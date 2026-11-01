"""Measure WorldSense Qwen2.5-Omni baseline and OmniZip efficiency on a fixed subset."""
import argparse
import csv
import time
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from moviepy import VideoFileClip
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from efficiency_profiler import EfficiencyProfiler, synchronize_all


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate OmniZip on Daily-Omni')
    p.add_argument('--model-path', default='Qwen/Qwen2.5-Omni-3B')
    p.add_argument('--data-path', default='/mnt/user10/datasets/WorldSense/worldsense_qa.json')
    p.add_argument('--video-dir', default='/mnt/user10/datasets/WorldSense/videos')
    p.add_argument('--WAPPER-METHOD', choices=['omnizip', 'none'], default='omnizip')
    p.add_argument('--OMNIZIP_RHO_AUDIO', type=float, default=0.3)
    p.add_argument('--OMNIZIP_RHO_VIDEO', type=float, default=0.6)
    p.add_argument('--OMNIZIP_G', type=int, default=3)
    p.add_argument('--OMNIZIP_CONTEXTUAL_RATIO', type=float, default=0.05)
    p.add_argument('--mini-test-num', type=int, default=None)
    p.add_argument('--output-dir', default='logs/results_worldsense_efficiency')
    p.add_argument('--subset-file', default=None, help='JSON list of fixed sample indices (same file for both methods)')
    p.add_argument('--device', default='cuda:0', help='Logical CUDA device, after CUDA_VISIBLE_DEVICES remapping')
    p.add_argument('--strict', action='store_true', help='Abort on the first failed sample')
    p.add_argument('--warmup-samples', type=int, default=1, help='Exclude first N samples from efficiency aggregate; retain their records')
    return p.parse_args()


def video_for(video_root: Path, video_id: str) -> Path:
    path = video_root / f'{video_id}.mp4'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def flatten_worldsense(qa_data):
    samples = []
    for vid, video in qa_data.items():
        for key, task in video.items():
            if not key.startswith('task'):
                continue
            samples.append({
                'video_id': video['video_id'],
                'question': task['question'],
                'choices': task['candidates'],
                'answer': task['answer'],
                'question_type': task.get('task_type', key),
                'domain': video.get('domain', ''),
                'data_type': task.get('task_domain', ''),
            })
    return samples


def video_has_audio(path: Path) -> bool:
    with VideoFileClip(str(path)) as clip:
        return clip.audio is not None


def extract_answer(text):
    """Extract an option letter without matching arbitrary words beginning A-D."""
    t = text.strip().upper()
    for pattern in (r'^\s*(?:ANSWER\s*(?:IS|:)?\s*)?\(?([A-D])\)?(?:\s*[.、:)\-]|\s*$)',
                    r'\b(?:OPTION|CHOICE|ANSWER)\s*(?:IS|:)?\s*\(?([A-D])\)?\b'):
        m = re.search(pattern, t)
        if m:
            return m.group(1)
    return None


def summarize(rows):
    valid = [r for r in rows if r['status'] == 'ok']
    groups = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in valid:
        g = groups[r['question_type']]
        g['total'] += 1
        g['correct'] += int(r['is_correct'])
    total_correct = sum(r['is_correct'] for r in valid)
    return {
        'total_samples': len(rows),
        'evaluated_samples': len(valid),
        'failed_samples': len(rows) - len(valid),
        'correct': total_correct,
        'accuracy': total_correct / len(valid) if valid else None,
        'by_type': {k: {**v, 'accuracy': v['correct'] / v['total']} for k, v in sorted(groups.items())}
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('worldsense')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check your PyTorch and GPU configuration')
    with open(args.data_path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('Expected WorldSense QA JSON object')
    data = flatten_worldsense(data)
    if args.subset_file:
        with open(args.subset_file, encoding='utf-8') as f:
            indices = json.load(f)
        if not isinstance(indices, list) or not all(isinstance(i, int) and 0 <= i < len(data) for i in indices):
            raise ValueError('subset-file must contain a JSON list of valid sample indices')
        data = [(i, data[i]) for i in indices]
    else:
        data = list(enumerate(data))
    if args.mini_test_num is not None:
        data = data[:args.mini_test_num]
    video_root = Path(args.video_dir)
    output = Path(args.output_dir) / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.WAPPER_METHOD)
    output.mkdir(parents=True, exist_ok=True)
    if args.WAPPER_METHOD == 'omnizip':
        from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration
    log.info('Loading model %s (%s)', args.model_path, args.WAPPER_METHOD)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype='auto', device_map=args.device,
        attn_implementation='flash_attention_2',
    )
    model.eval()
    if args.WAPPER_METHOD == 'omnizip':
        model.thinker.omnizip_config = {
            'rho_audio': args.OMNIZIP_RHO_AUDIO,
            'rho_video': args.OMNIZIP_RHO_VIDEO,
            'g': args.OMNIZIP_G,
            'contextual_ratio': args.OMNIZIP_CONTEXTUAL_RATIO,
        }
        log.info('OmniZip config: %s', model.thinker.omnizip_config)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    with open(output / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with open(output / 'sample_indices.json', 'w', encoding='utf-8') as f:
        json.dump([idx for idx, _ in data], f, indent=2)
    results = []
    profiler = EfficiencyProfiler(model)
    for run_index, (source_index, sample) in enumerate(tqdm(data, desc='WorldSense')):
        video_id = str(sample['video_id'])
        choices = sample['choices']
        answer = str(sample['answer']).strip().upper()
        if answer not in 'ABCD' or len(answer) != 1:
            raise ValueError(f'Invalid answer for {video_id}: {answer}')
        result = {
            'index': source_index, 'video_id': video_id, 'question': sample['question'],
            'choices': choices, 'correct_answer': answer,
            'question_type': sample.get('question_type', 'Unknown'),
            'domain': sample.get('domain', ''),
            'predicted_answer': None, 'model_response': '',
            'is_correct': False, 'status': 'error', 'error': None,
        }
        result['warmup'] = (run_index < args.warmup_samples)
        result['prefill_time_s'] = None
        result['generate_time_s'] = None
        result['sample_latency_s'] = None
        result['preprocess_time_s'] = None
        result['gpu_peak_allocated_gib'] = None
        result['gpu_peak_allocated_sum_gib'] = None
        result['gpu_peak_reserved_gib'] = None
        result['gpu_memory_per_device'] = {}
        synchronize_all()
        sample_start = time.perf_counter()
        try:
            video_path = video_for(video_root, video_id)
            has_audio = video_has_audio(video_path)
            prompt = (f"{sample['question']}\nOptions:\n" + '\n'.join(choices)
                      + "\nAnswer with the option's letter from the given choices directly.")
            conversation = [
                {'role': 'system', 'content': [{'type': 'text', 'text':
                  'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, '
                  'capable of perceiving auditory and visual inputs, as well as generating text and speech. '
                  'Please analyze the video carefully and select the most appropriate answer from the given options.'}]},
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': str(video_path)},
                    {'type': 'text', 'text': prompt},
                ]},
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=has_audio)
            if not videos:
                raise RuntimeError('Video processor returned zero videos')
            model.thinker.nframes = videos[0].shape[0]
            inputs = processor(text=text, audio=audios, images=images, videos=videos,
                               return_tensors='pt', padding=True, use_audio_in_video=has_audio)
            inputs = inputs.to(model.device).to(model.dtype)
            synchronize_all()
            result['preprocess_time_s'] = round(time.perf_counter() - sample_start, 6)
            with profiler.measure_generate():
              with torch.inference_mode():
                generated = model.generate(
                    **inputs, return_audio=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=True, temperature=1, top_p=None, num_beams=1,
                    max_new_tokens=100, use_cache=True, use_audio_in_video=has_audio,
                    thinker_do_sample=False,
                )
            result.update(profiler.result())
            response = processor.batch_decode(
                [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated)],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()
            pred = extract_answer(response)
            result.update(status='ok', predicted_answer=pred, model_response=response,
                          is_correct=(pred == answer), video_path=str(video_path),
                          has_audio=has_audio, frames=int(model.thinker.nframes))
            log.info('%s GT=%s Pred=%s Correct=%s', video_id, answer, pred, result['is_correct'])
        except Exception as e:
            if profiler.generate_s is not None and result["generate_time_s"] is None:
                result.update(profiler.result())
            result['error'] = f'{type(e).__name__}: {e}'
            log.exception('Failed sample %s', video_id)
            if args.strict:
                results.append(result)
                with open(output / 'evaluation_results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                raise
        synchronize_all()
        result["sample_latency_s"] = round(time.perf_counter() - sample_start, 6)
        results.append(result)
        # Persist each sample so partial runs are recoverable
        with open(output / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    metrics = summarize(results)
    efficiency_rows = [r for r in results if r['status'] == 'ok' and not r['warmup']]
    fields = ['index', 'video_id', 'status', 'warmup', 'frames', 'has_audio', 'preprocess_time_s',
              'prefill_time_s', 'generate_time_s', 'sample_latency_s',
              'gpu_peak_allocated_gib', 'gpu_peak_allocated_sum_gib', 'gpu_peak_reserved_gib']
    with open(output / 'efficiency_per_sample.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    def mean(field):
        values = [r[field] for r in efficiency_rows if r.get(field) is not None]
        return round(sum(values) / len(values), 6) if values else None
    summary = {'measured_samples': len(efficiency_rows), 'warmup_samples': args.warmup_samples,
               'prefill_time_s_mean': mean('prefill_time_s'),
               'generate_time_s_mean': mean('generate_time_s'),
               'sample_latency_s_mean': mean('sample_latency_s'),
               'gpu_peak_allocated_gib_mean': mean('gpu_peak_allocated_gib'),
               'gpu_peak_allocated_gib_max': max((r['gpu_peak_allocated_gib'] for r in efficiency_rows
                                                   if r.get('gpu_peak_allocated_gib') is not None), default=None)}
    with open(output / 'efficiency_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output / 'accuracy_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info('Results in %s', output)
    log.info('Metrics: %s', json.dumps(metrics, ensure_ascii=False, indent=2))
    log.info('Efficiency: %s', json.dumps(summary, ensure_ascii=False, indent=2))
    if metrics['failed_samples']:
        log.warning('%d samples failed; review them before reporting full benchmark accuracy', metrics['failed_samples'])

if __name__ == '__main__':
    main()
