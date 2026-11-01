"""Per-sample efficiency measurements for Qwen2.5-Omni/OmniZip inference.

Prefill is defined as the first complete forward() of model.thinker within
model.generate(). It includes the Thinker's multimodal encoders and prefill;
it is *not* a pure LLM-only prefill measure.
"""
import time
from contextlib import contextmanager

import torch


def synchronize_all():
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


class EfficiencyProfiler:
    def __init__(self, model):
        self.model = model
        self.devices = list(range(torch.cuda.device_count()))
        self._start = None
        self._prefill_start = None
        self.prefill_s = None
        self.thinker_calls = 0
        self._before_handle = None
        self._after_handle = None
        self.start_alloc = {}
        self.start_reserved = {}

    def _before(self, module, args):
        self.thinker_calls += 1
        if self.thinker_calls == 1:
            synchronize_all()
            self._prefill_start = time.perf_counter()

    def _after(self, module, args, output):
        if self.thinker_calls == 1 and self.prefill_s is None and self._prefill_start is not None:
            synchronize_all()
            self.prefill_s = time.perf_counter() - self._prefill_start

    @contextmanager
    def measure_generate(self):
        synchronize_all()
        for d in self.devices:
            torch.cuda.reset_peak_memory_stats(d)
            self.start_alloc[d] = torch.cuda.memory_allocated(d)
            self.start_reserved[d] = torch.cuda.memory_reserved(d)
        self.thinker_calls = 0
        self.prefill_s = None
        self._prefill_start = None
        self.generate_s = None
        self._before_handle = self.model.thinker.register_forward_pre_hook(self._before)
        self._after_handle = self.model.thinker.register_forward_hook(self._after)
        self._start = time.perf_counter()
        try:
            yield self
        finally:
            synchronize_all()
            self.generate_s = time.perf_counter() - self._start
            self._before_handle.remove()
            self._after_handle.remove()

    def result(self):
        gib = 1024 ** 3
        per_gpu = {}
        for d in self.devices:
            per_gpu[str(d)] = {
                'baseline_allocated_gib': round(self.start_alloc.get(d, 0) / gib, 4),
                'baseline_reserved_gib': round(self.start_reserved.get(d, 0) / gib, 4),
                'peak_allocated_gib': round(torch.cuda.max_memory_allocated(d) / gib, 4),
                'peak_reserved_gib': round(torch.cuda.max_memory_reserved(d) / gib, 4),
            }
        return {
            'prefill_time_s': round(self.prefill_s, 6) if self.prefill_s is not None else None,
            'generate_time_s': round(self.generate_s, 6),
            'thinker_forward_calls': self.thinker_calls,
            'gpu_peak_allocated_gib': max((v['peak_allocated_gib'] for v in per_gpu.values()), default=None),
            'gpu_peak_allocated_sum_gib': round(sum(v['peak_allocated_gib'] for v in per_gpu.values()), 4),
            'gpu_peak_reserved_gib': max((v['peak_reserved_gib'] for v in per_gpu.values()), default=None),
            'gpu_memory_per_device': per_gpu,
        }
