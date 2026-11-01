#!/usr/bin/env python3
"""Aggregate lmms-eval WorldSense sample JSONL into category accuracies.

Expected JSONL fields (as emitted by the user's lmms-eval build):
  doc_id: dataset row index
  worldsense_score: {
      question_id, category, score, domain, duration, audio_class
  }

Important mapping:
  worldsense_score.category -> official dataset task_type
  worldsense_score.domain   -> official dataset task_domain
  official dataset domain   -> 8 high-level domains (e.g. Film & TV, Music)
  official dataset sub_category -> 67 fine-grained subcategories

The high-level domain and sub_category are NOT present in the supplied
worldsense_score object. They are joined from lmms-lab/WorldSense by doc_id.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

HIGH_LEVEL_DOMAINS = [
    "Tech & Science", "Culture & Politics", "Daily Life", "Film & TV",
    "Performance", "Games", "Sports", "Music",
]
DIMENSIONS = [
    "domain", "sub_category", "task_domain", "task_type",
    "duration", "audio_class",
]


def as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None and str(x) != ""]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") or s.startswith("("):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if x is not None]
            except (ValueError, SyntaxError):
                pass
        return [s]
    return [str(value)]


def norm_id(value: Any) -> str:
    # Handles int IDs and numeric strings such as "0" / "0.0".
    if value is None:
        return ""
    try:
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def load_hf_rows(dataset_name: str, split: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install it in the active "
            "omnizip environment with: pip install datasets"
        ) from exc

    kwargs = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    print(f"Loading metadata: {dataset_name} [{split}]")
    ds = load_dataset(dataset_name, **kwargs)
    return ds


def load_local_json_rows(path: Path) -> list[dict[str, Any]]:
    """Fallback for the original WorldSense qa.json layout.

    The JSON is expected to be {video_id: {duration, domain, sub_category,
    audio_class, task1: {...}, task2: {...}, ...}}. It is flattened in insertion
    order, with task entries kept in their JSON order.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Local QA JSON must be a dictionary keyed by video ID.")

    rows: list[dict[str, Any]] = []
    for video_id, video in data.items():
        if not isinstance(video, dict):
            continue
        for task_key, task in video.items():
            if not (isinstance(task_key, str) and "task" in task_key
                    and isinstance(task, dict)):
                continue
            rows.append({
                "index": len(rows),
                "video": str(video_id),
                "video_path": video.get("video_path", ""),
                "duration": video.get("duration"),
                "domain": video.get("domain"),
                "sub_category": video.get("sub_category"),
                "audio_class": video.get("audio_class", []),
                "task_domain": task.get("task_domain"),
                "task_type": task.get("task_type"),
            })
    return rows


def get_score_record(obj: dict[str, Any], line_number: int) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"Line {line_number}: expected a JSON object.")

    metric = obj.get("worldsense_score")
    if not isinstance(metric, dict):
        raise ValueError(
            f"Line {line_number}: missing `worldsense_score` dictionary."
        )
    if "score" not in metric:
        raise ValueError(f"Line {line_number}: missing worldsense_score.score.")

    doc_id = obj.get("doc_id")
    question_id = metric.get("question_id", doc_id)
    if doc_id is None and question_id is None:
        raise ValueError(f"Line {line_number}: both doc_id and question_id are missing.")
    if doc_id is not None and question_id is not None:
        if norm_id(doc_id) != norm_id(question_id):
            raise ValueError(
                f"Line {line_number}: doc_id={doc_id!r} does not match "
                f"worldsense_score.question_id={question_id!r}."
            )

    score = float(metric["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Line {line_number}: score must be in [0, 1], got {score!r}."
        )

    return {
        "doc_id": norm_id(doc_id if doc_id is not None else question_id),
        "score": score,
        # In this lmms-eval task these fields are task-level metadata,
        # not the 8 high-level WorldSense content domains.
        "task_type_from_score": metric.get("category"),
        "task_domain_from_score": metric.get("domain"),
        "duration_from_score": metric.get("duration"),
        "audio_class_from_score": metric.get("audio_class"),
    }


def check_same(label: str, from_score: Any, from_dataset: Any, line_number: int):
    if from_score is None or from_dataset is None:
        return
    if label == "audio_class":
        a = set(as_list(from_score))
        b = set(as_list(from_dataset))
        if a != b:
            raise ValueError(
                f"Line {line_number}: {label} mismatch: JSONL={sorted(a)!r}, "
                f"dataset={sorted(b)!r}."
            )
    elif str(from_score).strip() != str(from_dataset).strip():
        raise ValueError(
            f"Line {line_number}: {label} mismatch: JSONL={from_score!r}, "
            f"dataset={from_dataset!r}. This may indicate doc_id/order mismatch."
        )


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: dict[str, list[float]], categories: list[str] | None = None):
    keys = list(categories or [])
    keys += sorted(k for k in values if k not in keys)
    out = []
    for key in keys:
        scores = values.get(key, [])
        if scores:
            out.append({
                "category": key,
                "correct": round(sum(scores), 6),
                "total": len(scores),
                "accuracy_pct": round(100.0 * sum(scores) / len(scores), 4),
            })
        else:
            out.append({
                "category": key,
                "correct": 0,
                "total": 0,
                "accuracy_pct": None,
            })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True,
                        help="lmms-eval *_samples_worldsense.jsonl")
    parser.add_argument("--output", type=Path, default=Path("./worldsense_category_report"))
    parser.add_argument("--dataset", default="lmms-lab/WorldSense",
                        help="HF dataset name; default: lmms-lab/WorldSense")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--metadata-json", type=Path, default=None,
                        help="Optional original WorldSense qa.json fallback.")
    parser.add_argument("--expected-overall", type=float, default=None,
                        help="Optional lmms-eval overall percentage, e.g. 44.4199.")
    parser.add_argument("--expected-count", type=int, default=3172,
                        help="Expected full split size; use 0 to disable.")
    args = parser.parse_args()

    if not args.samples.is_file():
        parser.error(f"Samples file not found: {args.samples}")

    # Load authoritative metadata. Prefer the same HF test split used by lmms-eval.
    if args.metadata_json:
        metadata_rows = load_local_json_rows(args.metadata_json)
        print(f"Loaded {len(metadata_rows)} metadata rows from {args.metadata_json}")
    else:
        metadata_rows = load_hf_rows(args.dataset, args.split, args.cache_dir)

    # Validate the fields needed for the high-level and fine-grained reports.
    required = {"index", "domain", "sub_category", "task_domain", "task_type",
                "duration", "audio_class"}
    if len(metadata_rows) == 0:
        raise ValueError("Metadata dataset is empty.")
    first = metadata_rows[0]
    missing = sorted(required - set(first.keys()))
    if missing:
        raise ValueError(f"Metadata source is missing required columns: {missing}")

    groups: dict[str, dict[str, list[float]]] = {
        dim: defaultdict(list) for dim in DIMENSIONS
    }
    detail_rows = []
    seen: set[str] = set()
    scores: list[float] = []

    with args.samples.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                record = get_score_record(obj, line_num)
                idx = int(record["doc_id"])
                if idx < 0 or idx >= len(metadata_rows):
                    raise ValueError(
                        f"doc_id {idx} is outside metadata range 0..{len(metadata_rows)-1}."
                    )
                if record["doc_id"] in seen:
                    raise ValueError(f"Duplicate doc_id {record['doc_id']}.")

                meta = metadata_rows[idx]
                meta_index = norm_id(meta.get("index"))
                if meta_index != record["doc_id"]:
                    raise ValueError(
                        f"doc_id/index mismatch at line {line_num}: JSONL doc_id="
                        f"{record['doc_id']}, metadata row index={meta_index}. "
                        "Do not assume row order; verify the correct split/metadata source."
                    )

                check_same("task_type", record["task_type_from_score"],
                           meta.get("task_type"), line_num)
                check_same("task_domain", record["task_domain_from_score"],
                           meta.get("task_domain"), line_num)
                check_same("duration", record["duration_from_score"],
                           meta.get("duration"), line_num)
                check_same("audio_class", record["audio_class_from_score"],
                           meta.get("audio_class"), line_num)

                row = {
                    "doc_id": record["doc_id"],
                    "score": record["score"],
                    "domain": meta.get("domain"),
                    "sub_category": meta.get("sub_category"),
                    "task_domain": meta.get("task_domain"),
                    "task_type": meta.get("task_type"),
                    "duration": meta.get("duration"),
                    "audio_class": as_list(meta.get("audio_class")),
                    "video_path": meta.get("video_path", ""),
                }
                if not row["domain"] or not row["sub_category"]:
                    raise ValueError(
                        f"Metadata row {idx} lacks domain or sub_category."
                    )

                seen.add(record["doc_id"])
                scores.append(record["score"])
                detail_rows.append(row)

                for dim in DIMENSIONS:
                    for value in dict.fromkeys(as_list(row[dim])):
                        groups[dim][value].append(row["score"])

            except Exception as exc:
                raise ValueError(f"{args.samples}:{line_num}: {exc}") from exc

    if not scores:
        raise ValueError("No scored samples found in JSONL.")

    args.output.mkdir(parents=True, exist_ok=True)
    categories = {}
    preferred = {
        "domain": HIGH_LEVEL_DOMAINS,
        "duration": ["<1min", "1-2min", "2-4min", "4-6min", "6-8min", ">8min"],
        "task_domain": ["Recognition", "Understanding", "Reasoning"],
        "audio_class": ["Speech", "Event", "Music"],
    }
    for dim in DIMENSIONS:
        rows = summarize(groups[dim], preferred.get(dim))
        csv_write(args.output / f"{dim}.csv", rows,
                  ["category", "correct", "total", "accuracy_pct"])
        categories[dim] = rows

    # Keep convenient aliases for commonly used paper tables.
    csv_write(args.output / "film_tv_and_other_domains.csv", categories["domain"],
              ["category", "correct", "total", "accuracy_pct"])
    csv_write(args.output / "per_question.csv", [
        {
            **r,
            "audio_class": json.dumps(r["audio_class"], ensure_ascii=False),
        }
        for r in detail_rows
    ], [
        "doc_id", "score", "domain", "sub_category", "task_domain",
        "task_type", "duration", "audio_class", "video_path",
    ])

    overall = 100.0 * sum(scores) / len(scores)
    summary = {
        "samples_file": str(args.samples.resolve()),
        "metadata_source": (
            str(args.metadata_json.resolve()) if args.metadata_json
            else f"{args.dataset}[{args.split}]"
        ),
        "questions_evaluated": len(scores),
        "expected_full_questions": args.expected_count,
        "overall_accuracy_pct": round(overall, 6),
        "categories": categories,
        "field_mapping": {
            "worldsense_score.category": "official dataset task_type",
            "worldsense_score.domain": "official dataset task_domain",
            "official dataset domain": "8 high-level content domains",
            "official dataset sub_category": "fine-grained content subcategories",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nSamples scored: {len(scores)}")
    print(f"Overall accuracy: {overall:.4f}%")
    print("\nEight WorldSense content domains:")
    print(f"{'Domain':<23}{'Correct':>10}{'Total':>9}{'Accuracy':>13}")
    print("-" * 55)
    for row in categories["domain"]:
        acc = "N/A" if row["accuracy_pct"] is None else f"{row['accuracy_pct']:.2f}%"
        print(f"{row['category']:<23}{row['correct']:>10g}{row['total']:>9d}{acc:>13}")

    if args.expected_count and len(scores) != args.expected_count:
        print(
            f"\nWARNING: partial evaluation: {len(scores)} / "
            f"{args.expected_count} samples."
        )
    if args.expected_overall is not None:
        diff = abs(overall - args.expected_overall)
        status = "PASS" if diff <= 0.01 else "MISMATCH"
        print(
            f"\nOverall cross-check vs lmms-eval "
            f"{args.expected_overall:.4f}%: {status} "
            f"(difference {diff:.6f} percentage points)"
        )
    print(f"\nReports saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
