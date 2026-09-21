#!/usr/bin/env python3
"""Summarize G1-D stage-evaluation JSONL logs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


STAGE_NAMES = ("reach", "grasp", "lift", "transport", "place")


def load_records(paths):
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                missing = {
                    "success", "partial_success", "geometric_sorted_fraction", "stage_rates",
                    "duration_s", "objects",
                } - record.keys()
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}"
                    )
                records.append(record)
    if not records:
        raise ValueError("No evaluation records found")
    return records


def _mean(records, key):
    return statistics.fmean(float(record[key]) for record in records)


def summarize(records):
    """Return JSON-serializable aggregate metrics for one group of episodes."""
    partial = [float(record["partial_success"]) for record in records]
    object_records = [obj for record in records for obj in record["objects"]]
    successful_durations = [float(record["duration_s"]) for record in records if record["success"]]
    drops = sum(int(obj.get("drop_count", 0)) for obj in object_records)
    wrong_bins = sum(int(obj.get("wrong_bin_count", 0)) for obj in object_records)
    reasons = defaultdict(int)
    for record in records:
        reasons[str(record.get("reason", "unknown"))] += 1
    return {
        "episodes": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "success_rate": statistics.fmean(bool(record["success"]) for record in records),
        "partial_success_mean": statistics.fmean(partial),
        "partial_success_std": statistics.stdev(partial) if len(partial) > 1 else 0.0,
        "geometric_sorted_fraction_mean": _mean(records, "geometric_sorted_fraction"),
        "stage_rates": {
            stage: statistics.fmean(float(record["stage_rates"][stage]) for record in records)
            for stage in STAGE_NAMES
        },
        "duration_s_mean": _mean(records, "duration_s"),
        "successful_duration_s_mean": (
            statistics.fmean(successful_durations) if successful_durations else None
        ),
        "drops_total": drops,
        "drops_per_episode": drops / len(records),
        "wrong_bins_total": wrong_bins,
        "wrong_bins_per_episode": wrong_bins / len(records),
        "reasons": dict(sorted(reasons.items())),
        "scene_config_hashes": sorted({
            record["scene_config_hash"] for record in records if "scene_config_hash" in record
        }),
        "evaluation_config_hashes": sorted({
            record["evaluation_config_hash"] for record in records
            if "evaluation_config_hash" in record
        }),
    }


def grouped_summary(records, group_by):
    groups = defaultdict(list)
    if group_by == "none":
        groups["all"] = records
    else:
        fallback = "(unlabelled)" if group_by == "run_label" else "(missing)"
        for record in records:
            groups[str(record.get(group_by) or fallback)].append(record)
    return {label: summarize(group) for label, group in sorted(groups.items())}


def print_table(groups):
    headers = (
        "group", "episodes", "success", "PSR", "PSR sd", "sorted", *STAGE_NAMES,
        "time s", "success s", "drop/ep", "wrong/ep",
    )
    rows = []
    for label, metrics in groups.items():
        rows.append((
            label,
            str(metrics["episodes"]),
            f"{metrics['success_rate']:.3f}",
            f"{metrics['partial_success_mean']:.3f}",
            f"{metrics['partial_success_std']:.3f}",
            f"{metrics['geometric_sorted_fraction_mean']:.3f}",
            *(f"{metrics['stage_rates'][stage]:.3f}" for stage in STAGE_NAMES),
            f"{metrics['duration_s_mean']:.1f}",
            ("-" if metrics["successful_duration_s_mean"] is None
             else f"{metrics['successful_duration_s_mean']:.1f}"),
            f"{metrics['drops_per_episode']:.2f}",
            f"{metrics['wrong_bins_per_episode']:.2f}",
        ))
    widths = [max(len(str(value)) for value in column) for column in zip(headers, *rows)]
    print("  ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="One or more episodes.jsonl files")
    parser.add_argument(
        "--group-by", choices=("run_label", "run_id", "none"), default="run_label",
        help="How to group episodes (default: run_label)",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args()
    try:
        groups = grouped_summary(load_records(args.logs), args.group_by)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(groups, indent=2, sort_keys=True))
    else:
        print_table(groups)


if __name__ == "__main__":
    main()
