import argparse
import csv
import json
import math
from collections import Counter
from typing import Dict, Iterable, List, Tuple


Key = Tuple[str, str, str, str, str, str, str, str, str]


def _truthy(value: str) -> bool:
    if value is None:
        return False
    return value.lower() in ("1", "true", "yes", "y")


def _key_value(value) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _finite_float(row: Dict[str, str], key: str):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalized_stop_ids(value: str) -> str:
    if value in ("", None):
        return ""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return ""
    try:
        ids = sorted({int(part) for part in parts})
    except ValueError:
        return str(value)
    return ",".join(str(token) for token in ids)


def _stop_key(policy: object, stop_token_ids: object) -> str:
    policy_text = _key_value(policy)
    ids_text = _normalized_stop_ids(_key_value(stop_token_ids))
    if policy_text == "tokenizer_eos":
        return "tokenizer_eos"
    if policy_text == "custom":
        return f"custom:{ids_text}"
    if ids_text:
        return f"custom:{ids_text}"
    return ""


def _row_stop_key(row: Dict[str, str]) -> str:
    return _stop_key(row.get("stop_token_policy", ""), row.get("stop_token_ids", ""))


def _int_value(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or "0"))
    except (TypeError, ValueError):
        return 0


def _row_key(row: Dict[str, str]) -> Key:
    return (
        row.get("algorithm", "speculative"),
        row.get("prompt_id", ""),
        row.get("prompt_sha256", ""),
        row.get("gamma", ""),
        row.get("drafter_freq_hz", ""),
        row.get("verifier_clock_mhz", ""),
        row.get("decoding_mode", "greedy"),
        row.get("max_new_tokens", ""),
        _row_stop_key(row),
    )


def _combo_key(
    algorithm: str,
    decoding_mode: str,
    max_new_tokens: str,
    stop_key: str,
    combo: Dict[str, object],
) -> Key:
    return (
        algorithm,
        _key_value(combo.get("prompt_id")),
        _key_value(combo.get("prompt_sha256")),
        _key_value(combo.get("gamma")),
        _key_value(combo.get("drafter_freq_hz")),
        _key_value(combo.get("verifier_clock_mhz")),
        decoding_mode,
        max_new_tokens,
        stop_key,
    )


def _run_index(row: Dict[str, str]) -> int:
    try:
        return int(float(row.get("run", "0") or "0"))
    except ValueError:
        return 0


def _measurement_order(row: Dict[str, str]) -> int:
    try:
        return int(float(row.get("measurement_order", "0") or "0"))
    except ValueError:
        return 0


def read_json(path: str) -> Dict[str, object]:
    with open(path) as f:
        return json.load(f)


def read_csvs(paths: Iterable[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path
                rows.append(row)
    return rows


def _preferred_rail(row: Dict[str, str]) -> str:
    if row.get("algorithm", "speculative") == "verifier_only":
        return row.get("verifier_primary_power_rail", "") or "verifier_gpu_power"
    return row.get("drafter_primary_power_rail", "") or "tot_power"


def _session_row_score(row: Dict[str, str]) -> Tuple[int, int, int]:
    rail = row.get("rail", "")
    preferred_rail = _preferred_rail(row)
    primary_rail = int(bool(preferred_rail) and rail == preferred_rail)
    has_total_energy = int(bool(row.get("system_total_energy_mj_per_generated_token", "")))
    has_active_energy = int(bool(row.get("system_active_energy_mj_per_generated_token", "")))
    return primary_rail, has_total_energy, has_active_energy


def session_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    by_session: Dict[str, Dict[str, str]] = {}
    for row in rows:
        session_id = row.get("session_id", "")
        if not session_id:
            continue
        current = by_session.get(session_id)
        if current is None or _session_row_score(row) > _session_row_score(current):
            by_session[session_id] = row
    return list(by_session.values())


def expected_counts(plans: Iterable[Dict[str, object]]) -> Counter:
    counts: Counter = Counter()
    for plan in plans:
        algorithm = str(plan.get("algorithm", "speculative"))
        decoding_mode = str(plan.get("decoding_mode", "greedy"))
        max_new_tokens = _key_value(plan.get("max_new_tokens"))
        stop_key = _stop_key(plan.get("stop_token_policy", ""), plan.get("stop_token_ids", ""))
        for combo in plan.get("combinations", []):
            key = _combo_key(algorithm, decoding_mode, max_new_tokens, stop_key, combo)
            counts[key] += int(combo.get("measured_runs", plan.get("measured_runs", 1)))
    return counts


def observed_counts(rows: Iterable[Dict[str, str]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        if _run_index(row) <= 0:
            continue
        counts[_row_key(row)] += 1
    return counts


def run_index_issues(
    plans: Iterable[Dict[str, object]],
    rows: Iterable[Dict[str, str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    expected = expected_counts(plans)
    observed: Dict[Key, List[int]] = {}
    for row in rows:
        run_index = _run_index(row)
        if run_index <= 0:
            continue
        key = _row_key(row)
        observed.setdefault(key, []).append(run_index)

    missing = []
    duplicate = []
    for key, expected_runs in sorted(expected.items()):
        expected_indices = set(range(1, expected_runs + 1))
        observed_indices = observed.get(key, [])
        observed_set = set(observed_indices)
        missing_indices = sorted(expected_indices - observed_set)
        duplicate_indices = sorted(
            run_index
            for run_index, count in Counter(observed_indices).items()
            if count > 1
        )
        if missing_indices:
            missing.append(
                {
                    "key": list(key),
                    "missing_run_indices": missing_indices,
                }
            )
        if duplicate_indices:
            duplicate.append(
                {
                    "key": list(key),
                    "duplicate_run_indices": duplicate_indices,
                }
            )
    return missing, duplicate


def measurement_order_issues(
    plans: Iterable[Dict[str, object]],
    rows: Iterable[Dict[str, str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    expected_by_algorithm: Dict[str, int] = {}
    for plan in plans:
        schedule = plan.get("measurement_schedule", [])
        if not schedule:
            continue
        algorithm = str(plan.get("algorithm", "speculative"))
        expected_by_algorithm[algorithm] = max(
            expected_by_algorithm.get(algorithm, 0),
            len(schedule),
        )

    if not expected_by_algorithm:
        return [], []

    observed: Dict[str, List[int]] = {}
    for row in rows:
        if _run_index(row) <= 0:
            continue
        algorithm = row.get("algorithm", "speculative")
        if algorithm not in expected_by_algorithm:
            continue
        observed.setdefault(algorithm, []).append(_measurement_order(row))

    missing = []
    duplicate = []
    for algorithm, expected_count in sorted(expected_by_algorithm.items()):
        expected_orders = set(range(1, expected_count + 1))
        observed_orders = observed.get(algorithm, [])
        observed_set = set(order for order in observed_orders if order > 0)
        missing_orders = sorted(expected_orders - observed_set)
        duplicate_orders = sorted(
            order
            for order, count in Counter(observed_orders).items()
            if order > 0 and count > 1
        )
        if missing_orders:
            missing.append(
                {
                    "algorithm": algorithm,
                    "missing_measurement_orders": missing_orders,
                }
            )
        if duplicate_orders:
            duplicate.append(
                {
                    "algorithm": algorithm,
                    "duplicate_measurement_orders": duplicate_orders,
                }
            )
    return missing, duplicate


def measurement_schedule_mismatches(
    plans: Iterable[Dict[str, object]],
    rows: Iterable[Dict[str, str]],
) -> List[Dict[str, object]]:
    expected_by_algorithm_order: Dict[str, Dict[int, Dict[str, object]]] = {}
    for plan in plans:
        algorithm = str(plan.get("algorithm", "speculative"))
        decoding_mode = str(plan.get("decoding_mode", "greedy"))
        max_new_tokens = _key_value(plan.get("max_new_tokens"))
        stop_key = _stop_key(plan.get("stop_token_policy", ""), plan.get("stop_token_ids", ""))
        combinations = list(plan.get("combinations", []))
        schedule = plan.get("measurement_schedule", [])
        if not schedule:
            continue

        expected_by_order = expected_by_algorithm_order.setdefault(algorithm, {})
        for item in schedule:
            try:
                order = int(item.get("order", 0) or 0)
                condition_index = int(item.get("condition_order", -1))
                run_index = int(item.get("run", 0) or 0)
            except (TypeError, ValueError):
                continue
            if order <= 0 or condition_index < 0 or condition_index >= len(combinations):
                continue
            combo = combinations[condition_index]
            expected_by_order[order] = {
                "key": _combo_key(
                    algorithm,
                    decoding_mode,
                    max_new_tokens,
                    stop_key,
                    combo,
                ),
                "run": run_index,
                "condition_order": condition_index,
            }

    mismatches = []
    for row in rows:
        if _run_index(row) <= 0:
            continue
        algorithm = row.get("algorithm", "speculative")
        expected_by_order = expected_by_algorithm_order.get(algorithm)
        if not expected_by_order:
            continue

        order = _measurement_order(row)
        if order <= 0:
            continue
        expected = expected_by_order.get(order)
        observed_key = _row_key(row)
        observed_run = _run_index(row)
        if expected is None:
            mismatches.append(
                {
                    "algorithm": algorithm,
                    "measurement_order": order,
                    "session_id": row.get("session_id", ""),
                    "reason": "measurement_order_not_in_plan_schedule",
                    "observed_key": list(observed_key),
                    "observed_run": observed_run,
                }
            )
            continue

        expected_key = expected["key"]
        expected_run = int(expected["run"])
        if observed_key != expected_key or observed_run != expected_run:
            mismatches.append(
                {
                    "algorithm": algorithm,
                    "measurement_order": order,
                    "session_id": row.get("session_id", ""),
                    "reason": "measurement_order_condition_mismatch",
                    "condition_order": expected["condition_order"],
                    "expected_key": list(expected_key),
                    "observed_key": list(observed_key),
                    "expected_run": expected_run,
                    "observed_run": observed_run,
                }
            )
    return mismatches


def measured_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if _run_index(row) > 0]


def incomplete_energy_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if not _truthy(row.get("system_energy_complete", ""))
    ]


def missing_idle_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    missing = []
    for row in rows:
        if not row.get("idle_baseline_s", ""):
            missing.append(row)
            continue
        algorithm = row.get("algorithm", "speculative")
        if algorithm == "speculative" or row.get("system_boundary", "") == "two_device_idle_drafter":
            if _int_value(row, "drafter_idle_power_samples") <= 0:
                missing.append(row)
                continue
        if _int_value(row, "verifier_idle_power_samples") <= 0:
            missing.append(row)
    return missing


def insufficient_power_sample_rows(
    rows: Iterable[Dict[str, str]],
    min_power_samples: int,
    require_idle_baseline: bool,
) -> List[Dict[str, str]]:
    if min_power_samples <= 0:
        return []

    invalid = []
    for row in rows:
        algorithm = row.get("algorithm", "speculative")
        if algorithm == "speculative":
            sample_keys = [
                "drafter_prefill_power_samples",
                "drafter_draft_power_samples",
                "drafter_commit_power_samples",
                "verifier_prefill_power_samples",
                "verifier_verify_power_samples",
            ]
        else:
            sample_keys = [
                "verifier_prefill_power_samples",
                "verifier_decode_power_samples",
            ]
        if require_idle_baseline:
            sample_keys.append("verifier_idle_power_samples")
            if algorithm == "speculative" or row.get("system_boundary", "") == "two_device_idle_drafter":
                sample_keys.append("drafter_idle_power_samples")

        if any(_int_value(row, key) < min_power_samples for key in sample_keys):
            invalid.append(row)
    return invalid


def invalid_generation_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    invalid = []
    for row in rows:
        try:
            generated_tokens = int(float(row.get("generated_tokens", "") or "0"))
            max_new_tokens = int(float(row.get("max_new_tokens", "") or "0"))
        except ValueError:
            invalid.append(row)
            continue

        if generated_tokens <= 0 or max_new_tokens <= 0:
            invalid.append(row)
            continue
        if generated_tokens > max_new_tokens:
            invalid.append(row)
            continue
        stop_reason = row.get("stop_reason", "")
        if generated_tokens < max_new_tokens and not (
            stop_reason.startswith("eos") or stop_reason.startswith("stop")
        ):
            invalid.append(row)
    return invalid


def invalid_metric_rows(
    rows: Iterable[Dict[str, str]],
    require_active_energy: bool,
) -> List[Dict[str, str]]:
    invalid = []
    for row in rows:
        required_positive = [
            "tokens_per_s",
            "wall_latency_ms",
            "system_total_energy_mj",
            "system_total_energy_mj_per_generated_token",
        ]
        if any((_finite_float(row, key) or 0.0) <= 0 for key in required_positive):
            invalid.append(row)
            continue

        if _truthy(row.get("system_energy_complete", "")):
            algorithm = row.get("algorithm", "speculative")
            if algorithm == "speculative":
                required_nonnegative = [
                    "drafter_total_energy_mj",
                    "verifier_total_energy_mj",
                    "drafter_draft_total_energy_mj",
                    "verifier_verify_total_energy_mj",
                ]
            else:
                required_nonnegative = [
                    "verifier_total_energy_mj",
                    "verifier_verify_total_energy_mj",
                ]
                if row.get("system_boundary", "") == "two_device_idle_drafter":
                    required_nonnegative.extend(
                        [
                            "drafter_total_energy_mj",
                            "drafter_idle_total_energy_mj",
                        ]
                    )
            for key in required_nonnegative:
                value = _finite_float(row, key)
                if value is None or value < 0:
                    invalid.append(row)
                    break
            else:
                if require_active_energy:
                    algorithm = row.get("algorithm", "speculative")
                    if algorithm == "speculative":
                        required_active = [
                            "drafter_active_energy_mj",
                            "verifier_active_energy_mj",
                            "system_active_energy_mj",
                            "system_active_energy_mj_per_generated_token",
                        ]
                    else:
                        required_active = [
                            "verifier_active_energy_mj",
                            "system_active_energy_mj",
                            "system_active_energy_mj_per_generated_token",
                        ]
                    if any(
                        (_finite_float(row, key) or 0.0) <= 0
                        for key in required_active
                    ):
                        invalid.append(row)
    return invalid


def plan_requires_idle(plans: Iterable[Dict[str, object]]) -> bool:
    return any(float(plan.get("idle_baseline_s", 0) or 0) > 0 for plan in plans)


def validate(
    plans: List[Dict[str, object]],
    raw_rows: List[Dict[str, str]],
    require_complete_energy: bool,
    require_idle_baseline: bool,
    min_power_samples: int = 1,
) -> Dict[str, object]:
    sessions = session_rows(raw_rows)
    measured_sessions = measured_rows(sessions)
    expected = expected_counts(plans)
    observed = observed_counts(measured_sessions)
    missing_run_indices, duplicate_run_indices = run_index_issues(
        plans,
        measured_sessions,
    )
    missing_measurement_orders, duplicate_measurement_orders = measurement_order_issues(
        plans,
        measured_sessions,
    )
    measurement_schedule_mismatch_rows = measurement_schedule_mismatches(
        plans,
        measured_sessions,
    )

    missing = []
    for key, expected_runs in sorted(expected.items()):
        observed_runs = observed.get(key, 0)
        if observed_runs < expected_runs:
            missing.append(
                {
                    "key": list(key),
                    "expected_runs": expected_runs,
                    "observed_runs": observed_runs,
                }
            )

    extras = [
        {
            "key": list(key),
            "observed_runs": count,
        }
        for key, count in sorted(observed.items())
        if key not in expected
    ]

    incomplete = (
        incomplete_energy_rows(measured_sessions) if require_complete_energy else []
    )
    idle_missing = (
        missing_idle_rows(measured_sessions) if require_idle_baseline else []
    )
    invalid_generation = invalid_generation_rows(measured_sessions)
    invalid_metrics = invalid_metric_rows(
        measured_sessions,
        require_active_energy=require_idle_baseline,
    )
    insufficient_samples = insufficient_power_sample_rows(
        measured_sessions,
        min_power_samples=min_power_samples,
        require_idle_baseline=require_idle_baseline,
    )
    ok = (
        not missing
        and not extras
        and not incomplete
        and not idle_missing
        and not invalid_generation
        and not invalid_metrics
        and not insufficient_samples
        and not missing_run_indices
        and not duplicate_run_indices
        and not missing_measurement_orders
        and not duplicate_measurement_orders
        and not measurement_schedule_mismatch_rows
    )
    return {
        "ok": ok,
        "expected_sessions": sum(expected.values()),
        "observed_sessions": sum(observed.values()),
        "expected_conditions": len(expected),
        "observed_conditions": len(observed),
        "missing": missing,
        "extra": extras,
        "incomplete_energy_sessions": [
            row.get("session_id", "") for row in incomplete
        ],
        "missing_idle_baseline_sessions": [
            row.get("session_id", "") for row in idle_missing
        ],
        "invalid_generation_sessions": [
            row.get("session_id", "") for row in invalid_generation
        ],
        "invalid_metric_sessions": [
            row.get("session_id", "") for row in invalid_metrics
        ],
        "insufficient_power_sample_sessions": [
            row.get("session_id", "") for row in insufficient_samples
        ],
        "missing_run_indices": missing_run_indices,
        "duplicate_run_indices": duplicate_run_indices,
        "missing_measurement_orders": missing_measurement_orders,
        "duplicate_measurement_orders": duplicate_measurement_orders,
        "measurement_schedule_mismatches": measurement_schedule_mismatch_rows[:20],
    }


def print_report(report: Dict[str, object]) -> None:
    print(f"ok={int(bool(report['ok']))}")
    print(f"expected_sessions={report['expected_sessions']}")
    print(f"observed_sessions={report['observed_sessions']}")
    print(f"expected_conditions={report['expected_conditions']}")
    print(f"observed_conditions={report['observed_conditions']}")
    print(f"missing_conditions={len(report['missing'])}")
    print(f"extra_conditions={len(report['extra'])}")
    print(f"incomplete_energy_sessions={len(report['incomplete_energy_sessions'])}")
    print(f"missing_idle_baseline_sessions={len(report['missing_idle_baseline_sessions'])}")
    print(f"invalid_generation_sessions={len(report['invalid_generation_sessions'])}")
    print(f"invalid_metric_sessions={len(report['invalid_metric_sessions'])}")
    print(
        "insufficient_power_sample_sessions="
        f"{len(report['insufficient_power_sample_sessions'])}"
    )
    print(f"missing_run_index_conditions={len(report['missing_run_indices'])}")
    print(f"duplicate_run_index_conditions={len(report['duplicate_run_indices'])}")
    print(f"missing_measurement_order_algorithms={len(report['missing_measurement_orders'])}")
    print(
        "duplicate_measurement_order_algorithms="
        f"{len(report['duplicate_measurement_orders'])}"
    )
    print(f"measurement_schedule_mismatches={len(report['measurement_schedule_mismatches'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw experiment CSV against plan JSON.")
    parser.add_argument("--plan", nargs="+", required=True, help="Plan JSON file(s)")
    parser.add_argument("--input", nargs="+", required=True, help="Raw result CSV file(s)")
    parser.add_argument("--out", default="", help="Optional JSON report path")
    parser.add_argument(
        "--allow-incomplete-energy",
        action="store_true",
        help="Do not fail on system_energy_complete=0.",
    )
    parser.add_argument(
        "--require-idle-baseline",
        action="store_true",
        help="Fail if idle baseline columns/samples are missing.",
    )
    parser.add_argument(
        "--min-power-samples",
        type=int,
        default=1,
        help="Minimum required power samples per required active phase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = [read_json(path) for path in args.plan]
    raw_rows = read_csvs(args.input)
    report = validate(
        plans=plans,
        raw_rows=raw_rows,
        require_complete_energy=not args.allow_incomplete_energy,
        require_idle_baseline=args.require_idle_baseline or plan_requires_idle(plans),
        min_power_samples=args.min_power_samples,
    )
    print_report(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote validation report to {args.out}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
