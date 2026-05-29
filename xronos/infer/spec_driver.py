import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


grpc = None
spec_pb2 = None
spec_pb2_grpc = None
RESULT_SCHEMA_VERSION = "xronos-spec-results-v12"
SPEC_RPC_SCHEMA_VERSION = "xronos-spec-rpc-v3"
SPEC_ALGORITHM_VERSION = "greedy-draft-verify-commit-v1"
BASELINE_ALGORITHM_VERSION = "verifier-greedy-v1"
DRAFTER_PRIMARY_POWER_RAIL = "tot_power"
VERIFIER_PRIMARY_POWER_RAIL = "verifier_gpu_power"


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    text: str
    source: str
    sha256: str


@dataclass(frozen=True)
class SpecCondition:
    prompt_case: PromptCase
    drafter_freq_hz: Optional[int]
    verifier_clock_mhz: Optional[int]
    gamma: int


@dataclass(frozen=True)
class SpecRunScheduleEntry:
    condition_index: int
    run_index: int


@dataclass(frozen=True)
class SpecWarmupScheduleEntry:
    condition_index: int
    warmup_index: int


def load_grpc_bindings() -> None:
    global grpc, spec_pb2, spec_pb2_grpc

    if grpc is not None:
        return

    import grpc as grpc_module
    from xronos.proto import spec_pb2 as spec_pb2_module
    from xronos.proto import spec_pb2_grpc as spec_pb2_grpc_module

    grpc = grpc_module
    spec_pb2 = spec_pb2_module
    spec_pb2_grpc = spec_pb2_grpc_module


def parse_int_csv(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",")]
    if any(not item for item in items):
        raise ValueError("CSV integer lists must not contain empty items.")
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integers, got: {value}") from exc
    if not parsed:
        raise ValueError("At least one integer value is required.")
    return parsed


def parse_optional_int_csv(value: str) -> List[Optional[int]]:
    if not value or not value.strip():
        return [None]
    items = [item.strip() for item in value.split(",")]
    if any(not item for item in items):
        raise ValueError(
            "At least one integer value is required, and empty CSV items are not allowed."
        )
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integers, got: {value}") from exc
    if not parsed:
        raise ValueError("At least one integer value is required, or leave it empty.")
    return parsed


def parse_stop_token_ids(value: str) -> List[int]:
    if not value or not value.strip():
        return []
    return parse_int_csv(value)


def _tokenizer_eos_ids(tokenizer) -> List[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, (list, tuple, set)):
        return [int(token) for token in eos_token_id]
    return [int(eos_token_id)]


def resolved_stop_token_ids(args: argparse.Namespace, tokenizer) -> List[int]:
    configured = parse_stop_token_ids(getattr(args, "stop_token_ids", ""))
    tokens = configured if configured else _tokenizer_eos_ids(tokenizer)
    return unique_token_ids(tokens)


def unique_token_ids(token_ids: Iterable[int]) -> List[int]:
    deduped: List[int] = []
    seen = set()
    for token in token_ids:
        token = int(token)
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def stop_token_policy(args: argparse.Namespace) -> str:
    return "custom" if getattr(args, "stop_token_ids", "").strip() else "tokenizer_eos"


def format_token_ids(token_ids: Iterable[int]) -> str:
    return ",".join(str(int(token)) for token in token_ids)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_set_hash(prompt_cases: List["PromptCase"]) -> str:
    prompt_hashes = sorted(prompt_case.sha256 for prompt_case in prompt_cases)
    return hashlib.sha256("\n".join(prompt_hashes).encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: List[int]) -> str:
    payload = json.dumps(
        [int(token) for token in token_ids],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(payload: Dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def plan_sha256(plan: Dict[str, object]) -> str:
    return canonical_json_sha256(plan_design_payload(plan))


PLAN_DESIGN_EXCLUDED_KEYS = {
    "metadata",
    "plan_sha256",
    "plan_design_sha256",
    "drafter_addr",
    "verifier_addr",
    "trace_out",
    "startup_timeout_s",
    "health_check_interval_s",
    "thermal_check_interval_s",
    "thermal_wait_timeout_s",
}


def plan_design_payload(plan: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in plan.items()
        if key not in PLAN_DESIGN_EXCLUDED_KEYS
    }


def plan_design_sha256(plan: Dict[str, object]) -> str:
    return canonical_json_sha256(plan_design_payload(plan))


def attach_plan_sha256(plan: Dict[str, object]) -> Dict[str, object]:
    plan["plan_design_sha256"] = plan_design_sha256(plan)
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def make_prompt_case(prompt_id: str, text: str, source: str) -> PromptCase:
    if not prompt_id:
        raise ValueError("Prompt id must not be empty.")
    if not text:
        raise ValueError(f"Prompt {prompt_id} is empty.")
    return PromptCase(
        prompt_id=prompt_id,
        text=text,
        source=source,
        sha256=prompt_hash(text),
    )


def selected_prompt_sources(args: argparse.Namespace) -> List[str]:
    return [
        name
        for name, value in (
            ("prompt", args.prompt),
            ("prompt_file", args.prompt_file),
            ("prompts_jsonl", args.prompts_jsonl),
        )
        if value
    ]


def load_prompt_cases(
    args: argparse.Namespace,
    allow_missing: bool = False,
) -> List[PromptCase]:
    sources = selected_prompt_sources(args)
    if len(sources) > 1:
        raise ValueError("Use only one of --prompt, --prompt-file, or --prompts-jsonl.")

    if args.prompt is not None:
        return [make_prompt_case("prompt_0", args.prompt, "inline")]

    if args.prompt_file:
        path = Path(args.prompt_file)
        return [
            make_prompt_case(
                path.stem or "prompt_0",
                path.read_text(encoding="utf-8"),
                f"file:{path}",
            )
        ]

    if args.prompts_jsonl:
        path = Path(args.prompts_jsonl)
        cases: List[PromptCase] = []
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {exc.msg}"
                    ) from exc

                if isinstance(item, str):
                    prompt_id = f"prompt_{len(cases)}"
                    text = item
                elif isinstance(item, dict):
                    text = item.get("prompt", item.get("text"))
                    prompt_id = str(
                        item.get("id", item.get("prompt_id", f"prompt_{len(cases)}"))
                    )
                else:
                    raise ValueError(
                        f"Expected JSON object or string in {path}:{line_number}."
                    )

                if not isinstance(text, str):
                    raise ValueError(
                        f"Prompt text must be a string in {path}:{line_number}."
                    )
                cases.append(
                    make_prompt_case(
                        prompt_id=prompt_id,
                        text=text,
                        source=f"jsonl:{path}:{line_number}",
                    )
                )

        if not cases:
            raise ValueError(f"No prompts were found in {path}.")
        return cases

    if allow_missing:
        return [
            PromptCase(
                prompt_id="prompt_0",
                text="",
                source="unspecified",
                sha256="",
            )
        ]
    raise ValueError("Provide --prompt, --prompt-file, or --prompts-jsonl.")


def validate_prompt_cases(prompt_cases: List[PromptCase]) -> None:
    seen = set()
    duplicates = set()
    for prompt_case in prompt_cases:
        if prompt_case.prompt_id in seen:
            duplicates.add(prompt_case.prompt_id)
        seen.add(prompt_case.prompt_id)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"Prompt ids must be unique. Duplicates: {joined}")


def run_git(args: List[str]) -> str:
    try:
        completed = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def collect_metadata() -> Dict[str, object]:
    status = run_git(["status", "--porcelain"])
    command_json = json.dumps(sys.argv, separators=(",", ":"))
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "command_sha256": hashlib.sha256(command_json.encode("utf-8")).hexdigest(),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "spec_rpc_schema_version": SPEC_RPC_SCHEMA_VERSION,
        "git_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "xronos_git_commit": os.environ.get("XRONOS_GIT_COMMIT", ""),
        "xronos_image": os.environ.get("XRONOS_IMAGE", ""),
        "pod_name": os.environ.get("POD_NAME", ""),
        "pod_namespace": os.environ.get("POD_NAMESPACE", ""),
        "node_name": os.environ.get("NODE_NAME", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
    }


def driver_metadata_fields(metadata: Optional[Dict[str, object]]) -> Dict[str, object]:
    metadata = metadata or {}
    return {
        "tokenizer_name_or_path": metadata.get("tokenizer_name_or_path", ""),
        "tokenizer_class": metadata.get("tokenizer_class", ""),
        "tokenizer_vocab_size": metadata.get("tokenizer_vocab_size", ""),
        "tokenizer_base_vocab_size": metadata.get("tokenizer_base_vocab_size", ""),
        "tokenizer_bos_token_id": metadata.get("tokenizer_bos_token_id", ""),
        "tokenizer_eos_token_id": metadata.get("tokenizer_eos_token_id", ""),
        "tokenizer_pad_token_id": metadata.get("tokenizer_pad_token_id", ""),
        "tokenizer_unk_token_id": metadata.get("tokenizer_unk_token_id", ""),
        "driver_hostname": metadata.get("hostname", ""),
        "driver_pod_name": metadata.get("pod_name", ""),
        "driver_pod_namespace": metadata.get("pod_namespace", ""),
        "driver_node_name": metadata.get("node_name", ""),
        "driver_python_version": metadata.get("python", ""),
        "driver_platform": metadata.get("platform", ""),
        "driver_result_schema_version": metadata.get("result_schema_version", ""),
        "driver_spec_rpc_schema_version": metadata.get("spec_rpc_schema_version", ""),
        "driver_git_branch": metadata.get("git_branch", ""),
        "driver_git_commit": metadata.get("git_commit", ""),
        "driver_git_dirty": int(bool(metadata.get("git_dirty", False))),
        "driver_xronos_git_commit": metadata.get("xronos_git_commit", ""),
        "driver_xronos_image": metadata.get("xronos_image", ""),
        "driver_command_sha256": metadata.get("command_sha256", ""),
        "driver_plan_sha256": metadata.get("plan_sha256", ""),
        "driver_plan_design_sha256": metadata.get("plan_design_sha256", ""),
        "driver_cuda_visible_devices": metadata.get("cuda_visible_devices", ""),
        "driver_nvidia_visible_devices": metadata.get("nvidia_visible_devices", ""),
    }


def build_sweep_plan(
    args: argparse.Namespace,
    prompt_cases: List[PromptCase],
    conditions: List[SpecCondition],
) -> Dict[str, object]:
    gammas = parse_int_csv(args.gammas)
    drafter_freqs_hz = parse_optional_int_csv(args.drafter_freqs_hz)
    verifier_clocks_mhz = parse_optional_int_csv(args.verifier_clocks_mhz)
    stop_policy = stop_token_policy(args)
    stop_token_ids_arg = format_token_ids(
        unique_token_ids(parse_stop_token_ids(args.stop_token_ids))
    )
    combos = [
        {
            "order": order,
            "prompt_id": condition.prompt_case.prompt_id,
            "prompt_sha256": condition.prompt_case.sha256,
            "gamma": condition.gamma,
            "drafter_freq_hz": condition.drafter_freq_hz,
            "verifier_clock_mhz": condition.verifier_clock_mhz,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
        }
        for order, condition in enumerate(conditions)
    ]
    measurement_schedule = build_spec_run_schedule(args, conditions)
    warmup_schedule = build_spec_warmup_schedule(args, conditions)
    if args.idle_baseline_s <= 0:
        total_idle_baselines = 0
    elif args.idle_baseline_policy == "run":
        total_idle_baselines = len(measurement_schedule)
    else:
        total_idle_baselines = len(conditions)
    plan = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": args.experiment,
        "algorithm": "speculative",
        "algorithm_version": SPEC_ALGORITHM_VERSION,
        "system_boundary": "two_device_active",
        "drafter_addr": args.drafter_addr or "",
        "verifier_addr": args.verifier_addr or "",
        "tokenizer": args.tokenizer or "",
        "gammas": gammas,
        "drafter_freqs_hz": drafter_freqs_hz,
        "verifier_clocks_mhz": verifier_clocks_mhz,
        "prompts": [
            {
                "prompt_id": prompt_case.prompt_id,
                "prompt_source": prompt_case.source,
                "prompt_sha256": prompt_case.sha256,
                "prompt_chars": len(prompt_case.text),
            }
            for prompt_case in prompt_cases
        ],
        "prompt_set_sha256": prompt_set_hash(prompt_cases),
        "prompt_count": len(prompt_cases),
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.runs,
        "max_new_tokens": args.max_new_tokens,
        "decoding_mode": args.decoding_mode,
        "stop_token_policy": stop_policy,
        "stop_token_ids": stop_token_ids_arg,
        "idle_baseline_s": args.idle_baseline_s,
        "idle_baseline_policy": args.idle_baseline_policy,
        "shuffle_conditions": args.shuffle_conditions,
        "shuffle_runs": args.shuffle_runs,
        "sample_runtime_metadata": args.sample_runtime_metadata,
        "startup_timeout_s": args.startup_timeout_s,
        "health_check_interval_s": args.health_check_interval_s,
        "max_start_temp_c": args.max_start_temp_c,
        "thermal_check_interval_s": args.thermal_check_interval_s,
        "thermal_wait_timeout_s": args.thermal_wait_timeout_s,
        "seed": args.seed,
        "trace_out": args.trace_out,
        "combinations": combos,
        "measurement_schedule": [
            {
                "order": order,
                "condition_order": entry.condition_index,
                "run": entry.run_index,
                "prompt_id": conditions[entry.condition_index].prompt_case.prompt_id,
                "prompt_sha256": conditions[entry.condition_index].prompt_case.sha256,
                "gamma": conditions[entry.condition_index].gamma,
                "drafter_freq_hz": conditions[entry.condition_index].drafter_freq_hz,
                "verifier_clock_mhz": conditions[entry.condition_index].verifier_clock_mhz,
            }
            for order, entry in enumerate(measurement_schedule, start=1)
        ],
        "warmup_schedule": [
            {
                "order": order,
                "condition_order": entry.condition_index,
                "warmup": entry.warmup_index,
                "prompt_id": conditions[entry.condition_index].prompt_case.prompt_id,
                "prompt_sha256": conditions[entry.condition_index].prompt_case.sha256,
                "gamma": conditions[entry.condition_index].gamma,
                "drafter_freq_hz": conditions[entry.condition_index].drafter_freq_hz,
                "verifier_clock_mhz": conditions[entry.condition_index].verifier_clock_mhz,
            }
            for order, entry in enumerate(warmup_schedule, start=1)
        ],
        "total_warmup_sessions": len(combos) * args.warmup_runs,
        "total_measured_sessions": len(combos) * args.runs,
        "total_idle_baseline_pairs": total_idle_baselines,
        "total_idle_baselines": total_idle_baselines,
        "metadata": collect_metadata(),
    }
    return attach_plan_sha256(plan)


def build_spec_conditions(
    args: argparse.Namespace,
    prompt_cases: List[PromptCase],
) -> List[SpecCondition]:
    conditions: List[SpecCondition] = []
    for prompt_case in prompt_cases:
        for drafter_freq_hz in parse_optional_int_csv(args.drafter_freqs_hz):
            for verifier_clock_mhz in parse_optional_int_csv(args.verifier_clocks_mhz):
                for gamma in parse_int_csv(args.gammas):
                    conditions.append(
                        SpecCondition(
                            prompt_case=prompt_case,
                            drafter_freq_hz=drafter_freq_hz,
                            verifier_clock_mhz=verifier_clock_mhz,
                            gamma=gamma,
                        )
                    )
    if args.shuffle_conditions:
        random.Random(args.seed).shuffle(conditions)
    return conditions


def build_spec_run_schedule(
    args: argparse.Namespace,
    conditions: List[SpecCondition],
) -> List[SpecRunScheduleEntry]:
    schedule = [
        SpecRunScheduleEntry(condition_index=condition_index, run_index=run_index)
        for run_index in range(1, args.runs + 1)
        for condition_index in range(len(conditions))
    ]
    if not args.shuffle_runs:
        schedule.sort(key=lambda entry: (entry.condition_index, entry.run_index))
    else:
        ordered = sorted(schedule, key=lambda entry: (entry.condition_index, entry.run_index))
        random.Random(args.seed).shuffle(schedule)
        if len(conditions) > 1 and len(schedule) > 1 and schedule == ordered:
            schedule = schedule[1:] + schedule[:1]
    return schedule


def build_spec_warmup_schedule(
    args: argparse.Namespace,
    conditions: List[SpecCondition],
) -> List[SpecWarmupScheduleEntry]:
    schedule = [
        SpecWarmupScheduleEntry(
            condition_index=condition_index,
            warmup_index=warmup_index,
        )
        for warmup_index in range(1, args.warmup_runs + 1)
        for condition_index in range(len(conditions))
    ]
    if not args.shuffle_runs:
        schedule.sort(key=lambda entry: (entry.condition_index, entry.warmup_index))
    else:
        ordered = sorted(
            schedule,
            key=lambda entry: (entry.condition_index, entry.warmup_index),
        )
        random.Random(f"{args.seed}:warmup").shuffle(schedule)
        if len(conditions) > 1 and len(schedule) > 1 and schedule == ordered:
            schedule = schedule[1:] + schedule[:1]
    return schedule


def print_sweep_plan(plan: Dict[str, object]) -> None:
    print(f"experiment={plan['experiment']}")
    print(f"prompts={plan['prompt_count']}")
    print(f"combinations={len(plan['combinations'])}")
    print(f"warmup_sessions={plan['total_warmup_sessions']}")
    print(f"measured_sessions={plan['total_measured_sessions']}")
    print(f"max_new_tokens={plan['max_new_tokens']}")
    print(f"plan_sha256={plan.get('plan_sha256', '')}")
    print(f"shuffle_runs={int(bool(plan.get('shuffle_runs', False)))}")
    for combo in plan["combinations"]:
        print(
            "prompt={prompt_id} gamma={gamma} fd={drafter_freq_hz} fv={verifier_clock_mhz} "
            "warmup={warmup_runs} runs={measured_runs}".format(**combo)
        )
    if plan.get("shuffle_runs"):
        for item in plan.get("warmup_schedule", [])[:20]:
            print(
                "warmup_order={order} condition={condition_order} warmup={warmup} "
                "prompt={prompt_id} gamma={gamma} fd={drafter_freq_hz} "
                "fv={verifier_clock_mhz}".format(**item)
            )
        for item in plan.get("measurement_schedule", [])[:20]:
            print(
                "schedule_order={order} condition={condition_order} run={run} "
                "prompt={prompt_id} gamma={gamma} fd={drafter_freq_hz} "
                "fv={verifier_clock_mhz}".format(**item)
            )


def write_json(path: str, payload: Dict[str, object]) -> None:
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: str, payload: Dict[str, object]) -> None:
    ensure_parent_dir(path)
    with open(path, "a") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def read_rows(path: str) -> List[Dict[str, object]]:
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _row_int(row: Dict[str, object], key: str) -> int:
    try:
        return int(float(str(row.get(key, "") or "0")))
    except (TypeError, ValueError):
        return 0


def load_resume_rows(
    args: argparse.Namespace,
    plan: Dict[str, object],
    expected_algorithm: str,
) -> Tuple[List[Dict[str, object]], Set[int]]:
    if not getattr(args, "resume", False):
        return [], set()
    if not args.out:
        raise ValueError("--resume requires --out.")

    output_path = Path(args.out)
    if not output_path.exists():
        print(f"Resume requested, but {args.out} does not exist; starting fresh.")
        return [], set()

    rows = read_rows(args.out)
    measured_rows = [row for row in rows if _row_int(row, "run") > 0]
    if not measured_rows:
        print(f"Resume requested from {args.out}, but no measured rows were found.")
        return rows, set()

    expected_plan_sha256 = str(plan.get("plan_sha256", ""))
    bad_plan_sessions = sorted(
        {
            str(row.get("session_id", ""))
            for row in measured_rows
            if str(row.get("plan_sha256", "")) != expected_plan_sha256
        }
    )
    if bad_plan_sessions:
        raise ValueError(
            "--resume output was produced by a different plan_sha256; "
            f"mismatched_sessions={bad_plan_sessions[:5]}."
        )

    bad_algorithm_sessions = sorted(
        {
            str(row.get("session_id", ""))
            for row in measured_rows
            if str(row.get("algorithm", "")) != expected_algorithm
        }
    )
    if bad_algorithm_sessions:
        raise ValueError(
            "--resume output has rows for a different algorithm; "
            f"mismatched_sessions={bad_algorithm_sessions[:5]}."
        )

    expected_orders = {
        int(item.get("order", 0) or 0)
        for item in plan.get("measurement_schedule", [])
    }
    expected_orders.discard(0)
    order_sessions: Dict[int, Set[str]] = defaultdict(set)
    for row in measured_rows:
        order = _row_int(row, "measurement_order")
        if order > 0:
            order_sessions[order].add(str(row.get("session_id", "")))

    unexpected_orders = sorted(set(order_sessions) - expected_orders)
    if unexpected_orders:
        raise ValueError(
            "--resume output contains measurement_order values outside the plan: "
            f"{unexpected_orders[:10]}."
        )

    duplicate_orders = sorted(
        order
        for order, sessions in order_sessions.items()
        if len({session for session in sessions if session}) > 1
    )
    if duplicate_orders:
        raise ValueError(
            "--resume output contains duplicate sessions for measurement_order: "
            f"{duplicate_orders[:10]}."
        )

    completed_orders = set(order_sessions)
    print(
        "Resume loaded {rows} rows from {path}; completed measured orders={done}/{total}.".format(
            rows=len(rows),
            path=args.out,
            done=len(completed_orders),
            total=len(expected_orders),
        )
    )
    return rows, completed_orders


def prepare_trace_output(
    args: argparse.Namespace,
    plan: Dict[str, object],
    resume_rows: List[Dict[str, object]],
) -> None:
    if not args.trace_out:
        return
    trace_path = Path(args.trace_out)
    ensure_parent_dir(args.trace_out)
    if getattr(args, "resume", False) and resume_rows:
        if not trace_path.exists():
            raise ValueError(
                "--resume found existing output rows, but --trace-out does not exist. "
                "Keep the original trace file or resume without trace output."
            )
    else:
        trace_path.write_text("")
    append_jsonl(args.trace_out, {"event": "plan", "plan": plan})


def rail_energy_map(response) -> Dict[str, Dict[str, float]]:
    return {
        rail.rail: {
            "mean_power_mw": rail.mean_power_mw,
            "energy_mj": rail.energy_mj,
        }
        for rail in response.rails
    }


def rail_mean_power_mw(response, preferred_rail: str) -> Optional[float]:
    values = {rail.rail: rail.mean_power_mw for rail in response.rails}
    if preferred_rail in values:
        return values[preferred_rail]
    if values:
        return sum(values.values())
    return None


def fmt_optional(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def active_energy_from_idle(
    total_energy_mj: float,
    idle_power_mw: Optional[float],
    active_latency_ms: float,
) -> Optional[float]:
    if idle_power_mw is None:
        return None
    return total_energy_mj - float(idle_power_mw) * active_latency_ms / 1000.0


def runtime_fingerprint_from_metadata(metadata: Dict[str, str], prefix: str) -> str:
    keys = [
        "python_version",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "gpu_name",
        "xronos_git_commit",
        "xronos_image",
    ]
    return "|".join(str(metadata.get(f"{prefix}_{key}", "")) for key in keys)


def runtime_status_fields(metadata: Dict[str, str], prefix: str) -> Dict[str, str]:
    return {
        f"{prefix}_runtime_temp_c": (
            metadata.get(f"{prefix}_thermal_max_temp_c", "")
            or metadata.get(f"{prefix}_nvidia_gpu_temp_c", "")
        ),
        f"{prefix}_thermal_zones": metadata.get(f"{prefix}_thermal_zones", ""),
        f"{prefix}_nvidia_pstate": metadata.get(f"{prefix}_nvidia_pstate", ""),
        f"{prefix}_nvidia_throttle_active": metadata.get(
            f"{prefix}_nvidia_throttle_active",
            "",
        ),
    }


def _metadata_float(metadata: Dict[str, str], key: str) -> Optional[float]:
    value = metadata.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_runtime_temp_c(
    metadata: Dict[str, str],
    prefixes: List[str],
) -> Optional[float]:
    temps = []
    for prefix in prefixes:
        for key in (
            f"{prefix}_thermal_max_temp_c",
            f"{prefix}_nvidia_gpu_temp_c",
            f"{prefix}_runtime_temp_c",
        ):
            value = _metadata_float(metadata, key)
            if value is not None:
                temps.append(value)
    return max(temps) if temps else None


async def thermal_guard_metadata(
    args: argparse.Namespace,
    initial_metadata: Dict[str, str],
    refresh_metadata,
    prefixes: List[str],
    label: str,
) -> Dict[str, str]:
    if args.max_start_temp_c is None:
        return initial_metadata

    metadata = initial_metadata
    deadline = time.monotonic() + args.thermal_wait_timeout_s
    while True:
        max_temp = max_runtime_temp_c(metadata, prefixes)
        if max_temp is None or max_temp <= args.max_start_temp_c:
            return metadata
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Thermal guard timed out before {label}: "
                f"max_temp_c={max_temp:.2f}, limit={args.max_start_temp_c:.2f}."
            )
        print(
            f"thermal guard before {label}: max_temp_c={max_temp:.2f} "
            f"> limit={args.max_start_temp_c:.2f}; waiting "
            f"{args.thermal_check_interval_s:.1f}s"
        )
        await asyncio.sleep(args.thermal_check_interval_s)
        metadata = await refresh_metadata()


def validate_args(args: argparse.Namespace) -> None:
    gammas = parse_int_csv(args.gammas)
    if any(gamma <= 0 for gamma in gammas):
        raise ValueError("--gammas values must be positive integers.")
    drafter_freqs_hz = parse_optional_int_csv(args.drafter_freqs_hz)
    if any(value is not None and value <= 0 for value in drafter_freqs_hz):
        raise ValueError("--drafter-freqs-hz values must be positive integers.")
    verifier_clocks_mhz = parse_optional_int_csv(args.verifier_clocks_mhz)
    if any(value is not None and value <= 0 for value in verifier_clocks_mhz):
        raise ValueError("--verifier-clocks-mhz values must be positive integers.")
    stop_token_ids = parse_stop_token_ids(args.stop_token_ids)
    if any(token < 0 for token in stop_token_ids):
        raise ValueError("--stop-token-ids values must not be negative.")
    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must not be negative.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than 0.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if args.startup_timeout_s < 0:
        raise ValueError("--startup-timeout-s must not be negative.")
    if args.health_check_interval_s <= 0:
        raise ValueError("--health-check-interval-s must be greater than 0.")
    if args.idle_baseline_s < 0:
        raise ValueError("--idle-baseline-s must not be negative.")
    if args.max_start_temp_c is not None and args.max_start_temp_c <= 0:
        raise ValueError("--max-start-temp-c must be greater than 0.")
    if args.thermal_check_interval_s <= 0:
        raise ValueError("--thermal-check-interval-s must be greater than 0.")
    if args.thermal_wait_timeout_s < 0:
        raise ValueError("--thermal-wait-timeout-s must not be negative.")
    if len(selected_prompt_sources(args)) > 1:
        raise ValueError("Use only one of --prompt, --prompt-file, or --prompts-jsonl.")
    if args.trace_warmups and not args.trace_out:
        raise ValueError("--trace-warmups requires --trace-out.")
    if args.dry_run:
        return
    missing = [
        name
        for name in ("drafter_addr", "verifier_addr", "tokenizer")
        if not getattr(args, name)
    ]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Missing required argument(s): {joined}")
    if not selected_prompt_sources(args):
        raise ValueError("Provide --prompt, --prompt-file, or --prompts-jsonl.")


async def check_health(
    name: str,
    stub,
    timeout_s: float,
    verbose: bool = True,
) -> Dict[str, str]:
    response = await stub.Health(spec_pb2.HealthRequest(), timeout=timeout_s)
    if not response.ok:
        raise RuntimeError(f"{name} health check failed: {response.message}")
    metadata = {f"{name}_{key}": value for key, value in dict(response.metadata).items()}
    metadata[f"{name}_model"] = response.model
    metadata[f"{name}_device"] = response.device
    if verbose:
        print(f"{name}: {response.message} model={response.model} device={response.device}")
    return metadata


async def wait_for_health(
    name: str,
    stub,
    timeout_s: float,
    startup_timeout_s: float,
    interval_s: float,
    verbose: bool = True,
) -> Dict[str, str]:
    deadline = time.monotonic() + startup_timeout_s
    last_error = ""
    while True:
        try:
            return await check_health(name, stub, timeout_s, verbose=verbose)
        except Exception as exc:
            last_error = str(exc)
            if startup_timeout_s <= 0 or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{name} health check did not become ready within "
                    f"{startup_timeout_s:.1f}s: {last_error}"
                ) from exc
            if verbose:
                print(
                    f"{name}: waiting for health check ({last_error}); "
                    f"retrying in {interval_s:.1f}s"
                )
            await asyncio.sleep(interval_s)


async def set_frequency(
    name: str,
    stub,
    timeout_s: float,
    jetson_gpu_freq_hz: Optional[int] = None,
    nvidia_smi_gpu_clock_mhz: Optional[int] = None,
) -> Dict[str, str]:
    response = await stub.SetFrequency(
        spec_pb2.SetFrequencyRequest(
            jetson_gpu_freq_hz=jetson_gpu_freq_hz or 0,
            nvidia_smi_gpu_clock_mhz=nvidia_smi_gpu_clock_mhz or 0,
        ),
        timeout=timeout_s,
    )
    if not response.ok:
        raise RuntimeError(f"{name} SetFrequency failed: {response.error}")
    metadata = {f"{name}_{key}": value for key, value in dict(response.metadata).items()}
    print(f"{name}: {response.message} {dict(response.metadata)}")
    return metadata


async def sample_idle_baseline(
    args: argparse.Namespace,
    drafter_stub,
    verifier_stub,
    drafter_freq_hz: Optional[int],
    verifier_clock_mhz: Optional[int],
) -> Dict[str, Optional[float]]:
    if args.idle_baseline_s <= 0:
        return {
            "idle_baseline_s": None,
            "drafter_idle_power_mw": None,
            "verifier_idle_power_mw": None,
            "drafter_idle_power_samples": 0,
            "verifier_idle_power_samples": 0,
        }

    request = spec_pb2.IdlePowerRequest(duration_s=args.idle_baseline_s)
    timeout_s = max(args.timeout, args.idle_baseline_s + 10.0)
    drafter_idle, verifier_idle = await asyncio.gather(
        drafter_stub.IdlePower(request, timeout=timeout_s),
        verifier_stub.IdlePower(request, timeout=timeout_s),
    )
    if drafter_idle.error:
        raise RuntimeError(f"Drafter IdlePower error: {drafter_idle.error}")
    if verifier_idle.error:
        raise RuntimeError(f"Verifier IdlePower error: {verifier_idle.error}")

    drafter_idle_power_mw = rail_mean_power_mw(drafter_idle, DRAFTER_PRIMARY_POWER_RAIL)
    verifier_idle_power_mw = rail_mean_power_mw(
        verifier_idle,
        VERIFIER_PRIMARY_POWER_RAIL,
    )
    baseline = {
        "idle_baseline_s": max(drafter_idle.duration_s, verifier_idle.duration_s),
        "drafter_idle_power_mw": drafter_idle_power_mw,
        "verifier_idle_power_mw": verifier_idle_power_mw,
        "drafter_idle_power_samples": int(drafter_idle.n_power_samples),
        "verifier_idle_power_samples": int(verifier_idle.n_power_samples),
    }
    print(
        "idle fd={fd} fv={fv} drafter_mw={dmw} verifier_mw={vmw}".format(
            fd=drafter_freq_hz or "",
            fv=verifier_clock_mhz or "",
            dmw=fmt_optional(drafter_idle_power_mw),
            vmw=fmt_optional(verifier_idle_power_mw),
        )
    )
    if args.trace_out:
        append_jsonl(
            args.trace_out,
            {
                "event": "idle_baseline",
                "algorithm": "speculative",
                "drafter_freq_hz": drafter_freq_hz,
                "verifier_clock_mhz": verifier_clock_mhz,
                "duration_s": baseline["idle_baseline_s"],
                "drafter_idle_power_mw": drafter_idle_power_mw,
                "verifier_idle_power_mw": verifier_idle_power_mw,
                "drafter_idle_power": rail_energy_map(drafter_idle),
                "verifier_idle_power": rail_energy_map(verifier_idle),
                "drafter_idle_power_samples": int(drafter_idle.n_power_samples),
                "verifier_idle_power_samples": int(verifier_idle.n_power_samples),
            },
        )
    return baseline


def empty_idle_baseline() -> Dict[str, Optional[float]]:
    return {
        "idle_baseline_s": None,
        "drafter_idle_power_mw": None,
        "verifier_idle_power_mw": None,
        "drafter_idle_power_samples": 0,
        "verifier_idle_power_samples": 0,
    }


async def idle_baseline_for_measured_run(
    args: argparse.Namespace,
    condition_idle_baseline: Dict[str, Optional[float]],
    drafter_stub,
    verifier_stub,
    drafter_freq_hz: Optional[int],
    verifier_clock_mhz: Optional[int],
) -> Dict[str, Optional[float]]:
    if args.idle_baseline_policy != "run":
        return condition_idle_baseline
    return await sample_idle_baseline(
        args=args,
        drafter_stub=drafter_stub,
        verifier_stub=verifier_stub,
        drafter_freq_hz=drafter_freq_hz,
        verifier_clock_mhz=verifier_clock_mhz,
    )


def add_power(
    response,
    energy_by_phase: Dict[str, Dict[str, float]],
    samples_by_phase: Dict[str, Dict[str, int]],
    phase: str,
) -> None:
    for rail in response.rails:
        energy_by_phase[phase][rail.rail] += rail.energy_mj
        samples_by_phase[phase][rail.rail] += int(response.n_power_samples)


def all_rails(energy_by_phase: Dict[str, Dict[str, float]]) -> List[str]:
    rails = sorted(
        {
            rail
            for phase_values in energy_by_phase.values()
            for rail in phase_values.keys()
        }
    )
    return rails or [""]


def phase_total(
    energy_by_phase: Dict[str, Dict[str, float]],
    phase: str,
    preferred_rail: str,
) -> float:
    values = energy_by_phase.get(phase, {})
    if preferred_rail in values:
        return values[preferred_rail]
    return sum(values.values())


def sample_total(
    samples_by_phase: Dict[str, Dict[str, int]],
    phase: str,
    preferred_rail: str,
) -> int:
    values = samples_by_phase.get(phase, {})
    if preferred_rail in values:
        return values[preferred_rail]
    return max(values.values(), default=0)


async def timed_rpc(awaitable):
    t0 = time.monotonic()
    response = await awaitable
    return response, (time.monotonic() - t0) * 1000.0


def _approx_message_value_size(value) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return 8
    if isinstance(value, float):
        return 8
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (list, tuple, set)):
        return sum(_approx_message_value_size(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _approx_message_value_size(key) + _approx_message_value_size(item)
            for key, item in value.items()
        )
    if hasattr(value, "__dict__"):
        return _approx_message_value_size(vars(value))
    return len(str(value).encode("utf-8"))


def message_byte_size(message) -> int:
    byte_size = getattr(message, "ByteSize", None)
    if callable(byte_size):
        try:
            return int(byte_size())
        except Exception:
            pass
    return _approx_message_value_size(message)


def add_rpc_bytes(
    rpc_bytes: Dict[str, int],
    phase: str,
    request,
    response,
) -> None:
    request_bytes = message_byte_size(request)
    response_bytes = message_byte_size(response)
    rpc_bytes["request_bytes"] += request_bytes
    rpc_bytes["response_bytes"] += response_bytes
    rpc_bytes[f"{phase}_request_bytes"] += request_bytes
    rpc_bytes[f"{phase}_response_bytes"] += response_bytes


async def init_sessions(
    session_id: str,
    context_tokens: List[int],
    drafter_stub,
    verifier_stub,
    timeout_s: float,
):
    drafter_init_request = spec_pb2.InitSessionRequest(
        session_id=session_id,
        context_tokens=context_tokens,
    )
    drafter_init, drafter_init_rpc_ms = await timed_rpc(
        drafter_stub.InitSession(
            drafter_init_request,
            timeout=timeout_s,
        )
    )
    if drafter_init.error:
        raise RuntimeError(f"Drafter InitSession error: {drafter_init.error}")

    verifier_init_request = spec_pb2.InitSessionRequest(
        session_id=session_id,
        context_tokens=context_tokens,
    )
    verifier_init, verifier_init_rpc_ms = await timed_rpc(
        verifier_stub.InitSession(
            verifier_init_request,
            timeout=timeout_s,
        )
    )
    if verifier_init.error:
        raise RuntimeError(f"Verifier InitSession error: {verifier_init.error}")
    rpc_bytes: Dict[str, int] = defaultdict(int)
    add_rpc_bytes(
        rpc_bytes,
        "drafter_init",
        drafter_init_request,
        drafter_init,
    )
    add_rpc_bytes(
        rpc_bytes,
        "verifier_init",
        verifier_init_request,
        verifier_init,
    )
    return (
        drafter_init,
        verifier_init,
        drafter_init_rpc_ms,
        verifier_init_rpc_ms,
        rpc_bytes,
    )

async def reset_sessions(session_id: str, drafter_stub, verifier_stub, timeout_s: float):
    await asyncio.gather(
        drafter_stub.ResetSession(
            spec_pb2.ResetSessionRequest(session_id=session_id),
            timeout=timeout_s,
        ),
        verifier_stub.ResetSession(
            spec_pb2.ResetSessionRequest(session_id=session_id),
            timeout=timeout_s,
        ),
    )


async def run_decode(
    args: argparse.Namespace,
    tokenizer,
    drafter_stub,
    verifier_stub,
    gamma: int,
    run_index: int,
    health_metadata: Dict[str, str],
    drafter_freq_hz: Optional[int],
    verifier_clock_mhz: Optional[int],
    prompt_case: PromptCase,
    prompt_set_sha256: str,
    idle_baseline: Dict[str, Optional[float]],
    measurement_order: int = 0,
    driver_metadata: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    prompt_tokens = tokenizer.encode(
        prompt_case.text,
        add_special_tokens=not args.no_special_tokens,
    )
    if not prompt_tokens:
        raise ValueError("Prompt produced zero tokens.")
    prompt_token_sha256 = token_ids_sha256(prompt_tokens)

    session_id = f"spec-{uuid.uuid4()}"
    stop_token_ids = resolved_stop_token_ids(args, tokenizer)
    stop_token_id_set = set(stop_token_ids)
    stop_token_ids_text = format_token_ids(stop_token_ids)
    stop_policy = stop_token_policy(args)
    generated_tokens: List[int] = []
    stop_reason = "max_new_tokens"
    accepted_draft_tokens = 0
    proposed_draft_tokens = 0
    replacement_tokens = 0
    steps = 0
    drafter_init_latency_ms = 0.0
    verifier_init_latency_ms = 0.0
    drafter_draft_latency_ms = 0.0
    drafter_commit_latency_ms = 0.0
    verifier_latency_ms = 0.0
    drafter_init_rpc_latency_ms = 0.0
    verifier_init_rpc_latency_ms = 0.0
    drafter_draft_rpc_latency_ms = 0.0
    drafter_commit_rpc_latency_ms = 0.0
    verifier_rpc_latency_ms = 0.0
    energy_by_phase: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    samples_by_phase: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rpc_bytes: Dict[str, int] = defaultdict(int)
    wall_t0 = time.monotonic()

    try:
        (
            drafter_init,
            verifier_init,
            drafter_init_rpc_latency_ms,
            verifier_init_rpc_latency_ms,
            init_rpc_bytes,
        ) = await init_sessions(
            session_id=session_id,
            context_tokens=prompt_tokens,
            drafter_stub=drafter_stub,
            verifier_stub=verifier_stub,
            timeout_s=args.timeout,
        )
        rpc_bytes.update(init_rpc_bytes)
        expected_context_tokens = len(prompt_tokens)
        if int(drafter_init.context_tokens) != expected_context_tokens:
            raise RuntimeError(
                "Drafter InitSession context length mismatch: "
                f"expected={expected_context_tokens}, "
                f"actual={int(drafter_init.context_tokens)}."
            )
        if int(verifier_init.context_tokens) != expected_context_tokens:
            raise RuntimeError(
                "Verifier InitSession context length mismatch: "
                f"expected={expected_context_tokens}, "
                f"actual={int(verifier_init.context_tokens)}."
            )
        drafter_init_latency_ms = drafter_init.latency_ms
        verifier_init_latency_ms = verifier_init.latency_ms
        add_power(drafter_init, energy_by_phase, samples_by_phase, "drafter_prefill")
        add_power(verifier_init, energy_by_phase, samples_by_phase, "verifier_prefill")

        while len(generated_tokens) < args.max_new_tokens:
            remaining = args.max_new_tokens - len(generated_tokens)
            draft_gamma = min(gamma, remaining)
            base_committed_tokens = len(prompt_tokens) + len(generated_tokens)

            draft_request = spec_pb2.DraftRequest(
                session_id=session_id,
                gamma=draft_gamma,
                base_committed_tokens=base_committed_tokens,
            )
            draft_response, draft_rpc_ms = await timed_rpc(
                drafter_stub.Draft(
                    draft_request,
                    timeout=args.timeout,
                )
            )
            if draft_response.error:
                raise RuntimeError(f"Drafter error: {draft_response.error}")
            add_rpc_bytes(rpc_bytes, "drafter_draft", draft_request, draft_response)

            draft_tokens = list(draft_response.draft_tokens)
            if not draft_tokens:
                raise RuntimeError("Drafter returned no tokens.")
            add_power(draft_response, energy_by_phase, samples_by_phase, "drafter_draft")

            verify_request = spec_pb2.VerifyRequest(
                session_id=session_id,
                draft_tokens=draft_tokens,
                append_replacement=remaining > len(draft_tokens),
                stop_token_ids=stop_token_ids,
                base_committed_tokens=base_committed_tokens,
            )
            verify_response, verify_rpc_ms = await timed_rpc(
                verifier_stub.Verify(
                    verify_request,
                    timeout=args.timeout,
                )
            )
            if verify_response.error:
                raise RuntimeError(f"Verifier error: {verify_response.error}")
            add_rpc_bytes(rpc_bytes, "verifier_verify", verify_request, verify_response)
            add_power(verify_response, energy_by_phase, samples_by_phase, "verifier_verify")

            accepted = int(verify_response.accepted_tokens)
            if accepted < 0 or accepted > len(draft_tokens):
                raise RuntimeError(
                    "Verifier returned accepted_tokens outside the draft length: "
                    f"accepted={accepted}, draft_length={len(draft_tokens)}."
                )
            accepted_tokens = draft_tokens[:accepted]
            available_for_accept = args.max_new_tokens - len(generated_tokens)
            used_accepted_tokens = accepted_tokens[:available_for_accept]
            generated_tokens.extend(used_accepted_tokens)

            append_replacement_to_output = len(generated_tokens) < args.max_new_tokens
            appended_replacement = bool(verify_response.appended_replacement)
            if appended_replacement and not append_replacement_to_output:
                raise RuntimeError(
                    "Verifier appended a replacement token beyond max_new_tokens."
                )
            if not used_accepted_tokens and not appended_replacement:
                raise RuntimeError(
                    "Speculative decode step made no token progress. "
                    "Verifier must accept at least one token or append a replacement."
                )
            expected_committed_tokens = (
                base_committed_tokens
                + len(used_accepted_tokens)
                + int(appended_replacement)
            )
            if int(verify_response.committed_tokens) != expected_committed_tokens:
                raise RuntimeError(
                    "Verifier committed token count mismatch: "
                    f"expected={expected_committed_tokens}, "
                    f"actual={int(verify_response.committed_tokens)}."
                )
            commit_request = spec_pb2.CommitRequest(
                session_id=session_id,
                accepted_tokens=accepted,
                replacement_token=int(verify_response.replacement_token),
                append_replacement=appended_replacement,
                base_committed_tokens=base_committed_tokens,
            )
            commit_response, commit_rpc_ms = await timed_rpc(
                drafter_stub.Commit(
                    commit_request,
                    timeout=args.timeout,
                )
            )
            if commit_response.error:
                raise RuntimeError(f"Drafter Commit error: {commit_response.error}")
            if int(commit_response.committed_tokens) != expected_committed_tokens:
                raise RuntimeError(
                    "Drafter committed token count mismatch: "
                    f"expected={expected_committed_tokens}, "
                    f"actual={int(commit_response.committed_tokens)}."
                )
            add_rpc_bytes(rpc_bytes, "drafter_commit", commit_request, commit_response)
            add_power(commit_response, energy_by_phase, samples_by_phase, "drafter_commit")

            if appended_replacement:
                replacement_token = int(verify_response.replacement_token)
                generated_tokens.append(replacement_token)
                replacement_tokens += 1
            else:
                replacement_token = None

            if args.trace_out and (run_index > 0 or args.trace_warmups):
                append_jsonl(
                    args.trace_out,
                    {
                        "event": "step",
                        "experiment": args.experiment,
                        "algorithm": "speculative",
                        "session_id": session_id,
                        "decoding_mode": args.decoding_mode,
                        "prompt_id": prompt_case.prompt_id,
                        "prompt_sha256": prompt_case.sha256,
                        "prompt_set_sha256": prompt_set_sha256,
                        "gamma": gamma,
                        "run": run_index,
                        "measurement_order": measurement_order,
                        "step": steps + 1,
                        "drafter_freq_hz": drafter_freq_hz,
                        "verifier_clock_mhz": verifier_clock_mhz,
                        "remaining_before_step": remaining,
                        "draft_gamma": draft_gamma,
                        "base_committed_tokens": base_committed_tokens,
                        "draft_tokens": draft_tokens,
                        "accepted_tokens": accepted,
                        "accepted_token_ids": used_accepted_tokens,
                        "replacement_token": replacement_token,
                        "appended_replacement": appended_replacement,
                        "generated_tokens_so_far": len(generated_tokens),
                        "committed_tokens_after_step": expected_committed_tokens,
                        "verifier_committed_tokens": int(
                            verify_response.committed_tokens
                        ),
                        "drafter_committed_tokens": int(
                            commit_response.committed_tokens
                        ),
                        "drafter_draft_latency_ms": draft_response.latency_ms,
                        "verifier_latency_ms": verify_response.latency_ms,
                        "drafter_commit_latency_ms": commit_response.latency_ms,
                        "drafter_draft_rpc_latency_ms": draft_rpc_ms,
                        "verifier_rpc_latency_ms": verify_rpc_ms,
                        "drafter_commit_rpc_latency_ms": commit_rpc_ms,
                        "drafter_draft_rpc_request_bytes": message_byte_size(
                            draft_request,
                        ),
                        "drafter_draft_rpc_response_bytes": message_byte_size(
                            draft_response,
                        ),
                        "verifier_verify_rpc_request_bytes": message_byte_size(
                            verify_request,
                        ),
                        "verifier_verify_rpc_response_bytes": message_byte_size(
                            verify_response,
                        ),
                        "drafter_commit_rpc_request_bytes": message_byte_size(
                            commit_request,
                        ),
                        "drafter_commit_rpc_response_bytes": message_byte_size(
                            commit_response,
                        ),
                        "estimated_step_rpc_overhead_ms": (
                            draft_rpc_ms
                            + verify_rpc_ms
                            + commit_rpc_ms
                            - draft_response.latency_ms
                            - verify_response.latency_ms
                            - commit_response.latency_ms
                        ),
                        "drafter_draft_power": rail_energy_map(draft_response),
                        "verifier_verify_power": rail_energy_map(verify_response),
                        "drafter_commit_power": rail_energy_map(commit_response),
                        "drafter_draft_power_samples": int(draft_response.n_power_samples),
                        "verifier_power_samples": int(verify_response.n_power_samples),
                        "drafter_commit_power_samples": int(commit_response.n_power_samples),
                    },
                )

            proposed_draft_tokens += len(draft_tokens)
            accepted_draft_tokens += accepted
            drafter_draft_latency_ms += draft_response.latency_ms
            drafter_commit_latency_ms += commit_response.latency_ms
            verifier_latency_ms += verify_response.latency_ms
            drafter_draft_rpc_latency_ms += draft_rpc_ms
            drafter_commit_rpc_latency_ms += commit_rpc_ms
            verifier_rpc_latency_ms += verify_rpc_ms
            steps += 1

            if stop_token_id_set:
                if any(token in stop_token_id_set for token in used_accepted_tokens):
                    stop_reason = "stop_accepted"
                    break
                if replacement_token in stop_token_id_set:
                    stop_reason = "stop_replacement"
                    break
    finally:
        await reset_sessions(session_id, drafter_stub, verifier_stub, args.timeout)

    wall_ms = (time.monotonic() - wall_t0) * 1000.0
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    output_token_sha256 = token_ids_sha256(generated_tokens)
    accept_rate = (
        accepted_draft_tokens / proposed_draft_tokens
        if proposed_draft_tokens
        else 0.0
    )
    tokens_per_s = len(generated_tokens) / (wall_ms / 1000.0) if generated_tokens else 0.0
    server_compute_latency_ms = (
        drafter_init_latency_ms
        + verifier_init_latency_ms
        + drafter_draft_latency_ms
        + drafter_commit_latency_ms
        + verifier_latency_ms
    )
    client_rpc_latency_ms = (
        drafter_init_rpc_latency_ms
        + verifier_init_rpc_latency_ms
        + drafter_draft_rpc_latency_ms
        + drafter_commit_rpc_latency_ms
        + verifier_rpc_latency_ms
    )
    estimated_rpc_overhead_ms = client_rpc_latency_ms - server_compute_latency_ms
    rpc_request_bytes = int(rpc_bytes.get("request_bytes", 0))
    rpc_response_bytes = int(rpc_bytes.get("response_bytes", 0))
    rpc_total_bytes = rpc_request_bytes + rpc_response_bytes

    rows: List[Dict[str, object]] = []
    drafter_prefill_total = phase_total(
        energy_by_phase,
        "drafter_prefill",
        DRAFTER_PRIMARY_POWER_RAIL,
    )
    drafter_draft_total = phase_total(
        energy_by_phase,
        "drafter_draft",
        DRAFTER_PRIMARY_POWER_RAIL,
    )
    drafter_commit_total = phase_total(
        energy_by_phase,
        "drafter_commit",
        DRAFTER_PRIMARY_POWER_RAIL,
    )
    drafter_total = drafter_prefill_total + drafter_draft_total + drafter_commit_total
    verifier_prefill_total = phase_total(
        energy_by_phase,
        "verifier_prefill",
        VERIFIER_PRIMARY_POWER_RAIL,
    )
    verifier_verify_total = phase_total(
        energy_by_phase,
        "verifier_verify",
        VERIFIER_PRIMARY_POWER_RAIL,
    )
    verifier_total = verifier_prefill_total + verifier_verify_total
    system_total = drafter_total + verifier_total
    drafter_idle_power_mw = idle_baseline.get("drafter_idle_power_mw")
    verifier_idle_power_mw = idle_baseline.get("verifier_idle_power_mw")
    drafter_prefill_active_energy = active_energy_from_idle(
        drafter_prefill_total,
        drafter_idle_power_mw,
        drafter_init_latency_ms,
    )
    drafter_draft_active_energy = active_energy_from_idle(
        drafter_draft_total,
        drafter_idle_power_mw,
        drafter_draft_latency_ms,
    )
    drafter_commit_active_energy = active_energy_from_idle(
        drafter_commit_total,
        drafter_idle_power_mw,
        drafter_commit_latency_ms,
    )
    verifier_prefill_active_energy = active_energy_from_idle(
        verifier_prefill_total,
        verifier_idle_power_mw,
        verifier_init_latency_ms,
    )
    verifier_verify_active_energy = active_energy_from_idle(
        verifier_verify_total,
        verifier_idle_power_mw,
        verifier_latency_ms,
    )
    drafter_active_energy = (
        drafter_prefill_active_energy
        + drafter_draft_active_energy
        + drafter_commit_active_energy
        if drafter_prefill_active_energy is not None
        and drafter_draft_active_energy is not None
        and drafter_commit_active_energy is not None
        else None
    )
    verifier_active_energy = (
        verifier_prefill_active_energy + verifier_verify_active_energy
        if verifier_prefill_active_energy is not None
        and verifier_verify_active_energy is not None
        else None
    )
    system_active_energy = (
        drafter_active_energy + verifier_active_energy
        if drafter_active_energy is not None and verifier_active_energy is not None
        else None
    )
    drafter_prefill_power_samples = sample_total(
        samples_by_phase,
        "drafter_prefill",
        DRAFTER_PRIMARY_POWER_RAIL,
    )
    drafter_draft_power_samples = sample_total(
        samples_by_phase,
        "drafter_draft",
        DRAFTER_PRIMARY_POWER_RAIL,
    )
    drafter_commit_power_samples = sample_total(
        samples_by_phase,
        "drafter_commit",
        DRAFTER_PRIMARY_POWER_RAIL,
    )
    verifier_prefill_power_samples = sample_total(
        samples_by_phase,
        "verifier_prefill",
        VERIFIER_PRIMARY_POWER_RAIL,
    )
    verifier_verify_power_samples = sample_total(
        samples_by_phase,
        "verifier_verify",
        VERIFIER_PRIMARY_POWER_RAIL,
    )
    drafter_power_samples = (
        drafter_prefill_power_samples
        + drafter_draft_power_samples
        + drafter_commit_power_samples
    )
    verifier_power_samples = (
        verifier_prefill_power_samples
        + verifier_verify_power_samples
    )
    drafter_energy_available = drafter_power_samples > 0
    verifier_energy_available = verifier_power_samples > 0
    system_energy_complete = drafter_energy_available and verifier_energy_available
    if args.trace_out and (run_index > 0 or args.trace_warmups):
        append_jsonl(
            args.trace_out,
            {
                "event": "speculative_run",
                "experiment": args.experiment,
                "algorithm": "speculative",
                "session_id": session_id,
                "decoding_mode": args.decoding_mode,
                "prompt_id": prompt_case.prompt_id,
                "prompt_sha256": prompt_case.sha256,
                "prompt_set_sha256": prompt_set_sha256,
                "gamma": gamma,
                "run": run_index,
                "measurement_order": measurement_order,
                "drafter_freq_hz": drafter_freq_hz,
                "verifier_clock_mhz": verifier_clock_mhz,
                "generated_tokens": len(generated_tokens),
                "generated_token_ids": generated_tokens,
                "stop_reason": stop_reason,
                "output_token_sha256": output_token_sha256,
                "stop_token_policy": stop_policy,
                "stop_token_ids": stop_token_ids_text,
                "steps": steps,
                "draft_tokens": proposed_draft_tokens,
                "accepted_draft_tokens": accepted_draft_tokens,
                "replacement_tokens": replacement_tokens,
                "drafter_total_energy_mj": drafter_total,
                "drafter_prefill_active_energy_mj": drafter_prefill_active_energy,
                "drafter_draft_active_energy_mj": drafter_draft_active_energy,
                "drafter_commit_active_energy_mj": drafter_commit_active_energy,
                "drafter_active_energy_mj": drafter_active_energy,
                "verifier_total_energy_mj": verifier_total,
                "verifier_prefill_active_energy_mj": verifier_prefill_active_energy,
                "verifier_verify_active_energy_mj": verifier_verify_active_energy,
                "system_total_energy_mj": system_total,
                "system_active_energy_mj": system_active_energy,
                "wall_latency_ms": wall_ms,
                "client_rpc_latency_ms": client_rpc_latency_ms,
                "server_compute_latency_ms": server_compute_latency_ms,
                "estimated_rpc_overhead_ms": estimated_rpc_overhead_ms,
                "rpc_request_bytes": rpc_request_bytes,
                "rpc_response_bytes": rpc_response_bytes,
                "rpc_total_bytes": rpc_total_bytes,
                "drafter_power_samples": drafter_power_samples,
                "verifier_power_samples": verifier_power_samples,
                "system_energy_complete": system_energy_complete,
            },
        )
    for rail in all_rails(energy_by_phase):
        drafter_prefill_energy = energy_by_phase["drafter_prefill"].get(rail, 0.0)
        drafter_draft_energy = energy_by_phase["drafter_draft"].get(rail, 0.0)
        drafter_commit_energy = energy_by_phase["drafter_commit"].get(rail, 0.0)
        verifier_prefill_energy = energy_by_phase["verifier_prefill"].get(rail, 0.0)
        verifier_verify_energy = energy_by_phase["verifier_verify"].get(rail, 0.0)
        role_total_energy = (
            drafter_prefill_energy
            + drafter_draft_energy
            + drafter_commit_energy
            + verifier_prefill_energy
            + verifier_verify_energy
        )
        total_samples = (
            samples_by_phase["drafter_prefill"].get(rail, 0)
            + samples_by_phase["drafter_draft"].get(rail, 0)
            + samples_by_phase["drafter_commit"].get(rail, 0)
            + samples_by_phase["verifier_prefill"].get(rail, 0)
            + samples_by_phase["verifier_verify"].get(rail, 0)
        )
        rows.append(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "experiment": args.experiment,
                "algorithm": "speculative",
                "algorithm_version": SPEC_ALGORITHM_VERSION,
                "system_boundary": "two_device_active",
                "plan_sha256": (driver_metadata or {}).get("plan_sha256", ""),
                "plan_design_sha256": (driver_metadata or {}).get(
                    "plan_design_sha256",
                    "",
                ),
                "session_id": session_id,
                "decoding_mode": args.decoding_mode,
                "prompt_id": prompt_case.prompt_id,
                "prompt_source": prompt_case.source,
                "prompt_sha256": prompt_case.sha256,
                "prompt_set_sha256": prompt_set_sha256,
                "prompt_chars": len(prompt_case.text),
                "max_new_tokens": args.max_new_tokens,
                "gamma": gamma,
                "run": run_index,
                "measurement_order": measurement_order,
                "drafter_freq_hz": drafter_freq_hz or "",
                "verifier_clock_mhz": verifier_clock_mhz or "",
                "rail": rail,
                "drafter_primary_power_rail": DRAFTER_PRIMARY_POWER_RAIL,
                "verifier_primary_power_rail": VERIFIER_PRIMARY_POWER_RAIL,
                "system_primary_power_rails": (
                    f"drafter:{DRAFTER_PRIMARY_POWER_RAIL},"
                    f"verifier:{VERIFIER_PRIMARY_POWER_RAIL}"
                ),
                "prompt_tokens": len(prompt_tokens),
                "prompt_token_sha256": prompt_token_sha256,
                "generated_tokens": len(generated_tokens),
                "stop_reason": stop_reason,
                "stop_token_policy": stop_policy,
                "stop_token_ids": stop_token_ids_text,
                "output_token_sha256": output_token_sha256,
                "steps": steps,
                "draft_tokens": proposed_draft_tokens,
                "accepted_draft_tokens": accepted_draft_tokens,
                "replacement_tokens": replacement_tokens,
                "accept_rate": f"{accept_rate:.6f}",
                "drafter_init_latency_ms": f"{drafter_init_latency_ms:.4f}",
                "verifier_init_latency_ms": f"{verifier_init_latency_ms:.4f}",
                "drafter_draft_latency_ms": f"{drafter_draft_latency_ms:.4f}",
                "drafter_commit_latency_ms": f"{drafter_commit_latency_ms:.4f}",
                "verifier_latency_ms": f"{verifier_latency_ms:.4f}",
                "drafter_init_rpc_latency_ms": f"{drafter_init_rpc_latency_ms:.4f}",
                "verifier_init_rpc_latency_ms": f"{verifier_init_rpc_latency_ms:.4f}",
                "drafter_draft_rpc_latency_ms": f"{drafter_draft_rpc_latency_ms:.4f}",
                "drafter_commit_rpc_latency_ms": f"{drafter_commit_rpc_latency_ms:.4f}",
                "verifier_rpc_latency_ms": f"{verifier_rpc_latency_ms:.4f}",
                "client_rpc_latency_ms": f"{client_rpc_latency_ms:.4f}",
                "server_compute_latency_ms": f"{server_compute_latency_ms:.4f}",
                "estimated_rpc_overhead_ms": f"{estimated_rpc_overhead_ms:.4f}",
                "rpc_request_bytes": rpc_request_bytes,
                "rpc_response_bytes": rpc_response_bytes,
                "rpc_total_bytes": rpc_total_bytes,
                "rpc_bytes_per_generated_token": (
                    f"{rpc_total_bytes / len(generated_tokens):.6f}"
                    if generated_tokens
                    else ""
                ),
                "drafter_init_rpc_request_bytes": int(
                    rpc_bytes.get("drafter_init_request_bytes", 0)
                ),
                "drafter_init_rpc_response_bytes": int(
                    rpc_bytes.get("drafter_init_response_bytes", 0)
                ),
                "verifier_init_rpc_request_bytes": int(
                    rpc_bytes.get("verifier_init_request_bytes", 0)
                ),
                "verifier_init_rpc_response_bytes": int(
                    rpc_bytes.get("verifier_init_response_bytes", 0)
                ),
                "drafter_draft_rpc_request_bytes": int(
                    rpc_bytes.get("drafter_draft_request_bytes", 0)
                ),
                "drafter_draft_rpc_response_bytes": int(
                    rpc_bytes.get("drafter_draft_response_bytes", 0)
                ),
                "verifier_verify_rpc_request_bytes": int(
                    rpc_bytes.get("verifier_verify_request_bytes", 0)
                ),
                "verifier_verify_rpc_response_bytes": int(
                    rpc_bytes.get("verifier_verify_response_bytes", 0)
                ),
                "drafter_commit_rpc_request_bytes": int(
                    rpc_bytes.get("drafter_commit_request_bytes", 0)
                ),
                "drafter_commit_rpc_response_bytes": int(
                    rpc_bytes.get("drafter_commit_response_bytes", 0)
                ),
                "verifier_decode_rpc_request_bytes": "",
                "verifier_decode_rpc_response_bytes": "",
                "wall_latency_ms": f"{wall_ms:.4f}",
                "tokens_per_s": f"{tokens_per_s:.4f}",
                "drafter_prefill_energy_mj": f"{drafter_prefill_energy:.6f}" if rail else "",
                "drafter_draft_energy_mj": f"{drafter_draft_energy:.6f}" if rail else "",
                "drafter_commit_energy_mj": f"{drafter_commit_energy:.6f}" if rail else "",
                "verifier_prefill_energy_mj": f"{verifier_prefill_energy:.6f}" if rail else "",
                "verifier_verify_energy_mj": f"{verifier_verify_energy:.6f}" if rail else "",
                "role_total_energy_mj": f"{role_total_energy:.6f}" if rail else "",
                "drafter_prefill_total_energy_mj": f"{drafter_prefill_total:.6f}",
                "drafter_draft_total_energy_mj": f"{drafter_draft_total:.6f}",
                "drafter_commit_total_energy_mj": f"{drafter_commit_total:.6f}",
                "drafter_idle_total_energy_mj": "",
                "drafter_idle_energy_mj_per_generated_token": "",
                "verifier_prefill_total_energy_mj": f"{verifier_prefill_total:.6f}",
                "verifier_verify_total_energy_mj": f"{verifier_verify_total:.6f}",
                "drafter_total_energy_mj": f"{drafter_total:.6f}",
                "verifier_total_energy_mj": f"{verifier_total:.6f}",
                "system_total_energy_mj": f"{system_total:.6f}",
                "system_total_energy_mj_per_generated_token": (
                    f"{system_total / len(generated_tokens):.6f}"
                    if rail and generated_tokens
                    else ""
                ),
                "idle_baseline_s": fmt_optional(idle_baseline.get("idle_baseline_s")),
                "idle_baseline_policy": args.idle_baseline_policy,
                "drafter_idle_power_mw": fmt_optional(drafter_idle_power_mw),
                "verifier_idle_power_mw": fmt_optional(verifier_idle_power_mw),
                "system_idle_power_mw": fmt_optional(
                    (
                        float(drafter_idle_power_mw) + float(verifier_idle_power_mw)
                        if drafter_idle_power_mw is not None
                        and verifier_idle_power_mw is not None
                        else None
                    )
                ),
                "drafter_idle_power_samples": int(
                    idle_baseline.get("drafter_idle_power_samples") or 0
                ),
                "verifier_idle_power_samples": int(
                    idle_baseline.get("verifier_idle_power_samples") or 0
                ),
                "drafter_prefill_active_energy_mj": fmt_optional(
                    drafter_prefill_active_energy
                ),
                "drafter_draft_active_energy_mj": fmt_optional(
                    drafter_draft_active_energy
                ),
                "drafter_commit_active_energy_mj": fmt_optional(
                    drafter_commit_active_energy
                ),
                "verifier_prefill_active_energy_mj": fmt_optional(
                    verifier_prefill_active_energy
                ),
                "verifier_verify_active_energy_mj": fmt_optional(
                    verifier_verify_active_energy
                ),
                "drafter_active_energy_mj": fmt_optional(drafter_active_energy),
                "verifier_active_energy_mj": fmt_optional(verifier_active_energy),
                "system_active_energy_mj": fmt_optional(system_active_energy),
                "system_active_energy_mj_per_generated_token": (
                    fmt_optional(system_active_energy / len(generated_tokens))
                    if system_active_energy is not None and generated_tokens
                    else ""
                ),
                "n_power_samples": total_samples,
                "drafter_prefill_power_samples": drafter_prefill_power_samples,
                "drafter_draft_power_samples": drafter_draft_power_samples,
                "drafter_commit_power_samples": drafter_commit_power_samples,
                "verifier_prefill_power_samples": verifier_prefill_power_samples,
                "verifier_verify_power_samples": verifier_verify_power_samples,
                "verifier_decode_power_samples": "",
                "drafter_power_samples": drafter_power_samples,
                "verifier_power_samples": verifier_power_samples,
                "drafter_energy_available": int(drafter_energy_available),
                "verifier_energy_available": int(verifier_energy_available),
                "system_energy_complete": int(system_energy_complete),
                "drafter_model": health_metadata.get("drafter_model", ""),
                "drafter_device": health_metadata.get("drafter_device", ""),
                "verifier_model": health_metadata.get("verifier_model", ""),
                "verifier_device": health_metadata.get("verifier_device", ""),
                "drafter_power_interval_s": health_metadata.get(
                    "drafter_power_interval_s",
                    "",
                ),
                "verifier_power_interval_s": health_metadata.get(
                    "verifier_power_interval_s",
                    "",
                ),
                "drafter_spec_rpc_schema_version": health_metadata.get(
                    "drafter_spec_rpc_schema_version",
                    "",
                ),
                "verifier_spec_rpc_schema_version": health_metadata.get(
                    "verifier_spec_rpc_schema_version",
                    "",
                ),
                "drafter_model_vocab_size": health_metadata.get(
                    "drafter_model_vocab_size",
                    "",
                ),
                "verifier_model_vocab_size": health_metadata.get(
                    "verifier_model_vocab_size",
                    "",
                ),
                "drafter_model_parameter_count": health_metadata.get(
                    "drafter_model_parameter_count",
                    "",
                ),
                "verifier_model_parameter_count": health_metadata.get(
                    "verifier_model_parameter_count",
                    "",
                ),
                "drafter_model_type": health_metadata.get("drafter_model_type", ""),
                "verifier_model_type": health_metadata.get("verifier_model_type", ""),
                "drafter_model_architectures": health_metadata.get(
                    "drafter_model_architectures",
                    "",
                ),
                "verifier_model_architectures": health_metadata.get(
                    "verifier_model_architectures",
                    "",
                ),
                "drafter_model_bos_token_id": health_metadata.get(
                    "drafter_model_bos_token_id",
                    "",
                ),
                "drafter_model_eos_token_id": health_metadata.get(
                    "drafter_model_eos_token_id",
                    "",
                ),
                "drafter_model_pad_token_id": health_metadata.get(
                    "drafter_model_pad_token_id",
                    "",
                ),
                "verifier_model_bos_token_id": health_metadata.get(
                    "verifier_model_bos_token_id",
                    "",
                ),
                "verifier_model_eos_token_id": health_metadata.get(
                    "verifier_model_eos_token_id",
                    "",
                ),
                "verifier_model_pad_token_id": health_metadata.get(
                    "verifier_model_pad_token_id",
                    "",
                ),
                "drafter_runtime_fingerprint": runtime_fingerprint_from_metadata(
                    health_metadata,
                    "drafter",
                ),
                "verifier_runtime_fingerprint": runtime_fingerprint_from_metadata(
                    health_metadata,
                    "verifier",
                ),
                "drafter_hostname": health_metadata.get("drafter_hostname", ""),
                "verifier_hostname": health_metadata.get("verifier_hostname", ""),
                "drafter_pod_name": health_metadata.get("drafter_pod_name", ""),
                "verifier_pod_name": health_metadata.get("verifier_pod_name", ""),
                "drafter_pod_namespace": health_metadata.get(
                    "drafter_pod_namespace", ""
                ),
                "verifier_pod_namespace": health_metadata.get(
                    "verifier_pod_namespace", ""
                ),
                "drafter_node_name": health_metadata.get("drafter_node_name", ""),
                "verifier_node_name": health_metadata.get("verifier_node_name", ""),
                "drafter_cuda_visible_devices": health_metadata.get(
                    "drafter_cuda_visible_devices", ""
                ),
                "verifier_cuda_visible_devices": health_metadata.get(
                    "verifier_cuda_visible_devices", ""
                ),
                "drafter_nvidia_visible_devices": health_metadata.get(
                    "drafter_nvidia_visible_devices", ""
                ),
                "verifier_nvidia_visible_devices": health_metadata.get(
                    "verifier_nvidia_visible_devices", ""
                ),
                "drafter_torch_version": health_metadata.get("drafter_torch_version", ""),
                "verifier_torch_version": health_metadata.get("verifier_torch_version", ""),
                "drafter_transformers_version": health_metadata.get(
                    "drafter_transformers_version",
                    "",
                ),
                "verifier_transformers_version": health_metadata.get(
                    "verifier_transformers_version",
                    "",
                ),
                "drafter_cuda_version": health_metadata.get("drafter_cuda_version", ""),
                "verifier_cuda_version": health_metadata.get("verifier_cuda_version", ""),
                "drafter_gpu_name": health_metadata.get("drafter_gpu_name", ""),
                "verifier_gpu_name": health_metadata.get("verifier_gpu_name", ""),
                "drafter_xronos_git_commit": health_metadata.get(
                    "drafter_xronos_git_commit",
                    "",
                ),
                "verifier_xronos_git_commit": health_metadata.get(
                    "verifier_xronos_git_commit",
                    "",
                ),
                "drafter_xronos_image": health_metadata.get("drafter_xronos_image", ""),
                "verifier_xronos_image": health_metadata.get("verifier_xronos_image", ""),
                "drafter_jetson_gpu_freq_hz": health_metadata.get(
                    "drafter_jetson_gpu_freq_hz", ""
                ),
                "verifier_gpu_clock_mhz": health_metadata.get(
                    "verifier_nvidia_smi_gpu_clock_mhz", ""
                ),
                "drafter_frequency_label": health_metadata.get("drafter_frequency_label", ""),
                "verifier_frequency_label": health_metadata.get("verifier_frequency_label", ""),
                "drafter_frequency_lock_ok": health_metadata.get(
                    "drafter_frequency_lock_ok",
                    "",
                ),
                "verifier_frequency_lock_ok": health_metadata.get(
                    "verifier_frequency_lock_ok",
                    "",
                ),
                **runtime_status_fields(health_metadata, "drafter"),
                **runtime_status_fields(health_metadata, "verifier"),
                **driver_metadata_fields(driver_metadata),
                "output_text": output_text,
            }
        )
    return rows


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "result_schema_version",
        "experiment",
        "algorithm",
        "algorithm_version",
        "system_boundary",
        "plan_sha256",
        "plan_design_sha256",
        "session_id",
        "decoding_mode",
        "prompt_id",
        "prompt_source",
        "prompt_sha256",
        "prompt_set_sha256",
        "prompt_chars",
        "max_new_tokens",
        "gamma",
        "run",
        "measurement_order",
        "drafter_freq_hz",
        "verifier_clock_mhz",
        "rail",
        "drafter_primary_power_rail",
        "verifier_primary_power_rail",
        "system_primary_power_rails",
        "prompt_tokens",
        "prompt_token_sha256",
        "generated_tokens",
        "stop_reason",
        "stop_token_policy",
        "stop_token_ids",
        "output_token_sha256",
        "steps",
        "draft_tokens",
        "accepted_draft_tokens",
        "replacement_tokens",
        "accept_rate",
        "drafter_init_latency_ms",
        "verifier_init_latency_ms",
        "drafter_draft_latency_ms",
        "drafter_commit_latency_ms",
        "verifier_latency_ms",
        "drafter_init_rpc_latency_ms",
        "verifier_init_rpc_latency_ms",
        "drafter_draft_rpc_latency_ms",
        "drafter_commit_rpc_latency_ms",
        "verifier_rpc_latency_ms",
        "client_rpc_latency_ms",
        "server_compute_latency_ms",
        "estimated_rpc_overhead_ms",
        "rpc_request_bytes",
        "rpc_response_bytes",
        "rpc_total_bytes",
        "rpc_bytes_per_generated_token",
        "drafter_init_rpc_request_bytes",
        "drafter_init_rpc_response_bytes",
        "verifier_init_rpc_request_bytes",
        "verifier_init_rpc_response_bytes",
        "drafter_draft_rpc_request_bytes",
        "drafter_draft_rpc_response_bytes",
        "verifier_verify_rpc_request_bytes",
        "verifier_verify_rpc_response_bytes",
        "drafter_commit_rpc_request_bytes",
        "drafter_commit_rpc_response_bytes",
        "verifier_decode_rpc_request_bytes",
        "verifier_decode_rpc_response_bytes",
        "wall_latency_ms",
        "tokens_per_s",
        "drafter_prefill_energy_mj",
        "drafter_draft_energy_mj",
        "drafter_commit_energy_mj",
        "verifier_prefill_energy_mj",
        "verifier_verify_energy_mj",
        "role_total_energy_mj",
        "drafter_prefill_total_energy_mj",
        "drafter_draft_total_energy_mj",
        "drafter_commit_total_energy_mj",
        "drafter_idle_total_energy_mj",
        "drafter_idle_energy_mj_per_generated_token",
        "verifier_prefill_total_energy_mj",
        "verifier_verify_total_energy_mj",
        "drafter_total_energy_mj",
        "verifier_total_energy_mj",
        "system_total_energy_mj",
        "system_total_energy_mj_per_generated_token",
        "idle_baseline_s",
        "idle_baseline_policy",
        "drafter_idle_power_mw",
        "verifier_idle_power_mw",
        "system_idle_power_mw",
        "drafter_idle_power_samples",
        "verifier_idle_power_samples",
        "drafter_prefill_active_energy_mj",
        "drafter_draft_active_energy_mj",
        "drafter_commit_active_energy_mj",
        "verifier_prefill_active_energy_mj",
        "verifier_verify_active_energy_mj",
        "drafter_active_energy_mj",
        "verifier_active_energy_mj",
        "system_active_energy_mj",
        "system_active_energy_mj_per_generated_token",
        "n_power_samples",
        "drafter_prefill_power_samples",
        "drafter_draft_power_samples",
        "drafter_commit_power_samples",
        "verifier_prefill_power_samples",
        "verifier_verify_power_samples",
        "verifier_decode_power_samples",
        "drafter_power_samples",
        "verifier_power_samples",
        "drafter_energy_available",
        "verifier_energy_available",
        "system_energy_complete",
        "drafter_model",
        "drafter_device",
        "verifier_model",
        "verifier_device",
        "drafter_power_interval_s",
        "verifier_power_interval_s",
        "drafter_spec_rpc_schema_version",
        "verifier_spec_rpc_schema_version",
        "drafter_model_vocab_size",
        "verifier_model_vocab_size",
        "drafter_model_parameter_count",
        "verifier_model_parameter_count",
        "drafter_model_type",
        "verifier_model_type",
        "drafter_model_architectures",
        "verifier_model_architectures",
        "drafter_model_bos_token_id",
        "drafter_model_eos_token_id",
        "drafter_model_pad_token_id",
        "verifier_model_bos_token_id",
        "verifier_model_eos_token_id",
        "verifier_model_pad_token_id",
        "drafter_runtime_fingerprint",
        "verifier_runtime_fingerprint",
        "drafter_hostname",
        "verifier_hostname",
        "drafter_pod_name",
        "verifier_pod_name",
        "drafter_pod_namespace",
        "verifier_pod_namespace",
        "drafter_node_name",
        "verifier_node_name",
        "drafter_cuda_visible_devices",
        "verifier_cuda_visible_devices",
        "drafter_nvidia_visible_devices",
        "verifier_nvidia_visible_devices",
        "drafter_torch_version",
        "verifier_torch_version",
        "drafter_transformers_version",
        "verifier_transformers_version",
        "drafter_cuda_version",
        "verifier_cuda_version",
        "drafter_gpu_name",
        "verifier_gpu_name",
        "drafter_xronos_git_commit",
        "verifier_xronos_git_commit",
        "drafter_xronos_image",
        "verifier_xronos_image",
        "drafter_jetson_gpu_freq_hz",
        "verifier_gpu_clock_mhz",
        "drafter_frequency_label",
        "verifier_frequency_label",
        "drafter_frequency_lock_ok",
        "verifier_frequency_lock_ok",
        "drafter_runtime_temp_c",
        "verifier_runtime_temp_c",
        "drafter_thermal_zones",
        "verifier_thermal_zones",
        "drafter_nvidia_pstate",
        "verifier_nvidia_pstate",
        "drafter_nvidia_throttle_active",
        "verifier_nvidia_throttle_active",
        "tokenizer_name_or_path",
        "tokenizer_class",
        "tokenizer_vocab_size",
        "tokenizer_base_vocab_size",
        "tokenizer_bos_token_id",
        "tokenizer_eos_token_id",
        "tokenizer_pad_token_id",
        "tokenizer_unk_token_id",
        "driver_hostname",
        "driver_pod_name",
        "driver_pod_namespace",
        "driver_node_name",
        "driver_python_version",
        "driver_platform",
        "driver_result_schema_version",
        "driver_spec_rpc_schema_version",
        "driver_git_branch",
        "driver_git_commit",
        "driver_git_dirty",
        "driver_xronos_git_commit",
        "driver_xronos_image",
        "driver_command_sha256",
        "driver_plan_sha256",
        "driver_plan_design_sha256",
        "driver_cuda_visible_devices",
        "driver_nvidia_visible_devices",
        "output_text",
    ]
    ensure_parent_dir(path)
    output_path = Path(path)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, output_path)


async def run(args: argparse.Namespace) -> None:
    validate_args(args)
    prompt_cases = load_prompt_cases(args, allow_missing=args.dry_run)
    validate_prompt_cases(prompt_cases)
    conditions = build_spec_conditions(args, prompt_cases)
    prompt_set_sha256 = prompt_set_hash(prompt_cases)
    plan = build_sweep_plan(args, prompt_cases, conditions)
    driver_metadata = {
        **plan["metadata"],
        "plan_sha256": plan["plan_sha256"],
        "plan_design_sha256": plan["plan_design_sha256"],
    }
    if args.plan_out:
        write_json(args.plan_out, plan)
        print(f"Wrote sweep plan to {args.plan_out}")
    if args.dry_run:
        print_sweep_plan(plan)
        return

    resume_rows, completed_measurement_orders = load_resume_rows(
        args,
        plan,
        expected_algorithm="speculative",
    )
    prepare_trace_output(args, plan, resume_rows)

    load_grpc_bindings()
    from xronos.infer.modeling import load_tokenizer, tokenizer_metadata

    tokenizer = load_tokenizer(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    driver_metadata = {
        **driver_metadata,
        **tokenizer_metadata(tokenizer),
    }

    max_message_bytes = args.max_message_mb * 1024 * 1024
    channel_options = [
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.max_receive_message_length", max_message_bytes),
    ]
    async with grpc.aio.insecure_channel(
        args.drafter_addr,
        options=channel_options,
    ) as drafter_channel, grpc.aio.insecure_channel(
        args.verifier_addr,
        options=channel_options,
    ) as verifier_channel:
        drafter_stub = spec_pb2_grpc.DrafterStub(drafter_channel)
        verifier_stub = spec_pb2_grpc.VerifierStub(verifier_channel)
        health_metadata = {}
        health_metadata.update(
            await wait_for_health(
                "drafter",
                drafter_stub,
                args.timeout,
                args.startup_timeout_s,
                args.health_check_interval_s,
            )
        )
        health_metadata.update(
            await wait_for_health(
                "verifier",
                verifier_stub,
                args.timeout,
                args.startup_timeout_s,
                args.health_check_interval_s,
            )
        )

        async def run_health_metadata(
            combo_metadata: Dict[str, str],
            force: bool = False,
        ) -> Dict[str, str]:
            if not force and not args.sample_runtime_metadata:
                return combo_metadata
            refreshed = dict(combo_metadata)
            refreshed.update(
                await check_health(
                    "drafter",
                    drafter_stub,
                    args.timeout,
                    verbose=False,
                )
            )
            refreshed.update(
                await check_health(
                    "verifier",
                    verifier_stub,
                    args.timeout,
                    verbose=False,
                )
            )
            return refreshed

        async def ready_run_metadata(
            combo_metadata: Dict[str, str],
            label: str,
        ) -> Dict[str, str]:
            initial = await run_health_metadata(combo_metadata)
            return await thermal_guard_metadata(
                args=args,
                initial_metadata=initial,
                refresh_metadata=lambda: run_health_metadata(combo_metadata, force=True),
                prefixes=["drafter", "verifier"],
                label=label,
            )

        async def prepare_condition(condition: SpecCondition):
            prompt_case = condition.prompt_case
            drafter_freq_hz = condition.drafter_freq_hz
            verifier_clock_mhz = condition.verifier_clock_mhz
            combo_metadata = dict(health_metadata)
            if drafter_freq_hz is not None:
                combo_metadata.update(
                    await set_frequency(
                        "drafter",
                        drafter_stub,
                        args.timeout,
                        jetson_gpu_freq_hz=drafter_freq_hz,
                    )
                )
            if verifier_clock_mhz is not None:
                combo_metadata.update(
                    await set_frequency(
                        "verifier",
                        verifier_stub,
                        args.timeout,
                        nvidia_smi_gpu_clock_mhz=verifier_clock_mhz,
                    )
                )
            if args.idle_baseline_policy == "condition":
                idle_baseline = await sample_idle_baseline(
                    args=args,
                    drafter_stub=drafter_stub,
                    verifier_stub=verifier_stub,
                    drafter_freq_hz=drafter_freq_hz,
                    verifier_clock_mhz=verifier_clock_mhz,
                )
            else:
                idle_baseline = empty_idle_baseline()
            return combo_metadata, idle_baseline

        rows: List[Dict[str, object]] = list(resume_rows)
        measurement_order = 0
        if not args.shuffle_runs:
            for condition in conditions:
                prompt_case = condition.prompt_case
                drafter_freq_hz = condition.drafter_freq_hz
                verifier_clock_mhz = condition.verifier_clock_mhz
                gamma = condition.gamma
                combo_metadata, idle_baseline = await prepare_condition(condition)
                for _ in range(args.warmup_runs):
                    await run_decode(
                        args=args,
                        tokenizer=tokenizer,
                        drafter_stub=drafter_stub,
                        verifier_stub=verifier_stub,
                        gamma=gamma,
                        run_index=0,
                        health_metadata=combo_metadata,
                        drafter_freq_hz=drafter_freq_hz,
                        verifier_clock_mhz=verifier_clock_mhz,
                        prompt_case=prompt_case,
                        prompt_set_sha256=prompt_set_sha256,
                        idle_baseline=idle_baseline,
                        measurement_order=0,
                        driver_metadata=driver_metadata,
                    )
                for run_index in range(1, args.runs + 1):
                    measurement_order += 1
                    if measurement_order in completed_measurement_orders:
                        print(
                            "Skipping completed spec measurement "
                            f"order={measurement_order} run={run_index}."
                        )
                        continue
                    run_rows = await run_decode(
                        args=args,
                        tokenizer=tokenizer,
                        drafter_stub=drafter_stub,
                        verifier_stub=verifier_stub,
                        gamma=gamma,
                        run_index=run_index,
                        health_metadata=await ready_run_metadata(
                            combo_metadata,
                            f"spec run {measurement_order}",
                        ),
                        drafter_freq_hz=drafter_freq_hz,
                        verifier_clock_mhz=verifier_clock_mhz,
                        prompt_case=prompt_case,
                        prompt_set_sha256=prompt_set_sha256,
                        idle_baseline=await idle_baseline_for_measured_run(
                            args=args,
                            condition_idle_baseline=idle_baseline,
                            drafter_stub=drafter_stub,
                            verifier_stub=verifier_stub,
                            drafter_freq_hz=drafter_freq_hz,
                            verifier_clock_mhz=verifier_clock_mhz,
                        ),
                        measurement_order=measurement_order,
                        driver_metadata=driver_metadata,
                    )
                    rows.extend(run_rows)
                    if args.out:
                        write_rows(args.out, rows)
                    total_row = next(
                        (row for row in run_rows if row["rail"] == "tot_power"),
                        run_rows[0],
                    )
                    print(
                        "prompt={prompt_id} fd={drafter_freq_hz} "
                        "fv={verifier_clock_mhz} gamma={gamma} run={run} "
                        "order={measurement_order} generated={generated_tokens} "
                        "accept={accept_rate} wall_ms={wall_latency_ms} "
                        "drafter_total_mj={drafter_total_energy_mj}".format(
                            **total_row
                        )
                    )
        else:
            prepared = {}
            for condition_index, condition in enumerate(conditions):
                combo_metadata, idle_baseline = await prepare_condition(condition)
                prepared[condition_index] = (combo_metadata, idle_baseline)

            current_drafter_freq = None
            current_verifier_clock = None
            for entry in build_spec_warmup_schedule(args, conditions):
                condition = conditions[entry.condition_index]
                combo_metadata, idle_baseline = prepared[entry.condition_index]
                combo_metadata = dict(combo_metadata)
                if (
                    condition.drafter_freq_hz is not None
                    and condition.drafter_freq_hz != current_drafter_freq
                ):
                    combo_metadata.update(
                        await set_frequency(
                            "drafter",
                            drafter_stub,
                            args.timeout,
                            jetson_gpu_freq_hz=condition.drafter_freq_hz,
                        )
                    )
                    current_drafter_freq = condition.drafter_freq_hz
                if (
                    condition.verifier_clock_mhz is not None
                    and condition.verifier_clock_mhz != current_verifier_clock
                ):
                    combo_metadata.update(
                        await set_frequency(
                            "verifier",
                            verifier_stub,
                            args.timeout,
                            nvidia_smi_gpu_clock_mhz=condition.verifier_clock_mhz,
                        )
                    )
                    current_verifier_clock = condition.verifier_clock_mhz
                await run_decode(
                    args=args,
                    tokenizer=tokenizer,
                    drafter_stub=drafter_stub,
                    verifier_stub=verifier_stub,
                    gamma=condition.gamma,
                    run_index=0,
                    health_metadata=combo_metadata,
                    drafter_freq_hz=condition.drafter_freq_hz,
                    verifier_clock_mhz=condition.verifier_clock_mhz,
                    prompt_case=condition.prompt_case,
                    prompt_set_sha256=prompt_set_sha256,
                    idle_baseline=idle_baseline,
                    measurement_order=0,
                    driver_metadata=driver_metadata,
                )

            current_drafter_freq = None
            current_verifier_clock = None
            for measurement_order, entry in enumerate(
                build_spec_run_schedule(args, conditions),
                start=1,
            ):
                condition = conditions[entry.condition_index]
                if measurement_order in completed_measurement_orders:
                    print(
                        "Skipping completed spec measurement "
                        f"order={measurement_order} run={entry.run_index}."
                    )
                    continue
                combo_metadata, idle_baseline = prepared[entry.condition_index]
                combo_metadata = dict(combo_metadata)
                if (
                    condition.drafter_freq_hz is not None
                    and condition.drafter_freq_hz != current_drafter_freq
                ):
                    combo_metadata.update(
                        await set_frequency(
                            "drafter",
                            drafter_stub,
                            args.timeout,
                            jetson_gpu_freq_hz=condition.drafter_freq_hz,
                        )
                    )
                    current_drafter_freq = condition.drafter_freq_hz
                if (
                    condition.verifier_clock_mhz is not None
                    and condition.verifier_clock_mhz != current_verifier_clock
                ):
                    combo_metadata.update(
                        await set_frequency(
                            "verifier",
                            verifier_stub,
                            args.timeout,
                            nvidia_smi_gpu_clock_mhz=condition.verifier_clock_mhz,
                        )
                    )
                    current_verifier_clock = condition.verifier_clock_mhz
                run_rows = await run_decode(
                    args=args,
                    tokenizer=tokenizer,
                    drafter_stub=drafter_stub,
                    verifier_stub=verifier_stub,
                    gamma=condition.gamma,
                    run_index=entry.run_index,
                    health_metadata=await ready_run_metadata(
                        combo_metadata,
                        f"spec run {measurement_order}",
                    ),
                    drafter_freq_hz=condition.drafter_freq_hz,
                    verifier_clock_mhz=condition.verifier_clock_mhz,
                    prompt_case=condition.prompt_case,
                    prompt_set_sha256=prompt_set_sha256,
                    idle_baseline=await idle_baseline_for_measured_run(
                        args=args,
                        condition_idle_baseline=idle_baseline,
                        drafter_stub=drafter_stub,
                        verifier_stub=verifier_stub,
                        drafter_freq_hz=condition.drafter_freq_hz,
                        verifier_clock_mhz=condition.verifier_clock_mhz,
                    ),
                    measurement_order=measurement_order,
                    driver_metadata=driver_metadata,
                )
                rows.extend(run_rows)
                if args.out:
                    write_rows(args.out, rows)
                total_row = next(
                    (row for row in run_rows if row["rail"] == "tot_power"),
                    run_rows[0],
                )
                print(
                    "prompt={prompt_id} fd={drafter_freq_hz} "
                    "fv={verifier_clock_mhz} gamma={gamma} run={run} "
                    "order={measurement_order} generated={generated_tokens} "
                    "accept={accept_rate} wall_ms={wall_latency_ms} "
                    "drafter_total_mj={drafter_total_energy_mj}".format(
                        **total_row
                    )
                )

    if args.out:
        write_rows(args.out, rows)
        print(f"Wrote {len(rows)} rows to {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed speculative decoding")
    parser.add_argument("--drafter-addr", help="host:port for Jetson drafter")
    parser.add_argument("--verifier-addr", help="host:port for verifier")
    parser.add_argument("--tokenizer", help="Shared tokenizer id/path")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument(
        "--prompts-jsonl",
        help="JSONL prompt set. Each line can be a string or {'id': ..., 'prompt': ...}.",
    )
    parser.add_argument("--gammas", default="1,2,4,8,16")
    parser.add_argument(
        "--drafter-freqs-hz",
        default="",
        help="Optional comma-separated Jetson GPU frequencies to sweep in Hz",
    )
    parser.add_argument(
        "--verifier-clocks-mhz",
        default="",
        help="Optional comma-separated verifier GPU graphics clocks to sweep in MHz",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--stop-token-ids",
        default="",
        help=(
            "Optional comma-separated token ids that stop generation. "
            "Defaults to the tokenizer eos_token_id value(s)."
        ),
    )
    parser.add_argument("--decoding-mode", choices=["greedy"], default="greedy")
    parser.add_argument(
        "--idle-baseline-s",
        type=float,
        default=0.0,
        help="Optional idle power sampling duration before each prompt/frequency combo.",
    )
    parser.add_argument(
        "--idle-baseline-policy",
        choices=["condition", "run"],
        default="condition",
        help=(
            "When --idle-baseline-s is positive, sample idle power once per "
            "condition or immediately before every measured run."
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for drafter/verifier health before a sweep starts.",
    )
    parser.add_argument(
        "--health-check-interval-s",
        type=float,
        default=5.0,
        help="Polling interval while waiting for startup health checks.",
    )
    parser.add_argument(
        "--shuffle-conditions",
        action="store_true",
        help="Shuffle prompt/frequency/gamma conditions to reduce order bias.",
    )
    parser.add_argument(
        "--shuffle-runs",
        action="store_true",
        help="Shuffle measured run order across conditions to reduce drift/order bias.",
    )
    parser.add_argument(
        "--sample-runtime-metadata",
        action="store_true",
        help="Refresh server health metadata before each measured run.",
    )
    parser.add_argument(
        "--max-start-temp-c",
        type=float,
        help="Wait before each measured run until recorded runtime temperature is at or below this value.",
    )
    parser.add_argument(
        "--thermal-check-interval-s",
        type=float,
        default=5.0,
        help="Polling interval while waiting for --max-start-temp-c.",
    )
    parser.add_argument(
        "--thermal-wait-timeout-s",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for --max-start-temp-c before failing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for condition and measured-run shuffling.",
    )
    parser.add_argument("--out", default="spec_gamma_sweep.csv")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load existing --out rows, verify they match the current plan, and "
            "skip completed measured orders."
        ),
    )
    parser.add_argument("--experiment", default="spec_gamma_sweep")
    parser.add_argument("--max-message-mb", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep matrix without connecting to services or loading models.",
    )
    parser.add_argument(
        "--plan-out",
        default="",
        help="Optional JSON path for sweep metadata.",
    )
    parser.add_argument(
        "--trace-out",
        default="",
        help="Optional JSONL path for per-step debug traces.",
    )
    parser.add_argument(
        "--trace-warmups",
        action="store_true",
        help="Include warmup sessions in --trace-out.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-special-tokens", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
