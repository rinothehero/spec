import argparse
import asyncio
import csv
import random
import time
import uuid
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Dict, List, Optional

from xronos.infer import spec_driver


@dataclass(frozen=True)
class BaselineCondition:
    prompt_case: spec_driver.PromptCase
    drafter_freq_hz: Optional[int]
    verifier_clock_mhz: Optional[int]


@dataclass(frozen=True)
class BaselineRunScheduleEntry:
    condition_index: int
    run_index: int


@dataclass(frozen=True)
class BaselineWarmupScheduleEntry:
    condition_index: int
    warmup_index: int


def build_baseline_plan(
    args: argparse.Namespace,
    prompt_cases: List[spec_driver.PromptCase],
    conditions: List[BaselineCondition],
) -> Dict[str, object]:
    drafter_freqs_hz = baseline_drafter_freqs(args)
    verifier_clocks_mhz = spec_driver.parse_optional_int_csv(args.verifier_clocks_mhz)
    stop_policy = spec_driver.stop_token_policy(args)
    stop_token_ids_arg = spec_driver.format_token_ids(
        spec_driver.unique_token_ids(spec_driver.parse_stop_token_ids(args.stop_token_ids))
    )
    combinations = [
        {
            "order": order,
            "prompt_id": condition.prompt_case.prompt_id,
            "prompt_sha256": condition.prompt_case.sha256,
            "drafter_freq_hz": condition.drafter_freq_hz,
            "verifier_clock_mhz": condition.verifier_clock_mhz,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
        }
        for order, condition in enumerate(conditions)
    ]
    measurement_schedule = build_baseline_run_schedule(args, conditions)
    warmup_schedule = build_baseline_warmup_schedule(args, conditions)
    if args.idle_baseline_s <= 0:
        total_idle_baselines = 0
    elif args.idle_baseline_policy == "run":
        total_idle_baselines = len(measurement_schedule)
    else:
        total_idle_baselines = len(combinations)
    plan = {
        "schema_version": spec_driver.RESULT_SCHEMA_VERSION,
        "experiment": args.experiment,
        "algorithm": "verifier_only",
        "algorithm_version": spec_driver.BASELINE_ALGORITHM_VERSION,
        "system_boundary": baseline_system_boundary(args),
        "drafter_addr": args.drafter_addr or "",
        "verifier_addr": args.verifier_addr or "",
        "tokenizer": args.tokenizer or "",
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
        "prompt_set_sha256": spec_driver.prompt_set_hash(prompt_cases),
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
        "combinations": combinations,
        "measurement_schedule": [
            {
                "order": order,
                "condition_order": entry.condition_index,
                "run": entry.run_index,
                "prompt_id": conditions[entry.condition_index].prompt_case.prompt_id,
                "prompt_sha256": conditions[entry.condition_index].prompt_case.sha256,
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
                "drafter_freq_hz": conditions[entry.condition_index].drafter_freq_hz,
                "verifier_clock_mhz": conditions[entry.condition_index].verifier_clock_mhz,
            }
            for order, entry in enumerate(warmup_schedule, start=1)
        ],
        "total_warmup_sessions": len(combinations) * args.warmup_runs,
        "total_measured_sessions": len(combinations) * args.runs,
        "total_idle_baselines": total_idle_baselines,
        "metadata": spec_driver.collect_metadata(),
    }
    return spec_driver.attach_plan_sha256(plan)


def baseline_system_boundary(args: argparse.Namespace) -> str:
    return "two_device_idle_drafter" if getattr(args, "drafter_addr", None) else "verifier_only"


def baseline_drafter_freqs(args: argparse.Namespace) -> List[Optional[int]]:
    if not getattr(args, "drafter_addr", None):
        return [None]
    return spec_driver.parse_optional_int_csv(args.drafter_freqs_hz)


def build_baseline_conditions(
    args: argparse.Namespace,
    prompt_cases: List[spec_driver.PromptCase],
) -> List[BaselineCondition]:
    conditions: List[BaselineCondition] = []
    for prompt_case in prompt_cases:
        for drafter_freq_hz in baseline_drafter_freqs(args):
            for verifier_clock_mhz in spec_driver.parse_optional_int_csv(
                args.verifier_clocks_mhz
            ):
                conditions.append(
                    BaselineCondition(
                        prompt_case=prompt_case,
                        drafter_freq_hz=drafter_freq_hz,
                        verifier_clock_mhz=verifier_clock_mhz,
                    )
                )
    if args.shuffle_conditions:
        random.Random(args.seed).shuffle(conditions)
    return conditions


def build_baseline_run_schedule(
    args: argparse.Namespace,
    conditions: List[BaselineCondition],
) -> List[BaselineRunScheduleEntry]:
    schedule = [
        BaselineRunScheduleEntry(condition_index=condition_index, run_index=run_index)
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


def build_baseline_warmup_schedule(
    args: argparse.Namespace,
    conditions: List[BaselineCondition],
) -> List[BaselineWarmupScheduleEntry]:
    schedule = [
        BaselineWarmupScheduleEntry(
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


def print_plan(plan: Dict[str, object]) -> None:
    print(f"experiment={plan['experiment']}")
    print(f"algorithm={plan['algorithm']}")
    print(f"prompts={plan['prompt_count']}")
    print(f"combinations={len(plan['combinations'])}")
    print(f"warmup_sessions={plan['total_warmup_sessions']}")
    print(f"measured_sessions={plan['total_measured_sessions']}")
    print(f"max_new_tokens={plan['max_new_tokens']}")
    print(f"plan_sha256={plan.get('plan_sha256', '')}")
    print(f"system_boundary={plan.get('system_boundary', '')}")
    print(f"shuffle_runs={int(bool(plan.get('shuffle_runs', False)))}")
    for combo in plan["combinations"]:
        print(
            "prompt={prompt_id} fd={drafter_freq_hz} fv={verifier_clock_mhz} "
            "warmup={warmup_runs} runs={measured_runs}".format(**combo)
        )
    if plan.get("shuffle_runs"):
        for item in plan.get("warmup_schedule", [])[:20]:
            print(
                "warmup_order={order} condition={condition_order} warmup={warmup} "
                "prompt={prompt_id} fd={drafter_freq_hz} fv={verifier_clock_mhz}".format(**item)
            )
        for item in plan.get("measurement_schedule", [])[:20]:
            print(
                "schedule_order={order} condition={condition_order} run={run} "
                "prompt={prompt_id} fd={drafter_freq_hz} fv={verifier_clock_mhz}".format(**item)
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must not be negative.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than 0.")
    stop_token_ids = spec_driver.parse_stop_token_ids(args.stop_token_ids)
    if any(token < 0 for token in stop_token_ids):
        raise ValueError("--stop-token-ids values must not be negative.")
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
    if len(spec_driver.selected_prompt_sources(args)) > 1:
        raise ValueError("Use only one of --prompt, --prompt-file, or --prompts-jsonl.")
    if args.trace_warmups and not args.trace_out:
        raise ValueError("--trace-warmups requires --trace-out.")
    if args.dry_run:
        return
    if args.drafter_freqs_hz and not args.drafter_addr:
        raise ValueError("--drafter-freqs-hz requires --drafter-addr.")
    if args.drafter_addr and args.idle_baseline_s <= 0:
        raise ValueError(
            "--drafter-addr baseline accounting requires --idle-baseline-s > 0."
        )
    missing = [
        name
        for name in ("verifier_addr", "tokenizer")
        if not getattr(args, name)
    ]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Missing required argument(s): {joined}")
    if not spec_driver.selected_prompt_sources(args):
        raise ValueError("Provide --prompt, --prompt-file, or --prompts-jsonl.")


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


async def check_health(
    verifier_stub,
    timeout_s: float,
    verbose: bool = True,
) -> Dict[str, str]:
    response = await verifier_stub.Health(spec_driver.spec_pb2.HealthRequest(), timeout=timeout_s)
    if not response.ok:
        raise RuntimeError(f"verifier health check failed: {response.message}")
    metadata = {f"verifier_{key}": value for key, value in dict(response.metadata).items()}
    metadata["verifier_model"] = response.model
    metadata["verifier_device"] = response.device
    if verbose:
        print(f"verifier: {response.message} model={response.model} device={response.device}")
    return metadata


async def wait_for_health(
    verifier_stub,
    timeout_s: float,
    startup_timeout_s: float,
    interval_s: float,
    verbose: bool = True,
) -> Dict[str, str]:
    deadline = time.monotonic() + startup_timeout_s
    last_error = ""
    while True:
        try:
            return await check_health(verifier_stub, timeout_s, verbose=verbose)
        except Exception as exc:
            last_error = str(exc)
            if startup_timeout_s <= 0 or time.monotonic() >= deadline:
                raise RuntimeError(
                    "verifier health check did not become ready within "
                    f"{startup_timeout_s:.1f}s: {last_error}"
                ) from exc
            if verbose:
                print(
                    "verifier: waiting for health check "
                    f"({last_error}); retrying in {interval_s:.1f}s"
                )
            await asyncio.sleep(interval_s)


async def set_verifier_clock(
    verifier_stub,
    timeout_s: float,
    verifier_clock_mhz: Optional[int],
) -> Dict[str, str]:
    if verifier_clock_mhz is None:
        return {}
    response = await verifier_stub.SetFrequency(
        spec_driver.spec_pb2.SetFrequencyRequest(
            nvidia_smi_gpu_clock_mhz=verifier_clock_mhz,
        ),
        timeout=timeout_s,
    )
    if not response.ok:
        raise RuntimeError(f"verifier SetFrequency failed: {response.error}")
    print(f"verifier: {response.message} {dict(response.metadata)}")
    return {f"verifier_{key}": value for key, value in dict(response.metadata).items()}


async def sample_idle_baseline(
    args: argparse.Namespace,
    verifier_stub,
    drafter_stub,
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

    request = spec_driver.spec_pb2.IdlePowerRequest(duration_s=args.idle_baseline_s)
    timeout = max(args.timeout, args.idle_baseline_s + 10.0)
    if drafter_stub is None:
        drafter_response = None
        verifier_response = await verifier_stub.IdlePower(request, timeout=timeout)
    else:
        drafter_response, verifier_response = await asyncio.gather(
            drafter_stub.IdlePower(request, timeout=timeout),
            verifier_stub.IdlePower(request, timeout=timeout),
        )
        if drafter_response.error:
            raise RuntimeError(f"Drafter IdlePower error: {drafter_response.error}")
    if verifier_response.error:
        raise RuntimeError(f"Verifier IdlePower error: {verifier_response.error}")

    drafter_idle_power_mw = (
        spec_driver.rail_mean_power_mw(
            drafter_response,
            spec_driver.DRAFTER_PRIMARY_POWER_RAIL,
        )
        if drafter_response is not None
        else None
    )

    verifier_idle_power_mw = spec_driver.rail_mean_power_mw(
        verifier_response,
        spec_driver.VERIFIER_PRIMARY_POWER_RAIL,
    )
    baseline = {
        "idle_baseline_s": (
            max(drafter_response.duration_s, verifier_response.duration_s)
            if drafter_response is not None
            else verifier_response.duration_s
        ),
        "drafter_idle_power_mw": drafter_idle_power_mw,
        "verifier_idle_power_mw": verifier_idle_power_mw,
        "drafter_idle_power_samples": (
            int(drafter_response.n_power_samples)
            if drafter_response is not None
            else 0
        ),
        "verifier_idle_power_samples": int(verifier_response.n_power_samples),
    }
    print(
        "idle baseline fd={fd} fv={fv} drafter_mw={dmw} verifier_mw={vmw}".format(
            fd=drafter_freq_hz or "",
            fv=verifier_clock_mhz or "",
            dmw=spec_driver.fmt_optional(drafter_idle_power_mw),
            vmw=spec_driver.fmt_optional(verifier_idle_power_mw),
        )
    )
    if args.trace_out:
        spec_driver.append_jsonl(
            args.trace_out,
            {
                "event": "idle_baseline",
                "algorithm": "verifier_only",
                "system_boundary": baseline_system_boundary(args),
                "drafter_freq_hz": drafter_freq_hz,
                "verifier_clock_mhz": verifier_clock_mhz,
                "duration_s": baseline["idle_baseline_s"],
                "drafter_idle_power_mw": drafter_idle_power_mw,
                "verifier_idle_power_mw": verifier_idle_power_mw,
                "drafter_idle_power": (
                    spec_driver.rail_energy_map(drafter_response)
                    if drafter_response is not None
                    else {}
                ),
                "verifier_idle_power": spec_driver.rail_energy_map(verifier_response),
                "drafter_idle_power_samples": baseline["drafter_idle_power_samples"],
                "verifier_idle_power_samples": baseline["verifier_idle_power_samples"],
            },
        )
    return baseline


def empty_verifier_idle_baseline() -> Dict[str, Optional[float]]:
    return {
        "idle_baseline_s": None,
        "drafter_idle_power_mw": None,
        "verifier_idle_power_mw": None,
        "drafter_idle_power_samples": 0,
        "verifier_idle_power_samples": 0,
    }


async def verifier_idle_baseline_for_measured_run(
    args: argparse.Namespace,
    condition_idle_baseline: Dict[str, Optional[float]],
    verifier_stub,
    drafter_stub,
    drafter_freq_hz: Optional[int],
    verifier_clock_mhz: Optional[int],
) -> Dict[str, Optional[float]]:
    if args.idle_baseline_policy != "run":
        return condition_idle_baseline
    return await sample_idle_baseline(
        args=args,
        verifier_stub=verifier_stub,
        drafter_stub=drafter_stub,
        drafter_freq_hz=drafter_freq_hz,
        verifier_clock_mhz=verifier_clock_mhz,
    )


async def reset_session(session_id: str, verifier_stub, timeout_s: float) -> None:
    await verifier_stub.ResetSession(
        spec_driver.spec_pb2.ResetSessionRequest(session_id=session_id),
        timeout=timeout_s,
    )


async def run_baseline_decode(
    args: argparse.Namespace,
    tokenizer,
    verifier_stub,
    prompt_case: spec_driver.PromptCase,
    prompt_set_sha256: str,
    drafter_freq_hz: Optional[int],
    verifier_clock_mhz: Optional[int],
    run_index: int,
    health_metadata: Dict[str, str],
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
    prompt_token_sha256 = spec_driver.token_ids_sha256(prompt_tokens)

    session_id = f"verifier-baseline-{uuid.uuid4()}"
    stop_token_ids = spec_driver.resolved_stop_token_ids(args, tokenizer)
    stop_token_id_set = set(stop_token_ids)
    stop_token_ids_text = spec_driver.format_token_ids(stop_token_ids)
    stop_policy = spec_driver.stop_token_policy(args)
    energy_by_phase: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    samples_by_phase: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rpc_bytes: Dict[str, int] = defaultdict(int)
    wall_t0 = time.monotonic()

    try:
        init_request = spec_driver.spec_pb2.InitSessionRequest(
            session_id=session_id,
            context_tokens=prompt_tokens,
        )
        init_response, init_rpc_ms = await spec_driver.timed_rpc(
            verifier_stub.InitSession(
                init_request,
                timeout=args.timeout,
            )
        )
        if init_response.error:
            raise RuntimeError(f"Verifier InitSession error: {init_response.error}")
        spec_driver.add_rpc_bytes(
            rpc_bytes,
            "verifier_init",
            init_request,
            init_response,
        )
        add_power(init_response, energy_by_phase, samples_by_phase, "verifier_prefill")

        generate_request = spec_driver.spec_pb2.GenerateRequest(
            session_id=session_id,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        generate_response, generate_rpc_ms = await spec_driver.timed_rpc(
            verifier_stub.Generate(
                generate_request,
                timeout=args.timeout,
            )
        )
        if generate_response.error:
            raise RuntimeError(f"Verifier Generate error: {generate_response.error}")
        spec_driver.add_rpc_bytes(
            rpc_bytes,
            "verifier_decode",
            generate_request,
            generate_response,
        )
        add_power(generate_response, energy_by_phase, samples_by_phase, "verifier_decode")

        generated_tokens = list(generate_response.generated_tokens)
        if not generated_tokens:
            raise RuntimeError("Verifier Generate returned no tokens.")
        if stop_token_id_set and any(token in stop_token_id_set for token in generated_tokens):
            stop_reason = "stop"
        elif len(generated_tokens) >= args.max_new_tokens:
            stop_reason = "max_new_tokens"
        else:
            stop_reason = "server_stopped_early"
        output_token_sha256 = spec_driver.token_ids_sha256(generated_tokens)
        if args.trace_out and (run_index > 0 or args.trace_warmups):
            spec_driver.append_jsonl(
                args.trace_out,
                {
                    "event": "verifier_baseline_run",
                    "experiment": args.experiment,
                    "algorithm": "verifier_only",
                    "system_boundary": baseline_system_boundary(args),
                    "session_id": session_id,
                    "decoding_mode": args.decoding_mode,
                    "prompt_id": prompt_case.prompt_id,
                    "prompt_sha256": prompt_case.sha256,
                    "prompt_set_sha256": prompt_set_sha256,
                    "run": run_index,
                    "measurement_order": measurement_order,
                    "drafter_freq_hz": drafter_freq_hz,
                    "verifier_clock_mhz": verifier_clock_mhz,
                    "generated_tokens": len(generated_tokens),
                    "stop_reason": stop_reason,
                    "stop_token_policy": stop_policy,
                    "stop_token_ids": stop_token_ids_text,
                    "output_token_sha256": output_token_sha256,
                    "generated_token_ids": generated_tokens,
                    "verifier_decode_latency_ms": generate_response.latency_ms,
                    "verifier_init_rpc_latency_ms": init_rpc_ms,
                    "verifier_rpc_latency_ms": generate_rpc_ms,
                    "verifier_init_rpc_request_bytes": spec_driver.message_byte_size(
                        init_request,
                    ),
                    "verifier_init_rpc_response_bytes": spec_driver.message_byte_size(
                        init_response,
                    ),
                    "verifier_decode_rpc_request_bytes": spec_driver.message_byte_size(
                        generate_request,
                    ),
                    "verifier_decode_rpc_response_bytes": spec_driver.message_byte_size(
                        generate_response,
                    ),
                    "estimated_rpc_overhead_ms": (
                        init_rpc_ms
                        + generate_rpc_ms
                        - init_response.latency_ms
                        - generate_response.latency_ms
                    ),
                    "verifier_decode_power": spec_driver.rail_energy_map(generate_response),
                    "verifier_decode_power_samples": int(generate_response.n_power_samples),
                },
            )
    finally:
        await reset_session(session_id, verifier_stub, args.timeout)

    wall_ms = (time.monotonic() - wall_t0) * 1000.0
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    output_token_sha256 = spec_driver.token_ids_sha256(generated_tokens)
    server_compute_latency_ms = init_response.latency_ms + generate_response.latency_ms
    client_rpc_latency_ms = init_rpc_ms + generate_rpc_ms
    estimated_rpc_overhead_ms = client_rpc_latency_ms - server_compute_latency_ms
    rpc_request_bytes = int(rpc_bytes.get("request_bytes", 0))
    rpc_response_bytes = int(rpc_bytes.get("response_bytes", 0))
    rpc_total_bytes = rpc_request_bytes + rpc_response_bytes
    verifier_prefill_total = spec_driver.phase_total(
        energy_by_phase,
        "verifier_prefill",
        spec_driver.VERIFIER_PRIMARY_POWER_RAIL,
    )
    verifier_decode_total = spec_driver.phase_total(
        energy_by_phase,
        "verifier_decode",
        spec_driver.VERIFIER_PRIMARY_POWER_RAIL,
    )
    verifier_total = verifier_prefill_total + verifier_decode_total
    drafter_idle_power_mw = idle_baseline.get("drafter_idle_power_mw")
    include_idle_drafter = baseline_system_boundary(args) == "two_device_idle_drafter"
    drafter_idle_total_energy = (
        float(drafter_idle_power_mw) * wall_ms / 1000.0
        if include_idle_drafter and drafter_idle_power_mw is not None
        else 0.0
    )
    system_total = verifier_total + drafter_idle_total_energy
    verifier_idle_power_mw = idle_baseline.get("verifier_idle_power_mw")
    verifier_prefill_active_energy = spec_driver.active_energy_from_idle(
        verifier_prefill_total,
        verifier_idle_power_mw,
        init_response.latency_ms,
    )
    verifier_decode_active_energy = spec_driver.active_energy_from_idle(
        verifier_decode_total,
        verifier_idle_power_mw,
        generate_response.latency_ms,
    )
    verifier_active_energy = (
        verifier_prefill_active_energy + verifier_decode_active_energy
        if verifier_prefill_active_energy is not None
        and verifier_decode_active_energy is not None
        else None
    )
    verifier_prefill_power_samples = spec_driver.sample_total(
        samples_by_phase,
        "verifier_prefill",
        spec_driver.VERIFIER_PRIMARY_POWER_RAIL,
    )
    verifier_decode_power_samples = spec_driver.sample_total(
        samples_by_phase,
        "verifier_decode",
        spec_driver.VERIFIER_PRIMARY_POWER_RAIL,
    )
    verifier_power_samples = verifier_prefill_power_samples + verifier_decode_power_samples
    verifier_energy_available = verifier_power_samples > 0
    drafter_idle_power_samples = int(idle_baseline.get("drafter_idle_power_samples") or 0)
    drafter_energy_available = (
        include_idle_drafter and drafter_idle_power_samples > 0
    )
    system_energy_complete = verifier_energy_available and (
        not include_idle_drafter or drafter_energy_available
    )
    idle_power_parts = [
        float(value)
        for value in (drafter_idle_power_mw, verifier_idle_power_mw)
        if value is not None
    ]
    system_idle_power_mw = sum(idle_power_parts) if idle_power_parts else None

    rows: List[Dict[str, object]] = []
    for rail in all_rails(energy_by_phase):
        verifier_prefill_energy = energy_by_phase["verifier_prefill"].get(rail, 0.0)
        verifier_decode_energy = energy_by_phase["verifier_decode"].get(rail, 0.0)
        role_total_energy = verifier_prefill_energy + verifier_decode_energy
        total_samples = (
            samples_by_phase["verifier_prefill"].get(rail, 0)
            + samples_by_phase["verifier_decode"].get(rail, 0)
        )
        rows.append(
            {
                "result_schema_version": spec_driver.RESULT_SCHEMA_VERSION,
                "experiment": args.experiment,
                "algorithm": "verifier_only",
                "algorithm_version": spec_driver.BASELINE_ALGORITHM_VERSION,
                "system_boundary": baseline_system_boundary(args),
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
                "gamma": "",
                "run": run_index,
                "measurement_order": measurement_order,
                "drafter_freq_hz": drafter_freq_hz or "",
                "verifier_clock_mhz": verifier_clock_mhz or "",
                "rail": rail,
                "drafter_primary_power_rail": (
                    spec_driver.DRAFTER_PRIMARY_POWER_RAIL
                    if include_idle_drafter
                    else ""
                ),
                "verifier_primary_power_rail": spec_driver.VERIFIER_PRIMARY_POWER_RAIL,
                "system_primary_power_rails": (
                    (
                        f"drafter_idle:{spec_driver.DRAFTER_PRIMARY_POWER_RAIL},"
                        f"verifier:{spec_driver.VERIFIER_PRIMARY_POWER_RAIL}"
                    )
                    if include_idle_drafter
                    else f"verifier:{spec_driver.VERIFIER_PRIMARY_POWER_RAIL}"
                ),
                "prompt_tokens": len(prompt_tokens),
                "prompt_token_sha256": prompt_token_sha256,
                "generated_tokens": len(generated_tokens),
                "stop_reason": stop_reason,
                "stop_token_policy": stop_policy,
                "stop_token_ids": stop_token_ids_text,
                "output_token_sha256": output_token_sha256,
                "steps": len(generated_tokens),
                "draft_tokens": "",
                "accepted_draft_tokens": "",
                "replacement_tokens": "",
                "accept_rate": "",
                "drafter_init_latency_ms": "",
                "verifier_init_latency_ms": f"{init_response.latency_ms:.4f}",
                "drafter_draft_latency_ms": "",
                "drafter_commit_latency_ms": "",
                "verifier_latency_ms": f"{generate_response.latency_ms:.4f}",
                "drafter_init_rpc_latency_ms": "",
                "verifier_init_rpc_latency_ms": f"{init_rpc_ms:.4f}",
                "drafter_draft_rpc_latency_ms": "",
                "drafter_commit_rpc_latency_ms": "",
                "verifier_rpc_latency_ms": f"{generate_rpc_ms:.4f}",
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
                "drafter_init_rpc_request_bytes": "",
                "drafter_init_rpc_response_bytes": "",
                "verifier_init_rpc_request_bytes": int(
                    rpc_bytes.get("verifier_init_request_bytes", 0)
                ),
                "verifier_init_rpc_response_bytes": int(
                    rpc_bytes.get("verifier_init_response_bytes", 0)
                ),
                "drafter_draft_rpc_request_bytes": "",
                "drafter_draft_rpc_response_bytes": "",
                "verifier_verify_rpc_request_bytes": "",
                "verifier_verify_rpc_response_bytes": "",
                "drafter_commit_rpc_request_bytes": "",
                "drafter_commit_rpc_response_bytes": "",
                "verifier_decode_rpc_request_bytes": int(
                    rpc_bytes.get("verifier_decode_request_bytes", 0)
                ),
                "verifier_decode_rpc_response_bytes": int(
                    rpc_bytes.get("verifier_decode_response_bytes", 0)
                ),
                "wall_latency_ms": f"{wall_ms:.4f}",
                "tokens_per_s": (
                    f"{len(generated_tokens) / (wall_ms / 1000.0):.4f}"
                    if generated_tokens
                    else "0.0000"
                ),
                "drafter_prefill_energy_mj": "",
                "drafter_draft_energy_mj": "",
                "drafter_commit_energy_mj": "",
                "verifier_prefill_energy_mj": f"{verifier_prefill_energy:.6f}" if rail else "",
                "verifier_verify_energy_mj": f"{verifier_decode_energy:.6f}" if rail else "",
                "role_total_energy_mj": f"{role_total_energy:.6f}" if rail else "",
                "drafter_prefill_total_energy_mj": "0.000000",
                "drafter_draft_total_energy_mj": "0.000000",
                "drafter_commit_total_energy_mj": "0.000000",
                "drafter_idle_total_energy_mj": f"{drafter_idle_total_energy:.6f}",
                "drafter_idle_energy_mj_per_generated_token": (
                    f"{drafter_idle_total_energy / len(generated_tokens):.6f}"
                    if include_idle_drafter and generated_tokens
                    else ""
                ),
                "verifier_prefill_total_energy_mj": f"{verifier_prefill_total:.6f}",
                "verifier_verify_total_energy_mj": f"{verifier_decode_total:.6f}",
                "drafter_total_energy_mj": f"{drafter_idle_total_energy:.6f}",
                "verifier_total_energy_mj": f"{verifier_total:.6f}",
                "system_total_energy_mj": f"{system_total:.6f}",
                "system_total_energy_mj_per_generated_token": (
                    f"{system_total / len(generated_tokens):.6f}"
                    if rail and generated_tokens
                    else ""
                ),
                "idle_baseline_s": spec_driver.fmt_optional(
                    idle_baseline.get("idle_baseline_s")
                ),
                "idle_baseline_policy": args.idle_baseline_policy,
                "drafter_idle_power_mw": spec_driver.fmt_optional(
                    drafter_idle_power_mw
                ),
                "verifier_idle_power_mw": spec_driver.fmt_optional(
                    verifier_idle_power_mw
                ),
                "system_idle_power_mw": spec_driver.fmt_optional(system_idle_power_mw),
                "drafter_idle_power_samples": drafter_idle_power_samples,
                "verifier_idle_power_samples": int(
                    idle_baseline.get("verifier_idle_power_samples") or 0
                ),
                "drafter_prefill_active_energy_mj": "0.000000"
                if include_idle_drafter
                else "",
                "drafter_draft_active_energy_mj": "0.000000"
                if include_idle_drafter
                else "",
                "drafter_commit_active_energy_mj": "0.000000"
                if include_idle_drafter
                else "",
                "verifier_prefill_active_energy_mj": spec_driver.fmt_optional(
                    verifier_prefill_active_energy
                ),
                "verifier_verify_active_energy_mj": spec_driver.fmt_optional(
                    verifier_decode_active_energy
                ),
                "drafter_active_energy_mj": "0.000000" if include_idle_drafter else "",
                "verifier_active_energy_mj": spec_driver.fmt_optional(
                    verifier_active_energy
                ),
                "system_active_energy_mj": spec_driver.fmt_optional(
                    verifier_active_energy
                ),
                "system_active_energy_mj_per_generated_token": (
                    spec_driver.fmt_optional(verifier_active_energy / len(generated_tokens))
                    if verifier_active_energy is not None and generated_tokens
                    else ""
                ),
                "n_power_samples": total_samples,
                "drafter_prefill_power_samples": 0,
                "drafter_draft_power_samples": 0,
                "drafter_commit_power_samples": 0,
                "verifier_prefill_power_samples": verifier_prefill_power_samples,
                "verifier_verify_power_samples": "",
                "verifier_decode_power_samples": verifier_decode_power_samples,
                "drafter_power_samples": drafter_idle_power_samples,
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
                "drafter_runtime_fingerprint": spec_driver.runtime_fingerprint_from_metadata(
                    health_metadata,
                    "drafter",
                )
                if include_idle_drafter
                else "",
                "verifier_runtime_fingerprint": spec_driver.runtime_fingerprint_from_metadata(
                    health_metadata,
                    "verifier",
                ),
                "drafter_hostname": health_metadata.get("drafter_hostname", ""),
                "verifier_hostname": health_metadata.get("verifier_hostname", ""),
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
                    "drafter_jetson_gpu_freq_hz",
                    "",
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
                "drafter_pod_name": health_metadata.get("drafter_pod_name", ""),
                "verifier_pod_name": health_metadata.get("verifier_pod_name", ""),
                "drafter_pod_namespace": health_metadata.get(
                    "drafter_pod_namespace",
                    "",
                ),
                "verifier_pod_namespace": health_metadata.get(
                    "verifier_pod_namespace", ""
                ),
                "drafter_node_name": health_metadata.get("drafter_node_name", ""),
                "verifier_node_name": health_metadata.get("verifier_node_name", ""),
                "drafter_cuda_visible_devices": health_metadata.get(
                    "drafter_cuda_visible_devices",
                    "",
                ),
                "verifier_cuda_visible_devices": health_metadata.get(
                    "verifier_cuda_visible_devices", ""
                ),
                "drafter_nvidia_visible_devices": health_metadata.get(
                    "drafter_nvidia_visible_devices",
                    "",
                ),
                "verifier_nvidia_visible_devices": health_metadata.get(
                    "verifier_nvidia_visible_devices", ""
                ),
                "drafter_runtime_temp_c": spec_driver.runtime_status_fields(
                    health_metadata,
                    "drafter",
                )["drafter_runtime_temp_c"],
                "verifier_runtime_temp_c": spec_driver.runtime_status_fields(
                    health_metadata,
                    "verifier",
                )["verifier_runtime_temp_c"],
                "drafter_thermal_zones": health_metadata.get(
                    "drafter_thermal_zones",
                    "",
                ),
                "verifier_thermal_zones": health_metadata.get(
                    "verifier_thermal_zones",
                    "",
                ),
                "drafter_nvidia_pstate": health_metadata.get(
                    "drafter_nvidia_pstate",
                    "",
                ),
                "verifier_nvidia_pstate": health_metadata.get(
                    "verifier_nvidia_pstate",
                    "",
                ),
                "drafter_nvidia_throttle_active": health_metadata.get(
                    "drafter_nvidia_throttle_active",
                    "",
                ),
                "verifier_nvidia_throttle_active": health_metadata.get(
                    "verifier_nvidia_throttle_active",
                    "",
                ),
                **spec_driver.driver_metadata_fields(driver_metadata),
                "output_text": output_text,
            }
        )
    return rows


async def run(args: argparse.Namespace) -> None:
    validate_args(args)
    prompt_cases = spec_driver.load_prompt_cases(args, allow_missing=args.dry_run)
    spec_driver.validate_prompt_cases(prompt_cases)
    conditions = build_baseline_conditions(args, prompt_cases)
    prompt_set_sha256 = spec_driver.prompt_set_hash(prompt_cases)
    plan = build_baseline_plan(args, prompt_cases, conditions)
    driver_metadata = {
        **plan["metadata"],
        "plan_sha256": plan["plan_sha256"],
        "plan_design_sha256": plan["plan_design_sha256"],
    }
    if args.plan_out:
        spec_driver.write_json(args.plan_out, plan)
        print(f"Wrote baseline plan to {args.plan_out}")
    if args.dry_run:
        print_plan(plan)
        return

    resume_rows, completed_measurement_orders = spec_driver.load_resume_rows(
        args,
        plan,
        expected_algorithm="verifier_only",
    )
    spec_driver.prepare_trace_output(args, plan, resume_rows)

    spec_driver.load_grpc_bindings()
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
    async with AsyncExitStack() as stack:
        verifier_channel = await stack.enter_async_context(
            spec_driver.grpc.aio.insecure_channel(
                args.verifier_addr,
                options=channel_options,
            )
        )
        verifier_stub = spec_driver.spec_pb2_grpc.VerifierStub(verifier_channel)
        drafter_stub = None
        if args.drafter_addr:
            drafter_channel = await stack.enter_async_context(
                spec_driver.grpc.aio.insecure_channel(
                    args.drafter_addr,
                    options=channel_options,
                )
            )
            drafter_stub = spec_driver.spec_pb2_grpc.DrafterStub(drafter_channel)

        health_metadata = {}
        if drafter_stub is not None:
            health_metadata.update(
                await spec_driver.wait_for_health(
                    "drafter",
                    drafter_stub,
                    args.timeout,
                    args.startup_timeout_s,
                    args.health_check_interval_s,
                )
            )
        health_metadata.update(
            await wait_for_health(
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
            if drafter_stub is not None:
                refreshed.update(
                    await spec_driver.check_health(
                        "drafter",
                        drafter_stub,
                        args.timeout,
                        verbose=False,
                    )
                )
            refreshed.update(
                await check_health(
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
            prefixes = ["verifier"]
            if drafter_stub is not None:
                prefixes.insert(0, "drafter")
            return await spec_driver.thermal_guard_metadata(
                args=args,
                initial_metadata=initial,
                refresh_metadata=lambda: run_health_metadata(combo_metadata, force=True),
                prefixes=prefixes,
                label=label,
            )

        async def prepare_condition(condition: BaselineCondition):
            prompt_case = condition.prompt_case
            drafter_freq_hz = condition.drafter_freq_hz
            verifier_clock_mhz = condition.verifier_clock_mhz
            combo_metadata = dict(health_metadata)
            if drafter_stub is not None and drafter_freq_hz is not None:
                combo_metadata.update(
                    await spec_driver.set_frequency(
                        "drafter",
                        drafter_stub,
                        args.timeout,
                        jetson_gpu_freq_hz=drafter_freq_hz,
                    )
                )
            combo_metadata.update(
                await set_verifier_clock(
                    verifier_stub,
                    args.timeout,
                    verifier_clock_mhz,
                )
            )
            if args.idle_baseline_policy == "condition":
                idle_baseline = await sample_idle_baseline(
                    args=args,
                    verifier_stub=verifier_stub,
                    drafter_stub=drafter_stub,
                    drafter_freq_hz=drafter_freq_hz,
                    verifier_clock_mhz=verifier_clock_mhz,
                )
            else:
                idle_baseline = empty_verifier_idle_baseline()
            return combo_metadata, idle_baseline

        rows: List[Dict[str, object]] = list(resume_rows)
        measurement_order = 0
        if not args.shuffle_runs:
            for condition in conditions:
                prompt_case = condition.prompt_case
                drafter_freq_hz = condition.drafter_freq_hz
                verifier_clock_mhz = condition.verifier_clock_mhz
                combo_metadata, idle_baseline = await prepare_condition(condition)
                for _ in range(args.warmup_runs):
                    await run_baseline_decode(
                        args=args,
                        tokenizer=tokenizer,
                        verifier_stub=verifier_stub,
                        prompt_case=prompt_case,
                        prompt_set_sha256=prompt_set_sha256,
                        drafter_freq_hz=drafter_freq_hz,
                        verifier_clock_mhz=verifier_clock_mhz,
                        run_index=0,
                        health_metadata=combo_metadata,
                        idle_baseline=idle_baseline,
                        measurement_order=0,
                        driver_metadata=driver_metadata,
                    )
                for run_index in range(1, args.runs + 1):
                    measurement_order += 1
                    if measurement_order in completed_measurement_orders:
                        print(
                            "Skipping completed baseline measurement "
                            f"order={measurement_order} run={run_index}."
                        )
                        continue
                    run_rows = await run_baseline_decode(
                        args=args,
                        tokenizer=tokenizer,
                        verifier_stub=verifier_stub,
                        prompt_case=prompt_case,
                        prompt_set_sha256=prompt_set_sha256,
                        drafter_freq_hz=drafter_freq_hz,
                        verifier_clock_mhz=verifier_clock_mhz,
                        run_index=run_index,
                        health_metadata=await ready_run_metadata(
                            combo_metadata,
                            f"baseline run {measurement_order}",
                        ),
                        idle_baseline=await verifier_idle_baseline_for_measured_run(
                            args=args,
                            condition_idle_baseline=idle_baseline,
                            verifier_stub=verifier_stub,
                            drafter_stub=drafter_stub,
                            drafter_freq_hz=drafter_freq_hz,
                            verifier_clock_mhz=verifier_clock_mhz,
                        ),
                        measurement_order=measurement_order,
                        driver_metadata=driver_metadata,
                    )
                    rows.extend(run_rows)
                    if args.out:
                        spec_driver.write_rows(args.out, rows)
                    total_row = next(
                        (row for row in run_rows if row["rail"] == "verifier_gpu_power"),
                        run_rows[0],
                    )
                    print(
                        "baseline prompt={prompt_id} fd={drafter_freq_hz} "
                        "fv={verifier_clock_mhz} "
                        "run={run} order={measurement_order} "
                        "generated={generated_tokens} wall_ms={wall_latency_ms} "
                        "system_total_mj={system_total_energy_mj}".format(**total_row)
                    )
        else:
            prepared = {}
            for condition_index, condition in enumerate(conditions):
                combo_metadata, idle_baseline = await prepare_condition(condition)
                prepared[condition_index] = (combo_metadata, idle_baseline)

            current_verifier_clock = None
            current_drafter_freq = None
            for entry in build_baseline_warmup_schedule(args, conditions):
                condition = conditions[entry.condition_index]
                combo_metadata, idle_baseline = prepared[entry.condition_index]
                combo_metadata = dict(combo_metadata)
                if (
                    drafter_stub is not None
                    and condition.drafter_freq_hz is not None
                    and condition.drafter_freq_hz != current_drafter_freq
                ):
                    combo_metadata.update(
                        await spec_driver.set_frequency(
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
                        await set_verifier_clock(
                            verifier_stub,
                            args.timeout,
                            condition.verifier_clock_mhz,
                        )
                    )
                    current_verifier_clock = condition.verifier_clock_mhz
                await run_baseline_decode(
                    args=args,
                    tokenizer=tokenizer,
                    verifier_stub=verifier_stub,
                    prompt_case=condition.prompt_case,
                    prompt_set_sha256=prompt_set_sha256,
                    drafter_freq_hz=condition.drafter_freq_hz,
                    verifier_clock_mhz=condition.verifier_clock_mhz,
                    run_index=0,
                    health_metadata=combo_metadata,
                    idle_baseline=idle_baseline,
                    measurement_order=0,
                    driver_metadata=driver_metadata,
                )

            current_verifier_clock = None
            current_drafter_freq = None
            for measurement_order, entry in enumerate(
                build_baseline_run_schedule(args, conditions),
                start=1,
            ):
                condition = conditions[entry.condition_index]
                if measurement_order in completed_measurement_orders:
                    print(
                        "Skipping completed baseline measurement "
                        f"order={measurement_order} run={entry.run_index}."
                    )
                    continue
                combo_metadata, idle_baseline = prepared[entry.condition_index]
                combo_metadata = dict(combo_metadata)
                if (
                    drafter_stub is not None
                    and condition.drafter_freq_hz is not None
                    and condition.drafter_freq_hz != current_drafter_freq
                ):
                    combo_metadata.update(
                        await spec_driver.set_frequency(
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
                        await set_verifier_clock(
                            verifier_stub,
                            args.timeout,
                            condition.verifier_clock_mhz,
                        )
                    )
                    current_verifier_clock = condition.verifier_clock_mhz
                run_rows = await run_baseline_decode(
                    args=args,
                    tokenizer=tokenizer,
                    verifier_stub=verifier_stub,
                    prompt_case=condition.prompt_case,
                    prompt_set_sha256=prompt_set_sha256,
                    drafter_freq_hz=condition.drafter_freq_hz,
                    verifier_clock_mhz=condition.verifier_clock_mhz,
                    run_index=entry.run_index,
                    health_metadata=await ready_run_metadata(
                        combo_metadata,
                        f"baseline run {measurement_order}",
                    ),
                    idle_baseline=await verifier_idle_baseline_for_measured_run(
                        args=args,
                        condition_idle_baseline=idle_baseline,
                        verifier_stub=verifier_stub,
                        drafter_stub=drafter_stub,
                        drafter_freq_hz=condition.drafter_freq_hz,
                        verifier_clock_mhz=condition.verifier_clock_mhz,
                    ),
                    measurement_order=measurement_order,
                    driver_metadata=driver_metadata,
                )
                rows.extend(run_rows)
                if args.out:
                    spec_driver.write_rows(args.out, rows)
                total_row = next(
                    (row for row in run_rows if row["rail"] == "verifier_gpu_power"),
                    run_rows[0],
                )
                print(
                    "baseline prompt={prompt_id} fd={drafter_freq_hz} "
                    "fv={verifier_clock_mhz} "
                    "run={run} order={measurement_order} "
                    "generated={generated_tokens} wall_ms={wall_latency_ms} "
                    "system_total_mj={system_total_energy_mj}".format(**total_row)
                )

    if args.out:
        spec_driver.write_rows(args.out, rows)
        print(f"Wrote {len(rows)} rows to {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verifier-only decode baseline")
    parser.add_argument(
        "--drafter-addr",
        default="",
        help=(
            "Optional drafter host:port. When set, the baseline includes Jetson "
            "idle energy for a two-device system boundary."
        ),
    )
    parser.add_argument("--verifier-addr", help="host:port for verifier")
    parser.add_argument("--tokenizer", help="Shared tokenizer id/path")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument(
        "--prompts-jsonl",
        help="JSONL prompt set. Each line can be a string or {'id': ..., 'prompt': ...}.",
    )
    parser.add_argument(
        "--drafter-freqs-hz",
        default="",
        help=(
            "Optional comma-separated Jetson GPU frequencies for idle-drafter "
            "baseline accounting. Used only with --drafter-addr."
        ),
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
        help="Maximum seconds to wait for verifier health before the baseline starts.",
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
        help="Shuffle prompt/frequency conditions to reduce order bias.",
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
    parser.add_argument("--out", default="verifier_baseline.csv")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load existing --out rows, verify they match the current plan, and "
            "skip completed measured orders."
        ),
    )
    parser.add_argument("--experiment", default="verifier_baseline")
    parser.add_argument("--max-message-mb", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-out", default="")
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--trace-warmups", action="store_true")
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
