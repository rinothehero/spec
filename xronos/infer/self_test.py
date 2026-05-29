import asyncio
import csv
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from xronos.infer import (
    analyze_gamma_effect,
    analyze_spec_results,
    artifact_audit,
    compare_to_baseline,
    experiment_doctor,
    experiment_report,
    frequency,
    k8s_manifest_audit,
    k8s_runbook,
    network_probe,
    paired_prompt_compare,
    plan_audit,
    power,
    runtime,
    select_best_config,
    spec_algorithm,
    spec_driver,
    validate_results,
    verifier_baseline_driver,
)


def _driver_args(**overrides):
    defaults = {
        "experiment": "self_test",
        "drafter_addr": None,
        "verifier_addr": None,
        "tokenizer": None,
        "prompt": None,
        "prompt_file": None,
        "prompts_jsonl": None,
        "gammas": "1,2,4",
        "drafter_freqs_hz": "408000000,612000000",
        "verifier_clocks_mhz": "810,1410",
        "runs": 2,
        "warmup_runs": 1,
        "max_new_tokens": 32,
        "stop_token_ids": "",
        "decoding_mode": "greedy",
        "idle_baseline_s": 0.0,
        "idle_baseline_policy": "condition",
        "shuffle_conditions": False,
        "shuffle_runs": False,
        "sample_runtime_metadata": False,
        "startup_timeout_s": 600.0,
        "health_check_interval_s": 5.0,
        "max_start_temp_c": None,
        "thermal_check_interval_s": 5.0,
        "thermal_wait_timeout_s": 300.0,
        "seed": 0,
        "timeout": 30.0,
        "resume": False,
        "dry_run": True,
        "plan_out": "",
        "trace_out": "",
        "trace_warmups": False,
        "no_special_tokens": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _single_run_schedule(condition_count: int, run_key: str):
    condition_order = list(range(condition_count))
    if condition_count > 1:
        condition_order = condition_order[1::2] + condition_order[0::2]
    return [
        {"order": order + 1, "condition_order": condition, run_key: 1}
        for order, condition in enumerate(condition_order)
    ]


def _strict_factorial_report_fixture():
    prompt_set_sha256 = "strict-prompt-set"
    prompts = [
        {"prompt_id": "p0", "prompt_sha256": "hash-p0", "prompt_chars": 10},
        {"prompt_id": "p1", "prompt_sha256": "hash-p1", "prompt_chars": 10},
    ]
    prompt_token_hashes = {
        "p0": spec_driver.token_ids_sha256([11, 12, 13]),
        "p1": spec_driver.token_ids_sha256([21, 22, 23]),
    }
    output_hashes = {
        "p0": spec_driver.token_ids_sha256([101, 102, 103]),
        "p1": spec_driver.token_ids_sha256([201, 202, 203]),
    }
    gammas = [1, 2]
    drafter_freqs = [408000000, 612000000]
    verifier_clocks = [810, 1410]

    baseline_combos = [
        {
            "prompt_id": prompt["prompt_id"],
            "prompt_sha256": prompt["prompt_sha256"],
            "drafter_freq_hz": drafter_freq,
            "verifier_clock_mhz": verifier_clock,
            "measured_runs": 1,
        }
        for prompt in prompts
        for drafter_freq in drafter_freqs
        for verifier_clock in verifier_clocks
    ]
    spec_combos = [
        {
            "prompt_id": prompt["prompt_id"],
            "prompt_sha256": prompt["prompt_sha256"],
            "gamma": gamma,
            "drafter_freq_hz": drafter_freq,
            "verifier_clock_mhz": verifier_clock,
            "measured_runs": 1,
        }
        for prompt in prompts
        for gamma in gammas
        for drafter_freq in drafter_freqs
        for verifier_clock in verifier_clocks
    ]

    def make_plan(algorithm: str, combos):
        plan = {
            "schema_version": spec_driver.RESULT_SCHEMA_VERSION,
            "algorithm": algorithm,
            "algorithm_version": (
                spec_driver.BASELINE_ALGORITHM_VERSION
                if algorithm == "verifier_only"
                else spec_driver.SPEC_ALGORITHM_VERSION
            ),
            "decoding_mode": "greedy",
            "max_new_tokens": 32,
            "idle_baseline_s": 5,
            "idle_baseline_policy": "run",
            "tokenizer": "test-tokenizer",
            "prompt_set_sha256": prompt_set_sha256,
            "prompts": prompts,
            "warmup_runs": 1,
            "shuffle_runs": True,
            "seed": 42,
            "combinations": combos,
            "measurement_schedule": _single_run_schedule(len(combos), "run"),
            "warmup_schedule": _single_run_schedule(len(combos), "warmup"),
        }
        if algorithm == "verifier_only":
            plan["system_boundary"] = "two_device_idle_drafter"
            plan["drafter_freqs_hz"] = drafter_freqs
        return plan

    plans = [
        make_plan("verifier_only", baseline_combos),
        make_plan("speculative", spec_combos),
    ]
    for plan in plans:
        spec_driver.attach_plan_sha256(plan)

    plan_sha_by_algorithm = {
        str(plan["algorithm"]): str(plan["plan_sha256"])
        for plan in plans
    }
    plan_design_sha_by_algorithm = {
        str(plan["algorithm"]): str(plan["plan_design_sha256"])
        for plan in plans
    }

    order_by_algorithm_and_condition = {
        str(plan["algorithm"]): {
            int(item["condition_order"]): int(item["order"])
            for item in plan["measurement_schedule"]
        }
        for plan in plans
    }

    def fmt(value: float) -> str:
        return f"{value:.6f}"

    def timing_fields(tokens_per_s: float, generated_tokens: int):
        wall_ms = generated_tokens / tokens_per_s * 1000.0
        server_ms = wall_ms * 0.55
        client_ms = wall_ms * 0.65
        return {
            "tokens_per_s": fmt(tokens_per_s),
            "wall_latency_ms": fmt(wall_ms),
            "client_rpc_latency_ms": fmt(client_ms),
            "server_compute_latency_ms": fmt(server_ms),
            "estimated_rpc_overhead_ms": fmt(client_ms - server_ms),
        }

    def split_two(total: float, first_ratio: float):
        first = total * first_ratio
        return first, total - first

    def split_three(total: float):
        prefill = total * 0.2
        draft = total * 0.6
        return prefill, draft, total - prefill - draft

    def baseline_energy_fields(system_active_per_token: float, generated_tokens: int):
        system_active = system_active_per_token * generated_tokens
        verifier_total = system_active + 1.0
        drafter_idle_total = 0.5
        system_total = verifier_total + drafter_idle_total
        verifier_prefill_active, verifier_verify_active = split_two(system_active, 0.2)
        verifier_prefill_total, verifier_verify_total = split_two(verifier_total, 0.2)
        return {
            "system_active_energy_mj": fmt(system_active),
            "system_active_energy_mj_per_generated_token": fmt(
                system_active / generated_tokens
            ),
            "system_total_energy_mj": fmt(system_total),
            "system_total_energy_mj_per_generated_token": fmt(
                system_total / generated_tokens
            ),
            "drafter_active_energy_mj": "",
            "drafter_prefill_active_energy_mj": "",
            "drafter_draft_active_energy_mj": "",
            "drafter_commit_active_energy_mj": "",
            "drafter_total_energy_mj": fmt(drafter_idle_total),
            "drafter_idle_total_energy_mj": fmt(drafter_idle_total),
            "drafter_idle_energy_mj_per_generated_token": fmt(
                drafter_idle_total / generated_tokens
            ),
            "drafter_prefill_total_energy_mj": "0",
            "drafter_draft_total_energy_mj": "0",
            "drafter_commit_total_energy_mj": "0",
            "verifier_active_energy_mj": fmt(system_active),
            "verifier_prefill_active_energy_mj": fmt(verifier_prefill_active),
            "verifier_verify_active_energy_mj": fmt(verifier_verify_active),
            "verifier_total_energy_mj": fmt(verifier_total),
            "verifier_prefill_total_energy_mj": fmt(verifier_prefill_total),
            "verifier_verify_total_energy_mj": fmt(verifier_verify_total),
        }

    def spec_energy_fields(
        system_active_per_token: float,
        drafter_active_per_token: float,
        generated_tokens: int,
    ):
        system_active = system_active_per_token * generated_tokens
        drafter_active = drafter_active_per_token * generated_tokens
        verifier_active = system_active - drafter_active
        assert verifier_active > 0
        drafter_total = drafter_active + 1.2
        verifier_total = verifier_active + 1.8
        drafter_prefill_active, drafter_draft_active, drafter_commit_active = split_three(
            drafter_active
        )
        verifier_prefill_active, verifier_verify_active = split_two(verifier_active, 0.2)
        drafter_prefill_total, drafter_draft_total, drafter_commit_total = split_three(
            drafter_total
        )
        verifier_prefill_total, verifier_verify_total = split_two(verifier_total, 0.2)
        system_total = drafter_total + verifier_total
        return {
            "system_active_energy_mj": fmt(system_active),
            "system_active_energy_mj_per_generated_token": fmt(
                system_active / generated_tokens
            ),
            "system_total_energy_mj": fmt(system_total),
            "system_total_energy_mj_per_generated_token": fmt(
                system_total / generated_tokens
            ),
            "drafter_active_energy_mj": fmt(drafter_active),
            "drafter_prefill_active_energy_mj": fmt(drafter_prefill_active),
            "drafter_draft_active_energy_mj": fmt(drafter_draft_active),
            "drafter_commit_active_energy_mj": fmt(drafter_commit_active),
            "drafter_total_energy_mj": fmt(drafter_total),
            "drafter_prefill_total_energy_mj": fmt(drafter_prefill_total),
            "drafter_draft_total_energy_mj": fmt(drafter_draft_total),
            "drafter_commit_total_energy_mj": fmt(drafter_commit_total),
            "verifier_active_energy_mj": fmt(verifier_active),
            "verifier_prefill_active_energy_mj": fmt(verifier_prefill_active),
            "verifier_verify_active_energy_mj": fmt(verifier_verify_active),
            "verifier_total_energy_mj": fmt(verifier_total),
            "verifier_prefill_total_energy_mj": fmt(verifier_prefill_total),
            "verifier_verify_total_energy_mj": fmt(verifier_verify_total),
        }

    common = {
        "decoding_mode": "greedy",
        "system_boundary": "two_device_active",
        "max_new_tokens": "32",
        "prompt_set_sha256": prompt_set_sha256,
        "verifier_model": "verifier-test",
        "system_energy_complete": "1",
        "run": "1",
        "stop_reason": "eos",
        "prompt_tokens": "3",
        "drafter_prefill_power_samples": "3",
        "drafter_draft_power_samples": "3",
        "drafter_commit_power_samples": "3",
        "verifier_prefill_power_samples": "3",
        "verifier_verify_power_samples": "3",
        "verifier_decode_power_samples": "3",
        "drafter_power_samples": "3",
        "verifier_power_samples": "3",
        "idle_baseline_s": "5",
        "idle_baseline_policy": "run",
        "drafter_idle_power_mw": "100",
        "verifier_idle_power_mw": "200",
        "system_idle_power_mw": "300",
        "drafter_idle_power_samples": "8",
        "verifier_idle_power_samples": "8",
        "drafter_model_vocab_size": "32000",
        "verifier_model_vocab_size": "32000",
        "drafter_model_bos_token_id": "1",
        "drafter_model_eos_token_id": "2",
        "drafter_model_pad_token_id": "2",
        "verifier_model_bos_token_id": "1",
        "verifier_model_eos_token_id": "2",
        "verifier_model_pad_token_id": "2",
        "drafter_power_interval_s": "0.010000",
        "verifier_power_interval_s": "0.010000",
        "result_schema_version": spec_driver.RESULT_SCHEMA_VERSION,
        "driver_result_schema_version": spec_driver.RESULT_SCHEMA_VERSION,
        "driver_spec_rpc_schema_version": spec_driver.SPEC_RPC_SCHEMA_VERSION,
        "verifier_spec_rpc_schema_version": spec_driver.SPEC_RPC_SCHEMA_VERSION,
        "tokenizer_name_or_path": "test-tokenizer",
        "tokenizer_class": "FakeTokenizer",
        "tokenizer_vocab_size": "32000",
        "tokenizer_base_vocab_size": "32000",
        "tokenizer_bos_token_id": "1",
        "tokenizer_eos_token_id": "2",
        "tokenizer_pad_token_id": "2",
        "tokenizer_unk_token_id": "0",
        "drafter_frequency_lock_ok": "1",
        "verifier_frequency_lock_ok": "1",
        "drafter_hostname": "jetson-node",
        "verifier_hostname": "gpu-node",
        "driver_git_commit": "commit-a",
        "driver_git_dirty": "0",
        "driver_xronos_git_commit": "commit-a",
        "driver_xronos_image": "xronos:gpu",
        "drafter_xronos_git_commit": "commit-a",
        "verifier_xronos_git_commit": "commit-a",
        "drafter_xronos_image": "xronos:jetson",
        "verifier_xronos_image": "xronos:gpu",
        "rpc_request_bytes": "120",
        "rpc_response_bytes": "130",
        "rpc_total_bytes": "250",
        "rpc_bytes_per_generated_token": "25.000000",
        "generated_tokens": "10",
    }

    baseline_energy = {
        (810, "p0"): 2.5,
        (810, "p1"): 2.7,
        (1410, "p0"): 3.0,
        (1410, "p1"): 3.2,
    }
    spec_system_energy = {
        (1, 408000000, 810): 1.4,
        (1, 408000000, 1410): 2.4,
        (1, 612000000, 810): 1.5,
        (1, 612000000, 1410): 2.5,
        (2, 408000000, 810): 1.2,
        (2, 408000000, 1410): 1.0,
        (2, 612000000, 810): 1.8,
        (2, 612000000, 1410): 1.3,
    }

    rows = []
    baseline_plan = plans[0]
    baseline_order = order_by_algorithm_and_condition["verifier_only"]
    for condition_index, combo in enumerate(baseline_plan["combinations"]):
        prompt_id = str(combo["prompt_id"])
        drafter_freq = int(combo["drafter_freq_hz"])
        verifier_clock = int(combo["verifier_clock_mhz"])
        tokens_per_s = 10.0 if verifier_clock == 810 else 12.0
        row = {
            **common,
            **baseline_energy_fields(
                baseline_energy[(verifier_clock, prompt_id)],
                generated_tokens=10,
            ),
            **timing_fields(tokens_per_s=tokens_per_s, generated_tokens=10),
            "algorithm": "verifier_only",
            "algorithm_version": spec_driver.BASELINE_ALGORITHM_VERSION,
            "system_boundary": "two_device_idle_drafter",
            "plan_sha256": plan_sha_by_algorithm["verifier_only"],
            "plan_design_sha256": plan_design_sha_by_algorithm["verifier_only"],
            "driver_plan_sha256": plan_sha_by_algorithm["verifier_only"],
            "driver_plan_design_sha256": plan_design_sha_by_algorithm[
                "verifier_only"
            ],
            "session_id": f"strict_b_{prompt_id}_{drafter_freq}_{verifier_clock}",
            "rail": "verifier_gpu_power",
            "drafter_primary_power_rail": "tot_power",
            "verifier_primary_power_rail": "verifier_gpu_power",
            "system_primary_power_rails": "drafter:tot_power,verifier:verifier_gpu_power",
            "prompt_id": prompt_id,
            "prompt_sha256": str(combo["prompt_sha256"]),
            "prompt_token_sha256": prompt_token_hashes[prompt_id],
            "gamma": "",
            "drafter_freq_hz": str(drafter_freq),
            "drafter_jetson_gpu_freq_hz": str(drafter_freq),
            "verifier_clock_mhz": str(verifier_clock),
            "verifier_gpu_clock_mhz": str(verifier_clock),
            "drafter_model": "",
            "drafter_model_parameter_count": "",
            "verifier_model_parameter_count": "1000000000",
            "drafter_spec_rpc_schema_version": "",
            "steps": "10",
            "output_token_sha256": output_hashes[prompt_id],
            "measurement_order": str(baseline_order[condition_index]),
        }
        rows.append(row)

    spec_plan = plans[1]
    spec_order = order_by_algorithm_and_condition["speculative"]
    for condition_index, combo in enumerate(spec_plan["combinations"]):
        prompt_id = str(combo["prompt_id"])
        gamma = int(combo["gamma"])
        drafter_freq = int(combo["drafter_freq_hz"])
        verifier_clock = int(combo["verifier_clock_mhz"])
        steps = 10 if gamma == 1 else 6
        draft_tokens = 10 if gamma == 1 else 12
        accepted = 6 if gamma == 1 else 9
        replacement = 10 - accepted
        drafter_active_per_token = (
            (0.46 if gamma == 1 else 0.34)
            + (0.02 if drafter_freq == 612000000 else 0.0)
            + (0.01 if verifier_clock == 1410 else 0.0)
        )
        tokens_per_s = (
            11.0
            + (4.0 if gamma == 2 else 0.0)
            + (1.0 if verifier_clock == 1410 else 0.0)
            - (0.5 if drafter_freq == 612000000 else 0.0)
        )
        row = {
            **common,
            **spec_energy_fields(
                spec_system_energy[(gamma, drafter_freq, verifier_clock)],
                drafter_active_per_token=drafter_active_per_token,
                generated_tokens=10,
            ),
            **timing_fields(tokens_per_s=tokens_per_s, generated_tokens=10),
            "algorithm": "speculative",
            "algorithm_version": spec_driver.SPEC_ALGORITHM_VERSION,
            "plan_sha256": plan_sha_by_algorithm["speculative"],
            "plan_design_sha256": plan_design_sha_by_algorithm["speculative"],
            "driver_plan_sha256": plan_sha_by_algorithm["speculative"],
            "driver_plan_design_sha256": plan_design_sha_by_algorithm[
                "speculative"
            ],
            "session_id": (
                f"strict_s_g{gamma}_d{drafter_freq}_"
                f"v{verifier_clock}_{prompt_id}"
            ),
            "rail": "tot_power",
            "drafter_primary_power_rail": "tot_power",
            "verifier_primary_power_rail": "verifier_gpu_power",
            "system_primary_power_rails": (
                "drafter:tot_power,verifier:verifier_gpu_power"
            ),
            "prompt_id": prompt_id,
            "prompt_sha256": str(combo["prompt_sha256"]),
            "prompt_token_sha256": prompt_token_hashes[prompt_id],
            "gamma": str(gamma),
            "drafter_freq_hz": str(drafter_freq),
            "verifier_clock_mhz": str(verifier_clock),
            "drafter_jetson_gpu_freq_hz": str(drafter_freq),
            "verifier_gpu_clock_mhz": str(verifier_clock),
            "drafter_model": "drafter-test",
            "drafter_model_parameter_count": "100000000",
            "verifier_model_parameter_count": "1000000000",
            "drafter_spec_rpc_schema_version": spec_driver.SPEC_RPC_SCHEMA_VERSION,
            "steps": str(steps),
            "draft_tokens": str(draft_tokens),
            "accepted_draft_tokens": str(accepted),
            "replacement_tokens": str(replacement),
            "accept_rate": fmt(accepted / draft_tokens),
            "output_token_sha256": output_hashes[prompt_id],
            "measurement_order": str(spec_order[condition_index]),
        }
        rows.append(row)

    return plans, rows


class _FakeRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSpecPb2:
    HealthRequest = _FakeRequest
    InitSessionRequest = _FakeRequest
    DraftRequest = _FakeRequest
    VerifyRequest = _FakeRequest
    CommitRequest = _FakeRequest
    ResetSessionRequest = _FakeRequest


def _fake_rail(name: str, energy_mj: float):
    return SimpleNamespace(
        rail=name,
        mean_power_mw=100.0,
        energy_mj=energy_mj,
    )


class _FakeTokenizer:
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True):
        assert text
        assert add_special_tokens is True
        return [101, 102]

    def decode(self, token_ids, skip_special_tokens: bool = False):
        assert skip_special_tokens is False
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeDrafterStub:
    def __init__(
        self,
        drafts,
        commit_offset: int = 0,
        init_context_offset: int = 0,
    ):
        self.drafts = list(drafts)
        self.commit_offset = commit_offset
        self.init_context_offset = init_context_offset
        self.committed_tokens = 0
        self.init_requests = []
        self.draft_requests = []
        self.commit_requests = []
        self.reset_requests = []

    async def InitSession(self, request, timeout):
        self.init_requests.append(request)
        self.committed_tokens = len(request.context_tokens)
        return SimpleNamespace(
            context_tokens=len(request.context_tokens) + self.init_context_offset,
            error="",
            latency_ms=1.0,
            rails=[_fake_rail("tot_power", 1.0)],
            n_power_samples=1,
        )

    async def Draft(self, request, timeout):
        self.draft_requests.append(request)
        if int(request.base_committed_tokens) != self.committed_tokens:
            return SimpleNamespace(
                error=(
                    "Drafter session length mismatch during Draft: "
                    f"expected={int(request.base_committed_tokens)}, "
                    f"actual={self.committed_tokens}."
                ),
                draft_tokens=[],
                latency_ms=0.0,
                rails=[],
                n_power_samples=0,
            )
        return SimpleNamespace(
            error="",
            draft_tokens=self.drafts.pop(0),
            latency_ms=2.0,
            rails=[_fake_rail("tot_power", 2.0)],
            n_power_samples=1,
        )

    async def Commit(self, request, timeout):
        self.commit_requests.append(request)
        if int(request.base_committed_tokens) != self.committed_tokens:
            return SimpleNamespace(
                error=(
                    "Drafter session length mismatch during Commit: "
                    f"expected={int(request.base_committed_tokens)}, "
                    f"actual={self.committed_tokens}."
                ),
                committed_tokens=self.committed_tokens,
                latency_ms=0.0,
                rails=[],
                n_power_samples=0,
            )
        self.committed_tokens += int(request.accepted_tokens) + int(
            bool(request.append_replacement)
        )
        return SimpleNamespace(
            error="",
            committed_tokens=self.committed_tokens + self.commit_offset,
            latency_ms=0.5,
            rails=[_fake_rail("tot_power", 0.5)],
            n_power_samples=1,
        )

    async def ResetSession(self, request, timeout):
        self.reset_requests.append(request)
        self.committed_tokens = 0
        return SimpleNamespace(ok=True, error="")


class _FakeVerifierStub:
    def __init__(
        self,
        verify_responses,
        commit_offset: int = 0,
        init_context_offset: int = 0,
    ):
        self.verify_responses = list(verify_responses)
        self.commit_offset = commit_offset
        self.init_context_offset = init_context_offset
        self.committed_tokens = 0
        self.init_requests = []
        self.verify_requests = []
        self.reset_requests = []

    async def InitSession(self, request, timeout):
        self.init_requests.append(request)
        self.committed_tokens = len(request.context_tokens)
        return SimpleNamespace(
            context_tokens=len(request.context_tokens) + self.init_context_offset,
            error="",
            latency_ms=1.5,
            rails=[_fake_rail("verifier_gpu_power", 3.0)],
            n_power_samples=1,
        )

    async def Verify(self, request, timeout):
        self.verify_requests.append(request)
        if int(request.base_committed_tokens) != self.committed_tokens:
            return SimpleNamespace(
                error=(
                    "Verifier session length mismatch during Verify: "
                    f"expected={int(request.base_committed_tokens)}, "
                    f"actual={self.committed_tokens}."
                ),
                accepted_tokens=0,
                replacement_token=0,
                appended_replacement=False,
                committed_tokens=self.committed_tokens,
                latency_ms=0.0,
                rails=[],
                n_power_samples=0,
            )
        accepted, replacement, appended = self.verify_responses.pop(0)
        self.committed_tokens += int(accepted) + int(bool(appended))
        return SimpleNamespace(
            error="",
            accepted_tokens=accepted,
            replacement_token=replacement,
            appended_replacement=appended,
            committed_tokens=self.committed_tokens + self.commit_offset,
            latency_ms=3.0,
            rails=[_fake_rail("verifier_gpu_power", 4.0)],
            n_power_samples=1,
        )

    async def ResetSession(self, request, timeout):
        self.reset_requests.append(request)
        self.committed_tokens = 0
        return SimpleNamespace(ok=True, error="")


class _FlakyHealthStub:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    async def Health(self, request, timeout):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("not ready")
        return SimpleNamespace(
            ok=True,
            message="ready",
            model="fake-model",
            device="cuda",
            metadata={"model_parameter_count": "1000"},
        )


def test_sweep_plan() -> None:
    args = _driver_args()
    spec_driver.validate_args(args)
    prompt_cases = spec_driver.load_prompt_cases(args, allow_missing=True)
    conditions = spec_driver.build_spec_conditions(args, prompt_cases)
    plan = spec_driver.build_sweep_plan(args, prompt_cases, conditions)
    assert plan["gammas"] == [1, 2, 4]
    assert plan["prompt_count"] == 1
    assert len(plan["combinations"]) == 12
    assert plan["total_warmup_sessions"] == 12
    assert plan["total_measured_sessions"] == 24
    assert plan["total_idle_baseline_pairs"] == 0
    assert plan["startup_timeout_s"] == 600.0
    assert plan["health_check_interval_s"] == 5.0
    assert len(plan["warmup_schedule"]) == 12

    condition_idle_args = _driver_args(idle_baseline_s=5, idle_baseline_policy="condition")
    condition_idle_plan = spec_driver.build_sweep_plan(
        condition_idle_args,
        prompt_cases,
        conditions,
    )
    assert condition_idle_plan["total_idle_baselines"] == len(conditions)

    run_idle_args = _driver_args(idle_baseline_s=5, idle_baseline_policy="run")
    run_idle_plan = spec_driver.build_sweep_plan(
        run_idle_args,
        prompt_cases,
        conditions,
    )
    assert run_idle_plan["idle_baseline_policy"] == "run"
    assert run_idle_plan["total_idle_baselines"] == len(conditions) * run_idle_args.runs


def test_two_device_baseline_plan() -> None:
    prompt_cases = [
        spec_driver.make_prompt_case("p0", "Edge AI is", "test"),
        spec_driver.make_prompt_case("p1", "Distributed decoding needs", "test"),
    ]
    args = _driver_args(
        drafter_addr="spec-drafter:50061",
        verifier_addr="spec-verifier:50062",
        tokenizer="test-tokenizer",
        drafter_freqs_hz="408000000,612000000",
        verifier_clocks_mhz="810",
        runs=1,
        warmup_runs=1,
        idle_baseline_s=5,
        idle_baseline_policy="run",
        shuffle_runs=True,
        seed=3,
    )
    conditions = verifier_baseline_driver.build_baseline_conditions(args, prompt_cases)
    plan = verifier_baseline_driver.build_baseline_plan(
        args,
        prompt_cases,
        conditions,
    )
    assert plan["system_boundary"] == "two_device_idle_drafter"
    assert plan["drafter_freqs_hz"] == [408000000, 612000000]
    assert len(plan["combinations"]) == 4
    assert {
        combo["drafter_freq_hz"] for combo in plan["combinations"]
    } == {408000000, 612000000}
    assert plan["total_idle_baselines"] == len(plan["measurement_schedule"])

    spec_args = _driver_args(
        tokenizer="test-tokenizer",
        gammas="1,2",
        drafter_freqs_hz="408000000,612000000",
        verifier_clocks_mhz="810",
        runs=1,
        warmup_runs=1,
        idle_baseline_s=5,
        idle_baseline_policy="run",
        shuffle_runs=True,
        seed=3,
    )
    spec_conditions = spec_driver.build_spec_conditions(spec_args, prompt_cases)
    spec_plan = spec_driver.build_sweep_plan(
        spec_args,
        prompt_cases,
        spec_conditions,
    )
    full_report = experiment_report.plan_design_report(
        [spec_plan, plan],
        min_prompts=2,
        min_gammas=2,
        require_two_device_boundary=True,
    )
    assert full_report["ok"] is True
    assert full_report["missing_two_device_baseline_conditions"] == 0

    generic_baseline_args = _driver_args(
        tokenizer="test-tokenizer",
        verifier_clocks_mhz="810",
        runs=1,
        warmup_runs=1,
        idle_baseline_s=5,
        idle_baseline_policy="run",
        shuffle_runs=True,
        seed=3,
    )
    generic_baseline_conditions = verifier_baseline_driver.build_baseline_conditions(
        generic_baseline_args,
        prompt_cases,
    )
    generic_baseline_plan = verifier_baseline_driver.build_baseline_plan(
        generic_baseline_args,
        prompt_cases,
        generic_baseline_conditions,
    )
    strict_generic_report = experiment_report.plan_design_report(
        [spec_plan, generic_baseline_plan],
        min_prompts=2,
        min_gammas=2,
        require_two_device_boundary=True,
    )
    assert strict_generic_report["ok"] is False
    assert (
        "two_device_boundary_requires_idle_drafter_baseline_plan"
        in strict_generic_report["errors"]
    )

    broken_plan = json.loads(json.dumps(plan))
    broken_plan["combinations"] = [
        combo
        for combo in broken_plan["combinations"]
        if not (
            combo["prompt_id"] == "p1"
            and str(combo["drafter_freq_hz"]) == "612000000"
            and str(combo["verifier_clock_mhz"]) == "810"
        )
    ]
    broken_report = experiment_report.plan_design_report(
        [spec_plan, broken_plan],
        min_prompts=2,
        min_gammas=2,
    )
    assert broken_report["ok"] is False
    assert (
        "two_device_baseline_missing_speculative_frequency_conditions"
        in broken_report["errors"]
    )
    assert broken_report["missing_two_device_baseline_conditions"] == 1


def test_arg_validation() -> None:
    try:
        spec_driver.validate_args(_driver_args(runs=0))
    except ValueError as exc:
        assert "--runs" in str(exc)
    else:
        raise AssertionError("Expected --runs validation error")

    try:
        spec_driver.validate_args(
            _driver_args(dry_run=False, prompt="hello", tokenizer="tok")
        )
    except ValueError as exc:
        assert "--drafter-addr" in str(exc)
    else:
        raise AssertionError("Expected missing address validation error")

    for kwargs, expected in (
        ({"gammas": "1,0"}, "--gammas"),
        ({"drafter_freqs_hz": "408000000,0"}, "--drafter-freqs-hz"),
        ({"verifier_clocks_mhz": "810,-1"}, "--verifier-clocks-mhz"),
        ({"verifier_clocks_mhz": ","}, "At least one integer"),
        ({"gammas": "1,,2"}, "empty items"),
        ({"drafter_freqs_hz": "408000000,"}, "empty CSV items"),
        ({"startup_timeout_s": -1}, "--startup-timeout-s"),
        ({"health_check_interval_s": 0}, "--health-check-interval-s"),
    ):
        try:
            spec_driver.validate_args(_driver_args(**kwargs))
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected validation error for {kwargs}")


def test_prompt_jsonl_plan() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prompts.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "p0", "prompt": "Edge AI is"}) + "\n")
            f.write(json.dumps({"id": "p1", "prompt": "Mobile inference needs"}) + "\n")

        args = _driver_args(prompts_jsonl=str(path))
        prompt_cases = spec_driver.load_prompt_cases(args, allow_missing=False)
        conditions = spec_driver.build_spec_conditions(args, prompt_cases)
        plan = spec_driver.build_sweep_plan(args, prompt_cases, conditions)
        assert [case.prompt_id for case in prompt_cases] == ["p0", "p1"]
        assert plan["prompt_count"] == 2
        assert len(plan["combinations"]) == 24
        assert plan["total_measured_sessions"] == 48


def test_plan_audit() -> None:
    prompt_cases = [
        spec_driver.make_prompt_case("p0", "Edge AI is", "test"),
        spec_driver.make_prompt_case("p1", "Distributed decoding needs", "test"),
    ]
    spec_args = _driver_args(
        tokenizer="test-tokenizer",
        gammas="1,2,4",
        drafter_freqs_hz="408000000",
        verifier_clocks_mhz="810",
        runs=3,
        warmup_runs=1,
        idle_baseline_s=5,
        idle_baseline_policy="run",
        shuffle_runs=True,
        seed=7,
    )
    spec_conditions = spec_driver.build_spec_conditions(spec_args, prompt_cases)
    spec_plan = spec_driver.build_sweep_plan(
        spec_args,
        prompt_cases,
        spec_conditions,
    )
    design_variant = {
        **spec_plan,
        "metadata": {"created_at_utc": "different"},
        "drafter_addr": "different-drafter:50061",
        "verifier_addr": "different-verifier:50062",
        "trace_out": "different_trace.jsonl",
        "startup_timeout_s": 999.0,
        "thermal_wait_timeout_s": 999.0,
    }
    assert spec_driver.plan_design_sha256(design_variant) == spec_plan[
        "plan_design_sha256"
    ]
    assert spec_driver.plan_sha256(design_variant) == spec_plan["plan_sha256"]
    assert spec_plan["plan_sha256"] == spec_plan["plan_design_sha256"]
    thermal_design_variant = {
        **spec_plan,
        "max_start_temp_c": 85.0,
    }
    assert spec_driver.plan_design_sha256(thermal_design_variant) != spec_plan[
        "plan_design_sha256"
    ]
    baseline_args = _driver_args(
        tokenizer="test-tokenizer",
        verifier_clocks_mhz="810",
        runs=3,
        warmup_runs=1,
        idle_baseline_s=5,
        idle_baseline_policy="run",
        shuffle_runs=True,
        seed=7,
    )
    baseline_conditions = verifier_baseline_driver.build_baseline_conditions(
        baseline_args,
        prompt_cases,
    )
    baseline_plan = verifier_baseline_driver.build_baseline_plan(
        baseline_args,
        prompt_cases,
        baseline_conditions,
    )
    plans = [spec_plan, baseline_plan]
    audit = plan_audit.build_audit(
        plans=plans,
        plan_paths=["spec_plan.json", "verifier_baseline_plan.json"],
        min_runs=3,
        min_prompts=2,
        min_gammas=3,
        summary_energy_key=experiment_report.DEFAULT_SUMMARY_ENERGY_KEY,
        paired_energy_key=experiment_report.DEFAULT_PAIRED_ENERGY_KEY,
    )
    assert audit["ok"] is True
    assert audit["expected"]["measured_sessions"] == 24
    assert audit["expected"]["warmup_sessions"] == 8
    assert audit["plan_design"]["min_runs"] == 3
    assert audit["plan_design"]["min_gammas"] == 3
    assert "speculative" in audit["plan_integrity"]["plan_design_hashes_by_algorithm"]

    low_run_audit = plan_audit.build_audit(
        plans=plans,
        plan_paths=[],
        min_runs=4,
        min_prompts=2,
        min_gammas=3,
        summary_energy_key=experiment_report.DEFAULT_SUMMARY_ENERGY_KEY,
        paired_energy_key=experiment_report.DEFAULT_PAIRED_ENERGY_KEY,
    )
    assert low_run_audit["ok"] is False
    assert (
        "plan_design:insufficient_plan_measured_runs"
        in low_run_audit["errors"]
    )

    low_gamma_audit = plan_audit.build_audit(
        plans=plans,
        plan_paths=[],
        min_runs=3,
        min_prompts=2,
        min_gammas=5,
        summary_energy_key=experiment_report.DEFAULT_SUMMARY_ENERGY_KEY,
        paired_energy_key=experiment_report.DEFAULT_PAIRED_ENERGY_KEY,
    )
    assert low_gamma_audit["ok"] is False
    assert "plan_design:insufficient_plan_gamma_values" in low_gamma_audit["errors"]

    broken_factorial_plan = json.loads(json.dumps(spec_plan))
    broken_factorial_plan["combinations"] = [
        combo
        for combo in broken_factorial_plan["combinations"]
        if not (
            combo["prompt_id"] == "p0"
            and str(combo["gamma"]) == "4"
            and str(combo["drafter_freq_hz"]) == "408000000"
            and str(combo["verifier_clock_mhz"]) == "810"
        )
    ]
    broken_factorial_report = experiment_report.plan_design_report(
        [broken_factorial_plan, baseline_plan],
        min_runs=3,
        min_prompts=2,
        min_gammas=3,
    )
    assert broken_factorial_report["ok"] is False
    assert "incomplete_spec_factorial_grid" in broken_factorial_report["errors"]
    assert broken_factorial_report["incomplete_spec_factorial_groups"][0][
        "missing_cell_count"
    ] == 1


def test_prompt_ids_must_be_unique() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prompts.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "dup", "prompt": "first"}) + "\n")
            f.write(json.dumps({"id": "dup", "prompt": "second"}) + "\n")

        args = _driver_args(prompts_jsonl=str(path))
        prompt_cases = spec_driver.load_prompt_cases(args, allow_missing=False)
        try:
            spec_driver.validate_prompt_cases(prompt_cases)
        except ValueError as exc:
            assert "Prompt ids must be unique" in str(exc)
            assert "dup" in str(exc)
        else:
            raise AssertionError("Expected duplicate prompt id validation error")


def test_doctor_prompt_source_checks_unique_prompts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        duplicate_path = Path(tmpdir) / "duplicate_prompts.jsonl"
        duplicate_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "p0", "prompt": "same prompt"}),
                    json.dumps({"id": "p1", "prompt": "same prompt"}),
                ]
            ),
            encoding="utf-8",
        )
        check = experiment_doctor.check_prompt_source(
            None,
            None,
            str(duplicate_path),
            min_prompts=2,
        )
        assert check.status == "fail"
        assert "unique" in check.message

        unique_path = Path(tmpdir) / "unique_prompts.jsonl"
        unique_path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "p0", "prompt": "first prompt"}),
                    json.dumps({"id": "p1", "prompt": "second prompt"}),
                ]
            ),
            encoding="utf-8",
        )
        check = experiment_doctor.check_prompt_source(
            None,
            None,
            str(unique_path),
            min_prompts=3,
        )
        assert check.status == "fail"
        assert "fewer unique prompts" in check.message

        check = experiment_doctor.check_prompt_source(
            None,
            None,
            str(unique_path),
            min_prompts=2,
        )
        assert check.status == "ok"
        assert check.details["unique_prompt_hashes"] == 2


def test_doctor_checks_k8s_manifest_template() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "k8s/spec-decoding.yaml"
    check = experiment_doctor.check_k8s_manifest_template(
        str(manifest_path),
        require_manifest=True,
    )
    assert check.status == "ok"
    assert check.details["document_count"] == 20
    assert check.details["sha256"]

    missing = experiment_doctor.check_k8s_manifest_template(
        str(manifest_path.parent / "missing-spec-decoding.yaml"),
        require_manifest=True,
    )
    assert missing.status == "fail"


def test_doctor_hf_token_and_cache_checks() -> None:
    previous_hf_token = os.environ.pop("HF_TOKEN", None)
    previous_hub_token = os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    try:
        check = experiment_doctor.check_hf_token(require_token=True)
        assert check.status == "fail"
        assert "HF_TOKEN" in check.message

        os.environ["HF_TOKEN"] = "test-token"
        check = experiment_doctor.check_hf_token(require_token=True)
        assert check.status == "ok"
        assert check.details["env_var"] == "HF_TOKEN"
    finally:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        if previous_hf_token is not None:
            os.environ["HF_TOKEN"] = previous_hf_token
        if previous_hub_token is not None:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = previous_hub_token

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "hf"
        check = experiment_doctor.check_hf_cache_dir(str(cache_dir))
        assert check.status == "ok"
        assert check.details["path"] == str(cache_dir)


def test_condition_shuffle_seed() -> None:
    args = _driver_args(shuffle_conditions=True, seed=17)
    prompt_cases = spec_driver.load_prompt_cases(args, allow_missing=True)
    first = spec_driver.build_spec_conditions(args, prompt_cases)
    second = spec_driver.build_spec_conditions(args, prompt_cases)
    assert first == second

    fixed = spec_driver.build_spec_conditions(
        _driver_args(shuffle_conditions=False),
        prompt_cases,
    )
    assert first != fixed

    baseline_args = _driver_args(verifier_clocks_mhz="810,1410", shuffle_conditions=True)
    baseline_first = verifier_baseline_driver.build_baseline_conditions(
        baseline_args,
        prompt_cases,
    )
    baseline_second = verifier_baseline_driver.build_baseline_conditions(
        baseline_args,
        prompt_cases,
    )
    assert baseline_first == baseline_second
    baseline_warmups = verifier_baseline_driver.build_baseline_warmup_schedule(
        _driver_args(verifier_clocks_mhz="810,1410", warmup_runs=2),
        baseline_first,
    )
    assert len(baseline_warmups) == 4

    schedule_args = _driver_args(
        gammas="1,2",
        drafter_freqs_hz="408000000",
        verifier_clocks_mhz="810",
        runs=3,
        warmup_runs=2,
    )
    schedule_conditions = spec_driver.build_spec_conditions(
        schedule_args,
        prompt_cases,
    )
    ordered_schedule = spec_driver.build_spec_run_schedule(
        schedule_args,
        schedule_conditions,
    )
    assert [(entry.condition_index, entry.run_index) for entry in ordered_schedule] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    ordered_warmups = spec_driver.build_spec_warmup_schedule(
        schedule_args,
        schedule_conditions,
    )
    assert [(entry.condition_index, entry.warmup_index) for entry in ordered_warmups] == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]

    shuffled_args = _driver_args(
        gammas="1,2",
        drafter_freqs_hz="408000000",
        verifier_clocks_mhz="810",
        runs=3,
        warmup_runs=2,
        shuffle_runs=True,
        seed=9,
    )
    shuffled_schedule = spec_driver.build_spec_run_schedule(
        shuffled_args,
        schedule_conditions,
    )
    shuffled_schedule_again = spec_driver.build_spec_run_schedule(
        shuffled_args,
        schedule_conditions,
    )
    assert shuffled_schedule == shuffled_schedule_again
    assert shuffled_schedule != ordered_schedule
    assert sorted((entry.condition_index, entry.run_index) for entry in shuffled_schedule) == sorted(
        (entry.condition_index, entry.run_index) for entry in ordered_schedule
    )
    shuffled_warmups = spec_driver.build_spec_warmup_schedule(
        shuffled_args,
        schedule_conditions,
    )
    shuffled_warmups_again = spec_driver.build_spec_warmup_schedule(
        shuffled_args,
        schedule_conditions,
    )
    assert shuffled_warmups == shuffled_warmups_again
    assert shuffled_warmups != ordered_warmups
    assert sorted(
        (entry.condition_index, entry.warmup_index) for entry in shuffled_warmups
    ) == sorted((entry.condition_index, entry.warmup_index) for entry in ordered_warmups)


def test_thermal_guard_metadata() -> None:
    args = _driver_args(
        max_start_temp_c=70.0,
        thermal_check_interval_s=0.001,
        thermal_wait_timeout_s=0.1,
    )
    calls = {"count": 0}

    async def refresh_metadata():
        calls["count"] += 1
        return {"drafter_thermal_max_temp_c": "65.0"}

    metadata = asyncio.run(
        spec_driver.thermal_guard_metadata(
            args=args,
            initial_metadata={"drafter_thermal_max_temp_c": "75.0"},
            refresh_metadata=refresh_metadata,
            prefixes=["drafter"],
            label="test run",
        )
    )
    assert metadata["drafter_thermal_max_temp_c"] == "65.0"
    assert calls["count"] == 1

    try:
        asyncio.run(
            spec_driver.thermal_guard_metadata(
                args=_driver_args(
                    max_start_temp_c=70.0,
                    thermal_check_interval_s=0.001,
                    thermal_wait_timeout_s=0.0,
                ),
                initial_metadata={"verifier_nvidia_gpu_temp_c": "80.0"},
                refresh_metadata=refresh_metadata,
                prefixes=["verifier"],
                label="hot run",
            )
        )
    except RuntimeError as exc:
        assert "Thermal guard timed out" in str(exc)
    else:
        raise AssertionError("expected thermal guard timeout")


def test_startup_health_wait() -> None:
    old_pb2 = spec_driver.spec_pb2
    spec_driver.spec_pb2 = _FakeSpecPb2
    try:
        drafter = _FlakyHealthStub(failures=2)
        metadata = asyncio.run(
            spec_driver.wait_for_health(
                "drafter",
                drafter,
                timeout_s=0.1,
                startup_timeout_s=1.0,
                interval_s=0.001,
                verbose=False,
            )
        )
        assert drafter.calls == 3
        assert metadata["drafter_model"] == "fake-model"
        assert metadata["drafter_model_parameter_count"] == "1000"

        verifier = _FlakyHealthStub(failures=1)
        baseline_metadata = asyncio.run(
            verifier_baseline_driver.wait_for_health(
                verifier,
                timeout_s=0.1,
                startup_timeout_s=1.0,
                interval_s=0.001,
                verbose=False,
            )
        )
        assert verifier.calls == 2
        assert baseline_metadata["verifier_model"] == "fake-model"
    finally:
        spec_driver.spec_pb2 = old_pb2


async def _run_fake_spec_decode(
    max_new_tokens: int = 4,
    verifier_responses=None,
    drafter_commit_offset: int = 0,
    verifier_commit_offset: int = 0,
    drafter_init_context_offset: int = 0,
    verifier_init_context_offset: int = 0,
):
    old_pb2 = spec_driver.spec_pb2
    spec_driver.spec_pb2 = _FakeSpecPb2
    drafter = _FakeDrafterStub(
        drafts=[[10, 99], [12, 13]],
        commit_offset=drafter_commit_offset,
        init_context_offset=drafter_init_context_offset,
    )
    verifier = _FakeVerifierStub(
        verify_responses=verifier_responses
        or [(1, 11, True), (2, 14, False)],
        commit_offset=verifier_commit_offset,
        init_context_offset=verifier_init_context_offset,
    )
    try:
        rows = await spec_driver.run_decode(
            args=_driver_args(max_new_tokens=max_new_tokens, timeout=5.0),
            tokenizer=_FakeTokenizer(),
            drafter_stub=drafter,
            verifier_stub=verifier,
            gamma=2,
            run_index=1,
            health_metadata={
                "drafter_model": "fake-drafter",
                "verifier_model": "fake-verifier",
                "drafter_model_vocab_size": "32000",
                "verifier_model_vocab_size": "32000",
                "drafter_model_parameter_count": "100000000",
                "verifier_model_parameter_count": "1000000000",
                "drafter_model_type": "llama",
                "verifier_model_type": "llama",
                "drafter_thermal_max_temp_c": "41.50",
                "drafter_thermal_zones": "gpu:41.50,cpu:39.00",
                "verifier_nvidia_gpu_temp_c": "55",
                "verifier_nvidia_pstate": "P2",
                "verifier_nvidia_throttle_active": "Not Active",
            },
            drafter_freq_hz=408,
            verifier_clock_mhz=810,
            prompt_case=spec_driver.make_prompt_case("p0", "prompt", "fake"),
            prompt_set_sha256="prompt-set",
            idle_baseline={
                "idle_baseline_s": None,
                "drafter_idle_power_mw": None,
                "verifier_idle_power_mw": None,
                "drafter_idle_power_samples": 0,
                "verifier_idle_power_samples": 0,
            },
        )
    finally:
        spec_driver.spec_pb2 = old_pb2
    return rows, drafter, verifier


def test_spec_driver_fake_orchestration() -> None:
    rows, drafter, verifier = asyncio.run(_run_fake_spec_decode())
    assert len(rows) == 2
    row = next(item for item in rows if item["rail"] == "tot_power")
    assert row["generated_tokens"] == 4
    assert row["stop_reason"] == "max_new_tokens"
    assert row["output_token_sha256"] == spec_driver.token_ids_sha256([10, 11, 12, 13])
    assert row["output_text"] == "10 11 12 13"
    assert row["steps"] == 2
    assert row["draft_tokens"] == 4
    assert row["accepted_draft_tokens"] == 3
    assert row["replacement_tokens"] == 1
    assert row["accept_rate"] == "0.750000"
    assert row["system_energy_complete"] == 1
    assert row["drafter_model"] == "fake-drafter"
    assert row["verifier_model"] == "fake-verifier"
    assert row["drafter_model_vocab_size"] == "32000"
    assert row["verifier_model_vocab_size"] == "32000"
    assert row["drafter_model_parameter_count"] == "100000000"
    assert row["verifier_model_parameter_count"] == "1000000000"
    assert row["drafter_primary_power_rail"] == "tot_power"
    assert row["verifier_primary_power_rail"] == "verifier_gpu_power"
    assert row["system_primary_power_rails"] == "drafter:tot_power,verifier:verifier_gpu_power"
    assert row["prompt_token_sha256"] == spec_driver.token_ids_sha256([101, 102])
    assert row["result_schema_version"] == spec_driver.RESULT_SCHEMA_VERSION
    assert row["algorithm_version"] == spec_driver.SPEC_ALGORITHM_VERSION
    assert row["idle_baseline_policy"] == "condition"
    assert row["drafter_runtime_temp_c"] == "41.50"
    assert row["drafter_thermal_zones"] == "gpu:41.50,cpu:39.00"
    assert row["verifier_runtime_temp_c"] == "55"
    assert row["verifier_nvidia_pstate"] == "P2"
    assert row["verifier_nvidia_throttle_active"] == "Not Active"
    assert row["rpc_total_bytes"] > 0
    assert float(row["rpc_bytes_per_generated_token"]) > 0

    assert [request.gamma for request in drafter.draft_requests] == [2, 2]
    assert [request.base_committed_tokens for request in drafter.draft_requests] == [
        2,
        4,
    ]
    assert [list(request.draft_tokens) for request in verifier.verify_requests] == [
        [10, 99],
        [12, 13],
    ]
    assert [request.base_committed_tokens for request in verifier.verify_requests] == [
        2,
        4,
    ]
    assert [request.append_replacement for request in verifier.verify_requests] == [
        True,
        False,
    ]
    assert [
        (
            request.accepted_tokens,
            request.replacement_token,
            request.append_replacement,
            request.base_committed_tokens,
        )
        for request in drafter.commit_requests
    ] == [(1, 11, True, 2), (2, 14, False, 4)]
    assert len(drafter.reset_requests) == 1
    assert len(verifier.reset_requests) == 1


def test_spec_driver_rejects_over_budget_replacement() -> None:
    try:
        asyncio.run(
            _run_fake_spec_decode(
                max_new_tokens=2,
                verifier_responses=[(2, 99, True)],
            )
        )
    except RuntimeError as exc:
        assert "beyond max_new_tokens" in str(exc)
    else:
        raise AssertionError("expected over-budget replacement validation error")


def test_spec_driver_rejects_invalid_verifier_progress() -> None:
    try:
        asyncio.run(
            _run_fake_spec_decode(
                verifier_responses=[(3, 99, False)],
            )
        )
    except RuntimeError as exc:
        assert "outside the draft length" in str(exc)
    else:
        raise AssertionError("expected invalid accepted_tokens validation error")

    try:
        asyncio.run(
            _run_fake_spec_decode(
                verifier_responses=[(0, 99, False)],
            )
        )
    except RuntimeError as exc:
        assert "made no token progress" in str(exc)
    else:
        raise AssertionError("expected no-progress validation error")


def test_spec_driver_rejects_drafter_commit_mismatch() -> None:
    try:
        asyncio.run(_run_fake_spec_decode(drafter_commit_offset=-1))
    except RuntimeError as exc:
        assert "Drafter committed token count mismatch" in str(exc)
    else:
        raise AssertionError("expected drafter commit mismatch validation error")


def test_spec_driver_rejects_verifier_commit_mismatch() -> None:
    try:
        asyncio.run(_run_fake_spec_decode(verifier_commit_offset=-1))
    except RuntimeError as exc:
        assert "Verifier committed token count mismatch" in str(exc)
    else:
        raise AssertionError("expected verifier commit mismatch validation error")


def test_spec_driver_rejects_init_context_mismatch() -> None:
    try:
        asyncio.run(_run_fake_spec_decode(drafter_init_context_offset=-1))
    except RuntimeError as exc:
        assert "Drafter InitSession context length mismatch" in str(exc)
    else:
        raise AssertionError("expected drafter init context mismatch validation error")

    try:
        asyncio.run(_run_fake_spec_decode(verifier_init_context_offset=-1))
    except RuntimeError as exc:
        assert "Verifier InitSession context length mismatch" in str(exc)
    else:
        raise AssertionError("expected verifier init context mismatch validation error")


def test_write_rows_checkpoint_replaces_output() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.csv"
        spec_driver.write_rows(
            str(path),
            [{"session_id": "first", "rail": "tot_power"}],
        )
        spec_driver.write_rows(
            str(path),
            [{"session_id": "second", "rail": "tot_power"}],
        )
        rows = list(csv.DictReader(path.open()))
        assert [row["session_id"] for row in rows] == ["second"]
        assert not (Path(tmpdir) / ".results.csv.tmp").exists()


def test_resume_rows_validate_plan_and_orders() -> None:
    plan = {
        "plan_sha256": "plan-a",
        "measurement_schedule": [
            {"order": 1},
            {"order": 2},
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.csv"
        spec_driver.write_rows(
            str(path),
            [
                {
                    "algorithm": "speculative",
                    "plan_sha256": "plan-a",
                    "session_id": "s1",
                    "run": 1,
                    "measurement_order": 1,
                    "rail": "tot_power",
                },
                {
                    "algorithm": "speculative",
                    "plan_sha256": "plan-a",
                    "session_id": "s1",
                    "run": 1,
                    "measurement_order": 1,
                    "rail": "gpu",
                },
            ],
        )
        loaded, completed = spec_driver.load_resume_rows(
            SimpleNamespace(resume=True, out=str(path)),
            plan,
            expected_algorithm="speculative",
        )
        assert len(loaded) == 2
        assert completed == {1}

        spec_driver.write_rows(
            str(path),
            [
                {
                    "algorithm": "speculative",
                    "plan_sha256": "other-plan",
                    "session_id": "s1",
                    "run": 1,
                    "measurement_order": 1,
                    "rail": "tot_power",
                }
            ],
        )
        try:
            spec_driver.load_resume_rows(
                SimpleNamespace(resume=True, out=str(path)),
                plan,
                expected_algorithm="speculative",
            )
        except ValueError as exc:
            assert "different plan_sha256" in str(exc)
        else:
            raise AssertionError("expected resume plan mismatch")

        spec_driver.write_rows(
            str(path),
            [
                {
                    "algorithm": "speculative",
                    "plan_sha256": "plan-a",
                    "session_id": "s1",
                    "run": 1,
                    "measurement_order": 1,
                    "rail": "tot_power",
                },
                {
                    "algorithm": "speculative",
                    "plan_sha256": "plan-a",
                    "session_id": "s2",
                    "run": 1,
                    "measurement_order": 1,
                    "rail": "tot_power",
                },
            ],
        )
        try:
            spec_driver.load_resume_rows(
                SimpleNamespace(resume=True, out=str(path)),
                plan,
                expected_algorithm="speculative",
            )
        except ValueError as exc:
            assert "duplicate sessions" in str(exc)
        else:
            raise AssertionError("expected resume duplicate-order mismatch")


def test_frequency_lock_uses_gpu_index() -> None:
    jetson_lock = frequency.FrequencyLock(
        jetson_gpu_freq_hz=612000000,
        jetson_gpu_devfreq_root="/tmp/fake-devfreq-gpu",
    )
    assert jetson_lock.metadata()["jetson_gpu_devfreq_root"] == "/tmp/fake-devfreq-gpu"

    lock = frequency.FrequencyLock(
        nvidia_smi_gpu_clock_mhz=1410,
        nvidia_smi_gpu_index=2,
    )
    calls = []

    def fake_run(cmd, label):
        calls.append((cmd, label))
        return True

    lock._run = fake_run
    assert lock.set_nvidia_smi_gpu_clock(810) is True
    lock.restore()
    assert calls == [
        (
            ["nvidia-smi", "-i", "2", "-lgc", "810"],
            "lock nvidia-smi GPU clocks",
        ),
        (
            ["nvidia-smi", "-i", "2", "-rgc"],
            "restore nvidia-smi GPU clocks",
        ),
    ]
    assert lock.metadata()["nvidia_smi_gpu_index"] == "2"
    assert lock.metadata()["nvidia_smi_gpu_clock_mhz"] == "810"
    assert lock.metadata()["frequency_label"] == "gpu_index=2,gpu_clock=810MHz"


def test_doctor_version_comparison() -> None:
    assert experiment_doctor.version_at_least("1.68.1", "1.68.1") is True
    assert experiment_doctor.version_at_least("1.68.2", "1.68.1") is True
    assert experiment_doctor.version_at_least("1.69.0", "1.68.1") is True
    assert experiment_doctor.version_at_least("1.68", "1.68.1") is False
    assert experiment_doctor.version_at_least("1.67.9", "1.68.1") is False
    assert experiment_doctor.version_at_least("", "1.68.1") is False


def test_analyzer_summary() -> None:
    rows = [
        {
            "session_id": "s1",
            "gamma": "1",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "810",
            "max_new_tokens": "32",
            "system_energy_complete": "1",
            "system_total_energy_mj": "30",
            "system_total_energy_mj_per_generated_token": "3",
            "tokens_per_s": "10",
            "accept_rate": "0.5",
            "wall_latency_ms": "100",
            "client_rpc_latency_ms": "120",
            "server_compute_latency_ms": "100",
            "estimated_rpc_overhead_ms": "20",
            "generated_tokens": "10",
            "stop_reason": "max_new_tokens",
            "drafter_total_energy_mj": "12",
            "verifier_total_energy_mj": "18",
            "drafter_draft_total_energy_mj": "8",
            "verifier_verify_total_energy_mj": "16",
            "drafter_model": "drafter-test",
            "verifier_model": "verifier-test",
        },
        {
            "session_id": "s1",
            "gamma": "1",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "810",
            "max_new_tokens": "32",
            "system_energy_complete": "1",
            "system_total_energy_mj": "30",
            "system_total_energy_mj_per_generated_token": "3",
            "tokens_per_s": "10",
            "accept_rate": "0.5",
            "wall_latency_ms": "100",
            "client_rpc_latency_ms": "120",
            "server_compute_latency_ms": "100",
            "estimated_rpc_overhead_ms": "20",
            "generated_tokens": "10",
            "stop_reason": "max_new_tokens",
            "drafter_total_energy_mj": "12",
            "verifier_total_energy_mj": "18",
            "drafter_draft_total_energy_mj": "8",
            "verifier_verify_total_energy_mj": "16",
            "drafter_model": "drafter-test",
            "verifier_model": "verifier-test",
        },
        {
            "session_id": "s2",
            "gamma": "1",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "810",
            "max_new_tokens": "32",
            "system_energy_complete": "1",
            "system_total_energy_mj": "40",
            "system_total_energy_mj_per_generated_token": "4",
            "tokens_per_s": "20",
            "accept_rate": "0.75",
            "wall_latency_ms": "90",
            "client_rpc_latency_ms": "90",
            "server_compute_latency_ms": "80",
            "estimated_rpc_overhead_ms": "10",
            "generated_tokens": "10",
            "stop_reason": "max_new_tokens",
            "drafter_total_energy_mj": "20",
            "verifier_total_energy_mj": "20",
            "drafter_draft_total_energy_mj": "10",
            "verifier_verify_total_energy_mj": "18",
            "drafter_model": "drafter-test",
            "verifier_model": "verifier-test",
        },
        {
            "session_id": "s3",
            "gamma": "2",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "810",
            "system_energy_complete": "0",
        },
    ]
    summary = analyze_spec_results.summarize(rows)
    assert len(summary) == 1
    row = summary[0]
    assert row["runs"] == 2
    assert row["prompts"] == 2
    assert row["max_new_tokens"] == "32"
    assert row["drafter_model"] == "drafter-test"
    assert row["verifier_model"] == "verifier-test"
    assert row["prompt_set_sha256"]
    assert row["mean_system_total_energy_mj"] == "35.000000"
    assert row["mean_client_rpc_latency_ms"] == "105.000000"
    assert row["mean_server_compute_latency_ms"] == "90.000000"
    assert row["mean_estimated_rpc_overhead_ms"] == "15.000000"
    assert row["stop_reasons"] == "max_new_tokens"
    assert row["max_token_stop_runs"] == 2
    assert row["stderr_system_energy_mj_per_token"] == "0.500000"
    assert row["ci95_system_energy_mj_per_token"] == "0.980000"
    assert row["mean_drafter_total_energy_mj_per_token"] == "1.600000"
    assert row["mean_drafter_active_energy_mj_per_token"] == "1.600000"
    assert row["mean_drafter_draft_energy_mj_per_token"] == "0.900000"
    assert row["mean_drafter_draft_active_energy_mj_per_token"] == "0.900000"
    assert row["mean_drafter_energy_share"] == "0.450000"


def test_gamma_effect_analysis() -> None:
    common = {
        "algorithm": "speculative",
        "prompt_set_sha256": "prompt-set",
        "drafter_freq_hz": "408",
        "verifier_clock_mhz": "810",
        "decoding_mode": "greedy",
        "max_new_tokens": "32",
        "drafter_model": "drafter-test",
        "verifier_model": "verifier-test",
        "system_energy_complete": "1",
        "rail": "tot_power",
        "generated_tokens": "10",
        "draft_tokens": "20",
        "tokens_per_s": "10",
        "wall_latency_ms": "100",
        "accept_rate": "0.5",
    }
    rows = [
        {
            **common,
            "session_id": "g1-p0",
            "gamma": "1",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "drafter_total_energy_mj": "10",
            "drafter_active_energy_mj": "8",
            "drafter_draft_total_energy_mj": "6",
            "drafter_draft_active_energy_mj": "6",
        },
        {
            **common,
            "session_id": "g1-p1",
            "gamma": "1",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "drafter_total_energy_mj": "20",
            "drafter_active_energy_mj": "16",
            "drafter_draft_total_energy_mj": "10",
            "drafter_draft_active_energy_mj": "10",
        },
        {
            **common,
            "session_id": "g2-p0",
            "gamma": "2",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "drafter_total_energy_mj": "5",
            "drafter_active_energy_mj": "4",
            "drafter_draft_total_energy_mj": "4",
            "drafter_draft_active_energy_mj": "4",
        },
        {
            **common,
            "session_id": "g2-p1",
            "gamma": "2",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "drafter_total_energy_mj": "15",
            "drafter_active_energy_mj": "12",
            "drafter_draft_total_energy_mj": "8",
            "drafter_draft_active_energy_mj": "8",
        },
    ]
    summary = analyze_gamma_effect.summarize(rows)
    assert len(summary) == 2
    baseline, gamma2 = summary
    assert baseline["gamma"] == "1"
    assert baseline["baseline_gamma"] == "1"
    assert baseline["complete_prompt_overlap"] == "1"
    assert baseline["common_prompts"] == "2"
    assert baseline["paired_prompts_vs_baseline_gamma"] == "2"
    assert baseline["complete_prompt_overlap_vs_baseline_gamma"] == "1"
    assert baseline["mean_drafter_total_energy_mj_per_generated_token"] == "1.500000"
    assert baseline["drafter_total_energy_ratio_vs_baseline_gamma"] == "1.000000"
    assert baseline[
        "paired_mean_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] == "0.000000"
    assert baseline[
        "paired_sign_test_p_value_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] == ""
    assert gamma2["gamma"] == "2"
    assert gamma2["mean_drafter_total_energy_mj_per_generated_token"] == "1.000000"
    assert gamma2["drafter_total_energy_ratio_vs_baseline_gamma"] == "0.666667"
    assert gamma2["drafter_total_energy_change_pct_vs_baseline_gamma"] == "-33.333333"
    assert gamma2[
        "paired_mean_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] == "-37.500000"
    assert gamma2[
        "paired_ci95_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] == "24.500000"
    assert gamma2[
        "paired_bootstrap_ci95_low_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] == "-50.000000"
    assert gamma2[
        "paired_bootstrap_ci95_high_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] == "-25.000000"
    assert gamma2[
        "paired_sign_test_p_value_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] == "0.500000"
    assert gamma2[
        "paired_mean_drafter_active_energy_change_pct_vs_baseline_gamma"
    ] == "-37.500000"
    assert gamma2[
        "paired_bootstrap_ci95_low_drafter_active_energy_change_pct_vs_baseline_gamma"
    ] == "-50.000000"
    assert gamma2[
        "paired_bootstrap_ci95_high_drafter_active_energy_change_pct_vs_baseline_gamma"
    ] == "-25.000000"
    assert gamma2[
        "paired_sign_test_p_value_drafter_active_energy_change_pct_vs_baseline_gamma"
    ] == "0.500000"
    assert gamma2[
        "paired_mean_drafter_draft_energy_change_pct_vs_baseline_gamma"
    ] == "-26.666667"
    assert gamma2[
        "paired_bootstrap_ci95_low_drafter_draft_energy_change_pct_vs_baseline_gamma"
    ] == "-33.333333"
    assert gamma2[
        "paired_bootstrap_ci95_high_drafter_draft_energy_change_pct_vs_baseline_gamma"
    ] == "-20.000000"
    assert gamma2[
        "paired_sign_test_p_value_drafter_draft_energy_change_pct_vs_baseline_gamma"
    ] == "0.500000"
    assert gamma2[
        "paired_mean_drafter_draft_active_energy_change_pct_vs_baseline_gamma"
    ] == "-26.666667"
    assert gamma2[
        "paired_bootstrap_ci95_low_drafter_draft_active_energy_change_pct_vs_baseline_gamma"
    ] == "-33.333333"
    assert gamma2[
        "paired_bootstrap_ci95_high_drafter_draft_active_energy_change_pct_vs_baseline_gamma"
    ] == "-20.000000"
    assert gamma2[
        "paired_sign_test_p_value_drafter_draft_active_energy_change_pct_vs_baseline_gamma"
    ] == "0.500000"
    assert gamma2["mean_drafter_draft_energy_mj_per_draft_token"] == "0.300000"
    assert (
        gamma2["mean_drafter_draft_active_energy_mj_per_draft_token"]
        == "0.300000"
    )
    assert gamma2[
        "paired_mean_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma"
    ] == "-26.666667"
    assert gamma2[
        "paired_bootstrap_ci95_low_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma"
    ] == "-33.333333"
    assert gamma2[
        "paired_bootstrap_ci95_high_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma"
    ] == "-20.000000"
    assert gamma2[
        "paired_sign_test_p_value_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma"
    ] == "0.500000"
    assert gamma2["log2_gamma_slope_drafter_total_energy_mj_per_token"] == "-0.500000"
    assert gamma2["log2_gamma_slope_drafter_active_energy_mj_per_token"] == "-0.400000"
    assert (
        gamma2["log2_gamma_slope_drafter_draft_active_energy_mj_per_token"]
        == "-0.200000"
    )
    assert gamma2["pearson_log2_gamma_drafter_active_energy"] == "-1.000000"
    assert gamma2["pearson_log2_gamma_drafter_draft_active_energy"] == "-1.000000"


def test_analyzer_keeps_incompatible_conditions_separate() -> None:
    base = {
        "algorithm": "speculative",
        "session_id": "s1",
        "gamma": "2",
        "prompt_id": "p0",
        "prompt_sha256": "hash-p0",
        "prompt_set_sha256": "prompt-set-a",
        "drafter_freq_hz": "408",
        "verifier_clock_mhz": "810",
        "decoding_mode": "greedy",
        "max_new_tokens": "32",
        "drafter_model": "drafter-a",
        "verifier_model": "verifier-a",
        "system_energy_complete": "1",
        "system_total_energy_mj": "30",
        "system_total_energy_mj_per_generated_token": "3",
        "tokens_per_s": "10",
        "accept_rate": "0.5",
        "wall_latency_ms": "100",
        "generated_tokens": "10",
    }
    rows = [
        base,
        {**base, "session_id": "s2", "max_new_tokens": "64"},
        {**base, "session_id": "s3", "prompt_set_sha256": "prompt-set-b"},
        {**base, "session_id": "s4", "verifier_model": "verifier-b"},
    ]
    summary = analyze_spec_results.summarize(rows)
    assert len(summary) == 4
    assert {row["max_new_tokens"] for row in summary} == {"32", "64"}
    assert {row["prompt_set_sha256"] for row in summary} == {
        "prompt-set-a",
        "prompt-set-b",
    }
    assert {row["verifier_model"] for row in summary} == {
        "verifier-a",
        "verifier-b",
    }


def test_analyzer_read_csvs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        first = Path(tmpdir) / "first.csv"
        second = Path(tmpdir) / "second.csv"
        fieldnames = ["session_id", "system_energy_complete"]
        for path, session_id in ((first, "s1"), (second, "s2")):
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "session_id": session_id,
                        "system_energy_complete": "1",
                    }
                )

        rows = analyze_spec_results.read_csvs([str(first), str(second)])
        assert [row["session_id"] for row in rows] == ["s1", "s2"]


def test_session_rows_prefer_primary_rail() -> None:
    rows = [
        {
            "algorithm": "speculative",
            "session_id": "s1",
            "run": "1",
            "rail": "gpu",
            "system_energy_complete": "1",
            "system_total_energy_mj_per_generated_token": "",
        },
        {
            "algorithm": "speculative",
            "session_id": "s1",
            "run": "1",
            "rail": "tot_power",
            "system_energy_complete": "1",
            "system_total_energy_mj_per_generated_token": "1.23",
        },
        {
            "algorithm": "verifier_only",
            "session_id": "b1",
            "run": "1",
            "rail": "other",
            "system_energy_complete": "1",
            "system_total_energy_mj_per_generated_token": "",
        },
        {
            "algorithm": "verifier_only",
            "session_id": "b1",
            "run": "1",
            "rail": "verifier_gpu_power",
            "system_energy_complete": "1",
            "system_total_energy_mj_per_generated_token": "2.34",
        },
    ]
    validate_rows = {row["session_id"]: row for row in validate_results.session_rows(rows)}
    assert validate_rows["s1"]["rail"] == "tot_power"
    assert validate_rows["b1"]["rail"] == "verifier_gpu_power"

    analyze_rows = {
        row["session_id"]: row
        for row in analyze_spec_results._session_rows(
            rows,
            allow_incomplete_energy=False,
        )
    }
    assert analyze_rows["s1"]["system_total_energy_mj_per_generated_token"] == "1.23"
    assert analyze_rows["b1"]["system_total_energy_mj_per_generated_token"] == "2.34"


def test_validate_results() -> None:
    plan = {
        "algorithm": "speculative",
        "decoding_mode": "greedy",
        "idle_baseline_s": 5,
        "max_new_tokens": 32,
        "combinations": [
            {
                "prompt_id": "p0",
                "prompt_sha256": "hash-p0",
                "gamma": 2,
                "drafter_freq_hz": 408,
                "verifier_clock_mhz": 810,
                "measured_runs": 2,
            }
        ],
    }
    base_row = {
        "algorithm": "speculative",
        "prompt_id": "p0",
        "prompt_sha256": "hash-p0",
        "gamma": "2",
        "drafter_freq_hz": "408",
        "verifier_clock_mhz": "810",
        "decoding_mode": "greedy",
        "max_new_tokens": "32",
        "system_energy_complete": "1",
        "idle_baseline_s": "5",
        "drafter_prefill_power_samples": "2",
        "drafter_draft_power_samples": "3",
        "drafter_commit_power_samples": "2",
        "verifier_prefill_power_samples": "2",
        "verifier_verify_power_samples": "3",
        "verifier_decode_power_samples": "",
        "drafter_power_samples": "6",
        "verifier_power_samples": "7",
        "drafter_idle_power_samples": "12",
        "verifier_idle_power_samples": "12",
        "generated_tokens": "32",
        "stop_reason": "max_new_tokens",
        "tokens_per_s": "10",
        "wall_latency_ms": "100",
        "system_total_energy_mj": "30",
        "system_total_energy_mj_per_generated_token": "0.9375",
        "drafter_total_energy_mj": "12",
        "verifier_total_energy_mj": "18",
        "drafter_draft_total_energy_mj": "8",
        "verifier_verify_total_energy_mj": "16",
        "drafter_active_energy_mj": "10",
        "verifier_active_energy_mj": "15",
        "system_active_energy_mj": "25",
        "system_active_energy_mj_per_generated_token": "0.78125",
    }
    rows = [
        {**base_row, "session_id": "s1", "run": "1", "rail": "tot_power"},
        {**base_row, "session_id": "s1", "run": "1", "rail": "gpu_power"},
        {**base_row, "session_id": "s2", "run": "2", "rail": "tot_power"},
    ]
    report = validate_results.validate(
        plans=[plan],
        raw_rows=rows,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is True
    assert report["expected_sessions"] == 2
    assert report["observed_sessions"] == 2
    assert report["missing_run_indices"] == []
    assert report["duplicate_run_indices"] == []

    eos_stop_plan = {
        **plan,
        "stop_token_policy": "tokenizer_eos",
        "stop_token_ids": "",
    }
    eos_stop_rows = [
        {
            **row,
            "stop_token_policy": "tokenizer_eos",
            "stop_token_ids": "2",
        }
        for row in rows
    ]
    report = validate_results.validate(
        plans=[eos_stop_plan],
        raw_rows=eos_stop_rows,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is True

    custom_stop_plan = {
        **plan,
        "stop_token_policy": "custom",
        "stop_token_ids": "2",
    }
    custom_stop_rows = [
        {
            **row,
            "stop_token_policy": "custom",
            "stop_token_ids": "3",
        }
        for row in rows
    ]
    report = validate_results.validate(
        plans=[custom_stop_plan],
        raw_rows=custom_stop_rows,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert len(report["missing"]) == 1
    assert len(report["extra"]) == 1

    scheduled_plan = {
        **plan,
        "measurement_schedule": [
            {"order": 1, "condition_order": 0, "run": 1},
            {"order": 2, "condition_order": 0, "run": 2},
        ],
    }
    scheduled_rows = [
        {**rows[0], "measurement_order": "1"},
        {**rows[1], "measurement_order": "1"},
        {**rows[2], "measurement_order": "2"},
    ]
    report = validate_results.validate(
        plans=[scheduled_plan],
        raw_rows=scheduled_rows,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is True
    assert report["missing_measurement_orders"] == []
    assert report["duplicate_measurement_orders"] == []
    assert report["measurement_schedule_mismatches"] == []

    duplicate_order_rows = [dict(row) for row in scheduled_rows]
    duplicate_order_rows[2]["measurement_order"] = "1"
    report = validate_results.validate(
        plans=[scheduled_plan],
        raw_rows=duplicate_order_rows,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert report["missing_measurement_orders"][0]["missing_measurement_orders"] == [2]
    assert report["duplicate_measurement_orders"][0]["duplicate_measurement_orders"] == [1]

    wrong_schedule_rows = [dict(row) for row in scheduled_rows]
    wrong_schedule_rows[0]["measurement_order"] = "2"
    wrong_schedule_rows[1]["measurement_order"] = "2"
    wrong_schedule_rows[2]["measurement_order"] = "1"
    report = validate_results.validate(
        plans=[scheduled_plan],
        raw_rows=wrong_schedule_rows,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert report["missing_measurement_orders"] == []
    assert report["duplicate_measurement_orders"] == []
    assert len(report["measurement_schedule_mismatches"]) == 2
    assert (
        report["measurement_schedule_mismatches"][0]["reason"]
        == "measurement_order_condition_mismatch"
    )

    duplicate_run = [dict(row) for row in rows]
    duplicate_run[2]["run"] = "1"
    duplicate_run[2]["session_id"] = "s3"
    report = validate_results.validate(
        plans=[plan],
        raw_rows=duplicate_run,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert report["missing_run_indices"][0]["missing_run_indices"] == [2]
    assert report["duplicate_run_indices"][0]["duplicate_run_indices"] == [1]

    wrong_mode = [dict(row) for row in rows]
    wrong_mode[2]["decoding_mode"] = "top_p"
    report = validate_results.validate(
        plans=[plan],
        raw_rows=wrong_mode,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert len(report["missing"]) == 1
    assert len(report["extra"]) == 1

    bad_generation = [dict(row) for row in rows]
    bad_generation[0]["generated_tokens"] = "33"
    report = validate_results.validate(
        plans=[plan],
        raw_rows=bad_generation,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert report["invalid_generation_sessions"] == ["s1"]

    bad_metric = [dict(row) for row in rows]
    bad_metric[0]["system_total_energy_mj_per_generated_token"] = "nan"
    report = validate_results.validate(
        plans=[plan],
        raw_rows=bad_metric,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert report["invalid_metric_sessions"] == ["s1"]

    bad_active_metric = [dict(row) for row in rows]
    bad_active_metric[0]["system_active_energy_mj_per_generated_token"] = "-1"
    report = validate_results.validate(
        plans=[plan],
        raw_rows=bad_active_metric,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert report["invalid_metric_sessions"] == ["s1"]

    low_samples = [dict(row) for row in rows]
    low_samples[0]["drafter_draft_power_samples"] = "1"
    report = validate_results.validate(
        plans=[plan],
        raw_rows=low_samples,
        require_complete_energy=True,
        require_idle_baseline=True,
        min_power_samples=2,
    )
    assert report["ok"] is False
    assert report["insufficient_power_sample_sessions"] == ["s1"]

    wrong_prompt = [dict(row) for row in rows]
    wrong_prompt[2]["prompt_sha256"] = "hash-other"
    report = validate_results.validate(
        plans=[plan],
        raw_rows=wrong_prompt,
        require_complete_energy=True,
        require_idle_baseline=True,
    )
    assert report["ok"] is False
    assert len(report["missing"]) == 1
    assert len(report["extra"]) == 1


def test_select_best_config() -> None:
    rows = [
        {
            "algorithm": "speculative",
            "gamma": "1",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "810",
            "runs": "3",
            "prompts": "2",
            "mean_system_energy_mj_per_token": "2.0",
            "mean_active_system_energy_mj_per_token": "1.2",
            "ci95_active_system_energy_mj_per_token": "0.05",
            "mean_drafter_active_energy_mj_per_token": "2.0",
            "ci95_drafter_active_energy_mj_per_token": "0.05",
            "mean_tokens_per_s": "20",
            "mean_wall_latency_ms": "100",
        },
        {
            "algorithm": "speculative",
            "gamma": "2",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "810",
            "runs": "3",
            "prompts": "2",
            "mean_system_energy_mj_per_token": "1.5",
            "mean_active_system_energy_mj_per_token": "1.4",
            "ci95_active_system_energy_mj_per_token": "0.20",
            "mean_drafter_active_energy_mj_per_token": "1.5",
            "ci95_drafter_active_energy_mj_per_token": "0.20",
            "mean_tokens_per_s": "18",
            "mean_wall_latency_ms": "110",
        },
        {
            "algorithm": "speculative",
            "gamma": "4",
            "drafter_freq_hz": "612",
            "verifier_clock_mhz": "1410",
            "runs": "3",
            "prompts": "2",
            "mean_system_energy_mj_per_token": "1.8",
            "mean_active_system_energy_mj_per_token": "1.0",
            "ci95_active_system_energy_mj_per_token": "0.05",
            "mean_drafter_active_energy_mj_per_token": "1.8",
            "ci95_drafter_active_energy_mj_per_token": "0.05",
            "mean_tokens_per_s": "25",
            "mean_wall_latency_ms": "90",
        },
    ]
    feasible = [
        row
        for row in rows
        if select_best_config.passes_constraints(
            row=row,
            algorithms=["speculative"],
            min_tokens_per_s=19.0,
            max_wall_latency_ms=None,
            min_runs=3,
            min_prompts=2,
        )
    ]
    assert len(feasible) == 2
    best = select_best_config.best_energy(feasible)
    assert best is not None
    assert best["gamma"] == "4"
    front = select_best_config.pareto_front(rows)
    assert {row["gamma"] for row in front} == {"2", "4"}
    active_best = select_best_config.best_energy(
        rows,
        energy_key="mean_active_system_energy_mj_per_token",
    )
    assert active_best is not None
    assert active_best["gamma"] == "4"
    optimization = select_best_config.optimization_summary(
        feasible,
        select_best_config.pareto_front(feasible),
        best,
    )
    assert optimization["ok"] is True
    assert optimization["best_joint_config"]["gamma"] == "4"
    assert optimization["runner_up_config"]["gamma"] == "1"
    assert optimization["energy_margin_pct_vs_runner_up"] == "11.111111"
    assert optimization["energy_ci95_margin_clear"] == "1"
    assert optimization["best_gamma_one_config"]["gamma"] == "1"
    assert optimization["energy_savings_pct_vs_best_gamma_one"] == "10.000000"
    assert optimization["throughput_ratio_vs_best_gamma_one"] == "1.250000"
    policy = select_best_config.policy_report(
        rows,
        energy_key="mean_active_system_energy_mj_per_token",
    )
    assert policy["ok"] is True
    assert policy["policy_rows"] == 3
    assert policy["uses_gamma_dependent_verifier_clock"] is True
    assert policy["uses_gamma_dependent_drafter_freq"] is True
    assert policy["policy_by_gamma"][0]["gamma"] == "1"
    assert policy["policy_by_gamma"][2]["gamma"] == "4"
    assert policy["policy_by_gamma"][2]["drafter_freq_hz"] == "612"
    assert policy["policy_by_gamma"][2]["verifier_clock_mhz"] == "1410"
    assert policy["policy_by_gamma"][2]["energy_ci95"] == "0.05"
    assert policy["policy_by_gamma"][2]["energy_margin_pct_vs_runner_up"] == ""

    assert (
        select_best_config.passes_constraints(
            row={**rows[0], "mean_tokens_per_s": "nan"},
            algorithms=["speculative"],
            min_tokens_per_s=1.0,
            max_wall_latency_ms=None,
            min_runs=1,
            min_prompts=1,
        )
        is False
    )
    assert (
        select_best_config.passes_constraints(
            row={**rows[0], "mean_drafter_active_energy_mj_per_token": "-1"},
            algorithms=["speculative"],
            min_tokens_per_s=1.0,
            max_wall_latency_ms=None,
            min_runs=1,
            min_prompts=1,
        )
        is False
    )


def test_interaction_report() -> None:
    common = {
        "algorithm": "speculative",
        "prompt_set_sha256": "prompt-set",
        "decoding_mode": "greedy",
        "max_new_tokens": "32",
        "drafter_model": "drafter-test",
        "verifier_model": "verifier-test",
        "runs": "3",
        "prompts": "2",
        "mean_tokens_per_s": "10",
        "mean_wall_latency_ms": "100",
    }
    energies = {
        ("1", "408", "810"): "10",
        ("1", "408", "1410"): "8",
        ("2", "408", "810"): "6",
        ("2", "408", "1410"): "9",
        ("1", "612", "810"): "7",
        ("1", "612", "1410"): "5",
        ("2", "612", "810"): "4",
        ("2", "612", "1410"): "12",
    }
    rows = [
        {
            **common,
            "gamma": gamma,
            "drafter_freq_hz": drafter_freq,
            "verifier_clock_mhz": verifier_clock,
            "mean_active_system_energy_mj_per_token": energy,
        }
        for (gamma, drafter_freq, verifier_clock), energy in energies.items()
    ]
    report = experiment_report.interaction_report(
        rows,
        energy_key="mean_active_system_energy_mj_per_token",
        require_interaction_analysis=True,
    )
    assert report["ok"] is True
    assert report["eligible"] is True
    assert report["verifier_clock_depends_on_gamma"] is True
    assert report["verifier_clock_change_groups"] == 2
    assert report["missing_factorial_cells_count"] == 0
    assert report["best_joint_config"]["gamma"] == "2"
    assert report["best_joint_config"]["drafter_freq_hz"] == "612"
    assert report["best_joint_config"]["verifier_clock_mhz"] == "810"
    assert report["marginal_independent_levels"] == {
        "gamma": "1",
        "drafter_freq_hz": "612",
        "verifier_clock_mhz": "810",
    }
    assert report["independent_energy_gap_pct_vs_joint_best"] == "75.000000"

    missing_cell_report = experiment_report.interaction_report(
        rows[:-1],
        energy_key="mean_active_system_energy_mj_per_token",
        require_interaction_analysis=True,
    )
    assert missing_cell_report["ok"] is False
    assert (
        "interaction_requires_complete_factorial_grid"
        in missing_cell_report["errors"]
    )


def test_compare_to_baseline() -> None:
    rows = [
        {
            "algorithm": "verifier_only",
            "gamma": "",
            "drafter_freq_hz": "",
            "verifier_clock_mhz": "810",
            "decoding_mode": "greedy",
            "max_new_tokens": "32",
            "prompt_set_sha256": "prompt-set",
            "verifier_model": "verifier-test",
            "runs": "6",
            "prompts": "2",
            "mean_system_energy_mj_per_token": "2.2",
            "mean_active_system_energy_mj_per_token": "2.0",
            "mean_tokens_per_s": "10",
            "mean_wall_latency_ms": "120",
        },
        {
            "algorithm": "speculative",
            "gamma": "2",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "810",
            "decoding_mode": "greedy",
            "max_new_tokens": "32",
            "prompt_set_sha256": "prompt-set",
            "drafter_model": "drafter-test",
            "verifier_model": "verifier-test",
            "runs": "6",
            "prompts": "2",
            "mean_system_energy_mj_per_token": "1.76",
            "mean_active_system_energy_mj_per_token": "1.5",
            "mean_tokens_per_s": "15",
            "mean_wall_latency_ms": "80",
            "mean_accept_rate": "0.75",
        },
        {
            "algorithm": "speculative",
            "gamma": "4",
            "drafter_freq_hz": "408",
            "verifier_clock_mhz": "1410",
            "decoding_mode": "greedy",
            "max_new_tokens": "32",
            "prompt_set_sha256": "prompt-set",
            "drafter_model": "drafter-test",
            "verifier_model": "verifier-test",
            "runs": "6",
            "prompts": "2",
            "mean_system_energy_mj_per_token": "1.5",
            "mean_active_system_energy_mj_per_token": "1.2",
            "mean_tokens_per_s": "18",
            "mean_wall_latency_ms": "70",
            "mean_accept_rate": "0.7",
        },
    ]
    compared, unmatched = compare_to_baseline.compare_rows(rows)
    assert unmatched == 1
    assert len(compared) == 1
    row = compared[0]
    assert row["gamma"] == "2"
    assert row["max_new_tokens"] == "32"
    assert row["prompt_set_sha256"] == "prompt-set"
    assert row["verifier_model"] == "verifier-test"
    assert row["energy_savings_pct_vs_baseline"] == "25.000000"
    assert row["system_energy_savings_pct_vs_baseline"] == "20.000000"
    assert row["tokens_per_s_ratio_vs_baseline"] == "1.500000"
    assert row["wall_latency_ratio_vs_baseline"] == "0.666667"


def test_paired_prompt_compare() -> None:
    p0_hash = spec_driver.token_ids_sha256([101, 102, 103])
    p1_hash = spec_driver.token_ids_sha256([201, 202, 203])
    wrong_hash = spec_driver.token_ids_sha256([999])
    common = {
        "decoding_mode": "greedy",
        "system_boundary": "two_device_active",
        "max_new_tokens": "32",
        "prompt_set_sha256": "prompt-set",
        "drafter_model": "",
        "verifier_model": "verifier-test",
        "verifier_clock_mhz": "810",
        "system_energy_complete": "1",
    }
    rows = [
        {
            **common,
            "algorithm": "verifier_only",
            "session_id": "b0",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "system_active_energy_mj_per_generated_token": "2.0",
            "system_total_energy_mj_per_generated_token": "2.2",
            "tokens_per_s": "10",
            "wall_latency_ms": "100",
            "output_token_sha256": p0_hash,
        },
        {
            **common,
            "algorithm": "verifier_only",
            "session_id": "b1",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "system_active_energy_mj_per_generated_token": "4.0",
            "system_total_energy_mj_per_generated_token": "4.4",
            "tokens_per_s": "20",
            "wall_latency_ms": "200",
            "output_token_sha256": p1_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "session_id": "s0",
            "gamma": "2",
            "drafter_freq_hz": "408",
            "drafter_model": "drafter-test",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "system_active_energy_mj_per_generated_token": "1.0",
            "system_total_energy_mj_per_generated_token": "1.1",
            "tokens_per_s": "15",
            "wall_latency_ms": "80",
            "accept_rate": "0.75",
            "output_token_sha256": p0_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "session_id": "s1",
            "gamma": "2",
            "drafter_freq_hz": "408",
            "drafter_model": "drafter-test",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "system_active_energy_mj_per_generated_token": "2.0",
            "system_total_energy_mj_per_generated_token": "2.2",
            "tokens_per_s": "30",
            "wall_latency_ms": "160",
            "accept_rate": "0.65",
            "output_token_sha256": p1_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "session_id": "s_unmatched",
            "gamma": "2",
            "drafter_freq_hz": "408",
            "drafter_model": "drafter-test",
            "prompt_id": "p_missing",
            "prompt_sha256": "hash-missing",
            "system_active_energy_mj_per_generated_token": "1.0",
            "system_total_energy_mj_per_generated_token": "1.1",
            "tokens_per_s": "30",
            "wall_latency_ms": "160",
            "accept_rate": "0.65",
        },
    ]
    summary, pairs = paired_prompt_compare.aggregate_pairs(rows)
    assert len(summary) == 1
    assert len(pairs) == 2
    row = summary[0]
    assert row["paired_prompts"] == "2"
    assert row["spec_runs"] == "2"
    assert row["baseline_runs"] == "2"
    assert row["mean_energy_savings_pct_vs_baseline"] == "50.000000"
    assert row["median_energy_savings_pct_vs_baseline"] == "50.000000"
    assert row["ci95_energy_savings_pct_vs_baseline"] == "0.000000"
    assert row["bootstrap_ci95_low_energy_savings_pct_vs_baseline"] == "50.000000"
    assert row["bootstrap_ci95_high_energy_savings_pct_vs_baseline"] == "50.000000"
    assert row["positive_energy_savings_prompts"] == "2"
    assert row["positive_energy_savings_fraction"] == "1.000000"
    assert row["sign_test_p_value_energy_savings"] == "0.500000"
    assert row["mean_tokens_per_s_ratio_vs_baseline"] == "1.500000"
    assert row["mean_wall_latency_ratio_vs_baseline"] == "0.800000"
    assert row["mean_accept_rate"] == "0.700000"
    assert row["mean_output_token_match"] == "1.000000"
    assert row["output_checked_prompts"] == "2"
    assert pairs[0]["output_token_match"] == "1.000000"
    assert pairs[0]["spec_output_hash_count"] == "1"
    assert pairs[0]["baseline_output_hash_count"] == "1"

    unstable_output = experiment_report.output_equivalence_report(
        [
            {
                "output_token_match": "1.000000",
                "spec_output_hash_count": "2",
                "baseline_output_hash_count": "2",
                "prompt_id": "p0",
                "prompt_sha256": "hash-p0",
            }
        ]
    )
    assert unstable_output["ok"] is False
    assert unstable_output["unstable_prompt_pairs"] == 1

    mismatched_rows = [dict(row) for row in rows]
    for row in mismatched_rows:
        if row.get("algorithm") == "speculative" and row.get("prompt_id") == "p1":
            row["output_token_sha256"] = wrong_hash
    mismatched_summary, _ = paired_prompt_compare.aggregate_pairs(mismatched_rows)
    assert mismatched_summary[0]["mean_output_token_match"] == "0.500000"

    exact_fd_rows = [
        {
            **common,
            "algorithm": "verifier_only",
            "system_boundary": "verifier_only",
            "session_id": "b_generic",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "drafter_freq_hz": "",
            "system_active_energy_mj_per_generated_token": "9.0",
            "system_total_energy_mj_per_generated_token": "9.0",
            "tokens_per_s": "10",
            "wall_latency_ms": "100",
            "output_token_sha256": p0_hash,
        },
        {
            **common,
            "algorithm": "verifier_only",
            "system_boundary": "two_device_idle_drafter",
            "session_id": "b_fd612",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "drafter_freq_hz": "612",
            "system_active_energy_mj_per_generated_token": "3.0",
            "system_total_energy_mj_per_generated_token": "3.3",
            "tokens_per_s": "10",
            "wall_latency_ms": "100",
            "output_token_sha256": p0_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "system_boundary": "two_device_active",
            "session_id": "s_fd612",
            "gamma": "2",
            "drafter_freq_hz": "612",
            "drafter_model": "drafter-test",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "system_active_energy_mj_per_generated_token": "1.5",
            "system_total_energy_mj_per_generated_token": "1.7",
            "tokens_per_s": "15",
            "wall_latency_ms": "80",
            "accept_rate": "0.75",
            "output_token_sha256": p0_hash,
        },
    ]
    exact_summary, exact_pairs = paired_prompt_compare.aggregate_pairs(exact_fd_rows)
    assert len(exact_summary) == 1
    assert len(exact_pairs) == 1
    assert exact_pairs[0]["baseline_energy_mj_per_token"] == "3.000000"
    assert exact_pairs[0]["baseline_system_boundary"] == "two_device_idle_drafter"
    assert exact_summary[0]["baseline_system_boundary"] == "two_device_idle_drafter"


def test_experiment_report() -> None:
    p0_hash = spec_driver.token_ids_sha256([101, 102, 103])
    p1_hash = spec_driver.token_ids_sha256([201, 202, 203])
    p0_prompt_token_hash = spec_driver.token_ids_sha256([11, 12, 13])
    p1_prompt_token_hash = spec_driver.token_ids_sha256([21, 22, 23])
    common = {
        "decoding_mode": "greedy",
        "system_boundary": "two_device_active",
        "max_new_tokens": "32",
        "prompt_set_sha256": "prompt-set",
        "verifier_model": "verifier-test",
        "verifier_clock_mhz": "810",
        "system_energy_complete": "1",
        "run": "1",
        "rail": "tot_power",
        "drafter_primary_power_rail": "tot_power",
        "verifier_primary_power_rail": "verifier_gpu_power",
        "system_primary_power_rails": "drafter:tot_power,verifier:verifier_gpu_power",
        "stop_reason": "eos",
        "prompt_tokens": "3",
        "drafter_prefill_power_samples": "3",
        "drafter_draft_power_samples": "3",
        "drafter_commit_power_samples": "3",
        "verifier_prefill_power_samples": "3",
        "verifier_verify_power_samples": "3",
        "verifier_decode_power_samples": "3",
        "drafter_power_samples": "3",
        "verifier_power_samples": "3",
        "idle_baseline_s": "5",
        "idle_baseline_policy": "run",
        "drafter_idle_power_mw": "100",
        "verifier_idle_power_mw": "200",
        "system_idle_power_mw": "300",
        "drafter_idle_power_samples": "8",
        "verifier_idle_power_samples": "8",
        "drafter_total_energy_mj": "0",
        "drafter_prefill_total_energy_mj": "0",
        "drafter_draft_total_energy_mj": "0",
        "drafter_commit_total_energy_mj": "0",
        "steps": "10",
        "drafter_model_vocab_size": "32000",
        "verifier_model_vocab_size": "32000",
        "drafter_model_parameter_count": "",
        "verifier_model_parameter_count": "1000000000",
        "drafter_model_bos_token_id": "1",
        "drafter_model_eos_token_id": "2",
        "drafter_model_pad_token_id": "2",
        "verifier_model_bos_token_id": "1",
        "verifier_model_eos_token_id": "2",
        "verifier_model_pad_token_id": "2",
        "drafter_power_interval_s": "0.010000",
        "verifier_power_interval_s": "0.010000",
        "driver_result_schema_version": spec_driver.RESULT_SCHEMA_VERSION,
        "driver_spec_rpc_schema_version": spec_driver.SPEC_RPC_SCHEMA_VERSION,
        "drafter_spec_rpc_schema_version": "",
        "verifier_spec_rpc_schema_version": spec_driver.SPEC_RPC_SCHEMA_VERSION,
        "tokenizer_name_or_path": "test-tokenizer",
        "tokenizer_class": "FakeTokenizer",
        "tokenizer_vocab_size": "32000",
        "tokenizer_base_vocab_size": "32000",
        "tokenizer_bos_token_id": "1",
        "tokenizer_eos_token_id": "2",
        "tokenizer_pad_token_id": "2",
        "tokenizer_unk_token_id": "0",
        "drafter_jetson_gpu_freq_hz": "408",
        "verifier_gpu_clock_mhz": "810",
        "drafter_frequency_lock_ok": "1",
        "verifier_frequency_lock_ok": "1",
        "drafter_hostname": "jetson-node",
        "verifier_hostname": "gpu-node",
        "driver_git_commit": "commit-a",
        "driver_git_dirty": "0",
        "driver_xronos_git_commit": "commit-a",
        "driver_xronos_image": "xronos:gpu",
        "drafter_xronos_git_commit": "commit-a",
        "verifier_xronos_git_commit": "commit-a",
        "drafter_xronos_image": "xronos:jetson",
        "verifier_xronos_image": "xronos:gpu",
        "rpc_request_bytes": "120",
        "rpc_response_bytes": "130",
        "rpc_total_bytes": "250",
        "rpc_bytes_per_generated_token": "25.000000",
        "client_rpc_latency_ms": "100",
        "server_compute_latency_ms": "80",
        "estimated_rpc_overhead_ms": "20",
    }
    plans = [
        {
            "schema_version": spec_driver.RESULT_SCHEMA_VERSION,
            "algorithm": "verifier_only",
            "algorithm_version": spec_driver.BASELINE_ALGORITHM_VERSION,
            "system_boundary": "two_device_idle_drafter",
            "decoding_mode": "greedy",
            "max_new_tokens": 32,
            "idle_baseline_s": 5,
            "idle_baseline_policy": "run",
            "tokenizer": "test-tokenizer",
            "prompts": [
                {
                    "prompt_id": "p0",
                    "prompt_sha256": "hash-p0",
                    "prompt_chars": 10,
                },
                {
                    "prompt_id": "p1",
                    "prompt_sha256": "hash-p1",
                    "prompt_chars": 10,
                },
            ],
            "warmup_runs": 1,
            "shuffle_runs": True,
            "seed": 42,
            "combinations": [
                {
                    "prompt_id": "p0",
                    "prompt_sha256": "hash-p0",
                    "drafter_freq_hz": 408,
                    "verifier_clock_mhz": 810,
                    "measured_runs": 1,
                },
                {
                    "prompt_id": "p1",
                    "prompt_sha256": "hash-p1",
                    "drafter_freq_hz": 408,
                    "verifier_clock_mhz": 810,
                    "measured_runs": 1,
                },
            ],
            "measurement_schedule": [
                {"order": 1, "condition_order": 1, "run": 1},
                {"order": 2, "condition_order": 0, "run": 1},
            ],
            "warmup_schedule": [
                {"order": 1, "condition_order": 1, "warmup": 1},
                {"order": 2, "condition_order": 0, "warmup": 1},
            ],
        },
        {
            "schema_version": spec_driver.RESULT_SCHEMA_VERSION,
            "algorithm": "speculative",
            "algorithm_version": spec_driver.SPEC_ALGORITHM_VERSION,
            "decoding_mode": "greedy",
            "max_new_tokens": 32,
            "idle_baseline_s": 5,
            "idle_baseline_policy": "run",
            "tokenizer": "test-tokenizer",
            "prompts": [
                {
                    "prompt_id": "p0",
                    "prompt_sha256": "hash-p0",
                    "prompt_chars": 10,
                },
                {
                    "prompt_id": "p1",
                    "prompt_sha256": "hash-p1",
                    "prompt_chars": 10,
                },
            ],
            "warmup_runs": 1,
            "shuffle_runs": True,
            "seed": 42,
            "combinations": [
                {
                    "prompt_id": "p0",
                    "prompt_sha256": "hash-p0",
                    "gamma": 1,
                    "drafter_freq_hz": 408,
                    "verifier_clock_mhz": 810,
                    "measured_runs": 1,
                },
                {
                    "prompt_id": "p1",
                    "prompt_sha256": "hash-p1",
                    "gamma": 1,
                    "drafter_freq_hz": 408,
                    "verifier_clock_mhz": 810,
                    "measured_runs": 1,
                },
                {
                    "prompt_id": "p0",
                    "prompt_sha256": "hash-p0",
                    "gamma": 2,
                    "drafter_freq_hz": 408,
                    "verifier_clock_mhz": 810,
                    "measured_runs": 1,
                },
                {
                    "prompt_id": "p1",
                    "prompt_sha256": "hash-p1",
                    "gamma": 2,
                    "drafter_freq_hz": 408,
                    "verifier_clock_mhz": 810,
                    "measured_runs": 1,
                },
            ],
            "measurement_schedule": [
                {"order": 1, "condition_order": 2, "run": 1},
                {"order": 2, "condition_order": 0, "run": 1},
                {"order": 3, "condition_order": 3, "run": 1},
                {"order": 4, "condition_order": 1, "run": 1},
            ],
            "warmup_schedule": [
                {"order": 1, "condition_order": 2, "warmup": 1},
                {"order": 2, "condition_order": 0, "warmup": 1},
                {"order": 3, "condition_order": 3, "warmup": 1},
                {"order": 4, "condition_order": 1, "warmup": 1},
            ],
        },
    ]
    for plan in plans:
        spec_driver.attach_plan_sha256(plan)
    plan_sha_by_algorithm = {
        str(plan["algorithm"]): str(plan["plan_sha256"])
        for plan in plans
    }
    plan_design_sha_by_algorithm = {
        str(plan["algorithm"]): str(plan["plan_design_sha256"])
        for plan in plans
    }
    rows = [
        {
            **common,
            "algorithm": "verifier_only",
            "system_boundary": "two_device_idle_drafter",
            "session_id": "b0",
            "rail": "verifier_gpu_power",
            "drafter_primary_power_rail": "tot_power",
            "system_primary_power_rails": "drafter:tot_power,verifier:verifier_gpu_power",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "prompt_token_sha256": p0_prompt_token_hash,
            "gamma": "",
            "drafter_freq_hz": "408",
            "system_active_energy_mj_per_generated_token": "2.0",
            "system_total_energy_mj_per_generated_token": "2.2",
            "system_total_energy_mj": "22",
            "verifier_active_energy_mj": "20",
            "verifier_prefill_active_energy_mj": "2",
            "verifier_verify_active_energy_mj": "18",
            "system_active_energy_mj": "20",
            "verifier_total_energy_mj": "20",
            "verifier_prefill_total_energy_mj": "2",
            "verifier_verify_total_energy_mj": "18",
            "drafter_total_energy_mj": "2",
            "drafter_idle_total_energy_mj": "2",
            "drafter_idle_energy_mj_per_generated_token": "0.2",
            "tokens_per_s": "10",
            "wall_latency_ms": "1000",
            "generated_tokens": "10",
            "output_token_sha256": p0_hash,
        },
        {
            **common,
            "algorithm": "verifier_only",
            "system_boundary": "two_device_idle_drafter",
            "session_id": "b1",
            "rail": "verifier_gpu_power",
            "drafter_primary_power_rail": "tot_power",
            "system_primary_power_rails": "drafter:tot_power,verifier:verifier_gpu_power",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "prompt_token_sha256": p1_prompt_token_hash,
            "gamma": "",
            "drafter_freq_hz": "408",
            "system_active_energy_mj_per_generated_token": "4.0",
            "system_total_energy_mj_per_generated_token": "4.4",
            "system_total_energy_mj": "44",
            "verifier_active_energy_mj": "40",
            "verifier_prefill_active_energy_mj": "4",
            "verifier_verify_active_energy_mj": "36",
            "system_active_energy_mj": "40",
            "verifier_total_energy_mj": "40",
            "verifier_prefill_total_energy_mj": "4",
            "verifier_verify_total_energy_mj": "36",
            "drafter_total_energy_mj": "4",
            "drafter_idle_total_energy_mj": "4",
            "drafter_idle_energy_mj_per_generated_token": "0.4",
            "tokens_per_s": "20",
            "wall_latency_ms": "500",
            "generated_tokens": "10",
            "output_token_sha256": p1_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "session_id": "s_base0",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "prompt_token_sha256": p0_prompt_token_hash,
            "gamma": "1",
            "drafter_freq_hz": "408",
            "drafter_model": "drafter-test",
            "drafter_model_parameter_count": "100000000",
            "system_active_energy_mj_per_generated_token": "1.5",
            "system_total_energy_mj_per_generated_token": "1.65",
            "system_total_energy_mj": "16.5",
            "drafter_active_energy_mj": "5",
            "drafter_prefill_active_energy_mj": "1",
            "drafter_draft_active_energy_mj": "3",
            "drafter_commit_active_energy_mj": "1",
            "verifier_active_energy_mj": "10",
            "verifier_prefill_active_energy_mj": "0.5",
            "verifier_verify_active_energy_mj": "9.5",
            "system_active_energy_mj": "15",
            "drafter_total_energy_mj": "6",
            "drafter_prefill_total_energy_mj": "1",
            "verifier_total_energy_mj": "10.5",
            "drafter_draft_total_energy_mj": "4",
            "drafter_commit_total_energy_mj": "1",
            "verifier_prefill_total_energy_mj": "0.5",
            "verifier_verify_total_energy_mj": "10",
            "tokens_per_s": "12",
            "wall_latency_ms": "833.333333",
            "generated_tokens": "10",
            "draft_tokens": "10",
            "accepted_draft_tokens": "6",
            "replacement_tokens": "4",
            "steps": "10",
            "accept_rate": "0.60",
            "output_token_sha256": p0_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "session_id": "s_base1",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "prompt_token_sha256": p1_prompt_token_hash,
            "gamma": "1",
            "drafter_freq_hz": "408",
            "drafter_model": "drafter-test",
            "drafter_model_parameter_count": "100000000",
            "system_active_energy_mj_per_generated_token": "3.0",
            "system_total_energy_mj_per_generated_token": "3.3",
            "system_total_energy_mj": "33",
            "drafter_active_energy_mj": "10",
            "drafter_prefill_active_energy_mj": "2",
            "drafter_draft_active_energy_mj": "6",
            "drafter_commit_active_energy_mj": "2",
            "verifier_active_energy_mj": "20",
            "verifier_prefill_active_energy_mj": "1",
            "verifier_verify_active_energy_mj": "19",
            "system_active_energy_mj": "30",
            "drafter_total_energy_mj": "12",
            "drafter_prefill_total_energy_mj": "2",
            "verifier_total_energy_mj": "21",
            "drafter_draft_total_energy_mj": "8",
            "drafter_commit_total_energy_mj": "2",
            "verifier_prefill_total_energy_mj": "1",
            "verifier_verify_total_energy_mj": "20",
            "tokens_per_s": "24",
            "wall_latency_ms": "416.666667",
            "generated_tokens": "10",
            "draft_tokens": "10",
            "accepted_draft_tokens": "5",
            "replacement_tokens": "5",
            "steps": "10",
            "accept_rate": "0.50",
            "output_token_sha256": p1_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "session_id": "s0",
            "prompt_id": "p0",
            "prompt_sha256": "hash-p0",
            "prompt_token_sha256": p0_prompt_token_hash,
            "gamma": "2",
            "drafter_freq_hz": "408",
            "drafter_model": "drafter-test",
            "drafter_model_parameter_count": "100000000",
            "system_active_energy_mj_per_generated_token": "1.0",
            "system_total_energy_mj_per_generated_token": "1.1",
            "system_total_energy_mj": "11",
            "drafter_active_energy_mj": "4",
            "drafter_prefill_active_energy_mj": "1",
            "drafter_draft_active_energy_mj": "2",
            "drafter_commit_active_energy_mj": "1",
            "verifier_active_energy_mj": "6",
            "verifier_prefill_active_energy_mj": "1",
            "verifier_verify_active_energy_mj": "5",
            "system_active_energy_mj": "10",
            "drafter_total_energy_mj": "5",
            "drafter_prefill_total_energy_mj": "1",
            "verifier_total_energy_mj": "6",
            "drafter_draft_total_energy_mj": "3",
            "drafter_commit_total_energy_mj": "1",
            "verifier_prefill_total_energy_mj": "1",
            "verifier_verify_total_energy_mj": "5",
            "tokens_per_s": "15",
            "wall_latency_ms": "666.666667",
            "generated_tokens": "10",
            "draft_tokens": "12",
            "accepted_draft_tokens": "9",
            "replacement_tokens": "1",
            "steps": "6",
            "accept_rate": "0.75",
            "output_token_sha256": p0_hash,
        },
        {
            **common,
            "algorithm": "speculative",
            "session_id": "s1",
            "prompt_id": "p1",
            "prompt_sha256": "hash-p1",
            "prompt_token_sha256": p1_prompt_token_hash,
            "gamma": "2",
            "drafter_freq_hz": "408",
            "drafter_model": "drafter-test",
            "drafter_model_parameter_count": "100000000",
            "system_active_energy_mj_per_generated_token": "2.0",
            "system_total_energy_mj_per_generated_token": "2.2",
            "system_total_energy_mj": "22",
            "drafter_active_energy_mj": "8",
            "drafter_prefill_active_energy_mj": "2",
            "drafter_draft_active_energy_mj": "4",
            "drafter_commit_active_energy_mj": "2",
            "verifier_active_energy_mj": "12",
            "verifier_prefill_active_energy_mj": "2",
            "verifier_verify_active_energy_mj": "10",
            "system_active_energy_mj": "20",
            "drafter_total_energy_mj": "10",
            "drafter_prefill_total_energy_mj": "2",
            "verifier_total_energy_mj": "12",
            "drafter_draft_total_energy_mj": "6",
            "drafter_commit_total_energy_mj": "2",
            "verifier_prefill_total_energy_mj": "2",
            "verifier_verify_total_energy_mj": "10",
            "tokens_per_s": "30",
            "wall_latency_ms": "333.333333",
            "generated_tokens": "10",
            "draft_tokens": "12",
            "accepted_draft_tokens": "8",
            "replacement_tokens": "2",
            "steps": "6",
            "accept_rate": "0.666667",
            "output_token_sha256": p1_hash,
        },
    ]
    for row in rows:
        row["result_schema_version"] = spec_driver.RESULT_SCHEMA_VERSION
        row["algorithm_version"] = (
            spec_driver.BASELINE_ALGORITHM_VERSION
            if row["algorithm"] == "verifier_only"
            else spec_driver.SPEC_ALGORITHM_VERSION
        )
        if row["algorithm"] == "speculative":
            row["drafter_spec_rpc_schema_version"] = spec_driver.SPEC_RPC_SCHEMA_VERSION
        row["plan_sha256"] = plan_sha_by_algorithm[row["algorithm"]]
        row["plan_design_sha256"] = plan_design_sha_by_algorithm[row["algorithm"]]
        row["driver_plan_sha256"] = plan_sha_by_algorithm[row["algorithm"]]
        row["driver_plan_design_sha256"] = plan_design_sha_by_algorithm[row["algorithm"]]
        row["measurement_order"] = {
            "b0": "2",
            "b1": "1",
            "s_base0": "2",
            "s_base1": "4",
            "s0": "1",
            "s1": "3",
        }[row["session_id"]]

    def build_trace_events():
        events = [{"event": "plan", "plan": plan} for plan in plans]
        for row in validate_results.session_rows(rows):
            algorithm = row["algorithm"]
            if algorithm == "verifier_only":
                events.append(
                    {
                        "event": "verifier_baseline_run",
                        "algorithm": "verifier_only",
                        "session_id": row["session_id"],
                        "decoding_mode": row["decoding_mode"],
                        "prompt_id": row["prompt_id"],
                        "prompt_sha256": row["prompt_sha256"],
                        "prompt_set_sha256": row["prompt_set_sha256"],
                        "run": int(row["run"]),
                        "measurement_order": int(row["measurement_order"]),
                        "verifier_clock_mhz": int(row["verifier_clock_mhz"]),
                        "generated_tokens": int(row["generated_tokens"]),
                        "stop_reason": row["stop_reason"],
                        "output_token_sha256": row["output_token_sha256"],
                    }
                )
                continue

            steps = int(row["steps"])
            gamma = int(row["gamma"])
            draft_left = int(row["draft_tokens"])
            accepted_left = int(row["accepted_draft_tokens"])
            replacement_left = int(row["replacement_tokens"])
            generated_so_far = 0
            prompt_tokens = int(row["prompt_tokens"])
            for step in range(1, steps + 1):
                base_committed_tokens = prompt_tokens + generated_so_far
                remaining_steps = steps - step + 1
                draft_count = min(gamma, max(1, draft_left - (remaining_steps - 1)))
                accepted = min(draft_count, accepted_left)
                append_replacement = replacement_left > 0 and (
                    accepted == 0 or replacement_left >= remaining_steps
                )
                draft_left -= draft_count
                accepted_left -= accepted
                replacement_left -= int(append_replacement)
                generated_so_far += accepted + int(append_replacement)
                committed_tokens = prompt_tokens + generated_so_far
                events.append(
                    {
                        "event": "step",
                        "algorithm": "speculative",
                        "session_id": row["session_id"],
                        "decoding_mode": row["decoding_mode"],
                        "prompt_id": row["prompt_id"],
                        "prompt_sha256": row["prompt_sha256"],
                        "prompt_set_sha256": row["prompt_set_sha256"],
                        "gamma": gamma,
                        "run": int(row["run"]),
                        "measurement_order": int(row["measurement_order"]),
                        "drafter_freq_hz": int(row["drafter_freq_hz"]),
                        "verifier_clock_mhz": int(row["verifier_clock_mhz"]),
                        "step": step,
                        "base_committed_tokens": base_committed_tokens,
                        "draft_tokens": list(range(draft_count)),
                        "accepted_tokens": accepted,
                        "accepted_token_ids": list(range(accepted)),
                        "replacement_token": 999 if append_replacement else None,
                        "appended_replacement": append_replacement,
                        "generated_tokens_so_far": generated_so_far,
                        "committed_tokens_after_step": committed_tokens,
                        "verifier_committed_tokens": committed_tokens,
                        "drafter_committed_tokens": committed_tokens,
                    }
                )
            assert draft_left == 0
            assert accepted_left == 0
            assert replacement_left == 0
            assert generated_so_far == int(row["generated_tokens"])
            events.append(
                {
                    "event": "speculative_run",
                    "algorithm": "speculative",
                    "session_id": row["session_id"],
                    "decoding_mode": row["decoding_mode"],
                    "prompt_id": row["prompt_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "prompt_set_sha256": row["prompt_set_sha256"],
                    "gamma": int(row["gamma"]),
                    "run": int(row["run"]),
                    "measurement_order": int(row["measurement_order"]),
                    "drafter_freq_hz": int(row["drafter_freq_hz"]),
                    "verifier_clock_mhz": int(row["verifier_clock_mhz"]),
                    "generated_tokens": int(row["generated_tokens"]),
                    "stop_reason": row["stop_reason"],
                    "output_token_sha256": row["output_token_sha256"],
                    "steps": int(row["steps"]),
                    "draft_tokens": int(row["draft_tokens"]),
                    "accepted_draft_tokens": int(row["accepted_draft_tokens"]),
                    "replacement_tokens": int(row["replacement_tokens"]),
                }
            )
        return events

    trace_events = build_trace_events()
    report = experiment_report.build_report(plans=plans, raw_rows=rows)
    assert report["ok"] is True
    assert report["plan_design"]["ok"] is True
    assert report["doctor_design"]["ok"] is True
    assert report["plan_audit"]["ok"] is True
    assert report["plan_audit"]["plan_audit_reports"] == 0
    assert report["k8s_manifest_audit"]["ok"] is True
    assert report["k8s_manifest_audit"]["reports"] == 0
    assert report["doctor_design"]["doctor_reports"] == 0
    assert report["plan_design"]["missing_baseline_conditions"] == 0
    assert report["plan_design"]["spec_gamma_values"] == ["1", "2"]
    assert report["plan_design"]["has_gamma_one_baseline"] is True
    assert report["plan_design"]["missing_tokenizer_plans"] == 0
    assert report["plan_design"]["missing_schema_version_plans"] == 0
    assert report["plan_design"]["missing_prompt_metadata_plans"] == []
    assert report["plan_design"]["duplicate_prompt_hash_plans"] == []
    assert report["plan_design"]["insufficient_unique_prompt_plans"] == []
    assert report["plan_design"]["missing_warmup_plans"] == []
    assert report["plan_design"]["nonpositive_warmup_plans"] == []
    assert report["plan_design"]["baseline_run_mismatches"] == []
    assert report["plan_design"]["baseline_warmup_mismatches"] == []
    assert report["plan_design"]["incomplete_gamma_groups"] == []
    assert report["plan_design"]["nonuniform_gamma_run_groups"] == []
    assert report["energy_design"]["ok"] is True
    assert report["energy_design"]["idle_baseline_policies"] == ["run"]
    assert report["energy_design"]["requires_run_idle_policy"] is True
    assert report["plan_integrity"]["ok"] is True
    assert report["plan_integrity"]["checked_sessions"] == 6
    assert report["plan_integrity"]["missing_plan_design_hashes"] == 0
    assert report["plan_integrity"]["mismatched_result_plan_design_hash_sessions"] == 0
    assert report["plan_integrity"]["mismatched_result_plan_hash_sessions"] == 0
    assert report["schema_contract"]["ok"] is True
    assert report["schema_contract"]["result_schema_versions"] == [
        spec_driver.RESULT_SCHEMA_VERSION
    ]
    assert report["schema_contract"]["driver_result_schema_mismatch_sessions"] == 0
    assert report["schema_contract"]["role_spec_rpc_schema_mismatch_sessions"] == 0
    assert report["frequency_consistency"]["ok"] is True
    assert report["measurement_setup"]["ok"] is True
    assert report["measurement_setup"]["power_intervals_by_role"] == {
        "drafter": ["0.010000"],
        "verifier": ["0.010000"],
    }
    assert report["measurement_setup"]["primary_power_rails_by_role"] == {
        "drafter": ["tot_power"],
        "verifier": ["verifier_gpu_power"],
    }
    assert report["measurement_setup"]["idle_baseline_policies"] == ["run"]
    assert report["provenance"]["ok"] is True
    assert report["provenance"]["commits_by_role"] == {
        "drafter": ["commit-a"],
        "driver": ["commit-a"],
        "verifier": ["commit-a"],
    }
    assert report["model_setup"]["ok"] is True
    assert report["model_setup"]["same_model_sessions"] == 0
    assert report["model_setup"]["missing_model_size_sessions"] == 0
    assert report["model_setup"]["non_smaller_drafter_sessions"] == 0
    assert report["input_consistency"]["ok"] is True
    assert report["input_consistency"]["tokenization_mismatch_groups"] == 0
    assert report["communication"]["ok"] is True
    assert report["trace_consistency"]["ok"] is True
    assert report["trace_consistency"]["trace_events"] == 0
    assert report["network_probe"]["ok"] is True
    assert report["network_probe"]["network_reports"] == 0
    assert report["timing"]["ok"] is True
    assert report["timing"]["throughput_mismatch_sessions"] == 0
    assert report["timing"]["latency_mismatch_sessions"] == 0
    assert report["accounting"]["ok"] is True
    assert report["accounting"]["token_error_sessions"] == 0
    assert report["accounting"]["energy_mismatch_sessions"] == 0
    assert report["energy_signal"]["ok"] is True
    assert report["energy_signal"]["nonpositive_total_energy_sessions"] == 0
    assert report["energy_signal"]["nonpositive_idle_power_sessions"] == 0
    assert report["runtime_status"]["ok"] is True
    assert report["token_compatibility"]["ok"] is True
    assert report["token_compatibility"]["missing_tokenizer_metadata_sessions"] == 0
    assert (
        report["token_compatibility"]["tokenizer_special_token_mismatch_sessions"]
        == 0
    )
    assert report["token_compatibility"]["mixed_tokenizer_metadata_keys"] == []
    assert report["gamma_design"]["ok"] is True
    assert report["gamma_design"]["multi_gamma_configs"] == 1
    assert report["gamma_design"]["min_gammas"] == 2
    assert report["gamma_design"]["min_gamma_configs"] == 1
    assert report["gamma_design"]["ready_configs"] == 1
    assert report["gamma_design"]["non_gamma_one_baseline_rows"] == 0
    assert report["gamma_statistics"]["ok"] is True
    assert report["gamma_statistics"]["checked_nonbaseline_rows"] == 1
    assert report["gamma_statistics"]["missing_stat_rows"] == 0
    assert report["gamma_statistics"]["invalid_stat_rows"] == 0
    assert report["gamma_trend"]["ok"] is True
    assert report["gamma_trend"]["valid_trend_groups"] == 1
    assert report["gamma_trend"]["missing_trend_rows"] == 0
    assert report["gamma_trend"]["invalid_trend_rows"] == 0
    assert report["measurement_stability"]["ok"] is True
    assert report["measurement_stability"]["checked_energy_rows"] == 2
    assert report["measurement_stability"]["checked_latency_rows"] == 3
    assert report["validation"]["ok"] is True
    assert report["validation"]["measurement_schedule_mismatches"] == []
    assert report["system_boundary"]["ok"] is True
    assert report["system_boundary"]["claim_ready"] is True
    assert report["system_boundary"]["paired_baseline_system_boundaries"] == [
        "two_device_idle_drafter"
    ]
    assert report["system_boundary"]["wrong_baseline_boundary_pairs"] == 0
    assert report["summary_rows"] == 3
    assert report["gamma_effect_rows"] == 2
    assert report["paired_summary_rows"] == 2
    assert report["paired_prompt_rows"] == 4
    assert report["unpaired_prompt_rows"] == 0
    assert report["output_equivalence"]["ok"] is True
    assert report["output_equivalence"]["checked_prompt_pairs"] == 4
    assert report["output_equivalence"]["unstable_prompt_pairs"] == 0
    assert report["best_paired_config"]["mean_energy_savings_pct_vs_baseline"] == "50.000000"
    assert report["optimization"]["ok"] is True
    assert report["optimization"]["best_joint_config"]["gamma"] == "2"
    assert report["optimization"]["best_gamma_one_config"]["gamma"] == "1"
    assert (
        report["optimization"]["energy_savings_pct_vs_best_gamma_one"]
        == "20.000000"
    )
    assert report["optimization"]["throughput_ratio_vs_best_gamma_one"] == "1.250000"
    assert report["optimization"]["latency_ratio_vs_best_gamma_one"] == "0.800000"
    assert report["system_optimization"]["ok"] is True
    assert (
        report["system_optimization"]["energy_key"]
        == "mean_active_system_energy_mj_per_token"
    )
    assert report["system_optimization"]["best_joint_config"]["gamma"] == "2"
    assert report["system_optimization"]["best_gamma_one_config"]["gamma"] == "1"
    assert (
        report["system_optimization"]["energy_savings_pct_vs_best_gamma_one"]
        == "33.333333"
    )
    assert report["gamma_policy"]["ok"] is True
    assert report["gamma_policy"]["policy_rows"] == 2
    assert report["gamma_policy"]["uses_gamma_dependent_verifier_clock"] is False
    assert report["interaction"]["ok"] is True
    assert report["interaction"]["eligible"] is False
    assert report["interaction"]["required"] is False
    assert report["claim_readiness"]["ok"] is False
    assert "drafter_gamma_energy" in report["claim_readiness"]["ready_claims"]
    assert (
        "system_energy_vs_verifier_baseline"
        in report["claim_readiness"]["ready_claims"]
    )
    assert (
        "adaptive_gamma_frequency_policy"
        in report["claim_readiness"]["ready_claims"]
    )
    assert (
        "gamma_frequency_interaction"
        in report["claim_readiness"]["blocked_claims"]
    )
    assert (
        "interaction_not_eligible"
        in report["claim_readiness"]["claims"]["gamma_frequency_interaction"][
            "blockers"
        ]
    )
    generic_boundary_report = experiment_report.build_report(
        plans=plans,
        raw_rows=[
            {
                **row,
                "system_boundary": (
                    "verifier_only"
                    if row["algorithm"] == "verifier_only"
                    else row["system_boundary"]
                ),
            }
            for row in rows
        ],
        require_two_device_boundary=True,
    )
    assert generic_boundary_report["ok"] is False
    assert (
        "paired_baseline_boundary_not_two_device_idle_drafter"
        in generic_boundary_report["system_boundary"]["errors"]
    )
    assert (
        "two_device_system_boundary_not_ready"
        in generic_boundary_report["claim_readiness"]["claims"][
            "system_energy_vs_verifier_baseline"
        ]["blockers"]
    )
    require_claim_readiness_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        require_claim_readiness=True,
    )
    assert require_claim_readiness_report["ok"] is False
    assert require_claim_readiness_report["claim_readiness"]["ok"] is False

    plan_audit_payload = plan_audit.build_audit(
        plans=plans,
        plan_paths=["spec_plan.json", "verifier_baseline_plan.json"],
        min_runs=1,
        min_prompts=2,
        min_gammas=2,
        summary_energy_key=experiment_report.DEFAULT_SUMMARY_ENERGY_KEY,
        paired_energy_key=experiment_report.DEFAULT_PAIRED_ENERGY_KEY,
    )
    required_plan_audit_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        plan_audit_reports=[plan_audit_payload],
        require_plan_audit=True,
    )
    assert required_plan_audit_report["ok"] is True
    assert required_plan_audit_report["plan_audit"]["plan_audit_reports"] == 1

    mismatched_plan_audit_payload = json.loads(json.dumps(plan_audit_payload))
    mismatched_plan_audit_payload["plan_integrity"][
        "plan_design_hashes_by_algorithm"
    ]["speculative"] = "wrong-design-hash"
    mismatched_plan_audit_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        plan_audit_reports=[mismatched_plan_audit_payload],
        require_plan_audit=True,
    )
    assert mismatched_plan_audit_report["ok"] is False
    assert (
        "plan_audit_design_hash_mismatch"
        in mismatched_plan_audit_report["plan_audit"]["errors"]
    )

    missing_plan_audit_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        require_plan_audit=True,
    )
    assert missing_plan_audit_report["ok"] is False
    assert (
        "missing_required_plan_audit"
        in missing_plan_audit_report["plan_audit"]["errors"]
    )

    k8s_manifest_path = Path(__file__).resolve().parents[2] / "k8s/spec-decoding.yaml"
    k8s_audit_payload = k8s_manifest_audit.build_audit(
        k8s_manifest_path.read_text(encoding="utf-8")
    )
    required_k8s_audit_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        k8s_manifest_audit_reports=[k8s_audit_payload],
        require_k8s_manifest_audit=True,
    )
    assert required_k8s_audit_report["ok"] is True
    assert required_k8s_audit_report["k8s_manifest_audit"]["reports"] == 1

    missing_k8s_audit_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        require_k8s_manifest_audit=True,
    )
    assert missing_k8s_audit_report["ok"] is False
    assert (
        "missing_required_k8s_manifest_audit"
        in missing_k8s_audit_report["k8s_manifest_audit"]["errors"]
    )

    failed_k8s_audit_payload = json.loads(json.dumps(k8s_audit_payload))
    failed_k8s_audit_payload["ok"] = False
    failed_k8s_audit_payload["errors"] = ["spec_plan_measured_design_mismatch"]
    failed_k8s_audit_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        k8s_manifest_audit_reports=[failed_k8s_audit_payload],
        require_k8s_manifest_audit=True,
    )
    assert failed_k8s_audit_report["ok"] is False
    assert (
        "k8s_manifest_audit_not_ok"
        in failed_k8s_audit_report["k8s_manifest_audit"]["errors"]
    )
    assert (
        "k8s_manifest_audit_design_mismatch"
        in failed_k8s_audit_report["k8s_manifest_audit"]["errors"]
    )

    failed_placement_payload = json.loads(json.dumps(k8s_audit_payload))
    failed_placement_payload["ok"] = False
    failed_placement_payload["placement"]["ok"] = False
    failed_placement_payload["placement"]["errors"] = [
        "placement_missing_required_token"
    ]
    failed_placement_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        k8s_manifest_audit_reports=[failed_placement_payload],
        require_k8s_manifest_audit=True,
    )
    assert failed_placement_report["ok"] is False
    assert (
        "k8s_manifest_audit_placement_mismatch"
        in failed_placement_report["k8s_manifest_audit"]["errors"]
    )

    traced_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        trace_events=trace_events,
        require_trace=True,
    )
    assert traced_report["ok"] is True
    assert traced_report["trace_consistency"]["checked_sessions"] == 6
    assert traced_report["trace_consistency"]["traced_sessions"] == 6
    assert traced_report["trace_consistency"]["missing_trace_sessions"] == 0
    assert traced_report["trace_consistency"]["missing_trace_summary_sessions"] == 0
    assert traced_report["trace_consistency"]["trace_summary_mismatch_sessions"] == 0

    missing_trace_events = [
        event
        for event in trace_events
        if event.get("session_id", "") != "s0"
    ]
    missing_trace_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        trace_events=missing_trace_events,
        require_trace=True,
    )
    assert missing_trace_report["ok"] is False
    assert (
        "missing_trace_sessions"
        in missing_trace_report["trace_consistency"]["errors"]
    )

    missing_summary_events = [
        event
        for event in trace_events
        if not (
            event.get("event") == "speculative_run"
            and event.get("session_id") == "s0"
        )
    ]
    missing_summary_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        trace_events=missing_summary_events,
        require_trace=True,
    )
    assert missing_summary_report["ok"] is False
    assert (
        "missing_trace_summary_sessions"
        in missing_summary_report["trace_consistency"]["errors"]
    )

    mismatched_trace_events = json.loads(json.dumps(trace_events))
    for event in mismatched_trace_events:
        if event.get("event") == "step" and event.get("session_id") == "s0":
            event["accepted_tokens"] = 0
            break
    mismatched_trace_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        trace_events=mismatched_trace_events,
        require_trace=True,
    )
    assert mismatched_trace_report["ok"] is False
    assert (
        "trace_session_mismatch"
        in mismatched_trace_report["trace_consistency"]["errors"]
    )

    mismatched_summary_events = json.loads(json.dumps(trace_events))
    for event in mismatched_summary_events:
        if event.get("event") == "speculative_run" and event.get("session_id") == "s0":
            event["output_token_sha256"] = spec_driver.token_ids_sha256([999])
            break
    mismatched_summary_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        trace_events=mismatched_summary_events,
        require_trace=True,
    )
    assert mismatched_summary_report["ok"] is False
    assert (
        "trace_summary_mismatch"
        in mismatched_summary_report["trace_consistency"]["errors"]
    )

    strict_gamma_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        min_gammas=3,
    )
    assert strict_gamma_report["ok"] is False
    assert "insufficient_gamma_values" in strict_gamma_report["gamma_design"]["errors"]
    assert "no_gamma_trend_eligible_config" in strict_gamma_report["gamma_trend"]["errors"]

    missing_trend_rows = [dict(row) for row in report["_gamma_effect_rows"]]
    for row in missing_trend_rows:
        row.pop("log2_gamma_slope_drafter_active_energy_mj_per_token", None)
    missing_trend_report = experiment_report.gamma_trend_report(
        missing_trend_rows,
        min_gammas=2,
        require_active_trend=True,
    )
    assert missing_trend_report["ok"] is False
    assert "missing_gamma_trend_statistics" in missing_trend_report["errors"]

    strict_stability_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        max_energy_cv=0.01,
        max_latency_cv=0.01,
    )
    assert strict_stability_report["ok"] is False
    assert "energy_cv_exceeds_limit" in strict_stability_report[
        "measurement_stability"
    ]["errors"]
    assert "latency_cv_exceeds_limit" in strict_stability_report[
        "measurement_stability"
    ]["errors"]

    mismatched_plan_hash_rows = [dict(row) for row in rows]
    mismatched_plan_hash_rows[0]["plan_sha256"] = "bad-plan-hash"
    mismatched_plan_report = experiment_report.build_report(
        plans=plans,
        raw_rows=mismatched_plan_hash_rows,
    )
    assert mismatched_plan_report["ok"] is False
    assert "result_plan_hash_mismatch" in mismatched_plan_report[
        "plan_integrity"
    ]["errors"]

    network_payload = {
        "schema_version": network_probe.SCHEMA_VERSION,
        "ok": True,
        "targets": {
            "drafter": {"ok": True, "rtt_ms_mean": "1.0", "rtt_ms_p95": "2.0"},
            "verifier": {"ok": True, "rtt_ms_mean": "1.5", "rtt_ms_p95": "2.5"},
        },
    }
    required_network_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        require_network_probe=True,
    )
    assert required_network_report["ok"] is False
    assert (
        "missing_required_network_probe"
        in required_network_report["network_probe"]["errors"]
    )
    passing_network_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        network_reports=[network_payload],
        require_network_probe=True,
        max_network_p95_rtt_ms=5.0,
    )
    assert passing_network_report["ok"] is True
    slow_network_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        network_reports=[network_payload],
        require_network_probe=True,
        max_network_p95_rtt_ms=2.0,
    )
    assert slow_network_report["ok"] is False
    assert (
        "network_p95_rtt_exceeds_limit"
        in slow_network_report["network_probe"]["errors"]
    )

    missing_gamma_stat_rows = [dict(row) for row in report["_gamma_effect_rows"]]
    missing_gamma_stat_rows[1].pop(
        "paired_bootstrap_ci95_low_drafter_total_energy_change_pct_vs_baseline_gamma",
        None,
    )
    missing_gamma_stats = experiment_report.gamma_statistics_report(
        missing_gamma_stat_rows,
        require_active_stats=True,
    )
    assert missing_gamma_stats["ok"] is False
    assert "missing_gamma_statistics" in missing_gamma_stats["errors"]

    invalid_gamma_stat_rows = [dict(row) for row in report["_gamma_effect_rows"]]
    invalid_gamma_stat_rows[1][
        "paired_sign_test_p_value_drafter_total_energy_change_pct_vs_baseline_gamma"
    ] = "2.0"
    invalid_gamma_stats = experiment_report.gamma_statistics_report(
        invalid_gamma_stat_rows,
        require_active_stats=True,
    )
    assert invalid_gamma_stats["ok"] is False
    assert "invalid_gamma_statistics" in invalid_gamma_stats["errors"]

    vocab_mismatch_rows = [
        {
            **row,
            "drafter_model_vocab_size": "32000"
            if row.get("algorithm") == "speculative"
            else "",
            "verifier_model_vocab_size": "32001"
            if row.get("algorithm") == "speculative"
            else "32001",
        }
        for row in rows
    ]
    vocab_report = experiment_report.build_report(
        plans=plans,
        raw_rows=vocab_mismatch_rows,
    )
    assert vocab_report["ok"] is False
    assert (
        "drafter_verifier_vocab_size_mismatch"
        in vocab_report["token_compatibility"]["errors"]
    )

    missing_model_vocab_rows = [dict(row) for row in rows]
    missing_model_vocab_rows[2].pop("drafter_model_vocab_size", None)
    missing_model_vocab_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_model_vocab_rows,
    )
    assert missing_model_vocab_report["ok"] is False
    assert (
        "missing_model_vocab_metadata"
        in missing_model_vocab_report["token_compatibility"]["errors"]
    )

    tokenizer_mismatch_rows = [
        {**row, "tokenizer_vocab_size": "32002"}
        for row in rows
    ]
    tokenizer_mismatch_report = experiment_report.build_report(
        plans=plans,
        raw_rows=tokenizer_mismatch_rows,
    )
    assert tokenizer_mismatch_report["ok"] is False
    assert (
        "tokenizer_model_vocab_size_mismatch"
        in tokenizer_mismatch_report["token_compatibility"]["errors"]
    )

    special_token_mismatch_rows = [
        {**row, "verifier_model_eos_token_id": "3"}
        for row in rows
    ]
    special_token_mismatch_report = experiment_report.build_report(
        plans=plans,
        raw_rows=special_token_mismatch_rows,
    )
    assert special_token_mismatch_report["ok"] is False
    assert (
        "tokenizer_model_special_token_mismatch"
        in special_token_mismatch_report["token_compatibility"]["errors"]
    )

    mixed_tokenizer_rows = [
        {
            **row,
            "tokenizer_class": "OtherTokenizer"
            if row["session_id"] == "s1"
            else row["tokenizer_class"],
        }
        for row in rows
    ]
    mixed_tokenizer_report = experiment_report.build_report(
        plans=plans,
        raw_rows=mixed_tokenizer_rows,
    )
    assert mixed_tokenizer_report["ok"] is False
    assert (
        "mixed_tokenizer_metadata"
        in mixed_tokenizer_report["token_compatibility"]["errors"]
    )

    missing_tokenizer_rows = [dict(row) for row in rows]
    missing_tokenizer_rows[0].pop("tokenizer_vocab_size", None)
    missing_tokenizer_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_tokenizer_rows,
    )
    assert missing_tokenizer_report["ok"] is False
    assert (
        "missing_tokenizer_vocab_metadata"
        in missing_tokenizer_report["token_compatibility"]["errors"]
    )

    missing_eos_rows = [dict(row) for row in rows]
    missing_eos_rows[0].pop("tokenizer_eos_token_id", None)
    missing_eos_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_eos_rows,
    )
    assert missing_eos_report["ok"] is False
    assert (
        "missing_tokenizer_eos_metadata"
        in missing_eos_report["token_compatibility"]["errors"]
    )

    missing_schema_rows = [dict(row) for row in rows]
    missing_schema_rows[0].pop("result_schema_version", None)
    schema_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_schema_rows,
    )
    assert schema_report["ok"] is False
    assert (
        "missing_result_schema_version"
        in schema_report["schema_contract"]["errors"]
    )

    driver_schema_rows = [dict(row) for row in rows]
    driver_schema_rows[0]["driver_result_schema_version"] = "old-result-schema"
    driver_schema_report = experiment_report.build_report(
        plans=plans,
        raw_rows=driver_schema_rows,
    )
    assert driver_schema_report["ok"] is False
    assert (
        "driver_result_schema_version_mismatch"
        in driver_schema_report["schema_contract"]["errors"]
    )

    missing_rpc_rows = [dict(row) for row in rows]
    missing_rpc_rows[2].pop("drafter_spec_rpc_schema_version", None)
    missing_rpc_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_rpc_rows,
    )
    assert missing_rpc_report["ok"] is False
    assert (
        "missing_role_spec_rpc_schema_version"
        in missing_rpc_report["schema_contract"]["errors"]
    )

    mismatched_rpc_rows = [dict(row) for row in rows]
    mismatched_rpc_rows[2]["verifier_spec_rpc_schema_version"] = "old-rpc-schema"
    mismatched_rpc_report = experiment_report.build_report(
        plans=plans,
        raw_rows=mismatched_rpc_rows,
    )
    assert mismatched_rpc_report["ok"] is False
    assert (
        "role_spec_rpc_schema_version_mismatch"
        in mismatched_rpc_report["schema_contract"]["errors"]
    )

    mixed_algorithm_rows = [dict(row) for row in rows]
    mixed_algorithm_rows[-1]["algorithm_version"] = "speculative-test-old"
    mixed_algorithm_report = experiment_report.build_report(
        plans=plans,
        raw_rows=mixed_algorithm_rows,
    )
    assert mixed_algorithm_report["ok"] is False
    assert (
        "mixed_algorithm_versions"
        in mixed_algorithm_report["schema_contract"]["errors"]
    )

    frequency_mismatch_rows = [dict(row) for row in rows]
    frequency_mismatch_rows[2]["drafter_jetson_gpu_freq_hz"] = "999"
    frequency_mismatch_report = experiment_report.build_report(
        plans=plans,
        raw_rows=frequency_mismatch_rows,
    )
    assert frequency_mismatch_report["ok"] is False
    assert (
        "reported_frequency_mismatch"
        in frequency_mismatch_report["frequency_consistency"]["errors"]
    )

    frequency_lock_rows = [dict(row) for row in rows]
    frequency_lock_rows[2]["drafter_frequency_lock_ok"] = "0"
    frequency_lock_report = experiment_report.build_report(
        plans=plans,
        raw_rows=frequency_lock_rows,
    )
    assert frequency_lock_report["ok"] is False
    assert (
        "frequency_lock_failed"
        in frequency_lock_report["frequency_consistency"]["errors"]
    )

    missing_power_interval_rows = [dict(row) for row in rows]
    missing_power_interval_rows[2].pop("drafter_power_interval_s", None)
    missing_power_interval_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_power_interval_rows,
    )
    assert missing_power_interval_report["ok"] is False
    assert (
        "missing_power_interval_metadata"
        in missing_power_interval_report["measurement_setup"]["errors"]
    )

    missing_idle_policy_rows = [dict(row) for row in rows]
    missing_idle_policy_rows[2].pop("idle_baseline_policy", None)
    missing_idle_policy_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_idle_policy_rows,
    )
    assert missing_idle_policy_report["ok"] is False
    assert (
        "missing_idle_baseline_policy"
        in missing_idle_policy_report["measurement_setup"]["errors"]
    )

    missing_primary_rail_rows = [dict(row) for row in rows]
    missing_primary_rail_rows[2].pop("drafter_primary_power_rail", None)
    missing_primary_rail_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_primary_rail_rows,
    )
    assert missing_primary_rail_report["ok"] is False
    assert (
        "missing_primary_power_rail_metadata"
        in missing_primary_rail_report["measurement_setup"]["errors"]
    )

    mismatched_primary_rail_rows = [dict(row) for row in rows]
    mismatched_primary_rail_rows[2]["drafter_primary_power_rail"] = "gpu_power"
    mismatched_primary_rail_report = experiment_report.build_report(
        plans=plans,
        raw_rows=mismatched_primary_rail_rows,
    )
    assert mismatched_primary_rail_report["ok"] is False
    assert (
        "selected_primary_power_rail_mismatch"
        in mismatched_primary_rail_report["measurement_setup"]["errors"]
    )

    same_host_rows = [dict(row) for row in rows]
    same_host_rows[2]["verifier_hostname"] = "jetson-node"
    same_host_report = experiment_report.build_report(
        plans=plans,
        raw_rows=same_host_rows,
    )
    assert same_host_report["ok"] is False
    assert "drafter_verifier_same_host" in same_host_report["measurement_setup"]["errors"]

    missing_host_rows = [dict(row) for row in rows]
    missing_host_rows[2].pop("drafter_hostname", None)
    missing_host_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_host_rows,
    )
    assert missing_host_report["ok"] is False
    assert (
        "missing_drafter_verifier_host_identity"
        in missing_host_report["measurement_setup"]["errors"]
    )

    missing_provenance_rows = [dict(row) for row in rows]
    missing_provenance_rows[2].pop("driver_xronos_git_commit", None)
    missing_provenance_rows[2].pop("driver_git_commit", None)
    missing_provenance_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_provenance_rows,
    )
    assert missing_provenance_report["ok"] is False
    assert (
        "missing_git_commit_metadata"
        in missing_provenance_report["provenance"]["errors"]
    )

    mismatched_provenance_rows = [dict(row) for row in rows]
    mismatched_provenance_rows[2]["verifier_xronos_git_commit"] = "commit-b"
    mismatched_provenance_report = experiment_report.build_report(
        plans=plans,
        raw_rows=mismatched_provenance_rows,
    )
    assert mismatched_provenance_report["ok"] is False
    assert (
        "role_git_commit_mismatch"
        in mismatched_provenance_report["provenance"]["errors"]
    )

    dirty_driver_rows = [dict(row) for row in rows]
    dirty_driver_rows[2]["driver_git_dirty"] = "1"
    dirty_driver_report = experiment_report.build_report(
        plans=plans,
        raw_rows=dirty_driver_rows,
    )
    assert dirty_driver_report["ok"] is False
    assert "driver_git_dirty" in dirty_driver_report["provenance"]["errors"]

    same_model_rows = [
        {
            **row,
            "drafter_model": row["verifier_model"]
            if row.get("algorithm") == "speculative"
            else row.get("drafter_model", ""),
        }
        for row in rows
    ]
    same_model_report = experiment_report.build_report(
        plans=plans,
        raw_rows=same_model_rows,
    )
    assert same_model_report["ok"] is False
    assert "drafter_verifier_same_model" in same_model_report["model_setup"]["errors"]

    missing_model_rows = [dict(row) for row in rows]
    missing_model_rows[2].pop("drafter_model", None)
    missing_model_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_model_rows,
    )
    assert missing_model_report["ok"] is False
    assert (
        "missing_drafter_or_verifier_model_name"
        in missing_model_report["model_setup"]["errors"]
    )

    missing_model_size_rows = [dict(row) for row in rows]
    missing_model_size_rows[2].pop("drafter_model_parameter_count", None)
    missing_model_size_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_model_size_rows,
    )
    assert missing_model_size_report["ok"] is False
    assert (
        "missing_drafter_or_verifier_model_size"
        in missing_model_size_report["model_setup"]["errors"]
    )

    non_smaller_model_rows = [
        {
            **row,
            "drafter_model_parameter_count": row["verifier_model_parameter_count"]
            if row.get("algorithm") == "speculative"
            else row.get("drafter_model_parameter_count", ""),
        }
        for row in rows
    ]
    non_smaller_model_report = experiment_report.build_report(
        plans=plans,
        raw_rows=non_smaller_model_rows,
    )
    assert non_smaller_model_report["ok"] is False
    assert (
        "drafter_model_must_be_smaller_than_verifier"
        in non_smaller_model_report["model_setup"]["errors"]
    )

    missing_input_rows = [dict(row) for row in rows]
    missing_input_rows[2].pop("prompt_token_sha256", None)
    missing_input_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_input_rows,
    )
    assert missing_input_report["ok"] is False
    assert "missing_input_metadata" in missing_input_report["input_consistency"]["errors"]

    inconsistent_input_rows = [dict(row) for row in rows]
    inconsistent_input_rows[2]["prompt_token_sha256"] = spec_driver.token_ids_sha256(
        [99, 100]
    )
    inconsistent_input_report = experiment_report.build_report(
        plans=plans,
        raw_rows=inconsistent_input_rows,
    )
    assert inconsistent_input_report["ok"] is False
    assert (
        "inconsistent_prompt_tokenization"
        in inconsistent_input_report["input_consistency"]["errors"]
    )

    missing_comm_rows = [dict(row) for row in rows]
    missing_comm_rows[2].pop("rpc_total_bytes", None)
    missing_comm_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_comm_rows,
    )
    assert missing_comm_report["ok"] is False
    assert "missing_communication_metrics" in missing_comm_report["communication"]["errors"]

    bad_timing_rows = [dict(row) for row in rows]
    bad_timing_rows[2]["tokens_per_s"] = "999"
    bad_timing_report = experiment_report.build_report(
        plans=plans,
        raw_rows=bad_timing_rows,
    )
    assert bad_timing_report["ok"] is False
    assert "throughput_latency_mismatch" in bad_timing_report["timing"]["errors"]

    bad_latency_rows = [dict(row) for row in rows]
    bad_latency_rows[2]["estimated_rpc_overhead_ms"] = "999"
    bad_latency_report = experiment_report.build_report(
        plans=plans,
        raw_rows=bad_latency_rows,
    )
    assert bad_latency_report["ok"] is False
    assert "latency_accounting_mismatch" in bad_latency_report["timing"]["errors"]

    bad_token_accounting_rows = [dict(row) for row in rows]
    bad_token_accounting_rows[2]["accepted_draft_tokens"] = "99"
    bad_token_accounting_report = experiment_report.build_report(
        plans=plans,
        raw_rows=bad_token_accounting_rows,
    )
    assert bad_token_accounting_report["ok"] is False
    assert (
        "token_accounting_inconsistent"
        in bad_token_accounting_report["accounting"]["errors"]
    )

    bad_energy_accounting_rows = [dict(row) for row in rows]
    bad_energy_accounting_rows[2]["system_total_energy_mj"] = "99"
    bad_energy_accounting_report = experiment_report.build_report(
        plans=plans,
        raw_rows=bad_energy_accounting_rows,
    )
    assert bad_energy_accounting_report["ok"] is False
    assert (
        "energy_accounting_inconsistent"
        in bad_energy_accounting_report["accounting"]["errors"]
    )

    missing_accounting_rows = [dict(row) for row in rows]
    missing_accounting_rows[2].pop("accepted_draft_tokens", None)
    missing_accounting_report = experiment_report.build_report(
        plans=plans,
        raw_rows=missing_accounting_rows,
    )
    assert missing_accounting_report["ok"] is False
    assert (
        "missing_accounting_fields"
        in missing_accounting_report["accounting"]["errors"]
    )

    zero_signal_rows = [dict(row) for row in rows]
    zero_signal_rows[2]["drafter_draft_total_energy_mj"] = "0"
    zero_signal_rows[2]["drafter_total_energy_mj"] = "2"
    zero_signal_rows[2]["system_total_energy_mj"] = "12.5"
    zero_signal_report = experiment_report.build_report(
        plans=plans,
        raw_rows=zero_signal_rows,
    )
    assert zero_signal_report["ok"] is False
    assert (
        "nonpositive_total_energy_signal"
        in zero_signal_report["energy_signal"]["errors"]
    )

    zero_active_phase_rows = [dict(row) for row in rows]
    zero_active_phase_rows[2]["drafter_draft_active_energy_mj"] = "0"
    zero_active_phase_report = experiment_report.energy_signal_report(
        zero_active_phase_rows
    )
    assert zero_active_phase_report["ok"] is False
    assert (
        "nonpositive_active_energy_signal"
        in zero_active_phase_report["errors"]
    )

    zero_idle_rows = [dict(row) for row in rows]
    zero_idle_rows[2]["drafter_idle_power_mw"] = "0"
    zero_idle_report = experiment_report.build_report(
        plans=plans,
        raw_rows=zero_idle_rows,
    )
    assert zero_idle_report["ok"] is False
    assert (
        "nonpositive_idle_power_signal"
        in zero_idle_report["energy_signal"]["errors"]
    )

    thermal_rows = [
        {
            **row,
            "drafter_runtime_temp_c": "71.5"
            if row.get("algorithm") == "speculative"
            else "",
            "verifier_runtime_temp_c": "64.0",
            "verifier_nvidia_throttle_active": "Not Active",
        }
        for row in rows
    ]
    thermal_report = experiment_report.build_report(
        plans=plans,
        raw_rows=thermal_rows,
        max_runtime_temp_c=70.0,
    )
    assert thermal_report["ok"] is False
    assert (
        "runtime_temperature_exceeds_limit"
        in thermal_report["runtime_status"]["errors"]
    )
    throttle_rows = [
        {
            **row,
            "verifier_nvidia_throttle_active": "Active"
            if row.get("algorithm") == "speculative"
            else "",
        }
        for row in rows
    ]
    throttle_report = experiment_report.build_report(
        plans=plans,
        raw_rows=throttle_rows,
        fail_on_throttle=True,
    )
    assert throttle_report["ok"] is False
    assert "runtime_throttle_active" in throttle_report["runtime_status"]["errors"]

    bad_rows = [dict(row) for row in rows]
    bad_rows[-1]["output_token_sha256"] = spec_driver.token_ids_sha256([999])
    bad_report = experiment_report.build_report(plans=plans, raw_rows=bad_rows)
    assert bad_report["ok"] is False
    assert bad_report["output_equivalence"]["mismatched_prompt_pairs"] == 1

    imbalanced_plans = [
        plans[0],
        {
            **plans[1],
            "combinations": [
                combo
                for combo in plans[1]["combinations"]
                if not (
                    combo["prompt_id"] == "p1"
                    and combo["gamma"] == 2
                )
            ],
        },
    ]
    imbalanced_report = experiment_report.build_report(
        plans=imbalanced_plans,
        raw_rows=rows,
    )
    assert imbalanced_report["ok"] is False
    assert "incomplete_gamma_factorial" in imbalanced_report["plan_design"]["errors"]

    nonuniform_plans = [
        plans[0],
        {
            **plans[1],
            "combinations": [
                {
                    **combo,
                    "measured_runs": 2
                    if combo["prompt_id"] == "p0" and combo["gamma"] == 1
                    else combo["measured_runs"],
                }
                for combo in plans[1]["combinations"]
            ],
        },
    ]
    nonuniform_report = experiment_report.build_report(
        plans=nonuniform_plans,
        raw_rows=rows,
    )
    assert nonuniform_report["ok"] is False
    assert "nonuniform_gamma_measured_runs" in nonuniform_report["plan_design"]["errors"]

    no_gamma_one_plans = [
        plans[0],
        {
            **plans[1],
            "combinations": [
                {
                    **combo,
                    "gamma": 4 if combo["gamma"] == 1 else combo["gamma"],
                }
                for combo in plans[1]["combinations"]
            ],
            "gammas": [2, 4],
        },
    ]
    no_gamma_one_rows = [
        {
            **row,
            "gamma": "4"
            if row.get("algorithm") == "speculative" and row.get("gamma") == "1"
            else row.get("gamma", ""),
        }
        for row in rows
    ]
    no_gamma_one_report = experiment_report.build_report(
        plans=no_gamma_one_plans,
        raw_rows=no_gamma_one_rows,
    )
    assert no_gamma_one_report["ok"] is False
    assert (
        "gamma_sweep_requires_gamma_one_baseline"
        in no_gamma_one_report["plan_design"]["errors"]
    )
    assert (
        "gamma_effect_baseline_must_be_one"
        in no_gamma_one_report["gamma_design"]["errors"]
    )

    unrandomized_report = experiment_report.build_report(
        plans=[{**plans[0], "shuffle_runs": False}, plans[1]],
        raw_rows=rows,
    )
    assert unrandomized_report["ok"] is False
    assert (
        "measurement_runs_must_be_randomized"
        in unrandomized_report["plan_design"]["errors"]
    )

    blocked_schedule_report = experiment_report.build_report(
        plans=[
            plans[0],
            {
                **plans[1],
                "measurement_schedule": [
                    {"order": 1, "condition_order": 0, "run": 1},
                    {"order": 2, "condition_order": 1, "run": 1},
                    {"order": 3, "condition_order": 2, "run": 1},
                    {"order": 4, "condition_order": 3, "run": 1},
                ],
            },
        ],
        raw_rows=rows,
    )
    assert blocked_schedule_report["ok"] is False
    assert "blocked_measurement_schedule" in blocked_schedule_report["plan_design"]["errors"]

    missing_warmup_schedule_plan = dict(plans[1])
    missing_warmup_schedule_plan.pop("warmup_schedule", None)
    missing_warmup_schedule_report = experiment_report.build_report(
        plans=[plans[0], missing_warmup_schedule_plan],
        raw_rows=rows,
    )
    assert missing_warmup_schedule_report["ok"] is False
    assert (
        "missing_warmup_schedule"
        in missing_warmup_schedule_report["plan_design"]["errors"]
    )

    blocked_warmup_schedule_report = experiment_report.build_report(
        plans=[
            plans[0],
            {
                **plans[1],
                "warmup_schedule": [
                    {"order": 1, "condition_order": 0, "warmup": 1},
                    {"order": 2, "condition_order": 1, "warmup": 1},
                    {"order": 3, "condition_order": 2, "warmup": 1},
                    {"order": 4, "condition_order": 3, "warmup": 1},
                ],
            },
        ],
        raw_rows=rows,
    )
    assert blocked_warmup_schedule_report["ok"] is False
    assert (
        "blocked_warmup_schedule"
        in blocked_warmup_schedule_report["plan_design"]["errors"]
    )

    missing_seed_plan = dict(plans[1])
    missing_seed_plan.pop("seed", None)
    missing_seed_report = experiment_report.build_report(
        plans=[plans[0], missing_seed_plan],
        raw_rows=rows,
    )
    assert missing_seed_report["ok"] is False
    assert "missing_randomization_seed" in missing_seed_report["plan_design"]["errors"]

    missing_warmup_plan = dict(plans[1])
    missing_warmup_plan.pop("warmup_runs", None)
    missing_warmup_report = experiment_report.build_report(
        plans=[plans[0], missing_warmup_plan],
        raw_rows=rows,
    )
    assert missing_warmup_report["ok"] is False
    assert "missing_warmup_runs" in missing_warmup_report["plan_design"]["errors"]

    zero_warmup_report = experiment_report.build_report(
        plans=[plans[0], {**plans[1], "warmup_runs": 0}],
        raw_rows=rows,
    )
    assert zero_warmup_report["ok"] is False
    assert (
        "measured_plan_requires_warmup"
        in zero_warmup_report["plan_design"]["errors"]
    )

    mismatched_baseline_runs = experiment_report.build_report(
        plans=[
            {
                **plans[0],
                "combinations": [
                    {**combo, "measured_runs": 2}
                    for combo in plans[0]["combinations"]
                ],
            },
            plans[1],
        ],
        raw_rows=rows,
    )
    assert mismatched_baseline_runs["ok"] is False
    assert (
        "baseline_measured_runs_mismatch"
        in mismatched_baseline_runs["plan_design"]["errors"]
    )

    mismatched_baseline_warmups = experiment_report.build_report(
        plans=[{**plans[0], "warmup_runs": 2}, plans[1]],
        raw_rows=rows,
    )
    assert mismatched_baseline_warmups["ok"] is False
    assert (
        "baseline_warmup_runs_mismatch"
        in mismatched_baseline_warmups["plan_design"]["errors"]
    )

    duplicate_prompt_plan = {
        **plans[1],
        "prompts": [
            {**plans[1]["prompts"][0]},
            {
                **plans[1]["prompts"][1],
                "prompt_sha256": plans[1]["prompts"][0]["prompt_sha256"],
            },
        ],
    }
    duplicate_prompt_report = experiment_report.build_report(
        plans=[plans[0], duplicate_prompt_plan],
        raw_rows=rows,
    )
    assert duplicate_prompt_report["ok"] is False
    assert (
        "duplicate_plan_prompt_hashes"
        in duplicate_prompt_report["plan_design"]["errors"]
    )

    insufficient_prompt_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        min_prompts=3,
    )
    assert insufficient_prompt_report["ok"] is False
    assert (
        "insufficient_unique_plan_prompts"
        in insufficient_prompt_report["plan_design"]["errors"]
    )

    unchecked_rows = [dict(row) for row in rows]
    for row in unchecked_rows:
        row.pop("output_token_sha256", None)
    unchecked_report = experiment_report.build_report(
        plans=plans,
        raw_rows=unchecked_rows,
    )
    assert unchecked_report["ok"] is False
    assert unchecked_report["output_equivalence"]["unchecked_prompt_pairs"] == 4
    allowed_unchecked_report = experiment_report.build_report(
        plans=plans,
        raw_rows=unchecked_rows,
        allow_unchecked_output=True,
    )
    assert allowed_unchecked_report["ok"] is True

    infeasible_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        min_runs=99,
    )
    assert infeasible_report["ok"] is False
    assert infeasible_report["feasible_rows"] == 0

    partial_plans = [
        {
            **plans[0],
            "combinations": plans[0]["combinations"][:1],
            "measurement_schedule": [
                {"order": 1, "condition_order": 0, "run": 1},
            ],
            "warmup_schedule": [
                {"order": 1, "condition_order": 0, "warmup": 1},
            ],
        },
        plans[1],
    ]
    spec_driver.attach_plan_sha256(partial_plans[0])
    partial_plan_sha_by_algorithm = {
        str(plan["algorithm"]): str(plan["plan_sha256"])
        for plan in partial_plans
    }
    partial_plan_design_sha_by_algorithm = {
        str(plan["algorithm"]): str(plan["plan_design_sha256"])
        for plan in partial_plans
    }
    partial_rows = [dict(row) for row in rows if row["session_id"] != "b1"]
    for row in partial_rows:
        row["plan_sha256"] = partial_plan_sha_by_algorithm[row["algorithm"]]
        row["driver_plan_sha256"] = partial_plan_sha_by_algorithm[row["algorithm"]]
        row["plan_design_sha256"] = partial_plan_design_sha_by_algorithm[row["algorithm"]]
        row["driver_plan_design_sha256"] = partial_plan_design_sha_by_algorithm[
            row["algorithm"]
        ]
        if row["session_id"] == "b0":
            row["measurement_order"] = "1"
    unpaired_report = experiment_report.build_report(
        plans=partial_plans,
        raw_rows=partial_rows,
    )
    assert unpaired_report["ok"] is False
    assert unpaired_report["plan_design"]["ok"] is False
    assert unpaired_report["plan_design"]["missing_baseline_conditions"] == 1
    assert unpaired_report["unpaired_prompt_rows"] == 2
    allowed_unpaired_report = experiment_report.build_report(
        plans=partial_plans,
        raw_rows=partial_rows,
        allow_unpaired=True,
    )
    assert allowed_unpaired_report["ok"] is True
    assert allowed_unpaired_report["plan_design"]["ok"] is True

    no_idle_plans = [{**plan, "idle_baseline_s": 0} for plan in plans]
    no_idle_report = experiment_report.build_report(
        plans=no_idle_plans,
        raw_rows=rows,
    )
    assert no_idle_report["ok"] is False
    assert no_idle_report["energy_design"]["ok"] is False

    condition_idle_policy_report = experiment_report.build_report(
        plans=[
            {**plans[0], "idle_baseline_policy": "condition"},
            {**plans[1], "idle_baseline_policy": "condition"},
        ],
        raw_rows=rows,
    )
    assert condition_idle_policy_report["ok"] is False
    assert (
        "active_energy_metrics_require_run_idle_baseline_policy"
        in condition_idle_policy_report["energy_design"]["errors"]
    )

    required_doctor_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        require_doctor=True,
    )
    assert required_doctor_report["ok"] is False
    assert "missing_drafter_doctor_report" in required_doctor_report["doctor_design"]["errors"]

    passing_doctors = [
        {
            "role": "drafter",
            "ok": True,
            "checks": [{"name": "ina3221", "status": "ok", "message": "", "details": {}}],
        },
        {
            "role": "verifier",
            "ok": True,
            "checks": [{"name": "nvidia_smi", "status": "ok", "message": "", "details": {}}],
        },
    ]
    doctor_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        doctor_reports=passing_doctors,
        require_doctor=True,
    )
    assert doctor_report["ok"] is True
    assert doctor_report["doctor_design"]["roles"] == ["drafter", "verifier"]

    driver_required_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        doctor_reports=passing_doctors,
        require_doctor=True,
        require_driver_doctor=True,
    )
    assert driver_required_report["ok"] is False
    assert (
        "missing_driver_doctor_report"
        in driver_required_report["doctor_design"]["errors"]
    )

    driver_doctor_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        doctor_reports=passing_doctors
        + [
            {
                "role": "driver",
                "ok": True,
                "checks": [
                    {"name": "prompts", "status": "ok", "message": "", "details": {}}
                ],
            }
        ],
        require_doctor=True,
        require_driver_doctor=True,
    )
    assert driver_doctor_report["ok"] is True
    assert driver_doctor_report["doctor_design"]["roles"] == [
        "drafter",
        "driver",
        "verifier",
    ]

    failing_doctor_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        doctor_reports=[
            passing_doctors[0],
            {
                "role": "verifier",
                "ok": False,
                "checks": [
                    {
                        "name": "nvidia_smi",
                        "status": "fail",
                        "message": "nvidia-smi missing",
                        "details": {},
                    }
                ],
            },
        ],
        require_doctor=True,
    )
    assert failing_doctor_report["ok"] is False
    assert "doctor_checks_failed" in failing_doctor_report["doctor_design"]["errors"]
    assert "doctor_report_not_ok" in failing_doctor_report["doctor_design"]["errors"]

    payload_not_ok_report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        doctor_reports=[
            passing_doctors[0],
            {"role": "verifier", "ok": False, "checks": []},
        ],
        require_doctor=True,
    )
    assert payload_not_ok_report["ok"] is False
    assert "doctor_report_not_ok" in payload_not_ok_report["doctor_design"]["errors"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "report"
        experiment_report.write_outputs(report, out_dir)
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "gamma_effect_summary.csv").exists()
        assert (out_dir / "paired_prompt_summary.csv").exists()
        assert (out_dir / "unpaired_prompt_rows.csv").exists()
        assert (out_dir / "pareto_configs.csv").exists()
        assert (out_dir / "gamma_frequency_policy.csv").exists()
        assert (out_dir / "doctor_report.json").exists()
        assert (out_dir / "plan_audit_report.json").exists()
        assert (out_dir / "k8s_manifest_audit_report.json").exists()
        assert (out_dir / "plan_integrity_report.json").exists()
        assert (out_dir / "schema_contract_report.json").exists()
        assert (out_dir / "frequency_consistency_report.json").exists()
        assert (out_dir / "measurement_setup_report.json").exists()
        assert (out_dir / "provenance_report.json").exists()
        assert (out_dir / "model_setup_report.json").exists()
        assert (out_dir / "input_consistency_report.json").exists()
        assert (out_dir / "communication_report.json").exists()
        assert (out_dir / "trace_consistency_report.json").exists()
        assert (out_dir / "network_probe_report.json").exists()
        assert (out_dir / "timing_report.json").exists()
        assert (out_dir / "accounting_report.json").exists()
        assert (out_dir / "energy_signal_report.json").exists()
        assert (out_dir / "runtime_status_report.json").exists()
        assert (out_dir / "token_compatibility_report.json").exists()
        assert (out_dir / "measurement_stability_report.json").exists()
        assert (out_dir / "gamma_statistics_report.json").exists()
        assert (out_dir / "gamma_trend_report.json").exists()
        assert (out_dir / "optimization_report.json").exists()
        assert (out_dir / "system_optimization_report.json").exists()
        assert (out_dir / "gamma_policy_report.json").exists()
        assert (out_dir / "interaction_report.json").exists()
        assert (out_dir / "claim_readiness_report.json").exists()
        assert (out_dir / "figures" / "plot_manifest.json").exists()
        assert (out_dir / "figures" / "gamma_drafter_energy.svg").exists()
        assert (out_dir / "figures" / "paired_energy_savings.svg").exists()
        assert (out_dir / "artifact_manifest.json").exists()
        manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
        assert manifest["schema_version"] == "xronos-artifact-manifest-v1"
        assert any(
            item["path"] == "summary.csv" and item["exists"]
            for item in manifest["outputs"]
        )
        assert all("sha256" in item for item in manifest["outputs"] if item["exists"])
        assert (out_dir / "REPORT.md").exists()


def test_strict_full_factorial_claim_report() -> None:
    plans, rows = _strict_factorial_report_fixture()
    report = experiment_report.build_report(
        plans=plans,
        raw_rows=rows,
        require_interaction_analysis=True,
        require_claim_readiness=True,
    )
    assert report["ok"] is True
    assert report["summary_rows"] == 12
    assert report["paired_summary_rows"] == 8
    assert report["paired_prompt_rows"] == 16
    assert report["unpaired_prompt_rows"] == 0
    assert report["gamma_effect_rows"] == 8
    assert len(plans[1]["combinations"]) == 16
    assert report["plan_design"]["ok"] is True
    assert report["plan_design"]["speculative_plan_conditions"] == 4
    assert report["plan_design"]["baseline_plan_conditions"] == 4
    assert report["plan_design"]["spec_gamma_balance_groups"] == 8
    assert report["plan_design"]["missing_baseline_conditions"] == 0
    assert report["validation"]["ok"] is True
    assert report["validation"]["observed_sessions"] == 24
    assert report["system_boundary"]["claim_ready"] is True
    assert report["system_boundary"]["paired_baseline_system_boundaries"] == [
        "two_device_idle_drafter"
    ]
    assert report["output_equivalence"]["ok"] is True
    assert report["output_equivalence"]["checked_prompt_pairs"] == 16
    assert report["system_optimization"]["ok"] is True
    assert (
        report["system_optimization"]["energy_key"]
        == "mean_active_system_energy_mj_per_token"
    )
    assert report["system_optimization"]["best_joint_config"]["gamma"] == "2"
    assert (
        report["system_optimization"]["best_joint_config"]["drafter_freq_hz"]
        == "408000000"
    )
    assert (
        report["system_optimization"]["best_joint_config"]["verifier_clock_mhz"]
        == "1410"
    )
    assert report["gamma_policy"]["ok"] is True
    assert report["gamma_policy"]["policy_rows"] == 2
    assert report["gamma_policy"]["uses_gamma_dependent_verifier_clock"] is True
    assert report["interaction"]["ok"] is True
    assert report["interaction"]["required"] is True
    assert report["interaction"]["eligible"] is True
    assert report["interaction"]["missing_factorial_cells_count"] == 0
    assert report["interaction"]["verifier_clock_depends_on_gamma"] is True
    assert (
        report["interaction"]["marginal_independent_config"]["verifier_clock_mhz"]
        == "810"
    )
    assert (
        report["interaction"]["best_joint_config"]["verifier_clock_mhz"]
        == "1410"
    )
    assert (
        report["interaction"]["independent_energy_gap_pct_vs_joint_best"]
        == "20.000000"
    )
    assert report["claim_readiness"]["ok"] is True
    assert report["claim_readiness"]["blocked_claims"] == []
    assert set(report["claim_readiness"]["ready_claims"]) == {
        "adaptive_gamma_frequency_policy",
        "drafter_gamma_energy",
        "gamma_frequency_interaction",
        "joint_system_energy_optimization",
        "system_energy_vs_verifier_baseline",
    }


def test_stop_token_acceptance() -> None:
    accepted, stopped = spec_algorithm.truncate_accepted_at_stop(
        draft_tokens=[10, 11, 2, 13],
        accepted_tokens=4,
        stop_token_ids=[2],
    )
    assert accepted == 3
    assert stopped is True
    assert (
        spec_algorithm.should_append_replacement(
            accepted_tokens=accepted,
            draft_length=4,
            append_replacement_requested=True,
            accepted_stop=stopped,
        )
        is False
    )

    accepted, stopped = spec_algorithm.truncate_accepted_at_stop(
        draft_tokens=[10, 11, 12],
        accepted_tokens=3,
        stop_token_ids=[2],
    )
    assert accepted == 3
    assert stopped is False
    assert (
        spec_algorithm.should_append_replacement(
            accepted_tokens=accepted,
            draft_length=3,
            append_replacement_requested=True,
            accepted_stop=stopped,
        )
        is True
    )


def test_verify_decision() -> None:
    decision = spec_algorithm.plan_verify_decision(
        draft_tokens=[10, 11],
        verifier_predictions=[10, 11, 12],
        append_replacement_requested=True,
        stop_token_ids=[],
    )
    assert decision.accepted_tokens == 2
    assert decision.replacement_token == 12
    assert decision.append_replacement is True
    assert decision.committed_token_delta == 3

    decision = spec_algorithm.plan_verify_decision(
        draft_tokens=[10, 99],
        verifier_predictions=[10, 11, 12],
        append_replacement_requested=False,
        stop_token_ids=[],
    )
    assert decision.accepted_tokens == 1
    assert decision.replacement_token == 11
    assert decision.append_replacement is True
    assert decision.committed_token_delta == 2

    decision = spec_algorithm.plan_verify_decision(
        draft_tokens=[99, 100],
        verifier_predictions=[10, 11, 12],
        append_replacement_requested=False,
        stop_token_ids=[],
    )
    assert decision.accepted_tokens == 0
    assert decision.replacement_token == 10
    assert decision.append_replacement is True
    assert decision.committed_token_delta == 1

    decision = spec_algorithm.plan_verify_decision(
        draft_tokens=[10, 2, 13],
        verifier_predictions=[10, 2, 14, 15],
        append_replacement_requested=True,
        stop_token_ids=[2],
    )
    assert decision.accepted_tokens == 2
    assert decision.accepted_stop is True
    assert decision.append_replacement is False
    assert decision.committed_token_delta == 2

    try:
        spec_algorithm.plan_verify_decision(
            draft_tokens=[10, 11],
            verifier_predictions=[10, 11],
            append_replacement_requested=True,
            stop_token_ids=[],
        )
    except ValueError as exc:
        assert "verifier_predictions" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing replacement prediction")


def test_power_summary_short_window_fallback() -> None:
    sampler = power.INA3221PowerSampler()
    sampler.timestamps = [9.9, 11.0]
    sampler.samples = [
        {"tot_power": 1000.0, "gpu": 400.0},
        {"tot_power": 3000.0, "gpu": 1200.0},
    ]

    summaries, n_samples = sampler.summarize(10.0, 10.1)
    by_rail = {summary.rail: summary for summary in summaries}
    assert n_samples == 1
    assert by_rail["tot_power"].mean_power_mw == 1000.0
    assert round(by_rail["tot_power"].energy_mj, 6) == 100.0
    assert by_rail["gpu"].mean_power_mw == 400.0

    sampler.timestamps = [9.0, 10.05, 11.0]
    sampler.samples = [
        {"tot_power": 1000.0},
        {"tot_power": 2000.0},
        {"tot_power": 3000.0},
    ]
    summaries, n_samples = sampler.summarize(10.0, 10.1)
    assert n_samples == 1
    assert summaries[0].mean_power_mw == 2000.0


def test_ina3221_total_power_rail_semantics() -> None:
    sampler = power.INA3221PowerSampler()
    vin_v = Path("vdd_in_voltage")
    vin_c = Path("vdd_in_current")
    pom_v = Path("pom_5v_in_voltage")
    pom_c = Path("pom_5v_in_current")
    gpu_v = Path("vdd_gpu_voltage")
    gpu_c = Path("vdd_gpu_current")
    soc_v = Path("vdd_soc_voltage")
    soc_c = Path("vdd_soc_current")
    sampler.rails = [
        ("VDD_IN", vin_v, vin_c),
        ("POM_5V_IN", pom_v, pom_c),
        ("VDD_CPU_GPU_CV", gpu_v, gpu_c),
        ("VDD_SOC", soc_v, soc_c),
    ]
    values = {
        vin_v: 5000.0,
        vin_c: 600.0,
        pom_v: 5000.0,
        pom_c: 200.0,
        gpu_v: 1000.0,
        gpu_c: 700.0,
        soc_v: 1000.0,
        soc_c: 300.0,
    }
    sampler._read_float = values.get

    sample = sampler._read_power()
    assert sample["VDD_IN"] == 3000.0
    assert sample["POM_5V_IN"] == 1000.0
    assert sample["VDD_CPU_GPU_CV"] == 700.0
    assert sample["VDD_SOC"] == 300.0
    assert sample["sum_rails_power"] == 1000.0
    assert sample["tot_power"] == 4000.0

    sampler.rails = [
        ("VDD_CPU_GPU_CV", gpu_v, gpu_c),
        ("VDD_SOC", soc_v, soc_c),
    ]
    sample = sampler._read_power()
    assert sample["sum_rails_power"] == 1000.0
    assert sample["tot_power"] == 1000.0


def test_thermal_zone_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        gpu = root / "thermal_zone0"
        cpu = root / "thermal_zone1"
        gpu.mkdir()
        cpu.mkdir()
        (gpu / "type").write_text("gpu", encoding="utf-8")
        (gpu / "temp").write_text("41500", encoding="utf-8")
        (cpu / "type").write_text("cpu", encoding="utf-8")
        (cpu / "temp").write_text("39.0", encoding="utf-8")

        metadata = runtime.thermal_zone_metadata(root=root)
        assert metadata["thermal_max_temp_c"] == "41.50"
        assert metadata["thermal_zones"] == "gpu:41.50,cpu:39.00"


def test_network_probe_summary() -> None:
    summary = network_probe.summarize_rtt_ms([1.0, 2.0, 3.0, 4.0])
    assert summary["sample_count"] == 4
    assert summary["rtt_ms_mean"] == "2.500000"
    assert summary["rtt_ms_median"] == "2.500000"
    assert summary["rtt_ms_p95"] == "3.850000"

    payload = {
        "schema_version": network_probe.SCHEMA_VERSION,
        "ok": True,
        "targets": {
            "drafter": {
                "ok": True,
                "sample_count": 4,
                "rtt_ms_mean": "2.500000",
                "rtt_ms_p95": "3.850000",
            },
            "verifier": {
                "ok": True,
                "sample_count": 4,
                "rtt_ms_mean": "3.000000",
                "rtt_ms_p95": "4.000000",
            },
        },
    }
    report = experiment_report.network_probe_report(
        [payload],
        require_network_probe=True,
        max_mean_rtt_ms=5.0,
        max_p95_rtt_ms=5.0,
    )
    assert report["ok"] is True
    assert report["roles"] == ["drafter", "verifier"]

    missing_report = experiment_report.network_probe_report(
        [],
        require_network_probe=True,
    )
    assert missing_report["ok"] is False
    assert "missing_required_network_probe" in missing_report["errors"]

    slow_report = experiment_report.network_probe_report(
        [payload],
        require_network_probe=True,
        max_mean_rtt_ms=2.0,
    )
    assert slow_report["ok"] is False
    assert "network_mean_rtt_exceeds_limit" in slow_report["errors"]


def _write_audit_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        path.write_text("col\nvalue\n", encoding="utf-8")
    elif path.suffix == ".jsonl":
        path.write_text("{}\n", encoding="utf-8")
    elif path.suffix == ".json":
        payload = {"ok": True} if path.name == "report.json" else {}
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    elif path.suffix == ".svg":
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
    else:
        path.write_text("ok\n", encoding="utf-8")


def test_artifact_audit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        report_dir = results_dir / "report_gamma_freq"
        for relative_path in artifact_audit.ROOT_REQUIRED_FILES:
            _write_audit_file(results_dir / relative_path)
        for relative_path in artifact_audit.REPORT_REQUIRED_FILES:
            if relative_path == "artifact_manifest.json":
                continue
            _write_audit_file(report_dir / relative_path)

        outputs = []
        for path in sorted(report_dir.rglob("*")):
            if path.is_file():
                outputs.append(
                    {
                        "role": "report_output",
                        "path": str(path.relative_to(report_dir)),
                        "exists": True,
                        "bytes": path.stat().st_size,
                        "sha256": artifact_audit.sha256_file(path),
                    }
                )
        inputs = [
            {
                "role": "input",
                "path": str(results_dir / relative_path),
                "exists": True,
                "sha256": artifact_audit.sha256_file(results_dir / relative_path),
            }
            for relative_path in artifact_audit.ROOT_REQUIRED_FILES
        ]
        (report_dir / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "xronos-artifact-manifest-v1",
                    "inputs": inputs,
                    "outputs": outputs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        audit = artifact_audit.build_audit(
            results_dir=results_dir,
            report_dir=report_dir,
            require_report_ok=True,
        )
        assert audit["ok"] is True

        (results_dir / artifact_audit.ROOT_REQUIRED_FILES[0]).write_text(
            "{\"changed\": true}\n",
            encoding="utf-8",
        )
        changed_input_audit = artifact_audit.build_audit(
            results_dir=results_dir,
            report_dir=report_dir,
            require_report_ok=True,
        )
        assert changed_input_audit["ok"] is False
        assert "manifest:manifest_input_hash_mismatch" in changed_input_audit["errors"]
        _write_audit_file(results_dir / artifact_audit.ROOT_REQUIRED_FILES[0])

        manifest_payload = json.loads(
            (report_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        manifest_payload["outputs"] = [
            item
            for item in manifest_payload["outputs"]
            if item["path"] != "summary.csv"
        ]
        (report_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        missing_manifest_output_audit = artifact_audit.build_audit(
            results_dir=results_dir,
            report_dir=report_dir,
            require_report_ok=True,
        )
        assert missing_manifest_output_audit["ok"] is False
        assert (
            "manifest:manifest_missing_required_outputs"
            in missing_manifest_output_audit["errors"]
        )
        outputs = []
        for path in sorted(report_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                outputs.append(
                    {
                        "role": "report_output",
                        "path": str(path.relative_to(report_dir)),
                        "exists": True,
                        "bytes": path.stat().st_size,
                        "sha256": artifact_audit.sha256_file(path),
                    }
                )
        manifest_payload["outputs"] = outputs
        (report_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_payload["inputs"] = [
            item
            for item in manifest_payload["inputs"]
            if not item["path"].endswith("network_probe.json")
        ]
        (report_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        missing_manifest_input_audit = artifact_audit.build_audit(
            results_dir=results_dir,
            report_dir=report_dir,
            require_report_ok=True,
        )
        assert missing_manifest_input_audit["ok"] is False
        assert (
            "manifest:manifest_missing_required_inputs"
            in missing_manifest_input_audit["errors"]
        )
        manifest_payload["inputs"] = inputs
        (report_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        (results_dir / "network_probe.json").unlink()
        missing_audit = artifact_audit.build_audit(
            results_dir=results_dir,
            report_dir=report_dir,
            require_report_ok=True,
        )
        assert missing_audit["ok"] is False
        assert "root:missing_root_files" in missing_audit["errors"]


def test_k8s_runbook_renderer() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "k8s/spec-decoding.yaml"
    manifest = manifest_path.read_text(encoding="utf-8")
    payload = k8s_runbook.build_runbook(
        manifest_text=manifest,
        manifest_path="k8s/spec-decoding.yaml",
        deployment_timeout_s=1800,
        job_timeout_s=7200,
    )
    assert payload["schema_version"] == k8s_runbook.SCHEMA_VERSION
    assert payload["ok"] is True
    assert payload["never_executes_kubectl"] is True
    assert payload["namespace"] == "xronos-spec"
    assert len(payload["preconditions"]) == 4
    assert len(payload["steps"]) == 12
    assert payload["steps"][0]["job"] == "drafter-doctor"
    assert payload["steps"][0]["resume_command"] == (
        "kubectl patch job drafter-doctor -n xronos-spec --type merge "
        "-p '{\"spec\":{\"suspend\":false}}'"
    )
    assert payload["steps"][8]["job"] == "verifier-baseline"
    assert payload["steps"][9]["job"] == "spec-driver"
    assert payload["steps"][9]["run_after"] == ["verifier-baseline"]
    assert "kubectl wait job/spec-driver" in payload["steps"][9]["wait_command"]
    assert "kubectl logs job/spec-driver" in payload["steps"][9]["logs_command"]
    markdown = "\n".join(k8s_runbook.markdown_lines(payload))
    assert "## Job Order" in markdown
    assert "kubectl rollout status deployment/spec-drafter" in markdown

    broken_manifest = manifest.replace(
        'xronos.run-after: "spec-driver"',
        'xronos.run-after: "plan-audit"',
        1,
    )
    broken_payload = k8s_runbook.build_runbook(
        manifest_text=broken_manifest,
        manifest_path="k8s/spec-decoding.yaml",
    )
    assert broken_payload["ok"] is False
    assert "manifest_audit_not_ok" in broken_payload["errors"]


def _k8s_doc(manifest_text: str, kind: str, name: str) -> str:
    for doc in manifest_text.split("\n---"):
        if f"kind: {kind}" in doc and f"\n  name: {name}\n" in doc:
            return doc
    raise AssertionError(f"Missing {kind}/{name} in k8s/spec-decoding.yaml")


def test_k8s_manifest_operational_guards() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "k8s/spec-decoding.yaml"
    manifest = manifest_path.read_text(encoding="utf-8")

    drafter_doctor = _k8s_doc(manifest, "Job", "drafter-doctor")
    assert "xronos-role: jetson-drafter" in drafter_doctor
    assert "nvidia.com/gpu: 1" in drafter_doctor
    assert "privileged: true" in drafter_doctor
    assert "mountPath: /sys" in drafter_doctor
    assert "mountPath: /dev" in drafter_doctor

    driver_doctor = _k8s_doc(manifest, "Job", "driver-doctor")
    assert "xronos-role: gpu-verifier" in driver_doctor
    assert "python -m xronos.infer.experiment_doctor" in driver_doctor
    assert "--role driver" in driver_doctor
    assert "--k8s-manifest k8s/spec-decoding.yaml" in driver_doctor
    assert "--require-k8s-manifest" in driver_doctor

    drafter_deployment = _k8s_doc(manifest, "Deployment", "spec-drafter")
    assert "xronos-role: jetson-drafter" in drafter_deployment
    assert "nvidia.com/gpu: 1" in drafter_deployment
    assert "startupProbe:" in drafter_deployment
    assert "readinessProbe:" in drafter_deployment
    assert "livenessProbe:" in drafter_deployment
    assert "mountPath: /sys" in drafter_deployment
    assert "mountPath: /dev" in drafter_deployment

    verifier_deployment = _k8s_doc(manifest, "Deployment", "spec-verifier")
    assert "xronos-role: gpu-verifier" in verifier_deployment
    assert "nvidia.com/gpu: 1" in verifier_deployment
    assert "startupProbe:" in verifier_deployment
    assert "readinessProbe:" in verifier_deployment
    assert "livenessProbe:" in verifier_deployment

    network_job = _k8s_doc(manifest, "Job", "network-probe")
    assert "xronos-role: gpu-verifier" in network_job
    assert "python -m xronos.infer.network_probe" in network_job
    assert "--drafter-addr spec-drafter:50061" in network_job
    assert "--verifier-addr spec-verifier:50062" in network_job

    k8s_audit_job = _k8s_doc(manifest, "Job", "k8s-manifest-audit")
    assert "python -m xronos.infer.k8s_manifest_audit" in k8s_audit_job
    assert "--out /results/k8s_manifest_audit.json" in k8s_audit_job
    assert "python -m xronos.infer.k8s_runbook" in k8s_audit_job
    assert "--out /results/k8s_runbook.json" in k8s_audit_job
    assert "--markdown-out /results/k8s_runbook.md" in k8s_audit_job
    assert 'xronos.run-order: "4"' in k8s_audit_job
    assert "/results/k8s_manifest_audit.json,/results/k8s_runbook.json" in k8s_audit_job

    baseline_plan_job = _k8s_doc(manifest, "Job", "verifier-baseline-plan")
    assert "python -m xronos.infer.verifier_baseline_driver" in baseline_plan_job
    assert "--dry-run" in baseline_plan_job
    assert "--resume" not in baseline_plan_job
    assert "--plan-out /results/verifier_baseline_plan.json" in baseline_plan_job
    assert 'xronos.run-after: "network-probe"' in baseline_plan_job

    spec_plan_job = _k8s_doc(manifest, "Job", "spec-plan")
    assert "python -m xronos.infer.spec_driver" in spec_plan_job
    assert "--dry-run" in spec_plan_job
    assert "--resume" not in spec_plan_job
    assert "--plan-out /results/spec_plan.json" in spec_plan_job
    assert 'xronos.run-after: "verifier-baseline-plan"' in spec_plan_job

    plan_audit_job = _k8s_doc(manifest, "Job", "plan-audit")
    assert "python -m xronos.infer.plan_audit" in plan_audit_job
    assert "--plan /results/spec_plan.json /results/verifier_baseline_plan.json" in plan_audit_job
    assert "--min-gammas 5" in plan_audit_job
    assert 'xronos.run-after: "spec-plan"' in plan_audit_job
    assert (
        'xronos.consumes: "/results/spec_plan.json,'
        '/results/verifier_baseline_plan.json"'
        in plan_audit_job
    )

    baseline_job = _k8s_doc(manifest, "Job", "verifier-baseline")
    assert "--resume" in baseline_job

    spec_driver_job = _k8s_doc(manifest, "Job", "spec-driver")
    assert "--resume" in spec_driver_job

    report_job = _k8s_doc(manifest, "Job", "spec-report")
    assert 'xronos.run-after: "spec-driver"' in report_job
    assert 'xronos.produces: "/results/report_gamma_freq"' in report_job
    assert "/results/k8s_runbook.json,/results/k8s_runbook.md" in report_job
    assert "--plan-audit-json /results/plan_audit.json" in report_job
    assert "--require-plan-audit" in report_job
    assert "--network-json /results/network_probe.json" in report_job
    assert "--require-network-probe" in report_job
    assert "--k8s-manifest-audit-json /results/k8s_manifest_audit.json" in report_job
    assert "--k8s-runbook-json /results/k8s_runbook.json" in report_job
    assert "--k8s-runbook-markdown /results/k8s_runbook.md" in report_job
    assert "--require-k8s-manifest-audit" in report_job
    assert (
        "--trace-jsonl /results/spec_trace.jsonl "
        "/results/verifier_baseline_trace.jsonl"
        in report_job
    )
    assert "--require-trace" in report_job
    assert "--require-interaction-analysis" in report_job
    assert "--require-claim-readiness" in report_job
    assert "--min-gammas 5" in report_job

    audit_job = _k8s_doc(manifest, "Job", "artifact-audit")
    assert "python -m xronos.infer.artifact_audit" in audit_job
    assert "--require-report-ok" in audit_job
    assert 'xronos.run-after: "spec-report"' in audit_job
    assert 'xronos.consumes: "/results/report_gamma_freq"' in audit_job

    audit = k8s_manifest_audit.build_audit(manifest)
    assert audit["ok"] is True
    assert audit["spec_parity"]["ok"] is True
    assert audit["baseline_parity"]["ok"] is True
    assert audit["runbook"]["ok"] is True
    assert audit["placement"]["ok"] is True
    assert audit["placement"]["checked_workloads"] == 16
    assert audit["runbook"]["job_count"] == 12
    assert audit["runbook"]["missing_input_producers"] == []
    assert audit["k8s_manifest_audit_output"] == "/results/k8s_manifest_audit.json"
    assert (
        audit["report_k8s_manifest_audit_input"]
        == "/results/k8s_manifest_audit.json"
    )

    broken_manifest = manifest.replace(
        '--max-start-temp-c "$MAX_START_TEMP_C" \\',
        "",
        1,
    )
    broken_audit = k8s_manifest_audit.build_audit(broken_manifest)
    assert broken_audit["ok"] is False
    assert "baseline_plan_measured_design_mismatch" in broken_audit["errors"]

    broken_placement = manifest.replace(
        "xronos-role: jetson-drafter",
        "xronos-role: gpu-verifier",
        1,
    )
    broken_placement_audit = k8s_manifest_audit.build_audit(broken_placement)
    assert broken_placement_audit["ok"] is False
    assert "placement_mismatch" in broken_placement_audit["errors"]
    assert "placement_missing_required_token" in broken_placement_audit[
        "placement"
    ]["errors"]

    mismatched_audit_input = manifest.replace(
        "--k8s-manifest-audit-json /results/k8s_manifest_audit.json",
        "--k8s-manifest-audit-json /results/other_k8s_manifest_audit.json",
    )
    mismatched_input_audit = k8s_manifest_audit.build_audit(mismatched_audit_input)
    assert mismatched_input_audit["ok"] is False
    assert (
        "k8s_manifest_audit_output_input_mismatch"
        in mismatched_input_audit["errors"]
    )

    broken_runbook = manifest.replace(
        'xronos.run-after: "spec-driver"',
        'xronos.run-after: "plan-audit"',
        1,
    )
    broken_runbook_audit = k8s_manifest_audit.build_audit(broken_runbook)
    assert broken_runbook_audit["ok"] is False
    assert "runbook_annotation_mismatch" in broken_runbook_audit["errors"]
    assert "runbook_after_mismatch" in broken_runbook_audit["runbook"]["errors"]


def test_dockerfile_includes_experiment_templates() -> None:
    dockerfile_path = Path(__file__).resolve().parents[2] / "docker/infer.Dockerfile"
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    assert "COPY xronos ./xronos" in dockerfile
    assert "COPY k8s ./k8s" in dockerfile


def main() -> None:
    test_sweep_plan()
    test_two_device_baseline_plan()
    test_arg_validation()
    test_prompt_jsonl_plan()
    test_plan_audit()
    test_prompt_ids_must_be_unique()
    test_doctor_prompt_source_checks_unique_prompts()
    test_doctor_checks_k8s_manifest_template()
    test_doctor_hf_token_and_cache_checks()
    test_condition_shuffle_seed()
    test_thermal_guard_metadata()
    test_startup_health_wait()
    test_spec_driver_fake_orchestration()
    test_spec_driver_rejects_over_budget_replacement()
    test_spec_driver_rejects_invalid_verifier_progress()
    test_spec_driver_rejects_drafter_commit_mismatch()
    test_spec_driver_rejects_verifier_commit_mismatch()
    test_spec_driver_rejects_init_context_mismatch()
    test_write_rows_checkpoint_replaces_output()
    test_resume_rows_validate_plan_and_orders()
    test_frequency_lock_uses_gpu_index()
    test_doctor_version_comparison()
    test_analyzer_summary()
    test_gamma_effect_analysis()
    test_analyzer_keeps_incompatible_conditions_separate()
    test_analyzer_read_csvs()
    test_session_rows_prefer_primary_rail()
    test_validate_results()
    test_select_best_config()
    test_interaction_report()
    test_compare_to_baseline()
    test_paired_prompt_compare()
    test_experiment_report()
    test_strict_full_factorial_claim_report()
    test_stop_token_acceptance()
    test_verify_decision()
    test_power_summary_short_window_fallback()
    test_ina3221_total_power_rail_semantics()
    test_thermal_zone_metadata()
    test_network_probe_summary()
    test_artifact_audit()
    test_k8s_runbook_renderer()
    test_k8s_manifest_operational_guards()
    test_dockerfile_includes_experiment_templates()
    print("xronos.infer self-test passed")


if __name__ == "__main__":
    main()
