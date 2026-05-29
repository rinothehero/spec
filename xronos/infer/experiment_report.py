import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from xronos.infer import (
    analyze_gamma_effect,
    analyze_spec_results,
    paired_prompt_compare,
    plot_experiment,
    select_best_config,
    spec_driver,
    validate_results,
)


DEFAULT_SUMMARY_ENERGY_KEY = "mean_drafter_active_energy_mj_per_token"
DEFAULT_PAIRED_ENERGY_KEY = "system_active_energy_mj_per_generated_token"
SYSTEM_ACTIVE_SUMMARY_ENERGY_KEY = "mean_active_system_energy_mj_per_token"
SYSTEM_TOTAL_SUMMARY_ENERGY_KEY = "mean_system_energy_mj_per_token"
PlanPairKey = Tuple[str, str, str, str, str, str, str]
PlanSystemKey = Tuple[str, str, str, str, str, str, str, str]
GammaDesignKey = Tuple[str, str, str, str, str, str, str, str, str, str, str]
SpecGammaBalanceKey = Tuple[str, str, str, str, str, str, str, str]
SpecFactorialKey = Tuple[str, str, str, str, str, str, str]


def _int_value(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or "0"))
    except (TypeError, ValueError):
        return 0


def write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl_events(paths: Sequence[str]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for path in paths:
        with open(path) as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if isinstance(event, dict):
                    event["_source"] = path
                    event["_line"] = line_number
                    events.append(event)
    return events


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, role: str = "", base_dir: Optional[Path] = None) -> Dict[str, object]:
    record_path = path
    if base_dir is not None:
        try:
            record_path = path.relative_to(base_dir)
        except ValueError:
            record_path = path
    if not path.exists():
        return {
            "role": role,
            "path": str(record_path),
            "exists": False,
        }
    return {
        "role": role,
        "path": str(record_path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def write_artifact_manifest(
    out_dir: Path,
    input_files: Sequence[Tuple[str, str]],
) -> Dict[str, object]:
    manifest_path = out_dir / "artifact_manifest.json"
    output_files = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": "xronos-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "command": sys.argv,
        "inputs": [
            file_record(Path(path), role=role)
            for role, path in input_files
        ],
        "outputs": [
            file_record(path, role="report_output", base_dir=out_dir)
            for path in output_files
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def write_markdown(path: Path, report: Dict[str, object]) -> None:
    validation = report["validation"]
    plan_design = report["plan_design"]
    plan_audit = report["plan_audit"]
    k8s_manifest_audit = report["k8s_manifest_audit"]
    energy_design = report["energy_design"]
    plan_integrity = report["plan_integrity"]
    gamma_design = report["gamma_design"]
    doctor_design = report["doctor_design"]
    schema_contract = report["schema_contract"]
    frequency_consistency = report["frequency_consistency"]
    measurement_setup = report["measurement_setup"]
    provenance = report["provenance"]
    model_setup = report["model_setup"]
    input_consistency = report["input_consistency"]
    communication = report["communication"]
    trace_consistency = report["trace_consistency"]
    network_probe = report["network_probe"]
    timing = report["timing"]
    accounting = report["accounting"]
    energy_signal = report["energy_signal"]
    runtime_status = report["runtime_status"]
    token_compatibility = report["token_compatibility"]
    gamma_statistics = report["gamma_statistics"]
    gamma_trend = report["gamma_trend"]
    measurement_stability = report["measurement_stability"]
    output_equivalence = report["output_equivalence"]
    system_boundary = report["system_boundary"]
    best_summary = report.get("best_summary_config")
    best_paired = report.get("best_paired_config")

    lines = [
        "# Xronos Speculative Decoding Experiment Report",
        "",
        "## Validation",
        "",
        f"- ok: {validation['ok']}",
        f"- plan design ok: {plan_design['ok']}",
        f"- pre-run plan audit ok: {plan_audit['ok']}",
        f"- pre-run plan audit reports: {plan_audit['plan_audit_reports']}",
        f"- pre-run plan audit not-ok reports: {plan_audit['not_ok_reports']}",
        f"- Kubernetes manifest audit ok: {k8s_manifest_audit['ok']}",
        f"- Kubernetes manifest audit reports: {k8s_manifest_audit['reports']}",
        f"- Kubernetes manifest audit not-ok reports: {k8s_manifest_audit['not_ok_reports']}",
        "- Kubernetes placement mismatch reports: "
        f"{k8s_manifest_audit['placement_mismatch_reports']}",
        f"- energy design ok: {energy_design['ok']}",
        f"- plan integrity ok: {plan_integrity['ok']}",
        f"- missing plan design hashes: {plan_integrity['missing_plan_design_hashes']}",
        f"- invalid plan design hashes: {plan_integrity['invalid_plan_design_hashes']}",
        f"- missing plan hashes: {plan_integrity['missing_plan_hashes']}",
        f"- invalid plan hashes: {plan_integrity['invalid_plan_hashes']}",
        "- result plan-hash mismatch sessions: "
        f"{plan_integrity['mismatched_result_plan_hash_sessions']}",
        "- result design-hash mismatch sessions: "
        f"{plan_integrity['mismatched_result_plan_design_hash_sessions']}",
        f"- gamma design ok: {gamma_design['ok']}",
        f"- min gamma values: {gamma_design['min_gammas']}",
        f"- configs meeting min gamma count: {gamma_design['min_gamma_configs']}",
        f"- gamma statistics ok: {gamma_statistics['ok']}",
        f"- gamma trend ok: {gamma_trend['ok']}",
        f"- gamma trend valid groups: {gamma_trend['valid_trend_groups']}",
        f"- missing gamma-trend rows: {gamma_trend['missing_trend_rows']}",
        f"- invalid gamma-trend rows: {gamma_trend['invalid_trend_rows']}",
        "- missing gamma-stat rows: "
        f"{gamma_statistics['missing_stat_rows']}",
        "- invalid gamma-stat rows: "
        f"{gamma_statistics['invalid_stat_rows']}",
        "- missing directional sign-test rows: "
        f"{gamma_statistics['missing_directional_sign_test_rows']}",
        f"- measurement stability ok: {measurement_stability['ok']}",
        "- max observed energy CV: "
        f"{measurement_stability['observed_max_energy_cv']}",
        "- max observed latency CV: "
        f"{measurement_stability['observed_max_latency_cv']}",
        "- energy CV violations: "
        f"{measurement_stability['energy_cv_violations']}",
        "- latency CV violations: "
        f"{measurement_stability['latency_cv_violations']}",
        f"- doctor design ok: {doctor_design['ok']}",
        f"- doctor reports: {doctor_design['doctor_reports']}",
        f"- doctor roles: {','.join(doctor_design['roles'])}",
        f"- doctor failures: {doctor_design['failures']}",
        f"- doctor warnings: {doctor_design['warnings']}",
        f"- doctor not-ok reports: {','.join(doctor_design['not_ok_reports'])}",
        f"- result schema ok: {schema_contract['ok']}",
        f"- result schema versions: {','.join(schema_contract['result_schema_versions'])}",
        "- missing result-schema sessions: "
        f"{schema_contract['missing_result_schema_sessions']}",
        "- missing algorithm-version sessions: "
        f"{schema_contract['missing_algorithm_version_sessions']}",
        "- missing driver result-schema sessions: "
        f"{schema_contract['missing_driver_result_schema_sessions']}",
        "- driver/result schema mismatch sessions: "
        f"{schema_contract['driver_result_schema_mismatch_sessions']}",
        "- missing driver RPC-schema sessions: "
        f"{schema_contract['missing_driver_spec_rpc_schema_sessions']}",
        "- missing role RPC-schema sessions: "
        f"{schema_contract['missing_role_spec_rpc_schema_sessions']}",
        "- role RPC-schema mismatch sessions: "
        f"{schema_contract['role_spec_rpc_schema_mismatch_sessions']}",
        f"- frequency consistency ok: {frequency_consistency['ok']}",
        "- frequency value mismatch sessions: "
        f"{frequency_consistency['mismatch_sessions']}",
        "- frequency lock failure sessions: "
        f"{frequency_consistency['lock_failure_sessions']}",
        f"- measurement setup ok: {measurement_setup['ok']}",
        "- missing power-interval sessions: "
        f"{measurement_setup['missing_power_interval_sessions']}",
        "- missing primary-rail sessions: "
        f"{measurement_setup['missing_primary_rail_sessions']}",
        "- selected primary-rail mismatch sessions: "
        f"{measurement_setup['selected_primary_rail_mismatch_sessions']}",
        "- same drafter/verifier host sessions: "
        f"{measurement_setup['same_drafter_verifier_host_sessions']}",
        "- missing drafter/verifier host sessions: "
        f"{measurement_setup['missing_host_identity_sessions']}",
        "- missing idle-baseline policy sessions: "
        f"{measurement_setup['missing_idle_baseline_policy_sessions']}",
        "- invalid idle-baseline policy sessions: "
        f"{measurement_setup['invalid_idle_baseline_policy_sessions']}",
        f"- provenance ok: {provenance['ok']}",
        f"- missing provenance sessions: {provenance['missing_git_commit_sessions']}",
        f"- role commit mismatch sessions: {provenance['role_commit_mismatch_sessions']}",
        f"- dirty driver sessions: {provenance['dirty_driver_sessions']}",
        f"- model setup ok: {model_setup['ok']}",
        f"- same drafter/verifier model sessions: {model_setup['same_model_sessions']}",
        f"- missing model-name sessions: {model_setup['missing_model_name_sessions']}",
        f"- missing model-size sessions: {model_setup['missing_model_size_sessions']}",
        f"- non-smaller drafter sessions: {model_setup['non_smaller_drafter_sessions']}",
        f"- input consistency ok: {input_consistency['ok']}",
        f"- missing input-metadata sessions: {input_consistency['missing_input_metadata_sessions']}",
        f"- tokenization mismatch groups: {input_consistency['tokenization_mismatch_groups']}",
        f"- prompt-id hash conflict groups: {input_consistency['prompt_id_hash_conflict_groups']}",
        f"- communication metrics ok: {communication['ok']}",
        f"- missing communication sessions: {communication['missing_metric_sessions']}",
        f"- invalid communication sessions: {communication['invalid_metric_sessions']}",
        f"- trace consistency ok: {trace_consistency['ok']}",
        f"- trace events: {trace_consistency['trace_events']}",
        f"- missing trace sessions: {trace_consistency['missing_trace_sessions']}",
        f"- trace mismatch sessions: {trace_consistency['trace_mismatch_sessions']}",
        "- missing trace summary sessions: "
        f"{trace_consistency['missing_trace_summary_sessions']}",
        "- trace summary mismatch sessions: "
        f"{trace_consistency['trace_summary_mismatch_sessions']}",
        f"- orphan trace sessions: {trace_consistency['orphan_trace_sessions']}",
        f"- network probe ok: {network_probe['ok']}",
        f"- network probe reports: {network_probe['network_reports']}",
        f"- network probe roles: {','.join(network_probe['roles'])}",
        f"- network mean RTT violations: {network_probe['mean_rtt_violations']}",
        f"- network p95 RTT violations: {network_probe['p95_rtt_violations']}",
        f"- timing consistency ok: {timing['ok']}",
        f"- throughput mismatch sessions: {timing['throughput_mismatch_sessions']}",
        f"- latency mismatch sessions: {timing['latency_mismatch_sessions']}",
        f"- missing timing sessions: {timing['missing_timing_sessions']}",
        f"- accounting ok: {accounting['ok']}",
        f"- token-accounting error sessions: {accounting['token_error_sessions']}",
        f"- energy-accounting mismatch sessions: {accounting['energy_mismatch_sessions']}",
        f"- missing accounting field sessions: {accounting['missing_field_sessions']}",
        f"- energy signal ok: {energy_signal['ok']}",
        "- nonpositive total-energy sessions: "
        f"{energy_signal['nonpositive_total_energy_sessions']}",
        "- nonpositive active-energy sessions: "
        f"{energy_signal['nonpositive_active_energy_sessions']}",
        "- nonpositive idle-power sessions: "
        f"{energy_signal['nonpositive_idle_power_sessions']}",
        f"- runtime status ok: {runtime_status['ok']}",
        f"- runtime temp rows: {runtime_status['temperature_rows']}",
        f"- max drafter temp C: {runtime_status['max_drafter_runtime_temp_c']}",
        f"- max verifier temp C: {runtime_status['max_verifier_runtime_temp_c']}",
        f"- throttle-active sessions: {runtime_status['throttle_active_sessions']}",
        f"- token compatibility ok: {token_compatibility['ok']}",
        f"- vocab mismatch sessions: {token_compatibility['vocab_mismatch_sessions']}",
        f"- missing vocab metadata sessions: {token_compatibility['missing_vocab_metadata_sessions']}",
        "- tokenizer/model vocab mismatch sessions: "
        f"{token_compatibility['tokenizer_vocab_mismatch_sessions']}",
        "- missing tokenizer metadata sessions: "
        f"{token_compatibility['missing_tokenizer_metadata_sessions']}",
        "- tokenizer special-token mismatch sessions: "
        f"{token_compatibility['tokenizer_special_token_mismatch_sessions']}",
        "- mixed tokenizer metadata keys: "
        f"{','.join(token_compatibility['mixed_tokenizer_metadata_keys'])}",
        f"- active energy requires idle baseline: {energy_design['requires_idle_baseline']}",
        f"- idle baseline policies: {','.join(energy_design['idle_baseline_policies'])}",
        "- active energy requires run-idle policy: "
        f"{energy_design['requires_run_idle_policy']}",
        f"- missing baseline plan conditions: {plan_design['missing_baseline_conditions']}",
        f"- plan gamma values: {','.join(plan_design['spec_gamma_values'])}",
        f"- plan has gamma=1 baseline: {plan_design['has_gamma_one_baseline']}",
        f"- plan tokenizers: {','.join(plan_design['tokenizers'])}",
        f"- missing schema-version plans: {plan_design['missing_schema_version_plans']}",
        f"- missing tokenizer plans: {plan_design['missing_tokenizer_plans']}",
        f"- missing prompt metadata plans: {len(plan_design['missing_prompt_metadata_plans'])}",
        f"- duplicate prompt-hash plans: {len(plan_design['duplicate_prompt_hash_plans'])}",
        "- insufficient unique-prompt plans: "
        f"{len(plan_design['insufficient_unique_prompt_plans'])}",
        f"- measurement-schedule plans: {plan_design['measurement_schedule_plans']}",
        f"- missing measurement schedules: {len(plan_design['missing_measurement_schedules'])}",
        f"- warmup-schedule plans: {plan_design['warmup_schedule_plans']}",
        f"- missing warmup schedules: {len(plan_design['missing_warmup_schedules'])}",
        f"- blocked warmup schedules: {len(plan_design['blocked_warmup_schedules'])}",
        f"- missing warmup plans: {len(plan_design['missing_warmup_plans'])}",
        f"- nonpositive warmup plans: {len(plan_design['nonpositive_warmup_plans'])}",
        f"- baseline run-count mismatches: {len(plan_design['baseline_run_mismatches'])}",
        f"- baseline warmup mismatches: {len(plan_design['baseline_warmup_mismatches'])}",
        f"- unrandomized run-order plans: {len(plan_design['unrandomized_run_order_plans'])}",
        f"- missing randomization seeds: {len(plan_design['missing_randomization_seed_plans'])}",
        f"- blocked measurement schedules: {len(plan_design['blocked_measurement_schedules'])}",
        "- invalid measurement schedules: "
        f"{len(plan_design['invalid_measurement_schedules'])}",
        f"- gamma balance groups: {plan_design['spec_gamma_balance_groups']}",
        f"- incomplete gamma groups: {len(plan_design['incomplete_gamma_groups'])}",
        f"- nonuniform gamma run groups: {len(plan_design['nonuniform_gamma_run_groups'])}",
        f"- spec factorial groups: {plan_design['spec_factorial_groups']}",
        "- incomplete spec factorial groups: "
        f"{len(plan_design['incomplete_spec_factorial_groups'])}",
        f"- gamma-ready configs: {gamma_design['ready_configs']}",
        f"- multi-gamma configs: {gamma_design['multi_gamma_configs']}",
        "- incomplete gamma prompt-overlap rows: "
        f"{gamma_design['incomplete_prompt_overlap_rows']}",
        "- insufficient gamma paired-prompt rows: "
        f"{gamma_design['insufficient_paired_prompt_rows']}",
        "- non-gamma-one baseline rows: "
        f"{gamma_design['non_gamma_one_baseline_rows']}",
        f"- expected sessions: {validation['expected_sessions']}",
        f"- observed sessions: {validation['observed_sessions']}",
        f"- missing conditions: {len(validation['missing'])}",
        f"- extra conditions: {len(validation['extra'])}",
        f"- incomplete energy sessions: {len(validation['incomplete_energy_sessions'])}",
        f"- missing idle baseline sessions: {len(validation['missing_idle_baseline_sessions'])}",
        f"- invalid generation sessions: {len(validation['invalid_generation_sessions'])}",
        f"- invalid metric sessions: {len(validation['invalid_metric_sessions'])}",
        "- insufficient power sample sessions: "
        f"{len(validation['insufficient_power_sample_sessions'])}",
        f"- missing run-index conditions: {len(validation['missing_run_indices'])}",
        f"- duplicate run-index conditions: {len(validation['duplicate_run_indices'])}",
        "- missing measurement-order algorithms: "
        f"{len(validation['missing_measurement_orders'])}",
        "- duplicate measurement-order algorithms: "
        f"{len(validation['duplicate_measurement_orders'])}",
        "- measurement-schedule mismatch sessions: "
        f"{len(validation['measurement_schedule_mismatches'])}",
        f"- system boundary ok: {system_boundary['ok']}",
        f"- system boundary claim ready: {system_boundary['claim_ready']}",
        "- paired baseline system boundaries: "
        f"{','.join(system_boundary['paired_baseline_system_boundaries'])}",
        "- wrong baseline boundary pairs: "
        f"{system_boundary['wrong_baseline_boundary_pairs']}",
        f"- output equivalence ok: {output_equivalence['ok']}",
        f"- output checked prompt-pairs: {output_equivalence['checked_prompt_pairs']}",
        f"- output unchecked prompt-pairs: {output_equivalence['unchecked_prompt_pairs']}",
        f"- output mismatched prompt-pairs: {output_equivalence['mismatched_prompt_pairs']}",
        f"- output unstable prompt-pairs: {output_equivalence['unstable_prompt_pairs']}",
        "",
        "## Outputs",
        "",
        f"- summary rows: {report['summary_rows']}",
        f"- paired configs: {report['paired_summary_rows']}",
        f"- paired prompt rows: {report['paired_prompt_rows']}",
        f"- unpaired speculative prompt rows: {report['unpaired_prompt_rows']}",
        f"- gamma effect rows: {report['gamma_effect_rows']}",
        f"- feasible configs: {report['feasible_rows']}",
        f"- pareto configs: {report['pareto_rows']}",
        f"- figure plots: {report.get('figure_manifest', {}).get('ok_plots', 0)}",
        "",
    ]
    if best_summary:
        lines.extend(
            [
                "## Best Summary Config",
                "",
                f"- gamma: {best_summary.get('gamma', '')}",
                f"- drafter_freq_hz: {best_summary.get('drafter_freq_hz', '')}",
                f"- verifier_clock_mhz: {best_summary.get('verifier_clock_mhz', '')}",
                f"- energy: {best_summary.get(report['summary_energy_key'], '')}",
                f"- tokens_per_s: {best_summary.get('mean_tokens_per_s', '')}",
                "",
            ]
        )
    optimization = report.get("optimization", {})
    if optimization:
        lines.extend(
            [
                "## Joint Optimization Summary",
                "",
                f"- optimization ok: {optimization.get('ok', '')}",
                "- energy_savings_pct_vs_best_gamma_one: "
                f"{optimization.get('energy_savings_pct_vs_best_gamma_one', '')}",
                "- throughput_ratio_vs_best_gamma_one: "
                f"{optimization.get('throughput_ratio_vs_best_gamma_one', '')}",
                "- latency_ratio_vs_best_gamma_one: "
                f"{optimization.get('latency_ratio_vs_best_gamma_one', '')}",
                "- energy_margin_pct_vs_runner_up: "
                f"{optimization.get('energy_margin_pct_vs_runner_up', '')}",
                "- energy_ci95_margin_clear: "
                f"{optimization.get('energy_ci95_margin_clear', '')}",
                "",
            ]
        )
    system_optimization = report.get("system_optimization", {})
    if system_optimization:
        lines.extend(
            [
                "## System Energy Optimization Summary",
                "",
                f"- optimization ok: {system_optimization.get('ok', '')}",
                f"- energy key: {system_optimization.get('energy_key', '')}",
                "- energy_savings_pct_vs_best_gamma_one: "
                f"{system_optimization.get('energy_savings_pct_vs_best_gamma_one', '')}",
                "- throughput_ratio_vs_best_gamma_one: "
                f"{system_optimization.get('throughput_ratio_vs_best_gamma_one', '')}",
                "- latency_ratio_vs_best_gamma_one: "
                f"{system_optimization.get('latency_ratio_vs_best_gamma_one', '')}",
                "- energy_margin_pct_vs_runner_up: "
                f"{system_optimization.get('energy_margin_pct_vs_runner_up', '')}",
                "- energy_ci95_margin_clear: "
                f"{system_optimization.get('energy_ci95_margin_clear', '')}",
                "",
            ]
        )
    gamma_policy = report.get("gamma_policy", {})
    if gamma_policy:
        lines.extend(
            [
                "## Gamma Frequency Policy",
                "",
                f"- policy ok: {gamma_policy.get('ok', '')}",
                f"- policy rows: {gamma_policy.get('policy_rows', '')}",
                "- uses_gamma_dependent_verifier_clock: "
                f"{gamma_policy.get('uses_gamma_dependent_verifier_clock', '')}",
                "- uses_gamma_dependent_drafter_freq: "
                f"{gamma_policy.get('uses_gamma_dependent_drafter_freq', '')}",
                "",
            ]
        )
    interaction = report.get("interaction", {})
    if interaction:
        lines.extend(
            [
                "## Gamma Frequency Interaction",
                "",
                f"- interaction ok: {interaction.get('ok', '')}",
                f"- required: {interaction.get('required', '')}",
                f"- eligible: {interaction.get('eligible', '')}",
                "- verifier_clock_depends_on_gamma: "
                f"{interaction.get('verifier_clock_depends_on_gamma', '')}",
                "- independent_energy_gap_pct_vs_joint_best: "
                f"{interaction.get('independent_energy_gap_pct_vs_joint_best', '')}",
                "- missing_factorial_cells: "
                f"{interaction.get('missing_factorial_cells_count', '')}",
                "",
            ]
        )
    claim_readiness = report.get("claim_readiness", {})
    if claim_readiness:
        lines.extend(
            [
                "## Claim Readiness",
                "",
                f"- claim readiness ok: {claim_readiness.get('ok', '')}",
                f"- required: {report.get('require_claim_readiness', False)}",
                "- ready claims: "
                f"{','.join(claim_readiness.get('ready_claims', []))}",
                "- blocked claims: "
                f"{','.join(claim_readiness.get('blocked_claims', []))}",
                "",
            ]
        )
    if best_paired:
        lines.extend(
            [
                "## Best Prompt-Paired Config",
                "",
                f"- gamma: {best_paired.get('gamma', '')}",
                f"- drafter_freq_hz: {best_paired.get('drafter_freq_hz', '')}",
                f"- verifier_clock_mhz: {best_paired.get('verifier_clock_mhz', '')}",
                f"- paired_prompts: {best_paired.get('paired_prompts', '')}",
                "- mean_energy_savings_pct_vs_baseline: "
                f"{best_paired.get('mean_energy_savings_pct_vs_baseline', '')}",
                "- median_energy_savings_pct_vs_baseline: "
                f"{best_paired.get('median_energy_savings_pct_vs_baseline', '')}",
                "- ci95_energy_savings_pct_vs_baseline: "
                f"{best_paired.get('ci95_energy_savings_pct_vs_baseline', '')}",
                "- bootstrap_ci95_energy_savings_pct_vs_baseline: "
                f"[{best_paired.get('bootstrap_ci95_low_energy_savings_pct_vs_baseline', '')}, "
                f"{best_paired.get('bootstrap_ci95_high_energy_savings_pct_vs_baseline', '')}]",
                "- sign_test_p_value_energy_savings: "
                f"{best_paired.get('sign_test_p_value_energy_savings', '')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plan_pair_key(plan: Dict[str, object], combo: Dict[str, object]) -> PlanPairKey:
    return (
        str(plan.get("prompt_set_sha256", "")),
        str(combo.get("prompt_id", "")),
        str(combo.get("prompt_sha256", "")),
        str(combo.get("verifier_clock_mhz", "")),
        str(plan.get("decoding_mode", "greedy")),
        str(plan.get("max_new_tokens", "")),
        _plan_stop_token_key(plan),
    )


def _key_dict(key: PlanPairKey) -> Dict[str, str]:
    return {
        "prompt_set_sha256": key[0],
        "prompt_id": key[1],
        "prompt_sha256": key[2],
        "verifier_clock_mhz": key[3],
        "decoding_mode": key[4],
        "max_new_tokens": key[5],
        "stop_token_key": key[6],
    }


def _plan_stop_token_key(plan: Dict[str, object]) -> str:
    return (
        f"{str(plan.get('stop_token_policy', ''))}:"
        f"{str(plan.get('stop_token_ids', ''))}"
    )


def _plan_system_key(
    plan: Dict[str, object],
    combo: Dict[str, object],
) -> PlanSystemKey:
    return (
        str(plan.get("prompt_set_sha256", "")),
        str(combo.get("prompt_id", "")),
        str(combo.get("prompt_sha256", "")),
        str(combo.get("drafter_freq_hz", "")),
        str(combo.get("verifier_clock_mhz", "")),
        str(plan.get("decoding_mode", "greedy")),
        str(plan.get("max_new_tokens", "")),
        _plan_stop_token_key(plan),
    )


def _system_key_dict(key: PlanSystemKey) -> Dict[str, str]:
    return {
        "prompt_set_sha256": key[0],
        "prompt_id": key[1],
        "prompt_sha256": key[2],
        "drafter_freq_hz": key[3],
        "verifier_clock_mhz": key[4],
        "decoding_mode": key[5],
        "max_new_tokens": key[6],
        "stop_token_key": key[7],
    }


def _spec_gamma_balance_key(
    plan: Dict[str, object],
    combo: Dict[str, object],
) -> SpecGammaBalanceKey:
    return (
        str(plan.get("prompt_set_sha256", "")),
        str(combo.get("prompt_id", "")),
        str(combo.get("prompt_sha256", "")),
        str(combo.get("drafter_freq_hz", "")),
        str(combo.get("verifier_clock_mhz", "")),
        str(plan.get("decoding_mode", "greedy")),
        str(plan.get("max_new_tokens", "")),
        _plan_stop_token_key(plan),
    )


def _spec_gamma_balance_key_dict(key: SpecGammaBalanceKey) -> Dict[str, str]:
    return {
        "prompt_set_sha256": key[0],
        "prompt_id": key[1],
        "prompt_sha256": key[2],
        "drafter_freq_hz": key[3],
        "verifier_clock_mhz": key[4],
        "decoding_mode": key[5],
        "max_new_tokens": key[6],
        "stop_token_key": key[7],
    }


def _spec_factorial_key(
    plan: Dict[str, object],
    combo: Dict[str, object],
) -> SpecFactorialKey:
    return (
        str(plan.get("prompt_set_sha256", "")),
        str(combo.get("prompt_id", "")),
        str(combo.get("prompt_sha256", "")),
        str(plan.get("decoding_mode", "greedy")),
        str(plan.get("max_new_tokens", "")),
        str(plan.get("tokenizer", "")),
        _plan_stop_token_key(plan),
    )


def _spec_factorial_key_dict(key: SpecFactorialKey) -> Dict[str, str]:
    return {
        "prompt_set_sha256": key[0],
        "prompt_id": key[1],
        "prompt_sha256": key[2],
        "decoding_mode": key[3],
        "max_new_tokens": key[4],
        "tokenizer": key[5],
        "stop_token_key": key[6],
    }


def _plan_level_values(values: object) -> List[str]:
    if isinstance(values, list):
        levels = ["" if value is None else str(value) for value in values]
    elif values in ("", None):
        levels = [""]
    else:
        levels = [str(values)]
    return sorted(set(levels))


def _gamma_design_key(row: Dict[str, str]) -> GammaDesignKey:
    return (
        row.get("drafter_freq_hz", ""),
        row.get("verifier_clock_mhz", ""),
        row.get("decoding_mode", "greedy"),
        row.get("max_new_tokens", ""),
        row.get("stop_token_policy", ""),
        row.get("stop_token_ids", ""),
        row.get("prompt_set_sha256", ""),
        row.get("drafter_model", ""),
        row.get("verifier_model", ""),
        row.get("drafter_runtime_fingerprint", ""),
        row.get("verifier_runtime_fingerprint", ""),
    )


def _gamma_key_dict(key: GammaDesignKey) -> Dict[str, str]:
    return {
        "drafter_freq_hz": key[0],
        "verifier_clock_mhz": key[1],
        "decoding_mode": key[2],
        "max_new_tokens": key[3],
        "stop_token_policy": key[4],
        "stop_token_ids": key[5],
        "prompt_set_sha256": key[6],
        "drafter_model": key[7],
        "verifier_model": key[8],
        "drafter_runtime_fingerprint": key[9],
        "verifier_runtime_fingerprint": key[10],
    }


def plan_design_report(
    plans: List[Dict[str, object]],
    allow_unpaired: bool = False,
    min_prompts: int = 1,
    min_runs: int = 1,
    min_gammas: int = 2,
    require_two_device_boundary: bool = False,
) -> Dict[str, object]:
    min_required_prompts = max(1, int(min_prompts))
    min_required_runs = max(1, int(min_runs))
    min_required_gammas = max(2, int(min_gammas))
    spec_keys: Set[PlanPairKey] = set()
    baseline_keys: Set[PlanPairKey] = set()
    spec_system_keys: Set[PlanSystemKey] = set()
    two_device_baseline_system_keys: Set[PlanSystemKey] = set()
    spec_runs_by_pair: Dict[PlanPairKey, Set[int]] = defaultdict(set)
    baseline_runs_by_pair: Dict[PlanPairKey, Set[int]] = defaultdict(set)
    spec_warmups_by_pair: Dict[PlanPairKey, Set[int]] = defaultdict(set)
    baseline_warmups_by_pair: Dict[PlanPairKey, Set[int]] = defaultdict(set)
    algorithms = []
    schema_versions = sorted(
        {str(plan.get("schema_version", "")) for plan in plans if plan.get("schema_version", "")}
    )
    missing_schema_version_plans = sum(
        1 for plan in plans if not str(plan.get("schema_version", ""))
    )
    tokenizers = sorted(
        {str(plan.get("tokenizer", "")) for plan in plans if plan.get("tokenizer", "")}
    )
    missing_tokenizer_plans = sum(
        1 for plan in plans if not str(plan.get("tokenizer", ""))
    )
    missing_prompt_metadata_plans = []
    duplicate_prompt_hash_plans = []
    insufficient_unique_prompt_plans = []
    measurement_schedule_plans = 0
    missing_measurement_schedules = []
    invalid_measurement_schedules = []
    warmup_schedule_plans = 0
    missing_warmup_schedules = []
    invalid_warmup_schedules = []
    blocked_warmup_schedules = []
    missing_warmup_plans = []
    nonpositive_warmup_plans = []
    insufficient_measured_run_plans = []
    unrandomized_run_order_plans = []
    missing_randomization_seed_plans = []
    blocked_measurement_schedules = []
    spec_gamma_values: Set[str] = set()
    spec_gamma_groups: Dict[SpecGammaBalanceKey, Dict[str, int]] = defaultdict(dict)
    spec_factorial_observed: Dict[SpecFactorialKey, Set[Tuple[str, str, str]]] = defaultdict(set)
    spec_factorial_expected: Dict[SpecFactorialKey, Tuple[List[str], List[str], List[str]]] = {}

    for plan in plans:
        algorithm = str(plan.get("algorithm", "speculative"))
        algorithms.append(algorithm)
        declared_gammas = _plan_level_values(plan.get("gammas", []))
        declared_drafter_freqs = _plan_level_values(plan.get("drafter_freqs_hz", [""]))
        declared_verifier_clocks = _plan_level_values(
            plan.get("verifier_clocks_mhz", [""])
        )
        prompts = plan.get("prompts", [])
        if not isinstance(prompts, list) or not prompts:
            missing_prompt_metadata_plans.append({"algorithm": algorithm})
        else:
            prompt_hashes = [
                str(prompt.get("prompt_sha256", ""))
                for prompt in prompts
                if isinstance(prompt, dict) and str(prompt.get("prompt_sha256", ""))
            ]
            unique_prompt_hashes = set(prompt_hashes)
            duplicate_prompt_hashes = sorted(
                {
                    prompt_hash
                    for prompt_hash in unique_prompt_hashes
                    if prompt_hashes.count(prompt_hash) > 1
                }
            )
            if duplicate_prompt_hashes:
                duplicate_prompt_hash_plans.append(
                    {
                        "algorithm": algorithm,
                        "duplicate_prompt_sha256": duplicate_prompt_hashes,
                    }
                )
            if len(unique_prompt_hashes) < min_required_prompts:
                insufficient_unique_prompt_plans.append(
                    {
                        "algorithm": algorithm,
                        "unique_prompt_hashes": len(unique_prompt_hashes),
                        "min_prompts": min_required_prompts,
                    }
                )
        schedule = plan.get("measurement_schedule", [])
        warmup_schedule = plan.get("warmup_schedule", [])
        warmup_runs = _int_value(plan, "warmup_runs") if "warmup_runs" in plan else 0
        expected_schedule_len = sum(
            int(combo.get("measured_runs", plan.get("measured_runs", 1)))
            for combo in plan.get("combinations", [])
        )
        expected_warmup_len = len(plan.get("combinations", [])) * max(0, warmup_runs)
        if expected_schedule_len > 0:
            if "warmup_runs" not in plan:
                missing_warmup_plans.append(
                    {
                        "algorithm": algorithm,
                        "expected_entries": expected_schedule_len,
                    }
                )
            elif _int_value(plan, "warmup_runs") <= 0:
                nonpositive_warmup_plans.append(
                    {
                        "algorithm": algorithm,
                        "expected_entries": expected_schedule_len,
                        "warmup_runs": plan.get("warmup_runs"),
                    }
                )
        if expected_warmup_len > 1 and not warmup_schedule:
            missing_warmup_schedules.append(
                {
                    "algorithm": algorithm,
                    "expected_entries": expected_warmup_len,
                }
            )
        if expected_schedule_len > 1 and not schedule:
            missing_measurement_schedules.append(
                {
                    "algorithm": algorithm,
                    "expected_entries": expected_schedule_len,
                }
            )
        if expected_schedule_len > 1 and not bool(plan.get("shuffle_runs", False)):
            unrandomized_run_order_plans.append(
                {
                    "algorithm": algorithm,
                    "expected_entries": expected_schedule_len,
                }
            )
        if (
            expected_schedule_len > 1
            and bool(plan.get("shuffle_runs", False))
            and "seed" not in plan
        ):
            missing_randomization_seed_plans.append(
                {
                    "algorithm": algorithm,
                    "expected_entries": expected_schedule_len,
                }
            )
        if schedule:
            measurement_schedule_plans += 1
            orders = [
                int(item.get("order", 0))
                for item in schedule
                if isinstance(item, dict)
            ]
            invalid_reasons = []
            if len(schedule) != expected_schedule_len:
                invalid_reasons.append("wrong_schedule_length")
            if sorted(orders) != list(range(1, expected_schedule_len + 1)):
                invalid_reasons.append("noncontiguous_schedule_order")
            schedule_pairs = []
            for item in schedule:
                if not isinstance(item, dict):
                    invalid_reasons.append("non_object_schedule_entry")
                    continue
                condition_order = _int_value(item, "condition_order")
                if condition_order < 0 or condition_order >= len(plan.get("combinations", [])):
                    invalid_reasons.append("condition_order_out_of_range")
                    continue
                run_index = _int_value(item, "run")
                combo = plan["combinations"][condition_order]
                measured_runs = int(combo.get("measured_runs", plan.get("measured_runs", 1)))
                if run_index <= 0 or run_index > measured_runs:
                    invalid_reasons.append("run_index_out_of_range")
                schedule_pairs.append((condition_order, run_index))
            condition_count = len(plan.get("combinations", []))
            if (
                expected_schedule_len > 1
                and condition_count > 1
                and bool(plan.get("shuffle_runs", False))
                and schedule_pairs == sorted(schedule_pairs)
            ):
                blocked_measurement_schedules.append(
                    {
                        "algorithm": algorithm,
                        "expected_entries": expected_schedule_len,
                        "conditions": condition_count,
                    }
                )
            if invalid_reasons:
                invalid_measurement_schedules.append(
                    {
                        "algorithm": algorithm,
                        "expected_entries": expected_schedule_len,
                        "observed_entries": len(schedule),
                        "reasons": sorted(set(invalid_reasons)),
                    }
                )
        if warmup_schedule:
            warmup_schedule_plans += 1
            warmup_orders = [
                int(item.get("order", 0))
                for item in warmup_schedule
                if isinstance(item, dict)
            ]
            invalid_reasons = []
            if len(warmup_schedule) != expected_warmup_len:
                invalid_reasons.append("wrong_warmup_schedule_length")
            if sorted(warmup_orders) != list(range(1, expected_warmup_len + 1)):
                invalid_reasons.append("noncontiguous_warmup_schedule_order")
            warmup_pairs = []
            for item in warmup_schedule:
                if not isinstance(item, dict):
                    invalid_reasons.append("non_object_warmup_schedule_entry")
                    continue
                condition_order = _int_value(item, "condition_order")
                if condition_order < 0 or condition_order >= len(plan.get("combinations", [])):
                    invalid_reasons.append("warmup_condition_order_out_of_range")
                    continue
                warmup_index = _int_value(item, "warmup")
                if warmup_index <= 0 or warmup_index > warmup_runs:
                    invalid_reasons.append("warmup_index_out_of_range")
                warmup_pairs.append((condition_order, warmup_index))
            condition_count = len(plan.get("combinations", []))
            if (
                expected_warmup_len > 1
                and condition_count > 1
                and bool(plan.get("shuffle_runs", False))
                and warmup_pairs == sorted(warmup_pairs)
            ):
                blocked_warmup_schedules.append(
                    {
                        "algorithm": algorithm,
                        "expected_entries": expected_warmup_len,
                        "conditions": condition_count,
                    }
                )
            if invalid_reasons:
                invalid_warmup_schedules.append(
                    {
                        "algorithm": algorithm,
                        "expected_entries": expected_warmup_len,
                        "observed_entries": len(warmup_schedule),
                        "reasons": sorted(set(invalid_reasons)),
                    }
                )
        target = baseline_keys if algorithm == "verifier_only" else spec_keys
        target_runs = (
            baseline_runs_by_pair if algorithm == "verifier_only" else spec_runs_by_pair
        )
        target_warmups = (
            baseline_warmups_by_pair
            if algorithm == "verifier_only"
            else spec_warmups_by_pair
        )
        is_two_device_baseline = (
            algorithm == "verifier_only"
            and str(plan.get("system_boundary", "")) == "two_device_idle_drafter"
        )
        for combo in plan.get("combinations", []):
            pair_key = _plan_pair_key(plan, combo)
            target.add(pair_key)
            if algorithm == "verifier_only":
                if is_two_device_baseline:
                    two_device_baseline_system_keys.add(_plan_system_key(plan, combo))
            else:
                spec_system_keys.add(_plan_system_key(plan, combo))
            measured_runs = int(combo.get("measured_runs", plan.get("measured_runs", 1)))
            if measured_runs < min_required_runs:
                insufficient_measured_run_plans.append(
                    {
                        "algorithm": algorithm,
                        "prompt_id": str(combo.get("prompt_id", "")),
                        "prompt_sha256": str(combo.get("prompt_sha256", "")),
                        "gamma": str(combo.get("gamma", "")),
                        "drafter_freq_hz": str(combo.get("drafter_freq_hz", "")),
                        "verifier_clock_mhz": str(combo.get("verifier_clock_mhz", "")),
                        "measured_runs": measured_runs,
                        "min_runs": min_required_runs,
                    }
                )
            target_runs[pair_key].add(measured_runs)
            if "warmup_runs" in plan:
                target_warmups[pair_key].add(_int_value(plan, "warmup_runs"))
            if algorithm != "verifier_only":
                gamma = str(combo.get("gamma", ""))
                factorial_key = _spec_factorial_key(plan, combo)
                spec_factorial_expected.setdefault(
                    factorial_key,
                    (
                        declared_gammas,
                        declared_drafter_freqs,
                        declared_verifier_clocks,
                    ),
                )
                spec_factorial_observed[factorial_key].add(
                    (
                        str(combo.get("gamma", "")),
                        "" if combo.get("drafter_freq_hz") is None else str(
                            combo.get("drafter_freq_hz", "")
                        ),
                        "" if combo.get("verifier_clock_mhz") is None else str(
                            combo.get("verifier_clock_mhz", "")
                        ),
                    )
                )
                if gamma:
                    spec_gamma_values.add(gamma)
                    spec_gamma_groups[_spec_gamma_balance_key(plan, combo)][gamma] = int(
                        combo.get("measured_runs", plan.get("measured_runs", 1))
                    )

    missing_baselines = sorted(spec_keys - baseline_keys)
    missing_two_device_baselines = (
        sorted(spec_system_keys - two_device_baseline_system_keys)
        if two_device_baseline_system_keys
        else []
    )
    incomplete_gamma_groups = []
    nonuniform_gamma_run_groups = []
    for key, runs_by_gamma in sorted(spec_gamma_groups.items()):
        missing_gamma_values = sorted(spec_gamma_values - set(runs_by_gamma))
        if missing_gamma_values:
            detail = _spec_gamma_balance_key_dict(key)
            detail["missing_gamma_values"] = missing_gamma_values
            detail["present_gamma_values"] = sorted(runs_by_gamma)
            incomplete_gamma_groups.append(detail)
        if len(set(runs_by_gamma.values())) > 1:
            detail = _spec_gamma_balance_key_dict(key)
            detail["measured_runs_by_gamma"] = {
                gamma: runs_by_gamma[gamma] for gamma in sorted(runs_by_gamma)
            }
            nonuniform_gamma_run_groups.append(detail)

    incomplete_spec_factorial_groups = []
    complete_spec_factorial_groups = 0
    for key, expected_levels in sorted(spec_factorial_expected.items()):
        gammas, drafter_freqs, verifier_clocks = expected_levels
        observed = spec_factorial_observed.get(key, set())
        missing_cells = []
        for gamma in gammas:
            for drafter_freq in drafter_freqs:
                for verifier_clock in verifier_clocks:
                    cell = (gamma, drafter_freq, verifier_clock)
                    if cell not in observed:
                        missing_cells.append(
                            {
                                "gamma": gamma,
                                "drafter_freq_hz": drafter_freq,
                                "verifier_clock_mhz": verifier_clock,
                            }
                        )
        if missing_cells:
            detail = _spec_factorial_key_dict(key)
            detail["expected_cells"] = len(gammas) * len(drafter_freqs) * len(
                verifier_clocks
            )
            detail["observed_cells"] = len(observed)
            detail["gamma_values"] = gammas
            detail["drafter_freq_hz_values"] = drafter_freqs
            detail["verifier_clock_mhz_values"] = verifier_clocks
            detail["missing_cells"] = missing_cells[:50]
            detail["missing_cell_count"] = len(missing_cells)
            incomplete_spec_factorial_groups.append(detail)
        else:
            complete_spec_factorial_groups += 1

    errors = []
    if not spec_keys:
        errors.append("no_speculative_plan_conditions")
    if not baseline_keys:
        errors.append("no_verifier_baseline_plan_conditions")
    if missing_baselines and not allow_unpaired:
        errors.append("baseline_plan_does_not_cover_all_speculative_conditions")
    if (
        require_two_device_boundary
        and spec_system_keys
        and not two_device_baseline_system_keys
        and not allow_unpaired
    ):
        errors.append("two_device_boundary_requires_idle_drafter_baseline_plan")
    if missing_two_device_baselines and not allow_unpaired:
        errors.append("two_device_baseline_missing_speculative_frequency_conditions")
    if len(spec_gamma_values) < min_required_gammas:
        errors.append("insufficient_plan_gamma_values")
    baseline_run_mismatches = []
    baseline_warmup_mismatches = []
    for key in sorted(spec_keys & baseline_keys):
        spec_runs = spec_runs_by_pair.get(key, set())
        baseline_runs = baseline_runs_by_pair.get(key, set())
        if spec_runs and baseline_runs and spec_runs != baseline_runs:
            detail = _key_dict(key)
            detail["spec_measured_runs"] = sorted(spec_runs)
            detail["baseline_measured_runs"] = sorted(baseline_runs)
            baseline_run_mismatches.append(detail)

        spec_warmups = spec_warmups_by_pair.get(key, set())
        baseline_warmups = baseline_warmups_by_pair.get(key, set())
        if spec_warmups and baseline_warmups and spec_warmups != baseline_warmups:
            detail = _key_dict(key)
            detail["spec_warmup_runs"] = sorted(spec_warmups)
            detail["baseline_warmup_runs"] = sorted(baseline_warmups)
            baseline_warmup_mismatches.append(detail)
    if len(spec_gamma_values) < 2:
        errors.append("speculative_plan_must_sweep_at_least_two_gamma_values")
    if spec_gamma_values and "1" not in spec_gamma_values:
        errors.append("gamma_sweep_requires_gamma_one_baseline")
    if len(schema_versions) > 1:
        errors.append("mixed_plan_schema_versions")
    if missing_schema_version_plans:
        errors.append("missing_plan_schema_version")
    if len(tokenizers) > 1:
        errors.append("mixed_tokenizers_between_plans")
    if missing_tokenizer_plans:
        errors.append("missing_plan_tokenizer")
    if missing_prompt_metadata_plans:
        errors.append("missing_plan_prompt_metadata")
    if duplicate_prompt_hash_plans:
        errors.append("duplicate_plan_prompt_hashes")
    if insufficient_unique_prompt_plans:
        errors.append("insufficient_unique_plan_prompts")
    if incomplete_gamma_groups:
        errors.append("incomplete_gamma_factorial")
    if nonuniform_gamma_run_groups:
        errors.append("nonuniform_gamma_measured_runs")
    if incomplete_spec_factorial_groups:
        errors.append("incomplete_spec_factorial_grid")
    if missing_measurement_schedules:
        errors.append("missing_measurement_schedule")
    if invalid_measurement_schedules:
        errors.append("invalid_measurement_schedule")
    if missing_warmup_schedules:
        errors.append("missing_warmup_schedule")
    if invalid_warmup_schedules:
        errors.append("invalid_warmup_schedule")
    if blocked_warmup_schedules:
        errors.append("blocked_warmup_schedule")
    if missing_warmup_plans:
        errors.append("missing_warmup_runs")
    if nonpositive_warmup_plans:
        errors.append("measured_plan_requires_warmup")
    if insufficient_measured_run_plans:
        errors.append("insufficient_plan_measured_runs")
    if baseline_run_mismatches:
        errors.append("baseline_measured_runs_mismatch")
    if baseline_warmup_mismatches:
        errors.append("baseline_warmup_runs_mismatch")
    if unrandomized_run_order_plans:
        errors.append("measurement_runs_must_be_randomized")
    if missing_randomization_seed_plans:
        errors.append("missing_randomization_seed")
    if blocked_measurement_schedules:
        errors.append("blocked_measurement_schedule")

    return {
        "ok": not errors,
        "algorithms": sorted(set(algorithms)),
        "schema_versions": schema_versions,
        "missing_schema_version_plans": missing_schema_version_plans,
        "tokenizers": tokenizers,
        "missing_tokenizer_plans": missing_tokenizer_plans,
        "missing_prompt_metadata_plans": missing_prompt_metadata_plans,
        "duplicate_prompt_hash_plans": duplicate_prompt_hash_plans,
        "insufficient_unique_prompt_plans": insufficient_unique_prompt_plans,
        "min_prompts": min_required_prompts,
        "min_runs": min_required_runs,
        "min_gammas": min_required_gammas,
        "require_two_device_boundary": require_two_device_boundary,
        "measurement_schedule_plans": measurement_schedule_plans,
        "missing_measurement_schedules": missing_measurement_schedules,
        "invalid_measurement_schedules": invalid_measurement_schedules,
        "warmup_schedule_plans": warmup_schedule_plans,
        "missing_warmup_schedules": missing_warmup_schedules,
        "invalid_warmup_schedules": invalid_warmup_schedules,
        "blocked_warmup_schedules": blocked_warmup_schedules,
        "missing_warmup_plans": missing_warmup_plans,
        "nonpositive_warmup_plans": nonpositive_warmup_plans,
        "insufficient_measured_run_plans": insufficient_measured_run_plans[:20],
        "baseline_run_mismatches": baseline_run_mismatches,
        "baseline_warmup_mismatches": baseline_warmup_mismatches,
        "unrandomized_run_order_plans": unrandomized_run_order_plans,
        "missing_randomization_seed_plans": missing_randomization_seed_plans,
        "blocked_measurement_schedules": blocked_measurement_schedules,
        "spec_gamma_values": sorted(spec_gamma_values),
        "has_gamma_one_baseline": "1" in spec_gamma_values,
        "spec_gamma_balance_groups": len(spec_gamma_groups),
        "incomplete_gamma_groups": incomplete_gamma_groups,
        "nonuniform_gamma_run_groups": nonuniform_gamma_run_groups,
        "spec_factorial_groups": len(spec_factorial_expected),
        "complete_spec_factorial_groups": complete_spec_factorial_groups,
        "incomplete_spec_factorial_groups": incomplete_spec_factorial_groups,
        "speculative_plan_conditions": len(spec_keys),
        "baseline_plan_conditions": len(baseline_keys),
        "speculative_system_conditions": len(spec_system_keys),
        "two_device_baseline_system_conditions": len(
            two_device_baseline_system_keys
        ),
        "missing_baseline_conditions": len(missing_baselines),
        "missing_baselines": [_key_dict(key) for key in missing_baselines],
        "missing_two_device_baseline_conditions": len(missing_two_device_baselines),
        "missing_two_device_baselines": [
            _system_key_dict(key) for key in missing_two_device_baselines
        ],
        "errors": errors,
        "allow_unpaired": allow_unpaired,
    }


def gamma_design_report(
    gamma_effect_rows: List[Dict[str, str]],
    min_prompts: int,
    min_gammas: int = 2,
) -> Dict[str, object]:
    groups: Dict[GammaDesignKey, List[Dict[str, str]]] = defaultdict(list)
    for row in gamma_effect_rows:
        groups[_gamma_design_key(row)].append(row)

    ready_configs = []
    multi_gamma_configs = 0
    min_gamma_configs = 0
    insufficient_gamma_value_groups = []
    incomplete_prompt_overlap_rows = 0
    insufficient_paired_prompt_rows = 0
    non_gamma_one_baseline_rows = []
    min_required_prompts = max(1, min_prompts)
    min_required_gammas = max(2, int(min_gammas))

    for key, rows in sorted(groups.items()):
        gamma_values = {
            row.get("gamma", "")
            for row in rows
            if row.get("gamma", "") not in ("", None)
        }
        gamma_count = max([_int_value(row, "gamma_count") for row in rows] or [0])
        has_multi_gamma = len(gamma_values) >= 2 and gamma_count >= 2
        if has_multi_gamma:
            multi_gamma_configs += 1
        has_enough_gammas = (
            len(gamma_values) >= min_required_gammas
            and gamma_count >= min_required_gammas
        )
        if has_enough_gammas:
            min_gamma_configs += 1
        else:
            detail = _gamma_key_dict(key)
            detail["gamma_count"] = str(gamma_count)
            detail["observed_gamma_values"] = ",".join(sorted(gamma_values))
            detail["min_gammas"] = str(min_required_gammas)
            insufficient_gamma_value_groups.append(detail)

        incomplete_rows = [
            row
            for row in rows
            if row.get("complete_prompt_overlap") != "1"
            or row.get("complete_prompt_overlap_vs_baseline_gamma") != "1"
        ]
        insufficient_rows = [
            row
            for row in rows
            if _int_value(row, "paired_prompts_vs_baseline_gamma")
            < min_required_prompts
        ]
        incomplete_prompt_overlap_rows += len(incomplete_rows)
        insufficient_paired_prompt_rows += len(insufficient_rows)
        non_gamma_one_rows = [
            row for row in rows if row.get("baseline_gamma", "") != "1"
        ]
        non_gamma_one_baseline_rows.extend(
            {
                **_gamma_key_dict(key),
                "gamma": row.get("gamma", ""),
                "baseline_gamma": row.get("baseline_gamma", ""),
            }
            for row in non_gamma_one_rows
        )

        if has_enough_gammas and not incomplete_rows and not insufficient_rows:
            ready_config = _gamma_key_dict(key)
            ready_config["gamma_count"] = str(gamma_count)
            ready_config["rows"] = str(len(rows))
            ready_config["min_paired_prompts_vs_baseline_gamma"] = str(
                min(
                    _int_value(row, "paired_prompts_vs_baseline_gamma")
                    for row in rows
                )
            )
            ready_configs.append(ready_config)

    errors = []
    if not gamma_effect_rows:
        errors.append("no_gamma_effect_rows")
    if multi_gamma_configs == 0:
        errors.append("no_multi_gamma_configs")
    if min_gamma_configs == 0:
        errors.append("insufficient_gamma_values")
    if incomplete_prompt_overlap_rows:
        errors.append("incomplete_gamma_prompt_overlap")
    if insufficient_paired_prompt_rows:
        errors.append("insufficient_gamma_paired_prompts")
    if non_gamma_one_baseline_rows:
        errors.append("gamma_effect_baseline_must_be_one")
    if not ready_configs:
        errors.append("no_gamma_ready_config")

    return {
        "ok": not errors,
        "gamma_effect_rows": len(gamma_effect_rows),
        "config_groups": len(groups),
        "multi_gamma_configs": multi_gamma_configs,
        "min_gamma_configs": min_gamma_configs,
        "ready_configs": len(ready_configs),
        "ready_config_details": ready_configs,
        "min_prompts": min_required_prompts,
        "min_gammas": min_required_gammas,
        "insufficient_gamma_value_groups": insufficient_gamma_value_groups[:20],
        "incomplete_prompt_overlap_rows": incomplete_prompt_overlap_rows,
        "insufficient_paired_prompt_rows": insufficient_paired_prompt_rows,
        "non_gamma_one_baseline_rows": len(non_gamma_one_baseline_rows),
        "non_gamma_one_baseline_details": non_gamma_one_baseline_rows[:20],
        "errors": errors,
    }


GAMMA_STAT_GROUPS = {
    "drafter_total": {
        "mean": "paired_mean_drafter_total_energy_change_pct_vs_baseline_gamma",
        "ci": "paired_ci95_drafter_total_energy_change_pct_vs_baseline_gamma",
        "bootstrap_low": "paired_bootstrap_ci95_low_drafter_total_energy_change_pct_vs_baseline_gamma",
        "bootstrap_high": "paired_bootstrap_ci95_high_drafter_total_energy_change_pct_vs_baseline_gamma",
        "sign_p": "paired_sign_test_p_value_drafter_total_energy_change_pct_vs_baseline_gamma",
    },
    "drafter_draft": {
        "mean": "paired_mean_drafter_draft_energy_change_pct_vs_baseline_gamma",
        "ci": "paired_ci95_drafter_draft_energy_change_pct_vs_baseline_gamma",
        "bootstrap_low": "paired_bootstrap_ci95_low_drafter_draft_energy_change_pct_vs_baseline_gamma",
        "bootstrap_high": "paired_bootstrap_ci95_high_drafter_draft_energy_change_pct_vs_baseline_gamma",
        "sign_p": "paired_sign_test_p_value_drafter_draft_energy_change_pct_vs_baseline_gamma",
    },
    "drafter_draft_per_draft_token": {
        "mean": "paired_mean_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
        "ci": "paired_ci95_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
        "bootstrap_low": "paired_bootstrap_ci95_low_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
        "bootstrap_high": "paired_bootstrap_ci95_high_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
        "sign_p": "paired_sign_test_p_value_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
    },
}

GAMMA_ACTIVE_STAT_GROUP = {
    "drafter_active": {
        "mean": "paired_mean_drafter_active_energy_change_pct_vs_baseline_gamma",
        "ci": "paired_ci95_drafter_active_energy_change_pct_vs_baseline_gamma",
        "bootstrap_low": "paired_bootstrap_ci95_low_drafter_active_energy_change_pct_vs_baseline_gamma",
        "bootstrap_high": "paired_bootstrap_ci95_high_drafter_active_energy_change_pct_vs_baseline_gamma",
        "sign_p": "paired_sign_test_p_value_drafter_active_energy_change_pct_vs_baseline_gamma",
    },
    "drafter_draft_active": {
        "mean": "paired_mean_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
        "ci": "paired_ci95_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
        "bootstrap_low": "paired_bootstrap_ci95_low_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
        "bootstrap_high": "paired_bootstrap_ci95_high_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
        "sign_p": "paired_sign_test_p_value_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
    },
    "drafter_draft_active_per_draft_token": {
        "mean": "paired_mean_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
        "ci": "paired_ci95_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
        "bootstrap_low": "paired_bootstrap_ci95_low_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
        "bootstrap_high": "paired_bootstrap_ci95_high_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
        "sign_p": "paired_sign_test_p_value_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
    }
}


def gamma_statistics_report(
    gamma_effect_rows: List[Dict[str, str]],
    require_active_stats: bool,
) -> Dict[str, object]:
    required_groups = dict(GAMMA_STAT_GROUPS)
    if require_active_stats:
        required_groups.update(GAMMA_ACTIVE_STAT_GROUP)

    missing_stats = []
    invalid_stats = []
    missing_directional_sign_tests = []
    checked_nonbaseline_rows = 0

    for row in gamma_effect_rows:
        gamma = row.get("gamma", "")
        baseline_gamma = row.get("baseline_gamma", "")
        if not gamma or gamma == baseline_gamma:
            continue

        checked_nonbaseline_rows += 1
        row_id = {
            "gamma": gamma,
            "baseline_gamma": baseline_gamma,
            "drafter_freq_hz": row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
            "prompt_set_sha256": row.get("prompt_set_sha256", ""),
        }
        for metric_name, keys in sorted(required_groups.items()):
            required_keys = [
                keys["mean"],
                keys["ci"],
                keys["bootstrap_low"],
                keys["bootstrap_high"],
            ]
            missing_keys = [
                key for key in required_keys if row.get(key, "") in ("", None)
            ]
            if missing_keys:
                missing_stats.append(
                    {
                        **row_id,
                        "metric": metric_name,
                        "missing": missing_keys,
                    }
                )
                continue

            mean_value = _float_value(row, keys["mean"])
            ci_value = _float_value(row, keys["ci"])
            low_value = _float_value(row, keys["bootstrap_low"])
            high_value = _float_value(row, keys["bootstrap_high"])
            if None in (mean_value, ci_value, low_value, high_value):
                invalid_stats.append(
                    {
                        **row_id,
                        "metric": metric_name,
                        "reason": "non_numeric_statistic",
                    }
                )
                continue
            assert mean_value is not None
            assert ci_value is not None
            assert low_value is not None
            assert high_value is not None

            if ci_value < 0 or low_value > high_value:
                invalid_stats.append(
                    {
                        **row_id,
                        "metric": metric_name,
                        "reason": "invalid_interval",
                        "ci95": f"{ci_value:.6f}",
                        "bootstrap_low": f"{low_value:.6f}",
                        "bootstrap_high": f"{high_value:.6f}",
                    }
                )

            sign_p_text = row.get(keys["sign_p"], "")
            if abs(mean_value) > 1e-9 and sign_p_text in ("", None):
                missing_directional_sign_tests.append(
                    {
                        **row_id,
                        "metric": metric_name,
                        "mean_change_pct": f"{mean_value:.6f}",
                        "missing": keys["sign_p"],
                    }
                )
                continue
            if sign_p_text not in ("", None):
                sign_p = _float_value(row, keys["sign_p"])
                if sign_p is None or sign_p < 0 or sign_p > 1:
                    invalid_stats.append(
                        {
                            **row_id,
                            "metric": metric_name,
                            "reason": "invalid_sign_test_p_value",
                            "sign_test_p_value": sign_p_text,
                        }
                    )

    errors = []
    if not gamma_effect_rows:
        errors.append("no_gamma_effect_rows")
    if checked_nonbaseline_rows == 0:
        errors.append("no_nonbaseline_gamma_statistics_rows")
    if missing_stats:
        errors.append("missing_gamma_statistics")
    if invalid_stats:
        errors.append("invalid_gamma_statistics")
    if missing_directional_sign_tests:
        errors.append("missing_directional_sign_test")

    return {
        "ok": not errors,
        "require_active_stats": require_active_stats,
        "checked_nonbaseline_rows": checked_nonbaseline_rows,
        "required_metrics": sorted(required_groups),
        "missing_stat_rows": len(missing_stats),
        "invalid_stat_rows": len(invalid_stats),
        "missing_directional_sign_test_rows": len(missing_directional_sign_tests),
        "missing_stats": missing_stats[:20],
        "invalid_stats": invalid_stats[:20],
        "missing_directional_sign_tests": missing_directional_sign_tests[:20],
        "errors": errors,
    }


def _summary_stat_key(mean_key: str, prefix: str) -> str:
    if not mean_key.startswith("mean_"):
        return ""
    return f"{prefix}_{mean_key[len('mean_'):]}"


def _summary_row_id(row: Dict[str, object]) -> Dict[str, str]:
    return {
        "algorithm": str(row.get("algorithm", "")),
        "gamma": str(row.get("gamma", "")),
        "drafter_freq_hz": str(row.get("drafter_freq_hz", "")),
        "verifier_clock_mhz": str(row.get("verifier_clock_mhz", "")),
        "decoding_mode": str(row.get("decoding_mode", "")),
        "max_new_tokens": str(row.get("max_new_tokens", "")),
        "prompt_set_sha256": str(row.get("prompt_set_sha256", "")),
    }


def measurement_stability_report(
    summary_rows: List[Dict[str, object]],
    energy_key: str,
    max_energy_cv: Optional[float] = None,
    max_latency_cv: Optional[float] = None,
) -> Dict[str, object]:
    specs = [
        (
            "energy",
            energy_key,
            _summary_stat_key(energy_key, "stdev"),
            _summary_stat_key(energy_key, "ci95"),
            max_energy_cv,
        ),
        (
            "latency",
            "mean_wall_latency_ms",
            "stdev_wall_latency_ms",
            "ci95_wall_latency_ms",
            max_latency_cv,
        ),
    ]
    observed_rows = []
    missing = []
    invalid = []
    violations = []
    checked = defaultdict(int)
    max_cv_by_metric: Dict[str, Optional[float]] = {
        "energy": None,
        "latency": None,
    }

    for row in summary_rows:
        row_id = _summary_row_id(row)
        row_as_strings = {key: str(value) for key, value in row.items()}
        for metric, mean_key, stdev_key, ci95_key, limit in specs:
            if (
                metric == "energy"
                and "drafter_" in mean_key
                and str(row.get("algorithm", "")) != "speculative"
            ):
                continue
            mean_value = _float_value(row_as_strings, mean_key)
            stdev_value = (
                _float_value(row_as_strings, stdev_key) if stdev_key else None
            )
            ci95_value = _float_value(row_as_strings, ci95_key) if ci95_key else None
            if mean_value is None or stdev_value is None:
                if limit is not None:
                    missing.append(
                        {
                            **row_id,
                            "metric": metric,
                            "mean_key": mean_key,
                            "stdev_key": stdev_key,
                        }
                    )
                continue

            if mean_value <= 0 or stdev_value < 0:
                if limit is not None:
                    invalid.append(
                        {
                            **row_id,
                            "metric": metric,
                            "mean": f"{mean_value:.6f}",
                            "stdev": f"{stdev_value:.6f}",
                        }
                    )
                continue

            cv = stdev_value / mean_value
            ci95_pct = (
                (ci95_value / mean_value * 100.0) if ci95_value is not None else None
            )
            checked[metric] += 1
            current_max = max_cv_by_metric[metric]
            max_cv_by_metric[metric] = cv if current_max is None else max(current_max, cv)
            record = {
                **row_id,
                "metric": metric,
                "mean_key": mean_key,
                "mean": f"{mean_value:.6f}",
                "stdev_key": stdev_key,
                "stdev": f"{stdev_value:.6f}",
                "cv": f"{cv:.6f}",
                "ci95_key": ci95_key,
                "ci95": "" if ci95_value is None else f"{ci95_value:.6f}",
                "ci95_pct_of_mean": (
                    "" if ci95_pct is None else f"{ci95_pct:.6f}"
                ),
                "cv_limit": "" if limit is None else f"{limit:.6f}",
            }
            observed_rows.append(record)
            if limit is not None and cv > limit:
                violations.append(record)

    energy_violations = [
        row for row in violations if row.get("metric") == "energy"
    ]
    latency_violations = [
        row for row in violations if row.get("metric") == "latency"
    ]
    errors = []
    if missing:
        errors.append("missing_stability_statistics")
    if invalid:
        errors.append("invalid_stability_statistics")
    if energy_violations:
        errors.append("energy_cv_exceeds_limit")
    if latency_violations:
        errors.append("latency_cv_exceeds_limit")

    return {
        "ok": not errors,
        "max_energy_cv": "" if max_energy_cv is None else f"{max_energy_cv:.6f}",
        "max_latency_cv": "" if max_latency_cv is None else f"{max_latency_cv:.6f}",
        "observed_max_energy_cv": (
            ""
            if max_cv_by_metric["energy"] is None
            else f"{max_cv_by_metric['energy']:.6f}"
        ),
        "observed_max_latency_cv": (
            ""
            if max_cv_by_metric["latency"] is None
            else f"{max_cv_by_metric['latency']:.6f}"
        ),
        "checked_energy_rows": checked["energy"],
        "checked_latency_rows": checked["latency"],
        "energy_cv_violations": len(energy_violations),
        "latency_cv_violations": len(latency_violations),
        "missing_stat_rows": len(missing),
        "invalid_stat_rows": len(invalid),
        "observed_rows": observed_rows,
        "violations": violations[:20],
        "missing": missing[:20],
        "invalid": invalid[:20],
        "errors": errors,
    }


def plan_integrity_report(
    plans: List[Dict[str, object]],
    raw_rows: List[Dict[str, str]],
) -> Dict[str, object]:
    expected_by_algorithm: Dict[str, str] = {}
    expected_design_by_algorithm: Dict[str, str] = {}
    missing_plan_hashes = []
    invalid_plan_hashes = []
    missing_plan_design_hashes = []
    invalid_plan_design_hashes = []
    duplicate_plan_algorithms = []

    for index, plan in enumerate(plans):
        algorithm = str(plan.get("algorithm", "speculative"))
        recorded_design = str(plan.get("plan_design_sha256", ""))
        computed_design = spec_driver.plan_design_sha256(plan)
        if not recorded_design:
            missing_plan_design_hashes.append(
                {
                    "plan_index": index,
                    "algorithm": algorithm,
                    "computed_plan_design_sha256": computed_design,
                }
            )
        elif recorded_design != computed_design:
            invalid_plan_design_hashes.append(
                {
                    "plan_index": index,
                    "algorithm": algorithm,
                    "recorded_plan_design_sha256": recorded_design,
                    "computed_plan_design_sha256": computed_design,
                }
            )

        recorded = str(plan.get("plan_sha256", ""))
        computed = spec_driver.plan_sha256(plan)
        if not recorded:
            missing_plan_hashes.append(
                {
                    "plan_index": index,
                    "algorithm": algorithm,
                    "computed_plan_sha256": computed,
                }
            )
        elif recorded != computed:
            invalid_plan_hashes.append(
                {
                    "plan_index": index,
                    "algorithm": algorithm,
                    "recorded_plan_sha256": recorded,
                    "computed_plan_sha256": computed,
                }
            )

        expected_hash = recorded or computed
        expected_design_hash = recorded_design or computed_design
        existing = expected_by_algorithm.get(algorithm)
        if existing is not None and existing != expected_hash:
            duplicate_plan_algorithms.append(
                {
                    "algorithm": algorithm,
                    "first_plan_sha256": existing,
                    "next_plan_sha256": expected_hash,
                }
            )
        expected_by_algorithm[algorithm] = expected_hash
        expected_design_by_algorithm[algorithm] = expected_design_hash

    missing_result_hashes = []
    mismatched_result_hashes = []
    inconsistent_driver_hashes = []
    missing_result_design_hashes = []
    mismatched_result_design_hashes = []
    inconsistent_driver_design_hashes = []
    unknown_algorithm_sessions = []
    checked_sessions = 0

    for row in validate_results.measured_rows(validate_results.session_rows(raw_rows)):
        checked_sessions += 1
        algorithm = row.get("algorithm", "speculative")
        plan_hash = row.get("plan_sha256", "")
        driver_hash = row.get("driver_plan_sha256", "")
        row_hash = plan_hash or driver_hash
        plan_design_hash = row.get("plan_design_sha256", "")
        driver_design_hash = row.get("driver_plan_design_sha256", "")
        row_design_hash = plan_design_hash or driver_design_hash
        row_id = {
            "session_id": row.get("session_id", ""),
            "algorithm": algorithm,
            "prompt_id": row.get("prompt_id", ""),
            "gamma": row.get("gamma", ""),
            "run": row.get("run", ""),
        }
        if plan_hash and driver_hash and plan_hash != driver_hash:
            inconsistent_driver_hashes.append(
                {
                    **row_id,
                    "plan_sha256": plan_hash,
                    "driver_plan_sha256": driver_hash,
                }
            )
        if not row_hash:
            missing_result_hashes.append(row_id)
        if plan_design_hash and driver_design_hash and plan_design_hash != driver_design_hash:
            inconsistent_driver_design_hashes.append(
                {
                    **row_id,
                    "plan_design_sha256": plan_design_hash,
                    "driver_plan_design_sha256": driver_design_hash,
                }
            )
        if not row_design_hash:
            missing_result_design_hashes.append(row_id)
        expected_hash = expected_by_algorithm.get(algorithm)
        expected_design_hash = expected_design_by_algorithm.get(algorithm)
        if expected_hash is None or expected_design_hash is None:
            unknown_algorithm_sessions.append(row_id)
            continue
        if row_hash and row_hash != expected_hash:
            mismatched_result_hashes.append(
                {
                    **row_id,
                    "result_plan_sha256": row_hash,
                    "expected_plan_sha256": expected_hash,
                }
            )
        if row_design_hash and row_design_hash != expected_design_hash:
            mismatched_result_design_hashes.append(
                {
                    **row_id,
                    "result_plan_design_sha256": row_design_hash,
                    "expected_plan_design_sha256": expected_design_hash,
                }
            )

    errors = []
    if missing_plan_design_hashes:
        errors.append("missing_plan_design_sha256")
    if invalid_plan_design_hashes:
        errors.append("invalid_plan_design_sha256")
    if missing_plan_hashes:
        errors.append("missing_plan_sha256")
    if invalid_plan_hashes:
        errors.append("invalid_plan_sha256")
    if duplicate_plan_algorithms:
        errors.append("duplicate_plan_algorithm_hashes")
    if missing_result_hashes:
        errors.append("missing_result_plan_sha256")
    if mismatched_result_hashes:
        errors.append("result_plan_hash_mismatch")
    if inconsistent_driver_hashes:
        errors.append("result_driver_plan_hash_mismatch")
    if missing_result_design_hashes:
        errors.append("missing_result_plan_design_sha256")
    if mismatched_result_design_hashes:
        errors.append("result_plan_design_hash_mismatch")
    if inconsistent_driver_design_hashes:
        errors.append("result_driver_plan_design_hash_mismatch")
    if unknown_algorithm_sessions:
        errors.append("result_algorithm_without_plan")

    return {
        "ok": not errors,
        "plans": len(plans),
        "checked_sessions": checked_sessions,
        "plan_hashes_by_algorithm": expected_by_algorithm,
        "plan_design_hashes_by_algorithm": expected_design_by_algorithm,
        "missing_plan_design_hashes": len(missing_plan_design_hashes),
        "invalid_plan_design_hashes": len(invalid_plan_design_hashes),
        "missing_plan_hashes": len(missing_plan_hashes),
        "invalid_plan_hashes": len(invalid_plan_hashes),
        "duplicate_plan_algorithms": len(duplicate_plan_algorithms),
        "missing_result_plan_hash_sessions": len(missing_result_hashes),
        "mismatched_result_plan_hash_sessions": len(mismatched_result_hashes),
        "inconsistent_driver_plan_hash_sessions": len(inconsistent_driver_hashes),
        "missing_result_plan_design_hash_sessions": len(missing_result_design_hashes),
        "mismatched_result_plan_design_hash_sessions": len(
            mismatched_result_design_hashes
        ),
        "inconsistent_driver_plan_design_hash_sessions": len(
            inconsistent_driver_design_hashes
        ),
        "unknown_algorithm_sessions": len(unknown_algorithm_sessions),
        "missing_design_plans": missing_plan_design_hashes[:20],
        "invalid_design_plans": invalid_plan_design_hashes[:20],
        "missing_plans": missing_plan_hashes[:20],
        "invalid_plans": invalid_plan_hashes[:20],
        "duplicate_plans": duplicate_plan_algorithms[:20],
        "missing_results": missing_result_hashes[:20],
        "mismatched_results": mismatched_result_hashes[:20],
        "inconsistent_driver_results": inconsistent_driver_hashes[:20],
        "missing_design_results": missing_result_design_hashes[:20],
        "mismatched_design_results": mismatched_result_design_hashes[:20],
        "inconsistent_driver_design_results": inconsistent_driver_design_hashes[:20],
        "unknown_algorithm_results": unknown_algorithm_sessions[:20],
        "errors": errors,
    }


def plan_audit_report(
    plans: List[Dict[str, object]],
    payloads: Sequence[Dict[str, object]],
    require_plan_audit: bool = False,
) -> Dict[str, object]:
    expected_design_hashes = {
        str(plan.get("algorithm", "speculative")): str(
            plan.get("plan_design_sha256", "")
        )
        or spec_driver.plan_design_sha256(plan)
        for plan in plans
    }
    invalid_schema_reports = []
    not_ok_reports = []
    missing_algorithm_reports = []
    mismatched_design_reports = []
    audited_algorithms = set()
    for index, payload in enumerate(payloads):
        if str(payload.get("schema_version", "")) not in (
            "xronos-plan-audit-v1",
            "xronos-plan-audit-v2",
        ):
            invalid_schema_reports.append(index)
        if payload.get("ok") is not True:
            not_ok_reports.append(index)
        plan_design_hashes = payload.get("plan_integrity", {}).get(
            "plan_design_hashes_by_algorithm",
            {},
        )
        if not isinstance(plan_design_hashes, dict):
            plan_design_hashes = {}
        audited_algorithms.update(str(key) for key in plan_design_hashes)
        for algorithm, expected_hash in expected_design_hashes.items():
            observed_hash = str(plan_design_hashes.get(algorithm, ""))
            if not observed_hash:
                missing_algorithm_reports.append(
                    {
                        "report_index": index,
                        "algorithm": algorithm,
                    }
                )
            elif observed_hash != expected_hash:
                mismatched_design_reports.append(
                    {
                        "report_index": index,
                        "algorithm": algorithm,
                        "expected_plan_design_sha256": expected_hash,
                        "observed_plan_design_sha256": observed_hash,
                    }
                )

    errors = []
    if require_plan_audit and not payloads:
        errors.append("missing_required_plan_audit")
    if invalid_schema_reports:
        errors.append("invalid_plan_audit_schema")
    if not_ok_reports:
        errors.append("plan_audit_not_ok")
    if require_plan_audit and missing_algorithm_reports:
        errors.append("plan_audit_missing_algorithm")
    if mismatched_design_reports:
        errors.append("plan_audit_design_hash_mismatch")

    return {
        "ok": not errors,
        "require_plan_audit": require_plan_audit,
        "plan_audit_reports": len(payloads),
        "expected_plan_design_hashes_by_algorithm": expected_design_hashes,
        "expected_algorithms": sorted(expected_design_hashes),
        "audited_algorithms": sorted(audited_algorithms),
        "invalid_schema_reports": len(invalid_schema_reports),
        "not_ok_reports": len(not_ok_reports),
        "missing_algorithm_reports": len(missing_algorithm_reports),
        "mismatched_design_reports": len(mismatched_design_reports),
        "invalid_schema_report_indices": invalid_schema_reports[:20],
        "not_ok_report_indices": not_ok_reports[:20],
        "missing_algorithm_details": missing_algorithm_reports[:20],
        "mismatched_design_details": mismatched_design_reports[:20],
        "errors": errors,
    }


def _doctor_payload(report: object) -> Dict[str, object]:
    if isinstance(report, dict):
        checks = report.get("checks", [])
        if isinstance(checks, list):
            return report
        return {**report, "checks": []}
    if isinstance(report, list):
        return {
            "role": "",
            "ok": not any(
                isinstance(check, dict) and check.get("status") == "fail"
                for check in report
            ),
            "checks": report,
        }
    return {"role": "", "ok": False, "checks": []}


def doctor_design_report(
    doctor_reports: List[object],
    require_doctor: bool = False,
    require_driver_doctor: bool = False,
) -> Dict[str, object]:
    payloads = [_doctor_payload(report) for report in doctor_reports]
    roles = sorted(
        {
            str(payload.get("role", ""))
            for payload in payloads
            if str(payload.get("role", ""))
        }
    )
    failures = 0
    warnings = 0
    failed_checks = []
    not_ok_reports = []
    for payload in payloads:
        role = str(payload.get("role", ""))
        if payload.get("ok") is False:
            not_ok_reports.append(role)
        checks = payload.get("checks", [])
        if not isinstance(checks, list):
            checks = []
        for check in checks:
            if not isinstance(check, dict):
                continue
            status = str(check.get("status", ""))
            if status == "fail":
                failures += 1
                failed_checks.append(
                    {
                        "role": role,
                        "name": str(check.get("name", "")),
                        "message": str(check.get("message", "")),
                    }
                )
            elif status == "warn":
                warnings += 1

    errors = []
    if require_doctor and not payloads:
        errors.append("missing_required_doctor_reports")
    if require_doctor and "drafter" not in roles:
        errors.append("missing_drafter_doctor_report")
    if require_doctor and "verifier" not in roles:
        errors.append("missing_verifier_doctor_report")
    if require_doctor and require_driver_doctor and "driver" not in roles:
        errors.append("missing_driver_doctor_report")
    if failures:
        errors.append("doctor_checks_failed")
    if not_ok_reports:
        errors.append("doctor_report_not_ok")

    return {
        "ok": not errors,
        "require_doctor": require_doctor,
        "require_driver_doctor": require_driver_doctor,
        "doctor_reports": len(payloads),
        "roles": roles,
        "failures": failures,
        "warnings": warnings,
        "failed_checks": failed_checks,
        "not_ok_reports": not_ok_reports,
        "errors": errors,
    }


def energy_design_report(
    plans: List[Dict[str, object]],
    summary_energy_key: str,
    paired_energy_key: str,
) -> Dict[str, object]:
    active_energy_keys = [
        key
        for key in (summary_energy_key, paired_energy_key)
        if "_active_" in key or key.startswith("active_")
    ]
    idle_baseline_requested = validate_results.plan_requires_idle(plans)
    idle_baseline_policies = sorted(
        {
            str(plan.get("idle_baseline_policy", ""))
            for plan in plans
            if float(plan.get("idle_baseline_s", 0) or 0) > 0
            and str(plan.get("idle_baseline_policy", ""))
        }
    )
    missing_idle_policy_plans = [
        {
            "algorithm": str(plan.get("algorithm", "")),
            "idle_baseline_s": str(plan.get("idle_baseline_s", "")),
        }
        for plan in plans
        if float(plan.get("idle_baseline_s", 0) or 0) > 0
        and not str(plan.get("idle_baseline_policy", ""))
    ]
    non_run_idle_policy_plans = [
        {
            "algorithm": str(plan.get("algorithm", "")),
            "idle_baseline_policy": str(plan.get("idle_baseline_policy", "")),
        }
        for plan in plans
        if float(plan.get("idle_baseline_s", 0) or 0) > 0
        and str(plan.get("idle_baseline_policy", "")) not in ("", "run")
    ]
    errors = []
    if active_energy_keys and not idle_baseline_requested:
        errors.append("active_energy_metrics_require_idle_baseline_plan")
    if active_energy_keys and missing_idle_policy_plans:
        errors.append("active_energy_metrics_require_idle_baseline_policy")
    if active_energy_keys and non_run_idle_policy_plans:
        errors.append("active_energy_metrics_require_run_idle_baseline_policy")
    return {
        "ok": not errors,
        "active_energy_keys": active_energy_keys,
        "requires_idle_baseline": bool(active_energy_keys),
        "requires_run_idle_policy": bool(active_energy_keys),
        "idle_baseline_requested": idle_baseline_requested,
        "idle_baseline_policies": idle_baseline_policies,
        "missing_idle_policy_plans": missing_idle_policy_plans,
        "non_run_idle_policy_plans": non_run_idle_policy_plans,
        "errors": errors,
    }


def schema_contract_report(
    plans: List[Dict[str, object]],
    raw_rows: List[Dict[str, str]],
) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    plan_schema_versions = sorted(
        {
            str(plan.get("schema_version", ""))
            for plan in plans
            if str(plan.get("schema_version", ""))
        }
    )
    result_schema_versions = sorted(
        {
            row.get("result_schema_version", "")
            for row in sessions
            if row.get("result_schema_version", "")
        }
    )
    missing_result_schema = [
        row.get("session_id", "")
        for row in sessions
        if not row.get("result_schema_version", "")
    ]
    missing_algorithm_version = [
        row.get("session_id", "")
        for row in sessions
        if not row.get("algorithm_version", "")
    ]
    missing_driver_result_schema = []
    driver_result_schema_mismatches = []
    missing_driver_rpc_schema = []
    missing_role_rpc_schema = []
    rpc_schema_mismatches = []

    algorithm_versions: Dict[str, Set[str]] = defaultdict(set)
    rpc_schema_versions: Dict[str, Set[str]] = defaultdict(set)
    unknown_algorithms = set()
    for row in sessions:
        algorithm = row.get("algorithm", "speculative")
        if algorithm not in ("speculative", "verifier_only"):
            unknown_algorithms.add(algorithm)
        version = row.get("algorithm_version", "")
        if version:
            algorithm_versions[algorithm].add(version)

        result_schema = row.get("result_schema_version", "")
        driver_result_schema = row.get("driver_result_schema_version", "")
        if not driver_result_schema:
            missing_driver_result_schema.append(row.get("session_id", ""))
        elif result_schema and driver_result_schema != result_schema:
            driver_result_schema_mismatches.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": algorithm,
                    "result_schema_version": result_schema,
                    "driver_result_schema_version": driver_result_schema,
                }
            )

        driver_rpc_schema = row.get("driver_spec_rpc_schema_version", "")
        roles = ["verifier"]
        if algorithm == "speculative":
            roles.append("drafter")
        if not driver_rpc_schema:
            missing_driver_rpc_schema.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": algorithm,
                    "role": "driver",
                }
            )
        else:
            rpc_schema_versions["driver"].add(driver_rpc_schema)
        for role in roles:
            server_rpc_schema = row.get(f"{role}_spec_rpc_schema_version", "")
            if not server_rpc_schema:
                missing_role_rpc_schema.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "role": role,
                    }
                )
                continue
            rpc_schema_versions[role].add(server_rpc_schema)
            if driver_rpc_schema and server_rpc_schema != driver_rpc_schema:
                rpc_schema_mismatches.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "role": role,
                        "driver_spec_rpc_schema_version": driver_rpc_schema,
                        f"{role}_spec_rpc_schema_version": server_rpc_schema,
                    }
                )

    mixed_algorithm_versions = {
        algorithm: sorted(versions)
        for algorithm, versions in sorted(algorithm_versions.items())
        if len(versions) > 1
    }

    errors = []
    if missing_result_schema:
        errors.append("missing_result_schema_version")
    if len(result_schema_versions) > 1:
        errors.append("mixed_result_schema_versions")
    if (
        plan_schema_versions
        and result_schema_versions
        and set(plan_schema_versions) != set(result_schema_versions)
    ):
        errors.append("plan_result_schema_version_mismatch")
    if missing_algorithm_version:
        errors.append("missing_algorithm_version")
    if mixed_algorithm_versions:
        errors.append("mixed_algorithm_versions")
    if unknown_algorithms:
        errors.append("unknown_algorithm")
    if missing_driver_result_schema:
        errors.append("missing_driver_result_schema_version")
    if driver_result_schema_mismatches:
        errors.append("driver_result_schema_version_mismatch")
    if missing_driver_rpc_schema:
        errors.append("missing_driver_spec_rpc_schema_version")
    if missing_role_rpc_schema:
        errors.append("missing_role_spec_rpc_schema_version")
    if rpc_schema_mismatches:
        errors.append("role_spec_rpc_schema_version_mismatch")

    return {
        "ok": not errors,
        "plan_schema_versions": plan_schema_versions,
        "result_schema_versions": result_schema_versions,
        "algorithm_versions": {
            algorithm: sorted(versions)
            for algorithm, versions in sorted(algorithm_versions.items())
        },
        "spec_rpc_schema_versions": {
            role: sorted(versions)
            for role, versions in sorted(rpc_schema_versions.items())
        },
        "spec_rpc_schema_versions_by_role": {
            role: sorted(versions)
            for role, versions in sorted(rpc_schema_versions.items())
        },
        "missing_result_schema_sessions": len(missing_result_schema),
        "missing_result_schema_session_ids": missing_result_schema[:20],
        "missing_algorithm_version_sessions": len(missing_algorithm_version),
        "missing_algorithm_version_session_ids": missing_algorithm_version[:20],
        "missing_driver_result_schema_sessions": len(missing_driver_result_schema),
        "missing_driver_result_schema_session_ids": missing_driver_result_schema[:20],
        "driver_result_schema_mismatch_sessions": len(driver_result_schema_mismatches),
        "driver_result_schema_mismatches": driver_result_schema_mismatches[:20],
        "missing_driver_spec_rpc_schema_sessions": len(
            {item["session_id"] for item in missing_driver_rpc_schema}
        ),
        "missing_driver_spec_rpc_schema": missing_driver_rpc_schema[:20],
        "missing_role_spec_rpc_schema_sessions": len(
            {item["session_id"] for item in missing_role_rpc_schema}
        ),
        "missing_role_spec_rpc_schema": missing_role_rpc_schema[:20],
        "role_spec_rpc_schema_mismatch_sessions": len(
            {item["session_id"] for item in rpc_schema_mismatches}
        ),
        "role_spec_rpc_schema_mismatches": rpc_schema_mismatches[:20],
        "missing_spec_rpc_schema_sessions": len(
            {
                item["session_id"]
                for item in missing_driver_rpc_schema + missing_role_rpc_schema
            }
        ),
        "missing_spec_rpc_schema": (
            missing_driver_rpc_schema + missing_role_rpc_schema
        )[:20],
        "spec_rpc_schema_mismatch_sessions": len(
            {item["session_id"] for item in rpc_schema_mismatches}
        ),
        "spec_rpc_schema_mismatches": rpc_schema_mismatches[:20],
        "mixed_algorithm_versions": mixed_algorithm_versions,
        "unknown_algorithms": sorted(unknown_algorithms),
        "errors": errors,
    }


def _int_text(value: str) -> str:
    if value in ("", None):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def frequency_consistency_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    mismatches = []
    missing_reported = []
    lock_failures = []

    checks = [
        (
            "drafter",
            "drafter_freq_hz",
            "drafter_jetson_gpu_freq_hz",
            "drafter_frequency_lock_ok",
        ),
        (
            "verifier",
            "verifier_clock_mhz",
            "verifier_gpu_clock_mhz",
            "verifier_frequency_lock_ok",
        ),
    ]
    for row in sessions:
        for role, requested_key, reported_key, lock_key in checks:
            requested = _int_text(row.get(requested_key, ""))
            if not requested:
                continue
            reported = _int_text(row.get(reported_key, ""))
            if not reported:
                missing_reported.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": row.get("algorithm", ""),
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                        "requested_key": requested_key,
                        "reported_key": reported_key,
                        "requested": requested,
                    }
                )
                continue
            if requested != reported:
                mismatches.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": row.get("algorithm", ""),
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                        "requested_key": requested_key,
                        "reported_key": reported_key,
                        "requested": requested,
                        "reported": reported,
                    }
                )
            if row.get(lock_key, "") == "0":
                lock_failures.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": row.get("algorithm", ""),
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                        "requested": requested,
                        "reported": reported,
                        "lock_key": lock_key,
                    }
                )

    errors = []
    if missing_reported:
        errors.append("missing_reported_frequency")
    if mismatches:
        errors.append("reported_frequency_mismatch")
    if lock_failures:
        errors.append("frequency_lock_failed")

    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "missing_reported_sessions": len(missing_reported),
        "mismatch_sessions": len(mismatches),
        "lock_failure_sessions": len(lock_failures),
        "missing_reported": missing_reported[:20],
        "mismatches": mismatches[:20],
        "lock_failures": lock_failures[:20],
        "errors": errors,
    }


def _positive_float_text(value: str) -> bool:
    if value in ("", None):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _host_identity(row: Dict[str, str], role: str) -> str:
    return row.get(f"{role}_node_name", "") or row.get(f"{role}_hostname", "")


def measurement_setup_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    missing_power_interval = []
    invalid_power_interval = []
    missing_primary_rail = []
    selected_primary_rail_mismatch = []
    same_host_sessions = []
    missing_host_identity = []
    missing_idle_baseline_policy = []
    invalid_idle_baseline_policy = []
    intervals_by_role: Dict[str, Set[str]] = defaultdict(set)
    primary_rails_by_role: Dict[str, Set[str]] = defaultdict(set)
    idle_baseline_policies: Set[str] = set()

    for row in sessions:
        algorithm = row.get("algorithm", "speculative")
        system_boundary = row.get("system_boundary", "")
        includes_idle_drafter = (
            algorithm != "speculative"
            and system_boundary == "two_device_idle_drafter"
        )
        if row.get("idle_baseline_s", ""):
            idle_policy = row.get("idle_baseline_policy", "")
            if not idle_policy:
                missing_idle_baseline_policy.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                    }
                )
            elif idle_policy not in ("condition", "run"):
                invalid_idle_baseline_policy.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "idle_baseline_policy": idle_policy,
                    }
                )
            else:
                idle_baseline_policies.add(idle_policy)

        roles = ["verifier"]
        if algorithm == "speculative" or includes_idle_drafter:
            roles.append("drafter")
        for role in roles:
            key = f"{role}_power_interval_s"
            value = row.get(key, "")
            if not value:
                missing_power_interval.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                    }
                )
                continue
            intervals_by_role[role].add(str(value))
            if not _positive_float_text(value):
                invalid_power_interval.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                        "power_interval_s": value,
                    }
                )
            rail_key = f"{role}_primary_power_rail"
            rail_value = row.get(rail_key, "")
            if not rail_value:
                missing_primary_rail.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                        "missing": rail_key,
                    }
                )
            else:
                primary_rails_by_role[role].add(str(rail_value))

        if not row.get("system_primary_power_rails", ""):
            missing_primary_rail.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": algorithm,
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "role": "system",
                    "missing": "system_primary_power_rails",
                }
            )

        selected_primary_rail = (
            row.get("verifier_primary_power_rail", "")
            if algorithm == "verifier_only"
            else row.get("drafter_primary_power_rail", "")
        )
        if selected_primary_rail and row.get("rail", "") != selected_primary_rail:
            selected_primary_rail_mismatch.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": algorithm,
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "rail": row.get("rail", ""),
                    "selected_primary_rail": selected_primary_rail,
                }
            )

        if algorithm == "speculative":
            drafter_host = _host_identity(row, "drafter")
            verifier_host = _host_identity(row, "verifier")
            if not drafter_host or not verifier_host:
                missing_host_identity.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "drafter_host": drafter_host,
                        "verifier_host": verifier_host,
                    }
                )
            elif drafter_host == verifier_host:
                same_host_sessions.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "drafter_host": drafter_host,
                        "verifier_host": verifier_host,
                    }
                )

    mixed_power_intervals = {
        role: sorted(values)
        for role, values in sorted(intervals_by_role.items())
        if len(values) > 1
    }
    mixed_idle_baseline_policies = sorted(idle_baseline_policies)

    errors = []
    if missing_power_interval:
        errors.append("missing_power_interval_metadata")
    if invalid_power_interval:
        errors.append("invalid_power_interval_metadata")
    if missing_primary_rail:
        errors.append("missing_primary_power_rail_metadata")
    if selected_primary_rail_mismatch:
        errors.append("selected_primary_power_rail_mismatch")
    if mixed_power_intervals:
        errors.append("mixed_power_intervals")
    if missing_host_identity:
        errors.append("missing_drafter_verifier_host_identity")
    if same_host_sessions:
        errors.append("drafter_verifier_same_host")
    if missing_idle_baseline_policy:
        errors.append("missing_idle_baseline_policy")
    if invalid_idle_baseline_policy:
        errors.append("invalid_idle_baseline_policy")
    if len(mixed_idle_baseline_policies) > 1:
        errors.append("mixed_idle_baseline_policies")

    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "power_intervals_by_role": {
            role: sorted(values) for role, values in sorted(intervals_by_role.items())
        },
        "primary_power_rails_by_role": {
            role: sorted(values) for role, values in sorted(primary_rails_by_role.items())
        },
        "missing_power_interval_sessions": len(missing_power_interval),
        "invalid_power_interval_sessions": len(invalid_power_interval),
        "missing_primary_rail_sessions": len(
            {item["session_id"] for item in missing_primary_rail}
        ),
        "selected_primary_rail_mismatch_sessions": len(
            {item["session_id"] for item in selected_primary_rail_mismatch}
        ),
        "mixed_power_intervals": mixed_power_intervals,
        "idle_baseline_policies": mixed_idle_baseline_policies,
        "same_drafter_verifier_host_sessions": len(same_host_sessions),
        "missing_host_identity_sessions": len(missing_host_identity),
        "missing_idle_baseline_policy_sessions": len(missing_idle_baseline_policy),
        "invalid_idle_baseline_policy_sessions": len(invalid_idle_baseline_policy),
        "missing_power_interval": missing_power_interval[:20],
        "invalid_power_interval": invalid_power_interval[:20],
        "missing_primary_rail": missing_primary_rail[:20],
        "selected_primary_rail_mismatch": selected_primary_rail_mismatch[:20],
        "missing_host_identity": missing_host_identity[:20],
        "same_drafter_verifier_host": same_host_sessions[:20],
        "missing_idle_baseline_policy": missing_idle_baseline_policy[:20],
        "invalid_idle_baseline_policy": invalid_idle_baseline_policy[:20],
        "errors": errors,
    }


def _truthy_value(value: str) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "y")


def _role_commit(row: Dict[str, str], role: str) -> str:
    if role == "driver":
        return row.get("driver_xronos_git_commit", "") or row.get(
            "driver_git_commit",
            "",
        )
    return row.get(f"{role}_xronos_git_commit", "")


def provenance_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    missing_git_commit = []
    role_commit_mismatches = []
    dirty_driver_sessions = []
    commits_by_role: Dict[str, Set[str]] = defaultdict(set)
    images_by_role: Dict[str, Set[str]] = defaultdict(set)

    for row in sessions:
        algorithm = row.get("algorithm", "speculative")
        roles = ["driver", "verifier"]
        if algorithm == "speculative":
            roles.append("drafter")

        row_commits = {}
        for role in roles:
            commit = _role_commit(row, role)
            if commit:
                row_commits[role] = commit
                commits_by_role[role].add(commit)
            else:
                missing_git_commit.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": algorithm,
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                    }
                )

            image = row.get(f"{role}_xronos_image", "")
            if image:
                images_by_role[role].add(image)

        if len(set(row_commits.values())) > 1:
            role_commit_mismatches.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": algorithm,
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "commits": row_commits,
                }
            )

        if _truthy_value(row.get("driver_git_dirty", "")):
            dirty_driver_sessions.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": algorithm,
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "driver_git_commit": row.get("driver_git_commit", ""),
                    "driver_xronos_git_commit": row.get(
                        "driver_xronos_git_commit",
                        "",
                    ),
                }
            )

    mixed_role_commits = {
        role: sorted(commits)
        for role, commits in sorted(commits_by_role.items())
        if len(commits) > 1
    }

    errors = []
    if missing_git_commit:
        errors.append("missing_git_commit_metadata")
    if role_commit_mismatches:
        errors.append("role_git_commit_mismatch")
    if mixed_role_commits:
        errors.append("mixed_role_git_commits")
    if dirty_driver_sessions:
        errors.append("driver_git_dirty")

    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "commits_by_role": {
            role: sorted(commits)
            for role, commits in sorted(commits_by_role.items())
        },
        "images_by_role": {
            role: sorted(images)
            for role, images in sorted(images_by_role.items())
        },
        "missing_git_commit_sessions": len(missing_git_commit),
        "role_commit_mismatch_sessions": len(role_commit_mismatches),
        "mixed_role_git_commits": mixed_role_commits,
        "dirty_driver_sessions": len(dirty_driver_sessions),
        "missing_git_commit": missing_git_commit[:20],
        "role_commit_mismatches": role_commit_mismatches[:20],
        "dirty_driver": dirty_driver_sessions[:20],
        "errors": errors,
    }


def model_setup_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    speculative_sessions = [
        row for row in sessions if row.get("algorithm", "speculative") == "speculative"
    ]
    missing_model_names = []
    same_model_sessions = []
    missing_model_sizes = []
    non_smaller_drafter = []
    model_pairs: Set[Tuple[str, str]] = set()
    model_size_pairs: Set[Tuple[str, str]] = set()

    for row in speculative_sessions:
        drafter_model = row.get("drafter_model", "")
        verifier_model = row.get("verifier_model", "")
        drafter_parameters = _float_value(row, "drafter_model_parameter_count")
        verifier_parameters = _float_value(row, "verifier_model_parameter_count")
        if not drafter_model or not verifier_model:
            missing_model_names.append(
                {
                    "session_id": row.get("session_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "drafter_model": drafter_model,
                    "verifier_model": verifier_model,
                }
            )
            continue

        model_pairs.add((drafter_model, verifier_model))
        if drafter_model == verifier_model:
            same_model_sessions.append(
                {
                    "session_id": row.get("session_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "model": drafter_model,
                }
            )

        if drafter_parameters is None or verifier_parameters is None:
            missing_model_sizes.append(
                {
                    "session_id": row.get("session_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "drafter_model": drafter_model,
                    "verifier_model": verifier_model,
                    "drafter_model_parameter_count": row.get(
                        "drafter_model_parameter_count",
                        "",
                    ),
                    "verifier_model_parameter_count": row.get(
                        "verifier_model_parameter_count",
                        "",
                    ),
                }
            )
        else:
            model_size_pairs.add(
                (
                    str(int(drafter_parameters)),
                    str(int(verifier_parameters)),
                )
            )
            if drafter_parameters >= verifier_parameters:
                non_smaller_drafter.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "drafter_model": drafter_model,
                        "verifier_model": verifier_model,
                        "drafter_model_parameter_count": str(
                            int(drafter_parameters)
                        ),
                        "verifier_model_parameter_count": str(
                            int(verifier_parameters)
                        ),
                    }
                )

    errors = []
    if missing_model_names:
        errors.append("missing_drafter_or_verifier_model_name")
    if same_model_sessions:
        errors.append("drafter_verifier_same_model")
    if missing_model_sizes:
        errors.append("missing_drafter_or_verifier_model_size")
    if non_smaller_drafter:
        errors.append("drafter_model_must_be_smaller_than_verifier")

    return {
        "ok": not errors,
        "speculative_sessions": len(speculative_sessions),
        "model_pairs": [
            {"drafter_model": drafter, "verifier_model": verifier}
            for drafter, verifier in sorted(model_pairs)
        ],
        "model_size_pairs": [
            {
                "drafter_model_parameter_count": drafter,
                "verifier_model_parameter_count": verifier,
            }
            for drafter, verifier in sorted(model_size_pairs)
        ],
        "missing_model_name_sessions": len(missing_model_names),
        "same_model_sessions": len(same_model_sessions),
        "missing_model_size_sessions": len(missing_model_sizes),
        "non_smaller_drafter_sessions": len(non_smaller_drafter),
        "missing_model_names": missing_model_names[:20],
        "same_model": same_model_sessions[:20],
        "missing_model_sizes": missing_model_sizes[:20],
        "non_smaller_drafter": non_smaller_drafter[:20],
        "errors": errors,
    }


def input_consistency_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    required = [
        "prompt_set_sha256",
        "prompt_id",
        "prompt_sha256",
        "prompt_tokens",
        "prompt_token_sha256",
    ]
    missing = []
    invalid_token_count = []
    tokenization_by_prompt: Dict[
        Tuple[str, str, str],
        Dict[str, Set[str]],
    ] = defaultdict(lambda: {"token_hashes": set(), "token_counts": set()})
    prompt_hashes_by_id: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for row in sessions:
        missing_keys = [key for key in required if not row.get(key, "")]
        if missing_keys:
            detail = {
                "session_id": row.get("session_id", ""),
                "algorithm": row.get("algorithm", ""),
                "prompt_id": row.get("prompt_id", ""),
                "gamma": row.get("gamma", ""),
                "missing": missing_keys,
            }
            missing.append(detail)
            continue

        prompt_tokens = _int_field(row, "prompt_tokens")
        if prompt_tokens is None or prompt_tokens <= 0:
            invalid_token_count.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": row.get("algorithm", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "prompt_tokens": row.get("prompt_tokens", ""),
                }
            )
            continue

        key = (
            row.get("prompt_set_sha256", ""),
            row.get("prompt_id", ""),
            row.get("prompt_sha256", ""),
        )
        tokenization_by_prompt[key]["token_hashes"].add(
            row.get("prompt_token_sha256", "")
        )
        tokenization_by_prompt[key]["token_counts"].add(str(prompt_tokens))
        prompt_hashes_by_id[
            (row.get("prompt_set_sha256", ""), row.get("prompt_id", ""))
        ].add(row.get("prompt_sha256", ""))

    tokenization_mismatches = []
    for key, values in sorted(tokenization_by_prompt.items()):
        token_hashes = sorted(values["token_hashes"])
        token_counts = sorted(values["token_counts"])
        if len(token_hashes) > 1 or len(token_counts) > 1:
            tokenization_mismatches.append(
                {
                    "prompt_set_sha256": key[0],
                    "prompt_id": key[1],
                    "prompt_sha256": key[2],
                    "prompt_token_sha256_values": token_hashes,
                    "prompt_token_count_values": token_counts,
                }
            )

    prompt_id_hash_conflicts = []
    for key, hashes in sorted(prompt_hashes_by_id.items()):
        if len(hashes) > 1:
            prompt_id_hash_conflicts.append(
                {
                    "prompt_set_sha256": key[0],
                    "prompt_id": key[1],
                    "prompt_sha256_values": sorted(hashes),
                }
            )

    errors = []
    if missing:
        errors.append("missing_input_metadata")
    if invalid_token_count:
        errors.append("invalid_prompt_token_count")
    if tokenization_mismatches:
        errors.append("inconsistent_prompt_tokenization")
    if prompt_id_hash_conflicts:
        errors.append("prompt_id_hash_conflict")

    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "prompt_groups": len(tokenization_by_prompt),
        "missing_input_metadata_sessions": len(
            {item["session_id"] for item in missing}
        ),
        "invalid_prompt_token_count_sessions": len(
            {item["session_id"] for item in invalid_token_count}
        ),
        "tokenization_mismatch_groups": len(tokenization_mismatches),
        "prompt_id_hash_conflict_groups": len(prompt_id_hash_conflicts),
        "missing": missing[:20],
        "invalid_token_count": invalid_token_count[:20],
        "tokenization_mismatches": tokenization_mismatches[:20],
        "prompt_id_hash_conflicts": prompt_id_hash_conflicts[:20],
        "errors": errors,
    }


def communication_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    required_keys = [
        "rpc_request_bytes",
        "rpc_response_bytes",
        "rpc_total_bytes",
        "rpc_bytes_per_generated_token",
    ]
    missing = []
    invalid = []

    for row in sessions:
        row_missing = [key for key in required_keys if not row.get(key, "")]
        if row_missing:
            missing.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": row.get("algorithm", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "missing": row_missing,
                }
            )
            continue

        row_invalid = [
            key for key in required_keys if not _positive_float_text(row.get(key, ""))
        ]
        if row_invalid:
            invalid.append(
                {
                    "session_id": row.get("session_id", ""),
                    "algorithm": row.get("algorithm", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "invalid": row_invalid,
                }
            )

    errors = []
    if missing:
        errors.append("missing_communication_metrics")
    if invalid:
        errors.append("invalid_communication_metrics")

    total_bytes = [
        _float_value(row, "rpc_total_bytes") or 0.0
        for row in sessions
        if row.get("rpc_total_bytes", "")
    ]
    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "missing_metric_sessions": len(missing),
        "invalid_metric_sessions": len(invalid),
        "max_rpc_total_bytes": f"{max(total_bytes):.0f}" if total_bytes else "",
        "missing": missing[:20],
        "invalid": invalid[:20],
        "errors": errors,
    }


def _event_int(event: Dict[str, object], key: str) -> Optional[int]:
    value = event.get(key)
    if value in ("", None):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _event_token_count(event: Dict[str, object], key: str) -> int:
    value = event.get(key)
    return len(value) if isinstance(value, list) else 0


def _normalized_trace_value(value: object) -> str:
    if value in ("", None):
        return ""
    try:
        return str(int(float(str(value))))
    except (TypeError, ValueError):
        return str(value)


def _trace_session_detail(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "session_id": row.get("session_id", ""),
        "algorithm": row.get("algorithm", ""),
        "prompt_id": row.get("prompt_id", ""),
        "gamma": row.get("gamma", ""),
        "run": row.get("run", ""),
        "measurement_order": row.get("measurement_order", ""),
    }


def _check_trace_metadata(
    row: Dict[str, str],
    event: Dict[str, object],
    keys: Sequence[str],
) -> List[str]:
    mismatches = []
    for key in keys:
        if _normalized_trace_value(row.get(key, "")) != _normalized_trace_value(
            event.get(key)
        ):
            mismatches.append(key)
    return mismatches


def _event_output_hash(event: Dict[str, object]) -> str:
    output_hash = str(event.get("output_token_sha256", "") or "")
    if output_hash:
        return output_hash

    token_ids = event.get("generated_token_ids")
    if not isinstance(token_ids, list):
        return ""
    try:
        return spec_driver.token_ids_sha256([int(token) for token in token_ids])
    except (TypeError, ValueError):
        return ""


def trace_consistency_report(
    plans: List[Dict[str, object]],
    raw_rows: List[Dict[str, str]],
    trace_events: Optional[List[Dict[str, object]]] = None,
    require_trace: bool = False,
) -> Dict[str, object]:
    trace_events = trace_events or []
    sessions = validate_results.session_rows(raw_rows)
    expected_plan_hashes = {
        str(plan.get("plan_sha256", ""))
        for plan in plans
        if str(plan.get("plan_sha256", ""))
    }
    trace_plan_hashes = {
        str(event.get("plan", {}).get("plan_sha256", ""))
        for event in trace_events
        if event.get("event") == "plan" and isinstance(event.get("plan"), dict)
    }
    trace_plan_hashes.discard("")

    spec_steps: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    spec_run_summaries: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    baseline_runs: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    invalid_events = []
    for event in trace_events:
        event_name = str(event.get("event", ""))
        session_id = str(event.get("session_id", ""))
        run_index = _event_int(event, "run")
        if event_name == "step":
            if not session_id:
                invalid_events.append({"reason": "missing_session_id", "event": event})
                continue
            if run_index is not None and run_index <= 0:
                continue
            draft_count = _event_token_count(event, "draft_tokens")
            accepted = _event_int(event, "accepted_tokens")
            generated_so_far = _event_int(event, "generated_tokens_so_far")
            if (
                draft_count <= 0
                or accepted is None
                or accepted < 0
                or accepted > draft_count
                or generated_so_far is None
                or generated_so_far <= 0
            ):
                invalid_events.append(
                    {
                        "reason": "invalid_spec_step_trace",
                        "session_id": session_id,
                        "source": event.get("_source", ""),
                        "line": event.get("_line", ""),
                    }
                )
                continue
            spec_steps[session_id].append(event)
        elif event_name == "speculative_run":
            if not session_id:
                invalid_events.append({"reason": "missing_session_id", "event": event})
                continue
            if run_index is not None and run_index <= 0:
                continue
            if (
                _event_int(event, "generated_tokens") is None
                or not _event_output_hash(event)
            ):
                invalid_events.append(
                    {
                        "reason": "invalid_spec_summary_trace",
                        "session_id": session_id,
                        "source": event.get("_source", ""),
                        "line": event.get("_line", ""),
                    }
                )
                continue
            spec_run_summaries[session_id].append(event)
        elif event_name == "verifier_baseline_run":
            if not session_id:
                invalid_events.append({"reason": "missing_session_id", "event": event})
                continue
            if run_index is not None and run_index <= 0:
                continue
            if _event_int(event, "generated_tokens") is None:
                invalid_events.append(
                    {
                        "reason": "invalid_baseline_trace",
                        "session_id": session_id,
                        "source": event.get("_source", ""),
                        "line": event.get("_line", ""),
                    }
                )
                continue
            baseline_runs[session_id].append(event)

    row_session_ids = {row.get("session_id", "") for row in sessions}
    traced_session_ids = set(spec_steps) | set(spec_run_summaries) | set(baseline_runs)
    missing_trace_sessions = []
    missing_trace_summary_sessions = []
    trace_mismatches = []
    trace_summary_mismatches = []

    for row in sessions:
        session_id = row.get("session_id", "")
        algorithm = row.get("algorithm", "speculative")
        if algorithm == "speculative":
            events = sorted(
                spec_steps.get(session_id, []),
                key=lambda event: _event_int(event, "step") or 0,
            )
            if not events:
                missing_trace_sessions.append(_trace_session_detail(row))
                continue
            metadata_mismatches = sorted(
                {
                    key
                    for event in events
                    for key in _check_trace_metadata(
                        row,
                        event,
                        [
                            "algorithm",
                            "decoding_mode",
                            "prompt_id",
                            "prompt_sha256",
                            "prompt_set_sha256",
                            "gamma",
                            "run",
                            "measurement_order",
                            "drafter_freq_hz",
                            "verifier_clock_mhz",
                        ],
                    )
                }
            )
            generated_progress = [
                _event_int(event, "generated_tokens_so_far") or 0 for event in events
            ]
            nonmonotonic = generated_progress != sorted(generated_progress)
            state_mismatches = []
            previous_generated = 0
            prompt_tokens = _int_field(row, "prompt_tokens")
            for event in events:
                step = _event_int(event, "step") or 0
                generated_so_far = _event_int(event, "generated_tokens_so_far") or 0
                expected_base = prompt_tokens + previous_generated
                expected_committed = prompt_tokens + generated_so_far
                observed_base = _event_int(event, "base_committed_tokens")
                observed_committed = _event_int(event, "committed_tokens_after_step")
                observed_verifier = _event_int(event, "verifier_committed_tokens")
                observed_drafter = _event_int(event, "drafter_committed_tokens")
                if observed_base != expected_base:
                    state_mismatches.append(
                        {
                            "step": step,
                            "field": "base_committed_tokens",
                            "expected": expected_base,
                            "observed": observed_base,
                        }
                    )
                for field, observed in (
                    ("committed_tokens_after_step", observed_committed),
                    ("verifier_committed_tokens", observed_verifier),
                    ("drafter_committed_tokens", observed_drafter),
                ):
                    if observed != expected_committed:
                        state_mismatches.append(
                            {
                                "step": step,
                                "field": field,
                                "expected": expected_committed,
                                "observed": observed,
                            }
                        )
                previous_generated = generated_so_far
            actual = {
                "steps": len(events),
                "draft_tokens": sum(
                    _event_token_count(event, "draft_tokens") for event in events
                ),
                "accepted_draft_tokens": sum(
                    _event_int(event, "accepted_tokens") or 0 for event in events
                ),
                "replacement_tokens": sum(
                    1 for event in events if bool(event.get("appended_replacement"))
                ),
                "generated_tokens": generated_progress[-1] if generated_progress else 0,
            }
            expected = {
                key: _int_field(row, key)
                for key in (
                    "steps",
                    "draft_tokens",
                    "accepted_draft_tokens",
                    "replacement_tokens",
                    "generated_tokens",
                )
            }
            mismatched_keys = [
                key for key, value in expected.items() if value != actual[key]
            ]
            if metadata_mismatches or mismatched_keys or nonmonotonic or state_mismatches:
                detail = _trace_session_detail(row)
                detail.update(
                    {
                        "metadata_mismatches": metadata_mismatches,
                        "count_mismatches": mismatched_keys,
                        "nonmonotonic_generated_tokens": nonmonotonic,
                        "state_mismatches": state_mismatches[:20],
                        "actual": actual,
                        "expected": expected,
                    }
                )
                trace_mismatches.append(detail)

            summaries = spec_run_summaries.get(session_id, [])
            if not summaries:
                if require_trace:
                    missing_trace_summary_sessions.append(_trace_session_detail(row))
                continue
            summary = summaries[0]
            summary_metadata_mismatches = _check_trace_metadata(
                row,
                summary,
                [
                    "algorithm",
                    "decoding_mode",
                    "prompt_id",
                    "prompt_sha256",
                    "prompt_set_sha256",
                    "gamma",
                    "run",
                    "measurement_order",
                    "drafter_freq_hz",
                    "verifier_clock_mhz",
                ],
            )
            summary_mismatches = []
            for key in (
                "steps",
                "draft_tokens",
                "accepted_draft_tokens",
                "replacement_tokens",
                "generated_tokens",
            ):
                if _event_int(summary, key) != _int_field(row, key):
                    summary_mismatches.append(key)
            if str(summary.get("stop_reason", "")) != row.get("stop_reason", ""):
                summary_mismatches.append("stop_reason")
            if _event_output_hash(summary) != row.get("output_token_sha256", ""):
                summary_mismatches.append("output_token_sha256")
            if len(summaries) != 1:
                summary_mismatches.append("duplicate_speculative_run_trace_events")
            if summary_metadata_mismatches or summary_mismatches:
                detail = _trace_session_detail(row)
                detail.update(
                    {
                        "metadata_mismatches": summary_metadata_mismatches,
                        "summary_mismatches": summary_mismatches,
                    }
                )
                trace_summary_mismatches.append(detail)
        elif algorithm == "verifier_only":
            events = baseline_runs.get(session_id, [])
            if not events:
                missing_trace_sessions.append(_trace_session_detail(row))
                continue
            event = events[0]
            metadata_mismatches = _check_trace_metadata(
                row,
                event,
                [
                    "algorithm",
                    "decoding_mode",
                    "prompt_id",
                    "prompt_sha256",
                    "prompt_set_sha256",
                    "run",
                    "measurement_order",
                    "verifier_clock_mhz",
                ],
            )
            actual_generated = _event_int(event, "generated_tokens")
            expected_generated = _int_field(row, "generated_tokens")
            mismatched_keys = []
            if actual_generated != expected_generated:
                mismatched_keys.append("generated_tokens")
            if str(event.get("stop_reason", "")) != row.get("stop_reason", ""):
                mismatched_keys.append("stop_reason")
            baseline_output_hash = _event_output_hash(event)
            if require_trace and not baseline_output_hash:
                mismatched_keys.append("output_token_sha256_missing")
            elif (
                baseline_output_hash
                and baseline_output_hash != row.get("output_token_sha256", "")
            ):
                mismatched_keys.append("output_token_sha256")
            if len(events) != 1:
                mismatched_keys.append("duplicate_baseline_trace_events")
            if metadata_mismatches or mismatched_keys:
                detail = _trace_session_detail(row)
                detail.update(
                    {
                        "metadata_mismatches": metadata_mismatches,
                        "count_mismatches": mismatched_keys,
                        "actual_generated_tokens": actual_generated,
                        "expected_generated_tokens": expected_generated,
                    }
                )
                trace_mismatches.append(detail)

    orphan_trace_sessions = sorted(traced_session_ids - row_session_ids)
    missing_plan_hashes = sorted(expected_plan_hashes - trace_plan_hashes)
    unexpected_plan_hashes = sorted(trace_plan_hashes - expected_plan_hashes)

    errors = []
    trace_was_provided = bool(trace_events)
    if require_trace and not trace_was_provided:
        errors.append("missing_trace_events")
    if invalid_events:
        errors.append("invalid_trace_events")
    if trace_was_provided or require_trace:
        if missing_plan_hashes:
            errors.append("missing_trace_plan_hash")
        if unexpected_plan_hashes:
            errors.append("unexpected_trace_plan_hash")
        if missing_trace_sessions:
            errors.append("missing_trace_sessions")
        if missing_trace_summary_sessions:
            errors.append("missing_trace_summary_sessions")
        if orphan_trace_sessions:
            errors.append("orphan_trace_sessions")
        if trace_mismatches:
            errors.append("trace_session_mismatch")
        if trace_summary_mismatches:
            errors.append("trace_summary_mismatch")

    return {
        "ok": not errors,
        "trace_required": require_trace,
        "trace_events": len(trace_events),
        "trace_plan_hashes": sorted(trace_plan_hashes),
        "expected_plan_hashes": sorted(expected_plan_hashes),
        "missing_trace_plan_hashes": missing_plan_hashes,
        "unexpected_trace_plan_hashes": unexpected_plan_hashes,
        "checked_sessions": len(sessions),
        "traced_sessions": len(traced_session_ids & row_session_ids),
        "missing_trace_sessions": len(missing_trace_sessions),
        "missing_trace_summary_sessions": len(missing_trace_summary_sessions),
        "orphan_trace_sessions": len(orphan_trace_sessions),
        "trace_mismatch_sessions": len(
            {item["session_id"] for item in trace_mismatches}
        ),
        "trace_summary_mismatch_sessions": len(
            {item["session_id"] for item in trace_summary_mismatches}
        ),
        "invalid_trace_events": len(invalid_events),
        "missing_trace_session_details": missing_trace_sessions[:20],
        "missing_trace_summary_session_details": missing_trace_summary_sessions[:20],
        "orphan_trace_session_ids": orphan_trace_sessions[:20],
        "trace_mismatches": trace_mismatches[:20],
        "trace_summary_mismatches": trace_summary_mismatches[:20],
        "invalid_events": invalid_events[:20],
        "errors": errors,
    }


GAMMA_TREND_GROUPS = {
    "drafter_total": {
        "slope": "log2_gamma_slope_drafter_total_energy_mj_per_token",
        "correlation": "pearson_log2_gamma_drafter_total_energy",
    },
    "drafter_draft": {
        "slope": "log2_gamma_slope_drafter_draft_energy_mj_per_token",
        "correlation": "pearson_log2_gamma_drafter_draft_energy",
    },
}

GAMMA_ACTIVE_TREND_GROUP = {
    "drafter_active": {
        "slope": "log2_gamma_slope_drafter_active_energy_mj_per_token",
        "correlation": "pearson_log2_gamma_drafter_active_energy",
    },
    "drafter_draft_active": {
        "slope": "log2_gamma_slope_drafter_draft_active_energy_mj_per_token",
        "correlation": "pearson_log2_gamma_drafter_draft_active_energy",
    },
}


def gamma_trend_report(
    gamma_effect_rows: List[Dict[str, str]],
    min_gammas: int,
    require_active_trend: bool,
) -> Dict[str, object]:
    min_required_gammas = max(2, int(min_gammas))
    required_groups = dict(GAMMA_TREND_GROUPS)
    if require_active_trend:
        required_groups.update(GAMMA_ACTIVE_TREND_GROUP)

    groups: Dict[GammaDesignKey, List[Dict[str, str]]] = defaultdict(list)
    for row in gamma_effect_rows:
        groups[_gamma_design_key(row)].append(row)

    eligible_groups = []
    valid_groups = []
    insufficient_groups = []
    missing_trends = []
    invalid_trends = []

    for key, rows in sorted(groups.items()):
        gamma_values = sorted(
            {
                row.get("gamma", "")
                for row in rows
                if row.get("gamma", "") not in ("", None)
            }
        )
        gamma_count = max([_int_value(row, "gamma_count") for row in rows] or [0])
        group_detail = _gamma_key_dict(key)
        group_detail["gamma_count"] = str(gamma_count)
        group_detail["gamma_values"] = ",".join(gamma_values)
        group_detail["min_gammas"] = str(min_required_gammas)
        if len(gamma_values) < min_required_gammas or gamma_count < min_required_gammas:
            insufficient_groups.append(group_detail)
            continue

        eligible_groups.append(group_detail)
        representative = rows[0]
        group_missing = []
        group_invalid = []
        for metric_name, keys in sorted(required_groups.items()):
            slope_text = representative.get(keys["slope"], "")
            corr_text = representative.get(keys["correlation"], "")
            missing_keys = [
                key
                for key, value in (
                    (keys["slope"], slope_text),
                    (keys["correlation"], corr_text),
                )
                if value in ("", None)
            ]
            if missing_keys:
                group_missing.append(
                    {
                        **group_detail,
                        "metric": metric_name,
                        "missing": missing_keys,
                    }
                )
                continue

            slope = _float_value(representative, keys["slope"])
            corr = _float_value(representative, keys["correlation"])
            if slope is None or corr is None:
                group_invalid.append(
                    {
                        **group_detail,
                        "metric": metric_name,
                        "reason": "non_numeric_trend",
                        "slope": slope_text,
                        "correlation": corr_text,
                    }
                )
                continue
            if corr < -1.000001 or corr > 1.000001:
                group_invalid.append(
                    {
                        **group_detail,
                        "metric": metric_name,
                        "reason": "correlation_out_of_range",
                        "slope": f"{slope:.6f}",
                        "correlation": f"{corr:.6f}",
                    }
                )

        missing_trends.extend(group_missing)
        invalid_trends.extend(group_invalid)
        if not group_missing and not group_invalid:
            valid_groups.append(group_detail)

    errors = []
    if not gamma_effect_rows:
        errors.append("no_gamma_effect_rows")
    if not eligible_groups:
        errors.append("no_gamma_trend_eligible_config")
    if not valid_groups:
        errors.append("no_gamma_trend_ready_config")
    if missing_trends:
        errors.append("missing_gamma_trend_statistics")
    if invalid_trends:
        errors.append("invalid_gamma_trend_statistics")

    return {
        "ok": not errors,
        "min_gammas": min_required_gammas,
        "require_active_trend": require_active_trend,
        "required_metrics": sorted(required_groups),
        "config_groups": len(groups),
        "eligible_trend_groups": len(eligible_groups),
        "valid_trend_groups": len(valid_groups),
        "insufficient_trend_groups": len(insufficient_groups),
        "missing_trend_rows": len(missing_trends),
        "invalid_trend_rows": len(invalid_trends),
        "eligible_group_details": eligible_groups[:20],
        "valid_group_details": valid_groups[:20],
        "insufficient_group_details": insufficient_groups[:20],
        "missing_trends": missing_trends[:20],
        "invalid_trends": invalid_trends[:20],
        "errors": errors,
    }


def _float_value(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _close_enough(actual: float, expected: float, tolerance: float = 1e-4) -> bool:
    return abs(actual - expected) <= max(tolerance, abs(expected) * tolerance)


def timing_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    required = [
        "generated_tokens",
        "wall_latency_ms",
        "tokens_per_s",
        "client_rpc_latency_ms",
        "server_compute_latency_ms",
        "estimated_rpc_overhead_ms",
    ]
    missing = []
    invalid = []
    throughput_mismatches = []
    latency_mismatches = []

    for row in sessions:
        missing_keys = [key for key in required if row.get(key, "") in ("", None)]
        if missing_keys:
            detail = _accounting_session(row)
            detail["missing"] = missing_keys
            missing.append(detail)
            continue

        generated = _float_value(row, "generated_tokens")
        wall_ms = _float_value(row, "wall_latency_ms")
        tokens_per_s = _float_value(row, "tokens_per_s")
        client_rpc_ms = _float_value(row, "client_rpc_latency_ms")
        server_compute_ms = _float_value(row, "server_compute_latency_ms")
        estimated_rpc_overhead_ms = _float_value(row, "estimated_rpc_overhead_ms")
        if None in (
            generated,
            wall_ms,
            tokens_per_s,
            client_rpc_ms,
            server_compute_ms,
            estimated_rpc_overhead_ms,
        ):
            detail = _accounting_session(row)
            detail["reason"] = "non_numeric_timing_field"
            invalid.append(detail)
            continue
        assert generated is not None
        assert wall_ms is not None
        assert tokens_per_s is not None
        assert client_rpc_ms is not None
        assert server_compute_ms is not None
        assert estimated_rpc_overhead_ms is not None

        if generated <= 0 or wall_ms <= 0 or tokens_per_s <= 0:
            detail = _accounting_session(row)
            detail["reason"] = "nonpositive_generation_latency_or_throughput"
            invalid.append(detail)
            continue

        expected_tokens_per_s = generated / (wall_ms / 1000.0)
        if not _close_enough(tokens_per_s, expected_tokens_per_s, tolerance=1e-3):
            detail = _accounting_session(row)
            detail["actual_tokens_per_s"] = f"{tokens_per_s:.6f}"
            detail["expected_tokens_per_s"] = f"{expected_tokens_per_s:.6f}"
            detail["generated_tokens"] = f"{generated:.6f}"
            detail["wall_latency_ms"] = f"{wall_ms:.6f}"
            throughput_mismatches.append(detail)

        if client_rpc_ms < 0 or server_compute_ms < 0:
            detail = _accounting_session(row)
            detail["reason"] = "negative_latency"
            detail["client_rpc_latency_ms"] = f"{client_rpc_ms:.6f}"
            detail["server_compute_latency_ms"] = f"{server_compute_ms:.6f}"
            invalid.append(detail)
        if not _close_enough(
            estimated_rpc_overhead_ms,
            client_rpc_ms - server_compute_ms,
            tolerance=1e-3,
        ):
            detail = _accounting_session(row)
            detail["check"] = "estimated_rpc_overhead_ms"
            detail["actual"] = f"{estimated_rpc_overhead_ms:.6f}"
            detail["expected"] = f"{client_rpc_ms - server_compute_ms:.6f}"
            latency_mismatches.append(detail)
        if client_rpc_ms > wall_ms * 1.001 or server_compute_ms > wall_ms * 1.001:
            detail = _accounting_session(row)
            detail["check"] = "latency_exceeds_wall_latency"
            detail["wall_latency_ms"] = f"{wall_ms:.6f}"
            detail["client_rpc_latency_ms"] = f"{client_rpc_ms:.6f}"
            detail["server_compute_latency_ms"] = f"{server_compute_ms:.6f}"
            latency_mismatches.append(detail)

    errors = []
    if missing:
        errors.append("missing_timing_fields")
    if invalid:
        errors.append("invalid_timing_fields")
    if throughput_mismatches:
        errors.append("throughput_latency_mismatch")
    if latency_mismatches:
        errors.append("latency_accounting_mismatch")

    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "missing_timing_sessions": len({item["session_id"] for item in missing}),
        "invalid_timing_sessions": len({item["session_id"] for item in invalid}),
        "throughput_mismatch_sessions": len(
            {item["session_id"] for item in throughput_mismatches}
        ),
        "latency_mismatch_sessions": len(
            {item["session_id"] for item in latency_mismatches}
        ),
        "missing": missing[:20],
        "invalid": invalid[:20],
        "throughput_mismatches": throughput_mismatches[:20],
        "latency_mismatches": latency_mismatches[:20],
        "errors": errors,
    }


def _accounting_session(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "session_id": row.get("session_id", ""),
        "algorithm": row.get("algorithm", ""),
        "prompt_id": row.get("prompt_id", ""),
        "gamma": row.get("gamma", ""),
        "run": row.get("run", ""),
    }


def _int_field(row: Dict[str, str], key: str) -> Optional[int]:
    value = _float_value(row, key)
    if value is None:
        return None
    return int(value)


def _append_missing(
    missing_fields: List[Dict[str, object]],
    row: Dict[str, str],
    fields: Sequence[str],
    group: str,
) -> bool:
    absent = [field for field in fields if row.get(field, "") in ("", None)]
    if absent:
        detail = _accounting_session(row)
        detail["group"] = group
        detail["missing"] = absent
        missing_fields.append(detail)
        return True
    return False


def _append_mismatch(
    mismatches: List[Dict[str, object]],
    row: Dict[str, str],
    check: str,
    actual: float,
    expected: float,
) -> None:
    if _close_enough(actual, expected):
        return
    detail = _accounting_session(row)
    detail.update(
        {
            "check": check,
            "actual": f"{actual:.6f}",
            "expected": f"{expected:.6f}",
        }
    )
    mismatches.append(detail)


def accounting_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    missing_fields: List[Dict[str, object]] = []
    token_errors: List[Dict[str, object]] = []
    energy_mismatches: List[Dict[str, object]] = []

    for row in sessions:
        algorithm = row.get("algorithm", "speculative")
        includes_idle_drafter = (
            algorithm != "speculative"
            and row.get("system_boundary", "") == "two_device_idle_drafter"
        )
        generated = _int_field(row, "generated_tokens")
        steps = _int_field(row, "steps")

        if algorithm == "speculative":
            token_required = [
                "generated_tokens",
                "draft_tokens",
                "accepted_draft_tokens",
                "replacement_tokens",
                "steps",
                "gamma",
                "accept_rate",
            ]
            if not _append_missing(missing_fields, row, token_required, "token_accounting"):
                draft = _int_field(row, "draft_tokens")
                accepted = _int_field(row, "accepted_draft_tokens")
                replacement = _int_field(row, "replacement_tokens")
                gamma = _int_field(row, "gamma")
                accept_rate = _float_value(row, "accept_rate")
                if (
                    generated is None
                    or draft is None
                    or accepted is None
                    or replacement is None
                    or steps is None
                    or gamma is None
                    or accept_rate is None
                ):
                    detail = _accounting_session(row)
                    detail["reason"] = "non_numeric_spec_token_accounting"
                    token_errors.append(detail)
                else:
                    checks = [
                        (
                            accepted + replacement == generated,
                            "generated_tokens_must_equal_accepted_plus_replacement",
                        ),
                        (0 <= accepted <= draft, "accepted_tokens_outside_draft_count"),
                        (replacement >= 0, "negative_replacement_tokens"),
                        (steps > 0, "nonpositive_spec_steps"),
                        (replacement <= steps, "replacement_tokens_exceed_steps"),
                        (
                            gamma > 0 and draft <= gamma * steps,
                            "draft_tokens_exceed_gamma_times_steps",
                        ),
                    ]
                    for ok, reason in checks:
                        if not ok:
                            detail = _accounting_session(row)
                            detail["reason"] = reason
                            detail.update(
                                {
                                    "generated_tokens": str(generated),
                                    "draft_tokens": str(draft),
                                    "accepted_draft_tokens": str(accepted),
                                    "replacement_tokens": str(replacement),
                                    "steps": str(steps),
                                    "gamma": str(gamma),
                                }
                            )
                            token_errors.append(detail)
                    expected_accept_rate = accepted / draft if draft > 0 else 0.0
                    if not _close_enough(accept_rate, expected_accept_rate):
                        detail = _accounting_session(row)
                        detail["reason"] = "accept_rate_mismatch"
                        detail["actual_accept_rate"] = f"{accept_rate:.6f}"
                        detail["expected_accept_rate"] = f"{expected_accept_rate:.6f}"
                        token_errors.append(detail)
        else:
            token_required = ["generated_tokens", "steps"]
            if not _append_missing(missing_fields, row, token_required, "token_accounting"):
                if generated is None or steps is None:
                    detail = _accounting_session(row)
                    detail["reason"] = "non_numeric_baseline_token_accounting"
                    token_errors.append(detail)
                elif generated != steps:
                    detail = _accounting_session(row)
                    detail["reason"] = "baseline_steps_must_equal_generated_tokens"
                    detail["generated_tokens"] = str(generated)
                    detail["steps"] = str(steps)
                    token_errors.append(detail)

        if not _truthy_value(row.get("system_energy_complete", "")):
            continue

        energy_required = [
            "generated_tokens",
            "system_total_energy_mj",
            "system_total_energy_mj_per_generated_token",
            "verifier_total_energy_mj",
            "verifier_prefill_total_energy_mj",
            "verifier_verify_total_energy_mj",
        ]
        if algorithm == "speculative":
            energy_required.extend(
                [
                    "drafter_total_energy_mj",
                    "drafter_prefill_total_energy_mj",
                    "drafter_draft_total_energy_mj",
                    "drafter_commit_total_energy_mj",
                ]
            )
        elif includes_idle_drafter:
            energy_required.extend(
                [
                    "drafter_total_energy_mj",
                    "drafter_idle_total_energy_mj",
                ]
            )
        active_energy_required = []
        if row.get("system_active_energy_mj", "") not in ("", None):
            active_energy_required.extend(
                [
                    "system_active_energy_mj",
                    "system_active_energy_mj_per_generated_token",
                    "verifier_active_energy_mj",
                    "verifier_prefill_active_energy_mj",
                    "verifier_verify_active_energy_mj",
                ]
            )
            if algorithm == "speculative":
                active_energy_required.extend(
                    [
                        "drafter_active_energy_mj",
                        "drafter_prefill_active_energy_mj",
                        "drafter_draft_active_energy_mj",
                        "drafter_commit_active_energy_mj",
                    ]
                )
        energy_required.extend(active_energy_required)
        if _append_missing(missing_fields, row, energy_required, "energy_accounting"):
            continue

        generated_float = _float_value(row, "generated_tokens")
        system_total = _float_value(row, "system_total_energy_mj")
        system_per_token = _float_value(row, "system_total_energy_mj_per_generated_token")
        verifier_total = _float_value(row, "verifier_total_energy_mj")
        verifier_prefill = _float_value(row, "verifier_prefill_total_energy_mj")
        verifier_verify = _float_value(row, "verifier_verify_total_energy_mj")
        values = [
            generated_float,
            system_total,
            system_per_token,
            verifier_total,
            verifier_prefill,
            verifier_verify,
        ]
        if algorithm == "speculative":
            drafter_total = _float_value(row, "drafter_total_energy_mj")
            drafter_prefill = _float_value(row, "drafter_prefill_total_energy_mj")
            drafter_draft = _float_value(row, "drafter_draft_total_energy_mj")
            drafter_commit = _float_value(row, "drafter_commit_total_energy_mj")
            values.extend([drafter_total, drafter_prefill, drafter_draft, drafter_commit])
        else:
            drafter_total = _float_value(row, "drafter_total_energy_mj") or 0.0
            drafter_idle = _float_value(row, "drafter_idle_total_energy_mj") or 0.0
            drafter_prefill = drafter_draft = drafter_commit = 0.0
        if active_energy_required:
            active_values = [_float_value(row, key) for key in active_energy_required]
            values.extend(active_values)

        if any(value is None for value in values) or not generated_float or generated_float <= 0:
            detail = _accounting_session(row)
            detail["reason"] = "non_numeric_energy_accounting"
            token_errors.append(detail)
            continue

        assert system_total is not None
        assert system_per_token is not None
        assert verifier_total is not None
        assert verifier_prefill is not None
        assert verifier_verify is not None
        assert drafter_total is not None
        assert drafter_prefill is not None
        assert drafter_draft is not None
        assert drafter_commit is not None

        _append_mismatch(
            energy_mismatches,
            row,
            "system_total_energy_mj",
            system_total,
            drafter_total + verifier_total,
        )
        _append_mismatch(
            energy_mismatches,
            row,
            "system_total_energy_mj_per_generated_token",
            system_per_token,
            system_total / generated_float,
        )
        if algorithm == "speculative":
            _append_mismatch(
                energy_mismatches,
                row,
                "drafter_phase_energy_sum",
                drafter_total,
                drafter_prefill + drafter_draft + drafter_commit,
            )
        elif includes_idle_drafter:
            _append_mismatch(
                energy_mismatches,
                row,
                "baseline_drafter_total_must_equal_idle",
                drafter_total,
                drafter_idle,
            )
        else:
            _append_mismatch(
                energy_mismatches,
                row,
                "baseline_drafter_energy_must_be_zero",
                drafter_total,
                0.0,
            )
        _append_mismatch(
            energy_mismatches,
            row,
            "verifier_phase_energy_sum",
            verifier_total,
            verifier_prefill + verifier_verify,
        )

        active_keys = [
            "system_active_energy_mj",
            "system_active_energy_mj_per_generated_token",
            "verifier_active_energy_mj",
            "verifier_prefill_active_energy_mj",
            "verifier_verify_active_energy_mj",
        ]
        if algorithm == "speculative":
            active_keys.extend(
                [
                    "drafter_active_energy_mj",
                    "drafter_prefill_active_energy_mj",
                    "drafter_draft_active_energy_mj",
                    "drafter_commit_active_energy_mj",
                ]
            )
        if all(row.get(key, "") not in ("", None) for key in active_keys):
            system_active = _float_value(row, "system_active_energy_mj")
            system_active_per_token = _float_value(
                row,
                "system_active_energy_mj_per_generated_token",
            )
            verifier_active = _float_value(row, "verifier_active_energy_mj")
            verifier_prefill_active = _float_value(
                row,
                "verifier_prefill_active_energy_mj",
            )
            verifier_verify_active = _float_value(
                row,
                "verifier_verify_active_energy_mj",
            )
            drafter_active = (
                _float_value(row, "drafter_active_energy_mj")
                if algorithm == "speculative"
                else 0.0
            )
            drafter_prefill_active = (
                _float_value(row, "drafter_prefill_active_energy_mj")
                if algorithm == "speculative"
                else 0.0
            )
            drafter_draft_active = (
                _float_value(row, "drafter_draft_active_energy_mj")
                if algorithm == "speculative"
                else 0.0
            )
            drafter_commit_active = (
                _float_value(row, "drafter_commit_active_energy_mj")
                if algorithm == "speculative"
                else 0.0
            )
            if None not in (
                system_active,
                system_active_per_token,
                verifier_active,
                verifier_prefill_active,
                verifier_verify_active,
                drafter_active,
                drafter_prefill_active,
                drafter_draft_active,
                drafter_commit_active,
            ):
                _append_mismatch(
                    energy_mismatches,
                    row,
                    "system_active_energy_mj",
                    float(system_active),
                    float(drafter_active) + float(verifier_active),
                )
                _append_mismatch(
                    energy_mismatches,
                    row,
                    "system_active_energy_mj_per_generated_token",
                    float(system_active_per_token),
                    float(system_active) / generated_float,
                )
                _append_mismatch(
                    energy_mismatches,
                    row,
                    "verifier_active_phase_energy_sum",
                    float(verifier_active),
                    float(verifier_prefill_active) + float(verifier_verify_active),
                )
                if algorithm == "speculative":
                    _append_mismatch(
                        energy_mismatches,
                        row,
                        "drafter_active_phase_energy_sum",
                        float(drafter_active),
                        float(drafter_prefill_active)
                        + float(drafter_draft_active)
                        + float(drafter_commit_active),
                    )

    token_error_sessions = {str(item.get("session_id", "")) for item in token_errors}
    energy_mismatch_sessions = {
        str(item.get("session_id", "")) for item in energy_mismatches
    }
    missing_field_sessions = {
        str(item.get("session_id", "")) for item in missing_fields
    }
    errors = []
    if missing_fields:
        errors.append("missing_accounting_fields")
    if token_errors:
        errors.append("token_accounting_inconsistent")
    if energy_mismatches:
        errors.append("energy_accounting_inconsistent")
    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "missing_field_sessions": len(missing_field_sessions),
        "token_error_sessions": len(token_error_sessions),
        "energy_mismatch_sessions": len(energy_mismatch_sessions),
        "missing_fields": missing_fields[:20],
        "token_errors": token_errors[:20],
        "energy_mismatches": energy_mismatches[:20],
        "errors": errors,
    }


def _positive_energy_signal(row: Dict[str, str], key: str) -> bool:
    value = _float_value(row, key)
    return value is not None and value > 0


def _append_nonpositive_signal(
    rows: List[Dict[str, object]],
    row: Dict[str, str],
    key: str,
    group: str,
) -> None:
    if _positive_energy_signal(row, key):
        return
    detail = _accounting_session(row)
    detail.update(
        {
            "group": group,
            "field": key,
            "value": row.get(key, ""),
        }
    )
    rows.append(detail)


def energy_signal_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    nonpositive_total_energy = []
    nonpositive_active_energy = []
    nonpositive_idle_power = []

    for row in sessions:
        if not _truthy_value(row.get("system_energy_complete", "")):
            continue

        algorithm = row.get("algorithm", "speculative")
        includes_idle_drafter = (
            algorithm != "speculative"
            and row.get("system_boundary", "") == "two_device_idle_drafter"
        )
        total_fields = [
            "system_total_energy_mj",
            "system_total_energy_mj_per_generated_token",
            "verifier_total_energy_mj",
            "verifier_prefill_total_energy_mj",
            "verifier_verify_total_energy_mj",
        ]
        if algorithm == "speculative":
            total_fields.extend(
                [
                    "drafter_total_energy_mj",
                    "drafter_prefill_total_energy_mj",
                    "drafter_draft_total_energy_mj",
                    "drafter_commit_total_energy_mj",
                ]
            )
        elif includes_idle_drafter:
            total_fields.extend(
                [
                    "drafter_total_energy_mj",
                    "drafter_idle_total_energy_mj",
                    "drafter_idle_energy_mj_per_generated_token",
                ]
            )
        for key in total_fields:
            _append_nonpositive_signal(nonpositive_total_energy, row, key, "total_energy")

        active_fields = [
            "system_active_energy_mj",
            "system_active_energy_mj_per_generated_token",
            "verifier_active_energy_mj",
            "verifier_prefill_active_energy_mj",
            "verifier_verify_active_energy_mj",
        ]
        if algorithm == "speculative":
            active_fields.extend(
                [
                    "drafter_active_energy_mj",
                    "drafter_prefill_active_energy_mj",
                    "drafter_draft_active_energy_mj",
                    "drafter_commit_active_energy_mj",
                ]
            )
        for key in active_fields:
            if row.get(key, "") not in ("", None):
                _append_nonpositive_signal(
                    nonpositive_active_energy,
                    row,
                    key,
                    "active_energy",
                )

        if _positive_float_text(row.get("idle_baseline_s", "")):
            idle_fields = ["verifier_idle_power_mw", "system_idle_power_mw"]
            if algorithm == "speculative" or includes_idle_drafter:
                idle_fields.append("drafter_idle_power_mw")
            for key in idle_fields:
                _append_nonpositive_signal(
                    nonpositive_idle_power,
                    row,
                    key,
                    "idle_power",
                )

    errors = []
    if nonpositive_total_energy:
        errors.append("nonpositive_total_energy_signal")
    if nonpositive_active_energy:
        errors.append("nonpositive_active_energy_signal")
    if nonpositive_idle_power:
        errors.append("nonpositive_idle_power_signal")

    return {
        "ok": not errors,
        "checked_sessions": len(sessions),
        "nonpositive_total_energy_sessions": len(
            {item["session_id"] for item in nonpositive_total_energy}
        ),
        "nonpositive_active_energy_sessions": len(
            {item["session_id"] for item in nonpositive_active_energy}
        ),
        "nonpositive_idle_power_sessions": len(
            {item["session_id"] for item in nonpositive_idle_power}
        ),
        "nonpositive_total_energy": nonpositive_total_energy[:20],
        "nonpositive_active_energy": nonpositive_active_energy[:20],
        "nonpositive_idle_power": nonpositive_idle_power[:20],
        "errors": errors,
    }


def _throttle_active(value: str) -> bool:
    normalized = (value or "").strip().lower()
    return normalized not in ("", "0", "false", "no", "none", "n/a", "not active")


def runtime_status_report(
    raw_rows: List[Dict[str, str]],
    max_runtime_temp_c: Optional[float] = None,
    fail_on_throttle: bool = False,
) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    drafter_temps = [
        temp
        for row in sessions
        for temp in [_float_value(row, "drafter_runtime_temp_c")]
        if temp is not None
    ]
    verifier_temps = [
        temp
        for row in sessions
        for temp in [_float_value(row, "verifier_runtime_temp_c")]
        if temp is not None
    ]
    throttle_rows = [
        {
            "session_id": row.get("session_id", ""),
            "algorithm": row.get("algorithm", ""),
            "prompt_id": row.get("prompt_id", ""),
            "gamma": row.get("gamma", ""),
            "drafter_throttle": row.get("drafter_nvidia_throttle_active", ""),
            "verifier_throttle": row.get("verifier_nvidia_throttle_active", ""),
        }
        for row in sessions
        if _throttle_active(row.get("drafter_nvidia_throttle_active", ""))
        or _throttle_active(row.get("verifier_nvidia_throttle_active", ""))
    ]
    max_drafter = max(drafter_temps) if drafter_temps else None
    max_verifier = max(verifier_temps) if verifier_temps else None
    temperature_violations = []
    if max_runtime_temp_c is not None:
        for role, temp in (("drafter", max_drafter), ("verifier", max_verifier)):
            if temp is not None and temp > max_runtime_temp_c:
                temperature_violations.append(
                    {
                        "role": role,
                        "max_temp_c": f"{temp:.2f}",
                    }
                )

    errors = []
    if temperature_violations:
        errors.append("runtime_temperature_exceeds_limit")
    if fail_on_throttle and throttle_rows:
        errors.append("runtime_throttle_active")

    return {
        "ok": not errors,
        "temperature_rows": len(drafter_temps) + len(verifier_temps),
        "max_drafter_runtime_temp_c": (
            f"{max_drafter:.2f}" if max_drafter is not None else ""
        ),
        "max_verifier_runtime_temp_c": (
            f"{max_verifier:.2f}" if max_verifier is not None else ""
        ),
        "max_runtime_temp_c": max_runtime_temp_c,
        "temperature_violations": temperature_violations,
        "fail_on_throttle": fail_on_throttle,
        "throttle_active_sessions": len(throttle_rows),
        "throttle_active_details": throttle_rows[:20],
        "errors": errors,
    }


def token_compatibility_report(raw_rows: List[Dict[str, str]]) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    speculative_sessions = [
        row for row in sessions if row.get("algorithm", "speculative") == "speculative"
    ]
    tokenizer_metadata_keys = [
        "tokenizer_name_or_path",
        "tokenizer_class",
        "tokenizer_vocab_size",
        "tokenizer_base_vocab_size",
        "tokenizer_bos_token_id",
        "tokenizer_eos_token_id",
        "tokenizer_pad_token_id",
        "tokenizer_unk_token_id",
    ]
    vocab_mismatches = []
    missing_vocab_metadata = 0
    tokenizer_vocab_mismatches = []
    missing_tokenizer_metadata = 0
    tokenizer_special_token_mismatches = []
    for row in speculative_sessions:
        drafter_vocab = row.get("drafter_model_vocab_size", "")
        verifier_vocab = row.get("verifier_model_vocab_size", "")
        if not drafter_vocab or not verifier_vocab:
            missing_vocab_metadata += 1
        elif str(drafter_vocab) != str(verifier_vocab):
            vocab_mismatches.append(
                {
                    "session_id": row.get("session_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "gamma": row.get("gamma", ""),
                    "drafter_model": row.get("drafter_model", ""),
                    "verifier_model": row.get("verifier_model", ""),
                    "drafter_model_vocab_size": drafter_vocab,
                    "verifier_model_vocab_size": verifier_vocab,
                }
            )

    for row in sessions:
        tokenizer_vocab = row.get("tokenizer_vocab_size", "")
        if not tokenizer_vocab:
            missing_tokenizer_metadata += 1
            continue
        model_vocab_by_role = {
            "verifier": row.get("verifier_model_vocab_size", ""),
        }
        if row.get("algorithm", "speculative") == "speculative":
            model_vocab_by_role["drafter"] = row.get("drafter_model_vocab_size", "")
        for role, model_vocab in sorted(model_vocab_by_role.items()):
            if not model_vocab:
                continue
            if str(tokenizer_vocab) != str(model_vocab):
                tokenizer_vocab_mismatches.append(
                    {
                        "session_id": row.get("session_id", ""),
                        "algorithm": row.get("algorithm", ""),
                        "prompt_id": row.get("prompt_id", ""),
                        "gamma": row.get("gamma", ""),
                        "role": role,
                        "tokenizer_name_or_path": row.get(
                            "tokenizer_name_or_path",
                            "",
                        ),
                        "tokenizer_vocab_size": tokenizer_vocab,
                        "model_vocab_size": model_vocab,
                        "model": row.get(f"{role}_model", ""),
                    }
                )
        for role in ("verifier", "drafter"):
            if (
                role == "drafter"
                and row.get("algorithm", "speculative") != "speculative"
            ):
                continue
            for token_name in ("bos", "eos", "pad"):
                tokenizer_value = row.get(f"tokenizer_{token_name}_token_id", "")
                model_value = row.get(f"{role}_model_{token_name}_token_id", "")
                if not tokenizer_value or not model_value:
                    continue
                if str(tokenizer_value) != str(model_value):
                    tokenizer_special_token_mismatches.append(
                        {
                            "session_id": row.get("session_id", ""),
                            "algorithm": row.get("algorithm", ""),
                            "prompt_id": row.get("prompt_id", ""),
                            "gamma": row.get("gamma", ""),
                            "role": role,
                            "token": token_name,
                            "tokenizer_name_or_path": row.get(
                                "tokenizer_name_or_path",
                                "",
                            ),
                            "tokenizer_token_id": tokenizer_value,
                            "model_token_id": model_value,
                            "model": row.get(f"{role}_model", ""),
                        }
                    )

    mixed_tokenizer_metadata = []
    for key in tokenizer_metadata_keys:
        values = sorted({row.get(key, "") for row in sessions if row.get(key, "")})
        if len(values) > 1:
            mixed_tokenizer_metadata.append({"key": key, "values": values})

    missing_tokenizer_eos_metadata = [
        {
            "session_id": row.get("session_id", ""),
            "algorithm": row.get("algorithm", ""),
            "prompt_id": row.get("prompt_id", ""),
            "gamma": row.get("gamma", ""),
        }
        for row in sessions
        if not row.get("tokenizer_eos_token_id", "")
    ]

    verifier_vocab_by_model: Dict[str, Set[str]] = defaultdict(set)
    for row in sessions:
        model = row.get("verifier_model", "")
        vocab = row.get("verifier_model_vocab_size", "")
        if model and vocab:
            verifier_vocab_by_model[model].add(vocab)
    inconsistent_verifier_vocab = [
        {
            "verifier_model": model,
            "vocab_sizes": sorted(vocabs),
        }
        for model, vocabs in sorted(verifier_vocab_by_model.items())
        if len(vocabs) > 1
    ]
    tokenizer_special_token_mismatch_session_ids = {
        item["session_id"] for item in tokenizer_special_token_mismatches
    }

    errors = []
    if vocab_mismatches:
        errors.append("drafter_verifier_vocab_size_mismatch")
    if missing_vocab_metadata:
        errors.append("missing_model_vocab_metadata")
    if tokenizer_vocab_mismatches:
        errors.append("tokenizer_model_vocab_size_mismatch")
    if missing_tokenizer_metadata:
        errors.append("missing_tokenizer_vocab_metadata")
    if missing_tokenizer_eos_metadata:
        errors.append("missing_tokenizer_eos_metadata")
    if mixed_tokenizer_metadata:
        errors.append("mixed_tokenizer_metadata")
    if tokenizer_special_token_mismatches:
        errors.append("tokenizer_model_special_token_mismatch")
    if inconsistent_verifier_vocab:
        errors.append("inconsistent_verifier_vocab_size")

    return {
        "ok": not errors,
        "speculative_sessions": len(speculative_sessions),
        "vocab_mismatch_sessions": len(vocab_mismatches),
        "missing_vocab_metadata_sessions": missing_vocab_metadata,
        "tokenizer_vocab_mismatch_sessions": len(tokenizer_vocab_mismatches),
        "missing_tokenizer_metadata_sessions": missing_tokenizer_metadata,
        "missing_tokenizer_eos_metadata_sessions": len(missing_tokenizer_eos_metadata),
        "mixed_tokenizer_metadata_keys": [
            item["key"] for item in mixed_tokenizer_metadata
        ],
        "tokenizer_special_token_mismatch_sessions": len(
            tokenizer_special_token_mismatch_session_ids
        ),
        "inconsistent_verifier_vocab_models": len(inconsistent_verifier_vocab),
        "vocab_mismatches": vocab_mismatches[:20],
        "tokenizer_vocab_mismatches": tokenizer_vocab_mismatches[:20],
        "missing_tokenizer_eos_metadata": missing_tokenizer_eos_metadata[:20],
        "mixed_tokenizer_metadata": mixed_tokenizer_metadata[:20],
        "tokenizer_special_token_mismatches": tokenizer_special_token_mismatches[:20],
        "inconsistent_verifier_vocab": inconsistent_verifier_vocab,
        "errors": errors,
    }


def k8s_manifest_audit_report(
    payloads: Sequence[object],
    require_k8s_manifest_audit: bool = False,
) -> Dict[str, object]:
    valid_payloads = [payload for payload in payloads if isinstance(payload, dict)]
    invalid_schema_reports = []
    not_ok_reports = []
    missing_workload_reports = []
    unsuspended_job_reports = []
    design_mismatch_reports = []
    placement_mismatch_reports = []
    output_input_mismatch_reports = []
    schema_versions = []

    for index, payload in enumerate(valid_payloads):
        schema_version = str(payload.get("schema_version", ""))
        schema_versions.append(schema_version)
        if schema_version not in (
            "xronos-k8s-manifest-audit-v1",
            "xronos-k8s-manifest-audit-v2",
        ):
            invalid_schema_reports.append(index)
        if payload.get("ok") is not True:
            not_ok_reports.append(index)
        if payload.get("missing_workloads"):
            missing_workload_reports.append(index)
        if payload.get("unsuspended_jobs"):
            unsuspended_job_reports.append(index)
        errors = payload.get("errors", [])
        if not isinstance(errors, list):
            errors = []
        if any(str(error).endswith("_design_mismatch") for error in errors):
            design_mismatch_reports.append(index)
        placement = payload.get("placement", {})
        if isinstance(placement, dict) and placement.get("ok") is False:
            placement_mismatch_reports.append(index)
        if "k8s_manifest_audit_output_input_mismatch" in errors:
            output_input_mismatch_reports.append(index)

    errors = []
    if require_k8s_manifest_audit and not valid_payloads:
        errors.append("missing_required_k8s_manifest_audit")
    if invalid_schema_reports:
        errors.append("invalid_k8s_manifest_audit_schema")
    if not_ok_reports:
        errors.append("k8s_manifest_audit_not_ok")
    if missing_workload_reports:
        errors.append("k8s_manifest_audit_missing_workloads")
    if unsuspended_job_reports:
        errors.append("k8s_manifest_audit_unsuspended_jobs")
    if design_mismatch_reports:
        errors.append("k8s_manifest_audit_design_mismatch")
    if placement_mismatch_reports:
        errors.append("k8s_manifest_audit_placement_mismatch")
    if output_input_mismatch_reports:
        errors.append("k8s_manifest_audit_output_input_mismatch")

    return {
        "ok": not errors,
        "require_k8s_manifest_audit": require_k8s_manifest_audit,
        "reports": len(valid_payloads),
        "schema_versions": sorted(set(schema_versions)),
        "invalid_schema_reports": len(invalid_schema_reports),
        "not_ok_reports": len(not_ok_reports),
        "missing_workload_reports": len(missing_workload_reports),
        "unsuspended_job_reports": len(unsuspended_job_reports),
        "design_mismatch_reports": len(design_mismatch_reports),
        "placement_mismatch_reports": len(placement_mismatch_reports),
        "output_input_mismatch_reports": len(output_input_mismatch_reports),
        "invalid_schema_report_indices": invalid_schema_reports[:20],
        "not_ok_report_indices": not_ok_reports[:20],
        "placement_mismatch_report_indices": placement_mismatch_reports[:20],
        "errors": errors,
    }


def _payload_float(payload: Dict[str, object], key: str) -> Optional[float]:
    value = payload.get(key, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def network_probe_report(
    network_reports: List[object],
    require_network_probe: bool = False,
    max_mean_rtt_ms: Optional[float] = None,
    max_p95_rtt_ms: Optional[float] = None,
) -> Dict[str, object]:
    payloads = [report for report in network_reports if isinstance(report, dict)]
    invalid_schema = []
    not_ok_reports = []
    failed_targets = []
    missing_targets = []
    mean_violations = []
    p95_violations = []
    roles: Set[str] = set()

    for index, payload in enumerate(payloads):
        schema = str(payload.get("schema_version", ""))
        if schema != "xronos-network-probe-v1":
            invalid_schema.append({"index": index, "schema_version": schema})
        if payload.get("ok") is False:
            not_ok_reports.append(index)
        targets = payload.get("targets", {})
        if not isinstance(targets, dict):
            targets = {}
        for role in ("drafter", "verifier"):
            target = targets.get(role)
            if not isinstance(target, dict):
                missing_targets.append({"index": index, "role": role})
                continue
            roles.add(role)
            if target.get("ok") is False:
                failed_targets.append(
                    {
                        "index": index,
                        "role": role,
                        "errors": target.get("errors", []),
                    }
                )
            mean_rtt = _payload_float(target, "rtt_ms_mean")
            p95_rtt = _payload_float(target, "rtt_ms_p95")
            if (
                max_mean_rtt_ms is not None
                and mean_rtt is not None
                and mean_rtt > max_mean_rtt_ms
            ):
                mean_violations.append(
                    {
                        "index": index,
                        "role": role,
                        "rtt_ms_mean": f"{mean_rtt:.6f}",
                        "limit_ms": f"{max_mean_rtt_ms:.6f}",
                    }
                )
            if (
                max_p95_rtt_ms is not None
                and p95_rtt is not None
                and p95_rtt > max_p95_rtt_ms
            ):
                p95_violations.append(
                    {
                        "index": index,
                        "role": role,
                        "rtt_ms_p95": f"{p95_rtt:.6f}",
                        "limit_ms": f"{max_p95_rtt_ms:.6f}",
                    }
                )

    errors = []
    if require_network_probe and not payloads:
        errors.append("missing_required_network_probe")
    if invalid_schema:
        errors.append("invalid_network_probe_schema")
    if require_network_probe and missing_targets:
        errors.append("missing_network_probe_target")
    if not_ok_reports:
        errors.append("network_probe_not_ok")
    if failed_targets:
        errors.append("network_probe_target_failed")
    if mean_violations:
        errors.append("network_mean_rtt_exceeds_limit")
    if p95_violations:
        errors.append("network_p95_rtt_exceeds_limit")

    return {
        "ok": not errors,
        "require_network_probe": require_network_probe,
        "network_reports": len(payloads),
        "roles": sorted(roles),
        "max_mean_rtt_ms": max_mean_rtt_ms,
        "max_p95_rtt_ms": max_p95_rtt_ms,
        "invalid_schema_reports": len(invalid_schema),
        "not_ok_reports": len(not_ok_reports),
        "missing_targets": len(missing_targets),
        "failed_targets": len(failed_targets),
        "mean_rtt_violations": len(mean_violations),
        "p95_rtt_violations": len(p95_violations),
        "invalid_schema": invalid_schema[:20],
        "not_ok_report_indices": not_ok_reports[:20],
        "missing_target_details": missing_targets[:20],
        "failed_target_details": failed_targets[:20],
        "mean_rtt_violation_details": mean_violations[:20],
        "p95_rtt_violation_details": p95_violations[:20],
        "errors": errors,
    }


def select_configs(
    summary_rows: List[Dict[str, object]],
    energy_key: str,
    min_tokens_per_s: float,
    max_wall_latency_ms: Optional[float],
    min_runs: int,
    min_prompts: int,
) -> tuple:
    feasible = [
        row
        for row in summary_rows
        if select_best_config.passes_constraints(
            row=row,
            algorithms=["speculative"],
            min_tokens_per_s=min_tokens_per_s,
            max_wall_latency_ms=max_wall_latency_ms,
            min_runs=min_runs,
            min_prompts=min_prompts,
            energy_key=energy_key,
        )
    ]
    return (
        feasible,
        select_best_config.pareto_front(feasible, energy_key=energy_key),
        select_best_config.best_energy(feasible, energy_key=energy_key),
    )


def _interaction_experiment_key(
    row: Dict[str, str],
) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("prompt_set_sha256", "")),
        str(row.get("decoding_mode", "")),
        str(row.get("max_new_tokens", "")),
        str(row.get("stop_token_policy", "")),
        str(row.get("stop_token_ids", "")),
        str(row.get("drafter_model", "")),
        str(row.get("verifier_model", "")),
    )


def _interaction_group_dict(
    key: Tuple[str, str, str, str, str, str, str],
) -> Dict[str, str]:
    return {
        "prompt_set_sha256": key[0],
        "decoding_mode": key[1],
        "max_new_tokens": key[2],
        "stop_token_policy": key[3],
        "stop_token_ids": key[4],
        "drafter_model": key[5],
        "verifier_model": key[6],
    }


def _sorted_factor_values(values: Set[str]) -> List[str]:
    def sort_key(value: str) -> Tuple[int, float, str]:
        try:
            return (0, float(value), value)
        except (TypeError, ValueError):
            return (1, 0.0, value)

    return sorted((value for value in values if value not in ("", None)), key=sort_key)


def _mean_by_factor(
    rows: List[Dict[str, str]],
    factor_key: str,
    energy_key: str,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = str(row.get(factor_key, ""))
        energy = _float_value(row, energy_key)
        if not value or energy is None or energy <= 0:
            continue
        grouped[value].append(float(energy))

    result = []
    for value in _sorted_factor_values(set(grouped)):
        values = grouped[value]
        result.append(
            {
                factor_key: value,
                "mean_energy": f"{(sum(values) / len(values)):.6f}",
                "configs": str(len(values)),
            }
        )
    return result


def _lowest_mean_level(means: List[Dict[str, str]], factor_key: str) -> str:
    if not means:
        return ""
    return min(
        means,
        key=lambda row: (
            _float_value(row, "mean_energy") or float("inf"),
            row.get(factor_key, ""),
        ),
    ).get(factor_key, "")


def interaction_report(
    summary_rows: List[Dict[str, str]],
    energy_key: str,
    require_interaction_analysis: bool = False,
) -> Dict[str, object]:
    rows = [
        row
        for row in summary_rows
        if row.get("algorithm", "") == "speculative"
        and (_float_value(row, energy_key) or 0.0) > 0
    ]
    gamma_values = _sorted_factor_values({str(row.get("gamma", "")) for row in rows})
    drafter_freqs = _sorted_factor_values(
        {str(row.get("drafter_freq_hz", "")) for row in rows}
    )
    verifier_clocks = _sorted_factor_values(
        {str(row.get("verifier_clock_mhz", "")) for row in rows}
    )
    eligible = bool(rows) and len(gamma_values) >= 2 and len(verifier_clocks) >= 2

    missing_factorial_cells = []
    experiment_groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        experiment_groups[_interaction_experiment_key(row)].append(row)
    for key, group_rows in sorted(experiment_groups.items()):
        group_gammas = _sorted_factor_values(
            {str(row.get("gamma", "")) for row in group_rows}
        )
        group_drafter_freqs = _sorted_factor_values(
            {str(row.get("drafter_freq_hz", "")) for row in group_rows}
        )
        group_verifier_clocks = _sorted_factor_values(
            {str(row.get("verifier_clock_mhz", "")) for row in group_rows}
        )
        if len(group_gammas) < 2 or len(group_verifier_clocks) < 2:
            continue
        observed = {
            (
                str(row.get("gamma", "")),
                str(row.get("drafter_freq_hz", "")),
                str(row.get("verifier_clock_mhz", "")),
            )
            for row in group_rows
        }
        for gamma in group_gammas:
            for drafter_freq in group_drafter_freqs:
                for verifier_clock in group_verifier_clocks:
                    if (gamma, drafter_freq, verifier_clock) not in observed:
                        detail = _interaction_group_dict(key)
                        detail.update(
                            {
                                "gamma": gamma,
                                "drafter_freq_hz": drafter_freq,
                                "verifier_clock_mhz": verifier_clock,
                            }
                        )
                        missing_factorial_cells.append(detail)

    clock_by_gamma_groups = []
    changing_clock_groups = 0
    for key, group_rows in sorted(experiment_groups.items()):
        group_drafter_freqs = _sorted_factor_values(
            {str(row.get("drafter_freq_hz", "")) for row in group_rows}
        )
        for drafter_freq in group_drafter_freqs:
            sub_rows = [
                row
                for row in group_rows
                if str(row.get("drafter_freq_hz", "")) == drafter_freq
            ]
            sub_gammas = _sorted_factor_values(
                {str(row.get("gamma", "")) for row in sub_rows}
            )
            sub_verifier_clocks = _sorted_factor_values(
                {str(row.get("verifier_clock_mhz", "")) for row in sub_rows}
            )
            if len(sub_gammas) < 2 or len(sub_verifier_clocks) < 2:
                continue
            best_clock_by_gamma: Dict[str, str] = {}
            best_energy_by_gamma: Dict[str, str] = {}
            for gamma in sub_gammas:
                best = select_best_config.best_energy(
                    [
                        row
                        for row in sub_rows
                        if str(row.get("gamma", "")) == gamma
                    ],
                    energy_key=energy_key,
                )
                if best is None:
                    continue
                best_clock_by_gamma[gamma] = str(best.get("verifier_clock_mhz", ""))
                best_energy_by_gamma[gamma] = str(best.get(energy_key, ""))
            changes = len(set(best_clock_by_gamma.values())) > 1
            changing_clock_groups += int(changes)
            detail = _interaction_group_dict(key)
            detail.update(
                {
                    "drafter_freq_hz": drafter_freq,
                    "best_verifier_clock_by_gamma": best_clock_by_gamma,
                    "best_energy_by_gamma": best_energy_by_gamma,
                    "verifier_clock_changes_with_gamma": changes,
                }
            )
            clock_by_gamma_groups.append(detail)

    best_joint = select_best_config.best_energy(rows, energy_key=energy_key)
    gamma_means = _mean_by_factor(rows, "gamma", energy_key)
    drafter_means = _mean_by_factor(rows, "drafter_freq_hz", energy_key)
    verifier_means = _mean_by_factor(rows, "verifier_clock_mhz", energy_key)
    marginal_gamma = _lowest_mean_level(gamma_means, "gamma")
    marginal_drafter_freq = _lowest_mean_level(drafter_means, "drafter_freq_hz")
    marginal_verifier_clock = _lowest_mean_level(verifier_means, "verifier_clock_mhz")
    marginal_matches = [
        row
        for row in rows
        if str(row.get("gamma", "")) == marginal_gamma
        and str(row.get("drafter_freq_hz", "")) == marginal_drafter_freq
        and str(row.get("verifier_clock_mhz", "")) == marginal_verifier_clock
    ]
    marginal_config = select_best_config.best_energy(
        marginal_matches,
        energy_key=energy_key,
    )
    best_joint_energy = _float_value(best_joint or {}, energy_key) or 0.0
    marginal_energy = _float_value(marginal_config or {}, energy_key) or 0.0
    independent_gap = (
        (marginal_energy - best_joint_energy) / best_joint_energy * 100.0
        if best_joint_energy > 0 and marginal_energy > 0
        else None
    )

    errors = []
    if require_interaction_analysis:
        if not rows:
            errors.append("missing_interaction_summary_rows")
        if len(gamma_values) < 2:
            errors.append("interaction_requires_multiple_gammas")
        if len(verifier_clocks) < 2:
            errors.append("interaction_requires_multiple_verifier_clocks")
        if missing_factorial_cells:
            errors.append("interaction_requires_complete_factorial_grid")
        if marginal_config is None:
            errors.append("missing_marginal_independent_config")

    return {
        "ok": not errors,
        "required": require_interaction_analysis,
        "eligible": eligible,
        "energy_key": energy_key,
        "configs": len(rows),
        "gamma_values": gamma_values,
        "drafter_freq_hz_values": drafter_freqs,
        "verifier_clock_mhz_values": verifier_clocks,
        "missing_factorial_cells": missing_factorial_cells[:50],
        "missing_factorial_cells_count": len(missing_factorial_cells),
        "verifier_clock_depends_on_gamma": changing_clock_groups > 0,
        "verifier_clock_change_groups": changing_clock_groups,
        "clock_by_gamma_groups": clock_by_gamma_groups[:50],
        "best_joint_config": select_best_config.compact_config(best_joint, energy_key),
        "marginal_factor_means": {
            "gamma": gamma_means,
            "drafter_freq_hz": drafter_means,
            "verifier_clock_mhz": verifier_means,
        },
        "marginal_independent_levels": {
            "gamma": marginal_gamma,
            "drafter_freq_hz": marginal_drafter_freq,
            "verifier_clock_mhz": marginal_verifier_clock,
        },
        "marginal_independent_config": select_best_config.compact_config(
            marginal_config,
            energy_key,
        ),
        "independent_energy_gap_pct_vs_joint_best": (
            f"{independent_gap:.6f}" if independent_gap is not None else ""
        ),
        "errors": errors,
    }


def _claim_entry(
    ready: bool,
    evidence: Dict[str, object],
    blockers: List[str],
) -> Dict[str, object]:
    return {
        "ready": ready,
        "evidence": evidence,
        "blockers": blockers,
    }


def system_boundary_report(
    raw_rows: List[Dict[str, str]],
    paired_prompt_rows: List[Dict[str, str]],
    paired_summary_rows: List[Dict[str, str]],
    require_two_device_boundary: bool = False,
) -> Dict[str, object]:
    sessions = validate_results.session_rows(raw_rows)
    spec_boundaries = sorted(
        {
            row.get("system_boundary", "")
            for row in sessions
            if row.get("algorithm", "") == "speculative"
        }
    )
    baseline_boundaries = sorted(
        {
            row.get("system_boundary", "")
            for row in sessions
            if row.get("algorithm", "") == "verifier_only"
        }
    )
    pair_spec_boundaries = sorted(
        {
            row.get("spec_system_boundary", "")
            for row in paired_prompt_rows
            if row.get("spec_system_boundary", "")
        }
    )
    pair_baseline_boundaries = sorted(
        {
            row.get("baseline_system_boundary", "")
            for row in paired_prompt_rows
            if row.get("baseline_system_boundary", "")
        }
    )
    summary_baseline_boundaries = sorted(
        {
            row.get("baseline_system_boundary", "")
            for row in paired_summary_rows
            if row.get("baseline_system_boundary", "")
        }
    )

    wrong_spec_pairs = [
        {
            "gamma": row.get("gamma", ""),
            "drafter_freq_hz": row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_sha256": row.get("prompt_sha256", ""),
            "spec_system_boundary": row.get("spec_system_boundary", ""),
        }
        for row in paired_prompt_rows
        if row.get("spec_system_boundary", "") != "two_device_active"
    ]
    wrong_baseline_pairs = [
        {
            "gamma": row.get("gamma", ""),
            "drafter_freq_hz": row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_sha256": row.get("prompt_sha256", ""),
            "baseline_system_boundary": row.get("baseline_system_boundary", ""),
        }
        for row in paired_prompt_rows
        if row.get("baseline_system_boundary", "") != "two_device_idle_drafter"
    ]
    wrong_summary_rows = [
        {
            "gamma": row.get("gamma", ""),
            "drafter_freq_hz": row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
            "baseline_system_boundary": row.get("baseline_system_boundary", ""),
        }
        for row in paired_summary_rows
        if row.get("baseline_system_boundary", "") != "two_device_idle_drafter"
    ]

    errors = []
    if require_two_device_boundary and not paired_prompt_rows:
        errors.append("missing_paired_system_boundary_rows")
    if require_two_device_boundary and "two_device_active" not in spec_boundaries:
        errors.append("missing_two_device_speculative_rows")
    if (
        require_two_device_boundary
        and "two_device_idle_drafter" not in pair_baseline_boundaries
    ):
        errors.append("missing_two_device_idle_drafter_baseline_pairs")
    if require_two_device_boundary and wrong_spec_pairs:
        errors.append("paired_spec_boundary_not_two_device_active")
    if require_two_device_boundary and wrong_baseline_pairs:
        errors.append("paired_baseline_boundary_not_two_device_idle_drafter")
    if require_two_device_boundary and wrong_summary_rows:
        errors.append("paired_summary_boundary_not_two_device_idle_drafter")

    claim_ready = (
        bool(paired_prompt_rows)
        and not wrong_spec_pairs
        and not wrong_baseline_pairs
        and not wrong_summary_rows
    )
    return {
        "ok": not errors,
        "claim_ready": claim_ready,
        "require_two_device_boundary": require_two_device_boundary,
        "spec_system_boundaries": spec_boundaries,
        "baseline_system_boundaries": baseline_boundaries,
        "paired_spec_system_boundaries": pair_spec_boundaries,
        "paired_baseline_system_boundaries": pair_baseline_boundaries,
        "paired_summary_baseline_system_boundaries": summary_baseline_boundaries,
        "paired_prompt_rows": len(paired_prompt_rows),
        "paired_summary_rows": len(paired_summary_rows),
        "wrong_spec_boundary_pairs": len(wrong_spec_pairs),
        "wrong_baseline_boundary_pairs": len(wrong_baseline_pairs),
        "wrong_summary_boundary_rows": len(wrong_summary_rows),
        "wrong_spec_boundary_examples": wrong_spec_pairs[:20],
        "wrong_baseline_boundary_examples": wrong_baseline_pairs[:20],
        "wrong_summary_boundary_examples": wrong_summary_rows[:20],
        "errors": errors,
    }


def claim_readiness_report(
    gamma_design: Dict[str, object],
    gamma_statistics: Dict[str, object],
    gamma_trend: Dict[str, object],
    output_equivalence: Dict[str, object],
    system_boundary: Dict[str, object],
    system_optimization: Dict[str, object],
    gamma_policy: Dict[str, object],
    interaction: Dict[str, object],
    best_paired: Optional[Dict[str, str]],
    unpaired_prompt_rows: List[Dict[str, str]],
    paired_summary_rows: List[Dict[str, str]],
    min_gammas: int,
) -> Dict[str, object]:
    claims: Dict[str, Dict[str, object]] = {}

    blockers = []
    if not gamma_design.get("ok", False):
        blockers.append("gamma_design_not_ready")
    if not gamma_statistics.get("ok", False):
        blockers.append("gamma_statistics_not_ready")
    if not gamma_trend.get("ok", False):
        blockers.append("gamma_trend_not_ready")
    claims["drafter_gamma_energy"] = _claim_entry(
        ready=not blockers,
        blockers=blockers,
        evidence={
            "gamma_effect_rows": gamma_design.get("gamma_effect_rows", 0),
            "ready_configs": gamma_design.get("ready_configs", 0),
            "gamma_trend_valid_groups": gamma_trend.get("valid_trend_groups", 0),
            "checked_nonbaseline_rows": gamma_statistics.get(
                "checked_nonbaseline_rows",
                0,
            ),
        },
    )

    blockers = []
    if not paired_summary_rows or best_paired is None:
        blockers.append("missing_paired_baseline_summary")
    if not output_equivalence.get("ok", False):
        blockers.append("output_equivalence_not_ready")
    if not system_boundary.get("claim_ready", False):
        blockers.append("two_device_system_boundary_not_ready")
    if unpaired_prompt_rows:
        blockers.append("unpaired_speculative_prompts")
    claims["system_energy_vs_verifier_baseline"] = _claim_entry(
        ready=not blockers,
        blockers=blockers,
        evidence={
            "paired_summary_rows": len(paired_summary_rows),
            "paired_prompt_rows": output_equivalence.get("checked_prompt_pairs", 0),
            "mean_energy_savings_pct_vs_baseline": (
                best_paired or {}
            ).get("mean_energy_savings_pct_vs_baseline", ""),
            "bootstrap_ci95_low_energy_savings_pct_vs_baseline": (
                best_paired or {}
            ).get("bootstrap_ci95_low_energy_savings_pct_vs_baseline", ""),
            "bootstrap_ci95_high_energy_savings_pct_vs_baseline": (
                best_paired or {}
            ).get("bootstrap_ci95_high_energy_savings_pct_vs_baseline", ""),
            "sign_test_p_value_energy_savings": (
                best_paired or {}
            ).get("sign_test_p_value_energy_savings", ""),
            "paired_baseline_system_boundaries": system_boundary.get(
                "paired_baseline_system_boundaries",
                [],
            ),
        },
    )

    blockers = []
    if not system_optimization.get("ok", False):
        blockers.append("system_optimization_not_ready")
    if not system_optimization.get("best_joint_config", {}):
        blockers.append("missing_joint_best_config")
    if not system_optimization.get("best_gamma_one_config", {}):
        blockers.append("missing_gamma_one_reference_config")
    claims["joint_system_energy_optimization"] = _claim_entry(
        ready=not blockers,
        blockers=blockers,
        evidence={
            "energy_key": system_optimization.get("energy_key", ""),
            "energy_savings_pct_vs_best_gamma_one": system_optimization.get(
                "energy_savings_pct_vs_best_gamma_one",
                "",
            ),
            "energy_margin_pct_vs_runner_up": system_optimization.get(
                "energy_margin_pct_vs_runner_up",
                "",
            ),
            "energy_ci95_margin_clear": system_optimization.get(
                "energy_ci95_margin_clear",
                "",
            ),
        },
    )

    blockers = []
    if not interaction.get("ok", False):
        blockers.append("interaction_report_not_ok")
    if not interaction.get("eligible", False):
        blockers.append("interaction_not_eligible")
    if int(interaction.get("missing_factorial_cells_count", 0) or 0) > 0:
        blockers.append("missing_interaction_factorial_cells")
    claims["gamma_frequency_interaction"] = _claim_entry(
        ready=not blockers,
        blockers=blockers,
        evidence={
            "required": interaction.get("required", False),
            "gamma_values": interaction.get("gamma_values", []),
            "verifier_clock_mhz_values": interaction.get(
                "verifier_clock_mhz_values",
                [],
            ),
            "verifier_clock_depends_on_gamma": interaction.get(
                "verifier_clock_depends_on_gamma",
                False,
            ),
            "independent_energy_gap_pct_vs_joint_best": interaction.get(
                "independent_energy_gap_pct_vs_joint_best",
                "",
            ),
        },
    )

    blockers = []
    if not gamma_policy.get("ok", False):
        blockers.append("gamma_policy_not_ready")
    if int(gamma_policy.get("policy_rows", 0) or 0) < max(1, min_gammas):
        blockers.append("insufficient_gamma_policy_rows")
    claims["adaptive_gamma_frequency_policy"] = _claim_entry(
        ready=not blockers,
        blockers=blockers,
        evidence={
            "policy_rows": gamma_policy.get("policy_rows", 0),
            "min_gammas": min_gammas,
            "uses_gamma_dependent_verifier_clock": gamma_policy.get(
                "uses_gamma_dependent_verifier_clock",
                False,
            ),
            "uses_gamma_dependent_drafter_freq": gamma_policy.get(
                "uses_gamma_dependent_drafter_freq",
                False,
            ),
        },
    )

    ready_claims = [
        name for name, claim in claims.items() if bool(claim.get("ready", False))
    ]
    blocked_claims = [
        name for name, claim in claims.items() if not bool(claim.get("ready", False))
    ]
    return {
        "schema_version": "xronos-claim-readiness-v1",
        "ok": len(blocked_claims) == 0,
        "ready_claims": ready_claims,
        "blocked_claims": blocked_claims,
        "claims": claims,
    }


def output_equivalence_report(
    pair_rows: List[Dict[str, str]],
    require_checked_output: bool = True,
) -> Dict[str, object]:
    checked = [
        row
        for row in pair_rows
        if row.get("output_token_match", "") not in ("", None)
    ]
    unchecked = [
        {
            "gamma": row.get("gamma", ""),
            "drafter_freq_hz": row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_sha256": row.get("prompt_sha256", ""),
            "spec_output_hash_count": row.get("spec_output_hash_count", ""),
            "baseline_output_hash_count": row.get("baseline_output_hash_count", ""),
        }
        for row in pair_rows
        if row.get("output_token_match", "") in ("", None)
    ]
    mismatched = [
        {
            "gamma": row.get("gamma", ""),
            "drafter_freq_hz": row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_sha256": row.get("prompt_sha256", ""),
        }
        for row in checked
        if row.get("output_token_match") != "1.000000"
    ]
    unstable = [
        {
            "gamma": row.get("gamma", ""),
            "drafter_freq_hz": row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_sha256": row.get("prompt_sha256", ""),
            "spec_output_hash_count": row.get("spec_output_hash_count", ""),
            "baseline_output_hash_count": row.get("baseline_output_hash_count", ""),
        }
        for row in checked
        if _int_value(row, "spec_output_hash_count") != 1
        or _int_value(row, "baseline_output_hash_count") != 1
    ]
    return {
        "ok": len(mismatched) == 0
        and len(unstable) == 0
        and (not require_checked_output or len(unchecked) == 0),
        "checked_prompt_pairs": len(checked),
        "mismatched_prompt_pairs": len(mismatched),
        "unchecked_prompt_pairs": len(unchecked),
        "unstable_prompt_pairs": len(unstable),
        "require_checked_output": require_checked_output,
        "mismatches": mismatched,
        "unchecked": unchecked,
        "unstable": unstable,
    }


def build_report(
    plans: List[Dict[str, object]],
    raw_rows: List[Dict[str, str]],
    doctor_reports: Optional[List[object]] = None,
    plan_audit_reports: Optional[List[Dict[str, object]]] = None,
    k8s_manifest_audit_reports: Optional[List[object]] = None,
    network_reports: Optional[List[object]] = None,
    trace_events: Optional[List[Dict[str, object]]] = None,
    summary_energy_key: str = DEFAULT_SUMMARY_ENERGY_KEY,
    paired_energy_key: str = DEFAULT_PAIRED_ENERGY_KEY,
    allow_incomplete_energy: bool = False,
    allow_unpaired: bool = False,
    allow_unchecked_output: bool = False,
    min_tokens_per_s: float = 0.0,
    max_wall_latency_ms: Optional[float] = None,
    min_runs: int = 1,
    min_prompts: int = 1,
    min_gammas: int = 2,
    min_power_samples: int = 1,
    paired_bootstrap_samples: int = 1000,
    paired_bootstrap_seed: int = 0,
    require_doctor: bool = False,
    require_driver_doctor: bool = False,
    require_plan_audit: bool = False,
    require_k8s_manifest_audit: bool = False,
    require_network_probe: bool = False,
    require_trace: bool = False,
    require_interaction_analysis: bool = False,
    require_claim_readiness: bool = False,
    require_two_device_boundary: bool = False,
    max_network_mean_rtt_ms: Optional[float] = None,
    max_network_p95_rtt_ms: Optional[float] = None,
    max_runtime_temp_c: Optional[float] = None,
    fail_on_throttle: bool = False,
    max_energy_cv: Optional[float] = None,
    max_latency_cv: Optional[float] = None,
) -> Dict[str, object]:
    plan_design = plan_design_report(
        plans,
        allow_unpaired=allow_unpaired,
        min_prompts=min_prompts,
        min_runs=min_runs,
        min_gammas=min_gammas,
        require_two_device_boundary=require_two_device_boundary,
    )
    doctor_design = doctor_design_report(
        doctor_reports or [],
        require_doctor=require_doctor,
        require_driver_doctor=require_driver_doctor,
    )
    plan_audit = plan_audit_report(
        plans,
        plan_audit_reports or [],
        require_plan_audit=require_plan_audit,
    )
    k8s_manifest_audit = k8s_manifest_audit_report(
        k8s_manifest_audit_reports or [],
        require_k8s_manifest_audit=require_k8s_manifest_audit,
    )
    energy_design = energy_design_report(
        plans,
        summary_energy_key=summary_energy_key,
        paired_energy_key=paired_energy_key,
    )
    plan_integrity = plan_integrity_report(plans, raw_rows)
    schema_contract = schema_contract_report(plans, raw_rows)
    frequency_consistency = frequency_consistency_report(raw_rows)
    measurement_setup = measurement_setup_report(raw_rows)
    provenance = provenance_report(raw_rows)
    model_setup = model_setup_report(raw_rows)
    input_consistency = input_consistency_report(raw_rows)
    communication = communication_report(raw_rows)
    trace_consistency = trace_consistency_report(
        plans=plans,
        raw_rows=raw_rows,
        trace_events=trace_events,
        require_trace=require_trace,
    )
    network_probe = network_probe_report(
        network_reports or [],
        require_network_probe=require_network_probe,
        max_mean_rtt_ms=max_network_mean_rtt_ms,
        max_p95_rtt_ms=max_network_p95_rtt_ms,
    )
    timing = timing_report(raw_rows)
    accounting = accounting_report(raw_rows)
    energy_signal = energy_signal_report(raw_rows)
    runtime_status = runtime_status_report(
        raw_rows,
        max_runtime_temp_c=max_runtime_temp_c,
        fail_on_throttle=fail_on_throttle,
    )
    token_compatibility = token_compatibility_report(raw_rows)
    validation = validate_results.validate(
        plans=plans,
        raw_rows=raw_rows,
        require_complete_energy=not allow_incomplete_energy,
        require_idle_baseline=validate_results.plan_requires_idle(plans),
        min_power_samples=min_power_samples,
    )
    summary_rows = analyze_spec_results.summarize(
        raw_rows,
        allow_incomplete_energy=allow_incomplete_energy,
    )
    measurement_stability = measurement_stability_report(
        summary_rows,
        energy_key=summary_energy_key,
        max_energy_cv=max_energy_cv,
        max_latency_cv=max_latency_cv,
    )
    gamma_effect_rows = analyze_gamma_effect.summarize(
        raw_rows,
        allow_incomplete_energy=allow_incomplete_energy,
        bootstrap_samples=paired_bootstrap_samples,
        bootstrap_seed=paired_bootstrap_seed,
    )
    gamma_design = gamma_design_report(
        gamma_effect_rows,
        min_prompts=min_prompts,
        min_gammas=min_gammas,
    )
    gamma_statistics = gamma_statistics_report(
        gamma_effect_rows,
        require_active_stats=validate_results.plan_requires_idle(plans),
    )
    gamma_trend = gamma_trend_report(
        gamma_effect_rows,
        min_gammas=min_gammas,
        require_active_trend=validate_results.plan_requires_idle(plans),
    )
    paired_summary_rows, paired_prompt_rows = paired_prompt_compare.aggregate_pairs(
        raw_rows,
        energy_key=paired_energy_key,
        allow_incomplete_energy=allow_incomplete_energy,
        bootstrap_samples=paired_bootstrap_samples,
        bootstrap_seed=paired_bootstrap_seed,
    )
    unpaired_prompt_rows = paired_prompt_compare.unpaired_spec_prompts(
        raw_rows,
        energy_key=paired_energy_key,
        allow_incomplete_energy=allow_incomplete_energy,
    )
    output_equivalence = output_equivalence_report(
        paired_prompt_rows,
        require_checked_output=not allow_unchecked_output,
    )
    system_boundary = system_boundary_report(
        raw_rows=raw_rows,
        paired_prompt_rows=paired_prompt_rows,
        paired_summary_rows=paired_summary_rows,
        require_two_device_boundary=require_two_device_boundary,
    )
    feasible_rows, pareto_rows, best_summary = select_configs(
        summary_rows,
        energy_key=summary_energy_key,
        min_tokens_per_s=min_tokens_per_s,
        max_wall_latency_ms=max_wall_latency_ms,
        min_runs=min_runs,
        min_prompts=min_prompts,
    )
    optimization = select_best_config.optimization_summary(
        feasible_rows=feasible_rows,
        pareto_rows=pareto_rows,
        selected=best_summary,
        energy_key=summary_energy_key,
        constraints={
            "algorithm": "speculative",
            "min_tokens_per_s": min_tokens_per_s,
            "max_wall_latency_ms": max_wall_latency_ms,
            "min_runs": min_runs,
            "min_prompts": min_prompts,
            "min_gammas": min_gammas,
        },
    )
    system_summary_energy_key = (
        SYSTEM_ACTIVE_SUMMARY_ENERGY_KEY
        if validate_results.plan_requires_idle(plans)
        else SYSTEM_TOTAL_SUMMARY_ENERGY_KEY
    )
    (
        system_feasible_rows,
        system_pareto_rows,
        best_system_summary,
    ) = select_configs(
        summary_rows,
        energy_key=system_summary_energy_key,
        min_tokens_per_s=min_tokens_per_s,
        max_wall_latency_ms=max_wall_latency_ms,
        min_runs=min_runs,
        min_prompts=min_prompts,
    )
    system_optimization = select_best_config.optimization_summary(
        feasible_rows=system_feasible_rows,
        pareto_rows=system_pareto_rows,
        selected=best_system_summary,
        energy_key=system_summary_energy_key,
        constraints={
            "algorithm": "speculative",
            "min_tokens_per_s": min_tokens_per_s,
            "max_wall_latency_ms": max_wall_latency_ms,
            "min_runs": min_runs,
            "min_prompts": min_prompts,
            "min_gammas": min_gammas,
        },
    )
    gamma_policy = select_best_config.policy_report(
        feasible_rows=system_feasible_rows,
        energy_key=system_summary_energy_key,
        constraints=system_optimization["constraints"],
    )
    interaction = interaction_report(
        system_feasible_rows,
        energy_key=system_summary_energy_key,
        require_interaction_analysis=require_interaction_analysis,
    )
    best_paired = paired_summary_rows[0] if paired_summary_rows else None
    claim_readiness = claim_readiness_report(
        gamma_design=gamma_design,
        gamma_statistics=gamma_statistics,
        gamma_trend=gamma_trend,
        output_equivalence=output_equivalence,
        system_boundary=system_boundary,
        system_optimization=system_optimization,
        gamma_policy=gamma_policy,
        interaction=interaction,
        best_paired=best_paired,
        unpaired_prompt_rows=unpaired_prompt_rows,
        paired_summary_rows=paired_summary_rows,
        min_gammas=min_gammas,
    )
    report_ok = (
        bool(plan_design["ok"])
        and bool(doctor_design["ok"])
        and bool(plan_audit["ok"])
        and bool(k8s_manifest_audit["ok"])
        and bool(energy_design["ok"])
        and bool(plan_integrity["ok"])
        and bool(schema_contract["ok"])
        and bool(frequency_consistency["ok"])
        and bool(measurement_setup["ok"])
        and bool(provenance["ok"])
        and bool(model_setup["ok"])
        and bool(input_consistency["ok"])
        and bool(communication["ok"])
        and bool(trace_consistency["ok"])
        and bool(network_probe["ok"])
        and bool(timing["ok"])
        and bool(accounting["ok"])
        and bool(energy_signal["ok"])
        and bool(runtime_status["ok"])
        and bool(token_compatibility["ok"])
        and bool(measurement_stability["ok"])
        and bool(gamma_design["ok"])
        and bool(gamma_statistics["ok"])
        and bool(gamma_trend["ok"])
        and bool(validation["ok"])
        and (not require_two_device_boundary or bool(system_boundary["ok"]))
        and bool(summary_rows)
        and bool(best_summary)
        and bool(optimization["ok"])
        and bool(system_optimization["ok"])
        and bool(gamma_policy["ok"])
        and bool(interaction["ok"])
        and (not require_claim_readiness or bool(claim_readiness["ok"]))
        and (bool(best_paired) or allow_unpaired)
        and (not unpaired_prompt_rows or allow_unpaired)
        and bool(output_equivalence["ok"])
    )
    return {
        "ok": report_ok,
        "summary_energy_key": summary_energy_key,
        "paired_energy_key": paired_energy_key,
        "paired_bootstrap_samples": paired_bootstrap_samples,
        "paired_bootstrap_seed": paired_bootstrap_seed,
        "require_claim_readiness": require_claim_readiness,
        "require_two_device_boundary": require_two_device_boundary,
        "plan_design": plan_design,
        "doctor_design": doctor_design,
        "plan_audit": plan_audit,
        "k8s_manifest_audit": k8s_manifest_audit,
        "energy_design": energy_design,
        "plan_integrity": plan_integrity,
        "schema_contract": schema_contract,
        "frequency_consistency": frequency_consistency,
        "measurement_setup": measurement_setup,
        "provenance": provenance,
        "model_setup": model_setup,
        "input_consistency": input_consistency,
        "communication": communication,
        "trace_consistency": trace_consistency,
        "network_probe": network_probe,
        "timing": timing,
        "accounting": accounting,
        "energy_signal": energy_signal,
        "runtime_status": runtime_status,
        "token_compatibility": token_compatibility,
        "measurement_stability": measurement_stability,
        "gamma_design": gamma_design,
        "gamma_statistics": gamma_statistics,
        "gamma_trend": gamma_trend,
        "validation": validation,
        "output_equivalence": output_equivalence,
        "system_boundary": system_boundary,
        "summary_rows": len(summary_rows),
        "paired_summary_rows": len(paired_summary_rows),
        "paired_prompt_rows": len(paired_prompt_rows),
        "unpaired_prompt_rows": len(unpaired_prompt_rows),
        "gamma_effect_rows": len(gamma_effect_rows),
        "feasible_rows": len(feasible_rows),
        "pareto_rows": len(pareto_rows),
        "optimization": optimization,
        "system_optimization": system_optimization,
        "gamma_policy": gamma_policy,
        "interaction": interaction,
        "claim_readiness": claim_readiness,
        "best_summary_config": best_summary,
        "best_paired_config": best_paired,
        "_summary_rows": summary_rows,
        "_paired_summary_rows": paired_summary_rows,
        "_paired_prompt_rows": paired_prompt_rows,
        "_unpaired_prompt_rows": unpaired_prompt_rows,
        "_gamma_effect_rows": gamma_effect_rows,
        "_pareto_rows": pareto_rows,
        "_gamma_policy_rows": gamma_policy["policy_by_gamma"],
    }


def write_outputs(
    report: Dict[str, object],
    out_dir: Path,
    input_files: Sequence[Tuple[str, str]] = (),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_spec_results.write_csv(
        str(out_dir / "summary.csv"),
        report["_summary_rows"],
    )
    paired_prompt_compare.write_csv(
        str(out_dir / "paired_prompt_summary.csv"),
        report["_paired_summary_rows"],
        paired_prompt_compare.SUMMARY_FIELDNAMES,
    )
    paired_prompt_compare.write_csv(
        str(out_dir / "paired_prompt_rows.csv"),
        report["_paired_prompt_rows"],
        paired_prompt_compare.PAIR_FIELDNAMES,
    )
    paired_prompt_compare.write_csv(
        str(out_dir / "unpaired_prompt_rows.csv"),
        report["_unpaired_prompt_rows"],
        paired_prompt_compare.UNPAIRED_FIELDNAMES,
    )
    analyze_gamma_effect.write_csv(
        str(out_dir / "gamma_effect_summary.csv"),
        report["_gamma_effect_rows"],
    )
    select_best_config.write_csv(
        str(out_dir / "pareto_configs.csv"),
        report["_pareto_rows"],
    )
    select_best_config.write_csv(
        str(out_dir / "gamma_frequency_policy.csv"),
        report["_gamma_policy_rows"],
    )
    public_report = {
        key: value for key, value in report.items() if not key.startswith("_")
    }
    report["figure_manifest"] = plot_experiment.write_plot_bundle(
        summary_rows=report["_summary_rows"],
        gamma_effect_rows=report["_gamma_effect_rows"],
        paired_summary_rows=report["_paired_summary_rows"],
        out_dir=out_dir / "figures",
        summary_energy_key=str(report["summary_energy_key"]),
    )
    public_report["figure_manifest"] = report["figure_manifest"]
    write_json(out_dir / "validation_report.json", report["validation"])
    write_json(out_dir / "doctor_report.json", report["doctor_design"])
    write_json(out_dir / "plan_audit_report.json", report["plan_audit"])
    write_json(out_dir / "k8s_manifest_audit_report.json", report["k8s_manifest_audit"])
    write_json(out_dir / "plan_integrity_report.json", report["plan_integrity"])
    write_json(out_dir / "schema_contract_report.json", report["schema_contract"])
    write_json(
        out_dir / "frequency_consistency_report.json",
        report["frequency_consistency"],
    )
    write_json(out_dir / "measurement_setup_report.json", report["measurement_setup"])
    write_json(out_dir / "provenance_report.json", report["provenance"])
    write_json(out_dir / "model_setup_report.json", report["model_setup"])
    write_json(out_dir / "input_consistency_report.json", report["input_consistency"])
    write_json(out_dir / "communication_report.json", report["communication"])
    write_json(out_dir / "trace_consistency_report.json", report["trace_consistency"])
    write_json(out_dir / "network_probe_report.json", report["network_probe"])
    write_json(out_dir / "timing_report.json", report["timing"])
    write_json(out_dir / "accounting_report.json", report["accounting"])
    write_json(out_dir / "energy_signal_report.json", report["energy_signal"])
    write_json(out_dir / "runtime_status_report.json", report["runtime_status"])
    write_json(out_dir / "token_compatibility_report.json", report["token_compatibility"])
    write_json(out_dir / "system_boundary_report.json", report["system_boundary"])
    write_json(
        out_dir / "measurement_stability_report.json",
        report["measurement_stability"],
    )
    write_json(out_dir / "gamma_statistics_report.json", report["gamma_statistics"])
    write_json(out_dir / "gamma_trend_report.json", report["gamma_trend"])
    write_json(out_dir / "optimization_report.json", report["optimization"])
    write_json(
        out_dir / "system_optimization_report.json",
        report["system_optimization"],
    )
    write_json(out_dir / "gamma_policy_report.json", report["gamma_policy"])
    write_json(out_dir / "interaction_report.json", report["interaction"])
    write_json(out_dir / "claim_readiness_report.json", report["claim_readiness"])
    write_json(out_dir / "report.json", public_report)
    write_markdown(out_dir / "REPORT.md", public_report)
    report["artifact_manifest"] = write_artifact_manifest(out_dir, input_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a postprocess report from Xronos experiment plans and raw CSVs."
    )
    parser.add_argument("--plan", nargs="+", required=True, help="Plan JSON file(s)")
    parser.add_argument("--input", nargs="+", required=True, help="Raw result CSV file(s)")
    parser.add_argument(
        "--doctor-json",
        nargs="*",
        default=[],
        help="Optional experiment_doctor JSON report(s).",
    )
    parser.add_argument(
        "--plan-audit-json",
        nargs="*",
        default=[],
        help="Optional pre-run plan_audit JSON report(s).",
    )
    parser.add_argument(
        "--network-json",
        nargs="*",
        default=[],
        help="Optional network_probe JSON report(s).",
    )
    parser.add_argument(
        "--k8s-manifest-audit-json",
        nargs="*",
        default=[],
        help="Optional k8s_manifest_audit JSON report(s).",
    )
    parser.add_argument(
        "--k8s-runbook-json",
        nargs="*",
        default=[],
        help="Optional rendered Kubernetes runbook JSON artifact(s).",
    )
    parser.add_argument(
        "--k8s-runbook-markdown",
        nargs="*",
        default=[],
        help="Optional rendered Kubernetes runbook Markdown artifact(s).",
    )
    parser.add_argument(
        "--trace-jsonl",
        nargs="*",
        default=[],
        help="Optional raw trace JSONL file(s) from the baseline/spec drivers.",
    )
    parser.add_argument("--out-dir", default="xronos_experiment_report")
    parser.add_argument("--summary-energy-key", default=DEFAULT_SUMMARY_ENERGY_KEY)
    parser.add_argument("--paired-energy-key", default=DEFAULT_PAIRED_ENERGY_KEY)
    parser.add_argument("--min-tokens-per-s", type=float, default=0.0)
    parser.add_argument("--max-wall-latency-ms", type=float)
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-prompts", type=int, default=1)
    parser.add_argument(
        "--min-gammas",
        type=int,
        default=2,
        help="Minimum distinct gamma values required for a claim-ready sweep.",
    )
    parser.add_argument(
        "--min-power-samples",
        type=int,
        default=1,
        help="Minimum required power samples per required active phase.",
    )
    parser.add_argument("--paired-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--paired-bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--max-runtime-temp-c",
        type=float,
        help="Fail if any recorded runtime temperature exceeds this value.",
    )
    parser.add_argument(
        "--fail-on-throttle",
        action="store_true",
        help="Fail if any recorded runtime throttle flag is active.",
    )
    parser.add_argument(
        "--max-energy-cv",
        type=float,
        help="Fail if summary energy coefficient of variation exceeds this value.",
    )
    parser.add_argument(
        "--max-latency-cv",
        type=float,
        help="Fail if summary wall-latency coefficient of variation exceeds this value.",
    )
    parser.add_argument(
        "--require-doctor",
        action="store_true",
        help="Fail unless drafter and verifier doctor reports are present and passing.",
    )
    parser.add_argument(
        "--require-driver-doctor",
        action="store_true",
        help="With --require-doctor, also require a passing driver doctor report.",
    )
    parser.add_argument(
        "--require-plan-audit",
        action="store_true",
        help="Fail unless a passing pre-run plan audit report is present.",
    )
    parser.add_argument(
        "--require-k8s-manifest-audit",
        action="store_true",
        help="Fail unless a passing Kubernetes manifest audit report is present.",
    )
    parser.add_argument(
        "--require-network-probe",
        action="store_true",
        help="Fail unless a passing drafter/verifier network probe report is present.",
    )
    parser.add_argument(
        "--require-trace",
        action="store_true",
        help="Fail unless trace JSONL covers every measured CSV session.",
    )
    parser.add_argument(
        "--require-interaction-analysis",
        action="store_true",
        help="Fail unless gamma/verifier-clock interaction can be evaluated on a complete factorial grid.",
    )
    parser.add_argument(
        "--require-claim-readiness",
        action="store_true",
        help="Fail unless every proposal claim category in claim_readiness_report.json is ready.",
    )
    parser.add_argument(
        "--require-two-device-boundary",
        action="store_true",
        help=(
            "Fail unless speculative/baseline prompt pairs use the two-device "
            "active/idle-drafter system boundary."
        ),
    )
    parser.add_argument(
        "--max-network-mean-rtt-ms",
        type=float,
        help="Fail if any network probe target mean RTT exceeds this value.",
    )
    parser.add_argument(
        "--max-network-p95-rtt-ms",
        type=float,
        help="Fail if any network probe target p95 RTT exceeds this value.",
    )
    parser.add_argument("--allow-incomplete-energy", action="store_true")
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Do not fail when speculative prompt rows lack verifier-only baselines.",
    )
    parser.add_argument(
        "--allow-unchecked-output",
        action="store_true",
        help="Do not fail when paired rows lack output token hashes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = [validate_results.read_json(path) for path in args.plan]
    doctor_reports = [validate_results.read_json(path) for path in args.doctor_json]
    plan_audit_reports = [
        validate_results.read_json(path) for path in args.plan_audit_json
    ]
    k8s_manifest_audit_reports = [
        validate_results.read_json(path) for path in args.k8s_manifest_audit_json
    ]
    network_reports = [validate_results.read_json(path) for path in args.network_json]
    trace_events = read_jsonl_events(args.trace_jsonl)
    raw_rows = validate_results.read_csvs(args.input)
    report = build_report(
        plans=plans,
        raw_rows=raw_rows,
        doctor_reports=doctor_reports,
        plan_audit_reports=plan_audit_reports,
        k8s_manifest_audit_reports=k8s_manifest_audit_reports,
        network_reports=network_reports,
        trace_events=trace_events,
        summary_energy_key=args.summary_energy_key,
        paired_energy_key=args.paired_energy_key,
        allow_incomplete_energy=args.allow_incomplete_energy,
        allow_unpaired=args.allow_unpaired,
        allow_unchecked_output=args.allow_unchecked_output,
        min_tokens_per_s=args.min_tokens_per_s,
        max_wall_latency_ms=args.max_wall_latency_ms,
        min_runs=args.min_runs,
        min_prompts=args.min_prompts,
        min_gammas=args.min_gammas,
        min_power_samples=args.min_power_samples,
        paired_bootstrap_samples=args.paired_bootstrap_samples,
        paired_bootstrap_seed=args.paired_bootstrap_seed,
        require_doctor=args.require_doctor,
        require_driver_doctor=args.require_driver_doctor,
        require_plan_audit=args.require_plan_audit,
        require_k8s_manifest_audit=args.require_k8s_manifest_audit,
        require_network_probe=args.require_network_probe,
        require_trace=args.require_trace,
        require_interaction_analysis=args.require_interaction_analysis,
        require_claim_readiness=args.require_claim_readiness,
        require_two_device_boundary=args.require_two_device_boundary,
        max_network_mean_rtt_ms=args.max_network_mean_rtt_ms,
        max_network_p95_rtt_ms=args.max_network_p95_rtt_ms,
        max_runtime_temp_c=args.max_runtime_temp_c,
        fail_on_throttle=args.fail_on_throttle,
        max_energy_cv=args.max_energy_cv,
        max_latency_cv=args.max_latency_cv,
    )
    out_dir = Path(args.out_dir)
    input_files = (
        [("plan", path) for path in args.plan]
        + [("result_csv", path) for path in args.input]
        + [("doctor_json", path) for path in args.doctor_json]
        + [("plan_audit_json", path) for path in args.plan_audit_json]
        + [("network_json", path) for path in args.network_json]
        + [
            ("k8s_manifest_audit_json", path)
            for path in args.k8s_manifest_audit_json
        ]
        + [("k8s_runbook_json", path) for path in args.k8s_runbook_json]
        + [
            ("k8s_runbook_markdown", path)
            for path in args.k8s_runbook_markdown
        ]
        + [("trace_jsonl", path) for path in args.trace_jsonl]
    )
    write_outputs(report, out_dir, input_files=input_files)
    print(f"report_ok={int(bool(report['ok']))}")
    print(f"plan_design_ok={int(bool(report['plan_design']['ok']))}")
    print(f"doctor_design_ok={int(bool(report['doctor_design']['ok']))}")
    print(f"plan_audit_ok={int(bool(report['plan_audit']['ok']))}")
    print(
        "k8s_manifest_audit_ok="
        f"{int(bool(report['k8s_manifest_audit']['ok']))}"
    )
    print(f"energy_design_ok={int(bool(report['energy_design']['ok']))}")
    print(f"plan_integrity_ok={int(bool(report['plan_integrity']['ok']))}")
    print(
        "plan_design_hash_mismatch_sessions="
        f"{report['plan_integrity']['mismatched_result_plan_design_hash_sessions']}"
    )
    print(f"schema_contract_ok={int(bool(report['schema_contract']['ok']))}")
    print(
        "frequency_consistency_ok="
        f"{int(bool(report['frequency_consistency']['ok']))}"
    )
    print(f"measurement_setup_ok={int(bool(report['measurement_setup']['ok']))}")
    print(f"provenance_ok={int(bool(report['provenance']['ok']))}")
    print(f"model_setup_ok={int(bool(report['model_setup']['ok']))}")
    print(f"input_consistency_ok={int(bool(report['input_consistency']['ok']))}")
    print(f"communication_ok={int(bool(report['communication']['ok']))}")
    print(f"trace_consistency_ok={int(bool(report['trace_consistency']['ok']))}")
    print(f"network_probe_ok={int(bool(report['network_probe']['ok']))}")
    print(f"timing_ok={int(bool(report['timing']['ok']))}")
    print(f"accounting_ok={int(bool(report['accounting']['ok']))}")
    print(f"energy_signal_ok={int(bool(report['energy_signal']['ok']))}")
    print(f"runtime_status_ok={int(bool(report['runtime_status']['ok']))}")
    print(f"token_compatibility_ok={int(bool(report['token_compatibility']['ok']))}")
    print(
        "measurement_stability_ok="
        f"{int(bool(report['measurement_stability']['ok']))}"
    )
    print(f"gamma_design_ok={int(bool(report['gamma_design']['ok']))}")
    print(f"gamma_statistics_ok={int(bool(report['gamma_statistics']['ok']))}")
    print(f"gamma_trend_ok={int(bool(report['gamma_trend']['ok']))}")
    print(f"interaction_ok={int(bool(report['interaction']['ok']))}")
    print(f"claim_readiness_ok={int(bool(report['claim_readiness']['ok']))}")
    print(f"system_boundary_ok={int(bool(report['system_boundary']['ok']))}")
    print(
        "system_boundary_claim_ready="
        f"{int(bool(report['system_boundary']['claim_ready']))}"
    )
    print(
        "missing_baseline_plan_conditions="
        f"{report['plan_design']['missing_baseline_conditions']}"
    )
    print(
        "missing_two_device_baseline_plan_conditions="
        f"{report['plan_design']['missing_two_device_baseline_conditions']}"
    )
    print(f"plan_gamma_values={','.join(report['plan_design']['spec_gamma_values'])}")
    print(f"missing_plan_tokenizers={report['plan_design']['missing_tokenizer_plans']}")
    print(f"incomplete_gamma_groups={len(report['plan_design']['incomplete_gamma_groups'])}")
    print(f"doctor_failures={report['doctor_design']['failures']}")
    print(f"plan_audit_reports={report['plan_audit']['plan_audit_reports']}")
    print(f"k8s_manifest_audit_reports={report['k8s_manifest_audit']['reports']}")
    print(f"gamma_ready_configs={report['gamma_design']['ready_configs']}")
    print(f"gamma_min_configs={report['gamma_design']['min_gamma_configs']}")
    print(f"gamma_trend_valid_groups={report['gamma_trend']['valid_trend_groups']}")
    print(f"summary_rows={report['summary_rows']}")
    print(f"paired_summary_rows={report['paired_summary_rows']}")
    print(f"paired_prompt_rows={report['paired_prompt_rows']}")
    print(f"unpaired_prompt_rows={report['unpaired_prompt_rows']}")
    print(f"gamma_effect_rows={report['gamma_effect_rows']}")
    print(
        "unstable_output_prompt_pairs="
        f"{report['output_equivalence']['unstable_prompt_pairs']}"
    )
    print(f"pareto_rows={report['pareto_rows']}")
    print(f"figure_plots={report.get('figure_manifest', {}).get('ok_plots', 0)}")
    print(
        "artifact_manifest_outputs="
        f"{len(report.get('artifact_manifest', {}).get('outputs', []))}"
    )
    print(f"Wrote report artifacts to {out_dir}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
