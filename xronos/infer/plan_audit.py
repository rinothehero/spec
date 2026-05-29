import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from xronos.infer import experiment_report, validate_results


SCHEMA_VERSION = "xronos-plan-audit-v2"


def expected_session_counts(plans: Sequence[Dict[str, object]]) -> Dict[str, int]:
    measured = 0
    warmup = 0
    idle = 0
    combinations = 0
    for plan in plans:
        plan_combinations = plan.get("combinations", [])
        if not isinstance(plan_combinations, list):
            continue
        combinations += len(plan_combinations)
        measured += sum(
            int(combo.get("measured_runs", plan.get("measured_runs", 1)))
            for combo in plan_combinations
            if isinstance(combo, dict)
        )
        warmup_runs = int(plan.get("warmup_runs", 0) or 0)
        warmup += len(plan_combinations) * max(0, warmup_runs)
        idle += int(plan.get("total_idle_baselines", 0) or 0)
    return {
        "combinations": combinations,
        "measured_sessions": measured,
        "warmup_sessions": warmup,
        "idle_baselines": idle,
    }


def build_audit(
    plans: List[Dict[str, object]],
    plan_paths: Sequence[str],
    min_runs: int,
    min_prompts: int,
    min_gammas: int,
    summary_energy_key: str,
    paired_energy_key: str,
    allow_unpaired: bool = False,
    require_two_device_boundary: bool = False,
) -> Dict[str, object]:
    plan_design = experiment_report.plan_design_report(
        plans,
        allow_unpaired=allow_unpaired,
        min_prompts=min_prompts,
        min_runs=min_runs,
        min_gammas=min_gammas,
        require_two_device_boundary=require_two_device_boundary,
    )
    energy_design = experiment_report.energy_design_report(
        plans,
        summary_energy_key=summary_energy_key,
        paired_energy_key=paired_energy_key,
    )
    plan_integrity = experiment_report.plan_integrity_report(plans, raw_rows=[])

    errors = []
    for section_name, section in (
        ("plan_design", plan_design),
        ("energy_design", energy_design),
        ("plan_integrity", plan_integrity),
    ):
        if not section.get("ok", False):
            errors.extend(
                f"{section_name}:{error}" for error in section.get("errors", [])
            )

    algorithms = sorted({str(plan.get("algorithm", "")) for plan in plans})
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": not errors,
        "plan_paths": list(plan_paths),
        "plan_count": len(plans),
        "algorithms": algorithms,
        "min_runs": min_runs,
        "min_prompts": min_prompts,
        "min_gammas": min_gammas,
        "summary_energy_key": summary_energy_key,
        "paired_energy_key": paired_energy_key,
        "allow_unpaired": allow_unpaired,
        "require_two_device_boundary": require_two_device_boundary,
        "expected": expected_session_counts(plans),
        "plan_design": plan_design,
        "energy_design": energy_design,
        "plan_integrity": plan_integrity,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Xronos experiment plan JSONs before running a long sweep."
    )
    parser.add_argument("--plan", nargs="+", required=True, help="Plan JSON file(s).")
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-prompts", type=int, default=1)
    parser.add_argument(
        "--min-gammas",
        type=int,
        default=2,
        help="Minimum distinct speculative gamma values required in the plan.",
    )
    parser.add_argument(
        "--summary-energy-key",
        default=experiment_report.DEFAULT_SUMMARY_ENERGY_KEY,
    )
    parser.add_argument(
        "--paired-energy-key",
        default=experiment_report.DEFAULT_PAIRED_ENERGY_KEY,
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Do not fail when the baseline plan does not cover every spec prompt.",
    )
    parser.add_argument(
        "--require-two-device-boundary",
        action="store_true",
        help=(
            "Require the verifier-only baseline plan to be the two-device "
            "idle-drafter baseline for the speculative frequency grid."
        ),
    )
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = [validate_results.read_json(path) for path in args.plan]
    audit = build_audit(
        plans=plans,
        plan_paths=args.plan,
        min_runs=args.min_runs,
        min_prompts=args.min_prompts,
        min_gammas=args.min_gammas,
        summary_energy_key=args.summary_energy_key,
        paired_energy_key=args.paired_energy_key,
        allow_unpaired=args.allow_unpaired,
        require_two_device_boundary=args.require_two_device_boundary,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"Wrote plan audit to {out_path}")

    print(f"plan_audit_ok={int(bool(audit['ok']))}")
    print(f"plans={audit['plan_count']}")
    print(f"algorithms={','.join(audit['algorithms'])}")
    print(f"expected_measured_sessions={audit['expected']['measured_sessions']}")
    print(f"expected_warmup_sessions={audit['expected']['warmup_sessions']}")
    print(f"plan_design_ok={int(bool(audit['plan_design']['ok']))}")
    print(f"energy_design_ok={int(bool(audit['energy_design']['ok']))}")
    print(f"plan_integrity_ok={int(bool(audit['plan_integrity']['ok']))}")
    print(f"require_two_device_boundary={int(bool(audit['require_two_device_boundary']))}")
    print(f"errors={len(audit['errors'])}")
    if not audit["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
