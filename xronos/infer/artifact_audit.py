import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


SCHEMA_VERSION = "xronos-artifact-audit-v3"

ROOT_REQUIRED_FILES = [
    "drafter_doctor.json",
    "driver_doctor.json",
    "k8s_manifest_audit.json",
    "k8s_runbook.json",
    "k8s_runbook.md",
    "network_probe.json",
    "plan_audit.json",
    "spec_gamma_sweep.csv",
    "spec_plan.json",
    "spec_trace.jsonl",
    "verifier_baseline.csv",
    "verifier_baseline_plan.json",
    "verifier_baseline_trace.jsonl",
    "verifier_doctor.json",
]

REPORT_REQUIRED_FILES = [
    "REPORT.md",
    "accounting_report.json",
    "artifact_manifest.json",
    "claim_readiness_report.json",
    "communication_report.json",
    "doctor_report.json",
    "energy_signal_report.json",
    "figures/gamma_drafter_energy.svg",
    "figures/paired_energy_savings.svg",
    "figures/plot_manifest.json",
    "frequency_consistency_report.json",
    "gamma_frequency_policy.csv",
    "gamma_effect_summary.csv",
    "gamma_policy_report.json",
    "gamma_statistics_report.json",
    "gamma_trend_report.json",
    "input_consistency_report.json",
    "interaction_report.json",
    "k8s_manifest_audit_report.json",
    "measurement_setup_report.json",
    "measurement_stability_report.json",
    "model_setup_report.json",
    "network_probe_report.json",
    "optimization_report.json",
    "paired_prompt_rows.csv",
    "paired_prompt_summary.csv",
    "pareto_configs.csv",
    "plan_audit_report.json",
    "plan_integrity_report.json",
    "provenance_report.json",
    "report.json",
    "runtime_status_report.json",
    "schema_contract_report.json",
    "summary.csv",
    "system_boundary_report.json",
    "system_optimization_report.json",
    "timing_report.json",
    "token_compatibility_report.json",
    "trace_consistency_report.json",
    "unpaired_prompt_rows.csv",
    "validation_report.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as f:
        return json.load(f)


def csv_data_rows(path: Path) -> int:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def jsonl_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def file_check(path: Path, base_dir: Path) -> Dict[str, object]:
    try:
        display_path = str(path.relative_to(base_dir))
    except ValueError:
        display_path = str(path)
    record: Dict[str, object] = {
        "path": display_path,
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    record["bytes"] = path.stat().st_size
    record["sha256"] = sha256_file(path)
    if path.suffix == ".json":
        try:
            read_json(path)
            record["json_ok"] = True
        except Exception as exc:
            record["json_ok"] = False
            record["error"] = str(exc)
    elif path.suffix == ".csv":
        try:
            record["data_rows"] = csv_data_rows(path)
            record["csv_ok"] = True
        except Exception as exc:
            record["csv_ok"] = False
            record["error"] = str(exc)
    elif path.suffix == ".jsonl":
        try:
            record["data_rows"] = jsonl_rows(path)
            record["jsonl_ok"] = True
        except Exception as exc:
            record["jsonl_ok"] = False
            record["error"] = str(exc)
    return record


def required_file_report(
    base_dir: Path,
    required_files: List[str],
    label: str,
) -> Dict[str, object]:
    files = [file_check(base_dir / relative_path, base_dir) for relative_path in required_files]
    missing = [item["path"] for item in files if not item.get("exists")]
    invalid_json = [
        item["path"]
        for item in files
        if item["path"].endswith(".json") and item.get("json_ok") is False
    ]
    invalid_csv = [
        item["path"]
        for item in files
        if item["path"].endswith(".csv") and item.get("csv_ok") is False
    ]
    empty_csv = [
        item["path"]
        for item in files
        if item["path"].endswith(".csv") and int(item.get("data_rows", 0) or 0) <= 0
    ]
    invalid_jsonl = [
        item["path"]
        for item in files
        if item["path"].endswith(".jsonl") and item.get("jsonl_ok") is False
    ]
    empty_jsonl = [
        item["path"]
        for item in files
        if item["path"].endswith(".jsonl") and int(item.get("data_rows", 0) or 0) <= 0
    ]
    errors = []
    if missing:
        errors.append(f"missing_{label}_files")
    if invalid_json:
        errors.append(f"invalid_{label}_json")
    if invalid_csv:
        errors.append(f"invalid_{label}_csv")
    if empty_csv:
        errors.append(f"empty_{label}_csv")
    if invalid_jsonl:
        errors.append(f"invalid_{label}_jsonl")
    if empty_jsonl:
        errors.append(f"empty_{label}_jsonl")
    return {
        "ok": not errors,
        "base_dir": str(base_dir),
        "required_count": len(required_files),
        "missing": missing,
        "invalid_json": invalid_json,
        "invalid_csv": invalid_csv,
        "empty_csv": empty_csv,
        "invalid_jsonl": invalid_jsonl,
        "empty_jsonl": empty_jsonl,
        "files": files,
        "errors": errors,
    }


def _resolve_manifest_path(path_text: str, report_dir: Path, results_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    candidates = [
        path,
        results_dir / path,
        report_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def manifest_report(report_dir: Path, results_dir: Path) -> Dict[str, object]:
    manifest_path = report_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return {
            "ok": False,
            "missing_manifest": True,
            "errors": ["missing_artifact_manifest"],
        }

    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return {
            "ok": False,
            "missing_manifest": False,
            "errors": ["invalid_artifact_manifest_json"],
            "error": str(exc),
        }

    mismatches = []
    missing_outputs = []
    missing_inputs = []
    input_mismatches = []
    manifest_outputs = set()
    manifest_inputs = set()
    for item in manifest.get("outputs", []):
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("path", ""))
        if not relative_path:
            continue
        manifest_outputs.add(relative_path)
        path = report_dir / relative_path
        if not path.exists():
            missing_outputs.append(relative_path)
            continue
        expected = str(item.get("sha256", ""))
        if expected and sha256_file(path) != expected:
            mismatches.append(relative_path)

    for item in manifest.get("inputs", []):
        if not isinstance(item, dict):
            continue
        input_path_text = str(item.get("path", ""))
        input_path = _resolve_manifest_path(input_path_text, report_dir, results_dir)
        if input_path_text:
            try:
                manifest_inputs.add(str(input_path.resolve()))
            except OSError:
                manifest_inputs.add(str(input_path))
        if input_path and not input_path.exists():
            missing_inputs.append(str(input_path))
            continue
        expected = str(item.get("sha256", ""))
        if expected and input_path and sha256_file(input_path) != expected:
            input_mismatches.append(str(input_path))

    expected_outputs = sorted(
        relative_path
        for relative_path in REPORT_REQUIRED_FILES
        if relative_path != "artifact_manifest.json"
    )
    missing_required_outputs = sorted(set(expected_outputs) - manifest_outputs)

    expected_inputs = []
    for relative_path in ROOT_REQUIRED_FILES:
        expected_path = results_dir / relative_path
        try:
            expected_inputs.append(str(expected_path.resolve()))
        except OSError:
            expected_inputs.append(str(expected_path))
    missing_required_inputs = sorted(set(expected_inputs) - manifest_inputs)

    errors = []
    if missing_required_outputs:
        errors.append("manifest_missing_required_outputs")
    if missing_required_inputs:
        errors.append("manifest_missing_required_inputs")
    if missing_outputs:
        errors.append("missing_manifest_outputs")
    if mismatches:
        errors.append("manifest_output_hash_mismatch")
    if missing_inputs:
        errors.append("missing_manifest_inputs")
    if input_mismatches:
        errors.append("manifest_input_hash_mismatch")
    return {
        "ok": not errors,
        "missing_manifest": False,
        "schema_version": manifest.get("schema_version", ""),
        "input_count": len(manifest.get("inputs", [])),
        "output_count": len(manifest.get("outputs", [])),
        "required_output_count": len(expected_outputs),
        "required_input_count": len(expected_inputs),
        "missing_required_outputs": missing_required_outputs,
        "missing_required_inputs": missing_required_inputs,
        "missing_outputs": missing_outputs,
        "hash_mismatches": mismatches,
        "missing_inputs": missing_inputs,
        "input_hash_mismatches": input_mismatches,
        "errors": errors,
    }


def report_status(report_dir: Path, require_report_ok: bool) -> Dict[str, object]:
    report_path = report_dir / "report.json"
    if not report_path.exists():
        return {
            "ok": not require_report_ok,
            "exists": False,
            "report_ok": None,
            "errors": ["missing_report_json"] if require_report_ok else [],
        }
    try:
        payload = read_json(report_path)
    except Exception as exc:
        return {
            "ok": False,
            "exists": True,
            "report_ok": None,
            "errors": ["invalid_report_json"],
            "error": str(exc),
        }
    report_ok = bool(payload.get("ok", False))
    errors = []
    if require_report_ok and not report_ok:
        errors.append("report_not_ok")
    return {
        "ok": not errors,
        "exists": True,
        "report_ok": report_ok,
        "errors": errors,
    }


def build_audit(
    results_dir: Path,
    report_dir: Path,
    require_report_ok: bool,
) -> Dict[str, object]:
    root = required_file_report(results_dir, ROOT_REQUIRED_FILES, "root")
    report_files = required_file_report(report_dir, REPORT_REQUIRED_FILES, "report")
    manifest = manifest_report(report_dir, results_dir)
    status = report_status(report_dir, require_report_ok=require_report_ok)
    errors = []
    for section_name, section in (
        ("root", root),
        ("report_files", report_files),
        ("manifest", manifest),
        ("report_status", status),
    ):
        if not section.get("ok", False):
            errors.extend(f"{section_name}:{error}" for error in section.get("errors", []))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": not errors,
        "results_dir": str(results_dir),
        "report_dir": str(report_dir),
        "require_report_ok": require_report_ok,
        "root": root,
        "report_files": report_files,
        "manifest": manifest,
        "report_status": status,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit claim-ready Xronos experiment result artifacts."
    )
    parser.add_argument("--results-dir", default=".")
    parser.add_argument("--report-dir", default="report_gamma_freq")
    parser.add_argument("--out", default="")
    parser.add_argument("--require-report-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = (results_dir / report_dir).resolve()
    audit = build_audit(
        results_dir=results_dir,
        report_dir=report_dir,
        require_report_ok=args.require_report_ok,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"Wrote artifact audit to {out_path}")
    print(f"artifact_audit_ok={int(bool(audit['ok']))}")
    print(f"root_missing={len(audit['root']['missing'])}")
    print(f"report_missing={len(audit['report_files']['missing'])}")
    print(f"manifest_errors={len(audit['manifest']['errors'])}")
    print(f"report_ok={audit['report_status']['report_ok']}")
    if not audit["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
