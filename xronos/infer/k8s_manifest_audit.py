import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


SCHEMA_VERSION = "xronos-k8s-manifest-audit-v2"

REQUIRED_WORKLOADS = [
    ("Job", "drafter-doctor"),
    ("Job", "verifier-doctor"),
    ("Job", "driver-doctor"),
    ("Deployment", "spec-drafter"),
    ("Deployment", "spec-verifier"),
    ("Service", "spec-drafter"),
    ("Service", "spec-verifier"),
    ("Job", "k8s-manifest-audit"),
    ("Job", "network-probe"),
    ("Job", "verifier-baseline-plan"),
    ("Job", "spec-plan"),
    ("Job", "plan-audit"),
    ("Job", "verifier-baseline"),
    ("Job", "spec-driver"),
    ("Job", "spec-report"),
    ("Job", "artifact-audit"),
]

SUSPENDED_JOBS = {
    "drafter-doctor",
    "verifier-doctor",
    "driver-doctor",
    "k8s-manifest-audit",
    "network-probe",
    "verifier-baseline-plan",
    "spec-plan",
    "plan-audit",
    "verifier-baseline",
    "spec-driver",
    "spec-report",
    "artifact-audit",
}

SPEC_DESIGN_FLAGS = [
    "--tokenizer",
    "--prompts-jsonl",
    "--gammas",
    "--drafter-freqs-hz",
    "--verifier-clocks-mhz",
    "--max-new-tokens",
    "--stop-token-ids",
    "--idle-baseline-s",
    "--idle-baseline-policy",
    "--shuffle-conditions",
    "--seed",
    "--shuffle-runs",
    "--sample-runtime-metadata",
    "--max-start-temp-c",
    "--warmup-runs",
    "--runs",
]

BASELINE_DESIGN_FLAGS = [
    "--drafter-addr",
    "--tokenizer",
    "--prompts-jsonl",
    "--drafter-freqs-hz",
    "--verifier-clocks-mhz",
    "--max-new-tokens",
    "--stop-token-ids",
    "--idle-baseline-s",
    "--idle-baseline-policy",
    "--shuffle-conditions",
    "--seed",
    "--shuffle-runs",
    "--sample-runtime-metadata",
    "--max-start-temp-c",
    "--warmup-runs",
    "--runs",
]

DESIGN_ENV_NAMES = [
    "TOKENIZER",
    "IDLE_BASELINE_S",
    "SWEEP_SEED",
    "EXPERIMENT_RUNS",
    "MAX_START_TEMP_C",
]

RUNBOOK_JOBS = [
    {
        "name": "drafter-doctor",
        "order": 1,
        "after": [],
        "produces": ["/results/drafter_doctor.json"],
        "consumes": [],
    },
    {
        "name": "verifier-doctor",
        "order": 2,
        "after": ["drafter-doctor"],
        "produces": ["/results/verifier_doctor.json"],
        "consumes": [],
    },
    {
        "name": "driver-doctor",
        "order": 3,
        "after": ["verifier-doctor"],
        "produces": ["/results/driver_doctor.json"],
        "consumes": [],
    },
    {
        "name": "k8s-manifest-audit",
        "order": 4,
        "after": ["driver-doctor"],
        "produces": [
            "/results/k8s_manifest_audit.json",
            "/results/k8s_runbook.json",
            "/results/k8s_runbook.md",
        ],
        "consumes": [],
    },
    {
        "name": "network-probe",
        "order": 5,
        "after": ["k8s-manifest-audit"],
        "produces": ["/results/network_probe.json"],
        "consumes": [],
    },
    {
        "name": "verifier-baseline-plan",
        "order": 6,
        "after": ["network-probe"],
        "produces": ["/results/verifier_baseline_plan.json"],
        "consumes": [],
    },
    {
        "name": "spec-plan",
        "order": 7,
        "after": ["verifier-baseline-plan"],
        "produces": ["/results/spec_plan.json"],
        "consumes": [],
    },
    {
        "name": "plan-audit",
        "order": 8,
        "after": ["spec-plan"],
        "produces": ["/results/plan_audit.json"],
        "consumes": [
            "/results/spec_plan.json",
            "/results/verifier_baseline_plan.json",
        ],
    },
    {
        "name": "verifier-baseline",
        "order": 9,
        "after": ["plan-audit"],
        "produces": [
            "/results/verifier_baseline.csv",
            "/results/verifier_baseline_plan.json",
            "/results/verifier_baseline_trace.jsonl",
        ],
        "consumes": [],
    },
    {
        "name": "spec-driver",
        "order": 10,
        "after": ["verifier-baseline"],
        "produces": [
            "/results/spec_gamma_sweep.csv",
            "/results/spec_plan.json",
            "/results/spec_trace.jsonl",
        ],
        "consumes": [],
    },
    {
        "name": "spec-report",
        "order": 11,
        "after": ["spec-driver"],
        "produces": ["/results/report_gamma_freq"],
        "consumes": [
            "/results/spec_plan.json",
            "/results/verifier_baseline_plan.json",
            "/results/spec_gamma_sweep.csv",
            "/results/verifier_baseline.csv",
            "/results/drafter_doctor.json",
            "/results/verifier_doctor.json",
            "/results/driver_doctor.json",
            "/results/plan_audit.json",
            "/results/network_probe.json",
            "/results/k8s_manifest_audit.json",
            "/results/k8s_runbook.json",
            "/results/k8s_runbook.md",
            "/results/spec_trace.jsonl",
            "/results/verifier_baseline_trace.jsonl",
        ],
    },
    {
        "name": "artifact-audit",
        "order": 12,
        "after": ["spec-report"],
        "produces": ["/results/artifact_audit.json"],
        "consumes": ["/results/report_gamma_freq"],
    },
]


PLACEMENT_REQUIREMENTS = [
    {
        "kind": "Job",
        "name": "drafter-doctor",
        "requires": [
            "xronos-role: jetson-drafter",
            "image: xronos:jetson",
            "nvidia.com/gpu: 1",
            "privileged: true",
            "mountPath: /sys",
            "mountPath: /dev",
        ],
    },
    {
        "kind": "Deployment",
        "name": "spec-drafter",
        "requires": [
            "xronos-role: jetson-drafter",
            "image: xronos:jetson",
            "nvidia.com/gpu: 1",
            "privileged: true",
            "mountPath: /sys",
            "mountPath: /dev",
            "startupProbe:",
            "readinessProbe:",
            "livenessProbe:",
        ],
    },
    {
        "kind": "Job",
        "name": "verifier-doctor",
        "requires": [
            "xronos-role: gpu-verifier",
            "image: xronos:gpu",
            "nvidia.com/gpu: 1",
        ],
    },
    {
        "kind": "Deployment",
        "name": "spec-verifier",
        "requires": [
            "xronos-role: gpu-verifier",
            "image: xronos:gpu",
            "nvidia.com/gpu: 1",
            "startupProbe:",
            "readinessProbe:",
            "livenessProbe:",
        ],
    },
    {
        "kind": "Service",
        "name": "spec-drafter",
        "requires": [
            "app: spec-drafter",
            "port: 50061",
            "targetPort: 50061",
        ],
    },
    {
        "kind": "Service",
        "name": "spec-verifier",
        "requires": [
            "app: spec-verifier",
            "port: 50062",
            "targetPort: 50062",
        ],
    },
]

GPU_DRIVER_JOB_NAMES = [
    "driver-doctor",
    "k8s-manifest-audit",
    "network-probe",
    "verifier-baseline-plan",
    "spec-plan",
    "plan-audit",
    "verifier-baseline",
    "spec-driver",
    "spec-report",
    "artifact-audit",
]


def split_docs(manifest_text: str) -> List[str]:
    return [doc.strip() for doc in manifest_text.split("\n---") if doc.strip()]


def _field_value(doc: str, field: str) -> str:
    prefix = f"{field}:"
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip().strip('"')
    return ""


def doc_key(doc: str) -> Tuple[str, str]:
    return _field_value(doc, "kind"), _field_value(doc, "name")


def docs_by_key(manifest_text: str) -> Dict[Tuple[str, str], str]:
    docs = {}
    for doc in split_docs(manifest_text):
        key = doc_key(doc)
        if all(key):
            docs[key] = doc
    return docs


def command_text(doc: str) -> str:
    start = doc.find("python -m ")
    if start < 0:
        return ""
    end_candidates = [
        index
        for index in (
            doc.find("\n          volumeMounts:", start),
            doc.find("\n          resources:", start),
            doc.find("\n      volumes:", start),
        )
        if index >= 0
    ]
    end = min(end_candidates) if end_candidates else len(doc)
    block = doc[start:end]
    return " ".join(
        line.strip().rstrip("\\").strip()
        for line in block.splitlines()
        if line.strip()
    )


def command_flags(doc: str) -> Dict[str, object]:
    text = command_text(doc)
    if not text:
        return {}
    tokens = shlex.split(text)
    flags: Dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            flags.setdefault(token, tokens[index + 1])
            index += 2
        else:
            flags.setdefault(token, True)
            index += 1
    return flags


def _annotation_values(doc: str) -> Dict[str, str]:
    values = {}
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped.startswith("xronos."):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        values[key] = value.strip().strip('"')
    return values


def _split_annotation_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_order(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def placement_report(docs: Dict[Tuple[str, str], str]) -> Dict[str, object]:
    missing_docs = []
    missing_tokens = []

    requirements = list(PLACEMENT_REQUIREMENTS)
    requirements.extend(
        {
            "kind": "Job",
            "name": name,
            "requires": [
                "xronos-role: gpu-verifier",
                "image: xronos:gpu",
            ],
        }
        for name in GPU_DRIVER_JOB_NAMES
    )

    for requirement in requirements:
        kind = str(requirement["kind"])
        name = str(requirement["name"])
        doc = docs.get((kind, name), "")
        if not doc:
            missing_docs.append({"kind": kind, "name": name})
            continue
        for token in requirement["requires"]:
            if str(token) not in doc:
                missing_tokens.append(
                    {
                        "kind": kind,
                        "name": name,
                        "missing": str(token),
                    }
                )

    errors = []
    if missing_docs:
        errors.append("placement_missing_workload")
    if missing_tokens:
        errors.append("placement_missing_required_token")

    return {
        "ok": not errors,
        "checked_workloads": len(requirements),
        "missing_docs": missing_docs,
        "missing_tokens": missing_tokens,
        "errors": errors,
    }


def env_values(doc: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    lines = doc.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped != "- name:" and not stripped.startswith("- name:"):
            continue
        if stripped == "- name:":
            continue
        name = stripped.split(":", 1)[1].strip().strip('"')
        for next_line in lines[index + 1 : index + 5]:
            next_stripped = next_line.strip()
            if next_stripped.startswith("- name:"):
                break
            if next_stripped.startswith("value:"):
                values[name] = next_stripped.split(":", 1)[1].strip().strip('"')
                break
    return values


def runbook_report(docs: Dict[Tuple[str, str], str]) -> Dict[str, object]:
    expected_jobs = {str(job["name"]): job for job in RUNBOOK_JOBS}
    order_by_job = {}
    producers_by_path: Dict[str, List[Tuple[str, int]]] = {}
    consumes_by_job: Dict[str, List[str]] = {}
    observed_rows = []
    missing_jobs = []
    order_mismatches = []
    after_mismatches = []
    produces_mismatches = []
    consumes_mismatches = []
    command_path_mismatches = []

    for job in RUNBOOK_JOBS:
        name = str(job["name"])
        expected_order = int(job["order"])
        expected_after = list(job["after"])
        expected_produces = list(job["produces"])
        expected_consumes = list(job["consumes"])
        doc = docs.get(("Job", name), "")
        if not doc:
            missing_jobs.append(name)
            continue

        annotations = _annotation_values(doc)
        observed_order = _parse_order(annotations.get("xronos.run-order", ""))
        observed_after = _split_annotation_list(
            annotations.get("xronos.run-after", "")
        )
        observed_produces = _split_annotation_list(
            annotations.get("xronos.produces", "")
        )
        observed_consumes = _split_annotation_list(
            annotations.get("xronos.consumes", "")
        )
        command = command_text(doc)
        observed_rows.append(
            {
                "name": name,
                "order": observed_order,
                "run_after": observed_after,
                "produces": observed_produces,
                "consumes": observed_consumes,
            }
        )
        if observed_order != expected_order:
            order_mismatches.append(
                {
                    "name": name,
                    "expected": expected_order,
                    "observed": observed_order,
                }
            )
        if observed_after != expected_after:
            after_mismatches.append(
                {
                    "name": name,
                    "expected": expected_after,
                    "observed": observed_after,
                }
            )
        if observed_produces != expected_produces:
            produces_mismatches.append(
                {
                    "name": name,
                    "expected": expected_produces,
                    "observed": observed_produces,
                }
            )
        if observed_consumes != expected_consumes:
            consumes_mismatches.append(
                {
                    "name": name,
                    "expected": expected_consumes,
                    "observed": observed_consumes,
                }
            )

        if observed_order is not None:
            order_by_job[name] = observed_order
            for path in expected_produces:
                producers_by_path.setdefault(path, []).append((name, observed_order))
            consumes_by_job[name] = expected_consumes

        for path in expected_produces + expected_consumes:
            if path not in command:
                command_path_mismatches.append(
                    {
                        "name": name,
                        "path": path,
                        "reason": "path_not_in_command",
                    }
                )

    duplicate_orders = []
    order_to_jobs: Dict[int, List[str]] = {}
    for name, order in order_by_job.items():
        order_to_jobs.setdefault(order, []).append(name)
    for order, names in sorted(order_to_jobs.items()):
        if len(names) > 1:
            duplicate_orders.append({"order": order, "jobs": sorted(names)})

    expected_orders = list(range(1, len(RUNBOOK_JOBS) + 1))
    missing_orders = sorted(set(expected_orders) - set(order_by_job.values()))

    invalid_after_order = []
    for job in RUNBOOK_JOBS:
        name = str(job["name"])
        order = order_by_job.get(name)
        if order is None:
            continue
        for dependency in job["after"]:
            dependency_name = str(dependency)
            dependency_order = order_by_job.get(dependency_name)
            if dependency_order is None or dependency_order >= order:
                invalid_after_order.append(
                    {
                        "name": name,
                        "dependency": dependency_name,
                        "job_order": order,
                        "dependency_order": dependency_order,
                    }
                )

    missing_input_producers = []
    for name, paths in sorted(consumes_by_job.items()):
        order = order_by_job.get(name)
        if order is None:
            continue
        for path in paths:
            earlier_producers = [
                producer
                for producer, producer_order in producers_by_path.get(path, [])
                if producer_order < order
            ]
            if not earlier_producers:
                missing_input_producers.append(
                    {
                        "name": name,
                        "path": path,
                        "job_order": order,
                    }
                )

    errors = []
    if missing_jobs:
        errors.append("missing_runbook_jobs")
    if order_mismatches:
        errors.append("runbook_order_mismatch")
    if after_mismatches:
        errors.append("runbook_after_mismatch")
    if produces_mismatches:
        errors.append("runbook_produces_mismatch")
    if consumes_mismatches:
        errors.append("runbook_consumes_mismatch")
    if command_path_mismatches:
        errors.append("runbook_command_path_mismatch")
    if duplicate_orders:
        errors.append("runbook_duplicate_order")
    if missing_orders:
        errors.append("runbook_missing_order")
    if invalid_after_order:
        errors.append("runbook_dependency_order_invalid")
    if missing_input_producers:
        errors.append("runbook_missing_input_producer")

    return {
        "ok": not errors,
        "job_count": len(observed_rows),
        "expected_job_count": len(expected_jobs),
        "jobs": sorted(observed_rows, key=lambda row: row.get("order") or 9999),
        "missing_jobs": missing_jobs,
        "order_mismatches": order_mismatches,
        "after_mismatches": after_mismatches,
        "produces_mismatches": produces_mismatches,
        "consumes_mismatches": consumes_mismatches,
        "command_path_mismatches": command_path_mismatches,
        "duplicate_orders": duplicate_orders,
        "missing_orders": missing_orders,
        "invalid_after_order": invalid_after_order,
        "missing_input_producers": missing_input_producers,
        "errors": errors,
    }


def parity_report(
    docs: Dict[Tuple[str, str], str],
    plan_job: str,
    measured_job: str,
    flags: List[str],
) -> Dict[str, object]:
    plan_doc = docs.get(("Job", plan_job), "")
    measured_doc = docs.get(("Job", measured_job), "")
    plan_flags = command_flags(plan_doc)
    measured_flags = command_flags(measured_doc)
    plan_env = env_values(plan_doc)
    measured_env = env_values(measured_doc)
    mismatched_flags = []
    missing_flags = []
    for flag in flags:
        plan_value = plan_flags.get(flag)
        measured_value = measured_flags.get(flag)
        if plan_value is None or measured_value is None:
            missing_flags.append(
                {
                    "flag": flag,
                    "plan_value": plan_value,
                    "measured_value": measured_value,
                }
            )
        elif plan_value != measured_value:
            mismatched_flags.append(
                {
                    "flag": flag,
                    "plan_value": plan_value,
                    "measured_value": measured_value,
                }
            )

    mismatched_env = []
    for name in DESIGN_ENV_NAMES:
        plan_value = plan_env.get(name)
        measured_value = measured_env.get(name)
        if plan_value is None and measured_value is None:
            continue
        if plan_value != measured_value:
            mismatched_env.append(
                {
                    "name": name,
                    "plan_value": plan_value,
                    "measured_value": measured_value,
                }
            )

    errors = []
    if not plan_doc:
        errors.append("missing_plan_job")
    if not measured_doc:
        errors.append("missing_measured_job")
    if missing_flags:
        errors.append("missing_design_flags")
    if mismatched_flags:
        errors.append("mismatched_design_flags")
    if mismatched_env:
        errors.append("mismatched_design_env")
    return {
        "ok": not errors,
        "plan_job": plan_job,
        "measured_job": measured_job,
        "checked_flags": flags,
        "missing_flags": missing_flags,
        "mismatched_flags": mismatched_flags,
        "mismatched_env": mismatched_env,
        "errors": errors,
    }


def build_audit(manifest_text: str) -> Dict[str, object]:
    docs = docs_by_key(manifest_text)
    missing_workloads = [
        {"kind": kind, "name": name}
        for kind, name in REQUIRED_WORKLOADS
        if (kind, name) not in docs
    ]
    unsuspended_jobs = [
        name
        for name in sorted(SUSPENDED_JOBS)
        if f"  name: {name}" in docs.get(("Job", name), "")
        and "suspend: true" not in docs.get(("Job", name), "")
    ]
    spec_parity = parity_report(
        docs,
        plan_job="spec-plan",
        measured_job="spec-driver",
        flags=SPEC_DESIGN_FLAGS,
    )
    baseline_parity = parity_report(
        docs,
        plan_job="verifier-baseline-plan",
        measured_job="verifier-baseline",
        flags=BASELINE_DESIGN_FLAGS,
    )
    runbook = runbook_report(docs)
    placement = placement_report(docs)

    plan_audit_doc = docs.get(("Job", "plan-audit"), "")
    k8s_manifest_audit_doc = docs.get(("Job", "k8s-manifest-audit"), "")
    report_doc = docs.get(("Job", "spec-report"), "")
    artifact_audit_doc = docs.get(("Job", "artifact-audit"), "")
    required_report_flags = [
        "--require-doctor",
        "--require-driver-doctor",
        "--require-plan-audit",
        "--require-k8s-manifest-audit",
        "--require-network-probe",
        "--require-trace",
        "--require-interaction-analysis",
        "--require-claim-readiness",
        "--require-two-device-boundary",
        "--k8s-manifest-audit-json",
        "--fail-on-throttle",
    ]
    report_flags = command_flags(report_doc)
    missing_report_flags = [
        flag for flag in required_report_flags if flag not in report_flags
    ]
    measured_operational_flags = {
        "verifier-baseline": ["--resume"],
        "spec-driver": ["--resume"],
    }
    missing_measured_operational_flags = []
    for job_name, required_flags in measured_operational_flags.items():
        job_flags = command_flags(docs.get(("Job", job_name), ""))
        for flag in required_flags:
            if flag not in job_flags:
                missing_measured_operational_flags.append(
                    {"job": job_name, "flag": flag}
                )

    errors = []
    if missing_workloads:
        errors.append("missing_required_workloads")
    if unsuspended_jobs:
        errors.append("jobs_not_suspended_by_default")
    if not spec_parity["ok"]:
        errors.append("spec_plan_measured_design_mismatch")
    if not baseline_parity["ok"]:
        errors.append("baseline_plan_measured_design_mismatch")
    k8s_manifest_audit_flags = command_flags(k8s_manifest_audit_doc)
    k8s_manifest_audit_output = k8s_manifest_audit_flags.get("--out", "")
    report_k8s_manifest_audit_input = report_flags.get(
        "--k8s-manifest-audit-json", ""
    )
    if "--out" not in k8s_manifest_audit_flags:
        errors.append("k8s_manifest_audit_missing_output")
    if (
        k8s_manifest_audit_output
        and report_k8s_manifest_audit_input
        and k8s_manifest_audit_output != report_k8s_manifest_audit_input
    ):
        errors.append("k8s_manifest_audit_output_input_mismatch")
    if "--min-gammas" not in command_flags(plan_audit_doc):
        errors.append("plan_audit_missing_min_gammas")
    if "--require-two-device-boundary" not in command_flags(plan_audit_doc):
        errors.append("plan_audit_missing_two_device_boundary_requirement")
    if missing_report_flags:
        errors.append("report_missing_required_flags")
    if "--require-report-ok" not in command_flags(artifact_audit_doc):
        errors.append("artifact_audit_missing_require_report_ok")
    if not runbook["ok"]:
        errors.append("runbook_annotation_mismatch")
    if not placement["ok"]:
        errors.append("placement_mismatch")
    if missing_measured_operational_flags:
        errors.append("measured_job_missing_operational_flags")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": not errors,
        "document_count": len(docs),
        "missing_workloads": missing_workloads,
        "unsuspended_jobs": unsuspended_jobs,
        "spec_parity": spec_parity,
        "baseline_parity": baseline_parity,
        "runbook": runbook,
        "placement": placement,
        "k8s_manifest_audit_output": k8s_manifest_audit_output,
        "report_k8s_manifest_audit_input": report_k8s_manifest_audit_input,
        "missing_report_flags": missing_report_flags,
        "missing_measured_operational_flags": missing_measured_operational_flags,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Xronos speculative-decoding Kubernetes manifest."
    )
    parser.add_argument("--manifest", default="k8s/spec-decoding.yaml")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_text = Path(args.manifest).read_text(encoding="utf-8")
    audit = build_audit(manifest_text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"Wrote Kubernetes manifest audit to {out_path}")
    print(f"k8s_manifest_audit_ok={int(bool(audit['ok']))}")
    print(f"documents={audit['document_count']}")
    print(f"missing_workloads={len(audit['missing_workloads'])}")
    print(f"unsuspended_jobs={len(audit['unsuspended_jobs'])}")
    print(f"spec_parity_ok={int(bool(audit['spec_parity']['ok']))}")
    print(f"baseline_parity_ok={int(bool(audit['baseline_parity']['ok']))}")
    print(f"runbook_ok={int(bool(audit['runbook']['ok']))}")
    print(f"placement_ok={int(bool(audit['placement']['ok']))}")
    print(f"errors={len(audit['errors'])}")
    if not audit["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
