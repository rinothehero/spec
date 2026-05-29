import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from xronos.infer import k8s_manifest_audit


SCHEMA_VERSION = "xronos-k8s-runbook-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_namespace(docs: Dict[Tuple[str, str], str], override: str = "") -> str:
    if override:
        return override
    namespaces = sorted(
        {
            k8s_manifest_audit._field_value(doc, "namespace")
            for doc in docs.values()
            if k8s_manifest_audit._field_value(doc, "namespace")
        }
    )
    return namespaces[0] if namespaces else "default"


def kubectl_patch_job_command(namespace: str, job_name: str) -> str:
    payload = json.dumps({"spec": {"suspend": False}}, separators=(",", ":"))
    return (
        f"kubectl patch job {job_name} -n {namespace} "
        f"--type merge -p '{payload}'"
    )


def kubectl_wait_job_command(namespace: str, job_name: str, timeout_s: int) -> str:
    return (
        f"kubectl wait job/{job_name} -n {namespace} "
        f"--for=condition=complete --timeout={timeout_s}s"
    )


def kubectl_logs_command(namespace: str, job_name: str) -> str:
    return f"kubectl logs job/{job_name} -n {namespace}"


def kubectl_get_job_command(namespace: str, job_name: str) -> str:
    return f"kubectl get job {job_name} -n {namespace} -o wide"


def build_preconditions(
    namespace: str,
    manifest_path: str,
    deployment_timeout_s: int,
) -> List[Dict[str, object]]:
    return [
        {
            "name": "apply_manifest",
            "description": "Create namespace, services, deployments, PVC bindings, and suspended jobs.",
            "command": f"kubectl apply -f {manifest_path}",
        },
        {
            "name": "wait_drafter_deployment",
            "description": "Wait until the Jetson drafter service pod has loaded the model and opened its gRPC port.",
            "command": (
                "kubectl rollout status deployment/spec-drafter "
                f"-n {namespace} --timeout={deployment_timeout_s}s"
            ),
        },
        {
            "name": "wait_verifier_deployment",
            "description": "Wait until the GPU verifier service pod has loaded the model and opened its gRPC port.",
            "command": (
                "kubectl rollout status deployment/spec-verifier "
                f"-n {namespace} --timeout={deployment_timeout_s}s"
            ),
        },
        {
            "name": "check_services",
            "description": "Confirm that both gRPC services exist before resuming jobs.",
            "command": (
                "kubectl get svc spec-drafter spec-verifier "
                f"-n {namespace} -o wide"
            ),
        },
    ]


def build_steps(
    runbook_report: Dict[str, object],
    namespace: str,
    job_timeout_s: int,
) -> List[Dict[str, object]]:
    steps = []
    jobs = sorted(
        [
            row
            for row in runbook_report.get("jobs", [])
            if isinstance(row, dict) and row.get("name")
        ],
        key=lambda row: int(row.get("order") or 9999),
    )
    for row in jobs:
        job_name = str(row["name"])
        order = int(row.get("order") or 0)
        steps.append(
            {
                "order": order,
                "job": job_name,
                "run_after": list(row.get("run_after", [])),
                "produces": list(row.get("produces", [])),
                "consumes": list(row.get("consumes", [])),
                "resume_command": kubectl_patch_job_command(namespace, job_name),
                "wait_command": kubectl_wait_job_command(
                    namespace,
                    job_name,
                    job_timeout_s,
                ),
                "status_command": kubectl_get_job_command(namespace, job_name),
                "logs_command": kubectl_logs_command(namespace, job_name),
            }
        )
    return steps


def build_runbook(
    manifest_text: str,
    manifest_path: str,
    namespace: str = "",
    deployment_timeout_s: int = 1800,
    job_timeout_s: int = 7200,
) -> Dict[str, object]:
    docs = k8s_manifest_audit.docs_by_key(manifest_text)
    audit = k8s_manifest_audit.build_audit(manifest_text)
    resolved_namespace = infer_namespace(docs, override=namespace)
    runbook = audit.get("runbook", {})
    preconditions = build_preconditions(
        namespace=resolved_namespace,
        manifest_path=manifest_path,
        deployment_timeout_s=deployment_timeout_s,
    )
    steps = build_steps(
        runbook_report=runbook if isinstance(runbook, dict) else {},
        namespace=resolved_namespace,
        job_timeout_s=job_timeout_s,
    )
    errors = []
    if not audit.get("ok", False):
        errors.append("manifest_audit_not_ok")
    if not isinstance(runbook, dict) or not runbook.get("ok", False):
        errors.append("runbook_annotation_not_ok")
    if not steps:
        errors.append("no_runbook_steps")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": not errors,
        "never_executes_kubectl": True,
        "manifest_path": manifest_path,
        "namespace": resolved_namespace,
        "deployment_timeout_s": deployment_timeout_s,
        "job_timeout_s": job_timeout_s,
        "preconditions": preconditions,
        "steps": steps,
        "audit_errors": list(audit.get("errors", [])),
        "runbook_errors": list(runbook.get("errors", []))
        if isinstance(runbook, dict)
        else ["missing_runbook_report"],
        "errors": errors,
    }


def write_json(path: str, payload: Dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def markdown_lines(payload: Dict[str, object]) -> List[str]:
    lines = [
        "# Xronos Kubernetes Experiment Runbook",
        "",
        f"- ok: {payload['ok']}",
        f"- namespace: {payload['namespace']}",
        f"- manifest: {payload['manifest_path']}",
        f"- generated_at_utc: {payload['created_at_utc']}",
        f"- never_executes_kubectl: {payload['never_executes_kubectl']}",
        "",
        "## Preconditions",
        "",
    ]
    for item in payload.get("preconditions", []):
        lines.extend(
            [
                f"### {item['name']}",
                "",
                str(item["description"]),
                "",
                "```bash",
                str(item["command"]),
                "```",
                "",
            ]
        )
    lines.extend(["## Job Order", ""])
    for step in payload.get("steps", []):
        lines.extend(
            [
                f"### {step['order']}. {step['job']}",
                "",
                f"- after: {', '.join(step['run_after']) if step['run_after'] else '(none)'}",
                f"- consumes: {', '.join(step['consumes']) if step['consumes'] else '(none)'}",
                f"- produces: {', '.join(step['produces']) if step['produces'] else '(none)'}",
                "",
                "```bash",
                str(step["resume_command"]),
                str(step["wait_command"]),
                str(step["logs_command"]),
                "```",
                "",
            ]
        )
    if payload.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in payload["errors"])
        lines.append("")
    return lines


def write_markdown(path: str, payload: Dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(markdown_lines(payload)), encoding="utf-8")


def print_commands(payload: Dict[str, object]) -> None:
    for item in payload.get("preconditions", []):
        print(item["command"])
    for step in payload.get("steps", []):
        print(step["resume_command"])
        print(step["wait_command"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Kubernetes runbook from xronos.run-order annotations. "
            "This command never executes kubectl."
        )
    )
    parser.add_argument("--manifest", default="k8s/spec-decoding.yaml")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--deployment-timeout-s", type=int, default=1800)
    parser.add_argument("--job-timeout-s", type=int, default=7200)
    parser.add_argument("--out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--print-commands", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    payload = build_runbook(
        manifest_text=manifest_text,
        manifest_path=args.manifest,
        namespace=args.namespace,
        deployment_timeout_s=args.deployment_timeout_s,
        job_timeout_s=args.job_timeout_s,
    )
    payload["manifest_sha256"] = file_sha256(manifest_path)
    if args.out:
        write_json(args.out, payload)
        print(f"Wrote Kubernetes runbook JSON to {args.out}")
    if args.markdown_out:
        write_markdown(args.markdown_out, payload)
        print(f"Wrote Kubernetes runbook Markdown to {args.markdown_out}")
    if args.print_commands:
        print_commands(payload)
    print(f"k8s_runbook_ok={int(bool(payload['ok']))}")
    print(f"namespace={payload['namespace']}")
    print(f"preconditions={len(payload['preconditions'])}")
    print(f"steps={len(payload['steps'])}")
    print(f"errors={len(payload['errors'])}")
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
