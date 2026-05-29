import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from xronos.infer import k8s_manifest_audit
from xronos.infer.frequency import discover_jetson_gpu_devfreq_root
from xronos.infer.power import DEFAULT_INA3221_ROOT, INA3221PowerSampler


REQUIRED_GRPC_VERSION = "1.68.1"
REQUIRED_PROTOBUF_VERSION = "5.28.1"


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: Dict[str, object]


def result(
    name: str,
    status: str,
    message: str,
    **details: object,
) -> CheckResult:
    return CheckResult(name=name, status=status, message=message, details=details)


def check_python() -> CheckResult:
    version = platform.python_version()
    if sys.version_info < (3, 8):
        return result("python", "fail", "Python >= 3.8 is required.", version=version)
    return result("python", "ok", "Python version is usable.", version=version)


def check_import(module_name: str) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return result(module_name, "fail", f"Cannot import {module_name}.", error=str(exc))
    version = getattr(module, "__version__", "")
    return result(module_name, "ok", f"{module_name} import works.", version=version)


def _version_tuple(value: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", str(value)))


def version_at_least(found: str, required: str) -> bool:
    found_parts = _version_tuple(found)
    required_parts = _version_tuple(required)
    if not found_parts or not required_parts:
        return False
    width = max(len(found_parts), len(required_parts))
    return found_parts + (0,) * (width - len(found_parts)) >= required_parts + (
        0,
    ) * (width - len(required_parts))


def check_import_min_version(
    module_name: str,
    min_version: str,
    reason: str,
) -> CheckResult:
    imported = check_import(module_name)
    if imported.status != "ok":
        return imported
    found_version = str(imported.details.get("version", ""))
    if not version_at_least(found_version, min_version):
        return result(
            module_name,
            "fail",
            f"{module_name} version is too old for this build.",
            version=found_version,
            required_min_version=min_version,
            reason=reason,
        )
    return result(
        module_name,
        "ok",
        f"{module_name} version is compatible.",
        version=found_version,
        required_min_version=min_version,
        reason=reason,
    )


def check_cuda(device: str) -> CheckResult:
    try:
        import torch
    except Exception as exc:
        return result("cuda", "fail", "Cannot import torch.", error=str(exc))

    if not device.startswith("cuda"):
        return result("cuda", "warn", "CUDA check skipped because device is not cuda.", device=device)

    if not torch.cuda.is_available():
        return result("cuda", "fail", "CUDA device was requested but is unavailable.", device=device)

    try:
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        name = torch.cuda.get_device_name(index)
    except Exception as exc:
        return result("cuda", "fail", "CUDA is available, but the requested device failed.", device=device, error=str(exc))

    return result("cuda", "ok", "CUDA device is visible.", device=device, gpu_name=name)


def check_model_ref(model: Optional[str]) -> CheckResult:
    if not model:
        return result("model", "warn", "No model path or Hugging Face id was provided.")

    path = Path(model).expanduser()
    if path.exists():
        return result("model", "ok", "Local model path exists.", model=model, resolved=str(path.resolve()))

    if "/" in model:
        return result(
            "model",
            "warn",
            "Model looks like a Hugging Face id; this check does not download weights or verify auth.",
            model=model,
        )

    return result("model", "warn", "Model is not a local path; runtime loading still needs to be tested.", model=model)


def check_tokenizer_ref(tokenizer: Optional[str]) -> CheckResult:
    if not tokenizer:
        return result("tokenizer", "fail", "Driver requires a tokenizer id/path.")

    path = Path(tokenizer).expanduser()
    if path.exists():
        return result(
            "tokenizer",
            "ok",
            "Local tokenizer path exists.",
            tokenizer=tokenizer,
            resolved=str(path.resolve()),
        )

    if "/" in tokenizer:
        return result(
            "tokenizer",
            "warn",
            "Tokenizer looks like a Hugging Face id; this check does not download files or verify auth.",
            tokenizer=tokenizer,
        )

    return result(
        "tokenizer",
        "warn",
        "Tokenizer is not a local path; runtime loading still needs to be tested.",
        tokenizer=tokenizer,
    )


def check_hf_token(require_token: bool) -> CheckResult:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return result(
            "hf_token",
            "ok",
            "Hugging Face token environment variable is present.",
            env_var=(
                "HF_TOKEN"
                if os.environ.get("HF_TOKEN")
                else "HUGGING_FACE_HUB_TOKEN"
            ),
            token_length=len(token),
        )
    status = "fail" if require_token else "warn"
    return result(
        "hf_token",
        status,
        "No Hugging Face token was found in HF_TOKEN or HUGGING_FACE_HUB_TOKEN.",
        required=require_token,
    )


def check_hf_cache_dir(cache_dir: str) -> CheckResult:
    resolved = Path(
        cache_dir
        or os.environ.get("HF_HOME", "")
        or os.environ.get("TRANSFORMERS_CACHE", "")
        or "~/.cache/huggingface"
    ).expanduser()
    if resolved.exists():
        if not resolved.is_dir():
            return result(
                "hf_cache",
                "fail",
                "Hugging Face cache path exists but is not a directory.",
                path=str(resolved),
            )
        if not os.access(resolved, os.W_OK):
            return result(
                "hf_cache",
                "fail",
                "Hugging Face cache directory is not writable.",
                path=str(resolved),
            )
        return result(
            "hf_cache",
            "ok",
            "Hugging Face cache directory is writable.",
            path=str(resolved),
        )

    parent = resolved.parent
    if parent.exists() and os.access(parent, os.W_OK):
        return result(
            "hf_cache",
            "ok",
            "Hugging Face cache directory can be created.",
            path=str(resolved),
            parent=str(parent),
        )
    return result(
        "hf_cache",
        "warn",
        "Hugging Face cache directory does not exist and parent is not writable.",
        path=str(resolved),
        parent=str(parent),
    )


def _prompt_text_from_json(item: object) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        value = item.get("prompt", item.get("text"))
        return value if isinstance(value, str) else None
    return None


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_count_result(
    prompt_count: int,
    prompt_hashes: Set[str],
    min_prompts: int,
    **details: object,
) -> Optional[CheckResult]:
    unique_prompt_hashes = len(prompt_hashes)
    if unique_prompt_hashes < min_prompts:
        return result(
            "prompts",
            "fail",
            "Prompt source has fewer unique prompts than required.",
            prompt_count=prompt_count,
            unique_prompt_hashes=unique_prompt_hashes,
            min_prompts=min_prompts,
            **details,
        )
    return None


def check_prompt_source(
    prompt: Optional[str],
    prompt_file: Optional[str],
    prompts_jsonl: Optional[str],
    min_prompts: int = 1,
) -> CheckResult:
    min_prompts = max(1, int(min_prompts))
    selected = [
        name
        for name, value in (
            ("prompt", prompt),
            ("prompt_file", prompt_file),
            ("prompts_jsonl", prompts_jsonl),
        )
        if value
    ]
    if not selected:
        return result("prompts", "fail", "Driver requires one prompt source.")
    if len(selected) > 1:
        return result(
            "prompts",
            "fail",
            "Use only one prompt source.",
            selected=selected,
        )

    if prompt is not None:
        if not prompt:
            return result("prompts", "fail", "Inline prompt is empty.")
        count_error = prompt_count_result(
            1,
            {prompt_sha256(prompt)},
            min_prompts,
        )
        if count_error:
            return count_error
        return result(
            "prompts",
            "ok",
            "Inline prompt is present.",
            prompt_chars=len(prompt),
            prompt_count=1,
            unique_prompt_hashes=1,
            min_prompts=min_prompts,
        )

    if prompt_file:
        path = Path(prompt_file).expanduser()
        if not path.exists():
            return result("prompts", "fail", "Prompt file does not exist.", path=str(path))
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return result("prompts", "fail", "Prompt file cannot be read.", path=str(path), error=str(exc))
        if not text:
            return result("prompts", "fail", "Prompt file is empty.", path=str(path))
        count_error = prompt_count_result(
            1,
            {prompt_sha256(text)},
            min_prompts,
            path=str(path),
        )
        if count_error:
            return count_error
        return result(
            "prompts",
            "ok",
            "Prompt file is readable.",
            path=str(path),
            prompt_chars=len(text),
            prompt_count=1,
            unique_prompt_hashes=1,
            min_prompts=min_prompts,
        )

    path = Path(str(prompts_jsonl)).expanduser()
    if not path.exists():
        return result("prompts", "fail", "Prompt JSONL file does not exist.", path=str(path))

    prompt_count = 0
    prompt_ids: Set[str] = set()
    duplicate_ids: Set[str] = set()
    prompt_hashes: Set[str] = set()
    duplicate_prompt_hashes: Set[str] = set()
    try:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    return result(
                        "prompts",
                        "fail",
                        "Prompt JSONL has invalid JSON.",
                        path=str(path),
                        line=line_number,
                        error=exc.msg,
                    )
                text = _prompt_text_from_json(item)
                if not isinstance(text, str) or not text:
                    return result(
                        "prompts",
                        "fail",
                        "Prompt JSONL entry must be a string or object with non-empty prompt/text.",
                        path=str(path),
                        line=line_number,
                    )
                if isinstance(item, dict):
                    prompt_id = str(item.get("id", item.get("prompt_id", f"prompt_{prompt_count}")))
                else:
                    prompt_id = f"prompt_{prompt_count}"
                if prompt_id in prompt_ids:
                    duplicate_ids.add(prompt_id)
                prompt_ids.add(prompt_id)
                prompt_hash = prompt_sha256(text)
                if prompt_hash in prompt_hashes:
                    duplicate_prompt_hashes.add(prompt_hash)
                prompt_hashes.add(prompt_hash)
                prompt_count += 1
    except OSError as exc:
        return result("prompts", "fail", "Prompt JSONL file cannot be read.", path=str(path), error=str(exc))

    if prompt_count == 0:
        return result("prompts", "fail", "Prompt JSONL contains no prompts.", path=str(path))
    if duplicate_ids:
        return result(
            "prompts",
            "fail",
            "Prompt ids must be unique.",
            path=str(path),
            duplicate_ids=sorted(duplicate_ids),
        )
    if duplicate_prompt_hashes:
        return result(
            "prompts",
            "fail",
            "Prompt texts must be unique.",
            path=str(path),
            duplicate_prompt_sha256=sorted(duplicate_prompt_hashes),
        )
    count_error = prompt_count_result(
        prompt_count,
        prompt_hashes,
        min_prompts,
        path=str(path),
    )
    if count_error:
        return count_error
    return result(
        "prompts",
        "ok",
        "Prompt JSONL is readable.",
        path=str(path),
        prompt_count=prompt_count,
        unique_prompt_hashes=len(prompt_hashes),
        min_prompts=min_prompts,
    )


def check_k8s_manifest_template(
    manifest_path: str,
    require_manifest: bool,
) -> CheckResult:
    path = Path(manifest_path).expanduser()
    if not path.exists():
        status = "fail" if require_manifest else "warn"
        return result(
            "k8s_manifest_template",
            status,
            "Kubernetes experiment manifest template was not found.",
            path=str(path),
            required=require_manifest,
        )
    try:
        manifest_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return result(
            "k8s_manifest_template",
            "fail",
            "Kubernetes experiment manifest template cannot be read.",
            path=str(path),
            error=str(exc),
        )

    audit = k8s_manifest_audit.build_audit(manifest_text)
    if not audit["ok"]:
        return result(
            "k8s_manifest_template",
            "fail",
            "Kubernetes experiment manifest template failed static audit.",
            path=str(path),
            sha256=file_sha256(path),
            errors=audit.get("errors", []),
            document_count=audit.get("document_count", 0),
        )
    return result(
        "k8s_manifest_template",
        "ok",
        "Kubernetes experiment manifest template is present and audit-clean.",
        path=str(path),
        sha256=file_sha256(path),
        document_count=audit.get("document_count", 0),
    )


def check_ina3221(root: Path, require_power: bool) -> CheckResult:
    sampler = INA3221PowerSampler(root=root)
    rails = [rail for rail, _, _ in sampler.rails]
    if not rails:
        status = "fail" if require_power else "warn"
        return result(
            "ina3221",
            status,
            "No Jetson INA3221 rails were found.",
            root=str(root),
        )
    return result("ina3221", "ok", "Jetson INA3221 rails are visible.", root=str(root), rails=rails)


def check_jetson_frequency(
    freq_hz: Optional[int],
    require_frequency: bool,
    devfreq_root: str,
) -> CheckResult:
    root = discover_jetson_gpu_devfreq_root(devfreq_root)
    min_file = root / "min_freq"
    max_file = root / "max_freq"
    if not min_file.exists() or not max_file.exists():
        status = "fail" if require_frequency else "warn"
        return result("jetson_frequency", status, "Jetson GPU devfreq files were not found.", root=str(root))

    details: Dict[str, object] = {"root": str(root)}
    try:
        details["current_min_freq"] = min_file.read_text().strip()
        details["current_max_freq"] = max_file.read_text().strip()
    except OSError as exc:
        return result("jetson_frequency", "fail", "Cannot read Jetson GPU frequency files.", error=str(exc))

    if freq_hz is not None:
        details["requested_freq_hz"] = freq_hz
        if not min_file.is_file() or not max_file.is_file():
            return result("jetson_frequency", "fail", "Jetson frequency paths are not regular files.", **details)
        if require_frequency and (
            not os.access(min_file, os.W_OK) or not os.access(max_file, os.W_OK)
        ):
            return result(
                "jetson_frequency",
                "fail",
                "Jetson frequency files are visible but not writable by this process.",
                **details,
            )
    return result("jetson_frequency", "ok", "Jetson GPU frequency files are visible.", **details)


def run_command(cmd: List[str], timeout_s: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def check_nvidia_smi(gpu_index: int) -> CheckResult:
    if shutil.which("nvidia-smi") is None:
        return result("nvidia_smi", "fail", "nvidia-smi was not found in PATH.")

    query = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=name,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = run_command(query)
    except Exception as exc:
        return result("nvidia_smi", "fail", "nvidia-smi query failed to run.", error=str(exc))

    if completed.returncode != 0:
        return result(
            "nvidia_smi",
            "fail",
            "nvidia-smi query returned an error.",
            stderr=completed.stderr.strip(),
        )
    return result("nvidia_smi", "ok", "nvidia-smi can read GPU power.", output=completed.stdout.strip())


def check_nvml() -> CheckResult:
    try:
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception as exc:
        return result("nvml", "warn", "NVML import/init failed; nvidia-smi fallback may still work.", error=str(exc))
    return result("nvml", "ok", "NVML import/init works.")


def check_nvidia_frequency(clock_mhz: Optional[int], require_frequency: bool) -> CheckResult:
    if clock_mhz is None:
        return result("nvidia_frequency", "warn", "No verifier GPU clock was requested.")
    if shutil.which("nvidia-smi") is None:
        status = "fail" if require_frequency else "warn"
        return result("nvidia_frequency", status, "nvidia-smi was not found; cannot lock verifier clock.")
    return result(
        "nvidia_frequency",
        "ok",
        "nvidia-smi is available for verifier GPU clock control.",
        requested_clock_mhz=clock_mhz,
    )


def collect_checks(args: argparse.Namespace) -> List[CheckResult]:
    checks = [
        check_python(),
        check_import_min_version(
            "grpc",
            REQUIRED_GRPC_VERSION,
            "xronos.proto.spec_pb2_grpc was generated with grpcio 1.68.1.",
        ),
        check_import_min_version(
            "google.protobuf",
            REQUIRED_PROTOBUF_VERSION,
            "xronos.proto.spec_pb2 was generated with protobuf 5.28.1.",
        ),
        check_import("torch"),
        check_import("transformers"),
    ]

    if args.role in ("drafter", "verifier"):
        checks.append(check_model_ref(args.model))
        checks.append(check_cuda(args.device))

    if args.role in ("drafter", "verifier", "driver"):
        checks.append(check_hf_token(args.require_hf_token))
        checks.append(check_hf_cache_dir(args.hf_cache_dir))

    if args.role == "drafter":
        checks.append(check_ina3221(Path(args.ina3221_root), args.require_power))
        checks.append(
            check_jetson_frequency(
                args.jetson_gpu_freq_hz,
                args.require_frequency_control,
                args.jetson_gpu_devfreq_root,
            )
        )

    if args.role == "verifier":
        checks.append(check_nvml())
        checks.append(check_nvidia_smi(args.gpu_index))
        checks.append(
            check_nvidia_frequency(
                args.gpu_clock_mhz,
                args.require_frequency_control,
            )
        )

    if args.role == "driver":
        checks.append(check_tokenizer_ref(args.tokenizer))
        checks.append(
            check_prompt_source(
                args.prompt,
                args.prompt_file,
                args.prompts_jsonl,
                args.min_prompts,
            )
        )
        checks.append(
            check_k8s_manifest_template(
                args.k8s_manifest,
                args.require_k8s_manifest,
            )
        )

    return checks


def print_checks(checks: List[CheckResult]) -> None:
    for check in checks:
        print(f"[{check.status.upper()}] {check.name}: {check.message}")
        for key, value in check.details.items():
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check host prerequisites before running spec decoding experiments."
    )
    parser.add_argument("--role", choices=["drafter", "verifier", "driver"], required=True)
    parser.add_argument("--model", help="Model id/path expected on this host")
    parser.add_argument("--tokenizer", help="Tokenizer id/path expected by the driver")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompts-jsonl")
    parser.add_argument(
        "--min-prompts",
        type=int,
        default=1,
        help="Minimum unique prompt texts required for the driver prompt source.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ina3221-root", default=str(DEFAULT_INA3221_ROOT))
    parser.add_argument("--jetson-gpu-freq-hz", type=int)
    parser.add_argument(
        "--jetson-gpu-devfreq-root",
        default="",
        help="Optional Jetson GPU devfreq root. Auto-discovered when omitted.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-clock-mhz", type=int)
    parser.add_argument("--require-power", action="store_true")
    parser.add_argument("--require-frequency-control", action="store_true")
    parser.add_argument(
        "--require-hf-token",
        action="store_true",
        help="Fail if neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN is present.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="",
        help="Optional Hugging Face cache directory to check for writability.",
    )
    parser.add_argument(
        "--k8s-manifest",
        default="k8s/spec-decoding.yaml",
        help="Kubernetes experiment manifest template expected in the driver image.",
    )
    parser.add_argument(
        "--require-k8s-manifest",
        action="store_true",
        help="Fail driver doctor if the Kubernetes experiment manifest template is missing or audit-failing.",
    )
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = collect_checks(args)
    print_checks(checks)
    payload = {
        "role": args.role,
        "ok": not any(check.status == "fail" for check in checks),
        "failures": sum(1 for check in checks if check.status == "fail"),
        "warnings": sum(1 for check in checks if check.status == "warn"),
        "checks": [asdict(check) for check in checks],
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote doctor report to {args.json_out}")

    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
