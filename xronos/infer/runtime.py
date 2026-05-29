import os
import platform
import shutil
import socket
import subprocess
from importlib import metadata
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def runtime_metadata(device: str, role: str) -> dict:
    metadata_dict = {
        "role": role,
        "hostname": socket.gethostname(),
        "device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "") if torch is not None else "",
        "transformers_version": package_version("transformers"),
        "cuda_available": str(torch.cuda.is_available()) if torch is not None else "False",
        "cuda_version": (
            getattr(torch.version, "cuda", "") or "" if torch is not None else ""
        ),
        "xronos_git_commit": os.environ.get("XRONOS_GIT_COMMIT", ""),
        "xronos_image": os.environ.get("XRONOS_IMAGE", ""),
        "pod_name": os.environ.get("POD_NAME", ""),
        "pod_namespace": os.environ.get("POD_NAMESPACE", ""),
        "node_name": os.environ.get("NODE_NAME", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
    }
    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        try:
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            metadata_dict["gpu_name"] = torch.cuda.get_device_name(index)
            metadata_dict["gpu_capability"] = ".".join(
                str(part) for part in torch.cuda.get_device_capability(index)
            )
        except Exception as exc:
            metadata_dict["gpu_metadata_error"] = str(exc)
    return metadata_dict


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def thermal_zone_metadata(root: Path = Path("/sys/class/thermal")) -> dict:
    zones = []
    for zone in sorted(root.glob("thermal_zone*")):
        temp_path = zone / "temp"
        if not temp_path.exists():
            continue
        try:
            raw_temp = float(temp_path.read_text().strip())
        except (OSError, ValueError):
            continue
        temp_c = raw_temp / 1000.0 if raw_temp > 1000.0 else raw_temp
        try:
            zone_type = (zone / "type").read_text().strip()
        except OSError:
            zone_type = zone.name
        zones.append((zone_type, temp_c))

    if not zones:
        return {}

    return {
        "thermal_max_temp_c": _format_float(max(temp for _, temp in zones)),
        "thermal_zones": ",".join(
            f"{name}:{_format_float(temp)}" for name, temp in zones
        ),
    }


def nvidia_smi_status(gpu_index: int = 0) -> dict:
    if shutil.which("nvidia-smi") is None:
        return {}
    query = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=temperature.gpu,pstate,clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            query,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception as exc:
        return {"nvidia_smi_status_error": str(exc)}
    if completed.returncode != 0:
        return {"nvidia_smi_status_error": completed.stderr.strip()}

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    status = {}
    if len(parts) >= 1 and parts[0]:
        status["nvidia_gpu_temp_c"] = parts[0]
    if len(parts) >= 2 and parts[1]:
        status["nvidia_pstate"] = parts[1]
    if len(parts) >= 3 and parts[2]:
        status["nvidia_throttle_active"] = parts[2]
    return status


def runtime_status_metadata(device: str, gpu_index: int = 0) -> dict:
    status = thermal_zone_metadata()
    if device.startswith("cuda"):
        status.update(nvidia_smi_status(gpu_index=gpu_index))
    return status


def runtime_fingerprint(metadata_dict: dict) -> str:
    keys = [
        "python_version",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "gpu_name",
        "xronos_git_commit",
        "xronos_image",
    ]
    return "|".join(str(metadata_dict.get(key, "")) for key in keys)
