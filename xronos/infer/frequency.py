import logging
import subprocess
from pathlib import Path
from typing import List, Optional


DEFAULT_JETSON_GPU_DEVFREQ_ROOT = Path("/sys/class/devfreq/17000000.gpu")
JETSON_GPU_DEVFREQ_PATTERNS = [
    "/sys/class/devfreq/*gpu*",
    "/sys/devices/*gpu*/devfreq/*gpu*",
    "/sys/devices/platform/*gpu*/devfreq/*gpu*",
]


def _has_devfreq_limits(path: Path) -> bool:
    return (path / "min_freq").exists() and (path / "max_freq").exists()


def discover_jetson_gpu_devfreq_root(requested_root: Optional[str] = None) -> Path:
    if requested_root:
        return Path(requested_root)

    candidates: List[Path] = [DEFAULT_JETSON_GPU_DEVFREQ_ROOT]
    for pattern in JETSON_GPU_DEVFREQ_PATTERNS:
        candidates.extend(Path("/").glob(pattern.lstrip("/")))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _has_devfreq_limits(candidate):
            return candidate
    return DEFAULT_JETSON_GPU_DEVFREQ_ROOT


class FrequencyLock:
    """Best-effort fixed GPU frequency helper for controlled experiments."""

    def __init__(
        self,
        jetson_gpu_freq_hz: Optional[int] = None,
        jetson_gpu_devfreq_root: Optional[str] = None,
        nvidia_smi_gpu_clock_mhz: Optional[int] = None,
        nvidia_smi_gpu_index: int = 0,
    ) -> None:
        self.jetson_gpu_freq_hz = jetson_gpu_freq_hz
        self.nvidia_smi_gpu_clock_mhz = nvidia_smi_gpu_clock_mhz
        self.nvidia_smi_gpu_index = nvidia_smi_gpu_index
        self.jetson_path = discover_jetson_gpu_devfreq_root(jetson_gpu_devfreq_root)
        self.original_min: Optional[str] = None
        self.original_max: Optional[str] = None
        self.active_label = ""
        self.last_lock_ok: Optional[bool] = None

    def apply(self) -> None:
        if self.jetson_gpu_freq_hz is not None:
            self.set_jetson_gpu_freq(self.jetson_gpu_freq_hz)
        if self.nvidia_smi_gpu_clock_mhz is not None:
            self.set_nvidia_smi_gpu_clock(self.nvidia_smi_gpu_clock_mhz)

    def restore(self) -> None:
        if self.jetson_gpu_freq_hz is not None:
            self._restore_jetson()
        if self.nvidia_smi_gpu_clock_mhz is not None:
            self._run(
                ["nvidia-smi", "-i", str(self.nvidia_smi_gpu_index), "-rgc"],
                "restore nvidia-smi GPU clocks",
            )

    def metadata(self) -> dict:
        return {
            "jetson_gpu_freq_hz": str(self.jetson_gpu_freq_hz or ""),
            "jetson_gpu_devfreq_root": str(self.jetson_path),
            "nvidia_smi_gpu_clock_mhz": str(self.nvidia_smi_gpu_clock_mhz or ""),
            "nvidia_smi_gpu_index": str(self.nvidia_smi_gpu_index),
            "frequency_label": self.active_label,
            "frequency_lock_ok": (
                "" if self.last_lock_ok is None else str(int(self.last_lock_ok))
            ),
        }

    def set_jetson_gpu_freq(self, freq_hz: int) -> bool:
        self.jetson_gpu_freq_hz = freq_hz
        return self._apply_jetson(freq_hz)

    def set_nvidia_smi_gpu_clock(self, clock_mhz: int) -> bool:
        self.nvidia_smi_gpu_clock_mhz = clock_mhz
        return self._apply_nvidia_smi(clock_mhz)

    def _apply_jetson(self, freq_hz: int) -> bool:
        min_file = self.jetson_path / "min_freq"
        max_file = self.jetson_path / "max_freq"
        if not min_file.exists() or not max_file.exists():
            logging.warning("Jetson GPU devfreq files not found under %s", self.jetson_path)
            self.last_lock_ok = False
            return False

        try:
            if self.original_min is None:
                self.original_min = min_file.read_text().strip()
            if self.original_max is None:
                self.original_max = max_file.read_text().strip()
            current_min = int(min_file.read_text().strip())
            current_max = int(max_file.read_text().strip())

            # devfreq validates min <= max on each write.
            if freq_hz > current_max:
                max_file.write_text(str(freq_hz))
                min_file.write_text(str(freq_hz))
            elif freq_hz < current_min:
                min_file.write_text(str(freq_hz))
                max_file.write_text(str(freq_hz))
            else:
                min_file.write_text(str(freq_hz))
                max_file.write_text(str(freq_hz))

            self.active_label = f"jetson_gpu={freq_hz}Hz"
            self.last_lock_ok = True
            logging.info("Locked Jetson GPU frequency to %s Hz", freq_hz)
            return True
        except (OSError, PermissionError, ValueError) as exc:
            logging.warning("Failed to lock Jetson GPU frequency: %s", exc)
            self.last_lock_ok = False
            return False

    def _restore_jetson(self) -> None:
        if self.original_min is None or self.original_max is None:
            return
        min_file = self.jetson_path / "min_freq"
        max_file = self.jetson_path / "max_freq"
        try:
            min_file.write_text(self.original_min)
            max_file.write_text(self.original_max)
            logging.info("Restored Jetson GPU frequency limits")
        except OSError as exc:
            logging.warning("Failed to restore Jetson GPU frequency: %s", exc)

    def _apply_nvidia_smi(self, clock_mhz: int) -> bool:
        ok = self._run(
            [
                "nvidia-smi",
                "-i",
                str(self.nvidia_smi_gpu_index),
                "-lgc",
                str(clock_mhz),
            ],
            "lock nvidia-smi GPU clocks",
        )
        if ok:
            self.active_label = (
                f"gpu_index={self.nvidia_smi_gpu_index},gpu_clock={clock_mhz}MHz"
            )
        self.last_lock_ok = ok
        return ok

    def _run(self, cmd: List[str], label: str) -> bool:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            logging.warning("Cannot %s because nvidia-smi was not found", label)
            return False
        except Exception as exc:
            logging.warning("Cannot %s: %s", label, exc)
            return False

        if result.returncode != 0:
            logging.warning("%s failed: %s", label, result.stderr.strip())
            return False
        else:
            logging.info("%s succeeded", label)
            return True
