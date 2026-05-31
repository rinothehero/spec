import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_INA3221_ROOT = Path("/sys/bus/i2c/drivers/ina3221")
INPUT_POWER_RAILS = {
    "POM_5V_IN",
    "VDD_5V_IN",
    "VDD_IN",
    "VIN_SYS_5V0",
}
SYNTHETIC_SUM_RAIL = "sum_rails_power"
SYNTHETIC_TOTAL_RAIL = "tot_power"


def _strip_prefix_suffix(value: str, prefix: str, suffix: str) -> str:
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value


def _base_rail_name(rail_name: str) -> str:
    return rail_name.split(":", 1)[-1].upper()


def _is_input_power_rail(rail_name: str) -> bool:
    return _base_rail_name(rail_name) in INPUT_POWER_RAILS


@dataclass
class PowerSummary:
    rail: str
    mean_power_mw: float
    energy_mj: float


def _samples_for_window(
    timestamps: List[float],
    samples: List[Dict[str, float]],
    t0: float,
    t1: float,
) -> List[Dict[str, float]]:
    non_empty = [
        (ts, sample)
        for ts, sample in zip(timestamps, samples)
        if sample
    ]
    window = [
        sample
        for ts, sample in non_empty
        if t0 <= ts <= t1
    ]
    if window:
        return window
    if not non_empty:
        return []

    midpoint = (t0 + t1) / 2.0
    _, nearest = min(non_empty, key=lambda item: abs(item[0] - midpoint))
    return [nearest]


class INA3221PowerSampler:
    """Small sysfs-only Jetson power sampler.

    It avoids the jtop dependency so the drafter server can import on non-Jetson
    hosts. On hosts without INA3221 rails it simply returns no power samples.
    """

    def __init__(
        self,
        interval_s: float = 0.01,
        root: Path = DEFAULT_INA3221_ROOT,
    ) -> None:
        self.interval_s = interval_s
        self.root = root
        self.rails = self._discover_rails()
        self.timestamps: List[float] = []
        self.samples: List[Dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _discover_rails(self) -> List[Tuple[str, Path, Path]]:
        if not self.root.exists():
            return []

        rails: List[Tuple[str, Path, Path]] = []
        hwmons = sorted(
            path for path in self.root.glob("*-00*/hwmon/hwmon*") if path.is_dir()
        )
        for hwmon in hwmons:
            monitor_name = hwmon.parent.parent.name
            for label_path in sorted(hwmon.glob("in*_label")):
                index = _strip_prefix_suffix(label_path.stem, "in", "_label")
                try:
                    label = label_path.read_text().strip()
                except OSError:
                    continue

                if label.lower().startswith("sum of"):
                    continue

                voltage_path = hwmon / f"in{index}_input"
                current_path = hwmon / f"curr{index}_input"
                if not voltage_path.exists() or not current_path.exists():
                    continue

                rail_name = label
                if any(existing[0] == rail_name for existing in rails):
                    rail_name = f"{monitor_name}:{label}"
                rails.append((rail_name, voltage_path, current_path))
        return rails

    def _read_float(self, path: Path) -> Optional[float]:
        try:
            return float(path.read_text().strip())
        except (OSError, ValueError):
            return None

    def _read_power(self) -> Dict[str, float]:
        sample: Dict[str, float] = {}
        sum_rails_power_mw = 0.0
        input_power_mw = 0.0
        has_input_power = False

        for rail_name, voltage_path, current_path in self.rails:
            voltage_mv = self._read_float(voltage_path)
            current_ma = self._read_float(current_path)
            if voltage_mv is None or current_ma is None:
                continue

            power_mw = voltage_mv * current_ma / 1000.0
            sample[rail_name] = power_mw
            if _is_input_power_rail(rail_name):
                input_power_mw += power_mw
                has_input_power = True
            else:
                sum_rails_power_mw += power_mw

        if sample:
            sample[SYNTHETIC_SUM_RAIL] = sum_rails_power_mw
            sample[SYNTHETIC_TOTAL_RAIL] = (
                input_power_mw
                if has_input_power
                else sum_rails_power_mw
            )
        return sample

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self.timestamps.append(time.monotonic())
            self.samples.append(self._read_power())
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.timestamps.clear()
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def summarize(self, t0: float, t1: float) -> Tuple[List[PowerSummary], int]:
        duration_s = max(t1 - t0, 0.0)
        window = _samples_for_window(self.timestamps, self.samples, t0, t1)
        if not window:
            return [], 0

        rail_names = sorted({rail for sample in window for rail in sample})
        summaries: List[PowerSummary] = []
        for rail_name in rail_names:
            values = [sample[rail_name] for sample in window if rail_name in sample]
            if not values:
                continue
            mean_power_mw = sum(values) / len(values)
            summaries.append(
                PowerSummary(
                    rail=rail_name,
                    mean_power_mw=mean_power_mw,
                    energy_mj=mean_power_mw * duration_s,
                )
            )
        return summaries, len(window)


class NvidiaSMIPowerSampler:
    """Power sampler for discrete NVIDIA GPUs.

    NVML is used when available because repeatedly spawning nvidia-smi is too
    expensive for short verify windows. The nvidia-smi subprocess path remains a
    fallback for minimal environments. Values are normalized to mW, so
    integration matches the Jetson sampler: mW * s = mJ.
    """

    def __init__(self, interval_s: float = 0.01, gpu_index: int = 0) -> None:
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.timestamps: List[float] = []
        self.samples: List[Dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._nvml_handle = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def _read_power(self) -> Dict[str, float]:
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                return {
                    "verifier_gpu_power": float(
                        self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                    )
                }
            except Exception:
                pass

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return {}

        if result.returncode != 0:
            return {}

        try:
            power_w = float(result.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            return {}
        return {"verifier_gpu_power": power_w * 1000.0}

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self.timestamps.append(time.monotonic())
            self.samples.append(self._read_power())
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.timestamps.clear()
        self.samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def summarize(self, t0: float, t1: float) -> Tuple[List[PowerSummary], int]:
        duration_s = max(t1 - t0, 0.0)
        window = _samples_for_window(self.timestamps, self.samples, t0, t1)
        if not window:
            return [], 0

        values = [
            sample["verifier_gpu_power"]
            for sample in window
            if "verifier_gpu_power" in sample
        ]
        if not values:
            return [], len(window)
        mean_power_mw = sum(values) / len(values)
        return [
            PowerSummary(
                rail="verifier_gpu_power",
                mean_power_mw=mean_power_mw,
                energy_mj=mean_power_mw * duration_s,
            )
        ], len(window)


class VerifierPowerSampler:
    """Verifier power sampler with discrete-GPU and Jetson fallbacks."""

    def __init__(self, interval_s: float = 0.01, gpu_index: int = 0) -> None:
        self.nvidia_sampler = NvidiaSMIPowerSampler(
            interval_s=interval_s,
            gpu_index=gpu_index,
        )
        self.jetson_sampler = INA3221PowerSampler(interval_s=interval_s)

    def start(self) -> None:
        self.nvidia_sampler.start()
        self.jetson_sampler.start()

    def stop(self) -> None:
        self.nvidia_sampler.stop()
        self.jetson_sampler.stop()

    def summarize(self, t0: float, t1: float) -> Tuple[List[PowerSummary], int]:
        nvidia_rails, nvidia_samples = self.nvidia_sampler.summarize(t0, t1)
        if nvidia_rails:
            return nvidia_rails, nvidia_samples
        return self.jetson_sampler.summarize(t0, t1)
