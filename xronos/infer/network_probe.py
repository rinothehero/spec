import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from xronos.infer import spec_driver


SCHEMA_VERSION = "xronos-network-probe-v1"


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rtt_ms(values: List[float]) -> Dict[str, object]:
    if not values:
        return {
            "sample_count": 0,
            "rtt_ms_min": "",
            "rtt_ms_mean": "",
            "rtt_ms_median": "",
            "rtt_ms_p95": "",
            "rtt_ms_max": "",
        }
    return {
        "sample_count": len(values),
        "rtt_ms_min": f"{min(values):.6f}",
        "rtt_ms_mean": f"{statistics.mean(values):.6f}",
        "rtt_ms_median": f"{statistics.median(values):.6f}",
        "rtt_ms_p95": f"{percentile(values, 0.95):.6f}",
        "rtt_ms_max": f"{max(values):.6f}",
    }


async def timed_health(name: str, stub, timeout_s: float):
    t0 = time.monotonic()
    metadata = await spec_driver.check_health(name, stub, timeout_s, verbose=False)
    t1 = time.monotonic()
    return (t1 - t0) * 1000.0, metadata


async def probe_target(
    name: str,
    addr: str,
    stub_class,
    samples: int,
    warmup_samples: int,
    timeout_s: float,
    startup_timeout_s: float,
    health_check_interval_s: float,
    channel_options,
) -> Dict[str, object]:
    async with spec_driver.grpc.aio.insecure_channel(
        addr,
        options=channel_options,
    ) as channel:
        stub = stub_class(channel)
        health_metadata = await spec_driver.wait_for_health(
            name,
            stub,
            timeout_s,
            startup_timeout_s,
            health_check_interval_s,
            verbose=True,
        )

        for _ in range(warmup_samples):
            await timed_health(name, stub, timeout_s)

        rtt_values = []
        errors = []
        for _ in range(samples):
            try:
                rtt_ms, health_metadata = await timed_health(name, stub, timeout_s)
                rtt_values.append(rtt_ms)
            except Exception as exc:
                errors.append(str(exc))

    summary = summarize_rtt_ms(rtt_values)
    return {
        "role": name,
        "addr": addr,
        "ok": len(errors) == 0 and len(rtt_values) == samples,
        "requested_samples": samples,
        "warmup_samples": warmup_samples,
        "errors": errors[:20],
        "health_metadata": health_metadata,
        **summary,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.drafter_addr and not args.verifier_addr:
        raise ValueError("Provide at least one of --drafter-addr or --verifier-addr.")
    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0.")
    if args.warmup_samples < 0:
        raise ValueError("--warmup-samples must not be negative.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if args.startup_timeout_s < 0:
        raise ValueError("--startup-timeout-s must not be negative.")
    if args.health_check_interval_s <= 0:
        raise ValueError("--health-check-interval-s must be greater than 0.")


async def run(args: argparse.Namespace) -> Dict[str, object]:
    validate_args(args)
    spec_driver.load_grpc_bindings()

    max_message_bytes = args.max_message_mb * 1024 * 1024
    channel_options = [
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.max_receive_message_length", max_message_bytes),
    ]
    tasks = []
    if args.drafter_addr:
        tasks.append(
            probe_target(
                "drafter",
                args.drafter_addr,
                spec_driver.spec_pb2_grpc.DrafterStub,
                args.samples,
                args.warmup_samples,
                args.timeout,
                args.startup_timeout_s,
                args.health_check_interval_s,
                channel_options,
            )
        )
    if args.verifier_addr:
        tasks.append(
            probe_target(
                "verifier",
                args.verifier_addr,
                spec_driver.spec_pb2_grpc.VerifierStub,
                args.samples,
                args.warmup_samples,
                args.timeout,
                args.startup_timeout_s,
                args.health_check_interval_s,
                channel_options,
            )
        )

    target_results = await asyncio.gather(*tasks)
    targets = {str(result["role"]): result for result in target_results}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": all(bool(result.get("ok")) for result in target_results),
        "samples": args.samples,
        "warmup_samples": args.warmup_samples,
        "timeout_s": args.timeout,
        "startup_timeout_s": args.startup_timeout_s,
        "health_check_interval_s": args.health_check_interval_s,
        "targets": targets,
        "metadata": spec_driver.collect_metadata(),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote network probe to {args.out}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure gRPC health-check RTTs to drafter/verifier services."
    )
    parser.add_argument("--drafter-addr", default="")
    parser.add_argument("--verifier-addr", default="")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--health-check-interval-s", type=float, default=5.0)
    parser.add_argument("--max-message-mb", type=int, default=64)
    parser.add_argument("--out", default="network_probe.json")
    return parser.parse_args()


def main() -> None:
    payload = asyncio.run(run(parse_args()))
    print(f"network_probe_ok={int(bool(payload['ok']))}")
    for role, target in sorted(payload["targets"].items()):
        print(
            "role={role} samples={sample_count} mean_ms={mean} "
            "p95_ms={p95} ok={ok}".format(
                role=role,
                sample_count=target.get("sample_count", 0),
                mean=target.get("rtt_ms_mean", ""),
                p95=target.get("rtt_ms_p95", ""),
                ok=int(bool(target.get("ok"))),
            )
        )
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
