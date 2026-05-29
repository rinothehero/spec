import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import grpc
import torch

from xronos.infer.frequency import FrequencyLock
from xronos.infer.modeling import (
    argmax_token,
    clone_or_crop_past_key_values,
    ensure_session_id,
    last_next_logits,
    load_causal_lm,
    maybe_clear_cuda_cache,
    maybe_synchronize,
    model_metadata,
    parse_dtype,
    single_token_tensor,
    token_tensor,
)
from xronos.infer.power import INA3221PowerSampler
from xronos.infer.runtime import runtime_metadata, runtime_status_metadata
from xronos.infer.spec_driver import SPEC_RPC_SCHEMA_VERSION
from xronos.proto import spec_pb2, spec_pb2_grpc


@dataclass
class DrafterSession:
    session_id: str
    seq_len: int
    past_key_values: Any
    next_logits: torch.Tensor
    speculative_base_len: int = 0
    speculative_past_key_values: Any = None
    speculative_next_logits: torch.Tensor = None
    speculative_next_logits_history: List[torch.Tensor] = field(default_factory=list)
    last_draft_tokens: List[int] = field(default_factory=list)


class DrafterServicer(spec_pb2_grpc.DrafterServicer):
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype,
        device: str,
        power_interval_s: float,
        trust_remote_code: bool,
        local_files_only: bool,
        frequency_lock: FrequencyLock,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.frequency_lock = frequency_lock
        self.frequency_lock.apply()
        self.model = load_causal_lm(
            model_name=model_name,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.model_metadata = model_metadata(self.model)
        self.power_interval_s = power_interval_s
        self.runtime_metadata = runtime_metadata(device=device, role="drafter")
        self.sessions: Dict[str, DrafterSession] = {}
        self._lock = asyncio.Lock()

    async def Health(self, request, context):
        return spec_pb2.HealthResponse(
            ok=True,
            message="drafter ready",
            model=self.model_name,
            device=self.device,
            metadata={
                **self.runtime_metadata,
                **self.model_metadata,
                "power_interval_s": f"{self.power_interval_s:.6f}",
                "spec_rpc_schema_version": SPEC_RPC_SCHEMA_VERSION,
                **runtime_status_metadata(device=self.device),
                **self.frequency_lock.metadata(),
            },
        )

    async def SetFrequency(self, request, context):
        async with self._lock:
            if request.jetson_gpu_freq_hz == 0:
                return spec_pb2.SetFrequencyResponse(
                    ok=True,
                    message="drafter frequency unchanged",
                    metadata=self.frequency_lock.metadata(),
                )
            ok = self.frequency_lock.set_jetson_gpu_freq(
                int(request.jetson_gpu_freq_hz)
            )
            return spec_pb2.SetFrequencyResponse(
                ok=ok,
                message="drafter frequency updated" if ok else "drafter frequency update failed",
                metadata=self.frequency_lock.metadata(),
                error="" if ok else "failed to set Jetson GPU frequency",
            )

    async def InitSession(self, request, context):
        async with self._lock:
            try:
                return self._init_session(request.session_id, list(request.context_tokens))
            except Exception as exc:
                logging.exception("Drafter InitSession failed")
                return spec_pb2.InitSessionResponse(error=str(exc))

    async def IdlePower(self, request, context):
        async with self._lock:
            try:
                return await self._idle_power(float(request.duration_s))
            except Exception as exc:
                logging.exception("Drafter IdlePower failed")
                return spec_pb2.IdlePowerResponse(error=str(exc))

    async def Draft(self, request, context):
        async with self._lock:
            try:
                return self._draft(
                    request.session_id,
                    int(request.gamma),
                    int(request.base_committed_tokens),
                )
            except Exception as exc:
                logging.exception("Draft request failed")
                return spec_pb2.DraftResponse(error=str(exc))

    async def Commit(self, request, context):
        async with self._lock:
            try:
                return self._commit(
                    session_id=request.session_id,
                    accepted_tokens=int(request.accepted_tokens),
                    replacement_token=int(request.replacement_token),
                    append_replacement=bool(request.append_replacement),
                    base_committed_tokens=int(request.base_committed_tokens),
                )
            except Exception as exc:
                logging.exception("Drafter Commit failed")
                return spec_pb2.CommitResponse(error=str(exc))

    async def ResetSession(self, request, context):
        async with self._lock:
            self.sessions.pop(request.session_id, None)
            maybe_clear_cuda_cache(self.device)
            return spec_pb2.ResetSessionResponse(ok=True)

    async def _idle_power(self, duration_s: float):
        if duration_s <= 0:
            raise ValueError("duration_s must be greater than 0.")

        sampler = INA3221PowerSampler(interval_s=self.power_interval_s)
        sampler.start()
        t0 = time.monotonic()
        await asyncio.sleep(duration_s)
        t1 = time.monotonic()
        sampler.stop()

        rails, n_samples = sampler.summarize(t0, t1)
        return spec_pb2.IdlePowerResponse(
            duration_s=t1 - t0,
            rails=self._rail_messages(rails),
            n_power_samples=n_samples,
        )

    def _init_session(self, requested_session_id: str, context_tokens: List[int]):
        if not context_tokens:
            raise ValueError("context_tokens must not be empty.")

        session_id = ensure_session_id(requested_session_id)
        input_ids = token_tensor(context_tokens, self.device)
        sampler = INA3221PowerSampler(interval_s=self.power_interval_s)

        maybe_synchronize(self.device)
        sampler.start()
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, use_cache=True)
            past_key_values = out.past_key_values
            next_logits = last_next_logits(out.logits)
        maybe_synchronize(self.device)
        t1 = time.monotonic()
        sampler.stop()

        self.sessions[session_id] = DrafterSession(
            session_id=session_id,
            seq_len=len(context_tokens),
            past_key_values=past_key_values,
            next_logits=next_logits,
        )
        rails, n_samples = sampler.summarize(t0, t1)
        return spec_pb2.InitSessionResponse(
            session_id=session_id,
            context_tokens=len(context_tokens),
            latency_ms=(t1 - t0) * 1000.0,
            rails=self._rail_messages(rails),
            n_power_samples=n_samples,
        )

    def _draft(self, session_id: str, gamma: int, base_committed_tokens: int):
        if gamma <= 0:
            raise ValueError("gamma must be greater than 0.")
        session = self._get_session(session_id)
        self._require_committed_len(
            session,
            base_committed_tokens,
            phase="Draft",
        )
        if session.last_draft_tokens:
            raise ValueError(
                "Previous speculative draft must be committed or the session reset before Draft."
            )

        draft_tokens: List[int] = []
        next_logits_history: List[torch.Tensor] = []
        past = session.past_key_values
        next_logits = session.next_logits
        sampler = INA3221PowerSampler(interval_s=self.power_interval_s)

        maybe_synchronize(self.device)
        sampler.start()
        t0 = time.monotonic()
        with torch.inference_mode():
            for _ in range(gamma):
                next_token_id = argmax_token(next_logits)
                draft_tokens.append(next_token_id)
                out = self.model(
                    input_ids=single_token_tensor(next_token_id, self.device),
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                next_logits = last_next_logits(out.logits)
                next_logits_history.append(next_logits)

        maybe_synchronize(self.device)
        t1 = time.monotonic()
        sampler.stop()

        session.speculative_base_len = session.seq_len
        session.speculative_past_key_values = past
        session.speculative_next_logits = next_logits
        session.speculative_next_logits_history = next_logits_history
        session.last_draft_tokens = draft_tokens

        rails, n_samples = sampler.summarize(t0, t1)
        return spec_pb2.DraftResponse(
            draft_tokens=draft_tokens,
            latency_ms=(t1 - t0) * 1000.0,
            rails=self._rail_messages(rails),
            n_power_samples=n_samples,
        )

    def _commit(
        self,
        session_id: str,
        accepted_tokens: int,
        replacement_token: int,
        append_replacement: bool,
        base_committed_tokens: int,
    ):
        session = self._get_session(session_id)
        if not session.last_draft_tokens:
            raise ValueError("No speculative draft is available to commit.")
        if accepted_tokens < 0:
            raise ValueError("accepted_tokens must not be negative.")
        if accepted_tokens > len(session.last_draft_tokens):
            raise ValueError("accepted_tokens cannot exceed the draft length.")
        self._require_committed_len(
            session,
            base_committed_tokens,
            phase="Commit",
        )
        if session.speculative_base_len != base_committed_tokens:
            raise ValueError(
                "Drafter speculative base length mismatch during Commit: "
                f"expected={base_committed_tokens}, "
                f"actual={session.speculative_base_len}."
            )

        sampler = INA3221PowerSampler(interval_s=self.power_interval_s)
        maybe_synchronize(self.device)
        sampler.start()
        t0 = time.monotonic()
        with torch.inference_mode():
            if accepted_tokens == 0:
                committed_past = session.past_key_values
                committed_next_logits = session.next_logits
                committed_len = session.speculative_base_len
            elif accepted_tokens == len(session.last_draft_tokens):
                committed_past = session.speculative_past_key_values
                committed_next_logits = session.speculative_next_logits
                committed_len = session.speculative_base_len + accepted_tokens
            else:
                committed_len = session.speculative_base_len + accepted_tokens
                committed_past = clone_or_crop_past_key_values(
                    session.speculative_past_key_values,
                    committed_len,
                )
                committed_next_logits = session.speculative_next_logits_history[
                    accepted_tokens - 1
                ]

            if append_replacement:
                out = self.model(
                    input_ids=single_token_tensor(replacement_token, self.device),
                    past_key_values=committed_past,
                    use_cache=True,
                )
                committed_past = out.past_key_values
                committed_next_logits = last_next_logits(out.logits)
                committed_len += 1

        maybe_synchronize(self.device)
        t1 = time.monotonic()
        sampler.stop()

        session.past_key_values = committed_past
        session.next_logits = committed_next_logits
        session.seq_len = committed_len
        session.speculative_past_key_values = None
        session.speculative_next_logits = None
        session.speculative_next_logits_history = []
        session.last_draft_tokens = []

        rails, n_samples = sampler.summarize(t0, t1)
        return spec_pb2.CommitResponse(
            committed_tokens=committed_len,
            latency_ms=(t1 - t0) * 1000.0,
            rails=self._rail_messages(rails),
            n_power_samples=n_samples,
        )

    def _get_session(self, session_id: str) -> DrafterSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"Unknown session_id: {session_id}") from exc

    def _require_committed_len(
        self,
        session: DrafterSession,
        base_committed_tokens: int,
        phase: str,
    ) -> None:
        if base_committed_tokens <= 0:
            raise ValueError(f"{phase} requires base_committed_tokens.")
        if session.seq_len != base_committed_tokens:
            raise ValueError(
                f"Drafter session length mismatch during {phase}: "
                f"expected={base_committed_tokens}, actual={session.seq_len}."
            )

    def _rail_messages(self, rails):
        return [
            spec_pb2.RailEnergy(
                rail=rail.rail,
                mean_power_mw=rail.mean_power_mw,
                energy_mj=rail.energy_mj,
            )
            for rail in rails
        ]


async def serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=args.log_level)
    frequency_lock = FrequencyLock(
        jetson_gpu_freq_hz=args.jetson_gpu_freq_hz,
        jetson_gpu_devfreq_root=args.jetson_gpu_devfreq_root,
    )
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", args.max_message_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", args.max_message_mb * 1024 * 1024),
        ]
    )
    spec_pb2_grpc.add_DrafterServicer_to_server(
        DrafterServicer(
            model_name=args.model,
            dtype=parse_dtype(args.dtype),
            device=args.device,
            power_interval_s=args.power_interval,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            frequency_lock=frequency_lock,
        ),
        server,
    )
    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    await server.start()
    logging.info("Drafter server listening on %s", bind_addr)
    try:
        await server.wait_for_termination()
    finally:
        if args.restore_frequency_on_exit:
            frequency_lock.restore()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speculative decoding drafter server")
    parser.add_argument("--model", required=True, help="Drafter causal LM id/path")
    parser.add_argument("--host", default="[::]")
    parser.add_argument("--port", type=int, default=50061)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--power-interval", type=float, default=0.01)
    parser.add_argument("--jetson-gpu-freq-hz", type=int, default=None)
    parser.add_argument(
        "--jetson-gpu-devfreq-root",
        default="",
        help="Optional Jetson GPU devfreq root. Auto-discovered when omitted.",
    )
    parser.add_argument("--restore-frequency-on-exit", action="store_true")
    parser.add_argument("--max-message-mb", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    asyncio.run(serve(parse_args()))


if __name__ == "__main__":
    main()
