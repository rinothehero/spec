import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
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
    next_logits_at,
    parse_dtype,
    single_token_tensor,
    token_tensor,
)
from xronos.infer.power import VerifierPowerSampler
from xronos.infer.runtime import runtime_metadata, runtime_status_metadata
from xronos.infer.spec_algorithm import plan_verify_decision
from xronos.infer.spec_driver import SPEC_RPC_SCHEMA_VERSION
from xronos.proto import spec_pb2, spec_pb2_grpc


@dataclass
class VerifierSession:
    session_id: str
    seq_len: int
    past_key_values: Any
    next_logits: torch.Tensor


class VerifierServicer(spec_pb2_grpc.VerifierServicer):
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype,
        device: str,
        trust_remote_code: bool,
        local_files_only: bool,
        frequency_lock: FrequencyLock,
        power_interval_s: float,
        gpu_index: int,
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
        self.runtime_metadata = runtime_metadata(device=device, role="verifier")
        self.sessions: Dict[str, VerifierSession] = {}
        self._lock = asyncio.Lock()
        self.power_interval_s = power_interval_s
        self.gpu_index = gpu_index

    async def Health(self, request, context):
        return spec_pb2.HealthResponse(
            ok=True,
            message="verifier ready",
            model=self.model_name,
            device=self.device,
            metadata={
                **self.runtime_metadata,
                **self.model_metadata,
                "power_interval_s": f"{self.power_interval_s:.6f}",
                "spec_rpc_schema_version": SPEC_RPC_SCHEMA_VERSION,
                **runtime_status_metadata(device=self.device, gpu_index=self.gpu_index),
                **self.frequency_lock.metadata(),
            },
        )

    async def SetFrequency(self, request, context):
        async with self._lock:
            if request.nvidia_smi_gpu_clock_mhz == 0:
                return spec_pb2.SetFrequencyResponse(
                    ok=True,
                    message="verifier frequency unchanged",
                    metadata=self.frequency_lock.metadata(),
                )
            ok = self.frequency_lock.set_nvidia_smi_gpu_clock(
                int(request.nvidia_smi_gpu_clock_mhz)
            )
            return spec_pb2.SetFrequencyResponse(
                ok=ok,
                message="verifier frequency updated" if ok else "verifier frequency update failed",
                metadata=self.frequency_lock.metadata(),
                error="" if ok else "failed to set verifier GPU clock",
            )

    async def InitSession(self, request, context):
        async with self._lock:
            try:
                return self._init_session(request.session_id, list(request.context_tokens))
            except Exception as exc:
                logging.exception("Verifier InitSession failed")
                return spec_pb2.InitSessionResponse(error=str(exc))

    async def IdlePower(self, request, context):
        async with self._lock:
            try:
                return await self._idle_power(float(request.duration_s))
            except Exception as exc:
                logging.exception("Verifier IdlePower failed")
                return spec_pb2.IdlePowerResponse(error=str(exc))

    async def Verify(self, request, context):
        async with self._lock:
            try:
                return self._verify(
                    session_id=request.session_id,
                    draft_tokens=list(request.draft_tokens),
                    append_replacement=bool(request.append_replacement),
                    stop_token_ids=set(int(token) for token in request.stop_token_ids),
                    base_committed_tokens=int(request.base_committed_tokens),
                )
            except Exception as exc:
                logging.exception("Verify request failed")
                return spec_pb2.VerifyResponse(error=str(exc))

    async def Generate(self, request, context):
        async with self._lock:
            try:
                return self._generate(
                    session_id=request.session_id,
                    max_new_tokens=int(request.max_new_tokens),
                    stop_token_ids=set(int(token) for token in request.stop_token_ids),
                )
            except Exception as exc:
                logging.exception("Generate request failed")
                return spec_pb2.GenerateResponse(error=str(exc))

    async def ResetSession(self, request, context):
        async with self._lock:
            self.sessions.pop(request.session_id, None)
            maybe_clear_cuda_cache(self.device)
            return spec_pb2.ResetSessionResponse(ok=True)

    async def _idle_power(self, duration_s: float):
        if duration_s <= 0:
            raise ValueError("duration_s must be greater than 0.")

        sampler = VerifierPowerSampler(
            interval_s=self.power_interval_s,
            gpu_index=self.gpu_index,
        )
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
        sampler = VerifierPowerSampler(
            interval_s=self.power_interval_s,
            gpu_index=self.gpu_index,
        )

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

        self.sessions[session_id] = VerifierSession(
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

    def _verify(
        self,
        session_id: str,
        draft_tokens: List[int],
        append_replacement: bool,
        stop_token_ids: set,
        base_committed_tokens: int,
    ):
        session = self._get_session(session_id)
        if not draft_tokens:
            raise ValueError("draft_tokens must not be empty.")
        self._require_committed_len(
            session,
            base_committed_tokens,
            phase="Verify",
        )
        sampler = VerifierPowerSampler(
            interval_s=self.power_interval_s,
            gpu_index=self.gpu_index,
        )

        maybe_synchronize(self.device)
        sampler.start()
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model(
                input_ids=token_tensor(draft_tokens, self.device),
                past_key_values=session.past_key_values,
                use_cache=True,
            )
            draft_logits = out.logits
            full_draft_past = out.past_key_values

            verifier_predictions = [
                argmax_token(
                    session.next_logits
                    if index == 0
                    else next_logits_at(draft_logits, index - 1)
                )
                for index in range(len(draft_tokens))
            ]
            verifier_predictions.append(argmax_token(last_next_logits(draft_logits)))
            decision = plan_verify_decision(
                draft_tokens=draft_tokens,
                verifier_predictions=verifier_predictions,
                append_replacement_requested=append_replacement,
                stop_token_ids=stop_token_ids,
            )
            accepted = decision.accepted_tokens
            replacement_token = decision.replacement_token

            if accepted == len(draft_tokens):
                committed_past = full_draft_past
                committed_next_logits = last_next_logits(draft_logits)
                committed_len = session.seq_len + accepted
            elif accepted == 0:
                committed_past = session.past_key_values
                committed_next_logits = session.next_logits
                committed_len = session.seq_len
            else:
                committed_len = session.seq_len + accepted
                committed_past = clone_or_crop_past_key_values(
                    full_draft_past,
                    committed_len,
                )
                committed_next_logits = next_logits_at(draft_logits, accepted - 1)

            appended_replacement = decision.append_replacement
            if appended_replacement:
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

        rails, n_samples = sampler.summarize(t0, t1)
        return spec_pb2.VerifyResponse(
            accepted_tokens=accepted,
            replacement_token=replacement_token,
            latency_ms=(t1 - t0) * 1000.0,
            rails=self._rail_messages(rails),
            n_power_samples=n_samples,
            appended_replacement=appended_replacement,
            committed_tokens=committed_len,
        )

    def _generate(
        self,
        session_id: str,
        max_new_tokens: int,
        stop_token_ids: set,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")
        session = self._get_session(session_id)
        sampler = VerifierPowerSampler(
            interval_s=self.power_interval_s,
            gpu_index=self.gpu_index,
        )

        generated_tokens: List[int] = []
        past = session.past_key_values
        next_logits = session.next_logits

        maybe_synchronize(self.device)
        sampler.start()
        t0 = time.monotonic()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                next_token_id = argmax_token(next_logits)
                generated_tokens.append(next_token_id)
                out = self.model(
                    input_ids=single_token_tensor(next_token_id, self.device),
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                next_logits = last_next_logits(out.logits)
                if int(next_token_id) in stop_token_ids:
                    break

        maybe_synchronize(self.device)
        t1 = time.monotonic()
        sampler.stop()

        session.past_key_values = past
        session.next_logits = next_logits
        session.seq_len += len(generated_tokens)

        rails, n_samples = sampler.summarize(t0, t1)
        return spec_pb2.GenerateResponse(
            generated_tokens=generated_tokens,
            latency_ms=(t1 - t0) * 1000.0,
            rails=self._rail_messages(rails),
            n_power_samples=n_samples,
        )

    def _get_session(self, session_id: str) -> VerifierSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"Unknown session_id: {session_id}") from exc

    def _require_committed_len(
        self,
        session: VerifierSession,
        base_committed_tokens: int,
        phase: str,
    ) -> None:
        if base_committed_tokens <= 0:
            raise ValueError(f"{phase} requires base_committed_tokens.")
        if session.seq_len != base_committed_tokens:
            raise ValueError(
                f"Verifier session length mismatch during {phase}: "
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
        nvidia_smi_gpu_clock_mhz=args.gpu_clock_mhz,
        nvidia_smi_gpu_index=args.gpu_index,
    )
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", args.max_message_mb * 1024 * 1024),
            ("grpc.max_receive_message_length", args.max_message_mb * 1024 * 1024),
        ]
    )
    spec_pb2_grpc.add_VerifierServicer_to_server(
        VerifierServicer(
            model_name=args.model,
            dtype=parse_dtype(args.dtype),
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            frequency_lock=frequency_lock,
            power_interval_s=args.power_interval,
            gpu_index=args.gpu_index,
        ),
        server,
    )
    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    await server.start()
    logging.info("Verifier server listening on %s", bind_addr)
    try:
        await server.wait_for_termination()
    finally:
        if args.restore_frequency_on_exit:
            frequency_lock.restore()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speculative decoding verifier server")
    parser.add_argument("--model", required=True, help="Verifier causal LM id/path")
    parser.add_argument("--host", default="[::]")
    parser.add_argument("--port", type=int, default=50062)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--power-interval", type=float, default=0.01)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-clock-mhz", type=int, default=None)
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
