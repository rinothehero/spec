from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_int_csv(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",")]
    if any(not item for item in items):
        raise ValueError("CSV integer lists must not contain empty items.")
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integers, got: {value}") from exc
    if not parsed:
        raise ValueError("At least one integer value is required.")
    return parsed


def parse_dtype(value: str) -> torch.dtype:
    if value not in DTYPES:
        choices = ", ".join(sorted(DTYPES))
        raise ValueError(f"Unsupported dtype '{value}'. Expected one of: {choices}")
    return DTYPES[value]


def maybe_synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def clone_or_crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    """Crop a HF KV cache to `max_length` tokens.

    Newer Transformers cache objects expose `crop`. Older tuple caches store
    keys/values as tensors with sequence length on the second-to-last axis.
    """
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop"):
        cropped = past_key_values
        result = cropped.crop(max_length)
        return cropped if result is None else result

    cropped_layers = []
    for layer in past_key_values:
        if not isinstance(layer, tuple):
            cropped_layers.append(layer)
            continue

        cropped_items = []
        for item in layer:
            if torch.is_tensor(item) and item.ndim >= 3:
                cropped_items.append(item[..., :max_length, :].contiguous())
            else:
                cropped_items.append(item)
        cropped_layers.append(tuple(cropped_items))
    return tuple(cropped_layers)


def single_token_tensor(token_id: int, device: str) -> torch.Tensor:
    return torch.tensor([[int(token_id)]], device=device, dtype=torch.long)


def token_tensor(token_ids: List[int], device: str) -> torch.Tensor:
    return torch.tensor([token_ids], device=device, dtype=torch.long)


def argmax_token(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def last_next_logits(output_logits: torch.Tensor) -> torch.Tensor:
    return output_logits[:, -1, :]


def next_logits_at(output_logits: torch.Tensor, index: int) -> torch.Tensor:
    return output_logits[:, index, :]


def maybe_clear_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_session_id(session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    import uuid

    return str(uuid.uuid4())


def load_tokenizer(model_or_tokenizer: str, trust_remote_code: bool, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_or_tokenizer,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _metadata_str(value) -> str:
    return "" if value is None else str(value)


def tokenizer_metadata(tokenizer) -> dict:
    try:
        tokenizer_len = len(tokenizer)
    except TypeError:
        tokenizer_len = ""
    return {
        "tokenizer_name_or_path": _metadata_str(
            getattr(tokenizer, "name_or_path", ""),
        ),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": _metadata_str(tokenizer_len),
        "tokenizer_base_vocab_size": _metadata_str(
            getattr(tokenizer, "vocab_size", ""),
        ),
        "tokenizer_bos_token_id": _metadata_str(
            getattr(tokenizer, "bos_token_id", ""),
        ),
        "tokenizer_eos_token_id": _metadata_str(
            getattr(tokenizer, "eos_token_id", ""),
        ),
        "tokenizer_pad_token_id": _metadata_str(
            getattr(tokenizer, "pad_token_id", ""),
        ),
        "tokenizer_unk_token_id": _metadata_str(
            getattr(tokenizer, "unk_token_id", ""),
        ),
    }


def load_causal_lm(
    model_name: str,
    dtype: torch.dtype,
    device: str,
    trust_remote_code: bool,
    local_files_only: bool,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested with device={device}, but CUDA is unavailable.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model


def model_metadata(model) -> dict:
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", "") if config is not None else ""
    if isinstance(architectures, (list, tuple)):
        architectures = ",".join(str(item) for item in architectures)
    try:
        parameter_count = int(model.num_parameters())
    except Exception:
        try:
            parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        except Exception:
            parameter_count = ""
    metadata = {
        "model_vocab_size": str(getattr(config, "vocab_size", "") if config is not None else ""),
        "model_parameter_count": _metadata_str(parameter_count),
        "model_type": str(getattr(config, "model_type", "") if config is not None else ""),
        "model_architectures": str(architectures or ""),
        "model_bos_token_id": _metadata_str(
            getattr(config, "bos_token_id", "") if config is not None else ""
        ),
        "model_eos_token_id": _metadata_str(
            getattr(config, "eos_token_id", "") if config is not None else ""
        ),
        "model_pad_token_id": _metadata_str(
            getattr(config, "pad_token_id", "") if config is not None else ""
        ),
    }
    return metadata
