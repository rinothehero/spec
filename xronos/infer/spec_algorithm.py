from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class VerifyDecision:
    accepted_tokens: int
    accepted_stop: bool
    replacement_token: int
    append_replacement: bool
    committed_token_delta: int


def truncate_accepted_at_stop(
    draft_tokens: List[int],
    accepted_tokens: int,
    stop_token_ids: Iterable[int],
) -> Tuple[int, bool]:
    if accepted_tokens < 0 or accepted_tokens > len(draft_tokens):
        raise ValueError("accepted_tokens must be within the draft length.")

    stops = {int(token) for token in stop_token_ids}
    if not stops:
        return accepted_tokens, False

    for index, token in enumerate(draft_tokens[:accepted_tokens]):
        if int(token) in stops:
            return index + 1, True
    return accepted_tokens, False


def should_append_replacement(
    accepted_tokens: int,
    draft_length: int,
    append_replacement_requested: bool,
    accepted_stop: bool,
) -> bool:
    if accepted_tokens < 0 or accepted_tokens > draft_length:
        raise ValueError("accepted_tokens must be within the draft length.")
    return (
        not accepted_stop
        and (accepted_tokens < draft_length or append_replacement_requested)
    )


def plan_verify_decision(
    draft_tokens: Sequence[int],
    verifier_predictions: Sequence[int],
    append_replacement_requested: bool,
    stop_token_ids: Iterable[int],
) -> VerifyDecision:
    if not draft_tokens:
        raise ValueError("draft_tokens must not be empty.")
    if len(verifier_predictions) < len(draft_tokens) + 1:
        raise ValueError(
            "verifier_predictions must include one prediction per draft token "
            "plus one extra replacement prediction."
        )

    accepted = 0
    for index, draft_token in enumerate(draft_tokens):
        if int(verifier_predictions[index]) != int(draft_token):
            accepted = index
            break
    else:
        accepted = len(draft_tokens)

    accepted, accepted_stop = truncate_accepted_at_stop(
        [int(token) for token in draft_tokens],
        accepted,
        stop_token_ids,
    )
    replacement_token = int(verifier_predictions[accepted])
    append_replacement = should_append_replacement(
        accepted_tokens=accepted,
        draft_length=len(draft_tokens),
        append_replacement_requested=append_replacement_requested,
        accepted_stop=accepted_stop,
    )
    return VerifyDecision(
        accepted_tokens=accepted,
        accepted_stop=accepted_stop,
        replacement_token=replacement_token,
        append_replacement=append_replacement,
        committed_token_delta=accepted + int(append_replacement),
    )
