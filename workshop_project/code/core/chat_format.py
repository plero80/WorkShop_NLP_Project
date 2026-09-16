from __future__ import annotations

import re
from typing import Any


# Anthropic HH-RLHF stores conversations as alternating transcript blocks such as
# ``\n\nHuman: ...\n\nAssistant: ...``.  Requiring the marker to be at the
# start of the transcript or after a blank line avoids treating a normal
# line like ``Assistant: is a label`` inside message content as a new turn.
_HH_ROLE_RE = re.compile(
    r"(?:\A|(?:\r?\n){2,})[ \t]*(Human|Assistant):[ \t]*"
)


def _without_duplicated_bos(tokenizer: Any, text: str) -> str:
    bos_token = getattr(tokenizer, "bos_token", None)
    if isinstance(bos_token, str) and bos_token and text.startswith(bos_token):
        return text[len(bos_token) :]
    return text


def parse_hh_transcript(text: str) -> list[dict[str, str]]:
    """Parse an Anthropic HH-style transcript into chat-template messages.

    Returns an empty list when ``text`` is not an HH transcript.  This keeps
    ordinary prompts backward compatible: callers can then treat the whole
    string as one user turn.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matches = list(_HH_ROLE_RE.finditer(text))
    if not matches:
        return []

    # Ignore only surrounding whitespace before the first role marker.  If
    # there is non-whitespace prose before it, this is not a canonical HH row.
    if text[: matches[0].start()].strip():
        return []

    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        content_start = match.end()
        content_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
        content = text[content_start:content_end].strip()
        if not content:
            # Empty role blocks are malformed and should not silently become
            # valid training examples.
            raise ValueError("HH transcript contains an empty role block")
        role = "user" if match.group(1) == "Human" else "assistant"
        messages.append({"role": role, "content": content})

    # HH-RLHF contains a small number of noisy transcripts with repeated role
    # markers, e.g. ``Human: ...\n\nHuman: ...`` or an extra ``Assistant:``
    # emitted inside an assistant answer.  Crashing the entire experiment on
    # one such row is undesirable.  Coalesce adjacent blocks with the same
    # role into a single chat message; this preserves the text while restoring
    # a valid role sequence for the tokenizer chat template.
    if messages[0]["role"] != "user":
        return []

    normalized: list[dict[str, str]] = []
    for message in messages:
        if normalized and normalized[-1]["role"] == message["role"]:
            normalized[-1]["content"] = (
                normalized[-1]["content"].rstrip()
                + "\n\n"
                + message["content"].lstrip()
            )
        else:
            normalized.append(dict(message))
    return normalized


def serialize_hh_messages(messages: list[dict[str, str]]) -> str:
    """Serialize parsed messages to a stable HH transcript string."""
    if not messages:
        raise ValueError("messages must be non-empty")
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("messages must contain user/assistant string content")
        if not content.strip():
            raise ValueError("message content must be non-empty")
        label = "Human" if role == "user" else "Assistant"
        parts.append(f"{label}: {content.strip()}")
    return "\n\n".join(parts)


def extract_hh_prompt(chosen: str) -> str:
    """Return the conversation context before the final chosen assistant turn.

    HH-RLHF ``chosen`` rows contain the full conversation including the chosen
    final assistant response.  Policy generation must receive all preceding
    turns, not a substring ending at the first question mark.
    """
    messages = parse_hh_transcript(chosen)
    if not messages:
        return ""
    if messages[-1]["role"] != "assistant":
        return ""
    context = messages[:-1]
    if not context or context[-1]["role"] != "user":
        return ""
    return serialize_hh_messages(context)


def prompt_messages(prompt: str) -> list[dict[str, str]]:
    """Convert a stored prompt into chat messages.

    A canonical HH transcript is preserved as a multi-turn conversation.
    Ordinary strings remain one user message for compatibility with other
    datasets and manually supplied prompts.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    parsed = parse_hh_transcript(prompt)
    if parsed:
        if parsed[-1]["role"] != "user":
            raise ValueError("A generation prompt must end with a user turn")
        return parsed
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    return [{"role": "user", "content": prompt}]


def format_user_prompt(tokenizer: Any, prompt: str) -> str:
    """Format a prompt context, ending at the assistant generation prefix."""
    formatted = tokenizer.apply_chat_template(
        prompt_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(formatted, str):
        raise TypeError("tokenizer chat template must return text")
    return _without_duplicated_bos(tokenizer, formatted)


def format_prompt_answer(tokenizer: Any, prompt: str, answer: str) -> str:
    """Format the prompt context followed by one generated assistant answer."""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    # Immediate EOS is a valid (usually poor) policy action. Score empty
    # completions instead of crashing or silently dropping their prompts.
    messages = [*prompt_messages(prompt), {"role": "assistant", "content": answer}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(formatted, str):
        raise TypeError("tokenizer chat template must return text")
    return _without_duplicated_bos(tokenizer, formatted)
