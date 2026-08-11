"""Prompt templates for single-turn factual QA generation."""

from __future__ import annotations

GROUNDED_SYSTEM_PROMPT = (
    "Answer the question using the provided retrieved context. If the context is "
    "insufficient or contradictory, answer as best as possible without inventing unsupported facts. "
    "Return only the shortest final answer needed to answer the question. For a yes/no question, "
    "return only Yes or No. Do not explain your reasoning, cite documents, restate the question, "
    "or add qualifications."
)

HYBRID_SYSTEM_PROMPT = (
    "Answer the question using the provided retrieved context as your primary evidence. "
    "If the context is incomplete or contradictory, you may also rely on your own knowledge "
    "to give the best answer you can, but do not fabricate facts and prefer context-supported information when possible. "
    "Return only the shortest final answer needed to answer the question. For a yes/no question, "
    "return only Yes or No. Do not explain your reasoning, cite documents, restate the question, "
    "or add qualifications."
)

USER_TEMPLATE = """Question:
{question}

Retrieved Context:
{context}

Return only the final answer:
"""


def build_user_prompt(question: str, context: str) -> str:
    """Format the user message."""
    return USER_TEMPLATE.format(question=question, context=context)


def resolve_system_prompt(
    prompt_mode: str = "grounded",
    system_prompt_override: str | None = None,
) -> str:
    """Return the system prompt for one generation mode."""
    if system_prompt_override:
        return system_prompt_override
    mode = str(prompt_mode).strip().lower()
    if mode in {"grounded", "default"}:
        return GROUNDED_SYSTEM_PROMPT
    if mode in {"hybrid", "parametric_hybrid", "hybrid_answer"}:
        return HYBRID_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def build_chat_messages(
    question: str,
    context: str,
    prompt_mode: str = "grounded",
    system_prompt_override: str | None = None,
) -> list[dict[str, str]]:
    """Return OpenAI-compatible chat messages."""
    system_prompt = resolve_system_prompt(
        prompt_mode=prompt_mode,
        system_prompt_override=system_prompt_override,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(question, context)},
    ]


def build_plain_prompt(
    question: str,
    context: str,
    prompt_mode: str = "grounded",
    system_prompt_override: str | None = None,
) -> str:
    """Return a plain prompt for local causal language models."""
    system_prompt = resolve_system_prompt(
        prompt_mode=prompt_mode,
        system_prompt_override=system_prompt_override,
    )
    return f"System:\n{system_prompt}\n\nUser:\n{build_user_prompt(question, context)}\n"
