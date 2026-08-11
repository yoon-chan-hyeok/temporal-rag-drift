"""Concurrent, rate-limited, resumable OpenAI sampling."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.build_contexts import build_context, parse_context_config
from src.data.load_dataset import DatasetRecord, load_dataset
from src.generation.base import GenerationConfig
from src.generation.prompting import build_chat_messages
from src.pipeline.sample_responses import _max_requested_samples, _prompt_question, _response_row
from src.utils.io import (
    append_jsonl,
    copy_config_to_run,
    init_run_subdirs,
    load_yaml,
    project_root,
    read_jsonl,
    resolve_path,
    setup_logging,
)
from src.utils.seed import derive_seed, set_seed
from src.utils.text import stable_text_hash

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingJob:
    record: DatasetRecord
    condition: str
    sample_idx: int
    question: str
    context: str
    context_hash: str
    sample_seed: int
    estimated_input_tokens: int
    estimated_tokens: int


class SlidingWindowLimiter:
    """Limit estimated requests and tokens over a rolling minute."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.events: deque[tuple[float, int]] = deque()
        self.token_total = 0
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        if tokens > self.tokens_per_minute:
            raise ValueError(
                f"One request is estimated at {tokens} tokens, above the "
                f"{self.tokens_per_minute} TPM limit."
            )
        while True:
            async with self.lock:
                now = time.monotonic()
                cutoff = now - 60.0
                while self.events and self.events[0][0] <= cutoff:
                    _, expired_tokens = self.events.popleft()
                    self.token_total -= expired_tokens
                request_ok = len(self.events) < self.requests_per_minute
                token_ok = self.token_total + tokens <= self.tokens_per_minute
                if request_ok and token_ok:
                    self.events.append((now, tokens))
                    self.token_total += tokens
                    return
                wait_seconds = (
                    max(0.05, self.events[0][0] + 60.0 - now)
                    if self.events
                    else 0.25
                )
            await asyncio.sleep(wait_seconds)


def estimate_message_input_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative input-token approximation for pacing and cost planning."""
    characters = sum(len(message.get("content", "")) for message in messages)
    return max(1, math.ceil(characters / 3.6) + 16)


def estimate_message_tokens(
    messages: list[dict[str, str]],
    max_output_tokens: int,
) -> int:
    """Conservative total-token approximation used for client-side TPM pacing."""
    return estimate_message_input_tokens(messages) + max_output_tokens


def completion_request_kwargs(
    generation_cfg: GenerationConfig,
    messages: list[dict[str, str]],
    sample_seed: int,
) -> dict[str, Any]:
    """Build Chat Completions arguments across legacy and GPT-5 model families."""
    extra = generation_cfg.extra or {}
    kwargs: dict[str, Any] = {
        "model": generation_cfg.model_name,
        "messages": messages,
    }
    if not bool(extra.get("omit_sampling_parameters", False)):
        kwargs["temperature"] = generation_cfg.temperature
        kwargs["top_p"] = generation_cfg.top_p
        if not bool(extra.get("omit_seed", False)):
            kwargs["seed"] = sample_seed
    if bool(extra.get("use_max_completion_tokens", False)):
        kwargs["max_completion_tokens"] = generation_cfg.max_new_tokens
    else:
        kwargs["max_tokens"] = generation_cfg.max_new_tokens
    reasoning_effort = extra.get("reasoning_effort")
    if reasoning_effort:
        kwargs["reasoning_effort"] = str(reasoning_effort)
    return kwargs


def build_jobs(
    config: dict[str, Any],
    completed: set[tuple[str, str, int]],
) -> tuple[list[SamplingJob], int]:
    root = project_root()
    seed = int(config.get("seed", 42))
    dataset_cfg = config.get("dataset", {})
    records = load_dataset(
        dataset_cfg.get("path"),
        max_questions=dataset_cfg.get("max_questions"),
        base_dir=root,
    )
    conditions = [str(value) for value in config.get("conditions", [])]
    generation_raw = config.get("generation", {})
    generation_cfg = GenerationConfig.from_mapping(generation_raw)
    n_samples = _max_requested_samples(generation_raw, generation_cfg.n_samples)
    context_cfg = parse_context_config(
        config.get("context") if isinstance(config.get("context"), dict) else None
    )
    jobs: list[SamplingJob] = []
    total = len(records) * len(conditions) * n_samples
    for record in records:
        for condition in conditions:
            context = build_context(
                record,
                condition,
                context_cfg,
                seed=derive_seed(seed, record.id, condition, "context"),
            )
            question = _prompt_question(record, condition, generation_cfg)
            messages = build_chat_messages(
                question,
                context,
                prompt_mode=str((generation_cfg.extra or {}).get("prompt_mode", "grounded")),
                system_prompt_override=(
                    str((generation_cfg.extra or {}).get("system_prompt_override"))
                    if (generation_cfg.extra or {}).get("system_prompt_override")
                    else None
                ),
            )
            estimated_tokens = estimate_message_tokens(
                messages,
                generation_cfg.max_new_tokens,
            )
            estimated_input_tokens = estimate_message_input_tokens(messages)
            context_hash = stable_text_hash(context)
            for sample_idx in range(n_samples):
                key = (record.id, condition, sample_idx)
                if key in completed:
                    continue
                jobs.append(
                    SamplingJob(
                        record=record,
                        condition=condition,
                        sample_idx=sample_idx,
                        question=question,
                        context=context,
                        context_hash=context_hash,
                        sample_seed=derive_seed(seed, record.id, condition, sample_idx),
                        estimated_input_tokens=estimated_input_tokens,
                        estimated_tokens=estimated_tokens,
                    )
                )
    return jobs, total


async def run_sampling_async(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    concurrency: int | None = None,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve_path(config_path, base_dir=root))
    seed = int(config.get("seed", 42))
    set_seed(seed)
    run_dir_path = resolve_path(run_dir, base_dir=root)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    subdirs = init_run_subdirs(run_dir_path)
    setup_logging(subdirs["logs"] / "sampling_async_openai.log")
    copy_config_to_run(config, run_dir_path)

    responses_path = subdirs["samples"] / "responses.jsonl"
    existing = read_jsonl(responses_path)
    completed = {
        (str(row.get("question_id")), str(row.get("condition")), int(row.get("sample_idx", -1)))
        for row in existing
        if str(row.get("answer", "")).strip() and not row.get("error")
    }
    jobs, total = build_jobs(config, completed)
    estimated_tokens = sum(job.estimated_tokens for job in jobs)
    estimated_input_tokens = sum(job.estimated_input_tokens for job in jobs)
    estimated_output_tokens_max = sum(
        job.estimated_tokens - job.estimated_input_tokens for job in jobs
    )
    generation_cfg = GenerationConfig.from_mapping(config.get("generation", {}))
    extra = generation_cfg.extra or {}
    concurrency_value = int(concurrency or extra.get("async_concurrency", 20))
    rpm_value = int(requests_per_minute or extra.get("requests_per_minute", 450))
    tpm_value = int(tokens_per_minute or extra.get("tokens_per_minute", 180000))
    plan = {
        "run_dir": str(run_dir_path),
        "total_requests": total,
        "already_completed": len(completed),
        "pending_requests": len(jobs),
        "estimated_pending_tokens": estimated_tokens,
        "estimated_pending_input_tokens": estimated_input_tokens,
        "estimated_pending_output_tokens_max": estimated_output_tokens_max,
        "concurrency": concurrency_value,
        "requests_per_minute": rpm_value,
        "tokens_per_minute": tpm_value,
    }
    if dry_run:
        return plan
    if not jobs:
        return plan

    api_key = os.getenv(generation_cfg.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing {generation_cfg.api_key_env}. Set it in the current PowerShell "
            "session before starting the pilot."
        )
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AsyncOpenAI,
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
        )
    except ImportError as exc:
        raise ImportError("Install openai to use asynchronous OpenAI sampling.") from exc

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": generation_cfg.request_timeout,
        "max_retries": 0,
    }
    if generation_cfg.base_url:
        client_kwargs["base_url"] = generation_cfg.base_url
    client = AsyncOpenAI(**client_kwargs)
    limiter = SlidingWindowLimiter(rpm_value, tpm_value)
    semaphore = asyncio.Semaphore(concurrency_value)
    write_lock = asyncio.Lock()
    failures_path = subdirs["logs"] / "failed_requests.jsonl"
    usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
    permanent_errors = (
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        PermissionDeniedError,
    )
    transient_errors = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        APIStatusError,
    )
    retryable_errors = transient_errors + (ValueError,)
    current_failures = 0

    async def sample_one(job: SamplingJob, progress: tqdm) -> None:
        nonlocal current_failures
        messages = build_chat_messages(
            job.question,
            job.context,
            prompt_mode=str(extra.get("prompt_mode", "grounded")),
            system_prompt_override=(
                str(extra.get("system_prompt_override"))
                if extra.get("system_prompt_override")
                else None
            ),
        )
        last_error: Exception | None = None
        for attempt in range(1, generation_cfg.max_retries + 1):
            try:
                await limiter.acquire(job.estimated_tokens)
                async with semaphore:
                    response = await client.chat.completions.create(
                        **completion_request_kwargs(
                            generation_cfg,
                            messages,
                            job.sample_seed,
                        )
                    )
                answer = (response.choices[0].message.content or "").strip()
                if not answer:
                    raise ValueError("OpenAI returned an empty answer.")
                row = _response_row(
                    record_id=job.record.id,
                    question=job.record.question,
                    gold_answer=job.record.gold_answer,
                    condition=job.condition,
                    sample_idx=job.sample_idx,
                    answer=answer,
                    context_hash=job.context_hash,
                    model_name=generation_cfg.model_name,
                    backend="openai_async",
                    sample_seed=job.sample_seed,
                )
                row["attempt"] = attempt
                row["api_request_id"] = getattr(response, "_request_id", None)
                row["resolved_model_name"] = str(
                    getattr(response, "model", generation_cfg.model_name)
                )
                response_usage = getattr(response, "usage", None)
                if response_usage is not None:
                    row["input_tokens"] = int(getattr(response_usage, "prompt_tokens", 0) or 0)
                    row["output_tokens"] = int(getattr(response_usage, "completion_tokens", 0) or 0)
                async with write_lock:
                    append_jsonl([row], responses_path)
                    usage["requests"] += 1
                    usage["retries"] += attempt - 1
                    usage["input_tokens"] += int(row.get("input_tokens", 0))
                    usage["output_tokens"] += int(row.get("output_tokens", 0))
                    progress.update(1)
                return
            except permanent_errors:
                raise
            except RateLimitError as exc:
                message = str(exc).lower()
                if any(
                    marker in message
                    for marker in (
                        "requests per day",
                        "rpd",
                        "daily limit",
                        "insufficient_quota",
                    )
                ):
                    raise RuntimeError(
                        "The account daily request/quota limit was reached. "
                        "Re-run the same command after the limit resets."
                    ) from exc
                last_error = exc
                if attempt < generation_cfg.max_retries:
                    delay = min(
                        generation_cfg.retry_max_seconds,
                        generation_cfg.retry_min_seconds * (2 ** (attempt - 1)),
                    )
                    await asyncio.sleep(delay + random.random())
            except retryable_errors as exc:
                last_error = exc
                if attempt < generation_cfg.max_retries:
                    delay = min(
                        generation_cfg.retry_max_seconds,
                        generation_cfg.retry_min_seconds * (2 ** (attempt - 1)),
                    )
                    await asyncio.sleep(delay + random.random())

        failure = {
            "question_id": job.record.id,
            "condition": job.condition,
            "sample_idx": job.sample_idx,
            "error": repr(last_error),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        async with write_lock:
            append_jsonl([failure], failures_path)
            current_failures += 1

    try:
        with tqdm(total=total, initial=len(completed), desc="openai async sampling") as progress:
            # Validate credentials/model access with one real checkpoint before fan-out.
            await sample_one(jobs[0], progress)
            if len(jobs) > 1:
                tasks = [
                    asyncio.create_task(sample_one(job, progress))
                    for job in jobs[1:]
                ]
                try:
                    await asyncio.gather(*tasks)
                except Exception:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
    finally:
        await client.close()

    plan.update(usage)
    plan["failed_requests"] = current_failures
    if current_failures:
        raise RuntimeError(
            f"{current_failures} requests failed after retries. Re-run the same command "
            "to resume only missing samples."
        )
    return plan
