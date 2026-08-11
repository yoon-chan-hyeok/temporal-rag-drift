"""Local Hugging Face causal language model backend."""

from __future__ import annotations

import logging

import torch

from src.generation.base import BaseGenerator, GenerationConfig
from src.generation.prompting import build_chat_messages, build_plain_prompt
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


class LocalHFGenerator(BaseGenerator):
    """Generate answers with a local Hugging Face causal LM."""

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers and torch to use backend=local_hf") from exc

        if not config.model_name:
            raise ValueError("generation.model_name is required for local_hf")
        extra = config.extra or {}
        local_files_only = bool(extra.get("local_files_only", False))
        LOGGER.info("Loading local HF model: %s", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=local_files_only)
        dtype_name = str(extra.get("torch_dtype", "float16" if torch.cuda.is_available() else "float32")).lower()
        dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported local_hf torch_dtype: {dtype_name}")
        device_map = extra.get("device_map", "auto" if torch.cuda.is_available() else None)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=local_files_only,
        )
        self.prompt_mode = str(extra.get("prompt_mode", "grounded"))
        self.system_prompt_override = extra.get("system_prompt_override")
        self.use_chat_template = bool(extra.get("use_chat_template", True))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, question: str, context: str, sample_seed: int | None = None) -> str:
        """Generate one answer from a local model."""
        set_seed(sample_seed)
        system_prompt_override = str(self.system_prompt_override) if self.system_prompt_override else None
        if self.use_chat_template and self.tokenizer.chat_template:
            messages = build_chat_messages(
                question,
                context,
                prompt_mode=self.prompt_mode,
                system_prompt_override=system_prompt_override,
            )
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            prompt = build_plain_prompt(
                question,
                context,
                prompt_mode=self.prompt_mode,
                system_prompt_override=system_prompt_override,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-5),
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
