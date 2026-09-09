"""Policy wrapper for Small Language Models."""

import copy
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from slm_rl.config import ModelConfig
from slm_rl.core.utils import selective_log_probs


class SLMPolicy(nn.Module):
    """
    Policy model wrapper around HuggingFace Causal Language Models.
    Provides generation, response-token log-probabilities, and reference model handling.
    """
    def __init__(
        self,
        config: ModelConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.torch_dtype, torch.float32)

        if model is not None and tokenizer is not None:
            self.model = model.to(device=self.device, dtype=self.dtype)
            self.tokenizer = tokenizer
        elif config.is_mock:
            # Create a tiny 2-layer GPT2 for offline/instant testing
            mock_cfg = GPT2Config(
                vocab_size=1000,
                n_positions=1024,
                n_embd=64,
                n_layer=2,
                n_head=2,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                attn_pdrop=0.0,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_implementation="eager" if self.device.type == "mps" else "sdpa",
            )
            self.model = GPT2LMHeadModel(mock_cfg).to(device=self.device, dtype=self.dtype)
            from transformers import PreTrainedTokenizerFast
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
            
            raw_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            raw_tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"], vocab_size=1000)
            raw_tokenizer.train_from_iterator(["hello world 1 2 3 4 5 6 7 8 9 0 <think> <answer>"], trainer)
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw_tokenizer,
                pad_token="<pad>",
                bos_token="<bos>",
                eos_token="<eos>",
                unk_token="<unk>",
            )
            self.tokenizer.pad_token_id = 0
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2
        else:
            import os
            import json
            model_path = config.model_name_or_path
            is_peft_checkpoint = False
            base_model_path = model_path

            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
                is_peft_checkpoint = True
                try:
                    with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                        meta = json.load(f)
                        base_model_path = meta.get("base_model_name_or_path", base_model_path)
                except Exception:
                    pass

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            attn_impl = "eager" if self.device.type == "mps" else "sdpa"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                dtype=self.dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            ).to(self.device)

            if is_peft_checkpoint:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
                self.is_lora = True
                print(f"✓ Loaded trained LoRA checkpoint from {model_path} on base {base_model_path}")
            else:
                self.model = base_model

        self.is_lora = getattr(self, "is_lora", False)
        if config.use_lora and not self.is_lora and not config.is_mock:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                target_modules = ["c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=target_modules,
                )
                self.model = get_peft_model(self.model, peft_config)
                self.is_lora = True
                print("✓ Attached LoRA trainable adapters. Base model frozen for zero-memory reference pass.")
            except Exception as e:
                print(f"Warning: Could not enable LoRA: {e}. Falling back to full parameters.")

        # Set tokenizer padding side to left for batched causal generation
        self.tokenizer.padding_side = "left"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw logits for input_ids."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def create_reference_policy(self) -> "SLMPolicy":
        """
        Create reference model for KL penalty.
        If LoRA is active, reuses the same model by disabling adapters during inference (zero memory overhead).
        """
        if getattr(self, "is_lora", False):
            # In LoRA mode, active model with adapter disabled acts as reference model
            return self

        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_policy = SLMPolicy(self.config, model=ref_model, tokenizer=self.tokenizer)
        return ref_policy


    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
    ) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        """
        Generate responses for a list of prompt strings.
        Supports sampling multiple completions per prompt (for GRPO).
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p

        # Tokenize prompts with left-padding
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_lens = attention_mask.sum(dim=-1).tolist()

        # If num_return_sequences > 1, replicate prompt lengths for each sample
        if num_return_sequences > 1:
            expanded_prompt_lens = []
            for plen in prompt_lens:
                expanded_prompt_lens.extend([plen] * num_return_sequences)
            prompt_lens = expanded_prompt_lens

        do_sample = temperature > 0.0
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": num_return_sequences,
        }

        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-4)
            generation_kwargs["top_p"] = top_p

        # Stop cleanly once </answer> is generated to prevent runaway hallucination loops
        if not self.config.is_mock:
            generation_kwargs["stop_strings"] = ["</answer>"]
            generation_kwargs["tokenizer"] = self.tokenizer

        seq_ids = self.model.generate(**generation_kwargs)
        seq_mask = (seq_ids != self.tokenizer.pad_token_id).long()

        # Decode full sequences and completion-only texts
        total_samples = seq_ids.size(0)
        full_texts = self.tokenizer.batch_decode(seq_ids, skip_special_tokens=False)
        completion_texts = []
        for i in range(total_samples):
            prompt_seq_len = input_ids.size(1)
            response_tokens = seq_ids[i, prompt_seq_len:]
            completion_texts.append(self.tokenizer.decode(response_tokens, skip_special_tokens=True))

        return {
            "seq_ids": seq_ids,
            "attention_mask": seq_mask,
            "prompt_lens": prompt_lens,
            "full_texts": full_texts,
            "completion_texts": completion_texts,
            "prompt_seq_len": input_ids.size(1),
        }

    def compute_response_log_probs(
        self,
        seq_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_seq_len: int,
        is_reference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log-probabilities for the generated response tokens only.
        Slices logits before vocabulary projection to minimize memory footprint.
        """
        if is_reference and getattr(self, "is_lora", False):
            with self.model.disable_adapter():
                logits = self.forward(seq_ids, attention_mask)
        else:
            logits = self.forward(seq_ids, attention_mask)

        # Slice to response tokens only:
        # Token at (prompt_seq_len - 1) predicts the first response token at prompt_seq_len.
        resp_start_idx = prompt_seq_len - 1
        resp_logits = logits[:, resp_start_idx:-1, :].contiguous()
        resp_labels = seq_ids[:, resp_start_idx + 1:].contiguous()
        response_mask = attention_mask[:, resp_start_idx + 1:].contiguous()

        # Compute log probs for response tokens only (avoids prompt token overhead)
        response_log_probs = selective_log_probs(resp_logits, resp_labels)

        # Precise Post-EOS Masking:
        if self.tokenizer.eos_token_id is not None:
            is_eos = (resp_labels == self.tokenizer.eos_token_id)
            eos_cumsum = torch.cumsum(is_eos.long(), dim=-1)
            valid_eos_mask = (eos_cumsum == 0) | ((eos_cumsum == 1) & is_eos)
            response_mask = response_mask * valid_eos_mask.long()

        return response_log_probs, response_mask

    @staticmethod
    def clear_memory():
        """Proactively clear accelerator cache on MPS/CUDA to avoid memory fragmentation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


