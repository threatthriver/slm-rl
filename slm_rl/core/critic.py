"""Value Network / Critic for PPO on Small Language Models."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from slm_rl.config import ModelConfig


class SLMCritic(nn.Module):
    """
    Critic (Value Network) that estimates token-level or sequence-level values.
    Built with a lightweight linear projection on top of language model hidden representations.
    """
    def __init__(
        self,
        config: ModelConfig,
        backbone: Optional[PreTrainedModel] = None,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.torch_dtype, torch.float32)

        if backbone is not None:
            self.backbone = backbone
            h_size = hidden_size or backbone.config.hidden_size
        elif config.is_mock:
            from transformers.models.gpt2 import GPT2Config, GPT2Model
            mock_cfg = GPT2Config(
                vocab_size=1000,
                n_positions=1024,
                n_embd=64,
                n_layer=2,
                n_head=2,
            )
            self.backbone = GPT2Model(mock_cfg).to(device=self.device, dtype=self.dtype)
            h_size = 64
        else:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(
                config.model_name_or_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)
            h_size = hidden_size or self.backbone.config.hidden_size

        self.value_head = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.Tanh(),
            nn.Linear(h_size, 1),
        ).to(device=self.device, dtype=self.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar values for all token positions.
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            values: (batch_size, seq_len)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # outputs[0] is last_hidden_state
        last_hidden_state = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        values = self.value_head(last_hidden_state).squeeze(-1)  # (batch_size, seq_len)
        return values

    def compute_response_values(
        self,
        seq_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract values corresponding to response tokens.
        """
        values = self.forward(seq_ids, attention_mask)
        resp_start_idx = prompt_seq_len - 1
        response_values = values[:, resp_start_idx:-1]
        response_mask = attention_mask[:, resp_start_idx + 1:]
        return response_values, response_mask
