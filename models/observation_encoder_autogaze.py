from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOGAZE_ROOT = REPO_ROOT / "AutoGaze"
if str(AUTOGAZE_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOGAZE_ROOT))

from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor  # noqa: E402
from autogaze.vision_encoders.siglip import SiglipVisionModel  # noqa: E402


ROLE_HIST = 0
ROLE_HEAD = 1
ROLE_LEFT = 2
ROLE_RIGHT = 3


class ObservationProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=960, hidden_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class AutoGazeObservationEncoder(nn.Module):
    SIMVLA_MEAN = (0.485, 0.456, 0.406)
    SIMVLA_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        out_dim: int,
        autogaze_model_path: str = "nvidia/AutoGaze",
        siglip_model_path: str = "google/siglip2-base-patch16-224",
        history_len: int = 8,
        projector_hidden_dim: int = 1536,
        gazing_ratio: float = 0.10,
        task_loss_requirement: float | None = None,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.gazing_ratio = float(gazing_ratio)
        self.task_loss_requirement = task_loss_requirement

        self.autogaze = AutoGaze.from_pretrained(autogaze_model_path)
        self.autogaze_processor = AutoGazeImageProcessor.from_pretrained(autogaze_model_path)

        self.siglip_processor = AutoImageProcessor.from_pretrained(siglip_model_path)
        self.siglip = SiglipVisionModel.from_pretrained(
            siglip_model_path,
            scales=self.autogaze.config.scales,
            attn_implementation="sdpa",
        )
        self.siglip.vision_model.embeddings.register_buffer(
            "position_ids",
            torch.arange(self.siglip.vision_model.embeddings.num_positions).expand((1, -1)),
            persistent=False,
        )

        self.patch_size = int(self.siglip.config.patch_size)
        self.scales = sorted(int(scale) for scale in str(self.siglip.config.scales).split("+"))
        self.num_tokens_each_group = sum((scale // self.patch_size) ** 2 for scale in self.scales)

        self.projector = ObservationProjector(
            in_dim=int(self.siglip.config.hidden_size),
            out_dim=out_dim,
            hidden_dim=projector_hidden_dim,
        )
        self.source_embed = nn.Embedding(4, out_dim)
        self.age_embed = nn.Embedding(self.history_len + 1, out_dim)

        self.register_buffer(
            "simvla_mean",
            torch.tensor(self.SIMVLA_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "simvla_std",
            torch.tensor(self.SIMVLA_STD, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        ag_mean = torch.tensor(self.autogaze_processor.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        ag_std = torch.tensor(self.autogaze_processor.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("autogaze_mean", ag_mean, persistent=False)
        self.register_buffer("autogaze_std", ag_std, persistent=False)
        self.autogaze_size = self._extract_size(self.autogaze_processor.size)

        sig_mean = torch.tensor(self.siglip_processor.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        sig_std = torch.tensor(self.siglip_processor.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("siglip_mean", sig_mean, persistent=False)
        self.register_buffer("siglip_std", sig_std, persistent=False)
        self.siglip_size = self._extract_size(self.siglip_processor.size)

    @staticmethod
    def _extract_size(size_cfg) -> int:
        if isinstance(size_cfg, dict):
            if "height" in size_cfg:
                return int(size_cfg["height"])
            if "shortest_edge" in size_cfg:
                return int(size_cfg["shortest_edge"])
            if "longest_edge" in size_cfg:
                return int(size_cfg["longest_edge"])
        if isinstance(size_cfg, int):
            return int(size_cfg)
        raise ValueError(f"Unsupported size config: {size_cfg}")

    def _simvla_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.simvla_std.to(x.device, x.dtype) + self.simvla_mean.to(x.device, x.dtype)).clamp(0.0, 1.0)

    def _prepare_autogaze_pixels(self, x: torch.Tensor) -> torch.Tensor:
        x = self._simvla_to_unit(x)
        B, T, C, _, _ = x.shape
        x = x.flatten(0, 1)
        x = F.interpolate(x, size=(self.autogaze_size, self.autogaze_size), mode="bicubic", align_corners=False, antialias=True)
        x = x.view(B, T, C, self.autogaze_size, self.autogaze_size)
        x = x * 2.0 - 1.0
        x = (x - self.autogaze_mean.to(x.device, x.dtype)) / self.autogaze_std.to(x.device, x.dtype)
        return x

    def _prepare_siglip_pixels(self, x: torch.Tensor) -> torch.Tensor:
        x = self._simvla_to_unit(x)
        B, T, C, _, _ = x.shape
        x = x.flatten(0, 1)
        x = F.interpolate(x, size=(self.siglip_size, self.siglip_size), mode="bicubic", align_corners=False, antialias=True)
        x = x.view(B, T, C, self.siglip_size, self.siglip_size)
        x = (x - self.siglip_mean.to(x.device, x.dtype)) / self.siglip_std.to(x.device, x.dtype)
        return x

    def _build_dense_group(self, batch_size: int, offset: int, device: torch.device):
        pos = torch.arange(self.num_tokens_each_group, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return {
            "gazing_pos": pos + offset,
            "if_padded_gazing": torch.zeros(batch_size, self.num_tokens_each_group, device=device, dtype=torch.bool),
            "num_gazing_each_frame": torch.tensor([self.num_tokens_each_group], device=device, dtype=torch.long),
        }

    def _build_unified_gazing_info(
        self,
        hist_info: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        parts_pos = []
        parts_pad = []
        group_lengths = []
        group_roles = []

        hist_lengths = hist_info["num_gazing_each_frame"].tolist()
        start = 0
        for length in hist_lengths:
            end = start + int(length)

            # history 的 gazing_pos 已经是 clip 内全局编号
            # 不要再加 hist_offset
            parts_pos.append(hist_info["gazing_pos"][:, start:end])
            parts_pad.append(hist_info["if_padded_gazing"][:, start:end])

            group_lengths.append(int(length))
            group_roles.append(ROLE_HIST)
            start = end

        # current groups 从 history 之后开始编号
        hist_offset = len(hist_lengths) * self.num_tokens_each_group

        head_info = self._build_dense_group(batch_size, hist_offset, device)
        left_info = self._build_dense_group(batch_size, hist_offset + self.num_tokens_each_group, device)
        right_info = self._build_dense_group(batch_size, hist_offset + self.num_tokens_each_group * 2, device)

        for role, info in (
            (ROLE_HEAD, head_info),
            (ROLE_LEFT, left_info),
            (ROLE_RIGHT, right_info),
        ):
            parts_pos.append(info["gazing_pos"])
            parts_pad.append(info["if_padded_gazing"])
            group_lengths.append(int(info["num_gazing_each_frame"][0]))
            group_roles.append(role)

        unified = {
            "gazing_pos": torch.cat(parts_pos, dim=1),
            "if_padded_gazing": torch.cat(parts_pad, dim=1),
            "num_gazing_each_frame": torch.tensor(group_lengths, device=device, dtype=torch.long),
            "group_roles": torch.tensor(group_roles, device=device, dtype=torch.long),
        }

        max_allowed = self.num_tokens_each_group * (len(hist_lengths) + 3)
        if unified["gazing_pos"].numel() > 0:
            max_pos = int(unified["gazing_pos"].max().item())
            if max_pos >= max_allowed:
                raise ValueError(
                    f"Unified gazing positions exceed range: "
                    f"max_pos={max_pos}, max_allowed={max_allowed - 1}, "
                    f"history_groups={len(hist_lengths)}, "
                    f"num_tokens_each_group={self.num_tokens_each_group}"
                )

        return unified

    def _build_token_metadata(
        self,
        unified_gazing_info: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ):
        group_lengths = unified_gazing_info["num_gazing_each_frame"].tolist()
        group_roles = unified_gazing_info["group_roles"].tolist()

        source_ids = []
        age_ids = []
        hist_group_count = sum(1 for role in group_roles if role == ROLE_HIST)
        hist_seen = 0
        for role, length in zip(group_roles, group_lengths):
            source_ids.append(torch.full((batch_size, int(length)), int(role), device=device, dtype=torch.long))
            if role == ROLE_HIST:
                age_val = hist_group_count - hist_seen
                age_ids.append(torch.full((batch_size, int(length)), age_val, device=device, dtype=torch.long))
                hist_seen += 1
            else:
                age_ids.append(torch.zeros(batch_size, int(length), device=device, dtype=torch.long))

        return torch.cat(source_ids, dim=1), torch.cat(age_ids, dim=1)

    def forward(
        self,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        head_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del image_mask  # RMBench always provides 3 views in this setup.
        batch_size = image_input.shape[0]
        device = image_input.device
        dtype = image_input.dtype

        head_history_ag = self._prepare_autogaze_pixels(head_history)
        autogaze_kwargs = {
            "gazing_ratio": self.gazing_ratio,
            "target_scales": self.scales,
            "target_patch_size": self.patch_size,
        }
        if self.task_loss_requirement is not None:
            autogaze_kwargs["task_loss_requirement"] = self.task_loss_requirement

        hist_info = self.autogaze({"video": head_history_ag}, generate_only=True, **autogaze_kwargs)

        current_views = image_input[:, :3]
        current_head = current_views[:, 0:1]
        current_left = current_views[:, 1:2]
        current_right = current_views[:, 2:3]
        siglip_pixels = torch.cat([head_history, current_head, current_left, current_right], dim=1)
        siglip_pixels = self._prepare_siglip_pixels(siglip_pixels)

        unified_gazing_info = self._build_unified_gazing_info(hist_info, batch_size, device)
        siglip_outputs = self.siglip(
            siglip_pixels,
            gazing_info=unified_gazing_info,
            output_hidden_states=False,
        )

        obs_tokens = self.projector(siglip_outputs.last_hidden_state.to(dtype))
        source_ids, age_ids = self._build_token_metadata(unified_gazing_info, batch_size, device)
        obs_tokens = obs_tokens + self.source_embed(source_ids)
        obs_tokens = obs_tokens + self.age_embed(age_ids)
        obs_attention_mask = (~unified_gazing_info["if_padded_gazing"]).long()

        return {
            "obs_tokens": obs_tokens,
            "obs_attention_mask": obs_attention_mask,
            "gazing_info": unified_gazing_info,
        }
