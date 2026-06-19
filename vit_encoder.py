# models/vit_encoder.py
import torch
import torch.nn as nn
from transformers import ViTModel
from config import SystemConfig

class SpatialViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained(SystemConfig.VIT_MODEL_NAME)
        self.feature_dim = self.vit.config.hidden_size

    def forward(self, pixel_values):
        # Extracts the [CLS] token representing the global desk state
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]