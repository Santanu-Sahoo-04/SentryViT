# models/fusion.py
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

class SentryViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the FULLY TRAINED model that already knows 1000 objects
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    def forward(self, pixel_values):
        # We only need the current frame to identify the object
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixel_values)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
            # Map the highest score index to its text label
            object_name = self.vit.config.id2label[predicted_class_idx]
            return object_name