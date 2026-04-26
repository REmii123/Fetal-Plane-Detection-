# model/vit_classifier.py
import os
import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 6, backbone_path: str | None = None):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        if backbone_path is not None and os.path.exists(backbone_path):
            state = torch.load(backbone_path, map_location="cpu")
            new_state = {}
            for k, v in state.items():
                if "fc.norm" in k:
                    new_state[k.replace("fc.norm.", "norm.")] = v
                else:
                    new_state[k] = v
            self.backbone.load_state_dict(new_state, strict=False)

        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return feats, logits
