
from __future__ import annotations

import os
import json
import hashlib
from functools import lru_cache
from typing import List, Dict, Tuple, Any


# ─── Skin validator (placeholder -- no validator model provided) ──────
class SkinValidator:
   

    def is_skin(self, image_path: str) -> Tuple[bool, float]:
        return True, 0.99


# ─── Fallback used only when torch or the checkpoint is unavailable ───
class DemoClassifier:
   
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def predict(self, image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        top_k = max(1, int(top_k))
        with open(image_path, "rb") as f:
            digest = hashlib.sha256(f.read()).digest()

        n = len(self.class_names)
        # Pick a pseudo-random but reproducible ordering of classes.
        scored = sorted(
            range(n),
            key=lambda i: hashlib.sha256(digest + str(i).encode()).hexdigest(),
        )
        out = []
        base_conf = 45.0
        for rank, idx in enumerate(scored[:top_k]):
            conf = max(5.0, base_conf - rank * 15.0)
            out.append({"class": self.class_names[idx], "confidence": float(conf)})
        return out


# ─── Real architecture (must match training notebook exactly) ─────────
def _build_real_model_class():
    
    import torch.nn as nn
    from torchvision import models

    class SkinDiseaseClassifier(nn.Module):
        

        def __init__(self, num_classes: int, dropout: float = 0.4):
            super().__init__()
            # weights=None: we're about to load our own trained state_dict,
            # so there's no need to also download ImageNet weights.
            backbone = models.efficientnet_b4(weights=None)

            self.features = backbone.features
            self.avgpool = backbone.avgpool
            in_features = backbone.classifier[1].in_features  # 1792 for B4

            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.SiLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout / 2),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(p=dropout / 4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            import torch

            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    return SkinDiseaseClassifier


class TorchClassifier:
    """Real classifier: loads best_model.pth and runs actual inference."""

    def __init__(self, checkpoint_path: str):
        import torch
        import torchvision.transforms as transforms

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Checkpoint may be a plain state_dict OR the dict the notebook
        # saves: {'model_state_dict': ..., 'class_names': ..., 'config': ...}
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            class_names = ckpt.get("class_names")
            cfg = ckpt.get("config", {})
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            class_names = ckpt.get("class_names")
            cfg = ckpt.get("config", {})
        else:
            # Assume the checkpoint *is* the state_dict.
            state_dict = ckpt
            class_names = None
            cfg = {}

        # Fall back to class_mapping.json next to this file if the
        # checkpoint didn't carry class names.
        if not class_names:
            class_names = _load_class_names_from_mapping()

        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_size = cfg.get("img_size", 256)
        mean = cfg.get("mean", [0.485, 0.456, 0.406])
        std = cfg.get("std", [0.229, 0.224, 0.225])
        dropout = cfg.get("dropout", 0.4)

        ModelClass = _build_real_model_class()
        self.model = ModelClass(num_classes=len(self.class_names), dropout=dropout)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Exact val/test transform pipeline from the training notebook.
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def predict(self, image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        top_k = max(1, min(int(top_k), len(self.class_names)))

        pil_img = Image.open(image_path).convert("RGB")
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_probs, top_idx = probs.topk(top_k)
        top_probs = top_probs.cpu().numpy()
        top_idx = top_idx.cpu().numpy()

        return [
            {"class": self.class_names[int(i)], "confidence": float(p) * 100.0}
            for i, p in zip(top_idx, top_probs)
        ]


# ─── Helpers ────────────────────────────────────────────────────────
def _load_class_names_from_mapping() -> List[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "class_mapping.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        names = mapping.get("class_names")
        if names:
            return names
    except Exception:
        pass
    # Last resort: keys from the disease database (kept in sync with
    # class_mapping.json, see skin_disease_data.py header comment).
    from skin_disease_data import SKIN_DISEASE_DATABASE

    return list(SKIN_DISEASE_DATABASE.keys())


def _checkpoint_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "model", "best_model.pth"),
        os.path.join(base_dir, "best_model.pth"),
        os.path.join(base_dir, "best_model (1).pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[1]  # default path used in the "not found" message


# ─── Public API used by app1.py ────────────────────────────────────
@lru_cache(maxsize=1)
def get_skin_validator() -> SkinValidator:
    return SkinValidator()


@lru_cache(maxsize=1)
def get_classifier():
 
    ckpt = _checkpoint_path()
    class_names = _load_class_names_from_mapping()

    if not os.path.exists(ckpt):
        print(
            f"[model] No checkpoint found at '{ckpt}'. "
            "Running in DEMO mode (not real predictions). "
            "Place best_model.pth next to this file (or under model/) "
            "to enable real inference."
        )
        return DemoClassifier(class_names)

    try:
        return TorchClassifier(ckpt)
    except ImportError as e:
        print(f"[model] torch/torchvision not available ({e}); using DEMO mode.")
        return DemoClassifier(class_names)
    except Exception as e:
        print(f"[model] Failed to load checkpoint '{ckpt}': {e}; using DEMO mode.")
        return DemoClassifier(class_names)
