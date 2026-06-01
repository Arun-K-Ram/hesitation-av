"""
backend/ml/scene_predictor.py

Connects CNN scene classifier + MLP weight predictor.

Pipeline:
  frame → CNN → scene_type
  scene_type + frame_features → MLP → [α, β] weights
  weights → AmbiguityFusion → A(t)
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from pathlib import Path
from PIL import Image

ML_DIR = Path(__file__).parent

CLASSES = {
    0: "pedestrian_curb",
    1: "merge_hesitation",
    2: "occluded_intersection",
}

# Scene-specific weight priors
# Based on ablation findings:
# pedestrian_curb:      behavioral ambiguity dominates
# merge_hesitation:     both equally important
# occluded_intersection: perceptual ambiguity dominates
SCENE_WEIGHT_PRIORS = {
    0: {"alpha": 0.35, "beta": 0.65},  # pedestrian: more behavioral
    1: {"alpha": 0.50, "beta": 0.50},  # merge: balanced
    2: {"alpha": 0.70, "beta": 0.30},  # occluded: more perceptual
}


#  CNN Scene Classifier 

class SceneClassifier:
    def __init__(self, model_path: str = None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build EfficientNetB2 (best from sweep)
        self.model = models.efficientnet_b2(
            weights=None
        )
        in_f = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 3),
        )

        # Load weights
        if model_path is None:
            model_path = ML_DIR / "cnn_best_v2.pth"

        if Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path,
                           map_location=self.device,
                           weights_only=True)
            )
            print(f"[CNN] Loaded from {model_path}")
        else:
            print(f"[CNN] No weights found at {model_path} "
                  f"- using random weights")

        self.model.to(self.device)
        self.model.eval()

        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Rolling prediction buffer for stability
        self._pred_buffer = []
        self._buffer_size = 10

    def predict(self, frame: np.ndarray) -> dict:
        """
        Args:
            frame: BGR numpy array from OpenCV

        Returns dict:
            scene_type:  int (0, 1, 2)
            scene_name:  str
            confidence:  float
            probs:       list of 3 class probabilities
        """
        # Convert BGR → RGB → PIL
        rgb   = frame[:, :, ::-1].copy()
        img   = Image.fromarray(rgb)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        # Rolling buffer for temporal smoothing
        self._pred_buffer.append(pred)
        if len(self._pred_buffer) > self._buffer_size:
            self._pred_buffer.pop(0)

        # Majority vote over buffer
        from collections import Counter
        stable_pred = Counter(self._pred_buffer).most_common(1)[0][0]

        return {
            "scene_type": stable_pred,
            "scene_name": CLASSES[stable_pred],
            "confidence": conf,
            "probs":      probs.tolist(),
            "raw_pred":   pred,
        }


#  MLP Ambiguity Weight Predictor 

class AmbiguityWeightPredictor:
    """
    Loads trained MLP and predicts optimal [α, β] weights
    based on 8 scene features.

    Falls back to scene-type priors if MLP not available.
    """

    def __init__(self):
        model_path = ML_DIR / "model_weights.npz"
        stats_path = ML_DIR / "feature_stats.npy"

        self._mlp_available = False

        if model_path.exists() and stats_path.exists():
            data = np.load(model_path)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.W3 = data["W3"]
            self.b3 = data["b3"]

            stats = np.load(stats_path)
            self.feature_mean = stats[0]
            self.feature_std  = stats[1]

            self._mlp_available = True
            print("[MLP] Ambiguity weight predictor loaded OK")
        else:
            print("[MLP] No weights found - using scene priors")

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass through numpy MLP."""
        x = (x - self.feature_mean) / (self.feature_std + 1e-8)
        a1 = self._relu(x @ self.W1 + self.b1)
        a2 = self._relu(a1 @ self.W2 + self.b2)
        out = a2 @ self.W3 + self.b3
        return float(np.clip(out[0], 0.0, 1.0))

    def predict_weights(self, scene_type: int,
                        features: np.ndarray = None) -> dict:
        """
        Predict optimal α, β for ambiguity fusion.

        Args:
            scene_type: CNN prediction (0, 1, 2)
            features:   8-dim scene feature vector (optional)

        Returns:
            {"alpha": float, "beta": float}
        """
        if self._mlp_available and features is not None:
            # Use MLP prediction
            ambiguity_score = self._forward(features)

            # Convert ambiguity score to weights
            # High ambiguity → trust behavioral more
            # Low ambiguity  → trust perceptual more
            beta  = float(np.clip(0.3 + ambiguity_score * 0.5, 0.3, 0.8))
            alpha = 1.0 - beta

            return {
                "alpha":  round(alpha, 4),
                "beta":   round(beta, 4),
                "source": "mlp",
                "ambiguity_score": round(ambiguity_score, 4),
            }

        # Fallback to scene-type priors
        prior = SCENE_WEIGHT_PRIORS.get(
            scene_type,
            {"alpha": 0.45, "beta": 0.55}
        )
        return {
            "alpha":  prior["alpha"],
            "beta":   prior["beta"],
            "source": "prior",
            "ambiguity_score": 0.0,
        }


#  Combined predictor

class SceneAmbiguityPredictor:
    """
    Single interface combining CNN + MLP.
    Used by live pipeline.
    """

    def __init__(self):
        self.cnn = SceneClassifier()
        self.mlp = AmbiguityWeightPredictor()

    def predict(self, frame: np.ndarray,
                scene_features: np.ndarray = None) -> dict:
        """
        Args:
            frame:          OpenCV BGR frame
            scene_features: 8-dim feature vector (optional)

        Returns combined prediction dict.
        """
        # CNN scene classification
        scene = self.cnn.predict(frame)

        # MLP weight prediction
        weights = self.mlp.predict_weights(
            scene["scene_type"], scene_features
        )

        return {
            "scene_type":      scene["scene_type"],
            "scene_name":      scene["scene_name"],
            "scene_confidence": scene["confidence"],
            "scene_probs":     scene["probs"],
            "alpha":           weights["alpha"],
            "beta":            weights["beta"],
            "weight_source":   weights["source"],
            "ambiguity_score": weights["ambiguity_score"],
        }