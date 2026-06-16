"""
Inferencia de autenticidad para Raspberry Pi Zero 2W.
Uso:
    from inferencia import AutenticidadDetector
    detector = AutenticidadDetector("modelos_autenticidad.joblib", "metadata.json")
    score = detector.evaluar(crop_imagen, label="serie_a", denominacion="100_Bs", lado="anverso")
    # score > 0 → probablemente real, score < 0 → probablemente falso
"""
import numpy as np
import joblib
import json
from pathlib import Path
from PIL import Image
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct


class AutenticidadDetector:
    def __init__(self, modelos_path, metadata_path):
        self.modelos = joblib.load(modelos_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

    def _extract_lbp(self, image, P=8, R=1, n_bins=32):
        lbp = local_binary_pattern(image, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-10
        return hist

    def _extract_dct(self, image, n_coeffs=64):
        dct2d = dct(dct(image.astype(np.float32).T, norm="ortho").T, norm="ortho")
        h, w = dct2d.shape
        zigzag = []
        for s in range(1, min(h, w)):
            for i in range(s + 1):
                j = s - i
                if i < h and j < w:
                    zigzag.append(dct2d[i, j])
                if j != i and j < h and i < w:
                    zigzag.append(dct2d[j, i])
            if len(zigzag) >= n_coeffs:
                break
        coeffs = np.array(zigzag[:n_coeffs], dtype=np.float32)
        std = coeffs.std() + 1e-10
        coeffs = (coeffs - coeffs.mean()) / std
        return coeffs

    def evaluar(self, crop, label, denominacion, lado):
        """
        Evalúa un crop y retorna (score_ensemble, es_real, detalles).
        score > umbral → real, score < umbral → falso/anómalo.
        """
        if isinstance(crop, (str, Path)):
            crop = np.array(Image.open(crop).convert("L"), dtype=np.uint8)
        elif isinstance(crop, Image.Image):
            crop = np.array(crop.convert("L"), dtype=np.uint8)

        group_key = (label, denominacion, lado)
        if group_key not in self.modelos:
            raise ValueError(f"Grupo no encontrado: {group_key}")

        md = self.modelos[group_key]

        # LBP score
        feat_lbp = self._extract_lbp(crop).reshape(1, -1)
        feat_lbp_s = md["scaler_lbp"].transform(feat_lbp)
        score_lbp = md["model_lbp"].decision_function(feat_lbp_s)[0]

        # DCT score
        feat_dct = self._extract_dct(crop).reshape(1, -1)
        feat_dct_s = md["scaler_dct"].transform(feat_dct)
        score_dct = md["model_dct"].decision_function(feat_dct_s)[0]

        # Ensemble
        score_ens = 0.6 * score_lbp + 0.4 * score_dct
        es_real = score_ens >= md["umbral_ensemble"]

        detalles = {
            "score_lbp": float(score_lbp),
            "score_dct": float(score_dct),
            "score_ensemble": float(score_ens),
            "umbral": md["umbral_ensemble"],
            "es_real": bool(es_real),
        }

        return float(score_ens), bool(es_real), detalles


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Uso: python inferencia.py <crop.png> <label> <denominacion> <lado>")
        print("Ej: python inferencia.py crop.png serie_a 100_Bs anverso")
        sys.exit(1)
    detector = AutenticidadDetector("modelos_autenticidad.joblib", "metadata.json")
    score, es_real, detalles = detector.evaluar(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print(f"Score: {score:.4f}")
    print(f"Es real: {es_real}")
    print(f"Detalles: {detalles}")
