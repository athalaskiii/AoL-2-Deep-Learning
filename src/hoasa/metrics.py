"""Metrik evaluasi HoASA.

`f1_indonlu` mereplikasi `absa_metrics_fn` di repositori IndoNLU (utils/metrics.py): semua sel
(ulasan x aspek) diratakan menjadi satu daftar, lalu dihitung macro-F1 sklearn atas kelas yang
muncul di label atau prediksi. Karena `neg_pos` tidak ada di valid/test, satu prediksi `neg_pos`
saja menambah kelas keempat dengan F1 = 0 dan menurunkan skor secara drastis.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

NEG, NEUT, POS, NEG_POS = 0, 1, 2, 3


def f1_indonlu(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true.ravel(), y_pred.ravel(), average="macro")


def f1_pair(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Micro-F1 atas pasangan (aspek, polaritas) yang dibahas (gaya SemEval ACSA); neut = tidak dibahas."""
    t, p = y_true.ravel(), y_pred.ravel()
    tp = ((t == p) & (t != NEUT)).sum()
    n_pred, n_true = (p != NEUT).sum(), (t != NEUT).sum()
    return float(2 * tp / (n_pred + n_true)) if n_pred + n_true else 0.0


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "f1_indonlu": f1_indonlu(y_true, y_pred),
        "f1_pair": f1_pair(y_true, y_pred),
        "acc": accuracy_score(y_true.ravel(), y_pred.ravel()),
        "exact": float((y_true == y_pred).all(1).mean()),
    }
