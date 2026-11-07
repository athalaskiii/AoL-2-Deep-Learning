"""Melatih ulang kedua anggota model akhir (seed 42) dan menyimpan bobotnya untuk demo prediksi di notebook 3.

Hasil laporan tetap berasal dari outputs/results.jsonl (3 seed). Run di sini ditulis ke outputs/models/ agar tidak
menimpa hasil tersebut; bobotnya tidak ikut di-commit karena ukurannya (~3,6 GB).

HOASA_COOLDOWN=60 uv run python scripts/save_final.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hoasa import train  # noqa: E402
from train import COOLDOWN, EXPERIMENTS  # noqa: E402

OUT = train.OUT_DIR / "models"
train.RESULTS = OUT / "results.jsonl"
train.PRED_DIR = OUT / "preds"

for key in ["indobert_large_aspdesc", "xlmr_large_aspdesc"]:
    kw = EXPERIMENTS[key][1]
    if (OUT / train.slug(kw["name"]) / "model.pt").exists():
        continue
    train.run_transformer(seed=42, save_dir=OUT / train.slug(kw["name"]), **kw)
    time.sleep(COOLDOWN)
