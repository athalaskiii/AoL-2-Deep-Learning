"""Menjalankan eksperimen pemodelan (notebook 2). Run yang sudah ada di outputs/results.jsonl dilewati.

uv run python scripts/train.py tfidf lstm bilstm bilstm_attn cnn
uv run python scripts/train.py indobert_base indobertweet mbert xlmr_base indobert_large xlmr_large
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers.utils import logging as hf_logging  # noqa: E402

from hoasa import train  # noqa: E402

hf_logging.disable_progress_bar()

# kunci -> (jenis, argumen). Model kata memakai text_norm, transformer memakai text_clean (rekomendasi notebook 1).
EXPERIMENTS = {
    "tfidf": ("tfidf", {"name": "TF-IDF + LR"}),
    "lstm": ("word", {"name": "LSTM", "arch": "lstm"}),
    "bilstm": ("word", {"name": "BiLSTM", "arch": "bilstm"}),
    "bilstm_attn": ("word", {"name": "BiLSTM + Attention", "arch": "bilstm_attn"}),
    "cnn": ("word", {"name": "TextCNN", "arch": "cnn"}),
    "mbert": ("tr", {"name": "mBERT-base", "backbone": "google-bert/bert-base-multilingual-uncased"}),
    "xlmr_base": ("tr", {"name": "XLM-R-base", "backbone": "FacebookAI/xlm-roberta-base", "lr": 2e-5}),
    "indobertweet": ("tr", {"name": "IndoBERTweet-base", "backbone": "indolem/indobertweet-base-uncased"}),
    "indobert_base": ("tr", {"name": "IndoBERT-base", "backbone": "indobenchmark/indobert-base-p1"}),
    "indobert_large": ("tr", {"name": "IndoBERT-large", "backbone": "indobenchmark/indobert-large-p2"}),
    "xlmr_large": ("tr", {"name": "XLM-R-large", "backbone": "FacebookAI/xlm-roberta-large", "lr": 1e-5}),
    # mitigasi hasil analisis error (dikembangkan di IndoBERT-base, dipilih dengan valid)
    "indobert_base_asp": ("tr", {"name": "IndoBERT-base + AspAttn", "backbone": "indobenchmark/indobert-base-p1",
                                 "head": "aspect", "head_lr": 1e-3}),
    "indobert_base_aspdesc": ("tr", {"name": "IndoBERT-base + AspAttn-Desc", "backbone": "indobenchmark/indobert-base-p1",
                                     "head": "aspect_desc", "head_lr": 1e-3}),
    "indobert_large_aspdesc": ("tr", {"name": "IndoBERT-large + AspAttn-Desc", "backbone": "indobenchmark/indobert-large-p2",
                                      "head": "aspect_desc", "head_lr": 1e-3}),
    "xlmr_large_aspdesc": ("tr", {"name": "XLM-R-large + AspAttn-Desc", "backbone": "FacebookAI/xlm-roberta-large",
                                  "lr": 1e-5, "head": "aspect_desc", "head_lr": 1e-3}),
    "indobert_base_ls": ("tr", {"name": "IndoBERT-base + LS0.1", "backbone": "indobenchmark/indobert-base-p1",
                                "smooth": 0.1}),
}
RUNNERS = {"word": train.run_word, "tr": train.run_transformer}
COOLDOWN = int(os.environ.get("HOASA_COOLDOWN", "0"))   # detik jeda antar run, agar GPU tidak terus-menerus penuh

if __name__ == "__main__":
    for key in sys.argv[1:]:
        kind, kw = EXPERIMENTS[key]
        have = train.done()
        if kind == "tfidf":
            if (kw["name"], 0) not in have:
                train.run_tfidf(**kw)
            continue
        for seed in train.SEEDS:
            if (kw["name"], seed) not in have:
                RUNNERS[kind](seed=seed, **kw)
                time.sleep(COOLDOWN)
