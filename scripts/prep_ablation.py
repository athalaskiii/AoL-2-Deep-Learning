"""Ablation preprocessing untuk model neural (notebook 1, bagian 7.8).

5-fold CV pada train + valid (fold dari data/processed/cv_folds.csv), konfigurasi tetap untuk semua
tingkat preprocessing. Hasil per fold ditambahkan ke outputs/prep_ablation.jsonl (run yang sudah ada dilewati).

uv run python scripts/prep_ablation.py bilstm
uv run python scripts/prep_ablation.py indobert
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hoasa.data import ASPECTS, PROC_DIR, ROOT, load_processed, to_ids  # noqa: E402
from hoasa.metrics import metrics  # noqa: E402
from hoasa.preprocess import TOKEN_RE  # noqa: E402

LEVELS = ["raw", "clean", "norm", "stem", "lemma", "stop"]
OUT = ROOT / "outputs" / "prep_ablation.jsonl"
DEV = "cuda"
N_CLS = 3


def load_cv():
    cv = pd.concat([load_processed("train"), load_processed("valid")], ignore_index=True)
    folds = pd.read_csv(PROC_DIR / "cv_folds.csv", keep_default_na=False)
    assert (folds.review.values == cv.review.values).all(), "urutan cv_folds.csv tidak cocok"
    cv["fold"] = folds.fold.values
    cv["text_raw"] = cv.review
    return cv, to_ids(cv)


def done():
    if not OUT.exists():
        return set()
    return {(r["model"], r["tingkat"], r["fold"]) for r in map(json.loads, OUT.read_text().splitlines())}


def log(model, level, fold, m, secs):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps({"model": model, "tingkat": level, "fold": fold, **m, "detik": round(secs)}) + "\n")
    print(f"{model} {level} fold{fold}: f1={m['f1_indonlu']:.4f} pair={m['f1_pair']:.4f} ({secs:.0f}s)", flush=True)


# --------------------------------------------------------------------------- BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float), freeze=False, padding_idx=0)
        self.drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(emb.shape[1], 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.att = nn.Linear(256, 1)
        self.out = nn.Linear(512, len(ASPECTS) * N_CLS)

    def forward(self, x):
        mask = x != 0
        h, _ = self.lstm(self.drop(self.emb(x)))
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        att = (a.unsqueeze(-1) * h).sum(1)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        return self.out(self.drop(torch.cat([att, mx], -1))).view(-1, len(ASPECTS), N_CLS)


def run_bilstm(cv, Y, ft, level, fold, epochs=30, bs=32, max_len=128):
    torch.manual_seed(fold)
    tr, te = (cv.fold != fold).values, (cv.fold == fold).values
    toks = cv[f"text_{level}"].str.lower().map(TOKEN_RE.findall)   # tokenisasi sama untuk semua tingkat
    vocab = {}
    for ws in toks[tr]:
        for w in ws:
            vocab.setdefault(w, len(vocab) + 2)
    emb = np.zeros((len(vocab) + 2, 300), dtype=np.float32)
    for w, k in vocab.items():
        emb[k] = ft[w]                       # FastText membentuk vektor dari n-gram karakter untuk kata OOV
    ids = [[vocab.get(w, 1) for w in ws][:max_len] or [1] for ws in toks]
    X = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long)
    for i, s in enumerate(ids):
        X[i, :len(s)] = torch.tensor(s)
    Xtr, Ytr = X[tr].to(DEV), torch.tensor(Y[tr].copy()).to(DEV)
    model = BiLSTM(emb).to(DEV)
    opt = torch.optim.Adam([{"params": model.emb.parameters(), "lr": 1e-4},
                            {"params": [p for n, p in model.named_parameters() if not n.startswith("emb")], "lr": 1e-3}])
    g = torch.Generator().manual_seed(fold)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(Xtr), generator=g).split(bs):
            loss = F.cross_entropy(model(Xtr[b]).reshape(-1, N_CLS), Ytr[b].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X[te].to(DEV)).argmax(-1).cpu().numpy()
    return metrics(Y[te], pred)


# --------------------------------------------------------------------------- IndoBERT
def run_indobert(cv, Y, level, fold, name="indobenchmark/indobert-base-p1", epochs=10, bs=32, lr=3e-5, max_len=128):
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(fold)
    tok = AutoTokenizer.from_pretrained(name)
    enc = tok(cv[f"text_{level}"].tolist(), max_length=max_len, truncation=True, padding=True, return_tensors="pt")
    tr, te = np.where(cv.fold != fold)[0], np.where(cv.fold == fold)[0]
    enc_model = AutoModel.from_pretrained(name).to(DEV)
    head = nn.Linear(enc_model.config.hidden_size, len(ASPECTS) * N_CLS).to(DEV)
    drop = nn.Dropout(0.1)
    params = list(enc_model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / (0.1 * steps), max(0.0, (steps - s) / (0.9 * steps))))
    Yt = torch.tensor(Y)

    def forward(idx):
        batch = {k: v[idx].to(DEV) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = enc_model(**batch).last_hidden_state[:, 0]
            return head(drop(h)).float().view(-1, len(ASPECTS), N_CLS)

    g = torch.Generator().manual_seed(fold)
    for _ in range(epochs):
        enc_model.train()
        for b in torch.tensor(tr)[torch.randperm(len(tr), generator=g)].split(bs):
            loss = F.cross_entropy(forward(b).reshape(-1, N_CLS), Yt[b].to(DEV).reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
    enc_model.eval()
    with torch.no_grad():
        pred = torch.cat([forward(b).argmax(-1).cpu() for b in torch.tensor(te).split(128)]).numpy()
    return metrics(Y[te], pred)


if __name__ == "__main__":
    which = sys.argv[1]
    cv, Y = load_cv()
    have = done()
    model_name = {"bilstm": "BiLSTM + FastText", "indobert": "IndoBERT-base"}[which]
    ft = None
    if which == "bilstm":
        from gensim.models.fasttext import load_facebook_vectors
        ft = load_facebook_vectors(ROOT / "data" / "embeddings" / "cc.id.300.bin")
    for level in LEVELS:
        for fold in range(5):
            if (model_name, level, fold) in have:
                continue
            t0 = time.time()
            m = run_bilstm(cv, Y, ft, level, fold) if which == "bilstm" else run_indobert(cv, Y, level, fold)
            log(model_name, level, fold, m, time.time() - t0)
