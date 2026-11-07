"""Arsitektur untuk ACSA multi-output: setiap model menghasilkan logit (batch, 10 aspek, N_CLS kelas)."""
import torch
import torch.nn as nn

from .data import ASPECTS

N_CLS = 4   # neg, neut, pos, neg_pos; saat prediksi probabilitas neg_pos digabung ke neg (lihat train.fold)


class Head(nn.Module):
    """Satu lapis linear bersama untuk 10 aspek x 3 kelas."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, len(ASPECTS) * N_CLS)

    def forward(self, h):
        return self.out(self.drop(h)).view(-1, len(ASPECTS), N_CLS)


# --------------------------------------------------------------------------- berbasis kata (embedding FastText)
class WordModel(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float), freeze=False, padding_idx=0)
        self.drop = nn.Dropout(0.3)


class LSTM(WordModel):
    """LSTM searah 2 lapis; representasi = state tersembunyi pada token terakhir (bukan padding)."""

    def __init__(self, emb, hidden=256):
        super().__init__(emb)
        self.rnn = nn.LSTM(emb.shape[1], hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.head = Head(hidden, 0.3)

    def forward(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        last = (x != 0).sum(1).clamp(min=1) - 1
        return self.head(h[torch.arange(len(x)), last])


class BiLSTM(WordModel):
    """BiLSTM 2 lapis; representasi = state akhir arah maju + state awal arah mundur."""

    def __init__(self, emb, hidden=128):
        super().__init__(emb)
        self.hidden = hidden
        self.rnn = nn.LSTM(emb.shape[1], hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = Head(2 * hidden, 0.3)

    def forward(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        last = (x != 0).sum(1).clamp(min=1) - 1
        fwd = h[torch.arange(len(x)), last, :self.hidden]
        bwd = h[:, 0, self.hidden:]
        return self.head(torch.cat([fwd, bwd], -1))


class BiLSTMAttn(WordModel):
    """BiLSTM 2 lapis + attention pooling dan max pooling atas semua token (Zhou et al., 2016)."""

    def __init__(self, emb, hidden=128):
        super().__init__(emb)
        self.rnn = nn.LSTM(emb.shape[1], hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = Head(4 * hidden, 0.3)

    def forward(self, x):
        mask = x != 0
        h, _ = self.rnn(self.drop(self.emb(x)))
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        att = (a.unsqueeze(-1) * h).sum(1)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        return self.head(torch.cat([att, mx], -1))


class TextCNN(WordModel):
    """CNN teks (Kim, 2014): filter lebar 2-5, max-over-time pooling."""

    def __init__(self, emb, n_filters=128, widths=(2, 3, 4, 5)):
        super().__init__(emb)
        self.convs = nn.ModuleList(nn.Conv1d(emb.shape[1], n_filters, w, padding=w - 1) for w in widths)
        self.head = Head(n_filters * len(widths), 0.5)

    def forward(self, x):
        mask = (x != 0).unsqueeze(1)
        e = self.drop(self.emb(x)).transpose(1, 2)
        feats = []
        for conv in self.convs:
            c = torch.relu(conv(e))[..., :x.shape[1]]
            feats.append(c.masked_fill(~mask, -1e9).max(-1).values)
        return self.head(torch.cat(feats, -1))


WORD_MODELS = {"lstm": LSTM, "bilstm": BiLSTM, "bilstm_attn": BiLSTMAttn, "cnn": TextCNN}


# --------------------------------------------------------------------------- transformer
class TransformerCLS(nn.Module):
    """Encoder pretrained + head linear di atas vektor token pertama ([CLS] / <s>)."""

    def __init__(self, name):
        from transformers import AutoModel
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.head = Head(self.enc.config.hidden_size, 0.1)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(h)


class TransformerAspect(nn.Module):
    """Encoder pretrained + attention per aspek: tiap aspek punya vektor query sendiri yang memilih token relevan,
    lalu classifier per aspek di atas [vektor hasil attention; vektor token pertama].

    `query_init` (10, hidden) opsional, mis. representasi encoder atas deskripsi aspek; tanpa itu query diacak.
    """

    def __init__(self, name, query_init=None):
        from transformers import AutoModel
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        d = self.enc.config.hidden_size
        self.query = nn.Parameter(torch.randn(len(ASPECTS), d) * 0.02 if query_init is None else query_init.clone())
        self.drop = nn.Dropout(0.1)
        self.w = nn.Parameter(torch.empty(len(ASPECTS), 2 * d, N_CLS))
        self.b = nn.Parameter(torch.zeros(len(ASPECTS), N_CLS))
        for w in self.w:
            nn.init.xavier_uniform_(w)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # (B, T, d)
        s = torch.einsum("btd,ad->bat", h, self.query) / h.shape[-1] ** 0.5
        a = s.masked_fill(attention_mask[:, None] == 0, -1e4).softmax(-1)                    # (B, 10, T)
        z = torch.cat([a @ h, h[:, :1].expand(-1, len(ASPECTS), -1)], -1)                    # (B, 10, 2d)
        return torch.einsum("bak,akc->bac", self.drop(z), self.w) + self.b
