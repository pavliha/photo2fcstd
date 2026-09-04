import os

import numpy as np
import torch
import torch.nn as nn

N_POINTS = 192
VOCAB = N_POINTS + 3
TOK_LINE, TOK_ARC, TOK_EOS = N_POINTS, N_POINTS + 1, N_POINTS + 2
MAX_LEN = 100


def features(x):
    d = np.roll(x, -1, axis=-2) - x
    ang = np.arctan2(d[..., 1], d[..., 0])
    turn = np.unwrap(np.diff(np.concatenate([ang, ang[..., :1]], axis=-1), axis=-1))
    turn = np.clip(turn, -1.0, 1.0)
    return np.concatenate([x, d, turn[..., None]], axis=-1).astype(np.float32)


def encode_target(seq):
    out = []
    for j in range(0, len(seq), 2):
        if seq[j] < 0:
            break
        out += [TOK_ARC if seq[j] == 1 else TOK_LINE, int(seq[j + 1])]
    return out + [TOK_EOS]


class SeqNet(nn.Module):
    def __init__(self, d=128, heads=4, enc_layers=4, dec_layers=4):
        super().__init__()
        self.inp = nn.Linear(5, d)
        pos = torch.arange(N_POINTS)[:, None] * torch.exp(
            -np.log(10000.0) * torch.arange(0, d, 2)[None] / d)
        pe = torch.zeros(N_POINTS, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos), torch.cos(pos)
        self.register_buffer("pe", pe)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True),
            enc_layers)
        self.tok = nn.Embedding(VOCAB, d)
        self.dpos = nn.Embedding(MAX_LEN, d)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True),
            dec_layers)
        self.out = nn.Linear(d, VOCAB)

    def encode(self, x):
        return self.encoder(self.inp(x) + self.pe[None])

    def forward(self, x, tgt):
        mem = self.encode(x)
        L = tgt.shape[1]
        h = self.tok(tgt) + self.dpos(torch.arange(L, device=tgt.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        return self.out(self.decoder(h, mem, tgt_mask=mask))

    @torch.no_grad()
    def decode(self, x):
        mem = self.encode(x)
        B = x.shape[0]
        tgt = torch.full((B, 1), TOK_LINE, dtype=torch.long, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        last_idx = torch.full((B,), -1, dtype=torch.long, device=x.device)
        outs = [[] for _ in range(B)]
        for step in range(MAX_LEN - 1):
            h = self.tok(tgt) + self.dpos(torch.arange(tgt.shape[1], device=x.device))[None]
            mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], device=x.device)
            logits = self.out(self.decoder(h, mem, tgt_mask=mask))[:, -1]
            want_type = step % 2 == 0
            bad = torch.full_like(logits, float("-inf"))
            if want_type:
                logits = torch.where(torch.zeros_like(logits, dtype=torch.bool).index_fill_(
                    1, torch.tensor([TOK_LINE, TOK_ARC, TOK_EOS], device=x.device), True), logits, bad)
            else:
                allow = torch.zeros_like(logits, dtype=torch.bool)
                allow[:, :N_POINTS] = True
                idx = torch.arange(N_POINTS, device=x.device)[None]
                allow[:, :N_POINTS] &= idx > last_idx[:, None]
                logits = torch.where(allow, logits, bad)
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, TOK_EOS), nxt)
            for b in range(B):
                if not done[b]:
                    if want_type and nxt[b] == TOK_EOS:
                        done[b] = True
                    else:
                        outs[b].append(int(nxt[b]))
                        if nxt[b] < N_POINTS:
                            last_idx[b] = nxt[b]
            if done.all():
                break
            tgt = torch.cat([tgt, nxt[:, None]], dim=1)
        spans = []
        for o in outs:
            s = []
            for j in range(0, len(o) - 1, 2):
                if o[j] in (TOK_LINE, TOK_ARC) and o[j + 1] < N_POINTS:
                    s.append((int(o[j + 1]), 1 if o[j] == TOK_ARC else 0))
            spans.append(s)
        return spans


def device():
    forced = os.environ.get("P2F_SEQ_DEVICE")
    if forced:
        return forced
    return "mps" if torch.backends.mps.is_available() else "cpu"
