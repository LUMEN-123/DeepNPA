#!/usr/bin/env python3
# ==============================================
# ProtBERT + FocalLoss Ablation Experiment Script (fixed)
# ==============================================
import os
import random
import itertools
import math
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # fixed
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForTokenClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------- 全局常量 ----------
BIO_TAGS = ['O', 'B', 'I']
LABEL2ID = {t: i for i, t in enumerate(BIO_TAGS)}         # {'O':0,'B':1,'I':2}
ID2LABEL = {i: t for t, i in LABEL2ID.items()}            # {0:'O',1:'B',2:'I'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("logs", exist_ok=True)
os.makedirs("../result", exist_ok=True)

# ---------- 可复现 ----------
def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- 数据 ----------
def load_data(file_path: str):
    data = pd.read_excel(file_path)
    sequences = data["Sequence"].tolist()
    labels = data["BIO"].tolist()
    return sequences, labels

def sliding_window(seq: str, lab: List[str], max_len: int, overlap: int):
    out_seq, out_lab = [], []
    start = 0
    while start < len(seq):
        end = start + max_len
        out_seq.append(seq[start:end])
        out_lab.append(lab[start:end])
        if end >= len(seq):
            break
        start = end - overlap
    return out_seq, out_lab

def preprocess_and_split(file_path, max_len=256, overlap=64, seed=42):
    seqs, labs = load_data(file_path)
    assert len(seqs) == len(labs)
    Xtr, Xtmp, ytr, ytmp = train_test_split(seqs, labs, test_size=0.2, random_state=seed)
    Xval, Xtest, yval, ytest = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed)

    def apply(seqs, labs):
        new_s, new_l = [], []
        for s, l in zip(seqs, labs):
            # labels may be space separated string or list
            if isinstance(l, str):
                lab_list = l.split()
            else:
                lab_list = list(l)
            if len(s) > max_len:
                sub_s, sub_l = sliding_window(s, lab_list, max_len, overlap)
                new_s.extend(sub_s)
                new_l.extend(sub_l)
            else:
                new_s.append(s)
                new_l.append(lab_list)
        return new_s, new_l

    Xtr, ytr = apply(Xtr, ytr)
    Xval, yval = apply(Xval, yval)

    print(f"Train: {len(Xtr)} | Val: {len(Xval)} | Test: {len(Xtest)}")
    return Xtr, ytr, Xval, yval, Xtest, ytest

# ---------- Dataset ----------
class ProtDataset(Dataset):
    def __init__(self, seqs, labs, tokenizer, max_len=256):
        self.seqs = seqs
        self.labs = labs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        lab = self.labs[idx]
        # tokenizers expect split tokens when is_split_into_words=True
        tokens = list(seq)  # each AA as token
        if isinstance(lab, str):
            lab = lab.split()
        # encode
        enc = self.tok(tokens,
                       is_split_into_words=True,
                       truncation=True,
                       padding='max_length',
                       max_length=self.max_len,
                       return_tensors="pt")
        # build labels aligned with tokens: keep -100 for special tokens
        # We assume tokenizer adds 1 special token at front and 1 at end (BERT)
        label_ids = [LABEL2ID.get(l, LABEL2ID["O"]) for l in lab]
        label_ids = label_ids[: (self.max_len - 2)]  # leave room for special tokens
        # add -100 for [CLS] and [SEP]
        label_ids = [-100] + label_ids + [-100]
        # pad with -100 to max_len
        pad_len = self.max_len - len(label_ids)
        if pad_len > 0:
            label_ids += [-100] * pad_len
        else:
            label_ids = label_ids[:self.max_len]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

def collate(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}

def make_loader(X, y, batch_size, max_len, seed, shuffle, tokenizer, num_workers=0):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    g = torch.Generator()
    g.manual_seed(seed)
    ds = ProtDataset(X, y, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, worker_init_fn=worker_init_fn, generator=g,
                      num_workers=num_workers)

# ---------- 损失函数 ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        # alpha: sequence or None
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.ignore = ignore_index

    def forward(self, logits, targets):
        """
        logits: (B, L, C)
        targets: (B, L)
        """
        b, l, c = logits.shape
        probs = torch.softmax(logits, dim=-1)  # (B, L, C)
        # prepare mask & safe targets
        mask = (targets != self.ignore)        # (B, L)
        if mask.sum() == 0:
            return logits.new_zeros(())
        safe_targets = targets.clone()
        safe_targets[~mask] = 0  # avoid invalid gather index
        # gather prob of true class
        pt = probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)  # (B, L)
        focal_w = (1 - pt) ** self.gamma
        loss = -focal_w * torch.log(pt.clamp(min=1e-8))
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[safe_targets]
        loss = loss * mask.float()
        return loss.sum() / mask.sum().clamp(min=1.0)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        """
        preds: (B*L, C) or (N, C)
        target: (B*L,) or (N,)
        We'll handle masking inside.
        """
        num_classes = preds.size(-1)
        mask = (target != self.ignore_index)
        if mask.sum() == 0:
            return preds.new_zeros(())
        preds = preds[mask]
        target = target[mask].long()
        log_probs = F.log_softmax(preds, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

loss_CE = lambda: torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

# ---------- 训练与验证 ----------
def acc_per_label(logits, labels, id2tag):
    """
    logits: (B,L,C) tensor
    labels: (B,L) tensor with -100 for ignored positions
    id2tag: dict id->tag
    """
    preds = logits.argmax(-1).cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()
    mask = labels_np != -100
    if mask.sum() == 0:
        return {"Overall": None, "B": None, "I": None, "O": None}
    overall = (preds[mask] == labels_np[mask]).mean()
    out = {"Overall": overall}
    # build reverse map id->tag already provided
    for lab in ["B", "I", "O"]:
        # find label id
        lab_id = next(k for k, v in id2tag.items() if v == lab)
        m = (labels_np == lab_id) & mask
        out[lab] = (preds[m] == labels_np[m]).mean() if m.sum() else None
    return out

def flatten_for_loss(logits, labels):
    # logits: (B,L,C) -> (B*L, C)
    # labels: (B,L) -> (B*L,)
    B, L, C = logits.shape
    return logits.view(-1, C), labels.view(-1)

def train_one_epoch(model, loader, loss_fn, optim, sched, id2tag, device):
    model.train()
    loss_sum, steps = 0.0, 0
    acc_cum = {"B": 0.0, "I": 0.0, "O": 0.0, "Overall": 0.0}
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits  # (B,L,C)
        # compute loss
        # some loss fns expect (B*L, C) and (B*L,), others (B,L,C) and (B,L)
        if isinstance(loss_fn, (FocalLoss,)):
            loss = loss_fn(logits, batch["labels"])
        else:
            flat_logits, flat_labels = flatten_for_loss(logits, batch["labels"])
            loss = loss_fn(flat_logits, flat_labels)
        loss.backward()
        optim.step()
        if sched is not None:
            sched.step()
        optim.zero_grad()
        acc = acc_per_label(logits.detach(), batch["labels"], id2tag)
        loss_sum += loss.item()
        steps += 1
        for k in acc_cum:
            acc_cum[k] += acc[k] if acc[k] is not None else 0.0
    if steps == 0:
        return 0.0, {k: None for k in acc_cum}
    return loss_sum / steps, {k: (v / steps) for k, v in acc_cum.items()}

def validate_one_epoch(model, loader, loss_fn, id2tag, device):
    model.eval()
    loss_sum, steps = 0.0, 0
    acc_cum = {"B": 0.0, "I": 0.0, "O": 0.0, "Overall": 0.0}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            if isinstance(loss_fn, (FocalLoss,)):
                loss = loss_fn(logits, batch["labels"])
            else:
                flat_logits, flat_labels = flatten_for_loss(logits, batch["labels"])
                loss = loss_fn(flat_logits, flat_labels)
            acc = acc_per_label(logits, batch["labels"], id2tag)
            loss_sum += loss.item()
            steps += 1
            for k in acc_cum:
                acc_cum[k] += acc[k] if acc[k] is not None else 0.0
    if steps == 0:
        return 0.0, {k: None for k in acc_cum}
    return loss_sum / steps, {k: (v / steps) for k, v in acc_cum.items()}

class EarlyStop:
    def __init__(self, patience=3, delta=0, path="result/best.pt"):
        self.pat = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best = 0.0   # start at 0.0 (accuracy scale)
        self.early_stop = False

    def __call__(self, val_b_acc, model):
        if val_b_acc is None:
            return
        if val_b_acc > self.best + self.delta:
            self.best = val_b_acc
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"💾 Best model saved | Val B-Acc: {val_b_acc:.4f} -> {self.path}")
        else:
            self.counter += 1
            if self.counter >= self.pat:
                self.early_stop = True

# ---------- 主实验 ----------
def run_experiment(maxlen, batch_size, lr, epochs, loss_fn, model_name, seed=42):
    set_random_seed(seed)
    # tokenizer/model selection
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    Xtr, ytr, Xval, yval, Xte, yte = preprocess_and_split("../BIO_labeled_sequences_1.xlsx", max_len=maxlen, overlap=64, seed=seed)
    train_loader = make_loader(Xtr, ytr, batch_size, maxlen, seed, True, tokenizer)
    val_loader = make_loader(Xval, yval, batch_size, maxlen, seed, False, tokenizer)

    model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(BIO_TAGS)).to(DEVICE)

    # scheduler & optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    num_steps = max(1, epochs * len(train_loader))
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=int(0.1 * num_steps),
                              num_training_steps=num_steps) if num_steps > 0 else None

    early_stop = EarlyStop(patience=5, path=f"result/best_maxlen{maxlen}_bs{batch_size}_lr{lr}_{os.path.basename(model_name)}.pt")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, ID2LABEL, DEVICE)
        vl_loss, vl_acc = validate_one_epoch(model, val_loader, loss_fn, ID2LABEL, DEVICE)
        b_acc = vl_acc.get("B") if vl_acc else None
        print(f"[{os.path.basename(model_name)}] Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | val_loss {vl_loss:.4f} | B_acc {b_acc}")
        if b_acc is not None:
            early_stop(b_acc, model)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break
    return early_stop.best

# ---------- 网格搜索 ----------
if __name__ == "__main__":
    # define losses and models to compare
    losses = [
        FocalLoss(alpha=[1.0, 3.0, 2.0], gamma=2.0, ignore_index=-100),
        LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=-100),
        loss_CE()
    ]
    loss_names = ["FocalLoss", "LabelSmoothingCE", "CrossEntropy"]

    models = [
        "./prot_bert",           # local protbert folder (if exists)
        "./bert-base-uncased"      # baseline
    ]

    grid = list(itertools.product([256], [16], [2e-5], [100], list(range(len(losses))), range(len(models))))
    results = []
    for maxlen, bs, lr, ep, loss_idx, model_idx in grid:
        loss_fn = losses[loss_idx]
        loss_name = loss_names[loss_idx]
        model_name = models[model_idx]
        print(f"\n=== Running: model={model_name} | loss={loss_name} | maxlen={maxlen} bs={bs} lr={lr} epochs={ep} ===")
        try:
            best_b = run_experiment(maxlen, bs, lr, ep, loss_fn, model_name)
        except Exception as e:
            print(f"ERROR running experiment: {e}")
            best_b = float("nan")
        results.append({
            "model": model_name, "loss": loss_name, "maxlen": maxlen,
            "batch_size": bs, "lr": lr, "epochs": ep, "best_B_acc": best_b
        })
        # save intermediate results so progress is not lost
        pd.DataFrame(results).to_csv("../result/grid_search.csv", index=False)
        torch.cuda.empty_cache()
    print("✅ All done → result/grid_search.csv")
