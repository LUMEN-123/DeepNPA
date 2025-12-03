#!/usr/bin/env python3
# ==============================================
# ProtBERT + FocalLoss Ablation Evaluation Script
# ==============================================

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, BertForTokenClassification
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    roc_auc_score, average_precision_score
)
import seaborn as sns
import os, random, itertools
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForTokenClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------- 常量 ----------
BIO_TAGS = ['O', 'B', 'I']
LABEL2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2LABEL = {i: t for t, i in LABEL2ID.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 全局常量 ----------
BIO_TAGS = ['O', 'B', 'I']
LABEL2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2LABEL = {i: t for t, i in LABEL2ID.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# os.makedirs("result/logs", exist_ok=True)
# os.makedirs("result/result", exist_ok=True)

# ---------- 可复现 ----------
def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- 数据 ----------
def load_data(file_path):
    data = pd.read_excel(file_path)
    sequences = data["Sequence"].tolist()
    labels = data["BIO"].tolist()
    return sequences, labels

def sliding_window(seq, lab, max_len, overlap):
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

def preprocess_and_split(file_path, max_len=512, overlap=64, seed=42):
    seqs, labs = load_data(file_path)
    assert len(seqs) == len(labs)
    Xtr, Xtmp, ytr, ytmp = train_test_split(seqs, labs, test_size=0.2, random_state=seed)
    Xval, Xtest, yval, ytest = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed)

    def apply(seqs, labs):
        new_s, new_l = [], []
        for s, l in zip(seqs, labs):
            if isinstance(l, str):
                l = l.split()
            if len(s) > max_len:
                sub_s, sub_l = sliding_window(s, l, max_len, overlap)
                new_s.extend(sub_s)
                new_l.extend(sub_l)
            else:
                new_s.append(s)
                new_l.append(l)
        return new_s, new_l

    Xtr, ytr = apply(Xtr, ytr)
    Xval, yval = apply(Xval, yval)
    Xtest, ytest = apply(Xtest, ytest)

    print(f"Train: {len(Xtr)} | Val: {len(Xval)} | Test: {len(Xtest)}")
    return Xtr, ytr, Xval, yval, Xtest, ytest

# ---------- Dataset ----------
class ProtDataset(Dataset):
    def __init__(self, seqs, labs, tokenizer, max_len=512):
        self.seqs, self.labs, self.tok, self.max_len = seqs, labs, tokenizer, max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq, lab = list(self.seqs[idx]), self.labs[idx]
        if isinstance(lab, str):
            lab = lab.split()

        enc = self.tok(seq,
                       is_split_into_words=True,
                       truncation=True,
                       padding='max_length',
                       max_length=self.max_len,
                       return_tensors="pt")

        label_ids = [LABEL2ID.get(l, LABEL2ID["O"]) for l in lab]
        label_ids = [-100] + label_ids[:self.max_len - 2] + [-100]
        pad_len = self.max_len - len(label_ids)
        if pad_len > 0:
            label_ids += [-100] * pad_len
        else:
            label_ids = label_ids[:self.max_len]

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

def collate(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}

def make_loader(X, y, batch_size, max_len, seed, shuffle, tokenizer):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    g = torch.Generator()
    g.manual_seed(seed)
    ds = ProtDataset(X, y, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, worker_init_fn=worker_init_fn, generator=g)


def predict_model(model_path, tokenizer, dataloader, id2label, device, model_name):
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(id2label)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing {os.path.basename(model_path)}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            labels = batch["labels"].cpu().numpy()

            for p, l, pr in zip(preds, labels, probs):
                for pi, li, proba in zip(p, l, pr):
                    if li == -100:
                        continue
                    all_labels.append(li)
                    all_preds.append(pi)
                    all_probs.append(proba)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ---------- 绘图 ----------
from matplotlib.colors import Normalize, LinearSegmentedColormap

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(ID2LABEL.keys()))
    fig, ax = plt.subplots(figsize=(5, 5))

    vmax = cm.max()

    # ---- 🎨 定义颜色映射：200 以下浅，以上快速变深 ----
    class PiecewiseNormalize(Normalize):
        def __call__(self, value, clip=None):
            result, is_scalar = self.process_value(value)
            v = result.data
            x = np.copy(v)
            x = np.clip(x, self.vmin, self.vmax)
            x = np.where(
                x <= 175,
                (x - self.vmin) / 175 * 0.7,  # 前70%颜色范围
                0.7 + (x - 175) / (self.vmax - 175) * 0.3  # 后30%颜色范围
            )
            return np.ma.array(x, mask=result.mask, copy=False)

    norm = PiecewiseNormalize(vmin=0, vmax=vmax)

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=BIO_TAGS, yticklabels=BIO_TAGS,
        cbar=True, norm=norm, square=True  # ✅ 强制正方形
    )

    # 🎯 自定义色条刻度
    cbar = plt.gca().collections[0].colorbar
    ticks = [0, 60, 120, 180, 200, vmax]
    cbar.set_ticks(ticks)
    tick_labels = [str(t) if t != vmax else f"{int(vmax)}" for t in ticks]
    tick_labels[-2] = "200+"
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    # ✅ 保证坐标比例一致（防止标注文字挤压导致非方形）
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    return fig

# ============================================================
# ✅ 计算指标
# ============================================================
def compute_metrics(y_true, y_pred, y_prob):
    results = {}
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    auprc = average_precision_score(pd.get_dummies(y_true), y_prob, average="macro")
    auroc = roc_auc_score(pd.get_dummies(y_true), y_prob, average="macro", multi_class="ovr")

    results.update({
        "ACC": acc, "MCC": mcc,
        "Precision": prec, "Recall": rec, "F1": f1,
        "AUPRC": auprc, "AUROC": auroc
    })

    # Per-class
    per_class = {}
    for i, tag in ID2LABEL.items():
        mask = y_true == i
        prec, rec, f1, _ = precision_recall_fscore_support(mask, (y_pred == i), average='binary', zero_division=0)
        try:
            auprc_c = average_precision_score(mask.astype(int), y_prob[:, i])
            auroc_c = roc_auc_score(mask.astype(int), y_prob[:, i])
        except:
            auprc_c, auroc_c = np.nan, np.nan
        per_class[tag] = {"Precision": prec, "Recall": rec, "F1": f1, "AUPRC": auprc_c, "AUROC": auroc_c}

    return results, per_class


# ============================================================
# ✅ 多类别 ROC + PRC 综合绘图
# ============================================================
def plot_roc_pr_curves_combined(y_true, y_prob, save_dir):
    y_true_bin = pd.get_dummies(y_true).values

    # ROC 图
    plt.figure(figsize=(6, 6))
    for i, tag in ID2LABEL.items():
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_roc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{tag} (AUC={auc_roc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (All Classes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ROC_combined.pdf"))
    plt.close()

    # PRC 图
    plt.figure(figsize=(6, 6))
    for i, tag in ID2LABEL.items():
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        auc_pr = auc(rec, prec)
        plt.plot(rec, prec, lw=2, label=f"{tag} (AUC={auc_pr:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PRC Curves (All Classes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "PRC_combined.pdf"))
    plt.close()

if __name__ == "__main__":

    # 测试集准备
    tokenizer_protbert = AutoTokenizer.from_pretrained("../prot_bert", do_lower_case=False)
    tokenizer_bert = AutoTokenizer.from_pretrained("../bert-base-uncased", do_lower_case=False)
    _, _, _, _, Xtest, ytest = preprocess_and_split("../BIO_labeled_sequences_1.xlsx",
                                                    max_len=512, overlap=64, seed=42)
    test_loader_protbert = make_loader(Xtest, ytest, batch_size=8, max_len=512, seed=42,
                                       shuffle=False, tokenizer=tokenizer_protbert)
    test_loader_bert = make_loader(Xtest, ytest, batch_size=8, max_len=512, seed=42,
                                   shuffle=False, tokenizer=tokenizer_bert)

    # 模型目录
    model_dirs = {
        "bert": ("result_model/bertmodel", "./bert-base-uncased", tokenizer_bert, test_loader_bert),
        "protbert": ("result_model/protbertmodel", "./prot_bert", tokenizer_protbert, test_loader_protbert)
    }

    for model_type, (result_dir, model_name, tokenizer, loader) in model_dirs.items():
        print(f"\n===== Evaluating {model_type.upper()} models =====")
        os.makedirs(os.path.join(result_dir, "eval_loss"), exist_ok=True)
        summary = []

        for file in os.listdir(result_dir):
            if not file.endswith(".pt"):
                continue
            model_path = os.path.join(result_dir, file)
            y_true, y_pred, y_prob = predict_model(model_path, tokenizer, loader, ID2LABEL, DEVICE, model_name)

            overall, per_class = compute_metrics(y_true, y_pred, y_prob)
            row = {"model": file, **overall}
            for tag, metrics in per_class.items():
                for k, v in metrics.items():
                    row[f"{tag}_{k}"] = v
            summary.append(row)

            save_dir = os.path.join(result_dir, "eval_loss", os.path.splitext(file)[0])
            os.makedirs(save_dir, exist_ok=True)
            plot_confusion_matrix(y_true, y_pred, os.path.join(save_dir, "confusion_matrix.pdf"))
            plot_roc_pr_curves_combined(y_true, y_prob, save_dir)

        pd.DataFrame(summary).to_csv(os.path.join(result_dir, f"{model_type}_evaluation_summary.csv"), index=False)
        print(f"✅ {model_type} evaluation done → {model_type}_evaluation_summary.csv")


