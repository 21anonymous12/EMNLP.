
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, MobileBertModel
from typing import List, Dict


# ───────────────────────────────────────────────────────────
# 1. 모델
# ───────────────────────────────────────────────────────────
class MobileBertWithFeatures(nn.Module):
    """
    MobileBERT + 수치/불리언 특성 → 다중라벨(여기선 2개) 로짓 반환.
    """
    def __init__(self, num_extra: int, dropout: float = 0.1):
        super().__init__()
        self.bert = MobileBertModel.from_pretrained("google/mobilebert-uncased")
        hidden = self.bert.config.hidden_size          # 512
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + num_extra, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)                          # 로짓 2개
        )

    def forward(self, input_ids, attention_mask, extra_feats):
        cls = self.bert(input_ids, attention_mask).last_hidden_state[:, 0]  # [B, 512]
        x = torch.cat([cls, extra_feats], dim=1)                            # [B, 512+E]
        logits = self.head(x)                                               # [B, 2]
        return logits              # BCEWithLogitsLoss에 바로 사용


# ───────────────────────────────────────────────────────────
# 2. 데이터셋
# ───────────────────────────────────────────────────────────
class SentFeatDataset(Dataset):
    """
    samples: List[Dict]  구조 예시는 아래 main() 참고.
    각 샘플에 'label_vec' → [0/1, 0/1]  (멀티라벨)
    """
    def __init__(
        self,
        samples: List[Dict],
        tokenizer: AutoTokenizer,
        max_len: int = 500
    ):
        self.samples = samples
        self.tk = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tk(
            s["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        # 수치/불리언 결합 → float32
        extra = torch.tensor(
            s["num_feats"] + [float(b) for b in s["bool_feats"]],
            dtype=torch.float32
        )
        label = torch.tensor(s["label_vec"], dtype=torch.float32)  # 다중라벨
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "extra": extra,
            "label": label
        }


