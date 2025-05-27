from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datasets import load_dataset
import time
import re
import os
import json
import requests
import torch
from dataloader.wikiTQ_loader import WikiTQ
from dataloader.tabMWP_loader import TabMWP
from dataloader.Tablebench_loader import Tablebench_Filtered
from dataloader.tabfact_loader import TabFact
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("Yale-LILY/reastap-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Yale-LILY/reastap-large")

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터셋 로드
dataset = load_dataset(path="Multilingual-Multimodal-NLP/TableBench", data_files="TableBench.jsonl", split="train")

answers_list = []
pred_list = []
for index in tqdm(range(693), desc="Processing dataset", unit="sample"):
# 객체 생성
    Tablebench = Tablebench_Filtered(dataset, index)
    Tablebench_table = Tablebench.get_table()
    Tablebench_question = Tablebench.get_question()
    Tablebench_answer = Tablebench.get_answer()

    try:
        encoding = tokenizer(table=Tablebench_table, query=Tablebench_question,   truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(**encoding)
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except AttributeError as e:
        pred = [""]
        print(f"Skipping sample {index} due to error: {e}")
    answers_list.append(Tablebench_answer)
    pred_list.append(pred)

result_df = pd.DataFrame({'answers': answers_list, 'pred': pred_list})
result_df.to_csv(f'Tablebench_reastap_test_final.csv', index=False)

