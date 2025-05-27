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
dataset = load_dataset('wenhu/tab_fact', 'tab_fact', trust_remote_code=True)

answers_list = []
pred_list = []
for index in tqdm(range(12779), desc="Processing dataset", unit="sample"):
# 객체 생성
    tabfact = TabFact(dataset['test'], index)
    tabfact_table = tabfact.get_table()
    tabfact_question = tabfact.get_statement()
    tabfact_label = tabfact.get_label()
    tabfact_caption = tabfact.get_caption()
    if tabfact_label == 1:
        tabfact_answer = 'True'
    elif tabfact_label == 0:
        tabfact_answer = 'False' 
    tabfact_question_refine = f'It is about {tabfact_caption}. Determine whether the statement is True or False : {tabfact_question}'

    encoding = tokenizer(table=tabfact_table, query=tabfact_question_refine,   truncation=True, return_tensors="pt").to(device)
    outputs = model.generate(**encoding)
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers_list.append(tabfact_answer)
    pred_list.append(pred)


result_df = pd.DataFrame({'answers': answers_list, 'pred': pred_list})
result_df.to_csv(f'tabfact_reastap_test_final.csv', index=False)

