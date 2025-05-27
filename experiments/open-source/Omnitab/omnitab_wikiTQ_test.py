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

tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large')
model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large')

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터셋 로드
dataset = load_dataset("Stanford/wikitablequestions")

answers_list = []
pred_list = []
for index in tqdm(range(len(dataset['test'])), desc="Processing dataset", unit="sample"):
# 객체 생성
    wiki_tq = WikiTQ(dataset, index, type = 'test') # type의 default는 test이고 train이나 validation으로 변경 가능
    wikiTQ_table = wiki_tq.get_table()
    wikiTQ_question = wiki_tq.get_question()
    wikiTQ_answer = wiki_tq.get_answers()

    encoding = tokenizer(table=wikiTQ_table, query=wikiTQ_question,   truncation=True, return_tensors="pt").to(device)
    outputs = model.generate(**encoding)
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers_list.append(wikiTQ_answer)
    pred_list.append(pred)

result_df = pd.DataFrame({'answers': answers_list, 'pred': pred_list})
result_df.to_csv(f'wikiTQ_omnitab_test_final.csv', index=False)

