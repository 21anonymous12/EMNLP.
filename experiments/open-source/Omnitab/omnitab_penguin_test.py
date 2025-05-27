from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datasets import load_dataset
import time
import re
import os
import json
import requests
import torch
import numpy as np
from dataloader.penguin_loader import Penguins_in_a_table
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large')
model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large')

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터셋 로드
with open("./penguins_in_a_table_dataframe_answer_extracted.json", "r", encoding="utf-8") as file:
    dataset = json.load(file) 

answers_list = []
pred_list = []
for index in tqdm(range(144), desc="Processing dataset", unit="sample"):
# 객체 생성
    penguins = Penguins_in_a_table(dataset, index)
    penguins_table = penguins.get_table()
    penguins_question = penguins.get_question()
    penguins_answer = penguins.get_answer_extract()

    encoding = tokenizer(table=penguins_table, query=penguins_question,   truncation=True, return_tensors="pt").to(device)
    outputs = model.generate(**encoding)
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers_list.append(penguins_answer)
    pred_list.append(pred)

result_df = pd.DataFrame({'answers': answers_list, 'pred': pred_list})
result_df.to_csv(f'penguins_omnitab_test_final.csv', index=False)

