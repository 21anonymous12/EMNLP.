import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
import pandas as pd
import torch
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from utils.create_query_format import *
from langchain_core.prompts import load_prompt
from utils.run_pandas_code import *
from utils.LLM_inference import *
from utils.selector_inference import *
from utils.convert_table_datatype import *
from utils.run_SQL_code import *
from utils.make_unique_columns import *
from utils.convert_selector_format import *
from utils.long_text_check import *
from utils.choose_ans_based_on_selector import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#from huggingface_hub import login
#login(token="~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

results_all = {}
llm = ChatOllama(
model="phi4:14b", 
format="json",  # 입출력 형식을 JSON으로 설정합니다.
temperature=0,
num_ctx = 2048,  # 컨텍스트 크기 증가
)


table_raw = pd.DataFrame(
    {
    "Day": ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "Boxes of cookies": [25, 27, 23, 26, 23]
    }
) # The input Table must have the format of Pandas DataFrame for MIAS.

question = 'A Girl Scout troop recorded how many boxes of cookies they sold each day for a week. According to the table, what was the rate of change between Wednesday and Thursday?'

table = make_unique_columns(table_raw) # If column names are duplicated, this function add additional numbers at columns. (ex. [Day, Month, Day, Month] => [Day 1, Month 1, Day 2, Month 2]) 

PoT_results = {}
text2sql_results = {}
CoT_results = {}

prompt_PoT = load_prompt("prompt/PoT_prompt.yaml", encoding="utf-8")
prompt_text2sql = load_prompt("prompt/text2sql_prompt.yaml", encoding="utf-8")
prompt_CoT = load_prompt("prompt/CoT_prompt.yaml", encoding="utf-8")
prompt_refine_PoT = load_prompt("prompt/PoT_self-refine.yaml", encoding="utf-8")
prompt_refine_text2sql = load_prompt("prompt/text2sql_self-refine.yaml", encoding="utf-8")


PoT_results, text2sql_results, CoT_results = LLM_inference(prompt_PoT, prompt_text2sql, prompt_CoT, prompt_refine_PoT, prompt_refine_text2sql, llm, table, question, PoT_results, text2sql_results, CoT_results, N = 3, sim_threshold = 0.9)


results_all = {
    'PoT': PoT_results,
    'text2sql' : text2sql_results,
    'CoT': CoT_results
}
# Grond Truth : -4

with open('LLM_inference_results.json', 'w', encoding='utf-8') as json_file:
    json.dump(results_all, json_file, ensure_ascii=False, indent=4)

model_path = "1anonymous1/MIAS_selector"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if results_all.get("inference") == "inference error":
    test_data = "inference error"
test_data = transform_json_to_special_tokens(table, question, results_all, window_number = 0)

if test_data == "inference error":
    print("An error occurred during the LLM inference process.")
long_checked_text_data = long_text_check(test_data, tokenizer, check_PoT_error=check_PoT_error, check_text2sql_error=check_text2sql_error)

selector_result = predict_label(long_checked_text_data, tokenizer, model)

selected_answer, is_hallucination = choose_ans_based_on_selector(selector_result, results_all, hallucination_threshold = 0.1)
print("================================FINAL ANSWER================================")
print(f"Answer : {selected_answer}")
print(f"Not hallucination : {is_hallucination}") 