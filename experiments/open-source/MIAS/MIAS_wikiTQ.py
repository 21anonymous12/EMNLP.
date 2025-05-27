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
from utils.LLM_inference_with_scheduler import *
from utils.LLM_inference_without_scheduler import *
from utils.selector_inference import *
from utils.convert_table_datatype import *
from utils.run_SQL_code import *
from utils.make_unique_columns import *
from utils.convert_selector_format import *
from utils.long_text_check import *
from utils.choose_ans_based_on_selector import *
from utils.extract_info_from_T_and_Q import *
from scheduler.scheduler import MobileBertWithFeatures
import argparse
import yaml
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#from huggingface_hub import login
#login(token="~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
parser.add_argument('--Use_Scheduler', type=lambda x: x.lower() == 'true', help='Override Use_Scheduler')
parser.add_argument('--N', type=int, help='Override N')
parser.add_argument('--hallucination_threshold', type=float, help='Override hallucination_threshold')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
if args.Use_Scheduler is not None:
    config['Use_Scheduler'] = args.Use_Scheduler
if args.N is not None:
    config['N'] = args.N
if args.hallucination_threshold is not None:
    config['hallucination_threshold'] = args.hallucination_threshold

Use_Scheduler = config['Use_Scheduler']
N = config['N']
hallucination_threshold = config['hallucination_threshold']



results_all = {}
llm = ChatOllama(
model="phi4:14b", 
format="json",  
temperature=0,
num_ctx = 2048,  
)

from dataloader.wikiTQ_loader import WikiTQ
dataset = load_dataset("Stanford/wikitablequestions", trust_remote_code=True)
index = 7
wikiTQ = WikiTQ(dataset, index, type = 'test') # The default value of `type` is `test`, but it can be changed to `train` or `validation`.
table_raw = wikiTQ.get_table()
question = wikiTQ.get_question()

table = make_unique_columns(table_raw) # If column names are duplicated, this function add additional numbers at columns. (ex. [Day, Month, Day, Month] => [Day 1, Month 1, Day 2, Month 2]) 

PoT_results = {}
text2sql_results = {}
CoT_results = {}

prompt_PoT = load_prompt("prompt/PoT_prompt.yaml", encoding="utf-8")
prompt_text2sql = load_prompt("prompt/text2sql_prompt.yaml", encoding="utf-8")
prompt_CoT = load_prompt("prompt/CoT_prompt.yaml", encoding="utf-8")
prompt_refine_PoT = load_prompt("prompt/PoT_self-refine.yaml", encoding="utf-8")
prompt_refine_text2sql = load_prompt("prompt/text2sql_self-refine.yaml", encoding="utf-8")

if Use_Scheduler == True :
    num_feats_cols = ['table_row', 'table_column', 'table_size', 'question_unique_word_count', 'question_numbers_count', 'table_question_duplicate_count']
    bool_feats_cols = ['table_int_check', 'table_float_check', 'table_text_check', 'table_NaN_check']
    num_extra = len(num_feats_cols) + len(bool_feats_cols)
    info = extract_table_info(table, question)
    scheduler_tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
    scheduler = MobileBertWithFeatures(num_extra).to(device)
    scheduler.load_state_dict(torch.load("./scheduler/mobilebert_multilabel_45.pt", map_location=device))
    scheduler.eval()
    sample = {}
    sample["sentence"] = question + " | " + str(table.columns.tolist())
    sample["num_feats"] = [info[key] for key in num_feats_cols]
    sample["bool_feats"] = [info[key] for key in bool_feats_cols]

    enc = scheduler_tokenizer(
        sample["sentence"],
        padding="max_length",
        truncation=True,
        max_length=500,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    extra_feats = torch.tensor(
        sample["num_feats"] + [float(b) for b in sample["bool_feats"]],
        dtype=torch.float32
    ).unsqueeze(0).to(device) 
    with torch.no_grad():
        logits = scheduler(input_ids, attention_mask, extra_feats)  # [1, 2]
        probs = torch.sigmoid(logits).cpu().numpy() 
    if probs[0][0] >= probs[0][1]:
        scheduler_result = 'PoT'    
    elif probs[0][0] < probs[0][1]:
        scheduler_result = 'text2sql'  
    PoT_results, text2sql_results, CoT_results = LLM_inference_with_scheduler(prompt_PoT, prompt_text2sql, prompt_CoT, prompt_refine_PoT, prompt_refine_text2sql, llm, table, question, PoT_results, text2sql_results, CoT_results, scheduler_result , N = N)
elif Use_Scheduler == False :
    PoT_results, text2sql_results, CoT_results = LLM_inference_without_scheduler(prompt_PoT, prompt_text2sql, prompt_CoT, prompt_refine_PoT, prompt_refine_text2sql, llm, table, question, PoT_results, text2sql_results, CoT_results, N = N)

results_all = {
    'PoT': PoT_results,
    'text2sql' : text2sql_results,
    'CoT': CoT_results
}

with open(f'LLM_inference_results_wikiTQ_index_{index}.json', 'w', encoding='utf-8') as json_file:
    json.dump(results_all, json_file, ensure_ascii=False, indent=4)

selector_path = "7anonymous7/MIAS_selector"
selector_tokenizer = AutoTokenizer.from_pretrained(selector_path)
selector = AutoModelForSequenceClassification.from_pretrained(selector_path)
selector.eval()

selector.to(device)

if results_all.get("inference") == "inference error":
    test_data = "inference error"
test_data = transform_json_to_special_tokens(table, question, results_all, window_number = 0)

if test_data == "inference error":
    print("An error occurred during the LLM inference process.")
long_checked_text_data = long_text_check(test_data, selector_tokenizer, check_PoT_error=check_PoT_error, check_text2sql_error=check_text2sql_error)

selector_result = predict_label(long_checked_text_data, selector_tokenizer, selector)

selected_answer, is_hallucination = choose_ans_based_on_selector(selector_result, results_all, hallucination_threshold = hallucination_threshold)
print("================================FINAL ANSWER================================")
print(f"Answer : {selected_answer}")
print(f"Not hallucination : {is_hallucination}") 
print(f"Ground Truth : {wikiTQ.get_answers()}") 