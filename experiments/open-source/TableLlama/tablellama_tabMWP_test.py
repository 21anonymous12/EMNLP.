import os
import json
import sys
import math
import torch
import argparse
# import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig
from tqdm import tqdm
import pandas as pd
from dataloader.wikiTQ_loader import WikiTQ
from dataloader.tabMWP_loader import TabMWP
from dataloader.Tablebench_loader import Tablebench_Filtered
from dataloader.tabfact_loader import TabFact
from tqdm import tqdm

# 데이터프레임을 포맷으로 변환하는 함수
def dataframe_to_custom_format(df):
    # 열 이름 동적으로 추가
    header = "[TAB] col: " + "|".join(df.columns) + "|"
    # 각 행 데이터를 [SEP]로 추가
    rows = "".join([f"[SEP]|{'|'.join(map(str, row))}" for row in df.values])
    return header + rows


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input_seg}\n\n### Question:\n{question}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def generate_prompt(instruction, question, input_seg=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

def build_generator(
    item, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(item):
    # def response(material, question, material_type="", material_title=None):
        # material = read_txt_file(material)
        # prompt = format_prompt(material, question, material_type, material_title)
        prompt = generate_prompt(instruction = item["instruction"], input_seg = item["input_seg"], question = item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache
        )
        out = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        out = out.split(prompt)[1].strip()
        return out

    return response

def main(args, test_data):
    # if args.flash_attn:
    #     replace_llama_attn()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        # padding_side="right",
        padding_side="left",
        use_fast=False,
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # with open(args.input_data_file, "r") as f:
    #     test_data = json.load(f)
    
    # import random
    # test_data = random.sample(test_data, k=3)

    test_data_pred = []
    for i in tqdm(range(len(test_data))):
        item = test_data[i]
        new_item = {}
        respond = build_generator(item, model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=not args.flash_attn)   # the temperature and top_p are highly different with previous alpaca exp, pay attention to this if there is sth wrong later
        output = respond(item)

        new_item["idx"] = i
        # new_item["table_id"] = test_data[i]["table_id"]
        new_item["instruction"] = test_data[i]["instruction"]
        new_item["input_seg"] = test_data[i]["input_seg"]
        new_item["question"] = test_data[i]["question"]
        # new_item["ground_truth"] = test_data[i]["ground_truth"]
        new_item["output"] = test_data[i]["output"]
        new_item["predict"] = output

        test_data_pred.append(new_item)
        # import pdb
        # pdb.set_trace() 
        
    return test_data_pred
        
    # with open(args.output_data_file, "w") as f:
    #     json.dump(test_data_pred, f, indent = 2)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="osunlp/TableLlama")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=1024, help='')
    parser.add_argument('--input_data_file', type=str, default='input_data/', help='')
    parser.add_argument('--output_data_file', type=str, default='output_data/', help='')
    args = parser.parse_args()
    return args

# 데이터셋 로드
tabMWP_df = pd.read_json('problems_test.json')


data_list = []
for index in tqdm(range(7686), desc="Processing dataset", unit="sample"):
    tab_mwp = TabMWP(tabMWP_df, index)
    table = tab_mwp.get_table()
    question = tab_mwp.get_question()
    serial_table = dataframe_to_custom_format(table)
    answers = tab_mwp.get_answer()
    infer_data = {
    'instruction': 'This is a table QA task. The goal of this task is to answer the question given the table',
    'input_seg': serial_table,
    'question': question,
    'output': answers
    }
    data_list.append(infer_data)

print(f"================tabMWP data length : {len(data_list)}=======================")



args = parse_config()
result = main(args, data_list)

# JSON 파일로 저장
with open('tabMWP_tablellama_json_test_final.json', 'w') as file:
    json.dump(result, file, indent=4)


