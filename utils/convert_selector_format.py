import json
import copy
import pandas as pd
import ast
from tqdm import tqdm


def cut_dataframe_if_too_large(df, window_number):
    max_size = 20
    rows, cols = df.shape
    total_size = rows * cols

    if total_size < max_size:
        return df.reset_index(drop=True)

    max_rows = max_size // cols

    # 윈도우 시작 인덱스 계산
    start = (window_number * max_rows) % rows

    # 끝 인덱스가 rows를 넘어가면 자투리만 가져옴
    end = min(start + max_rows, rows)

    return df.iloc[start:end].reset_index(drop=True)



def transform_json_to_special_tokens(table, question, json_data, window_number=0):
    output = ""
    # <Table_row_size> 처리
    output += "<Table_row_size>" + str(table.shape[0]) + "</Table_row_size>\n"

    # <Table_column_size> 처리
    output += "<Table_column_size>" + str(table.shape[1]) + "</Table_column_size>\n"    

    # <Table_size> 처리
    output += "<Table_size>" + str(table.shape[0]*table.shape[1]) + "</Table_size>\n"        
    

    # <Table> 처리
    table_cut = cut_dataframe_if_too_large(table, window_number)
    output += "<Table>\n" + table_cut.to_markdown(index=False) + "\n</Table>\n"  
    
    # <Question> 처리
    output += "<Question>" + question + "</Question>\n"

    # <PoT> 처리
    pot = json_data.get("PoT", {})
    output += "<PoT>\n"
    # PoT의 키는 "N=0", "N=0 execution_result", "N=1", ... 와 같이 구성되어 있음
    indices = []
    for key in pot.keys():
        if key.startswith("N=") and "execution_result" not in key:
            # 예: "N=0"에서 "0" 추출
            idx = key.split("N=")[1]
            indices.append(idx)
    indices = sorted(set(indices), key=int)
    for idx in indices:
        code_key = f"N={idx}"
        exec_key = f"N={idx} execution_result"
        code_entry = pot.get(code_key, {})
        if isinstance(code_entry, dict):
            code = code_entry.get("code", "")
        else:
            code = str(code_entry)
        exec_result = pot.get(exec_key, "")
        output += f"<N={idx}_code>{code}</N={idx}_code>\n"
        output += f"<N={idx}_execution_result>{exec_result}</N={idx}_execution_result>\n"
    output += "</PoT>\n"

    # <text2sql> 처리
    text2sql = json_data.get("text2sql", {})
    output += "<text2sql>\n"
    indices = []
    for key in text2sql.keys():
        if key.startswith("N=") and "execution_result" not in key:
            idx = key.split("N=")[1]
            indices.append(idx)
    indices = sorted(set(indices), key=int)
    for idx in indices:
        code_key = f"N={idx}"
        exec_key = f"N={idx} execution_result"
        code_entry = text2sql.get(code_key, {})
        if isinstance(code_entry, dict):
            code = code_entry.get("code", "")
        else:
            code = str(code_entry)
        exec_result = text2sql.get(exec_key, "")
        output += f"<N={idx}_code>{code}</N={idx}_code>\n"
        output += f"<N={idx}_execution_result>{exec_result}</N={idx}_execution_result>\n"
    output += "</text2sql>\n"

    # <CoT> 처리
    cot = json_data.get("CoT", {})
    output += "<CoT>\n"
    # CoT는 "N=0" 키 아래에 solution과 answer가 있다고 가정
    cot0 = cot.get("N=0", {})
    solution = cot0.get("solution", "")
    answer = cot0.get("answer", "")
    output += f"<solution>{solution}</solution>\n"
    output += f"<answer>{answer}</answer>\n"
    output += "</CoT>\n"

    return output