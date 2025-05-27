import json
from collections import Counter
import pandas as pd
import numpy as np
import re


def count_duplicate_words_in_headers(question: str, table_headers: list) -> int:
    """
    question: 사용자 질문 문자열
    table_headers: 테이블 헤더가 저장된 리스트
    return: 질문과 테이블 헤더 간의 중복 단어 수
    """
    # 1. 질문 문자열에서 불필요한 문장부호 제거 및 소문자 변환
    question_cleaned = re.sub(r'[^\w\s]', '', question).lower()
    # 2. 단어 단위로 분리
    question_words = question_cleaned.split()
    
    # 3. 테이블 헤더도 소문자로 변환하고, 공백이 있을 경우를 대비해 단어 단위로 분리
    #    예: 테이블 헤더가 "product name"이라면 ["product", "name"]로 만들어줌
    #    만약 이미 ["product", "sales"]와 같은 리스트 형태라면 하단 처리를 꼭 할 필요는 없음
    header_words = []
    for header in table_headers:
        # header가 "product name"처럼 공백 포함 문자열일 경우 분리
        # re.sub로 특수문자 제거 등 필요 시 추가 가능
        header_cleaned = re.sub(r'[^\w\s]', '', header).lower()
        header_words.extend(header_cleaned.split())
    
    # 4. set 자료구조를 사용하여 중복 단어를 효율적으로 확인
    question_word_set = set(question_words)
    header_word_set = set(header_words)
    
    # 5. 교집합(공통으로 등장하는 단어)을 구하고, 그 개수를 반환
    common_words = question_word_set.intersection(header_word_set)
    return len(common_words)

def convert_to_numeric(table):
    """
    데이터프레임에서 문자열로 저장된 숫자 값만 숫자 형식으로 변환하고,
    숫자로 변환할 수 없는 값은 원래 텍스트를 그대로 유지합니다.
    
    Parameters:
    table (pd.DataFrame): 변환할 데이터프레임

    Returns:
    pd.DataFrame: 변환된 데이터프레임
    """

    # 열 이름을 일시적으로 고유하게 변경
    original_columns = table.columns
    table.columns = [f"{col}_{i}" if list(table.columns).count(col) > 1 else col 
                     for i, col in enumerate(table.columns)]
    
    # object 타입의 열만 숫자로 변환
    for col in table.select_dtypes(include='object').columns:
        numeric_col = pd.to_numeric(table[col], errors='coerce')
        # 결합 동작을 명시적으로 처리
        table[col] = numeric_col.where(numeric_col.notna(), table[col])  # 숫자로 변환 불가능한 값은 원래 텍스트 유지
    
    # 원래 열 이름으로 복원
    table.columns = original_columns
    return table


def guess_column_type(dtype) -> str:
    """
    Pandas dtype을 보고 SQL에서 자주 쓰이는 자료형으로 매핑해주는 함수 예시입니다.
    상황과 DBMS에 따라 원하는 자료형을 추가/조정하세요.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DECIMAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    # 날짜/시계열 타입 등 추가적으로 처리하려면 여기에 elif 추가
    else:
        # 문자열(object) 등은 TEXT로 처리
        return "TEXT"


def extract_table_info(table, question):
    """
    테이블과 질문이 입력되었을 때, 테이블의 정보를 추출해주는 함수
    <추출되는 것>
    1. 테이블 row 개수
    2. 테이블 column 개수
    3. 1 X 2 결과
    4. 테이블 내의 타입의 종류 => int, float, bool, text, NaN
    
    5. 질문 내의 단어 개수
    6. 질문 내의 숫자 개수
    7. 질문의 string 길이
    
    8. 질문과 테이블 헤더 간의 중복 단어 수
    """  
    table_rows , table_columns  = table.shape
    table_size = table_rows * table_columns
    changed_table = convert_to_numeric(table)
    
    col_types = []
    for col in changed_table.columns:
        col_type = guess_column_type(changed_table[col].dtypes)
        col_types.append(col_type)
    
    int_check = False
    float_check = False
    bool_check = False
    text_check = False
    NaN_check = table.isna().any().any()
    if "INTEGER" in col_types:
        int_check = True
    
    if "DECIMAL" in col_types:
        float_check = True
    
    if "BOOLEAN" in col_types:
        bool_check = True

    if "TEXT" in col_types:
        text_check = True
    
    # 1) 단어 중복을 제거한(unique) 단어 개수
    #    - 알파벳(a~z, A~Z)로 이루어진 토큰만 추출
    #    - 대소문자는 구분하지 않는다고 가정하여 lower()로 통일
    words = re.findall(r"[A-Za-z0-9]+", question)
    unique_words = set(word.lower() for word in words)
    unique_word_count = len(unique_words)
    
    # 2) 숫자(중복 포함) 개수
    #    - 숫자(\d+)를 찾아 모두 세어줌
    numbers_in_text = re.findall(r"\d+", question)
    numbers_count = len(numbers_in_text)
    
    
    
    duplicate_count = count_duplicate_words_in_headers(question, table.columns.tolist())


    return {
        "table_row" : table_rows,
        "table_column" : table_columns,
        "table_size" : table_size,
        "table_int_check" : int_check,
        "table_float_check" : float_check,
        "table_bool_check" : bool_check,
        "table_text_check" : text_check,
        "table_NaN_check" : NaN_check,
        
        "question_unique_word_count" : unique_word_count,
        "question_numbers_count" : numbers_count,
        
        "table_question_duplicate_count" : duplicate_count
        
    }