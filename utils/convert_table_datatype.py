import pandas as pd
import numpy as np

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


def guess_sql_type(dtype) -> str:
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

def df_to_table_prompt(df: pd.DataFrame, table_name: str = "dataframe") -> str:
    """
    주어진 df(DataFrame)를 -- Table: ... 형태의 문자열로 변환합니다.
    """
    # 테이블 명 줄 만들기
    lines = [f"-- Table: {table_name}"]
    lines.append("-- Columns:")
    
    # 각 컬럼에 대한 자료형 매핑
    for col in df.columns:
        col_type = guess_sql_type(df[col].dtypes)
        lines.append(f"--   {col} ({col_type})")

    # 행(Row) 정보 구성
    lines.append("--")
    lines.append("-- Rows:")
    # 각 행을 출력 형식에 맞게 구성
    for idx, row in df.iterrows():
        # 예: "  Braden | 76"
        row_values = [str(val) for val in row.values]
        lines.append(f"--   {' | '.join(row_values)}")

    # 최종 문자열
    table_prompt = "\n".join(lines)
    return table_prompt


def df_to_table_prompt_rowX(df: pd.DataFrame, table_name: str = "dataframe") -> str:
    """
    주어진 df(DataFrame)를 -- Table: ... 형태의 문자열로 변환합니다.
    """
    # 테이블 명 줄 만들기
    lines = [f"-- Table: {table_name}"]
    lines.append("-- Columns:")
    
    # 각 컬럼에 대한 자료형 매핑
    for col in df.columns:
        col_type = guess_sql_type(df[col].dtypes)
        lines.append(f"--   {col} ({col_type})")

    # 최종 문자열
    table_prompt = "\n".join(lines)
    return table_prompt
