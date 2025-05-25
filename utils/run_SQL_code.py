import pandas as pd
import sqlite3


# SQLite 데이터베이스에 Pandas 데이터프레임 로드
connection = sqlite3.connect(":memory:")  # 메모리 상의 SQLite DB

# SQL 쿼리를 실행하고 결과를 반환하는 함수
def execute_single_query(df, query, table_name: str = "dataframe"):
    df.to_sql(table_name, connection, index=False, if_exists="replace")  # 데이터프레임을 "data"라는 테이블로 저장
    try:
        # SQL 실행
        result = pd.read_sql_query(query, connection)
        # 결과를 셀 데이터만 추출해서 리스트로 변환
        return result.values.tolist()
    except Exception as e:
        # 오류 발생 시 반환
        return f"Error: {e}"


def replace_spaces_with_underscores(df):
    """
    Replace spaces in column names of a DataFrame with underscores.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with updated column names.
    """
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.replace(' ', '_', regex=False)
    return df_copy