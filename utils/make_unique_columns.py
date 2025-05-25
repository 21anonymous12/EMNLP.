import pandas as pd
import numpy as np

def make_unique_columns(df):
    """
    중복된 컬럼 이름에 공백과 숫자를 붙여 고유하게 만드는 함수.
    예: ['A', 'B', 'A'] → ['A', 'B', 'A 1']
    """
    from collections import defaultdict

    col_count = defaultdict(int)
    new_columns = []

    for col in df.columns:
        count = col_count[col]
        if count == 0:
            new_columns.append(col)
        else:
            new_columns.append(f"{col} {count}")
        col_count[col] += 1

    df.columns = new_columns
    return df
