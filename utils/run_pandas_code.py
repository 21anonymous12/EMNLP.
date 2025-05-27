import pandas as pd

def run_code(code: str):
    """
    문자열로 된 파이썬 코드를 입력받아, 실행 후 마지막에 정의된 ans 변수를 반환하는 함수.
    code는 반드시 ans = ~~~ 형태의 구문으로 끝난다고 가정합니다.
    """
    # exec에서 사용할 로컬/글로벌 컨텍스트를 딕셔너리로 생성
    local_vars = {}
    
    try:
        exec(code, {}, local_vars)
        return local_vars['ans']  # 실행 완료 후 ans 값 반환
    except Exception as e:
        # 코드 실행 중 예외가 발생하면 예외 메시지를 문자열로 반환
        return str(e)


def generate_pandas_code(original_code: str, df_code: str) -> str:
    """
    사용자 코드(original_code)에
    1) import pandas as pd
    2) df_code (DataFrame 생성 등 사전 세팅 코드)
    를 맨 앞에 추가하여 최종 코드를 문자열로 만들어 반환.
    """
    # 1) pandas import
    new_code = "import pandas as pd\n\nimport numpy as np\n\n"

    # 2) df_code (DataFrame 준비 코드 등) 추가
    # df_code 자체는 여러 줄일 수 있으므로 그대로 이어붙임
    new_code += df_code.strip() + "\n\n"

    # 3) 마지막으로 사용자 코드를 이어붙임
    # original_code도 여러 줄일 수 있으니 그대로 이어붙임
    new_code += original_code.strip() + "\n"

    return new_code

def run_pandas_code(original_code: str, df_code: str):
    """
    original_code를 실행하기 전에
    1) import pandas as pd
    2) df_code
    를 위에 추가하고,
    마지막에 ans 변수를 반환하는 예시 함수
    """
    # 1) 판다스 준비 코드 자동 생성
    final_code = generate_pandas_code(original_code, df_code)
    
    # 2) ans 반환용 run_code 호출
    result = run_code(final_code.replace("'''", ""))
    return result