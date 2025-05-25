from transformers import AutoTokenizer
from langchain_ollama import ChatOllama

def measure_and_adjust_context(prompt, model_name="microsoft/phi-4", max_context=16000):
    
    # Qwen 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # 프롬프트를 토큰화
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)
    
    # context window 동적 조정
    adjusted_context = min(token_count, max_context) + 2048
    
    return {
        "token_count": token_count,
        "adjusted_context": adjusted_context,
        "truncated": adjusted_context > max_context
    }


def llm_adjusted_context(llm, ctx_window):
    llm = ChatOllama(
    model="phi4:14b", 
    format="json",  # 입출력 형식을 JSON으로 설정합니다.
    temperature=0,
    num_ctx = ctx_window,  # 컨텍스트 크기 증가
    )
    return llm