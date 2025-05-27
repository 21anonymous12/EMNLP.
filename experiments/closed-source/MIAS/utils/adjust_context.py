import tiktoken
from langchain_openai import ChatOpenAI

def measure_and_adjust_context(prompt, max_context=125800):
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)
    
    adjusted_context = min(token_count, max_context) + 2048
    
    return {
        "token_count": token_count,
        "adjusted_context": adjusted_context,
        "truncated": adjusted_context > max_context
    }


def llm_adjusted_context(llm, max_token):
    llm = ChatOpenAI(
        temperature=0,  
        model_name="gpt-4o",  
        max_tokens=max_token,
        timeout=300
    )
    return llm