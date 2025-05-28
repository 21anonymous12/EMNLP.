from Tablebench_loader import Tablebench_Filtered
from datasets import load_dataset
import pandas as pd

# Load TableBench
dataset = load_dataset("Multilingual-Multimodal-NLP/TableBench", trust_remote_code=True)
tablebench = Tablebench_Filtered(dataset, index=0)  # Start with index 0

def convert_to_chameleon_format(tablebench_instance):
    # Convert DataFrame to string representation (TabMWP style)
    table_df = tablebench_instance.get_table()
    table_str = table_df.to_string(index=False).replace('\n ', '\n')  # Remove extra spaces
    return {
        "pid": tablebench_instance.index,
        "question": tablebench_instance.get_question(),
        "answer": tablebench_instance.get_answer(),
        "table": table_str,
        "unit": None,  # TableBench doesn't specify units; adjust if needed
        "choices": None,  # Add if qtype indicates multiple-choice
        "ans_type": "extractive_text" if isinstance(tablebench_instance.get_answer(), str) else "integer_number",
        "grade": None,  # Optional metadata
        "ques_type": "multi_choice" if tablebench_instance.get_qsubtype() == "Comparison" else "free_text"
    }

# Test conversion
sample = convert_to_chameleon_format(tablebench)
print(sample)