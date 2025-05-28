from datasets import load_dataset
from Tablebench_loader import Tablebench_Filtered
import json
import pandas as pd
from tqdm import tqdm

# Load the TableBench dataset
dataset = load_dataset("Multilingual-Multimodal-NLP/TableBench", trust_remote_code=True)
tablebench = Tablebench_Filtered(dataset, 0)  # Initialize with dummy index

def tablebench_to_tabmwp_format(tablebench_instance):
    # Convert table to clean string with | delimiter
    table_df = tablebench_instance.get_table()
    
    # Replace NaN or empty values with "–" (en dash) for consistency with TabMWP
    table_df = table_df.fillna("–")
    
    # Get column names and join with " | "
    header = " | ".join(table_df.columns)
    
    # Convert each row to a string with " | " separator
    rows = []
    for _, row in table_df.iterrows():
        row_str = " | ".join(str(value) for value in row)
        rows.append(row_str)
    
    # Combine header and rows with newlines
    table_str = header + "\n" + "\n".join(rows)
    
    # Determine question and answer types
    answer = tablebench_instance.get_answer()
    qsubtype = tablebench_instance.get_qsubtype()
    
    # Infer ques_type and ans_type
    ques_type = "multi_choice" if qsubtype in {"Comparison", "Ranking"} else "free_text"
    if isinstance(answer, str):
        try:
            float(answer)
            ans_type = "decimal_number" if '.' in answer else "integer_number"
        except ValueError:
            ans_type = "extractive_text"
    else:
        ans_type = "integer_number" if isinstance(answer, int) else "decimal_number"
    
    # Construct dictionary
    return {
        "pid": str(tablebench_instance.index),
        "question": tablebench_instance.get_question(),
        "answer": str(answer),  # Ensure string for consistency
        "table": table_str,
        "unit": None,  # TableBench doesn’t provide units
        "choices": None,  # No explicit choices in TableBench; adjust if needed
        "ans_type": ans_type,
        "ques_type": ques_type,
        "grade": None,  # Not available in TableBench
        "split": "test"  # Mimic TabMWP
    }

# Get the length of the filtered dataset
filtered_length = len(tablebench.dataset)
print(f"Filtered dataset length: {filtered_length}")  # Should be 1779

# Convert dataset with progress bar over the filtered range
output_data = {}
for i in tqdm(range(filtered_length), desc="Converting TableBench to TabMWP JSON"):
    tablebench.index = i
    entry = tablebench_to_tabmwp_format(tablebench)
    output_data[entry["pid"]] = entry

# Save to JSON file
with open("tablebench_test.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("Saved to tablebench_test.json")