# Tablebench Testing
These codes are designed specifically to run the TableBench dataset.

# Tablebench Link: 
! https://tablebench.github.io/

# ReAcTable
Paper Link: ! https://arxiv.org/pdf/2310.00815
Source Code: ! https://github.com/yunjiazhang/ReAcTable

# result_file
./notebook/tablebench_eval_logs/tablebench_results.json
→ Provides an overall summary of the results.

# ./notebook/.env file
Create your own ./env file to enter your OpenAI API key (Handle with care).

# ./notebook/ReAcTable_TableBench_Evaluation.py 
run the above python file to evaluate Tablebench for ReAcTable

# ./tabqa directory
contains all necessary files to execute ReAcTable. DO NOT DELETE THIS DIRECTORY

# Changes in ./tabqa/GPTPrompter.py
In this code, pylcs was originally used for calculating the Longest Common Subsequence (LCS) similarity:
    sim = pylcs.lcs_sequence_length(utterance, c) / len(c)

However, due to environment constraints, pylcs was not used, and instead the following alternative approach was implemented:
    sim = sum(ch in utterance for ch in c) / len(c)

This method computes a simplified character-level overlap ratio between the utterance and each column name c. While it's not a true LCS metric, it serves as a reasonable heuristic for estimating similarity and functions correctly for the intended purpose in this context.