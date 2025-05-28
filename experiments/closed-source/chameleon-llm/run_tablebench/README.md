# Tablebench Testing
These codes are designed specifically to run the TableBench dataset.

# Tablebench Link: 
! https://tablebench.github.io/

# :lizard: Chameleon: Plug-and-Play Compositional Reasoning with GPT-4
Paper Link: ! https://arxiv.org/pdf/2304.09842 
Source Code: ! https://github.com/lupantech/chameleon-llm

# run.py file
parser.add_argument('--test_number', type=int, default=10)
→ Modify the default value to run the entire TableBench dataset.

# cache_file
./results/tablebench/chameleon_chatgpt_tablebench_test_cache.json
→ Shows the result for each question. true if correct, false if incorrect.

# result_file
./results/tablebench/chameleon_chatgpt_tablebench_test.json
→ Provides an overall summary of the results.

# ./env file
Create your own ./env file to enter your OpenAI API key (Handle with care).


# Execution process (run.py)
To run the full dataset, set the default value to the total number of TableBench questions and execute.
When testing with 10 samples, each question takes about 30 seconds to 1 minute.

