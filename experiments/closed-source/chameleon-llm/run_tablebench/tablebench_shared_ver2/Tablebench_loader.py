from datasets import Dataset, load_dataset
import pandas as pd
import json


class Tablebench_Filtered:
    def __init__(self, dataset, index):
        qsubtypes = {
            'Aggregation', 'ArithmeticCalculation', 'Comparison', 'Counting',
            'Domain-Specific', 'MatchBased', 'Multi-hop FactChecking',
            'Multi-hop NumericalReasoing', 'Ranking', 'StatisticalAnalysis',
            'Time-basedCalculation', 'TrendForecasting'
        }
        self.dataset = dataset['test'].filter(lambda example: example['qsubtype'] in qsubtypes)
        self.index = index

    # Getter Methods
    def get_table(self):
        data_dict = json.loads(self.dataset['table'][self.index])
        df = pd.DataFrame(data=data_dict["data"], columns=data_dict["columns"])
        return df

    def get_question(self):
        return self.dataset['question'][self.index]

    def get_answer(self):
        return self.dataset['answer'][self.index]

    def get_instruction_type(self):
        return self.dataset['instruction_type'][self.index]

    def get_qtype(self):
        return self.dataset['qtype'][self.index]

    def get_qsubtype(self):
        return self.dataset['qsubtype'][self.index]

    def display_all(self):
        print("Table DataFrame:")
        df = self.get_table()
        print(df)
        print("\nquestion:", self.dataset['question'][self.index])
        print("answer:", self.dataset['answer'][self.index])
        print("instruction_type:", self.dataset['instruction_type'][self.index])
        print("qtype:", self.dataset['qtype'][self.index])
        print("qsubtype:", self.dataset['qsubtype'][self.index])