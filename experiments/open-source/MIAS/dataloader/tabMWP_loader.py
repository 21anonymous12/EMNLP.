from datasets import load_dataset
import pandas as pd


class TabMWP:
    def __init__(self, df, number):
        """클래스 초기화: DataFrame과 인덱스(number) 저장"""
        self.df = df
        index_list = df.columns
        self.number = index_list[number]
        self._extract_data()  # 내부 데이터 추출 함수 호출
        

    def _extract_data(self):
        """df에서 데이터를 추출하여 클래스 변수에 저장"""
        entry = self.df[self.number]
        self.table = pd.DataFrame(entry['table_for_pd'])
        self.question = entry['question']
        self.choice = entry['choices']  # 선택지가 없을 수 있음
        self.solution = entry['solution']
        self.answer = entry['answer']
        self.ques_type = entry['ques_type']
        self.ans_type = entry['ans_type']
        self.grade = entry['grade']
        self.table_title = entry['table_title']

    def get_table(self):
        """테이블 데이터를 반환"""
        return self.table

    def get_question(self):
        """문제 문장을 반환"""
        return self.question

    def get_choices(self):
        """선택지를 반환"""
        return self.choice

    def get_solution(self):
        """풀이를 반환"""
        return self.solution

    def get_answer(self):
        """정답을 반환"""
        return self.answer

    def get_ques_type(self):
        """ques_type을 반환"""
        return self.ques_type
    
    def get_ans_type(self):
        """ans_type을 반환"""
        return self.ans_type
    
    def get_grade(self):
        """grade을 반환"""
        return self.grade
    
    def get_table_title(self):
        """table_title을 반환"""
        return self.table_title
    

    def display_all(self):
        """모든 데이터를 출력"""
        print("Table:")
        print(self.table)
        print("\nQuestion:", self.question)
        print("\nChoices:", self.choice)
        print("Solution: \n")
        print(self.solution)
        print("\nAnswer:", self.answer)
        print("\nques_type:", self.ques_type)
        print("\nans_type:", self.ans_type)
        print("\ngrade:", self.grade)
        print("\ntable_title:", self.table_title)
