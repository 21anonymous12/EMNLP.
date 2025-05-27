from datasets import load_dataset
import pandas as pd


class WikiTQ:
    def __init__(self, dataset, index, type='train'):
        """데이터셋과 인덱스를 받아 객체를 초기화합니다."""
        self.dataset = dataset
        self.index = index
        self.type = type
        self._extract_data()  # 내부 데이터 추출

    def _extract_data(self):
        """데이터에서 테이블과 관련된 정보를 추출하여 저장"""
        entry = self.dataset[self.type][self.index]

        # 테이블 정보 추출 및 DataFrame 생성
        self.table_header = entry['table']['header']
        self.table_rows = entry['table']['rows']
        self.df = pd.DataFrame(self.table_rows, columns=self.table_header)

        # 기타 정보 추출
        self.question = entry['question']
        self.answers = entry['answers']

    # Getter 메서드들
    def get_table(self):
        """테이블(DataFrame) 반환"""
        return self.df

    def get_question(self):
        """질문 반환"""
        return self.question

    def get_answers(self):
        """정답 목록 반환"""
        return self.answers

    def display_all(self):
        """모든 정보를 출력"""
        print("Table DataFrame:")
        print(self.df)
        print("\nQuestion:", self.question)
        print("\nAnswers:", self.answers)