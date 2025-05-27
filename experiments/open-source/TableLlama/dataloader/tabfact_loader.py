from datasets import load_dataset
import pandas as pd


class TabFact:
    def __init__(self, dataset, index):
        """Data와 인덱스를 받아 객체를 초기화합니다."""
        self.dataset = dataset
        self.index = index
        self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        # 1) 줄단위로 분리(앞뒤 공백 제거 후)
        lines = self.dataset['table_text'][self.index].strip().split('\n')

        # 2) 첫 줄을 컬럼명으로 분리
        columns = lines[0].split('#')

        # 3) 나머지 줄(데이터 라인들)을 # 구분자로 분리하여 2차원 리스트 생성
        data_lines = [line.split('#') for line in lines[1:] if line.strip()]

        # 4) pandas DataFrame으로 변환
        self.df = pd.DataFrame(data_lines, columns=columns)


    # Getter 메서드들
    def get_table(self):
        """테이블(DataFrame) 반환"""
        return self.df

    def get_statement(self):
        """주어진 문장(statement) 반환"""
        return self.dataset['statement'][self.index]

    def get_label(self):
        """정답 레이블(label) 반환"""
        return self.dataset['label'][self.index]

    def get_caption(self):
        """테이블의 캡션 반환"""
        return self.dataset['table_caption'][self.index]

    def display_all(self):
        """모든 데이터를 출력"""
        print("Table DataFrame:")
        print(self.df)
        print("\nStatement:", self.dataset['statement'][self.index])
        print("Label:", self.dataset['label'][self.index])
        print("Caption:", self.dataset['table_caption'][self.index])