import json
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_label(text, tokenizer, model):
    """입력 텍스트를 받아서 모델이 예측한 라벨을 반환하는 함수"""
    # 입력 텍스트 토큰화
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=2800, return_tensors="pt")

    # 입력 데이터를 GPU로 이동 (가능한 경우)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 시그모이드 함수로 변환 (multi-label classification)
    probabilities = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

    # 가장 확률이 높은 인덱스만 1로 설정하고 나머지는 0
    #predicted_labels = np.zeros_like(probabilities, dtype=int)
    #predicted_labels[np.argmax(probabilities)] = 1

    # 확률값 그대로 반환
    return probabilities

