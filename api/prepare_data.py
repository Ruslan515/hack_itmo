#!/usr/bin/env python

# для ВМ

# запись вероятностей принадлжености к классу
# модуль предобработки данных о резюме
# для автомодерации с внедрением API.
# на вход подается одна строка фрейма со множеством столбцов,
# на выходе получается фрейм с одной строкой и одним столбцом,
# который получается в ходе преобразования столбцов к строковой форме

import pandas as pd
from predict import predict_X

def text_clear(text):
    return text

def main_clear(data_json):        
    text = text_clear(data_json["comment"])
    return text

def main_predict(data_json):
    X = main_clear(data_json)
    X = pd.DataFrame(data={"text": [X]})
    code = predict_X(X)

    ans = {
        "version": data_json['version'],
        'event_type': data_json['event_type'],
        'event_date': data_json['event_date'],
        'body': {
            "answer": "toxic" if code[0] == 1 else "not toxic",
            "probability": code[1]
        }
    }
    return ans
