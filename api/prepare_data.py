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
import nltk
import pymorphy2
from nltk.corpus import stopwords

stop_words = set(stopwords.words('russian'))

def lemmatized(text):
    # нормализация текста: приведение к нижнему регистру, удаление различных символов
    text = text.lower()
    text = text.replace(',', ' ')
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')
    text = text.replace(';', ' ')
    text = text.replace(':', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('}', ' ')
    text = text.replace('{', ' ')
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')

    text = text.replace('!', ' ')
    text = text.replace(r'\d+', ' ')
    text = text.replace(r'[\W]+', ' ')

    return text


# приведение токенов входящих в текст к нормальной форме
def norm(text):
    morph = pymorphy2.MorphAnalyzer()
    text_norm = ''
    for token in nltk.word_tokenize(text):
        # print('token = ', token)
        token_norm = morph.parse(token)[0].normal_form
        if token_norm not in stop_words:
            text_norm = text_norm + ' ' + token_norm
        # print('text_norm', text_norm)
    return text_norm

def text_clear(text):
    text = lemmatized(text)
    text = norm(text)
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
