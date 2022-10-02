# !!! Модуль для локальной машины !!!

import os

from create_model import create_model

def predict_X(X):
    path_model = "../hack_ml/"
    file_model = "hack_model"
    model = create_model()
    model.load_model(os.path.join(path_model, file_model))
    y_pred = int(model.predict(X)[0])
    y_proba = model.predict_proba(X)[0]
    y_proba = y_proba[y_pred]
    print("y_pred ", y_pred, ". y_proba ", y_proba)
    return [y_pred, y_proba]