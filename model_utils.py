import joblib
import pandas as pd
from scipy.sparse import hstack

def cargar_modelo():

    modelo = joblib.load("modelo_logreg.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    encoder = joblib.load("encoder.pkl")

    return modelo, vectorizer, encoder


def predecir_dataset(df, modelo, vectorizer, encoder):

    textos = df["TEXTO_COMPLETO"].fillna("")

    X_text = vectorizer.transform(textos)

    X_cat = df[["PRIORIDAD","GRUPO","ORIGEN"]]

    X_cat_enc = encoder.transform(X_cat)

    X = hstack([X_text, X_cat_enc])

    probs = modelo.predict_proba(X)[:,1]

    df["PROB_RIESGO"] = probs

    return df


def predecir_riesgo(modelo, vectorizer, encoder, asunto, descripcion, prioridad, grupo, origen):

    texto = f"{asunto} {descripcion}"

    X_text = vectorizer.transform([texto])

    X_cat = pd.DataFrame([{
        "PRIORIDAD": prioridad,
        "GRUPO": grupo,
        "ORIGEN": origen
    }])

    X_cat_enc = encoder.transform(X_cat)

    X = hstack([X_text, X_cat_enc])

    proba = float(modelo.predict_proba(X)[0,1])

    if proba < 0.35:
        nivel = "Bajo"
    elif proba < 0.65:
        nivel = "Medio"
    else:
        nivel = "Alto"

    return proba, nivel