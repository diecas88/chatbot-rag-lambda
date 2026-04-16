# predict.py

import pandas as pd
import joblib
import boto3
from io import BytesIO, StringIO
import os

s3 = boto3.client("s3")
BUCKET = os.getenv("MY_BUCKET_NAME")


def load_s3_file(bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)

def load_model(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(BytesIO(obj["Body"].read()))

def load_csv(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

# carga modelos entrenados
# model_clf = joblib.load("model_clf.pkl")
# model_reg = joblib.load("model_reg.pkl")
# model_clf = joblib.load("../model/model_clf.pkl")
# model_reg = joblib.load("../model/model_reg.pkl")

model_clf = load_model(BUCKET, "model/model_clf.pkl")
model_reg = load_model(BUCKET, "model/model_reg.pkl")

# carga data
# ranking_df = pd.read_csv("ranking_fifa.csv") # ranking fifa 2026 abril
# ranking_df = pd.read_csv("../data/ranking_fifa.csv") # ranking fifa 2026 abril

ranking_df = load_csv(BUCKET, "data/ranking_fifa.csv")

ranking_dict = dict(zip(ranking_df["country"], ranking_df["rank"]))

#df_hist = pd.read_csv("clean_data.csv") # datos limpios
# df_hist = pd.read_csv("../data/clean_data.csv") # datos limpios
df_hist = load_csv(BUCKET, "data/clean_data.csv")

# funcion para obtener goles anotados y goles recibidos, solo toma los ultimos 5 partidos jugados por colombia
def get_team_form(team):
    
    df_team = df_hist[
        (df_hist["home_team"] == team) |
        (df_hist["away_team"] == team)
    ].copy()
    
    if df_team.empty:
        return 0, 0
    
    if "date" in df_team.columns:
        df_team = df_team.sort_values("date", ascending=False)
    
    df_last5 = df_team.head(5)
    
    goals_scored = []
    goals_conceded = []
    
    for _, row in df_last5.iterrows():
        
        if row["home_team"] == team:
            goals_scored.append(row["home_score"])
            goals_conceded.append(row["away_score"])
        else:
            goals_scored.append(row["away_score"])
            goals_conceded.append(row["home_score"])
    
    return (
        sum(goals_scored) / len(goals_scored),
        sum(goals_conceded) / len(goals_conceded)
    )


# obtener el ranking level de equipo
def get_level(rank):

    if rank > 200:
        return 0
    else:
        return 200 - rank


# obtener recencia
def calcular_recencia_equipo(equipo):
    
    df_team = df_hist[
        (df_hist["home_team"] == equipo) |
        (df_hist["away_team"] == equipo)
    ]
    
    if df_team.empty:
        return 0.5
    
    return df_team["recencia"].mean()


# predecir partido (gana, pierde, empata)
def predict_match(home, away, tournament="FIFA World Cup"):
    
    # raking
    home_rank = ranking_dict.get(home, 100)
    away_rank = ranking_dict.get(away, 100)
    
    # level
    home_level = get_level(home_rank)
    away_level = get_level(away_rank)
    level_diff = home_level - away_level
    
    # peso de partidos
    pesos = {
        "Friendly": 1,
        "FIFA World Cup": 3,
        "FIFA World Cup qualification": 2
    }
    
    peso = pesos.get(tournament, 1)
    
    # recencia
    recencia_home = calcular_recencia_equipo(home)
    recencia_away = calcular_recencia_equipo(away)
    recencia = (recencia_home + recencia_away) / 2
    
    # forma (goles a favor y en contra)
    home_scored, home_conceded = get_team_form(home)
    away_scored, away_conceded = get_team_form(away)
    
    # features
    nuevo = pd.DataFrame([{
        "peso": peso,
        "recencia": recencia,
        "home_level": home_level,
        "away_level": away_level,
        "level_diff": level_diff,
        "home_avg_goals_scored_5": home_scored,
        "home_avg_goals_conceded_5": home_conceded,
        "away_avg_goals_scored_5": away_scored,
        "away_avg_goals_conceded_5": away_conceded
    }])
    
    # modelo clasificacion
    prob = model_clf.predict_proba(nuevo)[0]
    
    # regresion lineal (goles)
    total_goals = model_reg.predict(nuevo)[0]
    
    # evitar valores negativos
    total_goals = max(total_goals, 0)
    
    # distribucion de goles
    total_attack = home_scored + away_scored
    
    if total_attack == 0:
        home_goals = total_goals / 2
        away_goals = total_goals / 2
    else:
        home_goals = total_goals * (home_scored / total_attack)
        away_goals = total_goals * (away_scored / total_attack)
    
    return {
        "home_team": home,
        "away_team": away,
        "probabilidades": {
            "home_win": float(prob[2]),
            "draw": float(prob[1]),
            "away_win": float(prob[0])
        },
        "prediccion_goles": {
            "home": float(round(home_goals, 2)),
            "away": float(round(away_goals, 2)),
            "total": float(round(total_goals, 2))
        },
        "features_usadas": {
            "recencia": float(recencia),
            "home_form_scored": float(home_scored),
            "away_form_scored": float(away_scored)
        }
    }


# hacer test
print(predict_match("Argentina", "France"))