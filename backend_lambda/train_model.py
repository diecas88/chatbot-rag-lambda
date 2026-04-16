# train_model.py

import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# leer archivo limpio
df = pd.read_csv("../data/clean_data.csv")

# function para definir si partido es ganado, perdido o empate
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2  # gana local
    elif row["home_score"] < row["away_score"]:
        return 0  # pierde local
    else:
        return 1  # empate

# crear columna resultado
df["resultado"] = df.apply(get_result, axis=1)

# enumerar las features a usar
features = [
    "peso", 
    "recencia",
    "home_level",
    "away_level", 
    "level_diff",
    "home_avg_goals_scored_5",
    "home_avg_goals_conceded_5",
    "away_avg_goals_scored_5",
    "away_avg_goals_conceded_5"
    ]

X = df[features]

# 1. aplicar modelo de clasificacion
y_clf = df["resultado"]

X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

model_clf = RandomForestClassifier()
model_clf.fit(X_train, y_train)

print("Accuracy (clf):", model_clf.score(X_test, y_test))


# 2. aplicar regression lineal

y_reg = df["total_goal"]

X_train_r, X_test_r, Y_train_r, Y_test_r, = train_test_split(X, y_reg, test_size= 0.2, random_state=42)

model_reg = RandomForestRegressor()
model_reg.fit(X_train_r, Y_train_r)

print("Score R2 (regression): ", model_reg.score(X_test_r, Y_test_r))

# guardar modelos

joblib.dump(model_clf, "../model/model_clf.pkl")
joblib.dump(model_reg, "../model/model_reg.pkl")