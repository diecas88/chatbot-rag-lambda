import json
import os

from bedrock_client import query_kb, generate_answer_from_fragments, detect_intent_and_entities
from predict import predict_match

import boto3

KNOWLEDGE_BASE_ID = os.getenv("KB_ID")

# -------------------------------
# 🧠 Lambda handler
# -------------------------------
def lambda_handler(event, context):
    
    user_query = event.get("query")
    
    if not user_query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No query provided"})
        }

    # -------------------------------
    # 🔥 1. Detectar intención con IA
    # -------------------------------
    intent_data = detect_intent_and_entities(user_query)

    intent = intent_data.get("intent")
    home = intent_data.get("home")
    away = intent_data.get("away")

    # -------------------------------
    # 🔵 2. Predicción ML
    # -------------------------------
    if intent == "prediction" and home and away:
        
        try:
            result = predict_match(home, away)

            result_clean = {
                "home_team": result["home_team"],
                "away_team": result["away_team"],
                "probabilidades": {
                    "home_win": float(result["probabilidades"]["home_win"]),
                    "draw": float(result["probabilidades"]["draw"]),
                    "away_win": float(result["probabilidades"]["away_win"]),
                },
                "prediccion_goles": {
                    "home": float(result["prediccion_goles"]["home"]),
                    "away": float(result["prediccion_goles"]["away"]),
                    "total": float(result["prediccion_goles"]["total"]),
                }
            }

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "type": "prediction",
                    "data": result_clean
                })
            }

        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)})
            }

    # -------------------------------
    # 🟢 3. RAG (fallback)
    # -------------------------------
    fragments = query_kb(KNOWLEDGE_BASE_ID, user_query)

    if not fragments:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "type": "rag",
                "data": "No se encontró información relevante en la KB"
            })
        }

    answer = generate_answer_from_fragments(fragments, user_query)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "type": "rag",
            "data": answer
        })
    }