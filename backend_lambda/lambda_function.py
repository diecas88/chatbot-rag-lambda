import json
import re 
import html
import os
from bedrock_client import (
    query_kb,
    generate_answer_from_fragments,
    detect_intent_and_entities,
    generate_prediction_explanation
)
from predict import predict_match

KNOWLEDGE_BASE_ID = os.getenv("KB_ID")


def lambda_handler(event, context):

    try:

        method = (
            event.get("requestContext", {}).get("http", {}).get("method")
            or event.get("httpMethod")
        )

        if method == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": cors_headers(),
                "body": ""
            }

        user_query = None

        if "query" in event:
            user_query = event.get("query")

        elif "body" in event:
            try:
                body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
                user_query = body.get("query")
            except Exception as e:
                print("ERROR PARSING BODY:", str(e))

        if not user_query:
            return response(400, {"error": "No query provided"})

        user_query = sanitize_input(user_query)
        # INTENT
        intent_data = detect_intent_and_entities(user_query)

        intent = intent_data.get("intent")
        home = intent_data.get("home")
        away = intent_data.get("away")

        # PREDICCIÓN

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

                answer_prediction = generate_prediction_explanation(
                    result_clean, user_query
                )

                return response(200, {
                    "type": "prediction",
                    "raw": result_clean,
                    "data": answer_prediction
                })

            except Exception as e:
                print("❌ ERROR PREDICTION:", str(e))
                return response(500, {"error": str(e)})

        # Guardrails - aws - bedrock
        
        if intent == "INTERVENED":
            return response(200, {
                "type": "INTERVENED",
                "data": "Lo siento, el modelo no puede responder a este tipo de preguntas."
            })
       
        # RAG
       
        fragments = query_kb(KNOWLEDGE_BASE_ID, user_query)

        if not fragments:
            return response(200, {
                "type": "rag",
                "data": "No se encontró información relevante en la KB"
            })

        answer = generate_answer_from_fragments(fragments, user_query)

        return response(200, {
            "type": "rag",
            "data": answer
        })

    except Exception as e:
        return response(500, {
            "error": "Internal server error",
            "details": str(e)
        })

# HELPERS
def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
    }


def response(status, body):
    return {
        "statusCode": status,
        "headers": cors_headers(),
        "body": json.dumps(body)
    }

def sanitize_input(text):
    # 1. Eliminar etiquetas HTML/Script para evitar inyecciones básicas
    clean_text = re.sub(r'<[^>]*?>', '', text)
    
    # 2. Escapar caracteres especiales (convierte " < " en "&lt;", etc.)
    clean_text = html.escape(clean_text)
    
    # 3. Limitar la longitud (evita ataques de denegación de servicio por texto infinito)
    max_length = 1000 
    clean_text = clean_text[:max_length]
    
    return clean_text.strip()