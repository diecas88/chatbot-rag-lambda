import json
import os
from backend_lambda.bedrock_client import query_kb, generate_answer_from_fragments
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_ID = os.getenv("KB_ID")

def lambda_handler(event, context):
    user_query = event.get("query", "")
    
    if not user_query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No query provided"})
        }

    # Recuperar fragmentos de la KB
    fragments = query_kb(KNOWLEDGE_BASE_ID, user_query)

    if not fragments:
        return {
            "statusCode": 200,
            "body": json.dumps({"results": "No se encontró información relevante en la KB"})
        }

    # Generar respuesta en lenguaje natural
    answer = generate_answer_from_fragments(fragments, user_query)

    return {
        "statusCode": 200,
        "body": json.dumps({"results": answer})
    }