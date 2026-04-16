import boto3
import os
import json
#from dotenv import load_dotenv
#load_dotenv()

region = os.getenv("MY_REG_AWS", "us-east-2")

# Cliente para KB (RAG)
kb_client = boto3.client("bedrock-agent-runtime", region_name=region)

# Cliente para generación de texto
model_client = boto3.client("bedrock-runtime", region_name=region)


def query_kb(knowledge_base_id, query_text, max_results=5):
    response = kb_client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={
            "text": query_text
        }
    )

    results = [
        item["content"]["text"]
        for item in response.get("retrievalResults", [])[:max_results]
    ]

    return results

def generate_answer_from_fragments(fragments, user_query, model_id=None):
    if not model_id:
        model_id = os.getenv(
            "INFERENCE_PROFILE_ARN",
            "us.amazon.nova-lite-v1:0"
        )

    prompt = f"""
Usa la siguiente información para responder la pregunta de forma clara y en español:

{''.join(fragments)}

Pregunta: {user_query}
"""

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 500,
            "temperature": 0.5
        }
    }

    response = model_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())

    return result["output"]["message"]["content"][0]["text"]

def detect_intent_and_entities(user_query):
    
    model_id = os.getenv(
        "INFERENCE_PROFILE_ARN",
        "us.amazon.nova-lite-v1:0"
    )

    prompt = f"""
Clasifica la siguiente consulta y extrae equipos si aplica.

Responde SOLO en JSON válido con este formato:

{{
  "intent": "prediction" o "rag",
  "home": "equipo_local" o null,
  "away": "equipo_visitante" o null
}}

Reglas:
- "prediction" si es pronóstico o partido
- "rag" si es pregunta general
- Si no hay equipos claros → null

Consulta: "{user_query}"
"""

    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 100,
            "temperature": 0
        }
    }

    response = model_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    text = result["output"]["message"]["content"][0]["text"]

    # 🔥 intentar parsear JSON del modelo
    try:
        parsed = json.loads(text)
        return parsed
    except:
        # fallback
        return {
            "intent": "rag",
            "home": None,
            "away": None
        }

