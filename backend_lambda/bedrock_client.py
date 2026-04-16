# bedrock_client.py

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
Eres un clasificador de intención.

REGLAS ESTRICTAS:

- SOLO puedes elegir entre:
  - "prediction"
  - "rag"
  - null

- NUNCA hagas predicciones numéricas
- NUNCA inventes probabilidades
- NUNCA generes análisis deportivo

REGLAS CLAVE:
- Si el texto contiene "X vs Y" pero habla en pasado pidiendo datos historicos → "rag"
- Si el texto contiene "X vs Y" → SIEMPRE "prediction"
- Si el texto contiene "X contra Y" → SIEMPRE "prediction"
- Si hay dos equipos de fútbol → "prediction"
- Si el usuario pide ganador o resultado → "prediction"
- Si el usuario pide datos historicos → "rag"
- Todo lo demás → "rag"

IMPORTANTE:
Tu trabajo es SOLO clasificar intención, NO responder la pregunta.

EJEMPLOS:

Input: Cual fue el resultado de Argentina vs Francia en el mundial de 2022?
Output:
{{
  "intent": "rag",
  "home": "Argentina",
  "away": "Francia"
}}

Input: Colombia vs Portugal
Output:
{{
  "intent": "prediction",
  "home": "Colombia",
  "away": "Portugal"
}}

Input: Colombia contra Portugal
Output:
{{
  "intent": "prediction",
  "home": "Colombia",
  "away": "Portugal"
}}

Input: Colombia aganist Portugal
Output:
{{
  "intent": "prediction",
  "home": "Colombia",
  "away": "Portugal"
}}

Input: who wins argentina vs brazil
Output:
{{
  "intent": "prediction",
  "home": "Argentina",
  "away": "Brazil"
}}

Input: who won world cup 2014
Output:
{{
  "intent": "rag",
  "home": null,
  "away": null
}}

Input: colombia world cup history
Output:
{{
  "intent": "rag",
  "home": null,
  "away": null
}}

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

    # intentar parsear JSON del modelo
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

def generate_prediction_explanation(result, user_query, model_id=None):
    import os
    import json

    if not model_id:
        model_id = os.getenv(
            "INFERENCE_PROFILE_ARN",
            "us.amazon.nova-lite-v1:0"
        )

    prompt = f"""
Eres un analista de fútbol.

Explica la siguiente predicción de manera clara, breve y en español.

DATOS (NO LOS CAMBIES NI INVENTES):
- Equipo local: {result["home_team"]}
- Equipo visitante: {result["away_team"]}
- Probabilidad local: {result["probabilidades"]["home_win"]}
- Probabilidad empate: {result["probabilidades"]["draw"]}
- Probabilidad visitante: {result["probabilidades"]["away_win"]}
- Goles esperados local: {result["prediccion_goles"]["home"]}
- Goles esperados visitante: {result["prediccion_goles"]["away"]}

REGLAS IMPORTANTES:
- NO inventes números nuevos
- NO cambies los valores
- SOLO explica los datos
- Sé natural y fácil de entender

Pregunta del usuario:
{user_query}
"""

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 300,
            "temperature": 0.3  
        }
    }

    response = model_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        body=json.dumps(body)
    )

    result_llm = json.loads(response["body"].read())

    return result_llm["output"]["message"]["content"][0]["text"]
