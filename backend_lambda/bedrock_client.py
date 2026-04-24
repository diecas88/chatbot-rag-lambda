# bedrock_client.py

import boto3
from io import StringIO
import os
import json
import pandas as pd

s3 = boto3.client("s3")
BUCKET = os.getenv("MY_BUCKET_NAME")
GUARD_ID = os.getenv("MY_GUARD_ID")
GUARD_VER_ID = os.getenv("MY_GUARD_VER_ID")
KEY = "data/ranking_fifa.csv"
region = os.getenv("MY_REG_AWS", "us-east-2")

kb_client = boto3.client("bedrock-agent-runtime", region_name=region)
model_client = boto3.client("bedrock-runtime", region_name=region)

# alias paises español/inglés
ALIASES = {
    "brasil": "brazil",
    "alemania": "germany",
    "corea del sur": "south korea",
    "corea": "south korea",
    "eeuu": "united states",
    "usa": "united states",
    "estados unidos": "united states",
    "inglaterra": "england",
    "holanda": "netherlands",
    "paises bajos": "netherlands",
    "suiza": "switzerland",
    "japon": "japan",
    "mexico": "mexico",
    "argentina": "argentina",
    "colombia": "colombia",
    "peru": "peru",
    "uruguay": "uruguay",
    "paraguay": "paraguay",
    "chile": "chile",
    "ecuador": "ecuador",
    "espana": "spain",
    "francia": "france",
    "portugal": "portugal",
    "italia": "italy",
    "belgica": "belgium",
    "croacia": "croatia",
    "serbia": "serbia",
    "dinamarca": "denmark",
    "suecia": "sweden",
    "noruega": "norway",
    "polonia": "poland",
    "rusia": "russia",
    "turquia": "turkey",
    "iran": "iran",
    "arabia saudita": "saudi arabia",
    "qatar": "qatar",
    "australia": "australia",
    "nueva zelanda": "new zealand",
    "sudafrica": "south africa",
    "costa rica": "costa rica"
}

def load_teams():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    
    return df["country"].dropna().str.strip().str.lower().unique().tolist()

VALID_TEAMS = load_teams()

def normalize_team(team):
    if not team:
        return None

    team = team.strip().lower()

    # aplicar alias primero
    team = ALIASES.get(team, team)

    for valid in VALID_TEAMS:
        if team == valid:
            return valid

        if team in valid or valid in team:
            return valid

    return None

def query_kb(knowledge_base_id, query_text, max_results=5):
    response = kb_client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={"text": query_text}
    )

    return [
        item["content"]["text"]
        for item in response.get("retrievalResults", [])[:max_results]
    ]

def generate_answer_from_fragments(fragments, user_query, model_id=None):
    if not model_id:
        model_id = os.getenv("INFERENCE_PROFILE_ARN", "us.amazon.nova-lite-v1:0")

    prompt = f"""
Usa la siguiente información para responder en español:

{''.join(fragments)}

Pregunta: {user_query}
"""

    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 500, "temperature": 0.5}
    }

    response = model_client.invoke_model(
        modelId=model_id,
        guardrailIdentifier=GUARD_ID,
        guardrailVersion=GUARD_VER_ID,
        contentType="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]

def detect_intent_and_entities(user_query):

    model_id = os.getenv("INFERENCE_PROFILE_ARN", "us.amazon.nova-lite-v1:0")

    prompt = f"""
Responde SOLO en JSON válido:

{{
  "intent": "prediction" | "rag",
  "home": string | null,
  "away": string | null
}}

Consulta: "{user_query}"
"""

    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 100, "temperature": 0}
    }

    response = model_client.invoke_model(
        modelId=model_id,
        guardrailIdentifier=GUARD_ID,
        guardrailVersion=GUARD_VER_ID,
        contentType="application/json",
        body=json.dumps(body)
    )
    #print("resultado : result: ", response)
    result = json.loads(response["body"].read())
    #print("the result: ", result)
    text = result["output"]["message"]["content"][0]["text"]

    # DEBUG
    #print("RAW LLM:", text)

    try:
        text = text.strip()
        # limpiar markdown si viene con ```json
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(text)
        #print("PARSED:", parsed)
    except:
        #print("❌ JSON ERROR")

        if result["amazon-bedrock-guardrailAction"] == "INTERVENED":
            return {"intent": "INTERVENED", "home": None, "away": None}
        else:
            return {"intent": "rag", "home": None, "away": None}

    intent = parsed.get("intent")
    home = parsed.get("home")
    away = parsed.get("away")

    #print("INTENT:", intent)
    #print("HOME RAW:", home)
    #print("AWAY RAW:", away)

    home_valid = normalize_team(home)
    away_valid = normalize_team(away)

    #print("HOME VALID:", home_valid)
    #print("AWAY VALID:", away_valid)

    if result["amazon-bedrock-guardrailAction"] == "INTERVENED":
        return {"intent": "INTERVENED", "home": None, "away": None}
    
    if intent == "prediction":
        if not home_valid or not away_valid:
            #print("FALLÓ VALIDACIÓN")
            return {"intent": "rag", "home": None, "away": None}

        return {
            "intent": "prediction",
            "home": home_valid,
            "away": away_valid
        }

    return {"intent": "rag", "home": None, "away": None}

def generate_prediction_explanation(result, user_query, model_id=None):

    if not model_id:
        model_id = os.getenv("INFERENCE_PROFILE_ARN", "us.amazon.nova-lite-v1:0")

    prompt = f"""
Explica esta predicción:

Local: {result["home_team"]}
Visitante: {result["away_team"]}
Probabilidades: {result["probabilidades"]}
Goles esperados: {result["prediccion_goles"]}

Pregunta: {user_query}
"""

    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 300, "temperature": 0.3}
    }

    response = model_client.invoke_model(
        modelId=model_id,
        guardrailIdentifier=GUARD_ID,
        guardrailVersion=GUARD_VER_ID,
        contentType="application/json",
        body=json.dumps(body)
    )

    result_llm = json.loads(response["body"].read())
    return result_llm["output"]["message"]["content"][0]["text"]