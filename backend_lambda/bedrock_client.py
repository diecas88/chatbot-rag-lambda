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