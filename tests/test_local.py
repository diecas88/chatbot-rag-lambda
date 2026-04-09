# test_local.py
import sys
import os
import json

# 👉 Agrega la raíz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_lambda.lambda_function import lambda_handler

text = input("escribe tu pregunta:")
# Simula el evento que recibiría la Lambda desde API Gateway
event = {
    "query": text
}

# Contexto falso
context = {}

try:
    response = lambda_handler(event, context)

    if "body" in response:
        body = json.loads(response["body"])
        print("Respuesta:\n")
        print(json.dumps(body, indent=4, ensure_ascii=False))
    else:
        print("Respuesta inesperada:")
        print(response)

except Exception as e:
    print("Error al ejecutar:")
    print(e)