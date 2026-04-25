# Chatbot RAG: Mundiales de Fútbol

## Un asistente conversacional impulsado por AWS Bedrock y datos históricos

Este proyecto tiene como objetivo desarrollar un chatbot de "Retrieval Augmented Generation" (RAG) que pueda responder preguntas relacionadas con los mundiales de fútbol. Utiliza una arquitectura serverless con AWS Lambda y aprovecha AWS Bedrock para las capacidades de generación de lenguaje natural. La base de conocimiento del chatbot se construye a partir de datos históricos de los mundiales de fútbol, permitiéndole ofrecer respuestas informadas y contextualizadas.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

-   **`backend_lambda/`**: Contiene el código fuente para el backend serverless del chatbot.
    -   `bedrock_client.py`: Módulo para interactuar con AWS Bedrock, facilitando la integración de modelos generativos.
    -   `lambda_function.py`: La función principal de AWS Lambda que expone el chatbot y maneja las solicitudes entrantes.
    -   `predict.py`: Contiene la lógica de negocio para predecir y generar respuestas basadas en el contexto y la base de conocimiento.
    -   `train_model.py`: Este archivo contiene la lógica para el preprocesamiento o la fine-tuning de modelos si fuera necesario.
    -   `utils.py`: Módulo con funciones de utilidad compartidas por el backend.

-   **`data/`**: Almacena los datasets utilizados para construir la base de conocimiento del chatbot.
    -   `ranking_fifa.csv`: Datos históricos del ranking FIFA de 2026.
    -   `results.csv`: Resultados de partidos de los mundiales de fútbol.

-   **`notebooks/`**: Cuadernos Jupyter para la exploración de datos, la explicación de conceptos y las instrucciones de despliegue.
    -   `instructions_ECR.ipynb`: Guía paso a paso sobre cómo construir y subir una imagen Docker a Amazon Elastic Container Registry (ECR).
    -   `instructions_KB.ipynb`: Instrucciones detalladas sobre cómo construir una Base de Conocimiento (Knowledge Base) utilizando los datos proporcionados.
    -   `prepare_5_last_results.ipynb`: Notebook para el preprocesamiento de los últimos 5 resultados,  para mantener la base de conocimiento actualizada o para análisis específicos.

-   **`scripts/`**: Scripts auxiliares para la automatización de tareas y pruebas de la solución.

-   **`tests/`**: Incluye pruebas unitarias y de integración para la lógica de negocio del chatbot.
    -   `test_local.py`: Pruebas diseñadas para ser ejecutadas en un entorno local.

-   **`requirements.txt`**: Lista todas las dependencias de Python necesarias para el proyecto.

## Cómo Empezar

Para poner en marcha el proyecto en tu entorno local, sigue los siguientes pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/chatbot-rag.git
    cd chatbot-rag
    ```

2.  **Configurar el entorno virtual e instalar dependencias:**
    
    puedes crear un entorno virtual manualmente e instalar las dependencias:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Explorar los notebooks:**
    Dirígete al directorio `notebooks/` para entender cómo construir la base de conocimiento y cómo desplegar la imagen en ECR.
    ```bash
    jupyter notebook
    ```

4.  **Ejecutar pruebas (opcional):**
    Para asegurarte de que la lógica de negocio funciona correctamente, puedes ejecutar las pruebas:
    ```bash
    make test
    ```
    O directamente con `pytest`:
    ```bash
    pytest tests/
    ```

## Uso

Una vez que el backend del chatbot esté desplegado (siguiendo las instrucciones en los notebooks para AWS Lambda y Bedrock), podrás interactuar con él a través de la API expuesta por la función Lambda.

Este proyecto no tiene definidas reglas de seguridad ya que es una prueba de concepto, si vas a ponerlo en produccion asegurate que tienes la configuración correcta para evitar problemas o vulnerabilidades.

### Despliegue en AWS (resumen):

1.  **Configurar AWS CLI:** Asegúrate de tener el AWS CLI configurado con las credenciales apropiadas.
2.  **Crear Base de Conocimiento:** Sigue `notebooks/instructions_KB.ipynb` para preparar y subir tus datos a una Knowledge Base en Bedrock.
3.  **Desplegar Lambda:** Sigue `notebooks/instructions_ECR.ipynb` para construir la imagen Docker del backend y subirla a ECR. Luego, crea y configura la función AWS Lambda para usar esta imagen.
4.  **Configurar Bedrock:** Asegúrate de que tu función Lambda tenga los permisos necesarios para interactuar con AWS Bedrock.

### Ejemplo de Uso (interacción con el chatbot):

Suponiendo que tu función Lambda está desplegada y accesible a través de un API Gateway, podrías interactuar con el chatbot enviando solicitudes HTTP.

**Pregunta de ejemplo:** "¿Quién ganó el Mundial de 2014?"

**Solicitud (ejemplo hipotético a tu endpoint de API Gateway):**

```json
{
    "question": "¿Quién ganó el Mundial de 2014?"
}
```

**Respuesta (ejemplo hipotético del chatbot):**

```json
{
    "answer": "Alemania ganó el Mundial de 2014, venciendo a Argentina en la final."
}
```

**Pregunta de ejemplo 2:** "¿Cuál fue el resultado del partido entre Brasil y Alemania en 2014?"

**Solicitud:**

```json
{
    "question": "¿Cuál fue el resultado del partido entre Brasil y Alemania en 2014?"
}
```

**Respuesta:**

```json
{
    "answer": "En la semifinal del Mundial de 2014, Alemania goleó a Brasil con un marcador de 7-1."
}
```