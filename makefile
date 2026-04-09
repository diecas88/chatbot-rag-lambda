# Nombre del zip final
PACKAGE_NAME=chatbot.zip

# Carpeta temporal de build
BUILD_DIR=zip-to-lambda

# Archivos que quieres incluir en Lambda
FILES=lambda_function.py bedrock_client.py

build:
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)

	# 👇 instalar dependencias dentro del build
	pip install -r requirements.txt -t $(BUILD_DIR)/

	# 👇 copiar tu código
	cp $(FILES) $(BUILD_DIR)/

zip: build
	cd $(BUILD_DIR) && zip -r ../$(PACKAGE_NAME) .

clean:
	rm -rf $(BUILD_DIR) $(PACKAGE_NAME)

deploy: zip
	aws lambda update-function-code \
		--function-name chatbot-call \
		--zip-file fileb://$(PACKAGE_NAME)

all: clean deploy