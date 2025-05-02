#!/bin/bash

# Carregar variáveis do arquivo .env
set -a
source .env
set +a

# Nome da imagem
IMAGE_NAME="model_api"

# Executar o container com as variáveis
docker run -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
           -e MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME" \
           -e MLFLOW_TRACKING_PASSWORD="$MLFLOW_TRACKING_PASSWORD" \
           -p 80:80 \
           $IMAGE_NAME
