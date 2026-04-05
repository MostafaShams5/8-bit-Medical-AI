#!/bin/bash

DB_DIR="/app/medical_qdrant_db/collection"

ZIP_URL="https://huggingface.co/datasets/Shams03/Ara-Medical-RAG/resolve/main/medical_qdrant_db.zip"

echo "Running Startup Checks..."

if [ ! -d "$DB_DIR" ] || [ -z "$(ls -A $DB_DIR 2>/dev/null)" ]; then
    echo "Qdrant database not found locally. Downloading from Hugging Face..."
    
    wget -qO db.zip "$ZIP_URL"
    
    unzip -qo db.zip -d /app/
    
    rm db.zip
    
    echo "Database downloaded and extracted successfully."
else
    echo "Local Qdrant database found. Skipping download."
fi

echo "Starting Medical RAG API..."
exec python -m app.main
