#!/bin/bash

# Script to create NovaEval API secrets
# Usage: ./create-secret.sh

set -e

echo "Creating NovaEval API Secret..."

# Function to base64 encode if value is provided
encode_if_present() {
    local value="$1"
    if [ -n "$value" ]; then
        printf %s "$value" | base64 | tr -d '\n'
    else
        echo ""
    fi
}

# Read API keys from environment or prompt
if [ -z "$OPENAI_API_KEY" ]; then
    read -p "Enter OpenAI API Key (or press Enter to skip): " OPENAI_API_KEY
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    read -p "Enter Anthropic API Key (or press Enter to skip): " ANTHROPIC_API_KEY
fi

if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    read -p "Enter Azure OpenAI API Key (or press Enter to skip): " AZURE_OPENAI_API_KEY
fi

if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    read -p "Enter Azure OpenAI Endpoint (or press Enter to skip): " AZURE_OPENAI_ENDPOINT
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    read -p "Enter Google API Key (or press Enter to skip): " GOOGLE_API_KEY
fi

# Create secret manifest
cat > novaeval-api-secret.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: novaeval-api-secrets
  labels:
    app: novaeval-api
type: Opaque
data:
EOF

# Add keys if they exist
if [ -n "$OPENAI_API_KEY" ]; then
    echo "  openai-api-key: $(encode_if_present "$OPENAI_API_KEY")" >> novaeval-api-secret.yaml
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  anthropic-api-key: $(encode_if_present "$ANTHROPIC_API_KEY")" >> novaeval-api-secret.yaml
fi

if [ -n "$AZURE_OPENAI_API_KEY" ]; then
    echo "  azure-openai-api-key: $(encode_if_present "$AZURE_OPENAI_API_KEY")" >> novaeval-api-secret.yaml
fi

if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "  azure-openai-endpoint: $(encode_if_present "$AZURE_OPENAI_ENDPOINT")" >> novaeval-api-secret.yaml
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    echo "  google-api-key: $(encode_if_present "$GOOGLE_API_KEY")" >> novaeval-api-secret.yaml
fi

echo ""
echo "Secret manifest created: novaeval-api-secret.yaml"
echo ""
echo "To apply to Kubernetes cluster:"
echo "  kubectl apply -f novaeval-api-secret.yaml"
echo ""
echo "To apply the full deployment:"
echo "  kubectl apply -f api-deployment.yaml"
echo ""
echo "To check status:"
echo "  kubectl get pods -l app=novaeval-api"
echo "  kubectl get services -l app=novaeval-api"
