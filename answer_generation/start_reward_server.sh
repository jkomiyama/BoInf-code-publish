#!/bin/bash

# Load .env file
if [ -f ".env" ]; then
    echo "📄 Loading .env file..."
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
else
    echo "⚠️  .env file not found. Using default values."
fi

# Set default values (only if not set in .env)
REWARD_MODEL_ID_DEFAULT="/workspace/ArmoRM-Llama3-8B-v0.1"

# Help display function
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script reads REWARD_MODEL_ID from the .env file."
    echo ""
    echo "Options:"
    echo "  -h, --help                   Show this help"
    echo ""
    echo ".env file example:"
    echo "  REWARD_MODEL_ID=/workspace/ArmoRM-Llama3-8B-v0.1"
}

# If model path is specified as positional argument (first argument), use it with highest priority
MODEL_OVERRIDE=""
if [[ $# -gt 0 && "${1#-}" = "$1" ]]; then
    MODEL_OVERRIDE="$1"
    shift
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Use default value if REWARD_MODEL_ID is not set
if [ -z "$REWARD_MODEL_ID" ]; then
    export REWARD_MODEL_ID="$REWARD_MODEL_ID_DEFAULT"
    echo "⚠️  REWARD_MODEL_ID is not set. Using default value."
fi

echo "🚀 Starting reward server..."
# If specified as first argument, prioritize that
if [ -n "$MODEL_OVERRIDE" ]; then
    export REWARD_MODEL_ID="$MODEL_OVERRIDE"
    echo "🔧 Using model specified as argument: $REWARD_MODEL_ID"
fi

echo "📦 Using model: $REWARD_MODEL_ID"
echo "🌐 Host: 0.0.0.0:9000"
echo ""

python -m uvicorn reward_server:app --host 0.0.0.0 --port 9000 --workers 1
