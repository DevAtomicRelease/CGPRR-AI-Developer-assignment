#!/bin/sh
# Pull the models into the local Ollama container (first run only), build the
# index if it is missing, then run the requested command.
set -e
OLLAMA_URL="${OLLAMA_URL:-http://ollama:11434}"
CHAT="${NEETRAG_CHAT_MODEL:-qwen2.5:7b-instruct}"
EMBED="${NEETRAG_EMBED_MODEL:-nomic-embed-text}"

echo "Waiting for Ollama at $OLLAMA_URL ..."
until python -c "import requests,sys; requests.get('$OLLAMA_URL/api/tags', timeout=2)" 2>/dev/null; do sleep 2; done

for m in "$EMBED" "$CHAT"; do
  echo "Ensuring model $m is present (first pull can take several minutes) ..."
  python - "$m" <<'EOF'
import json, os, sys, requests
url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
with requests.post(f"{url}/api/pull", json={"model": sys.argv[1]}, stream=True, timeout=None) as r:
    for line in r.iter_lines():
        status = json.loads(line).get("status", "")
        if status in ("success",) or status.startswith("pulling manifest"):
            print(" ", status)
EOF
done

if [ ! -f /app/index/embeddings.npy ]; then
  python -m neetrag build-index
fi

case "$1" in
  serve) exec python -m neetrag serve --host 0.0.0.0 --port 8080 ;;
  eval)  exec python eval/run_eval.py ;;
  *)     exec python -m neetrag "$@" ;;
esac
