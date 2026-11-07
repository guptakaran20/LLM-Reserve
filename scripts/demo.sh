#!/usr/bin/env bash
# predict -> reserve -> settle against a running service (`docker compose up --build`), with curl alone.
set -euo pipefail
API=${API:-http://localhost:8000}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PROMPT=${PROMPT:-Explain in two sentences why the sky is blue.}
INPUT_TOKENS=${INPUT_TOKENS:-20}
post() { curl -sS --fail-with-body -X POST "$API$1" -H 'content-type: application/json' -d "$2"; }
field() { python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"; }

echo "== GET /healthz";  curl -sS --fail-with-body "$API/healthz"; echo

echo "== POST /predict"
PRED=$(post /predict "{\"prompt\": \"$PROMPT\", \"model\": \"$MODEL\", \"input_tokens\": $INPUT_TOKENS, \"max_tokens\": 4096}")
echo "$PRED"
BOUND=$(echo "$PRED" | field bound)

echo "== POST /reserve"
RES=$(post /reserve "{\"input_tokens\": $INPUT_TOKENS, \"max_tokens\": 4096, \"bound\": $BOUND}")
echo "$RES"
RID=$(echo "$RES" | field reservation_id)
CAP=$(echo "$RES" | field max_tokens)

# Here a gateway forwards the call with max_tokens=$CAP. Pretend the answer used 60% of its bound.
OUTPUT_TOKENS=$(( BOUND * 6 / 10 ))
echo "== POST /settle (output_tokens=$OUTPUT_TOKENS, sent with max_tokens=$CAP)"
post /settle "{\"reservation_id\": \"$RID\", \"input_tokens\": $INPUT_TOKENS, \"output_tokens\": $OUTPUT_TOKENS}"; echo

echo "== GET /budget"; curl -sS --fail-with-body "$API/budget"; echo
