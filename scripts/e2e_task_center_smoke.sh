#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8000/api/v1}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl 未安装，无法执行冒烟测试" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 未安装，无法执行冒烟测试" >&2
  exit 1
fi

echo "[smoke] API_BASE=${API_BASE}"

extract_json() {
  local json_text="$1"
  local py_expr="$2"
  python3 - "${json_text}" "${py_expr}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
expr = sys.argv[2]
value = eval(expr, {"__builtins__": {}}, {"payload": payload})
if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

post_json() {
  local url="$1"
  local body="$2"
  curl -sS -X POST "${url}" \
    -H 'Content-Type: application/json' \
    -d "${body}"
}

get_json() {
  local url="$1"
  curl -sS "${url}"
}

echo "[smoke] 1/3 inference run create + execute + locate"
CREATE_INFER_RESP="$(post_json "${API_BASE}/task-center/runs" '{
  "task_type": "inference",
  "trigger_mode": "manual",
  "input_payload": {
    "model": "/tmp/model",
    "algorithm": "chatts",
    "input_files": ["/tmp/demo.csv"]
  },
  "steps": ["inference"],
  "auto_execute": false
}')"
INFER_RUN_ID="$(extract_json "${CREATE_INFER_RESP}" "payload.get('data', {}).get('run_id', '')")"
if [[ -z "${INFER_RUN_ID}" ]]; then
  echo "[smoke] inference create 未返回 run_id" >&2
  echo "${CREATE_INFER_RESP}" >&2
  exit 1
fi

post_json "${API_BASE}/task-center/runs/${INFER_RUN_ID}/execute" '{"simulate": false}' >/dev/null
LIST_INFER_RESP="$(get_json "${API_BASE}/task-center/runs?run_id=${INFER_RUN_ID}&limit=20&offset=0")"
INFER_TOTAL="$(extract_json "${LIST_INFER_RESP}" "payload.get('data', {}).get('total', 0)")"
INFER_ROW_RUN_ID="$(extract_json "${LIST_INFER_RESP}" "(payload.get('data', {}).get('runs', [{}])[0] or {}).get('run_id', '')")"
if [[ "${INFER_TOTAL}" != "1" || "${INFER_ROW_RUN_ID}" != "${INFER_RUN_ID}" ]]; then
  echo "[smoke] inference run 定位失败" >&2
  echo "${LIST_INFER_RESP}" >&2
  exit 1
fi
echo "[smoke] inference run=${INFER_RUN_ID} ok"

echo "[smoke] 2/3 training run create + execute + locate"
TRAINING_CONFIG_RESP="$(get_json "${API_BASE}/training/configs?model_family=chatts")"
TRAINING_CONFIG_NAME="$(extract_json "${TRAINING_CONFIG_RESP}" "(payload.get('data', {}).get('configs', [{}])[0] or {}).get('name', '')")"
if [[ -z "${TRAINING_CONFIG_NAME}" ]]; then
  echo "[smoke] 未找到训练配置，跳过 training 流程"
else
  CREATE_TRAIN_RESP="$(post_json "${API_BASE}/task-center/runs" "{
    \"task_type\": \"training\",
    \"trigger_mode\": \"manual\",
    \"input_payload\": {
      \"config_name\": \"${TRAINING_CONFIG_NAME}\",
      \"model_family\": \"chatts\"
    },
    \"steps\": [\"training\"],
    \"auto_execute\": false
  }")"
  TRAIN_RUN_ID="$(extract_json "${CREATE_TRAIN_RESP}" "payload.get('data', {}).get('run_id', '')")"
  if [[ -z "${TRAIN_RUN_ID}" ]]; then
    echo "[smoke] training create 未返回 run_id" >&2
    echo "${CREATE_TRAIN_RESP}" >&2
    exit 1
  fi

  post_json "${API_BASE}/task-center/runs/${TRAIN_RUN_ID}/execute" '{"simulate": false}' >/dev/null
  LIST_TRAIN_RESP="$(get_json "${API_BASE}/task-center/runs?run_id=${TRAIN_RUN_ID}&limit=20&offset=0")"
  TRAIN_TOTAL="$(extract_json "${LIST_TRAIN_RESP}" "payload.get('data', {}).get('total', 0)")"
  TRAIN_ROW_RUN_ID="$(extract_json "${LIST_TRAIN_RESP}" "(payload.get('data', {}).get('runs', [{}])[0] or {}).get('run_id', '')")"
  if [[ "${TRAIN_TOTAL}" != "1" || "${TRAIN_ROW_RUN_ID}" != "${TRAIN_RUN_ID}" ]]; then
    echo "[smoke] training run 定位失败" >&2
    echo "${LIST_TRAIN_RESP}" >&2
    exit 1
  fi
  echo "[smoke] training run=${TRAIN_RUN_ID} ok"
fi

echo "[smoke] 3/3 done"
echo "[smoke] 全部通过"
