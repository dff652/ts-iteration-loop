#!/usr/bin/env bash
# 本地一键启动脚本（支持安全基线默认值）
# 用法:
#   ./scripts/start_local.sh app
#   ./scripts/start_local.sh worker
#   ./scripts/start_local.sh legacy   # 旧版界面（8000/train-ui）
#   ./scripts/start_local.sh dual     # 新旧界面同启（8000 + 5173）
#   ./scripts/start_local.sh all      # dual 别名
#   ./scripts/start_local.sh frontend # 仅新版前端（5173）
#   ./scripts/start_local.sh smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

ENV_NAME="${START_LOCAL_ENV_NAME:-ts-iteration-loop}"
MODE="${1:-all}"
ARGS=("$@")
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
API_PORT="${API_PORT:-8000}"

load_env_file() {
  if [[ -f ".env" ]]; then
    # shellcheck disable=SC1091
    set -a && source ".env" && set +a
  fi
}

generate_jwt_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
  fi
}

ensure_security_env() {
  if [[ -z "${JWT_SECRET_KEY:-}" || "${#JWT_SECRET_KEY}" -lt 32 ]]; then
    export JWT_SECRET_KEY="$(generate_jwt_secret)"
    echo "⚠️  JWT_SECRET_KEY 未配置或过短，已生成临时开发密钥。"
  fi

  export CORS_ALLOW_ORIGINS="${CORS_ALLOW_ORIGINS:-http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173}"

  # 本地开发默认开启 Annotator 鉴权绕过，避免新前端标注中心代理请求出现 401。
  # 生产或联调可显式设置 ANNOTATOR_AUTH_BYPASS=false 关闭。
  export ANNOTATOR_AUTH_BYPASS="${ANNOTATOR_AUTH_BYPASS:-true}"
  export ANNOTATOR_AUTH_BYPASS_USER="${ANNOTATOR_AUTH_BYPASS_USER:-${DEFAULT_USER:-${USER:-douff}}}"
}

prepare_runtime_dirs() {
  export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-$PROJECT_ROOT/data/gradio_tmp}"
  mkdir -p "$GRADIO_TEMP_DIR" "$PROJECT_ROOT/logs"
}

is_target_env_active() {
  local current_name="${CONDA_DEFAULT_ENV:-}"
  local prefix_name=""
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    prefix_name="$(basename "$CONDA_PREFIX")"
  fi

  [[ "$current_name" == "$ENV_NAME" ]] && return 0
  [[ -n "$current_name" && "$(basename "$current_name")" == "$ENV_NAME" ]] && return 0
  [[ "$prefix_name" == "$ENV_NAME" ]] && return 0
  return 1
}

ensure_runtime_env() {
  if [[ "${_START_LOCAL_IN_CONDA:-0}" == "1" ]]; then
    return 0
  fi

  if is_target_env_active; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "❌ 未检测到 conda，且当前不在目标环境 '$ENV_NAME'。"
    echo "   请先安装 conda 或手动进入可用环境后再启动。"
    exit 1
  fi

  echo "↪ 当前环境: ${CONDA_DEFAULT_ENV:-<none>}，自动切换到 conda 环境 '$ENV_NAME'..."
  local conda_base=""
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME" || true
    if is_target_env_active; then
      return 0
    fi
  fi

  # Fallback: re-exec via conda run if activation in current shell failed.
  exec conda run -n "$ENV_NAME" env _START_LOCAL_IN_CONDA=1 bash "$0" "${ARGS[@]}"
}

preflight_checks() {
  if ! python -c "import scipy" >/dev/null 2>&1; then
    echo "❌ 当前 Python 环境缺少 scipy，主服务无法启动。"
    echo "   建议先执行:"
    echo "   1) conda activate ts-iteration-loop"
    echo "   2) 若仍缺依赖: pip install -r envs/requirements.txt"
    exit 1
  fi
}

ensure_frontend_ready() {
  local frontend_dir="$PROJECT_ROOT/src/frontend"
  if [[ ! -d "$frontend_dir" || ! -f "$frontend_dir/package.json" ]]; then
    echo "❌ 未找到新版前端目录: $frontend_dir"
    exit 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "❌ 未检测到 npm，无法启动新版前端。"
    exit 1
  fi
  if [[ ! -d "$frontend_dir/node_modules" ]]; then
    echo "📦 检测到前端依赖未安装，正在执行 npm install..."
    (cd "$frontend_dir" && npm install)
  fi
}

print_access_urls() {
  local with_frontend="${1:-0}"
  local host="${START_LOCAL_HOST:-127.0.0.1}"
  echo "🔗 访问地址:"
  echo "   旧版页面: http://$host:$API_PORT/train-ui"
  if [[ "$with_frontend" == "1" ]]; then
    echo "   新版页面: http://$host:$FRONTEND_PORT/task-center"
  else
    echo "   新版页面: 未启动（使用 dual/all/frontend 模式）"
  fi
  echo "   API 文档: http://$host:$API_PORT/docs"
}

start_app() {
  echo "🚀 启动主应用: python -m src.main"
  python -m src.main
}

start_worker() {
  echo "🔧 启动 Celery Worker: python -m celery -A src.core.tasks worker --loglevel=info"
  python -m celery -A src.core.tasks worker --loglevel=info
}

start_frontend() {
  echo "🎨 启动新版前端: npm run dev -- --host 0.0.0.0 --port $FRONTEND_PORT"
  cd "$PROJECT_ROOT/src/frontend"
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
}

start_stack() {
  local start_frontend_bg="${1:-0}"
  local worker_pid
  local frontend_pid=""

  cleanup() {
    set +e
    [[ -n "${worker_pid:-}" ]] && kill "$worker_pid" >/dev/null 2>&1 || true
    [[ -n "${frontend_pid:-}" ]] && kill "$frontend_pid" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM

  echo "🔧 后台启动 Celery Worker..."
  python -m celery -A src.core.tasks worker --loglevel=info > logs/celery.log 2>&1 &
  worker_pid=$!
  echo "   Celery PID: $worker_pid (logs/celery.log)"

  if [[ "$start_frontend_bg" == "1" ]]; then
    echo "🎨 后台启动新版前端..."
    (
      cd "$PROJECT_ROOT/src/frontend"
      npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
    ) > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
    frontend_pid=$!
    echo "   Frontend PID: $frontend_pid (logs/frontend.log)"
  fi

  print_access_urls "$start_frontend_bg"
  echo "🌐 前台启动主应用（会自动拉起 Annotator 子进程）..."
  python -m src.main
}

run_smoke() {
  local smoke_script="$PROJECT_ROOT/scripts/e2e_task_center_smoke.sh"
  if [[ ! -f "$smoke_script" ]]; then
    echo "❌ 未找到冒烟脚本: $smoke_script"
    exit 1
  fi
  if [[ ! -x "$smoke_script" ]]; then
    chmod +x "$smoke_script"
  fi

  local api_base="${API_BASE:-http://127.0.0.1:${API_PORT:-8000}/api/v1}"
  local health_url="${api_base%/api/v1}/health"
  echo "🧪 运行 Task Center 冒烟验证..."
  echo "   API_BASE: $api_base"

  if ! command -v curl >/dev/null 2>&1; then
    echo "❌ 未检测到 curl，请先安装后重试。"
    exit 1
  fi
  if ! curl -fsS --max-time 3 "$health_url" >/dev/null; then
    echo "❌ 健康检查失败: $health_url"
    echo "   请先启动服务，例如:"
    echo "   ./scripts/start_local.sh all"
    exit 1
  fi

  API_BASE="$api_base" "$smoke_script"
}

load_env_file
ensure_runtime_env
prepare_runtime_dirs

case "$MODE" in
  app|worker|legacy|dual|all)
    ensure_security_env
    preflight_checks
    [[ "$MODE" == "dual" || "$MODE" == "all" ]] && ensure_frontend_ready
    ;;
  frontend)
    ensure_frontend_ready
    ;;
  smoke)
    ;;
  *)
    echo "用法: $0 [app|worker|legacy|dual|all|frontend|smoke]"
    exit 1
    ;;
esac

case "$MODE" in
  app)
    start_app
    ;;
  worker)
    start_worker
    ;;
  legacy)
    start_stack 0
    ;;
  dual)
    start_stack 1
    ;;
  all)
    start_stack 1
    ;;
  frontend)
    print_access_urls 1
    start_frontend
    ;;
  smoke)
    run_smoke
    ;;
esac
