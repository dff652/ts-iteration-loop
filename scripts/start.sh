#!/bin/bash
# 本地开发启动脚本

set -e

echo "======================================"
echo "TS-Iteration-Loop 启动脚本"
echo "======================================"

# 检查 Redis
if ! command -v redis-cli &> /dev/null; then
    echo "⚠️  Redis 未安装，Celery 任务队列将不可用"
    echo "   请运行: sudo apt install redis-server"
else
    # 检查 Redis 服务
    if ! redis-cli ping &> /dev/null; then
        echo "🔄 启动 Redis..."
        redis-server --daemonize yes
    else
        echo "✅ Redis 已运行"
    fi
fi

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  建议在虚拟环境中运行"
fi

# 安装依赖
echo "📦 检查依赖..."
pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建数据目录
mkdir -p data

# 启动方式选择
MODE=${1:-app}

case $MODE in
    app)
        echo "🚀 启动主应用..."
        python -m src.main
        ;;
    worker)
        echo "🔧 启动 Celery Worker..."
        celery -A src.core.tasks worker --loglevel=info
        ;;
    all)
        echo "🚀 启动所有服务..."
        # 后台启动 Celery Worker
        celery -A src.core.tasks worker --loglevel=info &
        CELERY_PID=$!
        echo "   Celery Worker PID: $CELERY_PID"
        
        # 前台启动主应用
        python -m src.main
        
        # 清理
        kill $CELERY_PID 2>/dev/null
        ;;
    *)
        echo "用法: $0 [app|worker|all]"
        echo "  app    - 仅启动主应用"
        echo "  worker - 仅启动 Celery Worker"
        echo "  all    - 启动所有服务"
        exit 1
        ;;
esac
