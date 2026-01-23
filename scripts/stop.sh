#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}正在停止 TS-Iteration-Loop 服务...${NC}"

# 函数：检查并在必要时杀死进程
kill_process() {
    local pattern=$1
    local name=$2
    
    # 查找进程 ID，排除 grep 自身
    pids=$(pgrep -f "$pattern")
    
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}发现 $name 进程 (PID: $(echo $pids | tr '\n' ' '))...${NC}"
        
        # 尝试优雅停止 (SIGTERM)
        echo "$pids" | xargs kill -15
        
        # 等待最多 5 秒
        for i in {1..5}; do
            if ! pgrep -f "$pattern" > /dev/null; then
                echo -e "${GREEN}✅ $name 已停止${NC}"
                return
            fi
            sleep 1
        done
        
        # 如果还在运行，强制杀死 (SIGKILL)
        if pgrep -f "$pattern" > /dev/null; then
            echo -e "${RED}⚠️ $name 未响应，正在强制停止...${NC}"
            pids=$(pgrep -f "$pattern")
            if [ -n "$pids" ]; then
                echo "$pids" | xargs kill -9
                echo -e "${GREEN}✅ $name 已强制停止${NC}"
            fi
        fi
    else
        echo -e "${GREEN}✅ $name 未运行${NC}"
    fi
}

# 1. 停止 API 主程序
kill_process "src.main" "TS-Iteration-Loop API"

# 2. 停止 Celery Worker
kill_process "celery worker" "Celery Worker"

# 3. 停止多余的 Uvicorn 进程 (如果有)
kill_process "uvicorn.*src.main:app" "Uvicorn Server"

echo -e "${GREEN}所有服务已停止。${NC}"
