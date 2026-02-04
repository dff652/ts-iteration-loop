#!/bin/bash
# TS-Iteration-Loop 开发环境一键搭建脚本
#
# 使用方法:
#   chmod +x scripts/setup_dev.sh
#   ./scripts/setup_dev.sh
#
# 选项:
#   --conda    使用 Conda 创建环境（默认）
#   --pip      仅使用 pip 安装依赖
#   --no-torch 跳过 PyTorch 安装

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认选项
USE_CONDA=true
INSTALL_TORCH=true
ENV_NAME="ts-iteration-loop"

# 解析参数
for arg in "$@"; do
    case $arg in
        --pip)
            USE_CONDA=false
            ;;
        --no-torch)
            INSTALL_TORCH=false
            ;;
        --help|-h)
            echo "Usage: $0 [--conda|--pip] [--no-torch]"
            echo ""
            echo "Options:"
            echo "  --conda     使用 Conda 创建环境（默认）"
            echo "  --pip       仅使用 pip 安装依赖到当前环境"
            echo "  --no-torch  跳过 PyTorch 安装"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  🚀 TS-Iteration-Loop 开发环境搭建  ${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════${NC}"
echo ""

cd "$PROJECT_ROOT"

# Step 1: 创建/更新环境
if $USE_CONDA; then
    echo -e "${YELLOW}[1/4] 检查 Conda 环境...${NC}"
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo -e "${GREEN}  ✓ 环境 '${ENV_NAME}' 已存在，更新中...${NC}"
        conda env update -f envs/environment.yml --prune
    else
        echo -e "${GREEN}  → 创建新环境 '${ENV_NAME}'...${NC}"
        conda env create -f envs/environment.yml
    fi
    
    echo -e "${YELLOW}  ⚠ 请手动激活环境: conda activate ${ENV_NAME}${NC}"
    
    # 获取环境 Python 路径
    PYTHON_PATH="$(conda run -n $ENV_NAME which python)"
else
    echo -e "${YELLOW}[1/4] 使用当前 Python 环境安装依赖...${NC}"
    pip install -r envs/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    PYTHON_PATH="$(which python)"
fi

# Step 2: 安装 PyTorch (可选)
if $INSTALL_TORCH; then
    echo ""
    echo -e "${YELLOW}[2/4] 检查 PyTorch...${NC}"
    
    # 检测 CUDA 版本
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        echo -e "${GREEN}  ✓ 检测到 NVIDIA 驱动: $CUDA_VERSION${NC}"
        
        if $USE_CONDA; then
            echo -e "${YELLOW}  → 安装 PyTorch (CUDA 12.4)...${NC}"
            conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple
        else
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        fi
    else
        echo -e "${YELLOW}  ⚠ 未检测到 NVIDIA GPU，安装 CPU 版本 PyTorch${NC}"
        if $USE_CONDA; then
            conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        else
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
else
    echo -e "${YELLOW}[2/4] 跳过 PyTorch 安装${NC}"
fi

# Step 3: 初始化数据库
echo ""
echo -e "${YELLOW}[3/4] 初始化数据库...${NC}"
mkdir -p data

if $USE_CONDA; then
    conda run -n $ENV_NAME python -c "from src.db.database import init_db; init_db(); print('  ✓ 数据库初始化完成')"
else
    python -c "from src.db.database import init_db; init_db(); print('  ✓ 数据库初始化完成')"
fi

# Step 4: 验证安装
echo ""
echo -e "${YELLOW}[4/4] 验证安装...${NC}"

VERIFY_CMD="
import sys
print(f'  Python: {sys.version}')

try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA 可用: {torch.cuda.is_available()}')
except ImportError:
    print('  PyTorch: 未安装')

try:
    import transformers
    print(f'  Transformers: {transformers.__version__}')
except ImportError:
    print('  Transformers: 未安装')

try:
    import gradio
    print(f'  Gradio: {gradio.__version__}')
except ImportError:
    print('  Gradio: 未安装')

try:
    import fastapi
    print(f'  FastAPI: {fastapi.__version__}')
except ImportError:
    print('  FastAPI: 未安装')

print()
print('  ✓ 环境验证完成！')
"

if $USE_CONDA; then
    conda run -n $ENV_NAME python -c "$VERIFY_CMD"
else
    python -c "$VERIFY_CMD"
fi

# 完成
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ 开发环境搭建完成！${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}启动应用:${NC}"
if $USE_CONDA; then
    echo "  conda activate $ENV_NAME"
fi
echo "  python -m src.main"
echo ""
echo -e "${BLUE}访问地址:${NC}"
echo "  API 文档: http://localhost:8000/docs"
echo "  管理界面: http://localhost:8000/train-ui"
echo ""

echo -e "${BLUE}环境模式:${NC}"
echo "  默认使用统一环境 (推荐)。"
echo "  如需使用旧版独立环境，请设置: export ENV_MODE=legacy"
echo ""
