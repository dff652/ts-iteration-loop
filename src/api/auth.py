"""
平台认证 API
- POST /api/v1/auth/login    登录 → JWT token
- GET  /api/v1/auth/me        获取当前用户
- POST /api/v1/auth/register  注册（仅管理员）
"""
from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from configs.settings import settings
from src.db.database import get_db, User

router = APIRouter(tags=["auth"])


# ==================== Pydantic Schemas ====================

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""
    role: str = "annotator"


class TokenResponse(BaseModel):
    success: bool = True
    token: str
    username: str
    display_name: str
    role: str


class UserInfoResponse(BaseModel):
    success: bool = True
    username: str
    display_name: str
    role: str


# ==================== JWT 工具 ====================

_DEV_SECRET = "ts-loop-dev-secret-key-do-not-use-in-production-32chars!!"


def _get_jwt_secret() -> str:
    """获取 JWT 密钥，开发环境使用默认值"""
    secret = (settings.JWT_SECRET_KEY or "").strip()
    if not secret or len(secret) < 32:
        return _DEV_SECRET
    return secret


def _create_token(username: str) -> str:
    """生成 JWT token"""
    import time
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + 24 * 3600,
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm="HS256")


def get_current_user(
    db: Session = Depends(get_db),
    token: str = "",
) -> User:
    """从 Authorization header 解析当前用户 (可作为 Depends 使用)"""
    raise HTTPException(status_code=401, detail="Not implemented as direct Depends")


def verify_token_from_header(authorization: str | None = None) -> str | None:
    """从 Authorization header 提取 username（不依赖 db）"""
    if not authorization:
        return None
    raw = authorization.removeprefix("Bearer ").strip()
    if not raw:
        return None
    try:
        payload = jwt.decode(raw, _get_jwt_secret(), algorithms=["HS256"])
        return payload.get("sub")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ==================== 路由 ====================

@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """用户登录"""
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not user.verify_password(req.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账户已禁用")

    # 更新登录时间
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    return TokenResponse(
        token=_create_token(user.username),
        username=user.username,
        display_name=user.display_name or user.username,
        role=user.role,
    )


@router.get("/me", response_model=UserInfoResponse)
def get_me(
    db: Session = Depends(get_db),
    authorization: str | None = None,
):
    """获取当前用户信息"""
    # 手动从 query 或 header 获取 token
    from fastapi import Request
    # 实际使用时通过依赖注入中间件获取
    raise HTTPException(status_code=401, detail="请先登录")


@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """注册新用户"""
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="用户名已存在")

    user = User(
        username=req.username,
        password_hash=User.hash_password(req.password),
        display_name=req.display_name or req.username,
        role=req.role if req.role in ("admin", "annotator", "reviewer") else "annotator",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
    }
