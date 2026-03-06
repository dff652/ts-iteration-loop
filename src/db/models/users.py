"""
平台用户模型
"""
import hashlib
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text

from src.db.models.base import Base


class User(Base):
    """平台用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    display_name = Column(String(128), nullable=True)
    role = Column(String(32), nullable=False, default="annotator")  # admin / annotator / reviewer
    is_active = Column(Boolean, nullable=False, default=True)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    # ==================== 密码工具 ====================

    @staticmethod
    def hash_password(password: str) -> str:
        """SHA-256 哈希（与现有标注工具 manage_users.py 兼容）"""
        return "sha256:" + hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str) -> bool:
        """验证密码 - 支持 sha256 和 werkzeug pbkdf2 两种格式"""
        if self.password_hash.startswith("sha256:"):
            expected = self.password_hash[7:]
            actual = hashlib.sha256(password.encode()).hexdigest()
            return expected == actual
        # werkzeug 格式 (向后兼容)
        try:
            from werkzeug.security import check_password_hash
            return check_password_hash(self.password_hash, password)
        except ImportError:
            return False
