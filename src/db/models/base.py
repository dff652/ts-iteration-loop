"""
ORM 模型基类
"""
from sqlalchemy.orm import declarative_base

# 基类，所有具体的模型都会继承它
Base = declarative_base()
