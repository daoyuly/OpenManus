from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """使用给定参数执行工具。"""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """使用给定参数执行工具。"""

    def to_param(self) -> Dict:
        """将工具转换为函数调用格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """表示工具执行的结果。"""

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("无法合并工具结果")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        return f"错误: {self.error}" if self.error else self.output

    def replace(self, **kwargs):
        """返回一个新的ToolResult，其中指定的字段被替换。"""
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """一个可以作为CLI输出渲染的ToolResult。"""


class ToolFailure(ToolResult):
    """一个表示失败的ToolResult。"""
