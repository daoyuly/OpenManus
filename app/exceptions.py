class ToolError(Exception):
    """当工具遇到错误时抛出"""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """OpenManus所有错误的基类"""


class TokenLimitExceeded(OpenManusError):
    """当超出令牌限制时抛出的异常"""
