import os

import aiofiles

from app.config import WORKSPACE_ROOT
from app.tool.base import BaseTool


class FileSaver(BaseTool):
    name: str = "file_saver"
    description: str = """将内容保存到指定路径的本地文件。
当你需要将文本、代码或生成的内容保存到本地文件系统时使用此工具。
该工具接受内容和文件路径，并将内容保存到该位置。
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "（必需）要保存到文件的内容。",
            },
            "file_path": {
                "type": "string",
                "description": "（必需）文件应保存的路径，包括文件名和扩展名。",
            },
            "mode": {
                "type": "string",
                "description": "（可选）文件打开模式。默认为'w'表示写入。使用'a'表示追加。",
                "enum": ["w", "a"],
                "default": "w",
            },
        },
        "required": ["content", "file_path"],
    }

    async def execute(self, content: str, file_path: str, mode: str = "w") -> str:
        """
        将内容保存到指定路径的文件。

        参数:
            content (str): 要保存到文件的内容。
            file_path (str): 文件应保存的路径。
            mode (str, 可选): 文件打开模式。默认为'w'表示写入。使用'a'表示追加。

        返回:
            str: 表示操作结果的消息。
        """
        try:
            # 将生成的文件放在工作区目录中
            if os.path.isabs(file_path):
                file_name = os.path.basename(file_path)
                full_path = os.path.join(WORKSPACE_ROOT, file_name)
            else:
                full_path = os.path.join(WORKSPACE_ROOT, file_path)

            # 确保目录存在
            directory = os.path.dirname(full_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # 直接写入文件
            async with aiofiles.open(full_path, mode, encoding="utf-8") as file:
                await file.write(content)

            return f"内容已成功保存到 {full_path}"
        except Exception as e:
            return f"保存文件时出错: {str(e)}"
