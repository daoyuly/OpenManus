import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """一个用于执行Python代码的工具，具有超时和安全限制。"""

    name: str = "python_execute"
    description: str = "执行Python代码字符串。注意：只有print输出可见，函数返回值不会被捕获。使用print语句来查看结果。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的Python代码。",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        使用超时限制执行提供的Python代码。

        参数:
            code (str): 要执行的Python代码。
            timeout (int): 执行超时时间（秒）。

        返回:
            Dict: 包含执行输出或错误消息的'output'和成功状态的'success'。
        """

        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # 超时进程
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"执行超时，超过 {timeout} 秒",
                    "success": False,
                }
            return dict(result)
