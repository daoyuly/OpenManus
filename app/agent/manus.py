import json
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor

initial_working_directory = Path(os.getcwd()) / "workspace"


class Manus(ToolCallAgent):
    """
    一个使用规划来解决各种任务的通用代理。

    这个代理扩展了PlanningAgent，具有全面的工具和功能集，
    包括Python执行、网页浏览、文件操作和信息检索，
    可以处理广泛的用户请求。
    """

    name: str = "Manus"
    description: str = (
        "一个可以使用多种工具解决各种任务的通用代理"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=initial_working_directory)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # 向工具集合添加通用工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), BrowserUseTool(), StrReplaceEditor(), Terminate()
        )
    )

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return
        else:
            await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
            await super()._handle_special_tool(name, result, **kwargs)

    async def get_browser_state(self) -> Optional[dict]:
        """获取当前浏览器状态以用于下一步的上下文。"""
        browser_tool = self.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool:
            return None

        try:
            # 直接从工具获取浏览器状态，不带上下文参数
            result = await browser_tool.get_current_state()

            if result.error:
                logger.debug(f"浏览器状态错误: {result.error}")
                return None

            # 如果可用，存储截图
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

            # 解析状态信息
            return json.loads(result.output)

        except Exception as e:
            logger.debug(f"获取浏览器状态失败: {str(e)}")
            return None

    async def think(self) -> bool:
        # 在这里添加自定义预处理
        browser_state = await self.get_browser_state()

        # 临时修改next_step_prompt
        original_prompt = self.next_step_prompt
        if browser_state and not browser_state.get("error"):
            self.next_step_prompt += f"\n当前浏览器状态:\nURL: {browser_state.get('url', 'N/A')}\n标题: {browser_state.get('title', 'N/A')}\n"

        # 调用父类实现
        result = await super().think()

        # 恢复原始提示词
        self.next_step_prompt = original_prompt

        return result
