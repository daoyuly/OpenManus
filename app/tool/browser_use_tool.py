import asyncio
import json
from typing import Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch

_BROWSER_DESCRIPTION = """
与网页浏览器交互以执行各种操作，如导航、元素交互、内容提取和标签页管理。此工具提供全面的浏览器自动化功能：

导航：
- 'go_to_url'：在当前标签页中访问特定URL
- 'go_back'：返回上一页
- 'refresh'：刷新当前页面
- 'web_search'：在当前标签页中搜索查询，查询应该是像人类在网络上搜索那样的具体且不模糊或过长的查询。最好是单个最重要的项目。

元素交互：
- 'click_element'：通过索引点击元素
- 'input_text'：向表单元素输入文本
- 'scroll_down'/'scroll_up'：滚动页面（可选像素数量）
- 'scroll_to_text'：如果找不到要交互的内容，滚动到该内容
- 'send_keys'：发送特殊键字符串，如Escape、Backspace、Insert、PageDown、Delete、Enter，也支持快捷键如`Control+o`、`Control+Shift+T`。这用于keyboard.press。
- 'get_dropdown_options'：获取下拉菜单的所有选项
- 'select_dropdown_option'：通过要选择的选项文本为交互元素索引选择下拉选项

内容提取：
- 'extract_content'：提取页面内容以检索特定信息，例如所有公司名称、特定描述、所有相关信息、结构化格式的公司链接或简单链接

标签页管理：
- 'switch_tab'：切换到特定标签页
- 'open_tab'：使用URL打开新标签页
- 'close_tab'：关闭当前标签页

实用工具：
- 'wait'：等待指定的秒数
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "要执行的浏览器操作",
            },
            "url": {
                "type": "string",
                "description": "'go_to_url'或'open_tab'操作的URL",
            },
            "index": {
                "type": "integer",
                "description": "'click_element'、'input_text'、'get_dropdown_options'或'select_dropdown_option'操作的元素索引",
            },
            "text": {
                "type": "string",
                "description": "'input_text'、'scroll_to_text'或'select_dropdown_option'操作的文本",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "'scroll_down'或'scroll_up'操作要滚动的像素数（向下为正，向上为负）",
            },
            "tab_id": {
                "type": "integer",
                "description": "'switch_tab'操作的标签页ID",
            },
            "query": {
                "type": "string",
                "description": "'web_search'操作的搜索查询",
            },
            "goal": {
                "type": "string",
                "description": "'extract_content'操作的提取目标",
            },
            "keys": {
                "type": "string",
                "description": "'send_keys'操作要发送的键",
            },
            "seconds": {
                "type": "integer",
                "description": "'wait'操作要等待的秒数",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # 用于通用功能的上下文
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("参数不能为空")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """确保浏览器和上下文已初始化。"""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # 处理代理设置。
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # 如果配置中有上下文配置，使用它。
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        执行指定的浏览器操作。

        参数:
            action: 要执行的浏览器操作
            url: 导航或新标签页的URL
            index: 点击或输入操作的元素索引
            text: 输入操作或搜索查询的文本
            scroll_amount: 滚动操作的像素数
            tab_id: switch_tab操作的标签页ID
            query: Google搜索的搜索查询
            goal: 内容提取的提取目标
            keys: 键盘操作要发送的键
            seconds: 要等待的秒数
            **kwargs: 额外参数

        返回:
            包含操作输出或错误的ToolResult
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # 从配置获取最大内容长度
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # 导航操作
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="'go_to_url'操作需要URL"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)
                    await page.wait_for_load_state()
                    return ToolResult(output="成功导航到URL")

                elif action == "go_back":
                    page = await context.get_current_page()
                    await page.go_back()
                    return ToolResult(output="成功返回上一页")

                elif action == "refresh":
                    page = await context.get_current_page()
                    await page.reload()
                    return ToolResult(output="成功刷新页面")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="'web_search'操作需要搜索查询"
                        )
                    search_results = await self.web_search_tool.execute(query)

                    if search_results:
                        # 导航到第一个搜索结果
                        first_result = search_results[0]
                        if isinstance(first_result, dict) and "url" in first_result:
                            url_to_navigate = first_result["url"]
                        elif isinstance(first_result, str):
                            url_to_navigate = first_result
                        else:
                            return ToolResult(
                                error=f"搜索结果格式无效: {first_result}"
                            )

                        page = await context.get_current_page()
                        await page.goto(url_to_navigate)
                        await page.wait_for_load_state()

                        return ToolResult(
                            output=f"搜索 '{query}' 并导航到第一个结果: {url_to_navigate}\n所有结果:"
                            + "\n".join([str(r) for r in search_results])
                        )
                    else:
                        return ToolResult(
                            error=f"未找到 '{query}' 的搜索结果"
                        )

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(error="'click_element'操作需要元素索引")
                    page = await context.get_current_page()
                    elements = await self.dom_service.get_interactive_elements()
                    if 0 <= index < len(elements):
                        await elements[index].click()
                        return ToolResult(output=f"成功点击索引为{index}的元素")
                    return ToolResult(error=f"无效的元素索引: {index}")

                elif action == "input_text":
                    if index is None or text is None:
                        return ToolResult(error="'input_text'操作需要元素索引和文本")
                    page = await context.get_current_page()
                    elements = await self.dom_service.get_interactive_elements()
                    if 0 <= index < len(elements):
                        await elements[index].fill(text)
                        return ToolResult(output=f"成功向索引为{index}的元素输入文本")
                    return ToolResult(error=f"无效的元素索引: {index}")

                elif action == "scroll_down":
                    if scroll_amount is None:
                        scroll_amount = 100
                    page = await context.get_current_page()
                    await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                    return ToolResult(output=f"成功向下滚动{scroll_amount}像素")

                elif action == "scroll_up":
                    if scroll_amount is None:
                        scroll_amount = 100
                    page = await context.get_current_page()
                    await page.evaluate(f"window.scrollBy(0, -{scroll_amount})")
                    return ToolResult(output=f"成功向上滚动{scroll_amount}像素")

                elif action == "scroll_to_text":
                    if text is None:
                        return ToolResult(error="'scroll_to_text'操作需要文本")
                    page = await context.get_current_page()
                    await page.evaluate(f"""
                        const elements = Array.from(document.querySelectorAll('*'));
                        const element = elements.find(el => el.textContent.includes('{text}'));
                        if (element) element.scrollIntoView();
                    """)
                    return ToolResult(output=f"成功滚动到包含文本'{text}'的元素")

                elif action == "send_keys":
                    if keys is None:
                        return ToolResult(error="'send_keys'操作需要键")
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"成功发送键: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(error="'get_dropdown_options'操作需要元素索引")
                    page = await context.get_current_page()
                    elements = await self.dom_service.get_interactive_elements()
                    if 0 <= index < len(elements):
                        options = await elements[index].evaluate("""
                            el => {
                                if (el.tagName === 'SELECT') {
                                    return Array.from(el.options).map(opt => opt.text);
                                }
                                return [];
                            }
                        """)
                        return ToolResult(output=f"下拉选项: {options}")
                    return ToolResult(error=f"无效的元素索引: {index}")

                elif action == "select_dropdown_option":
                    if index is None or text is None:
                        return ToolResult(error="'select_dropdown_option'操作需要元素索引和选项文本")
                    page = await context.get_current_page()
                    elements = await self.dom_service.get_interactive_elements()
                    if 0 <= index < len(elements):
                        await elements[index].evaluate(f"""
                            el => {{
                                if (el.tagName === 'SELECT') {{
                                    const options = Array.from(el.options);
                                    const option = options.find(opt => opt.text === '{text}');
                                    if (option) el.selectedIndex = option.index;
                                }}
                            }}
                        """)
                        return ToolResult(output=f"成功选择选项: {text}")
                    return ToolResult(error=f"无效的元素索引: {index}")

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(error="'extract_content'操作需要提取目标")
                    page = await context.get_current_page()
                    content = await page.content()
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    return ToolResult(output=f"页面内容: {content}")

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(error="'switch_tab'操作需要标签页ID")
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"成功切换到标签页 {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="'open_tab'操作需要URL")
                    new_tab = await context.new_page()
                    await new_tab.goto(url)
                    return ToolResult(output=f"成功打开新标签页并导航到 {url}")

                elif action == "close_tab":
                    page = await context.get_current_page()
                    await page.close()
                    return ToolResult(output="成功关闭当前标签页")

                # Utility actions
                elif action == "wait":
                    if seconds is None:
                        return ToolResult(error="'wait'操作需要等待秒数")
                    await asyncio.sleep(seconds)
                    return ToolResult(output=f"成功等待 {seconds} 秒")

                else:
                    return ToolResult(error=f"未知的操作: {action}")

            except Exception as e:
                return ToolResult(error=f"执行操作时出错: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        获取当前浏览器状态作为ToolResult。
        如果未提供context，则使用self.context。
        """
        try:
            # 使用提供的context或回退到self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="浏览器上下文未初始化")

            state = await ctx.get_state()

            # 如果不存在则创建viewport_info字典
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # 为状态拍摄截图
            screenshot = await ctx.take_screenshot(full_page=True)

            # 构建包含所有必需字段的状态信息
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2] 等表示与列出的元素相对应的可点击索引。点击这些索引将导航到或与它们后面的相应内容交互。",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"获取浏览器状态失败: {str(e)}")

    async def cleanup(self):
        """清理浏览器资源。"""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """确保在对象被销毁时进行清理。"""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """工厂方法，用于创建具有特定上下文的BrowserUseTool。"""
        tool = cls()
        tool.tool_context = context
        return tool
