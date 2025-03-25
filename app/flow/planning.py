import json
import time
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow, PlanStepStatus
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanningFlow(BaseFlow):
    """使用代理管理任务规划和执行的流程"""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # 在super().__init__之前设置执行器键
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 如果提供了计划ID则设置
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 如果未提供则初始化规划工具
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 使用处理后的数据调用父类的init
        super().__init__(agents, **data)

        # 如果未指定执行器键，则设置为所有代理键
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        获取当前步骤的适当执行器代理。
        可以扩展为基于步骤类型/要求选择代理。
        """
        # 如果提供了步骤类型且匹配代理键，使用该代理
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 否则使用第一个可用的执行器或回退到主代理
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 回退到主代理
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """使用代理执行规划流程"""
        try:
            if not self.primary_agent:
                raise ValueError("没有可用的主代理")

            # 如果提供了输入则创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 验证计划是否成功创建
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"计划创建失败。在规划工具中未找到计划ID {self.active_plan_id}。"
                    )
                    return f"为以下内容创建计划失败: {input_text}"

            result = ""
            while True:
                # 获取要执行的当前步骤
                self.current_step_index, step_info = await self._get_current_step_info()

                # 如果没有更多步骤或计划完成则退出
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 使用适当的代理执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 检查代理是否想要终止
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"PlanningFlow中出错: {str(e)}")
            return f"执行失败: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """使用流程的LLM和PlanningTool基于请求创建初始计划"""
        logger.info(f"创建ID为 {self.active_plan_id} 的初始计划")

        # 创建用于计划创建的系统消息
        system_message = Message.system_message(
            "你是一个规划助手。创建一个简洁、可执行的计划，包含清晰的步骤。"
            "关注关键里程碑而不是详细的子步骤。"
            "优化清晰度和效率。"
        )

        # 创建包含请求的用户消息
        user_message = Message.user_message(
            f"创建一个合理的计划，包含清晰的步骤来完成以下任务: {request}"
        )

        # 使用PlanningTool调用LLM
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 处理工具调用（如果存在）
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析参数
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"解析工具参数失败: {args}")
                            continue

                    # 确保正确设置plan_id并执行工具
                    args["plan_id"] = self.active_plan_id

                    # 通过ToolCollection而不是直接执行工具
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"计划创建结果: {str(result)}")
                    return

        # 如果执行到达这里，创建默认计划
        logger.warning("创建默认计划")

        # 使用ToolCollection创建默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"计划: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["分析请求", "执行任务", "验证结果"],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        解析当前计划以识别第一个未完成步骤的索引和信息。
        如果未找到活动步骤，返回 (None, None)。
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"未找到ID为 {self.active_plan_id} 的计划")
            return None, None

        try:
            # 直接从规划工具存储访问计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤类型/类别（如果可用）
                    step_info = {"text": step}

                    # 尝试从文本中提取步骤类型（例如，[SEARCH]或[CODE]）
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 将当前步骤标记为进行中
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"将步骤标记为进行中时出错: {e}")
                        # 如果需要，直接更新步骤状态
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # 未找到活动步骤

        except Exception as e:
            logger.warning(f"查找当前步骤索引时出错: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """使用agent.run()通过指定代理执行当前步骤"""
        # 使用当前计划状态为代理准备上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"步骤 {self.current_step_index}")

        # 创建提示供代理执行当前步骤
        step_prompt = f"""
        当前计划状态:
        {plan_status}

        你的当前任务:
        你现在正在处理步骤 {self.current_step_index}: "{step_text}"

        请使用适当的工具执行此步骤。完成后，提供你完成内容的摘要。
        """

        # 使用agent.run()执行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 成功执行后将步骤标记为已完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"执行步骤时出错: {e}")
            return f"步骤执行失败: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """将当前步骤标记为已完成"""
        if self.current_step_index is not None:
            try:
                await self.planning_tool.execute(
                    command="mark_step",
                    plan_id=self.active_plan_id,
                    step_index=self.current_step_index,
                    step_status=PlanStepStatus.COMPLETED.value,
                )
            except Exception as e:
                logger.warning(f"将步骤标记为已完成时出错: {e}")

    async def _get_plan_text(self) -> str:
        """获取当前计划的文本表示"""
        try:
            # 尝试从规划工具获取计划文本
            result = await self.planning_tool.execute(
                command="get_plan_text",
                plan_id=self.active_plan_id,
            )
            if result and not result.error:
                return result.output
        except Exception as e:
            logger.warning(f"从规划工具获取计划文本时出错: {e}")

        # 如果从工具获取失败，从存储生成
        return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """从规划工具存储生成计划文本"""
        if not self.active_plan_id or self.active_plan_id not in self.planning_tool.plans:
            return "未找到计划"

        try:
            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "未命名计划")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 获取状态标记
            status_marks = PlanStepStatus.get_status_marks()

            # 构建计划文本
            plan_text = f"计划: {title}\n\n步骤:\n"
            for i, step in enumerate(steps):
                # 获取步骤状态标记
                status = (
                    step_statuses[i]
                    if i < len(step_statuses)
                    else PlanStepStatus.NOT_STARTED.value
                )
                status_mark = status_marks.get(status, "[ ]")

                # 添加步骤到文本
                plan_text += f"{status_mark} {step}\n"

            return plan_text
        except Exception as e:
            logger.warning(f"从存储生成计划文本时出错: {e}")
            return "生成计划文本时出错"

    async def _finalize_plan(self) -> str:
        """完成计划并返回摘要"""
        try:
            # 获取最终计划状态
            plan_status = await self._get_plan_text()

            # 创建完成消息
            completion_message = f"""
            计划完成！

            最终计划状态:
            {plan_status}

            所有步骤已完成。计划已成功执行。
            """

            return completion_message
        except Exception as e:
            logger.warning(f"完成计划时出错: {e}")
            return "完成计划时出错"
