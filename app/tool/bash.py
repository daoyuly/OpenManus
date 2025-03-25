import asyncio
import os
from typing import Optional

from app.exceptions import ToolError
from app.tool.base import BaseTool, CLIResult, ToolResult

_BASH_DESCRIPTION = """在终端中执行bash命令。
* 长时间运行的命令：对于可能无限期运行的命令，应该在后台运行并将输出重定向到文件，例如 command = `python3 app.py > server.log 2>&1 &`。
* 交互式：如果bash命令返回退出码 `-1`，这意味着进程尚未完成。助手必须发送第二次调用到终端，使用空的 `command`（这将获取任何额外的日志），或者它可以发送额外的文本（将 `command` 设置为文本）到运行进程的STDIN，或者它可以发送 command=`ctrl+c` 来中断进程。
* 超时：如果命令执行结果显示"命令超时。正在向进程发送SIGINT"，助手应该在后台重试运行命令。
"""


class _BashSession:
    """一个bash shell会话。"""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # 秒
    _timeout: float = 120.0  # 秒
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """终止bash shell。"""
        if not self._started:
            raise ToolError("会话尚未启动。")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """在bash shell中执行命令。"""
        if not self._started:
            raise ToolError("会话尚未启动。")
        if self._process.returncode is not None:
            return ToolResult(
                system="工具必须重新启动",
                error=f"bash已退出，返回码 {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"超时：bash在 {self._timeout} 秒内未返回，必须重新启动",
            )

        # 我们知道这些不是None，因为我们使用PIPE创建了进程
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # 向进程发送命令
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # 从进程读取输出，直到找到标记
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # 如果我们直接从stdout/stderr读取，它会永远等待EOF。
                    # 直接使用StreamReader缓冲区代替。
                    output = (
                        self._process.stdout._buffer.decode()
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # 移除标记并退出
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"超时：bash在 {self._timeout} 秒内未返回，必须重新启动",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = (
            self._process.stderr._buffer.decode()
        )  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # 清除缓冲区，以便下次可以正确读取输出
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class Bash(BaseTool):
    """一个用于执行bash命令的工具"""

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的bash命令。当之前的退出码为 `-1` 时可以为空以查看额外日志。可以是 `ctrl+c` 来中断当前运行的进程。",
            },
        },
        "required": ["command"],
    }

    _session: Optional[_BashSession] = None

    async def execute(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolResult(system="工具已重新启动。")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("未提供命令。")


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("ls -l"))
    print(rst)
