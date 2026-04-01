#!/usr/bin/env python3
"""MCP server providing Python code generation and execution tools."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import anyio
import click
import mcp.types as types
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from openai import AsyncOpenAI

load_dotenv()

ContentList = list[types.ContentBlock]


@lru_cache(maxsize=None)
def _cached_client(api_key: str, base_url: str | None) -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def _json_content(payload: Any) -> ContentList:
    return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]


class PythonCodeTools:
    """Utilities for generating Python code via OpenAI-compatible APIs."""

    def __init__(self) -> None:
        remote_api_key = os.getenv("OPENAI_API_KEY")
        local_api_key = os.getenv("LOCAL_API_KEY")
        if not remote_api_key and not local_api_key:
            raise RuntimeError(
                "Either OPENAI_API_KEY or LOCAL_API_KEY must be set for code tools."
            )
        self._remote_api_key = remote_api_key
        self._remote_base_url = os.getenv("OPENAI_BASE_URL")
        self._default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self._force_local = os.getenv("AI_LANGUAGE_USE_LOCAL") == "1"
        self._local_base_url = os.getenv("LOCAL_LANGUAGE_BASE_URL")
        self._local_model = os.getenv("LOCAL_LANGUAGE_MODEL")
        self._local_api_key = local_api_key or remote_api_key

        if self._force_local and self._local_model:
            self._default_model = self._local_model

    def _client(self) -> AsyncOpenAI:
        if self._force_local:
            if not self._local_base_url:
                raise RuntimeError(
                    "LOCAL_LANGUAGE_BASE_URL must be set when AI_LANGUAGE_USE_LOCAL=1"
                )
            if not self._local_api_key:
                raise RuntimeError(
                    "LOCAL_API_KEY must be set when AI_LANGUAGE_USE_LOCAL=1"
                )
            return _cached_client(self._local_api_key, self._local_base_url)
        if not self._remote_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set when AI_LANGUAGE_USE_LOCAL!=1"
            )
        return _cached_client(self._remote_api_key, self._remote_base_url)

    async def generate_python(
        self,
        description: str,
        model: str | None,
        extra_instructions: str | None,
    ) -> str:
        client = self._client()
        default_model = self._default_model
        if self._force_local and self._local_model:
            default_model = self._local_model
        chosen_model = model or default_model
        system_prompt = (
            "You are an expert Python 3 engineer. Produce a standalone, executable script that "
            "solves the user's request. The code must be self-contained, rely only on the Python "
            "standard library unless explicitly permitted, and include helpful entry points such "
            'as "if __name__ == \'__main__\':" when applicable. Return only the Python code with no '
            "additional commentary or markdown fencing."
        )

        user_sections = [description.strip()]
        if extra_instructions:
            user_sections.append(f"Additional guidance:\n{extra_instructions.strip()}")
        user_prompt = "\n\n".join(user_sections).strip()

        response = await client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        message_content = response.choices[0].message.content
        if isinstance(message_content, list):
            text = "".join(part.get("text", "") for part in message_content).strip()
        else:
            text = (message_content or "").strip()
        return self._strip_code_fences(text)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if not lines:
            return text
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)


class PythonExecutor:
    """Execute Python source code and capture stdout/stderr."""

    def __init__(self) -> None:
        self._python_executable = os.getenv("PYTHON_EXECUTABLE", "python3")

    async def run(self, code: str, timeout: float | None) -> dict[str, Any]:
        if not code.strip():
            raise ValueError("'code' must be a non-empty string")

        tmp_path = self._write_temp_script(code)
        try:
            run_coro = anyio.run_process(
                [self._python_executable, str(tmp_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if timeout is not None:
                if timeout <= 0:
                    raise ValueError("'timeout_seconds' must be positive when provided")
                try:
                    with anyio.fail_after(timeout):
                        result = await run_coro
                except TimeoutError as exc:
                    raise RuntimeError(
                        f"Python execution timed out after {timeout} seconds"
                    ) from exc
            else:
                result = await run_coro
        finally:
            tmp_path.unlink(missing_ok=True)

        stdout = (result.stdout or b"").decode("utf-8", errors="replace")
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
        }

    @staticmethod
    def _write_temp_script(code: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(code)
            handle.flush()
            return Path(handle.name)


def _build_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="python_generate_code",
            title="Python Code Generator",
            description=(
                "Generate standalone Python 3 code that fulfils the provided description using an OpenAI-compatible model. "
                "The response is raw Python (no markdown fences) so it can be written straight to disk or passed into python_execute_code."
            ),
            inputSchema={
                "type": "object",
                "required": ["description"],
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Problem statement or functionality the script should implement.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model override for the OpenAI-compatible client.",
                    },
                    "extra_instructions": {
                        "type": "string",
                        "description": "Additional constraints or notes for the generated script.",
                    },
                },
            },
        ),
        types.Tool(
            name="python_execute_code",
            title="Python Code Executor",
            description=(
                "Execute provided Python code locally and capture stdout/stderr output. "
                "Returns JSON text containing stdout, stderr, and exit_code fields."
            ),
            inputSchema={
                "type": "object",
                "required": ["code"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Optional timeout for execution (seconds).",
                        "minimum": 0.1,
                    },
                },
            },
        ),
    ]


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    show_default=True,
)
@click.option(
    "--host", default="127.0.0.1", show_default=True, help="Host for SSE transport"
)
@click.option("--port", default=8000, show_default=True, help="Port for SSE transport")
def main(transport: str, host: str, port: int) -> int:
    tools_impl = PythonCodeTools()
    executor = PythonExecutor()
    tools = _build_tools()
    server = Server("mcp-python-tools")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> ContentList:
        args = arguments or {}

        if name == "python_generate_code":
            description = str(args.get("description") or "").strip()
            if not description:
                raise ValueError("'description' must be a non-empty string")
            model = args.get("model")
            if model is not None and not isinstance(model, str):
                raise ValueError("'model' must be a string when provided")
            extra_instructions = args.get("extra_instructions")
            if extra_instructions is not None and not isinstance(extra_instructions, str):
                raise ValueError("'extra_instructions' must be a string when provided")

            code = await tools_impl.generate_python(
                description,
                model if isinstance(model, str) else None,
                extra_instructions,
            )
            return [types.TextContent(type="text", text=code)]

        if name == "python_execute_code":
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("'code' must be a non-empty string")
            timeout = args.get("timeout_seconds")
            if timeout is not None:
                try:
                    timeout_value = float(timeout)
                except (TypeError, ValueError) as exc:
                    raise ValueError("'timeout_seconds' must be numeric when provided") from exc
            else:
                timeout_value = None

            result = await executor.run(code, timeout_value)
            return _json_content(result)

        raise ValueError(f"Unknown tool: {name}")

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:  # type: ignore[attr-defined]
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=False,
            routes=[
                Route("/messages/", handle_sse, methods=["GET"]),
                Mount("/", app=sse.app),
            ],
        )

        import uvicorn

        uvicorn.run(starlette_app, host=host, port=port)
        return 0

    else:
        from mcp.server.stdio import stdio_server  # type: ignore

        async def run_stdio() -> None:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )

        anyio.run(run_stdio)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
