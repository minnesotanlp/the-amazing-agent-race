"""ReAct agent for solving AAR puzzles.

Uses OpenAI function calling with ToolRegistry for tool execution.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from mcp_servers.registry import ToolRegistry

logger = logging.getLogger(__name__)

REACT_SYSTEM = """\
You are a ReAct-style assistant solving a scavenger-hunt puzzle.
Use a Thought -> Action -> Observation loop to reason step by step.

1) Thought: explain what you will do next and why.
2) Action: if you need a tool, use function calling. Do NOT print tool JSON in text.
3) Observation: after a tool returns, reflect on the result in a new Thought.
4) Final Answer: when ready, write exactly:
   Final Answer: X
   where X is a single digit (0-9).

Guidelines:
- Always start with a Thought.
- Use tools via function calling, never by printing JSON.
- Follow the riddle clues in order.
- Keep track of extracted values for the final computation.
- Double-check your arithmetic before answering.
"""

_FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:\s*(\d)", re.IGNORECASE)


@dataclass
class AgentResult:
    """Result from a single puzzle solve attempt."""

    answer: int | None = None
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    hit_step_limit: bool = False


class ReActAgent:
    """ReAct agent using OpenAI function calling + ToolRegistry."""

    def __init__(
        self,
        llm_client: OpenAI,
        model: str,
        tool_registry: ToolRegistry,
        max_steps: int = 25,
    ):
        self._llm = llm_client
        self._model = model
        self._registry = tool_registry
        self._max_steps = max_steps

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        """Convert ToolRegistry specs to OpenAI function-calling schema."""
        tools: list[dict[str, Any]] = []
        for name, spec in sorted(self._registry.available_tools().items()):
            if not spec.availability.is_available:
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description or "",
                    "parameters": dict(spec.input_schema) if spec.input_schema else {
                        "type": "object",
                        "properties": {},
                    },
                },
            })
        return tools

    async def solve(self, prompt: str, *, max_steps: int | None = None) -> AgentResult:
        """Run the ReAct loop on a puzzle prompt."""
        limit = max_steps or self._max_steps
        openai_tools = self._build_openai_tools()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REACT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        result = AgentResult()
        result.transcript.append({"role": "user", "content": prompt})

        for step in range(limit):
            result.steps = step + 1

            try:
                call_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                }
                if openai_tools:
                    call_kwargs["tools"] = openai_tools
                    call_kwargs["tool_choice"] = "auto"
                completion = self._llm.chat.completions.create(**call_kwargs)
            except Exception as exc:
                logger.error("LLM call failed at step %d: %s", step + 1, exc)
                break

            msg = completion.choices[0].message
            text = msg.content or ""
            tool_calls = msg.tool_calls or []

            # Build assistant message for history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)
            result.transcript.append(assistant_msg)

            # Check for final answer in text
            if text:
                match = _FINAL_ANSWER_RE.search(text)
                if match:
                    result.answer = int(match.group(1))
                    result.final_text = text
                    return result

            # No tool calls and no final answer — model is done without answering
            if not tool_calls:
                result.final_text = text
                return result

            # Execute tool calls
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                logger.debug("  Tool call: %s(%s)", name, json.dumps(args)[:200])

                try:
                    spec = self._registry.available_tools().get(name)
                    if spec is None:
                        tool_result_text = f"Error: tool '{name}' not found"
                    else:
                        exec_result = await spec.execute(args)
                        tool_result_text = exec_result.output_text or "(empty result)"
                except Exception as exc:
                    tool_result_text = f"Error: {exc}"

                # Truncate very long results
                if len(tool_result_text) > 8000:
                    tool_result_text = tool_result_text[:8000] + "\n...[truncated]"

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_text,
                }
                messages.append(tool_msg)
                result.transcript.append(tool_msg)
                result.tool_calls.append({
                    "tool_name": name,
                    "arguments": args,
                    "result_preview": tool_result_text[:500],
                })

        result.hit_step_limit = True
        result.final_text = "(step limit reached)"
        return result
