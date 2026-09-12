"""The agent loop: sends the user's message to Claude with tool definitions,
executes any tool calls Claude makes, feeds results back, and repeats until
Claude returns a final text answer.
"""
import json

import anthropic

from app.config import settings
from app.agent.tools import TOOL_DEFINITIONS, TOOL_IMPLEMENTATIONS

SYSTEM_PROMPT = """You are a research copilot for a single user's personal library \
of papers, articles, and notes. You have tools to search that library, search \
the web, and summarize the library's contents.

When answering:
- Search the library first for anything that might be in it.
- If the library doesn't have what's needed, say so plainly, then use web search \
if it's available and relevant.
- Cite which document(s) your answer draws on by title.
- If related past queries are returned alongside search results, mention them \
briefly when they're genuinely relevant — this is what makes you a research \
memory rather than a one-shot search tool.
- Be direct. Don't pad answers with unnecessary hedging or repetition.
"""


class ResearchAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def run(self, user_message: str, history: list[dict] = None, max_turns: int = 5) -> dict:
        messages = list(history) if history else []
        messages.append({"role": "user", "content": user_message})

        tool_calls_made = []

        for _ in range(max_turns):
            response = self.client.messages.create(
                model=settings.generation_model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return {
                    "answer": final_text,
                    "tool_calls": tool_calls_made,
                    "messages": messages + [{"role": "assistant", "content": response.content}],
                }

            # Claude wants to use one or more tools — execute them and feed
            # results back in the same conversation.
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_name = block.name
                tool_input = block.input
                tool_calls_made.append({"tool": tool_name, "input": tool_input})

                try:
                    impl = TOOL_IMPLEMENTATIONS[tool_name]
                    result = impl(**tool_input)
                    result_content = json.dumps(result)
                    is_error = False
                except Exception as e:
                    result_content = f"Tool error: {e}"
                    is_error = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})

        return {
            "answer": "Reached max tool-use turns without a final answer. Try a narrower query.",
            "tool_calls": tool_calls_made,
            "messages": messages,
        }
