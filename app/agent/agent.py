"""The agent loop, powered by Google's Gemini API (free tier).

Gemini's SDK supports "automatic function calling": you hand it plain
Python functions (with docstrings and type hints), and it inspects them,
decides when to call them, executes them, and feeds results back into the
conversation automatically. This is simpler than manually building a
tool-use loop, at the cost of a little visibility into individual tool
calls (Gemini handles that internally).
"""
import google.generativeai as genai

from app.config import settings
from app.agent.tools import search_library, search_web, library_summary

genai.configure(api_key=settings.gemini_api_key)

SYSTEM_PROMPT = """You are a research copilot for a single user's personal library \
of papers, articles, and notes. You have tools to search that library, search \
the web, and summarize the library's contents.

When answering:
- Search the library first for anything that might be in it.
- If the library doesn't have what's needed, say so plainly. If search_web \
returns an error saying it isn't connected yet, tell the user that instead \
of pretending you searched the web.
- Cite which document(s) your answer draws on by title.
- If related past queries are returned alongside search results, mention them \
briefly when they're genuinely relevant — this is what makes you a research \
memory rather than a one-shot search tool.
- Be direct. Don't pad answers with unnecessary hedging or repetition.
"""


class ResearchAgent:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=settings.generation_model,
            tools=[search_library, search_web, library_summary],
            system_instruction=SYSTEM_PROMPT,
        )

    def run(self, user_message: str, history: list = None) -> dict:
        chat = self.model.start_chat(
            history=history or [],
            enable_automatic_function_calling=True,
        )
        response = chat.send_message(user_message)

        # Pull out which tools were actually called, by scanning the chat
        # history for function_call parts Gemini added during this turn.
        tool_calls = []
        for msg in chat.history:
            for part in getattr(msg, "parts", []):
                fc = getattr(part, "function_call", None)
                if fc is not None and fc.name:
                    tool_calls.append({"tool": fc.name, "input": dict(fc.args)})

        return {
            "answer": response.text,
            "tool_calls": tool_calls,
            "history": chat.history,
        }
