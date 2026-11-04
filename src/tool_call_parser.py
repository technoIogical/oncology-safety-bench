from __future__ import annotations

import ast
import json
import re
from typing import Optional

from src.model_clients import extract_and_parse_json
from src.tools import TOOLS

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def _norm(obj) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name, args = obj.get("name"), obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if name not in TOOL_NAMES or not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


# Parses Gemma-style 'call:tool_name{arg: val}' tool invocations into standard (name, args) tuples.
def _parse_pseudo_call(text: str) -> Optional[dict]:
    text_content = text
    try:
        start_idx = text_content.find("call:")
        if start_idx != -1:
            func_name_end = text_content.find("{", start_idx)
            if func_name_end != -1:
                end_idx = text_content.find("}", func_name_end)

                if end_idx == -1:
                    end_idx = len(text_content)

                func_name = text_content[start_idx+5:func_name_end]
                args_str = text_content[func_name_end+1:end_idx]

                args_dict = {}
                if args_str:
                    if args_str.count('"') % 2 != 0:
                        args_str += '"'

                    stack = []
                    for char in args_str:
                        if char in "[{": stack.append(char)
                        elif char == "]" and stack and stack[-1] == "[": stack.pop()
                        elif char == "}" and stack and stack[-1] == "{": stack.pop()
                    for char in reversed(stack):
                        if char == "[": args_str += "]"
                        if char == "{": args_str += "}"

                    pairs = re.split(r',\s*(?![^\[]*\])(?=(?:[^"]*"[^"]*")*[^"]*$)', args_str)
                    for pair in pairs:
                        if ":" in pair:
                            k, v = pair.split(":", 1)
                            k = k.strip().strip('"').strip("'")
                            v = v.strip().strip('"').strip("'")

                            if v.startswith("[") and v.endswith("]"):
                                try:
                                    parsed_list = ast.literal_eval(v)
                                    if isinstance(parsed_list, list):
                                        v = parsed_list
                                except Exception:
                                    v = [item.strip().strip('"').strip("'") for item in v[1:-1].split(",")]

                            args_dict[k] = v

                return {"name": func_name, "arguments": args_dict}
    except Exception:
        pass
    return None


def parse_text_tool_call(text: str) -> Optional[dict]:
    if not text:
        return None
    if "call:" in text and "{" in text:
        r = _parse_pseudo_call(text)
        if r and r["name"] in TOOL_NAMES:
            return r
    return _norm(extract_and_parse_json(text))


def looks_like_tool_attempt(text: str) -> bool:
    s = (text or "").strip()
    if s.startswith(("{", "```json", "call:")):
        return True
    return "{" in s and any(f'"{n}"' in s for n in TOOL_NAMES)
