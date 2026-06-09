from __future__ import annotations

import re
from typing import Any

import numpy as np

from ml.schema import LANGUAGE_MAP, PromptFeatures

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
URL_PATTERN = re.compile(r"https?://[^\s,)'\"]+")
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>|\[tool_call\]|\"tool_calls\"|tool_call_id|function_call"
)
REASONING_TRIGGERS = [
    "explain", "reason", "think step by step", "analyze", "compare",
    "contrast", "why", "how does", "what if", "derive", "prove",
    "evaluate", "synthesize", "critique", "justify", "elaborate",
    "break down", "discuss", "interpret", "investigate",
]


def token_count(text: str) -> int:
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text) // 4


def has_code_blocks(text: str) -> bool:
    if CODE_BLOCK_PATTERN.search(text):
        return True
    code_keywords = [
        "def ", "class ", "import ", "from ", "return ",
        "function", "const ", "let ", "var ", "=>",
        "if __name__", "console.log", "print(", "int main",
        "<html", "<div", "public static", "extends ",
    ]
    return sum(1 for kw in code_keywords if kw in text) >= 2


def has_urls(text: str) -> bool:
    return bool(URL_PATTERN.search(text))


def has_images(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
        if isinstance(content, str) and re.search(
            r'!\[.*?\]\(.*?\)|<img[^>]*src=',
            content
        ):
            return True
    return False


def has_tool_calls(messages: list[dict]) -> bool:
    for msg in messages:
        if msg.get("tool_calls"):
            return True
        if msg.get("tool_call_id"):
            return True
        content = msg.get("content", "") or ""
        if isinstance(content, str) and TOOL_CALL_PATTERN.search(content):
            return True
    return False


def detect_dominant_language(text: str) -> str:
    if not text:
        return "unknown"
    text_lower = text.lower()
    code_score = 0
    code_keywords = {
        "def ", "class ", "import ", "fn ", "func", "function",
        "const ", "let ", "var ", "=>", "return ", "if ", "else ",
        "for ", "while ", "try:", "except", "with ", "async ",
        "await ", "print(", "console.", "export ", "interface ",
        "type ", "struct ", "impl ", "pub fn",
    }
    for kw in code_keywords:
        if kw in text_lower:
            code_score += 1
    if code_score >= 4:
        return "code"

    math_keywords = {
        "solve", "equation", "derivative", "integral", "calculate",
        "compute", "matrix", "vector", "theorem", "proof", "formula",
        "∑", "∫", "∂", "Δ", "√", "π", "algebra", "calculus",
    }
    if sum(1 for kw in math_keywords if kw in text_lower) >= 2:
        return "math"

    translation_indicators = [
        "translate", "translation", "translate to", "in french",
        "in spanish", "in german", "in chinese", "in japanese",
        "in russian", "en français", "en español",
    ]
    if any(ind in text_lower for ind in translation_indicators):
        return "translation"

    return "natural_language"


def compute_reasoning_complexity(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    score = 0.0
    for trigger in REASONING_TRIGGERS:
        count = text_lower.count(trigger)
        score += count * 0.15
    code_density = text.count("\n") / max(len(text), 1) * 100
    if code_density > 10:
        score += 0.2
    score += min(text.count("?") * 0.05, 0.3)
    complexity_keywords = [
        "comprehensive", "detailed", "thorough", "in-depth", "complex",
        "sophisticated", "multi-step", "hierarchical", "comparative",
        "exhaustive", "systematic", "rigorous", "elaborate",
    ]
    for kw in complexity_keywords:
        if kw in text_lower:
            score += 0.1
    return min(round(score, 2), 1.0)


def extract_features_from_messages(messages: list[dict]) -> PromptFeatures:
    full_text = " ".join(
        m.get("content") or "" for m in messages if isinstance(m.get("content"), str)
    )
    return PromptFeatures(
        token_count=token_count(full_text),
        char_length=len(full_text),
        has_code_blocks=has_code_blocks(full_text),
        has_urls=has_urls(full_text),
        has_images=has_images(messages),
        has_tool_calls=has_tool_calls(messages),
        dominant_language=detect_dominant_language(full_text),
        reasoning_complexity=compute_reasoning_complexity(full_text),
        hour_of_day=__import__("datetime").datetime.utcnow().hour,
    )


def features_to_vector(features: PromptFeatures) -> np.ndarray:
    lang_encoded = LANGUAGE_MAP.get(features.dominant_language, 4)
    return np.array([
        features.token_count,
        features.char_length,
        1.0 if features.has_code_blocks else 0.0,
        1.0 if features.has_urls else 0.0,
        1.0 if features.has_images else 0.0,
        1.0 if features.has_tool_calls else 0.0,
        lang_encoded,
        features.reasoning_complexity,
        features.hour_of_day,
    ], dtype=np.float64)


def feature_names() -> list[str]:
    return [
        "token_count",
        "char_length",
        "has_code_blocks",
        "has_urls",
        "has_images",
        "has_tool_calls",
        "dominant_language",
        "reasoning_complexity",
        "hour_of_day",
    ]


def extract_features_dict(data: dict[str, Any]) -> PromptFeatures:
    return PromptFeatures(
        token_count=data.get("token_count", 0),
        char_length=data.get("char_length", 0),
        has_code_blocks=data.get("has_code_blocks", False),
        has_urls=data.get("has_urls", False),
        has_images=data.get("has_images", False),
        has_tool_calls=data.get("has_tool_calls", False),
        dominant_language=data.get("dominant_language", "unknown"),
        reasoning_complexity=data.get("reasoning_complexity", 0.0),
        hour_of_day=data.get("hour_of_day", 0),
    )
