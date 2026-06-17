"""Content-aware prompt complexity analysis (Phase 1 & 2).

Separates three dimensions:
- context_load: how much the model must read (size)
- task_difficulty: how hard the cognitive work is (content)
- requirement_load: output constraints and strict formatting demands

Routing compares task_difficulty (+ requirement bump) against model max_complexity_score.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Task types recognised by the rule classifier
TASK_CHITCHAT = "chitchat"
TASK_EXTRACTION = "extraction"
TASK_SUMMARIZATION = "summarization"
TASK_CODE_EDIT = "code_edit"
TASK_DEBUG = "debug"
TASK_PLANNING = "planning"
TASK_MULTI_STEP = "multi_step_research"
TASK_GENERAL = "general"
TASK_UNKNOWN = "unknown"

TASK_DIFFICULTY_PRIORS: dict[str, float] = {
    TASK_CHITCHAT: 0.08,
    TASK_EXTRACTION: 0.18,
    TASK_SUMMARIZATION: 0.22,
    TASK_GENERAL: 0.22,
    TASK_CODE_EDIT: 0.48,
    TASK_DEBUG: 0.62,
    TASK_PLANNING: 0.68,
    TASK_MULTI_STEP: 0.72,
    TASK_UNKNOWN: 0.25,
}

# (task_type, pattern, score) — highest total score wins
_TASK_CLASSIFICATION_RULES: list[tuple[str, re.Pattern[str], float]] = [
    (TASK_DEBUG, re.compile(
        r"\b(debug|fix\s+the\s+bug|race\s+condition|stack\s+trace|root\s+cause|"
        r"why\s+(is|does|did)\s+.+\s+(fail|break|error)|trace\s+through)\b",
        re.I,
    ), 3.0),
    (TASK_DEBUG, re.compile(r"\b(bug|broken|not\s+working|fails?\s+with|exception)\b", re.I), 1.5),
    (TASK_PLANNING, re.compile(
        r"\b(design|architect|migration\s+plan|roadmap|strategy|trade[\s-]?offs?|"
        r"system\s+design|refactor\s+.+\s+to\s+use)\b",
        re.I,
    ), 2.5),
    (TASK_PLANNING, re.compile(r"\b(implement\s+.+\s+with\s+.+\s+and\s+)", re.I), 2.0),
    (TASK_MULTI_STEP, re.compile(
        r"\b(research|investigate|compare\s+.+\s+and\s+.+\s+then|step[\s-]?by[\s-]?step|"
        r"multi[\s-]?step|first\s+.+\s+then\s+.+\s+finally)\b",
        re.I,
    ), 2.5),
    (TASK_CODE_EDIT, re.compile(
        r"\b(implement|write\s+(a\s+)?function|add\s+(a\s+)?method|create\s+(a\s+)?class|"
        r"refactor|rewrite|port\s+to|convert\s+to)\b",
        re.I,
    ), 2.0),
    (TASK_CODE_EDIT, re.compile(r"\b(pull\s+request|code\s+review)\b", re.I), 1.5),
    (TASK_SUMMARIZATION, re.compile(
        r"\b(summarize|summary|tl;dr|tldr|brief\s+overview|key\s+points)\b", re.I,
    ), 2.5),
    (TASK_EXTRACTION, re.compile(
        r"\b(extract|list\s+all|find\s+all|pull\s+out|parse\s+.+\s+from|"
        r"what\s+are\s+the\s+.+\s+in)\b",
        re.I,
    ), 2.0),
    (TASK_CHITCHAT, re.compile(
        r"^(hi|hello|hey|thanks|thank\s+you|ok|okay|yes|no)[!.?\s]*$", re.I,
    ), 3.0),
    (TASK_CHITCHAT, re.compile(
        r"\b(how\s+are\s+you|good\s+morning|good\s+afternoon)\b", re.I,
    ), 2.0),
]

_REASONING_TRIGGERS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bthink\s+step\s+by\s+step\b", re.I), 0.20),
    (re.compile(r"\b(prove|derive|justify)\b", re.I), 0.18),
    (re.compile(r"\b(analyze|analyse|evaluate|synthesize|critique)\b", re.I), 0.15),
    (re.compile(r"\b(compare|contrast)\b", re.I), 0.12),
    (re.compile(r"\bhow\s+does\b", re.I), 0.12),
    (re.compile(r"\bwhat\s+if\b", re.I), 0.12),
    (re.compile(r"\b(explain|reason)\b", re.I), 0.06),
    (re.compile(r"\bwhy\b", re.I), 0.05),
]

_COMPLEXITY_KEYWORDS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(in[\s-]?depth|comprehensive|exhaustive|rigorous)\b", re.I), 0.10),
    (re.compile(r"\b(thorough|detailed|sophisticated|hierarchical)\b", re.I), 0.08),
    (re.compile(r"\b(multi[\s-]?step|complex)\b", re.I), 0.08),
]

_CONSTRAINT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(must|shall|required?)\b", re.I),
    re.compile(r"\b(only|exactly|without\s+changing|do\s+not\s+change|preserve)\b", re.I),
    re.compile(r"\b(json\s+schema|valid\s+json|type[\s-]?safe|strict)\b", re.I),
    re.compile(r"\b(unit\s+tests?|test\s+coverage|pytest|jest)\b", re.I),
    re.compile(r"\b(follow\s+.+\s+format|output\s+format|markdown\s+table)\b", re.I),
]

_REFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(this|that|the)\s+(function|class|method|file|module|component)\b", re.I),
    re.compile(r"\b(above|below|earlier|previous)\s+(code|message|function)\b", re.I),
    re.compile(r"\bin\s+[`'\"][^`'\"]+[`'\"]", re.I),
    re.compile(r"```[\s\S]*?```"),
]

_SUB_TASK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\d+[.)]\s", re.M),
    re.compile(r"^\s*[-*]\s", re.M),
    re.compile(r"\b(also|additionally|furthermore|then|finally|next)\b", re.I),
]

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_CODE_DEF_RE = re.compile(r"\b(def |class |function |fn |func |const |let |var )", re.I)


@dataclass
class ComplexityAnalysis:
    context_load: float
    task_difficulty: float
    requirement_load: float
    task_type: str
    complexity_score: float
    reasoning_complexity: float
    sub_task_count: int
    constraint_count: int
    reference_count: int
    heuristic_task_difficulty: float = 0.0
    embedding_difficulty: float | None = None
    embedding_routing_applied: bool = False


def user_text_from_messages(messages: list[dict]) -> str:
    """Latest user message text, or all user messages if the latest is empty."""
    user_texts: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            user_texts.append(content)
        elif isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") in ("text", "input_text")
            ]
            joined = " ".join(parts).strip()
            if joined:
                user_texts.append(joined)
    if user_texts:
        return user_texts[-1]
    return ""


def classify_task_type(user_text: str, full_text_lower: str, has_code_blocks: bool) -> str:
    """Rule-based task classifier; highest pattern score wins."""
    if not user_text.strip() and not full_text_lower.strip():
        return TASK_UNKNOWN

    scores: dict[str, float] = {}
    for task_type, pattern, weight in _TASK_CLASSIFICATION_RULES:
        target = user_text if user_text.strip() else full_text_lower
        if pattern.search(target):
            scores[task_type] = scores.get(task_type, 0.0) + weight

    if has_code_blocks and scores.get(TASK_CODE_EDIT, 0) < 1.5:
        if re.search(r"\b(fix|debug|error|bug)\b", user_text, re.I):
            scores[TASK_DEBUG] = scores.get(TASK_DEBUG, 0.0) + 1.0
        else:
            scores[TASK_CODE_EDIT] = scores.get(TASK_CODE_EDIT, 0.0) + 1.0

    if not scores:
        return TASK_GENERAL

    return max(scores, key=scores.get)


def compute_context_load(token_count: int) -> float:
    """Log-scaled size score (0–1). Long prompts score higher but with diminishing returns."""
    if token_count <= 0:
        return 0.0
    # Reference: ~128k tokens ≈ 1.0
    return min(1.0, round(math.log1p(token_count) / math.log1p(128_000), 3))


def compute_reasoning_complexity(text: str) -> float:
    """Content-based reasoning demand (0–1), with capped category contributions."""
    if not text:
        return 0.0

    score = 0.0
    reasoning_hits = 0.0
    for pattern, weight in _REASONING_TRIGGERS:
        if pattern.search(text):
            reasoning_hits += weight
    score += min(reasoning_hits, 0.45)

    complexity_hits = 0.0
    for pattern, weight in _COMPLEXITY_KEYWORDS:
        if pattern.search(text):
            complexity_hits += weight
    score += min(complexity_hits, 0.20)

    code_density = text.count("\n") / max(len(text), 1) * 100
    if code_density > 10:
        score += 0.12

    question_marks = text.count("?")
    score += min(question_marks * 0.02, 0.06)

    return min(round(score, 2), 1.0)


def _count_pattern_matches(text: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(1 for p in patterns if p.search(text))


def _analyze_code_structure(text: str) -> tuple[int, int]:
    """Return (code_line_count, definition_count)."""
    blocks = _CODE_BLOCK_RE.findall(text)
    code_lines = sum(block.count("\n") for block in blocks)
    definition_count = len(_CODE_DEF_RE.findall(text))
    return code_lines, definition_count


def _conversation_depth(messages: list[dict]) -> tuple[int, int]:
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    tool_turns = sum(1 for m in messages if m.get("role") == "tool")
    return user_turns, tool_turns


def compute_requirement_load(user_text: str, full_text: str, has_tool_calls: bool) -> tuple[float, int]:
    """Constraint / output-format burden (0–1) and raw constraint count."""
    target = user_text if user_text.strip() else full_text
    constraint_count = _count_pattern_matches(target, _CONSTRAINT_PATTERNS)

    score = min(constraint_count * 0.08, 0.40)
    if has_tool_calls:
        score += 0.10
    if re.search(r"\b(json|yaml|xml|csv|schema)\b", target, re.I):
        score += 0.08
    if re.search(r"\b(tests?|coverage|lint|type[\s-]?check)\b", target, re.I):
        score += 0.08

    return min(round(score, 3), 1.0), constraint_count


def compute_structural_signals(
    user_text: str,
    full_text: str,
    messages: list[dict],
) -> tuple[int, int, int, float]:
    """Return sub_task_count, reference_count, code_bonus, conversation_bonus."""
    target = user_text if user_text.strip() else full_text

    sub_task_count = _count_pattern_matches(target, _SUB_TASK_PATTERNS)
    sub_task_count += min(target.count("?"), 3)

    reference_count = _count_pattern_matches(target, _REFERENCE_PATTERNS)

    code_lines, def_count = _analyze_code_structure(full_text)
    code_bonus = 0.0
    if code_lines > 20:
        code_bonus += 0.08
    elif code_lines > 5:
        code_bonus += 0.04
    if def_count >= 3:
        code_bonus += 0.06
    elif def_count >= 1:
        code_bonus += 0.03

    user_turns, tool_turns = _conversation_depth(messages)
    conversation_bonus = 0.0
    if user_turns >= 4:
        conversation_bonus += 0.06
    elif user_turns >= 2:
        conversation_bonus += 0.03
    if tool_turns >= 2:
        conversation_bonus += 0.05

    return sub_task_count, reference_count, code_bonus, conversation_bonus


def compute_task_difficulty(
    task_type: str,
    reasoning_complexity: float,
    dominant_language: str,
    sub_task_count: int,
    reference_count: int,
    code_bonus: float,
    conversation_bonus: float,
    has_images: bool,
) -> float:
    """Content difficulty (0–1), independent of prompt length."""
    base = TASK_DIFFICULTY_PRIORS.get(task_type, TASK_DIFFICULTY_PRIORS[TASK_UNKNOWN])

    structural = (
        min(sub_task_count * 0.04, 0.12)
        + min(reference_count * 0.03, 0.09)
        + code_bonus
        + conversation_bonus
    )

    reasoning_component = reasoning_complexity * 0.18

    domain_bonus = {
        "math": 0.12,
        "code": 0.06,
        "translation": 0.04,
    }.get(dominant_language, 0.0)

    multimodal_bonus = 0.04 if has_images else 0.0

    score = base + structural + reasoning_component + domain_bonus + multimodal_bonus
    return min(round(score, 3), 1.0)


def compute_composite_complexity(
    task_difficulty: float,
    requirement_load: float,
    context_load: float,
) -> float:
    """Weighted composite for logging / backward compatibility."""
    return min(
        round(0.65 * task_difficulty + 0.20 * requirement_load + 0.15 * context_load, 3),
        1.0,
    )


def get_routing_difficulty(
    task_difficulty: float,
    requirement_load: float,
    *,
    legacy_complexity_score: float = 0.0,
) -> float:
    """Difficulty compared against model max_complexity_score."""
    if task_difficulty > 0:
        return min(round(task_difficulty + requirement_load * 0.15, 3), 1.0)
    return legacy_complexity_score


def analyze_prompt_complexity(
    messages: list[dict],
    *,
    token_count: int,
    dominant_language: str,
    has_code_blocks: bool,
    has_images: bool,
    has_tool_calls: bool,
    full_text: str,
    apply_embeddings: bool = True,
) -> ComplexityAnalysis:
    """Full complexity analysis for a prompt (Phases 1–3)."""
    user_text = user_text_from_messages(messages)
    text_for_reasoning = user_text if user_text.strip() else full_text
    full_text_lower = full_text.lower()

    task_type = classify_task_type(user_text, full_text_lower, has_code_blocks)
    context_load = compute_context_load(token_count)
    reasoning_complexity = compute_reasoning_complexity(text_for_reasoning)

    requirement_load, constraint_count = compute_requirement_load(
        user_text, full_text, has_tool_calls,
    )
    sub_task_count, reference_count, code_bonus, conversation_bonus = compute_structural_signals(
        user_text, full_text, messages,
    )

    heuristic_task_difficulty = compute_task_difficulty(
        task_type,
        reasoning_complexity,
        dominant_language,
        sub_task_count,
        reference_count,
        code_bonus,
        conversation_bonus,
        has_images,
    )

    task_difficulty = heuristic_task_difficulty
    embedding_difficulty: float | None = None
    embedding_routing_applied = False

    if apply_embeddings:
        from app.services.embedding_complexity import (
            blend_task_difficulty,
            embedding_difficulty_for_text,
        )
        from app.core.config import settings as app_settings

        embed_text = text_for_reasoning
        if embed_text.strip() and app_settings.embedding_routing_enabled:
            embedding_difficulty = embedding_difficulty_for_text(embed_text)
            if embedding_difficulty is not None:
                task_difficulty = blend_task_difficulty(
                    heuristic_task_difficulty,
                    embedding_difficulty,
                    app_settings.embedding_blend_weight,
                )
                embedding_routing_applied = True

    complexity_score = compute_composite_complexity(
        task_difficulty, requirement_load, context_load,
    )

    return ComplexityAnalysis(
        context_load=context_load,
        task_difficulty=task_difficulty,
        requirement_load=requirement_load,
        task_type=task_type,
        complexity_score=complexity_score,
        reasoning_complexity=reasoning_complexity,
        sub_task_count=sub_task_count,
        constraint_count=constraint_count,
        reference_count=reference_count,
        heuristic_task_difficulty=heuristic_task_difficulty,
        embedding_difficulty=embedding_difficulty,
        embedding_routing_applied=embedding_routing_applied,
    )


def build_complexity_explanation(
    messages: list[dict],
    features: object,
    *,
    token_count: int,
    dominant_language: str,
    has_code_blocks: bool,
    has_images: bool,
    has_tool_calls: bool,
    full_text: str,
) -> dict:
    """Human-readable breakdown of how complexity scores were calculated."""
    from app.core.config import settings as app_settings
    from app.core.models import PromptFeatures

    assert isinstance(features, PromptFeatures)
    user_text = user_text_from_messages(messages)
    text_for_reasoning = user_text if user_text.strip() else full_text

    sub_task_count, reference_count, code_bonus, conversation_bonus = compute_structural_signals(
        user_text, full_text, messages,
    )
    task_type_prior = TASK_DIFFICULTY_PRIORS.get(
        features.task_type, TASK_DIFFICULTY_PRIORS[TASK_UNKNOWN],
    )
    reasoning_component = round(features.reasoning_complexity * 0.18, 3)
    structural_component = round(
        min(sub_task_count * 0.04, 0.12)
        + min(reference_count * 0.03, 0.09)
        + code_bonus
        + conversation_bonus,
        3,
    )
    domain_bonus = {
        "math": 0.12,
        "code": 0.06,
        "translation": 0.04,
    }.get(dominant_language, 0.0)
    multimodal_bonus = 0.04 if has_images else 0.0
    requirement_bump = round(features.requirement_load * 0.15, 3)

    routing_difficulty = get_routing_difficulty(
        features.task_difficulty,
        features.requirement_load,
    )

    explanation: dict = {
        "user_message_preview": (user_text[:200] + "…") if len(user_text) > 200 else user_text,
        "dimensions": {
            "context_load": {
                "value": features.context_load,
                "description": "Log-scaled prompt size (token_count); does not affect routing difficulty directly",
                "inputs": {"token_count": token_count},
            },
            "task_difficulty": {
                "value": features.task_difficulty,
                "description": "Content/cognitive difficulty used for model matching",
            },
            "requirement_load": {
                "value": features.requirement_load,
                "description": "Strict output constraints (must/json/tests/etc.)",
                "inputs": {"constraint_pattern_matches": features.constraint_count},
            },
            "composite_complexity_score": {
                "value": features.complexity_score,
                "description": "0.65×task + 0.20×requirement + 0.15×context (logging/UI)",
            },
            "routing_difficulty": {
                "value": routing_difficulty,
                "description": "Compared to model max_complexity_score: task_difficulty + requirement_load×0.15",
                "formula": f"{features.task_difficulty} + {features.requirement_load} × 0.15 = {routing_difficulty}",
            },
        },
        "task_difficulty_breakdown": {
            "task_type": features.task_type,
            "task_type_prior": task_type_prior,
            "structural_bonus": structural_component,
            "structural_signals": {
                "sub_task_count": sub_task_count,
                "reference_count": reference_count,
                "code_bonus": round(code_bonus, 3),
                "conversation_bonus": round(conversation_bonus, 3),
            },
            "reasoning_complexity": features.reasoning_complexity,
            "reasoning_component": reasoning_component,
            "domain": dominant_language,
            "domain_bonus": domain_bonus,
            "multimodal_bonus": multimodal_bonus,
            "heuristic_total": features.heuristic_task_difficulty,
            "sum_before_cap": round(
                task_type_prior + structural_component + reasoning_component
                + domain_bonus + multimodal_bonus,
                3,
            ),
        },
        "embedding": {
            "enabled": app_settings.embedding_routing_enabled,
            "applied": features.embedding_routing_applied,
            "embedding_difficulty": features.embedding_difficulty,
            "blend_weight": app_settings.embedding_blend_weight if app_settings.embedding_routing_enabled else None,
            "model": app_settings.embedding_model_name if app_settings.embedding_routing_enabled else None,
        },
        "signals": {
            "has_code_blocks": has_code_blocks,
            "has_images": has_images,
            "has_tool_calls": has_tool_calls,
            "has_urls": features.has_urls,
            "dominant_language": dominant_language,
            "reasoning_text_sample": (text_for_reasoning[:120] + "…") if len(text_for_reasoning) > 120 else text_for_reasoning,
        },
    }
    return explanation
