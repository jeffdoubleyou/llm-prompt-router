from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChatMessage(BaseModel):
    role: Role
    content: str | list | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: Any | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]


class ModelCapability(str, Enum):
    text = "text"
    vision = "vision"
    tool_calling = "tool_calling"
    function_calling = "function_calling"
    streaming = "streaming"
    json_mode = "json_mode"
    reasoning = "reasoning"
    code = "code"
    long_context = "long_context"
    multilingual = "multilingual"
    image_generation = "image_generation"
    audio = "audio"
    embedding = "embedding"


class ModelProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    google = "google"
    azure = "azure"
    aws_bedrock = "aws_bedrock"
    together = "together"
    fireworks = "fireworks"
    groq = "groq"
    deepseek = "deepseek"
    mistral = "mistral"
    cohere = "cohere"
    llama = "llama"
    ollama = "ollama"
    custom = "custom"


class ModelRegistryEntry(BaseModel):
    id: str
    display_name: str
    provider: ModelProvider
    base_url: str | None = None
    api_key_encrypted: str | None = None
    capabilities: list[ModelCapability] = []
    tags: list[str] = []
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    context_window: int = 8192
    rpm_limit: int = 60
    tpm_limit: int = 100000
    is_active: bool = True
    priority: int = 0
    estimated_parameters_billions: float | None = None
    estimated_tokens_per_second: float | None = None
    max_complexity_score: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ModelRegistryCreate(BaseModel):
    id: str
    display_name: str
    provider: ModelProvider
    base_url: str | None = None
    api_key: str | None = None
    capabilities: list[ModelCapability] = []
    tags: list[str] = []
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    context_window: int = 8192
    rpm_limit: int = 60
    tpm_limit: int = 100000
    is_active: bool = True
    priority: int = 0
    timeout: float | None = None
    estimated_parameters_billions: float | None = None
    estimated_tokens_per_second: float | None = None
    max_complexity_score: float | None = None


class ModelRegistryUpdate(BaseModel):
    display_name: str | None = None
    provider: ModelProvider | None = None
    base_url: str | None = None
    api_key: str | None = None
    capabilities: list[ModelCapability] | None = None
    tags: list[str] | None = None
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    is_active: bool | None = None
    priority: int | None = None
    timeout: float | None = None
    estimated_parameters_billions: float | None = None
    estimated_tokens_per_second: float | None = None
    max_complexity_score: float | None = None


class PromptFeatures(BaseModel):
    token_count: int = 0
    char_length: int = 0
    has_code_blocks: bool = False
    has_urls: bool = False
    has_images: bool = False
    has_tool_calls: bool = False
    dominant_language: str = "unknown"
    reasoning_complexity: float = 0.0
    hour_of_day: int = 0
    sub_task_count: int = 0
    constraint_count: int = 0
    reference_count: int = 0
    # Phase 1+2: split complexity dimensions
    context_load: float = 0.0
    task_difficulty: float = 0.0
    requirement_load: float = 0.0
    task_type: str = "unknown"
    # Phase 3: embedding-enhanced difficulty (when EMBEDDING_ROUTING_ENABLED=true)
    heuristic_task_difficulty: float = 0.0
    embedding_difficulty: float | None = None
    embedding_routing_applied: bool = False
    # Composite + legacy routing field (see get_routing_difficulty)
    complexity_score: float = 0.0


class ClassifierPrediction(BaseModel):
    model_id: str
    confidence: float
    features: PromptFeatures


class MetricsSnapshot(BaseModel):
    model_id: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    period_seconds: int = 3600


class RequestLogEntry(BaseModel):
    id: str
    request_id: str
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    is_error: bool = False
    error_message: str | None = None
    model_used: str | None = None
    created_at: datetime | None = None


class ClassifierStatus(BaseModel):
    model_version: str | None = None
    accuracy: float | None = None
    training_data_count: int = 0
    last_trained_at: datetime | None = None
    is_training: bool = False
    embedding_routing_enabled: bool = False
    embedding_model_loaded: bool = False
    embedding_exemplar_count: int = 0
    embedding_model_name: str | None = None


class QueueStatus(BaseModel):
    depth: int = 0
    workers_active: int = 0
    avg_processing_time_ms: float = 0.0
    consumed_total: int = 0
    failed_total: int = 0


class LiveMetric(BaseModel):
    request_rate: float = 0.0
    active_requests: int = 0
    queue_depth: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    total_requests: int = 0
    total_cost: float = 0.0
    top_model: str = ""
    timestamp: str = ""


class ClassifierSampleUpdate(BaseModel):
    is_correct: bool


class PromptDebugEntry(BaseModel):
    request_id: str
    model_id: str | None = None
    messages: list[dict]
    features: dict
    created_at: str


class DebugRouteRequest(BaseModel):
    messages: list[ChatMessage]


class DebugRouteResponse(BaseModel):
    model_id: str | None
    routing_difficulty: float
    features: PromptFeatures


class ModelRoutingEvaluation(BaseModel):
    model_id: str
    eligible: bool
    exclusion_reason: str | None = None
    max_complexity_score: float | None = None
    rule_score: float | None = None
    selected: bool = False


class DebugComplexityResponse(BaseModel):
    model_id: str | None
    routing_method: str
    routing_confidence: float
    routing_difficulty: float
    would_enqueue_classifier: bool
    features: PromptFeatures
    complexity_explanation: dict
    model_evaluations: list[ModelRoutingEvaluation]
    complexity_candidate: str | None = None
    rule_candidate: str | None = None
