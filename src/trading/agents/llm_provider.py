from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

PROVIDERS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5",
    ],
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
}


class LLMConfig(BaseModel):
    provider: str
    model: str


DEFAULT_CONFIG = LLMConfig(provider="anthropic", model="claude-opus-4-5")


def create_llm_client(config: LLMConfig) -> BaseChatModel:
    """
    Instantiate the appropriate LangChain chat model for the given config.

    Args:
        config: Provider and model name to use.

    Returns:
        A LangChain BaseChatModel ready to invoke.

    Raises:
        ValueError: If the provider is not recognised.
    """
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model_name=config.model)  # type: ignore[call-arg]
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        # max_retries=2: retry transient 5xx / connection errors automatically.
        # Quota errors (429 insufficient_quota) are also retried by the SDK,
        # but they propagate after 2 attempts and get caught by AgentAbortError.
        return ChatOpenAI(model=config.model, max_retries=2)
    raise ValueError(f"Unknown provider: {config.provider!r}")
