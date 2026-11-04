import pytest

from core.providers.hints import failure_hint


@pytest.mark.parametrize("error,expect", [
    ("ProviderError: NotFoundError: Error code: 404 - model 'llama3:latest' not found", "ollama list"),
    ("failed: model_not_found: The model `gpt-x` does not exist", "named exactly"),
    ("AuthError: AuthenticationError: 401 invalid api key", "key was refused"),
    ("RateLimitError: 429 quota exceeded", "limiting you"),
    ("TransientError: APIConnectionError: Connection error.", "could not reach the server"),
    ("failed: reply is not valid JSON", "not in the format"),
    ("ProviderError: BadRequestError: 400 this model does not support image input", "cannot read images"),
    ("ProviderError: 400 something else", "rejected the request"),
    ("TransientError: 503 overloaded", "problem on its side"),
])
def test_known_failures_get_a_next_step(error, expect):
    assert expect in failure_hint(error, "openai_compat", "llama3")


def test_hint_mentions_the_model_and_only_suggests_ollama_for_local_servers():
    local = failure_hint("404 model not found", "openai_compat", "my-model")
    assert "'my-model'" in local and "ollama list" in local
    cloud = failure_hint("404 model not found", "gemini", "my-model")
    assert "ollama" not in cloud.lower() and "'my-model'" in cloud


def test_unknown_or_empty_errors_give_no_hint():
    assert failure_hint("", "gemini") == "" and failure_hint(None) == ""
    assert failure_hint("KeyError: 'x'", "gemini") == ""


from core.providers.hints import clean_model_name  # noqa: E402


@pytest.mark.parametrize("pasted,clean", [
    ("llama3.2-vision:11b", "llama3.2-vision:11b"),
    ("  llama3.2-vision:11b\n", "llama3.2-vision:11b"),
    ('"qwen2.5vl:7b"', "qwen2.5vl:7b"),
    ("`gemma3:12b`", "gemma3:12b"),
    ("ollama run qwen2.5vl:7b", "qwen2.5vl:7b"),
    ("ollama pull hf.co/org/model:Q4_K_M\n", "hf.co/org/model:Q4_K_M"),
    ("model-a\nmodel-b", "model-a"),
    ("", ""), (None, ""),
])
def test_pasted_model_names_are_tidied(pasted, clean):
    assert clean_model_name(pasted) == clean
