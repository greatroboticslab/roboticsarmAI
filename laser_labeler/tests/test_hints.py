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
    ("BadRequestError: Error code: 400 - request (4638 tokens) exceeds the available context size (4096 tokens)",
     "context window"),
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


def test_context_size_hint_names_the_ollama_fix_only_for_local_servers():
    err = "Error code: 400 - request (4638 tokens) exceeds the available context size (4096 tokens), try increasing it"
    assert "OLLAMA_CONTEXT_LENGTH" in failure_hint(err, "openai_compat", "qwen3-vl:4b")
    assert "OLLAMA_CONTEXT_LENGTH" not in failure_hint(err, "gemini", "g")


def test_timeout_gets_a_speed_hint():
    hint = failure_hint("ProviderError: Request timed out: APITimeoutError: Request timed out.", "openai_compat", "qwen3-vl:4b")
    assert "ollama ps" in hint and "Turn off thinking" in hint


def test_ran_out_of_tokens_gets_its_own_hint():
    hint = failure_hint("ReplyTruncated: The model ran out of tokens: its reply was cut off after 596 tokens, with the "
                        "request at 3500 tokens.", "openai_compat", "qwen3-vl:4b")
    assert "Spots shown per image" in hint and "Turn off thinking" in hint and "video memory" in hint


def test_ran_out_of_tokens_hint_does_the_arithmetic_and_blames_a_long_reply():
    hint = failure_hint("failed: The model ran out of tokens: its reply was cut off after 2009 tokens, with the request "
                        "at 6183 tokens. Together that is 8192 tokens, so ...", "openai_compat", "qwen3-vl:4b")
    assert "6183 + reply 2009 = 8192" in hint and "thinking or rambling" in hint
    assert "Longest reply allowed" in hint


def test_a_short_cut_off_reply_does_not_blame_thinking_first():
    hint = failure_hint("The model ran out of tokens: its reply was cut off after 300 tokens, with the request at 3796 "
                        "tokens.", "openai_compat", "m")
    assert "3796 + reply 300 = 4096" in hint and "alone used a large part" not in hint


def test_a_timeout_hint_says_the_run_moved_on_and_warns_about_bigger_windows():
    hint = failure_hint("ProviderError: Request timed out after 300 seconds: APITimeoutError", "openai_compat", "m")
    assert "16k" in hint and "Longest reply allowed" in hint and "smaller model" in hint
