# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for cross-encoder score prompt rendering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.entrypoints.pooling.scoring import io_processor as io_processor_module
from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor


def test_score_template_receives_flattened_and_structured_content(monkeypatch):
    processor = CrossEncoderIOProcessor.__new__(CrossEncoderIOProcessor)
    processor.model_config = SimpleNamespace()
    processor.tokenizer = MagicMock(return_value={"input_ids": [1, 2, 3]})
    processor.supports_score_template = False
    processor.use_sep_token = False
    processor.model = None

    query = "query"
    document = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": "document"},
    ]
    mm_data = {"image": [object()]}
    monkeypatch.setattr(
        io_processor_module,
        "parse_score_data",
        lambda *_args: ("query", "[IMG]document", mm_data, None),
    )

    captured = {}

    def fake_apply_chat_template(
        model_config,
        tokenizer,
        messages,
        **kwargs,
    ):
        captured["messages"] = messages
        return "rendered prompt"

    monkeypatch.setattr(
        io_processor_module,
        "safe_apply_chat_template",
        fake_apply_chat_template,
    )

    full_prompt, engine_prompt = processor.get_score_prompt(
        query,
        document,
        encode_kwargs={},
        chat_template="{{ messages }}",
    )

    assert full_prompt == "rendered prompt"
    assert captured["messages"] == [
        {"role": "query", "content": "query", "content_parts": query},
        {
            "role": "document",
            "content": "[IMG]document",
            "content_parts": document,
        },
    ]
    assert engine_prompt["prompt_token_ids"] == [1, 2, 3]
    assert engine_prompt["multi_modal_data"] is mm_data
