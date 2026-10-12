# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ScoringIOProcessor helpers."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import TokenizersBackend

from vllm import TokensPrompt
from vllm.entrypoints.pooling.scoring.io_processor import (
    BiEncoderIOProcessor,
    CrossEncoderIOProcessor,
    JinaRankingIOProcessor,
    LateInteractionIOProcessor,
    _apply_post_tokenization_to_token_type_ids,
)
from vllm.entrypoints.pooling.scoring.utils import compress_token_type_ids
from vllm.multimodal.utils import encode_image_url
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError
from vllm.renderers import TokenizeParams
from vllm.renderers.chat_utils import MultiModalItemTracker

pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class _DummyTokenizer:
    truncation_side: str = "left"
    # Outside the range of the prompt ids below, so a test can tell a pad
    # token apart from a real one.
    pad_token_id: int = 99999


@pytest.fixture
def llm_reranker_processor(monkeypatch) -> CrossEncoderIOProcessor:
    vocab = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "a": 3,
        "bc": 4,
        "##b": 5,
        "##c": 6,
    }
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    processor = CrossEncoderIOProcessor.__new__(CrossEncoderIOProcessor)
    processor.model_config = SimpleNamespace(
        enable_prompt_embeds=False,
        revision=None,
        code_revision=None,
        trust_remote_code=False,
    )
    monkeypatch.setattr(
        processor, "chat_template_content_format", "auto", raising=False
    )
    processor.tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    processor.supports_score_template = False
    processor.use_sep_token = False
    processor.model = None
    return processor


def test_llm_reranker_tokenization_is_independent_of_nonbinding_doc_limit(
    llm_reranker_processor: CrossEncoderIOProcessor,
):
    encode_kwargs = {"add_special_tokens": True}

    _, uncapped = llm_reranker_processor.get_score_prompt("a", "bc", encode_kwargs)
    _, nonbinding = llm_reranker_processor.get_score_prompt(
        "a", "bc", encode_kwargs, max_tokens_per_doc=10
    )

    expected = [1, 3, 5, 6, 2]
    assert uncapped["prompt_token_ids"] == expected
    assert nonbinding["prompt_token_ids"] == expected


def test_token_type_ids_stay_aligned_with_a_truncated_padded_prompt():
    """The cross-encoder segment boundary must survive truncate + pad.

    `token_type_ids` are parallel to `prompt_token_ids` and are reduced to a
    single boundary index by `compress_token_type_ids`. If the two arrays are
    truncated and padded in different orders they no longer describe the same
    positions, and the model is told the query segment is empty.
    """
    tokenizer = _DummyTokenizer()
    num_query, num_doc = 20, 30
    prompt = TokensPrompt(prompt_token_ids=list(range(num_query + num_doc)))
    token_type_ids = [0] * num_query + [1] * num_doc

    tok_params = TokenizeParams(
        max_total_tokens=100,
        pad_prompt_tokens=-1,
        truncate_prompt_tokens=40,
        truncation_side="left",
    )

    processed_prompt = tok_params.apply_post_tokenization(
        tokenizer,
        prompt,
    )
    assert "prompt_token_ids" in processed_prompt
    token_type_ids = _apply_post_tokenization_to_token_type_ids(
        tokenizer, tok_params, token_type_ids
    )

    prompt_token_ids = processed_prompt["prompt_token_ids"]
    assert len(token_type_ids) == len(prompt_token_ids)

    # Keeping the last 40 tokens drops the first 10 query tokens, so 10 query
    # tokens survive and the document starts at index 10.
    first_doc = compress_token_type_ids(token_type_ids)
    assert first_doc == 10
    assert prompt_token_ids[:first_doc] == list(range(10, num_query))
    assert prompt_token_ids[first_doc:40] == list(range(num_query, 50))


def _pooling_output(
    request_id: str,
    data: torch.Tensor,
    error: RequestError | None = None,
) -> PoolingRequestOutput:
    return PoolingRequestOutput(
        request_id=request_id,
        outputs=PoolingOutput(data),
        prompt_token_ids=[1],
        num_cached_tokens=0,
        finished=True,
        error=error,
    )


@pytest.mark.parametrize(
    "processor_type", [BiEncoderIOProcessor, LateInteractionIOProcessor]
)
@pytest.mark.parametrize("failed_index", [0, 1])
def test_paired_scoring_preserves_request_error(processor_type, failed_index):
    processor = processor_type.__new__(processor_type)
    processor.pad_token_id = None
    error = RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )
    outputs = [
        _pooling_output("query", torch.tensor([1.0, 2.0])),
        _pooling_output("document", torch.tensor([3.0, 4.0])),
    ]
    outputs[failed_index] = _pooling_output(
        outputs[failed_index].request_id,
        torch.empty(0),
        error,
    )

    (result,) = processor._post_process(outputs, n_queries=1)

    assert result.request_id == "query_document"
    assert result.error is error
    assert result.outputs is outputs[failed_index].outputs


def test_jina_scoring_preserves_request_error_before_indexing_output():
    processor = JinaRankingIOProcessor.__new__(JinaRankingIOProcessor)
    error = RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )
    failed = _pooling_output("request", torch.empty(0), error)

    assert processor._post_process([failed], n_queries=1) == [failed]


_BLOCK_TEMPLATE = """
{%- for message in messages -%}
  {{- message.role + ':' -}}
  {%- for part in message.content -%}
    {%- if part.type == 'text' -%}
      {{- part.text | replace('[IMG]', '[literal IMG]') -}}
    {%- elif part.type == 'image' -%}
      {{- '[IMG]' -}}
    {%- endif -%}
  {%- endfor -%}
{%- endfor -%}
"""


@pytest.fixture
def multimodal_reranker_processor(llm_reranker_processor, monkeypatch):
    processor = llm_reranker_processor
    config = processor.model_config
    config.hf_config = SimpleNamespace()
    config.multimodal_config = None
    config.is_multimodal_model = True
    config.allowed_local_media_path = ""
    config.allowed_media_domains = None
    # Model-specific image processing is unnecessary for testing prompt rendering.
    monkeypatch.setattr(
        MultiModalItemTracker,
        "mm_processor",
        SimpleNamespace(info=SimpleNamespace(validate_num_items=Mock())),
    )
    monkeypatch.setattr(
        MultiModalItemTracker,
        "model_cls",
        SimpleNamespace(get_placeholder_str=lambda modality, i: None),
    )
    return processor


@pytest.mark.parametrize("placeholder", [None, "<image>"])
@pytest.mark.parametrize("text", [None, "", "a", "literal [IMG]"])
@pytest.mark.parametrize("with_image", [False, True])
def test_scoring_template_preserves_image_blocks_and_literal_text(
    multimodal_reranker_processor, monkeypatch, placeholder, text, with_image
):
    """Image identity must not depend on model placeholders or literal user text."""
    monkeypatch.setattr(
        MultiModalItemTracker,
        "model_cls",
        SimpleNamespace(get_placeholder_str=lambda modality, i: placeholder),
    )
    processor = multimodal_reranker_processor
    document = [] if text is None else [{"type": "text", "text": text}]
    if with_image:
        document.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2), "red"))},
                "uuid": "document-image",
            }
        )
    prompt, inputs = processor.get_score_prompt(
        "a", document, {}, chat_template=_BLOCK_TEMPLATE
    )
    expected = "query:adocument:" + (text or "").replace("[IMG]", "[literal IMG]")
    expected += "[IMG]" if with_image else ""
    assert prompt == expected
    assert inputs["prompt_token_ids"] == processor.tokenizer.encode(expected)
    if with_image:
        assert inputs["multi_modal_data"]["image"][0].getpixel((0, 0)) == (255, 0, 0)
        assert inputs["multi_modal_uuids"] == {"image": ["document-image"]}
    else:
        assert "multi_modal_data" not in inputs


@pytest.mark.parametrize(
    ("doc_limit", "expected_document"),
    [(1, "a[IMG]"), (3, "a bc[IMG]bc"), (10, "a bc[IMG]bc a")],
)
def test_scoring_text_limits_share_budget_across_blocks_and_keep_images(
    multimodal_reranker_processor,
    doc_limit,
    expected_document,
):
    processor = multimodal_reranker_processor
    last_text = {"type": "text", "text": "bc a"}
    document = [
        {"type": "text", "text": "a bc"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2)))},
        },
        last_text,
    ]
    prompt, inputs = processor.get_score_prompt(
        "a bc",
        document,
        {},
        chat_template=_BLOCK_TEMPLATE,
        max_tokens_per_query=1,
        max_tokens_per_doc=doc_limit,
    )
    assert prompt == "query:adocument:" + expected_document
    assert len(inputs["multi_modal_data"]["image"]) == 1
    assert last_text["text"] == "bc a"


@pytest.mark.parametrize("content_format", ["auto", "string"])
def test_string_scoring_templates_keep_model_placeholders(
    multimodal_reranker_processor, monkeypatch, content_format
):
    processor = multimodal_reranker_processor
    processor.chat_template_content_format = content_format
    monkeypatch.setattr(
        MultiModalItemTracker,
        "model_cls",
        SimpleNamespace(get_placeholder_str=lambda modality, i: "<image>"),
    )
    prompt, _ = processor.get_score_prompt(
        "a",
        [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2)))},
            }
        ],
        {},
        chat_template="{{ messages[0].content + messages[1].content }}",
    )
    assert prompt == "a<image>"


@pytest.mark.parametrize("content_format", ["auto", "openai"])
def test_scoring_template_formats_plain_strings_as_text_blocks(
    llm_reranker_processor, content_format
):
    processor = llm_reranker_processor
    prompt, _ = processor.get_score_prompt(
        "a",
        "bc",
        {},
        chat_template=_BLOCK_TEMPLATE,
        chat_template_content_format=content_format,
    )
    assert prompt == "query:adocument:bc"


@pytest.mark.parametrize("failure_stage", ["resolve", "apply"])
def test_missing_scoring_template_keeps_plain_pair_fallback(
    llm_reranker_processor, monkeypatch, failure_stage
):
    from vllm.entrypoints.pooling.scoring import io_processor
    from vllm.renderers.chat_utils import ChatTemplateResolutionError

    def missing_template(*args, **kwargs):
        raise ChatTemplateResolutionError("No template")

    monkeypatch.setattr(
        io_processor,
        "resolve_chat_template_content_format"
        if failure_stage == "resolve"
        else "safe_apply_chat_template",
        missing_template,
    )
    processor = llm_reranker_processor
    expected = processor.get_score_prompt("a", "bc", {})
    actual = processor.get_score_prompt("a", "bc", {}, chat_template=_BLOCK_TEMPLATE)
    assert actual == expected


def test_scoring_content_format_override_for_indirect_template(llm_reranker_processor):
    import json

    # The detector cannot infer structured content from a filter alone.
    prompt, _ = llm_reranker_processor.get_score_prompt(
        "a",
        "bc",
        {},
        chat_template="{{ messages[1].content | tojson }}",
        chat_template_content_format="openai",
    )
    assert json.loads(prompt) == [{"type": "text", "text": "bc"}]
