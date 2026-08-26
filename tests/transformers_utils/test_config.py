# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from transformers import PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import (
    get_pooling_config,
    get_safetensors_params_metadata,
    try_get_dense_modules,
    try_get_generation_config,
    try_get_sentence_transformer_config,
)


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_model_config_generation_fallback_forwards_code_revision():
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            generation_config="auto",
            hf_config_path=None,
            model="org/model",
            trust_remote_code=True,
            revision="model-pin",
            code_revision="code-pin",
            config_format="auto",
            hf_token=None,
        ),
    )

    with (
        patch.object(
            config_module.GenerationConfig,
            "from_pretrained",
            side_effect=OSError,
        ),
        patch.object(
            config_module,
            "get_config",
            return_value=PretrainedConfig(),
        ) as get_config,
    ):
        ModelConfig.try_get_generation_config(model_config)

    get_config.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
        revision="model-pin",
        code_revision="code-pin",
        config_format="auto",
        token=None,
    )


def test_safetensors_metadata_of_repo_without_safetensors():
    """A repo storing its weights in another format is an answer, not a failure,
    so it must not be retried."""
    from huggingface_hub.errors import LocalEntryNotFoundError, NotASafetensorsRepoError

    get_safetensors_metadata = MagicMock(
        side_effect=NotASafetensorsRepoError("not a safetensors repo")
    )
    api = SimpleNamespace(
        get_safetensors_metadata=get_safetensors_metadata,
        snapshot_download=MagicMock(side_effect=LocalEntryNotFoundError("no cache")),
    )

    with patch.object(config_module, "hf_api", lambda: api):
        assert get_safetensors_params_metadata("some/pytorch-only-model") == {}

    get_safetensors_metadata.assert_called_once()


def test_current_sentence_transformers_pooling_and_dense_metadata(tmp_path):
    pooling_dir = tmp_path / "1_Pooling"
    dense_dir = tmp_path / "2_Dense"
    pooling_dir.mkdir()
    dense_dir.mkdir()
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "CrossEncoder", "activation_fn": "torch.nn.Identity"})
    )
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 1,
                    "path": "1_Pooling",
                    "type": (
                        "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
                    ),
                },
                {
                    "idx": 2,
                    "path": "2_Dense",
                    "type": "sentence_transformers.base.modules.dense.Dense",
                },
            ]
        )
    )
    (pooling_dir / "config.json").write_text(
        json.dumps(
            {
                "embedding_dimension": 4,
                "pooling_mode": "mean",
                "include_prompt": True,
            }
        )
    )
    dense_config = {
        "in_features": 4,
        "out_features": 1,
        "bias": False,
        "activation_function": "torch.nn.modules.linear.Identity",
        "module_input_name": "sentence_embedding",
        "module_output_name": "scores",
    }
    (dense_dir / "config.json").write_text(json.dumps(dense_config))

    assert try_get_sentence_transformer_config(str(tmp_path), revision=None) == {
        "model_type": "CrossEncoder",
        "activation_fn": "torch.nn.Identity",
    }
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            registry=SimpleNamespace(get_supported_archs=lambda: []),
            model=str(tmp_path),
            revision=None,
        ),
    )
    assert (
        ModelConfig._get_default_convert_type(
            model_config, ["Mistral3Model"], "pooling"
        )
        == "classify"
    )
    assert get_pooling_config(str(tmp_path), revision=None) == {
        "use_activation": False,
        "seq_pooling_type": "MEAN",
    }
    assert try_get_dense_modules(str(tmp_path), revision=None) == [
        {**dense_config, "folder": "2_Dense"}
    ]
