# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from importlib import import_module
from importlib.metadata import requires

import pytest
import torch
import transformers
from packaging.requirements import Requirement
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    FalconConfig,
    FalconForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.pipelines import TASK_ALIASES

from optimum.exporters.onnx import main_export
from optimum.onnxruntime import (
    ORTModelForCausalLM,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    pipeline,
)
from optimum.onnxruntime.modeling import ORTModel
from optimum.onnxruntime.pipelines import (
    ORT_TASKS_MAPPING,
    REMOVED_IN_TRANSFORMERS_V5,
    normalize_task,
    ort_load_model,
    patch_pipelines_to_load_ort_model,
)


VOCAB_SIZE = 32


def _requirement(name: str) -> Requirement:
    requirement = next(value for value in requires("optimum-onnx") or [] if Requirement(value).name == name)
    return Requirement(requirement)


def _tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "hello": 4,
        "world": 5,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def test_installed_distribution_requires_transformers_v5():
    requirement = _requirement("transformers")

    assert requirement.specifier.contains(transformers.__version__)
    assert requirement.specifier.contains("5.14.0")
    assert not requirement.specifier.contains("4.57.0")
    assert not requirement.specifier.contains("6.0.0")


@pytest.mark.parametrize(
    ("library_name", "task", "expected_kwarg"),
    [
        ("transformers", "text-classification", "dtype"),
        ("timm", "image-classification", "torch_dtype"),
    ],
)
def test_main_export_routes_dtype_to_the_selected_library(
    monkeypatch, tmp_path, library_name: str, task: str, expected_kwarg: str
):
    export_main = import_module("optimum.exporters.onnx.__main__")
    captured_kwargs = {}

    if library_name == "transformers":
        monkeypatch.setattr(export_main.AutoConfig, "from_pretrained", lambda *args, **kwargs: BertConfig())
    else:
        monkeypatch.setattr(export_main, "is_timm_available", lambda: True)

    def stop_after_model_loading(*args, **kwargs):
        captured_kwargs.update(kwargs)
        raise RuntimeError("stop after model loading")

    monkeypatch.setattr(export_main.TasksManager, "get_model_from_task", stop_after_model_loading)

    with pytest.raises(RuntimeError, match="stop after model loading"):
        main_export(
            "unused-local-model",
            output=tmp_path,
            task=task,
            framework="pt",
            library_name=library_name,
            dtype="fp16",
        )

    other_kwarg = "torch_dtype" if expected_kwarg == "dtype" else "dtype"
    assert captured_kwargs[expected_kwarg] is torch.float16
    assert other_kwarg not in captured_kwargs


def test_every_registered_pipeline_task_exists_in_transformers():
    unknown = sorted(set(ORT_TASKS_MAPPING) - set(transformers.pipelines.SUPPORTED_TASKS))

    assert unknown == []


def test_pipeline_docstring_advertises_no_unsupported_task():
    advertised = set(re.findall(r'- `"([a-z0-9_-]+)"`', pipeline.__doc__))
    advertised -= set(TASK_ALIASES)

    assert sorted(advertised - set(ORT_TASKS_MAPPING)) == []


def test_transformers_exposes_the_patched_pipeline_loader():
    assert hasattr(transformers.pipelines, "load_model")

    with patch_pipelines_to_load_ort_model():
        assert transformers.pipelines.load_model is ort_load_model

    assert transformers.pipelines.load_model is not ort_load_model


@pytest.mark.parametrize("task, model_class_name", REMOVED_IN_TRANSFORMERS_V5.items())
def test_pipeline_tasks_removed_in_transformers_v5_raise_a_clear_error(task, model_class_name):
    with pytest.raises(ValueError) as error:
        pipeline(task, model="unused-local-model")

    assert task in str(error.value)
    assert model_class_name in str(error.value)


def test_unsupported_pipeline_task_raises_a_clear_error():
    with pytest.raises(ValueError, match="not supported by ONNX Runtime"):
        pipeline("not-a-real-task", model="unused-local-model")


@pytest.mark.parametrize(
    "alias, task",
    [(alias, task) for alias, task in TASK_ALIASES.items() if task in ORT_TASKS_MAPPING],
)
def test_pipeline_task_aliases_resolve_to_registered_tasks(alias, task):
    assert normalize_task(alias) == task


def test_encoder_export_runtime_and_pipeline_are_offline(tmp_path):
    torch.manual_seed(0)
    source_dir = tmp_path / "source"
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    ).eval()
    model.save_pretrained(source_dir)

    ort_model = ORTModelForSequenceClassification.from_pretrained(
        source_dir,
        export=True,
        local_files_only=True,
    )
    inputs = {
        "input_ids": torch.tensor([[2, 4, 5, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "token_type_ids": torch.zeros((1, 4), dtype=torch.long),
    }

    with torch.no_grad():
        torch_logits = model(**inputs).logits
    ort_logits = ort_model(**inputs).logits
    torch.testing.assert_close(torch_logits, ort_logits, atol=1e-4, rtol=1e-4)

    classifier = pipeline("text-classification", model=ort_model, tokenizer=_tokenizer())
    assert isinstance(classifier.model, ORTModel)
    assert classifier("hello world")[0]["label"] in model.config.id2label.values()


def test_causal_export_and_greedy_generation_match_torch(tmp_path):
    torch.manual_seed(0)
    source_dir = tmp_path / "source"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    model.save_pretrained(source_dir)

    ort_model = ORTModelForCausalLM.from_pretrained(
        source_dir,
        export=True,
        local_files_only=True,
        use_cache=True,
    )
    inputs = {
        "input_ids": torch.tensor([[1, 4, 5]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    generation_kwargs = {"do_sample": False, "max_new_tokens": 2}

    with torch.no_grad():
        torch_tokens = model.generate(**inputs, **generation_kwargs)
    ort_tokens = ort_model.generate(**inputs, **generation_kwargs)

    torch.testing.assert_close(torch_tokens, ort_tokens)


def test_seq2seq_export_and_greedy_generation_match_torch(tmp_path):
    torch.manual_seed(0)
    source_dir = tmp_path / "source"
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=VOCAB_SIZE,
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dropout_rate=0,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
    ).eval()
    model.save_pretrained(source_dir)

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        source_dir,
        export=True,
        local_files_only=True,
        use_cache=True,
    )
    inputs = {
        "input_ids": torch.tensor([[2, 4, 5, 1]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    generation_kwargs = {"do_sample": False, "max_new_tokens": 2}

    with torch.no_grad():
        torch_tokens = model.generate(**inputs, **generation_kwargs)
    ort_tokens = ort_model.generate(**inputs, **generation_kwargs)

    torch.testing.assert_close(torch_tokens, ort_tokens)


def test_falcon_alibi_export_uses_the_traceable_mask_path(tmp_path):
    torch.manual_seed(0)
    source_dir = tmp_path / "source"
    model = FalconForCausalLM(
        FalconConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_kv_heads=1,
            ffn_hidden_size=32,
            alibi=True,
            multi_query=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    model.save_pretrained(source_dir)

    ort_model = ORTModelForCausalLM.from_pretrained(
        source_dir,
        export=True,
        local_files_only=True,
        use_cache=True,
    )
    outputs = ort_model(
        input_ids=torch.tensor([[1, 4, 5]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )

    assert outputs.logits.shape == (1, 3, VOCAB_SIZE)
