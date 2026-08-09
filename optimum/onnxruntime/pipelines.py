# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Pipelines running different backends."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch
import transformers.pipelines
from transformers import AutoConfig, Pipeline
from transformers import pipeline as transformers_pipeline
from transformers.pipelines import TASK_ALIASES

from optimum.utils import is_onnxruntime_available
from optimum.utils.logging import get_logger


if TYPE_CHECKING:
    from transformers import (
        BaseImageProcessor,
        FeatureExtractionMixin,
        PretrainedConfig,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )


logger = get_logger(__name__)

if is_onnxruntime_available():
    from optimum.onnxruntime import (
        ORTModelForAudioClassification,
        ORTModelForCausalLM,
        ORTModelForCTC,
        ORTModelForFeatureExtraction,
        ORTModelForImageClassification,
        ORTModelForMaskedLM,
        ORTModelForSemanticSegmentation,
        ORTModelForSequenceClassification,
        ORTModelForSpeechSeq2Seq,
        ORTModelForTokenClassification,
        ORTModelForZeroShotImageClassification,
    )
    from optimum.onnxruntime.modeling import ORTModel

    # Only tasks that exist in transformers.pipelines.SUPPORTED_TASKS can be registered here:
    # anything else would advertise a task that raises inside transformers. See
    # `REMOVED_IN_TRANSFORMERS_V5` for the tasks this mapping used to carry.
    ORT_TASKS_MAPPING = {
        "audio-classification": (ORTModelForAudioClassification,),
        "automatic-speech-recognition": (ORTModelForCTC, ORTModelForSpeechSeq2Seq),
        "feature-extraction": (ORTModelForFeatureExtraction,),
        "fill-mask": (ORTModelForMaskedLM,),
        "image-classification": (ORTModelForImageClassification,),
        "image-segmentation": (ORTModelForSemanticSegmentation,),  # TODO: we need to add ORTModelForImageSegmentation
        "text-classification": (ORTModelForSequenceClassification,),
        "text-generation": (ORTModelForCausalLM,),
        "token-classification": (ORTModelForTokenClassification,),
        "zero-shot-classification": (ORTModelForSequenceClassification,),
        "zero-shot-image-classification": (ORTModelForZeroShotImageClassification,),
    }
else:
    ORT_TASKS_MAPPING = {}


# Pipeline tasks transformers v5 removed. The corresponding ORT model classes and export tasks are
# unaffected -- only the pipeline registrations are gone -- so point users at the model classes.
REMOVED_IN_TRANSFORMERS_V5 = {
    "image-to-image": "ORTModelForImageToImage",
    "image-to-text": "ORTModelForVision2Seq",
    "question-answering": "ORTModelForQuestionAnswering",
    "summarization": "ORTModelForSeq2SeqLM",
    "text2text-generation": "ORTModelForSeq2SeqLM",
    "translation": "ORTModelForSeq2SeqLM",
}


def normalize_task(task: str) -> str:
    """Resolves `task` to its `ORT_TASKS_MAPPING` key, raising if it cannot be served.

    Aliases (`ner`, `sentiment-analysis`, ...) have to be resolved here because transformers passes
    the task through to the model loader verbatim. Unsupported tasks raise here rather than as a
    bare `KeyError` from transformers' own registry, which gives no hint that the ORT model class
    still exists for the tasks v5 removed as pipelines.
    """
    normalized_task = TASK_ALIASES.get(task, task)
    if normalized_task.startswith("translation"):
        normalized_task = "translation"

    if normalized_task in ORT_TASKS_MAPPING:
        return normalized_task

    if normalized_task in REMOVED_IN_TRANSFORMERS_V5:
        raise ValueError(
            f"Task '{task}' is no longer available as a pipeline: transformers v5 removed it. "
            f"`{REMOVED_IN_TRANSFORMERS_V5[normalized_task]}` still supports this model family, so "
            "use it directly instead of through `pipeline`. Exporting to this task is unaffected."
        )

    raise ValueError(
        f"Task '{task}' is not supported by ONNX Runtime. Only {sorted(ORT_TASKS_MAPPING)} are supported."
    )


def get_ort_model_class(
    task: str, config: PretrainedConfig | None = None, model_id: str | None = None, **model_kwargs
):
    task = normalize_task(task)

    if task == "automatic-speech-recognition":
        if config is None:
            hub_kwargs = {
                "trust_remote_code": model_kwargs.pop("trust_remote_code", False),
                "revision": model_kwargs.pop("revision", None),
                "token": model_kwargs.pop("token", None),
            }
            config = AutoConfig.from_pretrained(model_id, **hub_kwargs)
        if any(arch.endswith("ForCTC") for arch in config.architectures):
            ort_model_class = ORT_TASKS_MAPPING[task][0]
        else:
            ort_model_class = ORT_TASKS_MAPPING[task][1]
    else:
        ort_model_class = ORT_TASKS_MAPPING[task][0]

    return ort_model_class


# a modified transformers.pipelines.base.load_model that loads ORT models
def ort_load_model(
    model,
    config: PretrainedConfig | None = None,
    model_classes: tuple[type, ...] | None = None,
    task: str | None = None,
    **model_kwargs,
):
    if isinstance(model, str):
        model_kwargs.pop("framework", None)
        model_kwargs.pop("dtype", None)  # not supported for ORTModel
        model_kwargs.pop("torch_dtype", None)  # not supported for ORTModel
        model_kwargs.pop("_commit_hash", None)  # not supported for ORTModel
        model_kwargs.pop("_from_pipeline", None)  # not supported for ORTModel
        ort_model_class = get_ort_model_class(task, config, model, **model_kwargs)
        ort_model = ort_model_class.from_pretrained(model, **model_kwargs)
    elif isinstance(model, ORTModel):
        ort_model = model
    else:
        raise TypeError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide None as the model to use a default one."""
        )

    return ort_model


@contextlib.contextmanager
def patch_pipelines_to_load_ort_model():
    # `transformers.pipelines.pipeline` resolves `load_model` in its own module namespace, so the
    # patch has to go there rather than on `transformers.pipelines.base`, which would also capture
    # the unrelated assistant-model load. If the hook is ever renamed again, raise instead of
    # yielding: an unpatched `pipeline()` silently returns a torch model while the caller believes
    # it is ORT-backed, which is worse than failing.
    if not hasattr(transformers.pipelines, "load_model"):
        raise RuntimeError(
            "Could not patch `transformers.pipelines.load_model`, which "
            f"`optimum.onnxruntime.pipeline` needs in order to load ONNX Runtime models. "
            f"transformers {transformers.__version__} does not expose it. Please open an issue at "
            "https://github.com/huggingface/optimum-onnx/issues."
        )

    original_load_model = transformers.pipelines.load_model
    transformers.pipelines.load_model = ort_load_model
    try:
        yield
    finally:
        transformers.pipelines.load_model = original_load_model


# The docstring is simply a copy of transformers.pipelines.pipeline's doc with minor modifications
# to reflect the fact that this pipeline loads ONNX Runtime models that inherit from ORTModel
def pipeline(  # noqa: D417
    task: str | None = None,
    model: str | ORTModel | None = None,
    config: str | PretrainedConfig | None = None,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    feature_extractor: str | FeatureExtractionMixin | None = None,
    image_processor: str | BaseImageProcessor | None = None,
    processor: str | ProcessorMixin | None = None,
    # framework: Optional[str] = None, # we leave it as None to trigger model loading with ort_infer_framework_load_model
    revision: str | None = None,
    use_fast: bool = True,
    token: str | bool | None = None,
    device: int | str | torch.device | None = None,
    # device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None, # we do not support device_map with ORTModel yet
    # torch_dtype: Optional[Union[str, "torch.dtype"]] = "auto", we don't support torch_dtype with ORTModel yet
    trust_remote_code: bool | None = None,
    model_kwargs: dict[str, Any] | None = None,
    pipeline_class: Any | None = None,
    **kwargs: Any,
) -> Pipeline:
    """Utility factory method to build a [`Pipeline`] with an ONNX Runtime model, similar to `transformers.pipeline`.

    A pipeline consists of:

        - One or more components for pre-processing model inputs, such as a [tokenizer](tokenizer),
        [image_processor](image_processor), [feature_extractor](feature_extractor), or [processor](processors).
        - A [model](model) that generates predictions from the inputs.
        - Optional post-processing steps to refine the model's output, which can also be handled by processors.

    <Tip>
    While there are such optional arguments as `tokenizer`, `feature_extractor`, `image_processor`, and `processor`,
    they shouldn't be specified all at once. If these components are not provided, `pipeline` will try to load
    required ones automatically. In case you want to provide these components explicitly, please refer to a
    specific pipeline in order to get more details regarding what components are required.
    </Tip>

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
            - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
            - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
            - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
            - `"image-classification"`: will return a [`ImageClassificationPipeline`].
            - `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a
              [`TextClassificationPipeline`].
            - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
            - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
            - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
            - `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].

        model (`str` or [`ORTModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a ONNX Runtime model inheriting from [`ORTModel`].

            If not provided, the default for the `task` will be loaded.
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].

            If not provided, the default configuration file for the requested model will be used. That means that if
            `model` is given, its default configuration will be used. However, if `model` is not supplied, this
            `task`'s default model's config is used instead.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from [`PreTrainedFeatureExtractor`].

            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.

            If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
            is a string). However, if `config` is also not given or not a string, then the default feature extractor
            for the given `task` will be loaded.
        image_processor (`str` or [`BaseImageProcessor`], *optional*):
            The image processor that will be used by the pipeline to preprocess images for the model. This can be a
            model identifier or an actual image processor inheriting from [`BaseImageProcessor`].

            Image processors are used for Vision models and multi-modal models that require image inputs. Multi-modal
            models will also require a tokenizer to be passed.

            If not provided, the default image processor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default image processor for `config` is loaded (if it is
            a string).
        processor (`str` or [`ProcessorMixin`], *optional*):
            The processor that will be used by the pipeline to preprocess data for the model. This can be a model
            identifier or an actual processor inheriting from [`ProcessorMixin`].

            Processors are used for multi-modal models that require multi-modal inputs, for example, a model that
            requires both text and image inputs.

            If not provided, the default processor for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default processor for `config` is loaded (if it is a string).
        revision (`str`, *optional*, defaults to `"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        device (`int` or `str` or `torch.device`):
            Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this
            pipeline will be allocated.
        device_map (`str` or `dict[str, Union[int, str, torch.device]`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set
            `device_map="auto"` to compute the most optimized `device_map` automatically (see
            [here](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling#accelerate.cpu_offload)
            for more information).

            <Tip warning={true}>

            Do not use `device_map` AND `device` at the same time as they will conflict

            </Tip>

        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,
            tokenization or even pipeline files. This option should only be set to `True` for repositories you trust
            and in which you have read the code, as it will execute code present on the Hub on your local machine.
        model_kwargs (`dict[str, Any]`, *optional*):
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.
        kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.

    Examples:
    ```python
    >>> from optimum.onnxruntime import pipeline

    >>> # Sentiment analysis pipeline
    >>> analyzer = pipeline("sentiment-analysis")

    >>> # Fill-mask pipeline, specifying the checkpoint identifier
    >>> unmasker = pipeline(
    ...     "fill-mask", model="google-bert/bert-base-uncased", tokenizer="google-bert/bert-base-uncased"
    ... )

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = ORTModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```
    """
    if kwargs.pop("accelerator", None) is not None:
        logger.warning(
            "The `accelerator` argument should not be passed when using `optimum.optimum.onnxruntime.pipelines.pipeline`"
            " as ONNX Runtime is the only supported backend. Please remove the `accelerator` argument."
        )

    if task is not None:
        # Fail before delegating, so an unsupported task does not surface as a transformers KeyError.
        normalize_task(task)

    with patch_pipelines_to_load_ort_model():
        pipeline_with_ort_model = transformers_pipeline(
            task=task,
            model=model,
            config=config,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            # framework=framework,
            revision=revision,
            use_fast=use_fast,
            token=token,
            device=device,
            # device_map=device_map,
            # torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            pipeline_class=pipeline_class,
            **kwargs,
        )

    return pipeline_with_ort_model
