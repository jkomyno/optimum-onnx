# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import importlib
import inspect
from unittest import TestCase

from transformers import BertConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.tasks import TasksManager


# Model types registered under `library_name="transformers"` that are not transformers model types,
# and so are expected not to resolve in CONFIG_MAPPING_NAMES.
NON_TRANSFORMERS_MODEL_TYPES = {
    # Remote-code only architecture; it was never shipped by transformers.
    "internlm2",
    # Synthetic names for the text sub-component of a SigLIP export, not transformers model types.
    "siglip-text",
    "siglip-text-with-projection",
}


class TasksManagerTestCase(TestCase):
    def _check_all_models_are_registered(
        self, backend: str, class_prefix: str, classes_to_ignore: set[str] | None = None
    ):
        registered_classes = set()
        for mappings in TasksManager._SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)
        for mappings in TasksManager._TIMM_SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)
        for mappings in TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)
        for mappings in TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE.values():
            for class_ in mappings.get(backend, {}).values():
                registered_classes.add(class_.func.__name__)

        if classes_to_ignore is None:
            classes_to_ignore = set()

        module_name = f"optimum.exporters.{backend}.model_configs"

        def predicate(member):
            name = getattr(member, "__name__", "")
            module = getattr(member, "__module__", "")
            return all(
                (
                    inspect.isclass(member),
                    module == module_name,
                    name.endswith(class_prefix),
                    name not in classes_to_ignore,
                )
            )

        defined_classes = inspect.getmembers(importlib.import_module(module_name), predicate)

        # inspect.getmembers returns a list of (name, value) tuples, so we retrieve the names here.
        defined_classes = {x[0] for x in defined_classes}

        diff = defined_classes - registered_classes
        if diff:
            raise ValueError(
                f"Some models were defined for the {backend} backend, but never registered in the TasksManager: "
                f"{', '.join(diff)}."
            )

    def test_all_onnx_models_are_registered(self):
        return self._check_all_models_are_registered("onnx", "OnnxConfig")

    def test_registered_transformers_model_types_exist_in_transformers(self):
        """Every model type registered for the transformers library must still exist in transformers.

        This guards against the exporter advertising an architecture that transformers has removed,
        which is how `mctct` survived until the transformers v5 migration. Only the `transformers`
        library registrations are checked: the diffusers, timm and sentence-transformers ones use
        sub-component identifiers (`vae-encoder`, `default-timm-config`, ...) that are not, and are
        not meant to be, transformers model types.
        """
        registered = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES["transformers"]

        unknown = {
            model_type
            for model_type, mappings in registered.items()
            if "onnx" in mappings
            and model_type not in NON_TRANSFORMERS_MODEL_TYPES
            # TasksManager keys use hyphens where transformers uses underscores.
            and model_type.replace("-", "_") not in CONFIG_MAPPING_NAMES
            and model_type not in CONFIG_MAPPING_NAMES
        }

        self.assertEqual(
            unknown,
            set(),
            f"Some model types are registered for the onnx backend but no longer exist in transformers "
            f"{importlib.import_module('transformers').__version__}: {', '.join(sorted(unknown))}. "
            "Remove their exporter configs and test fixtures, or add them to NON_TRANSFORMERS_MODEL_TYPES "
            "if they are not transformers architectures.",
        )

    def test_registration_guard_must_filter_on_library_name(self):
        """The guard above must not be written naively over every registered model type.

        The diffusers, timm and sentence-transformers registrations use synthetic sub-component
        names (`vae-encoder`, `default-timm-config`, ...) that deliberately do not exist in
        transformers. This pins that fact, so that anyone rewriting the guard to iterate over all
        libraries sees why it has to filter on `library_name` first.
        """
        for library in ("diffusers", "timm", "sentence_transformers"):
            registered = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library]
            self.assertTrue(registered, f"Expected registrations for the {library} library.")

            synthetic = {
                model_type
                for model_type in registered
                if model_type.replace("-", "_") not in CONFIG_MAPPING_NAMES and model_type not in CONFIG_MAPPING_NAMES
            }
            self.assertTrue(
                synthetic,
                f"Expected the {library} registrations to use names absent from transformers; "
                "if that is no longer true, the library_name filter in the guard above may be "
                "hiding real removals.",
            )

    def test_register(self):
        # Case 1: We try to register a config that was already registered, it should not register anything.
        register_for_onnx = TasksManager.create_register("onnx")

        @register_for_onnx("bert", "text-classification")
        class BadBertOnnxConfig(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "onnx",
            model_type="bert",
            task="text-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertNotEqual(
            bert_onnx_config.__class__,
            BadBertOnnxConfig,
            "Registering an already existing config constructor should not do anything unless overwrite_existing=True.",
        )

        # Case 2: We try to register a config that was already registered, but authorize overwriting, it should register
        # the new config.
        register_for_onnx = TasksManager.create_register("onnx", overwrite_existing=True)

        @register_for_onnx("bert", "text-classification")
        class BadBertOnnxConfig2(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "onnx",
            model_type="bert",
            task="text-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertEqual(
            bert_onnx_config.__class__,
            BadBertOnnxConfig2,
            (
                "Registering an already existing config constructor with overwrite_existing=True should overwrite the "
                "old config constructor."
            ),
        )

        # Case 3: Registering an unknown task.
        with self.assertRaisesRegex(ValueError, "The TasksManager does not know the task called"):

            @register_for_onnx("bert", "this is a wrong name for a task")
            class UnknownTask(BertOnnxConfig):
                pass

        # Case 4: Registering for a new backend.
        register_for_new_backend = TasksManager.create_register("new-backend")

        @register_for_new_backend("bert", "text-classification")
        class BertNewBackendConfig(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "new-backend",
            model_type="bert",
            task="text-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertEqual(
            bert_onnx_config.__class__, BertNewBackendConfig, "Wrong config class compared to the registered one."
        )

        # Case 5: Registering a new task for a already existing backend.
        @register_for_new_backend("bert", "token-classification")
        class BertNewBackendConfigTaskSpecific(BertOnnxConfig):
            pass

        bert_config_constructor = TasksManager.get_exporter_config_constructor(
            "new-backend",
            model_type="bert",
            task="token-classification",
        )
        bert_onnx_config = bert_config_constructor(BertConfig())

        self.assertEqual(
            bert_onnx_config.__class__,
            BertNewBackendConfigTaskSpecific,
            "Wrong config class compared to the registered one.",
        )
