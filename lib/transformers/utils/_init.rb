# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

module Transformers
  WEIGHTS_NAME = "pytorch_model.bin"
  WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
  TF2_WEIGHTS_NAME = "tf_model.h5"
  TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
  TF_WEIGHTS_NAME = "model.ckpt"
  FLAX_WEIGHTS_NAME = "flax_model.msgpack"
  FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
  SAFE_WEIGHTS_NAME = "model.safetensors"
  SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
  CONFIG_NAME = "config.json"
  FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
  IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
  PROCESSOR_NAME = "processor_config.json"
  GENERATION_CONFIG_NAME = "generation_config.json"
  MODEL_CARD_NAME = "modelcard.json"
end
