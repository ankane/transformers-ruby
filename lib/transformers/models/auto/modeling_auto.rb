# Copyright 2018 The HuggingFace Inc. team.
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
  MODEL_MAPPING_NAMES = {
    "bert" => "BertModel",
    "distilbert" => "DistilBertModel",
    "vit" => "ViTModel"
  }

  MODEL_FOR_MASKED_LM_MAPPING_NAMES = {
    "bert" => "BertForMaskedLM"
  }

  MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = {
    "distilbert" => "DistilBertForSequenceClassification"
  }

  MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = {
    "distilbert" => "DistilBertForQuestionAnswering"
  }

  MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = {
    "vit" => "ViTForImageClassification"
  }

  MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = {
    "bert" => "BertForTokenClassification"
  }

  MODEL_MAPPING = LazyAutoMapping.new(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
  MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = LazyAutoMapping.new(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
  )
  MODEL_FOR_MASKED_LM_MAPPING = LazyAutoMapping.new(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)
  MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = LazyAutoMapping.new(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
  )
  MODEL_FOR_QUESTION_ANSWERING_MAPPING = LazyAutoMapping.new(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
  )
  MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = LazyAutoMapping.new(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
  )

  class AutoModel < BaseAutoModelClass
    self._model_mapping = MODEL_MAPPING
  end

  class AutoModelForMaskedLM < BaseAutoModelClass
    self._model_mapping = MODEL_FOR_MASKED_LM_MAPPING
  end

  class AutoModelForSequenceClassification < BaseAutoModelClass
    self._model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
  end

  class AutoModelForQuestionAnswering < BaseAutoModelClass
    self._model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING
  end

  class AutoModelForTokenClassification < BaseAutoModelClass
    self._model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
  end

  class AutoModelForImageClassification < BaseAutoModelClass
    self._model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
  end
end
