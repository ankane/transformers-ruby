# Copyright 2020 The HuggingFace Team. All rights reserved.
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
  class BaseModelOutput < ModelOutput
    attribute :last_hidden_state
    attribute :hidden_states
    attribute :attentions
  end

  class BaseModelOutputWithPooling < ModelOutput
    attribute :last_hidden_state
    attribute :pooler_output
    attribute :hidden_states
    attribute :attentions
  end

  class BaseModelOutputWithPoolingAndCrossAttentions < ModelOutput
    attribute :last_hidden_state
    attribute :pooler_output
    attribute :hidden_states
    attribute :past_key_values
    attribute :attentions
    attribute :cross_attentions
  end

  class BaseModelOutputWithPastAndCrossAttentions < ModelOutput
    attribute :last_hidden_state
    attribute :past_key_values
    attribute :hidden_states
    attribute :attentions
    attribute :cross_attentions
  end

  class MaskedLMOutput < ModelOutput
    attribute :loss
    attribute :logits
    attribute :hidden_states
    attribute :attentions
  end

  class SequenceClassifierOutput < ModelOutput
    attribute :loss
    attribute :logits
    attribute :hidden_states
    attribute :attentions
  end

  class TokenClassifierOutput < ModelOutput
    attribute :loss
    attribute :logits
    attribute :hidden_states
    attribute :attentions
  end

  class QuestionAnsweringModelOutput < ModelOutput
    attribute :loss
    attribute :start_logits
    attribute :end_logits
    attribute :hidden_states
    attribute :attentions
  end

  class ImageClassifierOutput < ModelOutput
    attribute :loss
    attribute :logits
    attribute :hidden_states
    attribute :attentions
  end
end
