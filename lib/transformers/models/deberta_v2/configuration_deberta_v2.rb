# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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
  module DebertaV2
    class DebertaV2Config < PretrainedConfig
      self.model_type = "deberta-v2"

      attr_reader :vocab_size, :hidden_size, :num_hidden_layers, :num_attention_heads,
        :intermediate_size, :hidden_act, :hidden_dropout_prob, :attention_probs_dropout_prob,
        :max_position_embeddings, :type_vocab_size, :initializer_range, :layer_norm_eps,
        :relative_attention, :max_relative_positions, :pad_token_id, :position_biased_input,
        :pos_att_type, :pooler_dropout, :pooler_hidden_act, :pooler_hidden_size

      def initialize(
        vocab_size: 128100,
        hidden_size: 1536,
        num_hidden_layers: 24,
        num_attention_heads: 24,
        intermediate_size: 6144,
        hidden_act: "gelu",
        hidden_dropout_prob: 0.1,
        attention_probs_dropout_prob: 0.1,
        max_position_embeddings: 512,
        type_vocab_size: 0,
        initializer_range: 0.02,
        layer_norm_eps: 1e-07,
        relative_attention: false,
        max_relative_positions: -1,
        pad_token_id: 0,
        position_biased_input: true,
        pos_att_type: nil,
        pooler_dropout: 0,
        pooler_hidden_act: "gelu",
        **kwargs
      )
        super(**kwargs)

        @hidden_size = hidden_size
        @num_hidden_layers = num_hidden_layers
        @num_attention_heads = num_attention_heads
        @intermediate_size = intermediate_size
        @hidden_act = hidden_act
        @hidden_dropout_prob = hidden_dropout_prob
        @attention_probs_dropout_prob = attention_probs_dropout_prob
        @max_position_embeddings = max_position_embeddings
        @type_vocab_size = type_vocab_size
        @initializer_range = initializer_range
        @relative_attention = relative_attention
        @max_relative_positions = max_relative_positions
        @pad_token_id = pad_token_id
        @position_biased_input = position_biased_input

        # Backwards compatibility
        if pos_att_type.is_a?(String)
          pos_att_type = pos_att_type.downcase.split("|").map { |x| x.strip }
        end

        @pos_att_type = pos_att_type
        @vocab_size = vocab_size
        @layer_norm_eps = layer_norm_eps

        @pooler_hidden_size = kwargs[:pooler_hidden_size] || hidden_size
        @pooler_dropout = pooler_dropout
        @pooler_hidden_act = pooler_hidden_act
      end
    end
  end
end
