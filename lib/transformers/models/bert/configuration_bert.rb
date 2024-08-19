# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
  module Bert
    class BertConfig < PretrainedConfig
      self.model_type = "bert"

      attr_reader :vocab_size, :hidden_size, :num_hidden_layers, :num_attention_heads,
        :intermediate_size, :hidden_act, :hidden_dropout_prob, :attention_probs_dropout_prob,
        :max_position_embeddings, :type_vocab_size, :initializer_range, :layer_norm_eps,
        :position_embedding_type, :use_cache, :classifier_dropout

      def initialize(
        vocab_size: 30522,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        hidden_act: "gelu",
        hidden_dropout_prob: 0.1,
        attention_probs_dropout_prob: 0.1,
        max_position_embeddings: 512,
        type_vocab_size: 2,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
        position_embedding_type: "absolute",
        use_cache: true,
        classifier_dropout: nil,
        **kwargs
      )
        super(pad_token_id: pad_token_id, **kwargs)

        @vocab_size = vocab_size
        @hidden_size = hidden_size
        @num_hidden_layers = num_hidden_layers
        @num_attention_heads = num_attention_heads
        @hidden_act = hidden_act
        @intermediate_size = intermediate_size
        @hidden_dropout_prob = hidden_dropout_prob
        @attention_probs_dropout_prob = attention_probs_dropout_prob
        @max_position_embeddings = max_position_embeddings
        @type_vocab_size = type_vocab_size
        @initializer_range = initializer_range
        @layer_norm_eps = layer_norm_eps
        @position_embedding_type = position_embedding_type
        @use_cache = use_cache
        @classifier_dropout = classifier_dropout
      end
    end
  end
end
