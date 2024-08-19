# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
  module Distilbert
    class DistilBertConfig < PretrainedConfig
      self.model_type = "distilbert"
      self.attribute_map = {
        hidden_size: "dim",
        num_attention_heads: "n_heads",
        num_hidden_layers: "n_layers",
      }

     attr_reader :vocab_size, :max_position_embeddings, :sinusoidal_pos_embds, :n_layers, :n_heads,
      :dim, :hidden_dim, :dropout, :attention_dropout, :activation, :initializer_range, :qa_dropout,
      :seq_classif_dropout, :pad_token_id

      def initialize(
        vocab_size: 30522,
        max_position_embeddings: 512,
        sinusoidal_pos_embds: false,
        n_layers: 6,
        n_heads: 12,
        dim: 768,
        hidden_dim: 4 * 768,
        dropout: 0.1,
        attention_dropout: 0.1,
        activation: "gelu",
        initializer_range: 0.02,
        qa_dropout: 0.1,
        seq_classif_dropout: 0.2,
        pad_token_id: 0,
        **kwargs
      )
        @vocab_size = vocab_size
        @max_position_embeddings = max_position_embeddings
        @sinusoidal_pos_embds = sinusoidal_pos_embds
        @n_layers = n_layers
        @n_heads = n_heads
        @dim = dim
        @hidden_dim = hidden_dim
        @dropout = dropout
        @attention_dropout = attention_dropout
        @activation = activation
        @initializer_range = initializer_range
        @qa_dropout = qa_dropout
        @seq_classif_dropout = seq_classif_dropout
        super(**kwargs, pad_token_id: pad_token_id)
      end
    end
  end
end
