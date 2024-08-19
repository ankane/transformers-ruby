# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
  module Vit
    class ViTConfig < PretrainedConfig
      self.model_type = "vit"

      attr_reader :hidden_size, :num_hidden_layers, :num_attention_heads, :intermediate_size,
        :hidden_act, :hidden_dropout_prob, :attention_probs_dropout_prob, :initializer_range,
        :layer_norm_eps, :image_size, :patch_size, :num_channels, :qkv_bias, :encoder_stride

      def initialize(
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        hidden_act: "gelu",
        hidden_dropout_prob: 0.0,
        attention_probs_dropout_prob: 0.0,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        image_size: 224,
        patch_size: 16,
        num_channels: 3,
        qkv_bias: true,
        encoder_stride: 16,
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
        @initializer_range = initializer_range
        @layer_norm_eps = layer_norm_eps
        @image_size = image_size
        @patch_size = patch_size
        @num_channels = num_channels
        @qkv_bias = qkv_bias
        @encoder_stride = encoder_stride
      end
    end
  end
end
