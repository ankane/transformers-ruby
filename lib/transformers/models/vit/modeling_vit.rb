# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
    class ViTEmbeddings < Torch::NN::Module
      def initialize(config, use_mask_token: false)
        super()

        @cls_token = Torch::NN::Parameter.new(Torch.randn(1, 1, config.hidden_size))
        @mask_token = use_mask_token ? Torch::NN::Parameter.new(Torch.zeros(1, 1, config.hidden_size)) : nil
        @patch_embeddings = ViTPatchEmbeddings.new(config)
        num_patches = @patch_embeddings.num_patches
        @position_embeddings = Torch::NN::Parameter.new(Torch.randn(1, num_patches + 1, config.hidden_size))
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
        @config = config
      end

      def forward(
        pixel_values,
        bool_masked_pos: nil,
        interpolate_pos_encoding: false
      )
        batch_size, _num_channels, height, width = pixel_values.shape
        embeddings = @patch_embeddings.(pixel_values, interpolate_pos_encoding: interpolate_pos_encoding)

        if !bool_masked_pos.nil?
          seq_length = embeddings.shape[1]
          mask_tokens = @mask_token.expand(batch_size, seq_length, -1)
          # replace the masked visual tokens by mask_tokens
          mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
          embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        end

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = @cls_token.expand(batch_size, -1, -1)
        embeddings = Torch.cat([cls_tokens, embeddings], dim: 1)

        # add positional encoding to each token
        if interpolate_pos_encoding
          embeddings = embeddings + @interpolate_pos_encoding.(embeddings, height, width)
        else
          embeddings = embeddings + @position_embeddings
        end

        embeddings = @dropout.(embeddings)

        embeddings
      end
    end

    class ViTPatchEmbeddings < Torch::NN::Module
      attr_reader :num_patches

      def initialize(config)
        super()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size.is_a?(Enumerable) ? image_size : [image_size, image_size]
        patch_size = patch_size.is_a?(Enumerable) ? patch_size : [patch_size, patch_size]
        num_patches = image_size[1].div(patch_size[1]) * image_size[0].div(patch_size[0])
        @image_size = image_size
        @patch_size = patch_size
        @num_channels = num_channels
        @num_patches = num_patches

        @projection = Torch::NN::Conv2d.new(num_channels, hidden_size, patch_size, stride: patch_size)
      end

      def forward(pixel_values, interpolate_pos_encoding: false)
        _batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != @num_channels
          raise ArgumentError,
            "Make sure that the channel dimension of the pixel values match with the one set in the configuration." +
            " Expected #{@num_channels} but got #{num_channels}."
        end
        if !interpolate_pos_encoding
          if height != @image_size[0] || width != @image_size[1]
            raise ArgumentError,
              "Input image size (#{height}*#{width}) doesn't match model" +
              " (#{@image_size[0]}*#{@image_size[1]})."
          end
        end
        embeddings = @projection.(pixel_values).flatten(2).transpose(1, 2)
        embeddings
      end
    end

    class ViTSelfAttention < Torch::NN::Module
      def initialize(config)
        super()
        if config.hidden_size % config.num_attention_heads != 0 && !config.instance_variable_defined?(:@embedding_size)
          raise ArgumentError,
            "The hidden size #{config.hidden_size} is not a multiple of the number of attention " +
            "heads #{config.num_attention_heads}."
        end

        @num_attention_heads = config.num_attention_heads
        @attention_head_size = (config.hidden_size / config.num_attention_heads).to_i
        @all_head_size = @num_attention_heads * @attention_head_size

        @query = Torch::NN::Linear.new(config.hidden_size, @all_head_size, bias: config.qkv_bias)
        @key = Torch::NN::Linear.new(config.hidden_size, @all_head_size, bias: config.qkv_bias)
        @value = Torch::NN::Linear.new(config.hidden_size, @all_head_size, bias: config.qkv_bias)

        @dropout = Torch::NN::Dropout.new(p: config.attention_probs_dropout_prob)
      end

      def transpose_for_scores(x)
        new_x_shape = x.size[...-1] + [@num_attention_heads, @attention_head_size]
        x = x.view(new_x_shape)
        x.permute(0, 2, 1, 3)
      end

      def forward(
        hidden_states, head_mask: nil, output_attentions: false
      )
        mixed_query_layer = @query.(hidden_states)

        key_layer = transpose_for_scores(@key.(hidden_states))
        value_layer = transpose_for_scores(@value.(hidden_states))
        query_layer = transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = Torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / Math.sqrt(@attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = Torch::NN::Functional.softmax(attention_scores, dim: -1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = @dropout.(attention_probs)

        # Mask heads if we want to
        if !head_mask.nil?
          attention_probs = attention_probs * head_mask
        end

        context_layer = Torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous
        new_context_layer_shape = context_layer.size[...-2] + [@all_head_size]
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = output_attentions ? [context_layer, attention_probs] : [context_layer]

        outputs
      end
    end

    class ViTSelfOutput < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
      end

      def forward(hidden_states, input_tensor)
        hidden_states = @dense.(hidden_states)
        hidden_states = @dropout.(hidden_states)

        hidden_states
      end
    end

    class ViTAttention < Torch::NN::Module
      def initialize(config)
        super()
        @attention = ViTSelfAttention.new(config)
        @output = ViTSelfOutput.new(config)
        @pruned_heads = Set.new
      end

      def prune_heads(heads)
        raise Todo
      end

      def forward(
        hidden_states,
        head_mask: nil,
        output_attentions: false
      )
        self_outputs = @attention.(hidden_states, head_mask: head_mask, output_attentions: output_attentions)

        attention_output = @output.(self_outputs[0], hidden_states)

        outputs = [attention_output] + self_outputs[1..]  # add attentions if we output them
        outputs
      end
    end

    class ViTIntermediate < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.intermediate_size)
        if config.hidden_act.is_a?(String)
          @intermediate_act_fn = ACT2FN[config.hidden_act]
        else
          @intermediate_act_fn = config.hidden_act
        end
      end

      def forward(hidden_states)
        hidden_states = @dense.(hidden_states)
        hidden_states = @intermediate_act_fn.(hidden_states)

        hidden_states
      end
    end

    class ViTOutput < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.intermediate_size, config.hidden_size)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
      end

      def forward(hidden_states, input_tensor)
        hidden_states = @dense.(hidden_states)
        hidden_states = @dropout.(hidden_states)

        hidden_states = hidden_states + input_tensor

        hidden_states
      end
    end

    VIT_ATTENTION_CLASSES = {
      "eager" => ViTAttention
    }

    class ViTLayer < Torch::NN::Module
      def initialize(config)
        super()
        @chunk_size_feed_forward = config.chunk_size_feed_forward
        @seq_len_dim = 1
        @attention = VIT_ATTENTION_CLASSES.fetch(config._attn_implementation).new(config)
        @intermediate = ViTIntermediate.new(config)
        @output = ViTOutput.new(config)
        @layernorm_before = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @layernorm_after = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
      end

      def forward(
        hidden_states,
        head_mask: nil,
        output_attentions: false
      )
        self_attention_outputs = @attention.(
          @layernorm_before.(hidden_states),  # in ViT, layernorm is applied before self-attention
          head_mask: head_mask,
          output_attentions: output_attentions
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1..]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = @layernorm_after.(hidden_states)
        layer_output = @intermediate.(layer_output)

        # second residual connection is done here
        layer_output = @output.(layer_output, hidden_states)

        outputs = [layer_output] + outputs

        outputs
      end
    end

    class ViTEncoder < Torch::NN::Module
      def initialize(config)
        super()
        @config = config
        @layer = Torch::NN::ModuleList.new(config.num_hidden_layers.times.map { ViTLayer.new(config) })
        @gradient_checkpointing = false
      end

      def forward(
        hidden_states,
        head_mask: nil,
        output_attentions: false,
        output_hidden_states: false,
        return_dict: true
      )
        all_hidden_states = output_hidden_states ? [] : nil
        all_self_attentions = output_attentions ? [] : nil

        @layer.each_with_index do |layer_module, i|
          if output_hidden_states
            all_hidden_states = all_hidden_states + [hidden_states]
          end

          layer_head_mask = !head_mask.nil? ? head_mask[i] : nil

          if @gradient_checkpointing && @training
            raise Todo
          else
            layer_outputs = layer_module.(hidden_states, head_mask: layer_head_mask, output_attentions: output_attentions)
          end

          hidden_states = layer_outputs[0]

          if output_attentions
            all_self_attentions = all_self_attentions + [layer_outputs[1]]
          end
        end

        if output_hidden_states
          all_hidden_states = all_hidden_states + [hidden_states]
        end

        if !return_dict
          raise Todo
        end
        BaseModelOutput.new(
          last_hidden_state: hidden_states,
          hidden_states: all_hidden_states,
          attentions: all_self_attentions
        )
      end
    end

    class ViTPreTrainedModel < PreTrainedModel
      self.config_class = ViTConfig
      self.base_model_prefix = "vit"
      self.main_input_name = "pixel_values"

      def _init_weights(mod)
        # TODO
      end
    end

    class ViTModel < ViTPreTrainedModel
      def initialize(config, add_pooling_layer: true, use_mask_token: false)
        super(config)
        @config = config

        @embeddings = ViTEmbeddings.new(config, use_mask_token: use_mask_token)
        @encoder = ViTEncoder.new(config)

        @layernorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @pooler = add_pooling_layer ? ViTPooler.new(config) : nil

        # Initialize weights and apply final processing
        post_init
      end

      def _prune_heads(heads_to_prune)
        heads_to_prune.each do |layer, heads|
          @encoder.layer[layer].attention.prune_heads(heads)
        end
      end

      def forward(
        pixel_values: nil,
        bool_masked_pos: nil,
        head_mask: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        interpolate_pos_encoding: nil,
        return_dict: nil
      )
        output_attentions = !output_attentions.nil? ? output_attentions : @config.output_attentions
        output_hidden_states = (
          !output_hidden_states.nil? ? output_hidden_states : @config.output_hidden_states
        )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        if pixel_values.nil?
          raise ArgumentError, "You have to specify pixel_values"
        end

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = get_head_mask(head_mask, @config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = @embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype
          pixel_values = pixel_values.to(expected_dtype)
        end

        embedding_output = @embeddings.(
          pixel_values, bool_masked_pos: bool_masked_pos, interpolate_pos_encoding: interpolate_pos_encoding
        )

        encoder_outputs = @encoder.(
          embedding_output,
          head_mask: head_mask,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          return_dict: return_dict
        )
        sequence_output = encoder_outputs[0]
        sequence_output = @layernorm.(sequence_output)
        pooled_output = @pooler ? @pooler.(sequence_output) : nil

        if !return_dict
          raise Todo
        end

        BaseModelOutputWithPooling.new(
          last_hidden_state: sequence_output,
          pooler_output: pooled_output,
          hidden_states: encoder_outputs.hidden_states,
          attentions: encoder_outputs.attentions
        )
      end
    end

    class ViTPooler < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        @activation = Torch::NN::Tanh.new
      end

      def forward(hidden_states)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[0.., 0]
        pooled_output = @dense.(first_token_tensor)
        pooled_output = @activation.(pooled_output)
        pooled_output
      end
    end

    class ViTForImageClassification < ViTPreTrainedModel
      def initialize(config)
        super(config)

        @num_labels = config.num_labels
        @vit = ViTModel.new(config, add_pooling_layer: false)

        # Classifier head
        @classifier = config.num_labels > 0 ? Torch::NN::Linear.new(config.hidden_size, config.num_labels) : Torch::NN::Identity.new

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        pixel_values: nil,
        head_mask: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        interpolate_pos_encoding: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @vit.(
          pixel_values: pixel_values,
          head_mask: head_mask,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          interpolate_pos_encoding: interpolate_pos_encoding,
          return_dict: return_dict
        )

        sequence_output = outputs[0]

        logits = @classifier.(sequence_output[0.., 0, 0..])

        loss = nil
        if !labels.nil?
          raise Todo
        end

        if !return_dict
          raise Todo
        end

        ImageClassifierOutput.new(
          loss: loss,
          logits: logits,
          hidden_states: outputs.hidden_states,
          attentions: outputs.attentions
        )
      end
    end
  end

  ViTForImageClassification = Vit::ViTForImageClassification
end
