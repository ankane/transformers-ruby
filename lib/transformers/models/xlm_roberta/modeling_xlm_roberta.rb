# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
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
  module XlmRoberta
    class XLMRobertaEmbeddings < Torch::NN::Module
      def initialize(config)
        super()
        @word_embeddings = Torch::NN::Embedding.new(config.vocab_size, config.hidden_size, padding_idx: config.pad_token_id)
        @position_embeddings = Torch::NN::Embedding.new(config.max_position_embeddings, config.hidden_size)
        @token_type_embeddings = Torch::NN::Embedding.new(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        @position_embedding_type = config.getattr("position_embedding_type", "absolute")
        register_buffer("position_ids", Torch.arange(config.max_position_embeddings).expand([1, -1]), persistent: false)
        register_buffer("token_type_ids", Torch.zeros(@position_ids.size, dtype: Torch.long), persistent: false)

        @padding_idx = config.pad_token_id
        @position_embeddings = Torch::NN::Embedding.new(config.max_position_embeddings, config.hidden_size, padding_idx: @padding_idx)
      end

      def forward(input_ids: nil, token_type_ids: nil, position_ids: nil, inputs_embeds: nil, past_key_values_length: 0)
        if position_ids.nil?
          if !input_ids.nil?
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(input_ids, @padding_idx, past_key_values_length:)
          else
            position_ids = create_position_ids_from_inputs_embeds(inputs_embeds)
          end
        end

        if !input_ids.nil?
          input_shape = input_ids.size
        else
          input_shape = inputs_embeds.size[...-1]
        end

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids.nil?
          if respond_to?(:token_type_ids)
            buffered_token_type_ids = token_type_ids[0.., ...seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
          else
            token_type_ids = Torch.zeros(input_shape, dtype: Torch.long, device: @position_ids.device)
          end
        end

        if inputs_embeds.nil?
          inputs_embeds = @word_embeddings.(input_ids)
        end
        token_type_embeddings = @token_type_embeddings.(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if @position_embedding_type == "absolute"
          position_embeddings = @position_embeddings.(position_ids)
          embeddings += position_embeddings
        end
        embeddings = @LayerNorm.(embeddings)
        embeddings = @dropout.(embeddings)
        embeddings
      end

      def create_position_ids_from_inputs_embeds(inputs_embeds)
        input_shape = inputs_embeds.size[...-1]
        sequence_length = input_shape[1]

        position_ids = Torch.arange(@padding_idx + 1, sequence_length + @padding_idx + 1, dtype: Torch.long, device: inputs_embeds.device)
        position_ids.unsqueeze(0).expand(input_shape)
      end

      def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length: 0)
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int
        incremental_indices = (Torch.cumsum(mask, dim: 1).type_as(mask) + past_key_values_length) * mask
        incremental_indices.long + padding_idx
      end
    end

    class XLMRobertaSelfAttention < Torch::NN::Module
      def initialize(config, position_embedding_type: nil)
        super()
        if config.hidden_size % config.num_attention_heads != 0 && !config.hasattr("embedding_size")
          raise ArgumentError, "The hidden size (#{config.hidden_size}) is not a multiple of the number of attention heads (#{config.num_attention_heads})"
        end

        @num_attention_heads = config.num_attention_heads
        @attention_head_size = (config.hidden_size / config.num_attention_heads).to_i
        @all_head_size = @num_attention_heads * @attention_head_size

        @query = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
        @key = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
        @value = Torch::NN::Linear.new(config.hidden_size, @all_head_size)

        @dropout = Torch::NN::Dropout.new(p: config.attention_probs_dropout_prob)
        @position_embedding_type = position_embedding_type || config.getattr("position_embedding_type", "absolute")
        if @position_embedding_type == "relative_key" || @position_embedding_type == "relative_key_query"
          @max_position_embeddings = config.max_position_embeddings
          @distance_embedding = Torch::NN::Embedding.new((2 * config.max_position_embeddings) - 1, @attention_head_size)
        end

        @is_decoder = config.is_decoder
      end

      def transpose_for_scores(x)
        new_x_shape = x.size[...-1] + [@num_attention_heads, @attention_head_size]
        x = x.view(new_x_shape)
        x.permute(0, 2, 1, 3)
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        past_key_value: nil,
        output_attentions: false
      )
        mixed_query_layer = @query.(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = !encoder_hidden_states.nil?

        if is_cross_attention && !past_key_value.nil?
          # reuse k,v, cross_attentions
          key_layer = past_key_value[0]
          value_layer = past_key_value[1]
          attention_mask = encoder_attention_mask
        elsif is_cross_attention
          key_layer = transpose_for_scores(@key.(encoder_hidden_states))
          value_layer = transpose_for_scores(@value.(encoder_hidden_states))
          attention_mask = encoder_attention_mask
        elsif !past_key_value.nil?
          key_layer = transpose_for_scores(@key.(hidden_states))
          value_layer = transpose_for_scores(@value.(hidden_states))
          key_layer = Torch.cat([past_key_value[0], key_layer], dim: 2)
          value_layer = Torch.cat([past_key_value[1], value_layer], dim: 2)
        else
          key_layer = transpose_for_scores(@key.(hidden_states))
          value_layer = transpose_for_scores(@value.(hidden_states))
        end

        query_layer = transpose_for_scores(mixed_query_layer)

        use_cache = !past_key_value.nil?
        if @is_decoder
          # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
          # Further calls to cross_attention layer can then reuse all cross-attention
          # key/value_states (first "if" case)
          # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
          # all previous decoder key/value_states. Further calls to uni-directional self-attention
          # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
          # if encoder bi-directional self-attention `past_key_value` is always `None`
          past_key_value = [key_layer, value_layer]
        end

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = Torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if @position_embedding_type == "relative_key" || @position_embedding_type == "relative_key_query"
          query_length, key_length = [query_layer.shape[2], key_layer.shape[2]]
          if use_cache
            position_ids_l = Torch.tensor(key_length - 1, dtype: Torch.long, device: hidden_states.device).view(-1, 1)
          else
            position_ids_l = Torch.arange(query_length, dtype: Torch.long, device: hidden_states.device).view(-1, 1)
          end
          position_ids_r = Torch.arange(key_length, dtype: Torch.long, device: hidden_states.device).view(1, -1)
          distance = position_ids_l - position_ids_r

          positional_embedding = @distance_embedding.((distance + @max_position_embeddings) - 1)
          positional_embedding = positional_embedding.to(dtype: query_layer.dtype)

          if @position_embedding_type == "relative_key"
            relative_position_scores = Torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
          elsif @position_embedding_type == "relative_key_query"
            relative_position_scores_query = Torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = Torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
          end
        end

        attention_scores = attention_scores / Math.sqrt(@attention_head_size)
        if !attention_mask.nil?
          # Apply the attention mask is (precomputed for all layers in XLMRobertaModel forward() function)
          attention_scores = attention_scores + attention_mask
        end

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

        if @is_decoder
          outputs = outputs + [past_key_value]
        end
        outputs
      end
    end

    class XLMRobertaSdpaSelfAttention < XLMRobertaSelfAttention
      def initialize(config, position_embedding_type: nil)
        super(config, position_embedding_type: position_embedding_type)
        @dropout_prob = config.attention_probs_dropout_prob
        @require_contiguous_qkv = Packaging::Version.parse(Utils.get_torch_version) < Packaging::Version.parse("2.2.0")
      end

      # Adapted from XLMRobertaSelfAttention
      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        past_key_value: nil,
        output_attentions: false
      )
        if @position_embedding_type != "absolute" || output_attentions || !head_mask.nil?
          # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
          Transformers.logger.warn("XLMRobertaSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support non-absolute `position_embedding_type` or `output_attentions: true` or `head_mask`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation: \"eager\"` when loading the model.")
          return super(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        end

        bsz, tgt_len, _ = hidden_states.size

        query_layer = transpose_for_scores(@query.(hidden_states))

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = !encoder_hidden_states.nil?

        current_states = is_cross_attention ? encoder_hidden_states : hidden_states
        attention_mask = is_cross_attention ? encoder_attention_mask : attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if is_cross_attention && past_key_value && past_key_value[0].shape[2] == current_states.shape[1]
          key_layer, value_layer = past_key_value
        else
          key_layer = transpose_for_scores(@key.(current_states))
          value_layer = transpose_for_scores(@value.(current_states))
          if !past_key_value.nil? && !is_cross_attention
            key_layer = Torch.cat([past_key_value[0], key_layer], dim: 2)
            value_layer = Torch.cat([past_key_value[1], value_layer], dim: 2)
          end
        end

        if @is_decoder
          # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
          # Further calls to cross_attention layer can then reuse all cross-attention
          # key/value_states (first "if" case)
          # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
          # all previous decoder key/value_states. Further calls to uni-directional self-attention
          # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
          # if encoder bi-directional self-attention `past_key_value` is always `None`
          past_key_value = [key_layer, value_layer]
        end

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if @require_contiguous_qkv && query_layer.device.type == "cuda" && !attention_mask.nil?
          query_layer = query_layer.contiguous
          key_layer = key_layer.contiguous
          value_layer = value_layer.contiguous
        end

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = @is_decoder && !is_cross_attention && attention_mask.nil? && tgt_len > 1 ? true : false

        attn_output = Torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask: attention_mask, dropout_p: @training ? @dropout_prob : 0.0, is_causal: is_causal)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, @all_head_size)

        outputs = [attn_output]
        if @is_decoder
          outputs = outputs + [past_key_value]
        end
        outputs
      end
    end

    class XLMRobertaSelfOutput < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
      end

      def forward(hidden_states, input_tensor)
        hidden_states = @dense.(hidden_states)
        hidden_states = @dropout.(hidden_states)
        hidden_states = @LayerNorm.(hidden_states + input_tensor)
        hidden_states
      end
    end

    XLM_ROBERTA_SELF_ATTENTION_CLASSES = {"eager" => XLMRobertaSelfAttention, "sdpa" => XLMRobertaSdpaSelfAttention}

    class XLMRobertaAttention < Torch::NN::Module
      def initialize(config, position_embedding_type: nil)
        super()
        @self = XLM_ROBERTA_SELF_ATTENTION_CLASSES.fetch(config._attn_implementation).new(config, position_embedding_type: position_embedding_type)
        @output = XLMRobertaSelfOutput.new(config)
        @pruned_heads = Set.new
      end

      def prune_heads(heads)
        if heads.length == 0
          return
        end
        heads, index = TorchUtils.find_pruneable_heads_and_indices(heads, @self.num_attention_heads, @self.attention_head_size, @pruned_heads)

        # Prune linear layers
        @query = TorchUtils.prune_linear_layer(@self.query, index)
        @key = TorchUtils.prune_linear_layer(@self.key, index)
        @value = TorchUtils.prune_linear_layer(@self.value, index)
        @dense = TorchUtils.prune_linear_layer(@output.dense, index, dim: 1)

        # Update hyper params and store pruned heads
        @num_attention_heads = @self.num_attention_heads - heads.length
        @all_head_size = @self.attention_head_size * @self.num_attention_heads
        @pruned_heads = @pruned_heads.union(heads)
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        past_key_value: nil,
        output_attentions: false
      )
        self_outputs = @self.(hidden_states, attention_mask:, head_mask:, encoder_hidden_states:, encoder_attention_mask:, past_key_value:, output_attentions:)
        attention_output = @output.(self_outputs[0], hidden_states)
        outputs = [attention_output] + self_outputs[1..]
        outputs
      end
    end

    class XLMRobertaIntermediate < Torch::NN::Module
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

    class XLMRobertaOutput < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.intermediate_size, config.hidden_size)
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
      end

      def forward(hidden_states, input_tensor)
        hidden_states = @dense.(hidden_states)
        hidden_states = @dropout.(hidden_states)
        hidden_states = @LayerNorm.(hidden_states + input_tensor)
        hidden_states
      end
    end

    class XLMRobertaLayer < Torch::NN::Module
      def initialize(config)
        super()
        @chunk_size_feed_forward = config.chunk_size_feed_forward
        @seq_len_dim = 1
        @attention = XLMRobertaAttention.new(config)
        @is_decoder = config.is_decoder
        @add_cross_attention = config.add_cross_attention
        if @add_cross_attention
          if !@is_decoder
            raise ArgumentError, "#{self} should be used as a decoder model if cross attention is added"
          end
          @crossattention = XLMRobertaAttention.new(config, position_embedding_type: "absolute")
        end
        @intermediate = XLMRobertaIntermediate.new(config)
        @output = XLMRobertaOutput.new(config)
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        past_key_value: nil,
        output_attentions: false
      )
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = !past_key_value.nil? ? past_key_value[...2] : nil
        self_attention_outputs = @attention.(hidden_states, attention_mask:, head_mask:, output_attentions: output_attentions, past_key_value: self_attn_past_key_value)
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if @is_decoder
          outputs = self_attention_outputs[1...-1]
          present_key_value = self_attention_outputs[-1]
        else
          outputs = self_attention_outputs[1..]
        end

        cross_attn_present_key_value = nil
        if @is_decoder && !encoder_hidden_states.nil?
          if instance_variable_defined?(:@crossattention)
            raise ArgumentError, "If `encoder_hidden_states` are passed, #{self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
          end

          # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
          cross_attn_past_key_value = !past_key_value.nil? ? past_key_value[-2..] : nil
          cross_attention_outputs = @crossattention.(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
          attention_output = cross_attention_outputs[0]
          outputs = outputs + cross_attention_outputs[1...-1]

          # add cross-attn cache to positions 3,4 of present_key_value tuple
          cross_attn_present_key_value = cross_attention_outputs[-1]
          present_key_value = present_key_value + cross_attn_present_key_value
        end

        layer_output = TorchUtils.apply_chunking_to_forward(method(:feed_forward_chunk), @chunk_size_feed_forward, @seq_len_dim, attention_output)
        outputs = [layer_output] + outputs

        # if decoder, return the attn key/values as the last output
        if @is_decoder
          outputs = outputs + [present_key_value]
        end

        outputs
      end

      def feed_forward_chunk(attention_output)
        intermediate_output = @intermediate.(attention_output)
        layer_output = @output.(intermediate_output, attention_output)
        layer_output
      end
    end

    class XLMRobertaEncoder < Torch::NN::Module
      def initialize(config)
        super()
        @config = config
        @layer = Torch::NN::ModuleList.new(config.num_hidden_layers.times.map { |_| XLMRobertaLayer.new(config) })
        @gradient_checkpointing = false
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        past_key_values: nil,
        use_cache: nil,
        output_attentions: false,
        output_hidden_states: false,
        return_dict: true
      )
        all_hidden_states = output_hidden_states ? [] : nil
        all_self_attentions = output_attentions ? [] : nil
        all_cross_attentions = output_attentions && @config.add_cross_attention ? [] : nil

        if @gradient_checkpointing && @training
          if use_cache
            Transformers.logger.warn("`use_cache: true` is incompatible with gradient checkpointing. Setting `use_cache: false`...")
            use_cache = false
          end
        end

        next_decoder_cache = use_cache ? [] : nil
        @layer.each_with_index do |layer_module, i|
          if output_hidden_states
            all_hidden_states = all_hidden_states + [hidden_states]
          end

          layer_head_mask = !head_mask.nil? ? head_mask[i] : nil
          past_key_value = !past_key_values.nil? ? past_key_values[i] : nil

          if @gradient_checkpointing && @training
            layer_outputs = _gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
          else
            layer_outputs = layer_module.(hidden_states, attention_mask:, head_mask: layer_head_mask, encoder_hidden_states:, encoder_attention_mask:, past_key_value:, output_attentions:)
          end

          hidden_states = layer_outputs[0]
          if use_cache
            next_decoder_cache += [layer_outputs[-1]]
          end
          if output_attentions
            all_self_attentions = all_self_attentions + [layer_outputs[1]]
            if @config.add_cross_attention
              all_cross_attentions = all_cross_attentions + [layer_outputs[2]]
            end
          end
        end

        if output_hidden_states
          all_hidden_states = all_hidden_states + [hidden_states]
        end

        if !return_dict
          return Array([hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions].select { |v| !v.nil? })
        end
        BaseModelOutputWithPastAndCrossAttentions.new(last_hidden_state: hidden_states, past_key_values: next_decoder_cache, hidden_states: all_hidden_states, attentions: all_self_attentions, cross_attentions: all_cross_attentions)
      end
    end

    class XLMRobertaPooler < Torch::NN::Module
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

    class XLMRobertaPreTrainedModel < PreTrainedModel
      self.config_class = XLMRobertaConfig
      self.base_model_prefix = "roberta"
      # self.supports_gradient_checkpointing = true
      # self._no_split_modules = ["XLMRobertaEmbeddings", "XLMRobertaSelfAttention", "XLMRobertaSdpaSelfAttention"]
      # self._supports_sdpa = true

      def _init_weights(module_)
        if module_.is_a?(Torch::NN::Linear)
          # Slightly different from the TF version which uses truncated_normal for initialization
          # cf https://github.com/pytorch/pytorch/pull/5617
          module_.weight.data.normal!(mean: 0.0, std: @config.initializer_range)
          if !module_.bias.nil?
            module_.bias.data.zero!
          end
        elsif module_.is_a?(Torch::NN::Embedding)
          module_.weight.data.normal!(mean: 0.0, std: @config.initializer_range)
          if !module_.padding_idx.nil?
            module_.weight.data.fetch(module_.padding_idx).zero!
          end
        elsif module_.is_a?(Torch::NN::LayerNorm)
          module_.bias.data.zero!
          module_.weight.data.fill!(1.0)
        end
      end
    end

    class XLMRobertaModel < XLMRobertaPreTrainedModel
      # self._no_split_modules = ["XLMRobertaEmbeddings", "XLMRobertaLayer"]

      def initialize(config, add_pooling_layer: true)
        super(config)
        @config = config

        @embeddings = XLMRobertaEmbeddings.new(config)
        @encoder = XLMRobertaEncoder.new(config)

        @pooler = add_pooling_layer ? XLMRobertaPooler.new(config) : nil

        @attn_implementation = config._attn_implementation
        @position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        post_init
      end

      def get_input_embeddings
        @embeddings.word_embeddings
      end

      def set_input_embeddings(value)
        @word_embeddings = value
      end

      def _prune_heads(heads_to_prune)
        heads_to_prune.each do |layer, heads|
          @encoder.layer[layer].attention.prune_heads(heads)
        end
      end

      def forward(
        input_ids,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        past_key_values: nil,
        use_cache: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        output_attentions = !output_attentions.nil? ? output_attentions : @config.output_attentions
        output_hidden_states = !output_hidden_states.nil? ? output_hidden_states : @config.output_hidden_states
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        if @config.is_decoder
          use_cache = !use_cache.nil? ? use_cache : @config.use_cache
        else
          use_cache = false
        end

        if !input_ids.nil? && !inputs_embeds.nil?
          raise ArgumentError, "You cannot specify both input_ids and inputs_embeds at the same time"
        elsif !input_ids.nil?
          warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
          input_shape = input_ids.size
        elsif !inputs_embeds.nil?
          input_shape = inputs_embeds.size[...-1]
        else
          raise ArgumentError, "You have to specify either input_ids or inputs_embeds"
        end

        batch_size, seq_length = input_shape
        device = !input_ids.nil? ? input_ids.device : inputs_embeds.device

        # past_key_values_length
        past_key_values_length = !past_key_values.nil? ? past_key_values[0][0].shape[2] : 0

        if token_type_ids.nil?
          if @embeddings.respond_to?(:token_type_ids)
            buffered_token_type_ids = @embeddings.token_type_ids[0.., ...seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
          else
            token_type_ids = Torch.zeros(input_shape, dtype: Torch.long, device: device)
          end
        end

        embedding_output = @embeddings.(input_ids: input_ids, position_ids: position_ids, token_type_ids: token_type_ids, inputs_embeds: inputs_embeds, past_key_values_length: past_key_values_length)

        if attention_mask.nil?
          attention_mask = Torch.ones([batch_size, seq_length + past_key_values_length], device: device)
        end

        use_sdpa_attention_masks = @attn_implementation == "sdpa" && @position_embedding_type == "absolute" && head_mask.nil? && !output_attentions

        # Expand the attention mask
        if use_sdpa_attention_masks && attention_mask.dim == 2
          # Expand the attention mask for SDPA.
          # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
          if @config.is_decoder
            extended_attention_mask = ModelingAttnMaskUtils._prepare_4d_causal_attention_mask_for_sdpa(attention_mask, input_shape, embedding_output, past_key_values_length)
          else
            extended_attention_mask = ModelingAttnMaskUtils._prepare_4d_attention_mask_for_sdpa(attention_mask, embedding_output.dtype, tgt_len: seq_length)
          end
        else
          # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
          # ourselves in which case we just need to make it broadcastable to all heads.
          extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape)
        end

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if @config.is_decoder && !encoder_hidden_states.nil?
          encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size
          encoder_hidden_shape = [encoder_batch_size, encoder_sequence_length]
          if encoder_attention_mask.nil?
            encoder_attention_mask = Torch.ones(encoder_hidden_shape, device: device)
          end

          if use_sdpa_attention_masks && encoder_attention_mask.dim == 2
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            encoder_extended_attention_mask = ModelingAttnMaskUtils._prepare_4d_attention_mask_for_sdpa(encoder_attention_mask, embedding_output.dtype, tgt_len: seq_length)
          else
            encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
          end
        else
          encoder_extended_attention_mask = nil
        end

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = get_head_mask(head_mask, @config.num_hidden_layers)

        encoder_outputs = @encoder.(embedding_output, attention_mask: extended_attention_mask, head_mask: head_mask, encoder_hidden_states: encoder_hidden_states, encoder_attention_mask: encoder_extended_attention_mask, past_key_values: past_key_values, use_cache: use_cache, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = !@pooler.nil? ? @pooler.(sequence_output) : nil

        if !return_dict
          return [sequence_output, pooled_output] + encoder_outputs[1..]
        end

        BaseModelOutputWithPoolingAndCrossAttentions.new(last_hidden_state: sequence_output, pooler_output: pooled_output, past_key_values: encoder_outputs.past_key_values, hidden_states: encoder_outputs.hidden_states, attentions: encoder_outputs.attentions, cross_attentions: encoder_outputs.cross_attentions)
      end
    end

    class XLMRobertaForCausalLM < XLMRobertaPreTrainedModel
      self._tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

      def initialize(config)
        super(config)

        if !config.is_decoder
          Transformers.logger.warn("If you want to use `XLMRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")
        end

        @roberta = XLMRobertaModel.new(config, add_pooling_layer: false)
        @lm_head = XLMRobertaLMHead.new(config)

        # Initialize weights and apply final processing
        post_init
      end

      def get_output_embeddings
        @lm_head.decoder
      end

      def set_output_embeddings(new_embeddings)
        @decoder = new_embeddings
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        labels: nil,
        past_key_values: nil,
        use_cache: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict
        if !labels.nil?
          use_cache = false
        end

        outputs = @roberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, encoder_hidden_states: encoder_hidden_states, encoder_attention_mask: encoder_attention_mask, past_key_values: past_key_values, use_cache: use_cache, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        sequence_output = outputs[0]
        prediction_scores = @lm_head.(sequence_output)

        lm_loss = nil
        if !labels.nil?
          # move labels to correct device to enable model parallelism
          labels = labels.to(prediction_scores.device)
          # we are doing next-token prediction; shift prediction scores and input ids by one
          shifted_prediction_scores = prediction_scores[0.., ...-1, 0..].contiguous
          labels = labels[0.., 1..].contiguous
          loss_fct = Torch::NN::CrossEntropyLoss.new
          lm_loss = loss_fct.(shifted_prediction_scores.view(-1, @config.vocab_size), labels.view(-1))
        end

        if !return_dict
          output = [prediction_scores] + outputs[2..]
          return !lm_loss.nil? ? [lm_loss] + output : output
        end

        CausalLMOutputWithCrossAttentions.new(loss: lm_loss, logits: prediction_scores, past_key_values: outputs.past_key_values, hidden_states: outputs.hidden_states, attentions: outputs.attentions, cross_attentions: outputs.cross_attentions)
      end

      def prepare_inputs_for_generation(input_ids, past_key_values: nil, attention_mask: nil, **model_kwargs)
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask.nil?
          attention_mask = input_ids.new_ones(input_shape)
        end

        # cut decoder_input_ids if past_key_values is used
        if !past_key_values.nil?
          past_length = past_key_values[0][0].shape[2]

          # Some generation methods already pass only the last input ID
          if input_ids.shape[1] > past_length
            remove_prefix_length = past_length
          else
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1
          end

          input_ids = input_ids[0.., remove_prefix_length..]
        end

        {"input_ids" => input_ids, "attention_mask" => attention_mask, "past_key_values" => past_key_values}
      end

      def _reorder_cache(past_key_values, beam_idx)
        reordered_past = []
        past_key_values.each do |layer_past|
          reordered_past += [Array(layer_past.select { |past_state| past_state })]
        end
        reordered_past
      end
    end

    class XLMRobertaForMaskedLM < XLMRobertaPreTrainedModel
      self._tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

      def initialize(config)
        super(config)

        if config.is_decoder
          Transformers.logger.warn("If you want to use `XLMRobertaForMaskedLM` make sure `config.is_decoder: false` for bi-directional self-attention.")
        end

        @roberta = XLMRobertaModel.new(config, add_pooling_layer: false)
        @lm_head = XLMRobertaLMHead.new(config)

        # Initialize weights and apply final processing
        post_init
      end

      def get_output_embeddings
        @lm_head.decoder
      end

      def set_output_embeddings(new_embeddings)
        @decoder = new_embeddings
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @roberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, encoder_hidden_states: encoder_hidden_states, encoder_attention_mask: encoder_attention_mask, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)
        sequence_output = outputs[0]
        prediction_scores = @lm_head.(sequence_output)

        masked_lm_loss = nil
        if !labels.nil?
          # move labels to correct device to enable model parallelism
          labels = labels.to(prediction_scores.device)
          loss_fct = Torch::NN::CrossEntropyLoss.new
          masked_lm_loss = loss_fct.(prediction_scores.view(-1, @config.vocab_size), labels.view(-1))
        end

        if !return_dict
          output = [prediction_scores] + outputs[2..]
          return !masked_lm_loss.nil? ? [masked_lm_loss] + output : output
        end

        MaskedLMOutput.new(loss: masked_lm_loss, logits: prediction_scores, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class XLMRobertaLMHead < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        @layer_norm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)

        @decoder = Torch::NN::Linear.new(config.hidden_size, config.vocab_size)
        @bias = Torch::NN::Parameter.new(Torch.zeros(config.vocab_size))
        @bias = @bias
      end

      def forward(features, **kwargs)
        x = @dense.(features)
        x = Activations.gelu(x)
        x = @layer_norm.(x)

        # project back to size of vocabulary with bias
        x = @decoder.(x)

        x
      end

      def _tie_weights
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if @decoder.bias.device.type == "meta"
          @bias = @bias
        else
          @bias = @decoder.bias
        end
      end
    end

    class XLMRobertaForSequenceClassification < XLMRobertaPreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels
        @config = config

        @roberta = XLMRobertaModel.new(config, add_pooling_layer: false)
        @classifier = XLMRobertaClassificationHead.new(config)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @roberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)
        sequence_output = outputs[0]
        logits = @classifier.(sequence_output)

        loss = nil
        if !labels.nil?
          # move labels to correct device to enable model parallelism
          labels = labels.to(logits.device)
          if @config.problem_type.nil?
            if @num_labels == 1
              @problem_type = "regression"
            elsif @num_labels > 1 && labels.dtype == Torch.long || labels.dtype == Torch.int
              @problem_type = "single_label_classification"
            else
              @problem_type = "multi_label_classification"
            end
          end

          if @config.problem_type == "regression"
            loss_fct = Torch::NN::MSELoss.new
            if @num_labels == 1
              loss = loss_fct.(logits.squeeze, labels.squeeze)
            else
              loss = loss_fct.(logits, labels)
            end
          elsif @config.problem_type == "single_label_classification"
            loss_fct = Torch::NN::CrossEntropyLoss.new
            loss = loss_fct.(logits.view(-1, @num_labels), labels.view(-1))
          elsif @config.problem_type == "multi_label_classification"
            loss_fct = Torch::NN::BCEWithLogitsLoss.new
            loss = loss_fct.(logits, labels)
          end
        end

        if !return_dict
          output = [logits] + outputs[2..]
          return !loss.nil? ? [loss] + output : output
        end

        SequenceClassifierOutput.new(loss: loss, logits: logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class XLMRobertaForMultipleChoice < XLMRobertaPreTrainedModel
      def initialize(config)
        super(config)

        @roberta = XLMRobertaModel.new(config)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
        @classifier = Torch::NN::Linear.new(config.hidden_size, 1)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        token_type_ids: nil,
        attention_mask: nil,
        labels: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict
        num_choices = !input_ids.nil? ? input_ids.shape[1] : inputs_embeds.shape[1]

        flat_input_ids = !input_ids.nil? ? input_ids.view(-1, input_ids.size(-1)) : nil
        flat_position_ids = !position_ids.nil? ? position_ids.view(-1, position_ids.size(-1)) : nil
        flat_token_type_ids = !token_type_ids.nil? ? token_type_ids.view(-1, token_type_ids.size(-1)) : nil
        flat_attention_mask = !attention_mask.nil? ? attention_mask.view(-1, attention_mask.size(-1)) : nil
        flat_inputs_embeds = !inputs_embeds.nil? ? inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) : nil

        outputs = @roberta.(flat_input_ids, position_ids: flat_position_ids, token_type_ids: flat_token_type_ids, attention_mask: flat_attention_mask, head_mask: head_mask, inputs_embeds: flat_inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)
        pooled_output = outputs[1]

        pooled_output = @dropout.(pooled_output)
        logits = @classifier.(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = nil
        if !labels.nil?
          # move labels to correct device to enable model parallelism
          labels = labels.to(reshaped_logits.device)
          loss_fct = Torch::NN::CrossEntropyLoss.new
          loss = loss_fct.(reshaped_logits, labels)
        end

        if !return_dict
          output = [reshaped_logits] + outputs[2..]
          return !loss.nil? ? [loss] + output : output
        end

        MultipleChoiceModelOutput.new(loss: loss, logits: reshaped_logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class XLMRobertaForTokenClassification < XLMRobertaPreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels

        @roberta = XLMRobertaModel.new(config, add_pooling_layer: false)
        classifier_dropout = !config.classifier_dropout.nil? ? config.classifier_dropout : config.hidden_dropout_prob
        @dropout = Torch::NN::Dropout.new(p: classifier_dropout)
        @classifier = Torch::NN::Linear.new(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @roberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        sequence_output = outputs[0]

        sequence_output = @dropout.(sequence_output)
        logits = @classifier.(sequence_output)

        loss = nil
        if !labels.nil?
          # move labels to correct device to enable model parallelism
          labels = labels.to(logits.device)
          loss_fct = Torch::NN::CrossEntropyLoss.new
          loss = loss_fct.(logits.view(-1, @num_labels), labels.view(-1))
        end

        if !return_dict
          output = [logits] + outputs[2..]
          return !loss.nil? ? [loss] + output : output
        end

        TokenClassifierOutput.new(loss: loss, logits: logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class XLMRobertaClassificationHead < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        classifier_dropout = !config.classifier_dropout.nil? ? config.classifier_dropout : config.hidden_dropout_prob
        @dropout = Torch::NN::Dropout.new(p: classifier_dropout)
        @out_proj = Torch::NN::Linear.new(config.hidden_size, config.num_labels)
      end

      def forward(features, **kwargs)
        x = features[0.., 0, 0..]
        x = @dropout.(x)
        x = @dense.(x)
        x = Torch.tanh(x)
        x = @dropout.(x)
        x = @out_proj.(x)
        x
      end
    end

    class XLMRobertaForQuestionAnswering < XLMRobertaPreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels

        @roberta = XLMRobertaModel.new(config, add_pooling_layer: false)
        @qa_outputs = Torch::NN::Linear.new(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        start_positions: nil,
        end_positions: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @roberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        sequence_output = outputs[0]

        logits = @qa_outputs.(sequence_output)
        start_logits, end_logits = logits.split(1, dim: -1)
        start_logits = start_logits.squeeze(-1).contiguous
        end_logits = end_logits.squeeze(-1).contiguous

        total_loss = nil
        if !start_positions.nil? && !end_positions.nil?
          # If we are on multi-GPU, split add a dimension
          if start_positions.size.length > 1
            start_positions = start_positions.squeeze(-1)
          end
          if end_positions.size.length > 1
            end_positions = end_positions.squeeze(-1)
          end
          # sometimes the start/end positions are outside our model inputs, we ignore these terms
          ignored_index = start_logits.size(1)
          start_positions = start_positions.clamp(0, ignored_index)
          end_positions = end_positions.clamp(0, ignored_index)

          loss_fct = Torch::NN::CrossEntropyLoss.new(ignore_index: ignored_index)
          start_loss = loss_fct.(start_logits, start_positions)
          end_loss = loss_fct.(end_logits, end_positions)
          total_loss = (start_loss + end_loss) / 2
        end

        if !return_dict
          output = [start_logits, end_logits] + outputs[2..]
          return !total_loss.nil? ? [total_loss] + output : output
        end

        QuestionAnsweringModelOutput.new(loss: total_loss, start_logits: start_logits, end_logits: end_logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end
  end

  XLMRobertaForSequenceClassification = XlmRoberta::XLMRobertaForSequenceClassification
end
