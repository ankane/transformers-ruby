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
    class BertEmbeddings < Torch::NN::Module
      def initialize(config)
        super()
        @word_embeddings = Torch::NN::Embedding.new(config.vocab_size, config.hidden_size, padding_idx: config.pad_token_id)
        @position_embeddings = Torch::NN::Embedding.new(config.max_position_embeddings, config.hidden_size)
        @token_type_embeddings = Torch::NN::Embedding.new(config.type_vocab_size, config.hidden_size)

        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        @position_embedding_type = config.position_embedding_type || "absolute"
        register_buffer(
          "position_ids", Torch.arange(config.max_position_embeddings).expand([1, -1]), persistent: false
        )
        register_buffer(
          "token_type_ids", Torch.zeros(position_ids.size, dtype: Torch.long), persistent: false
        )
      end

      def forward(
        input_ids: nil,
        token_type_ids: nil,
        position_ids: nil,
        inputs_embeds: nil,
        past_key_values_length: 0
      )
        if !input_ids.nil?
          input_shape = input_ids.size
        else
          input_shape = inputs_embeds.size[...-1]
        end

        seq_length = input_shape[1]

        if position_ids.nil?
          position_ids = @position_ids[0.., past_key_values_length...(seq_length + past_key_values_length)]
        end

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids.nil?
          raise Todo
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
    end

    class BertSelfAttention < Torch::NN::Module
      def initialize(config, position_embedding_type: nil)
        super()
        if config.hidden_size % config.num_attention_heads != 0 && !config.embedding_size
          raise ArgumentError,
            "The hidden size (#{config.hidden_size}) is not a multiple of the number of attention " +
            "heads (#{config.num_attention_heads})"
        end

        @num_attention_heads = config.num_attention_heads
        @attention_head_size = (config.hidden_size / config.num_attention_heads).to_i
        @all_head_size = @num_attention_heads * @attention_head_size

        @query = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
        @key = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
        @value = Torch::NN::Linear.new(config.hidden_size, @all_head_size)

        @dropout = Torch::NN::Dropout.new(p: config.attention_probs_dropout_prob)
        @position_embedding_type = position_embedding_type || config.position_embedding_type || "absolute"
        if @position_embedding_type == "relative_key" || @position_embedding_type == "relative_key_query"
          @max_position_embeddings = config.max_position_embeddings
          @distance_embedding = Torch::NN::Embedding.new(2 * config.max_position_embeddings - 1, @attention_head_size)
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

        _use_cache = !past_key_value.nil?
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
          raise Todo
        end

        attention_scores = attention_scores / Math.sqrt(@attention_head_size)
        if !attention_mask.nil?
          # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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

    class BertSelfOutput < Torch::NN::Module
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

    BERT_SELF_ATTENTION_CLASSES = {
      "eager" => BertSelfAttention
    }

    class BertAttention < Torch::NN::Module
      def initialize(config, position_embedding_type: nil)
        super()
        @self = BERT_SELF_ATTENTION_CLASSES.fetch(config._attn_implementation).new(
          config, position_embedding_type: position_embedding_type
        )
        @output = BertSelfOutput.new(config)
        @pruned_heads = Set.new
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
        self_outputs = @self.(
          hidden_states,
          attention_mask: attention_mask,
          head_mask: head_mask,
          encoder_hidden_states: encoder_hidden_states,
          encoder_attention_mask: encoder_attention_mask,
          past_key_value: past_key_value,
          output_attentions: output_attentions
        )
        attention_output = @output.(self_outputs[0], hidden_states)
        outputs = [attention_output] + self_outputs[1..]  # add attentions if we output them
        outputs
      end
    end

    class BertIntermediate < Torch::NN::Module
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

    class BertOutput < Torch::NN::Module
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

    class BertLayer < Torch::NN::Module
      def initialize(config)
        super()
        @chunk_size_feed_forward = config.chunk_size_feed_forward
        @seq_len_dim = 1
        @attention = BertAttention.new(config)
        @is_decoder = config.is_decoder
        @add_cross_attention = config.add_cross_attention
        if @add_cross_attention
          if !@is_decoder
            raise ArgumentError, "#{self} should be used as a decoder model if cross attention is added"
          end
          @crossattention = BertAttention.new(config, position_embedding_type: "absolute")
        end
        @intermediate = BertIntermediate.new(config)
        @output = BertOutput.new(config)
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
        self_attention_outputs = @attention.(
          hidden_states,
          attention_mask: attention_mask,
          head_mask: head_mask,
          output_attentions: output_attentions,
          past_key_value: self_attn_past_key_value
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if @is_decoder
          outputs = self_attention_outputs[1...-1]
          present_key_value = self_attention_outputs[-1]
        else
          outputs = self_attention_outputs[1..]  # add self attentions if we output attention weights
        end

        _cross_attn_present_key_value = nil
        if @is_decoder && !encoder_hidden_states.nil?
          raise Todo
        end

        layer_output = TorchUtils.apply_chunking_to_forward(
          method(:feed_forward_chunk), @chunk_size_feed_forward, @seq_len_dim, attention_output
        )
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
        return layer_output
      end
    end

    class BertEncoder < Torch::NN::Module
      def initialize(config)
        super()
        @config = config
        @layer = Torch::NN::ModuleList.new(config.num_hidden_layers.times.map { BertLayer.new(config) })
        @gradient_checkpointing = false
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        encoder_hidden_states: nil,
        encoder_attention_mask:nil,
        past_key_values: nil,
        use_cache: nil,
        output_attentions: false,
        output_hidden_states: false,
        return_dict: true
      )
        all_hidden_states = output_hidden_states ? [] : nil
        all_self_attentions = output_attentions ? [] : nil
        all_cross_attentions = output_attentions && @config.add_cross_attention ? [] : nil

        if @gradient_checkpointing && @raining
          raise Todo
        end

        next_decoder_cache = use_cache ? [] : nil
        @layer.each_with_index do |layer_module, i|
          if output_hidden_states
            all_hidden_states = all_hidden_states + [hidden_states]
          end

          layer_head_mask = !head_mask.nil? ? head_mask[i] : nil
          past_key_value = !past_key_values.nil? ? past_key_values[i] : nil

          if @gradient_checkpointing && @training
            raise Todo
          else
            layer_outputs = layer_module.(
              hidden_states,
              attention_mask: attention_mask,
              head_mask: layer_head_mask,
              encoder_hidden_states: encoder_hidden_states,
              encoder_attention_mask: encoder_attention_mask,
              past_key_value: past_key_value,
              output_attentions: output_attentions
            )
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
          raise Todo
        end
        BaseModelOutputWithPastAndCrossAttentions.new(
          last_hidden_state: hidden_states,
          past_key_values: next_decoder_cache,
          hidden_states: all_hidden_states,
          attentions: all_self_attentions,
          cross_attentions: all_cross_attentions
        )
      end
    end

    class BertPooler < Torch::NN::Module
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

    class BertPredictionHeadTransform < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        if config.hidden_act.is_a?(String)
          @transform_act_fn = ACT2FN[config.hidden_act]
        else
          @transform_act_fn = config.hidden_act
        end
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
      end

      def forward(hidden_states)
        hidden_states = @dense.(hidden_states)
        hidden_states = @transform_act_fn.(hidden_states)
        hidden_states = @LayerNorm.(hidden_states)
        hidden_states
      end
    end

    class BertLMPredictionHead < Torch::NN::Module
      def initialize(config)
        super()
        @transform = BertPredictionHeadTransform.new(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        @decoder = Torch::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)

        @bias = Torch::NN::Parameter.new(Torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        @decoder.instance_variable_set(:@bias, @bias)
      end

      def _tie_weights
        @decoder.instance_variable_set(:@bias, @bias)
      end

      def forward(hidden_states)
        hidden_states = @transform.(hidden_states)
        hidden_states = @decoder.(hidden_states)
        hidden_states
      end
    end

    class BertOnlyMLMHead < Torch::NN::Module
      def initialize(config)
        super()
        @predictions = BertLMPredictionHead.new(config)
      end

      def forward(sequence_output)
        prediction_scores = @predictions.(sequence_output)
        prediction_scores
      end
    end

    class BertPreTrainedModel < PreTrainedModel
      self.config_class = BertConfig
      self.base_model_prefix = "bert"

      def _init_weights(mod)
        if mod.is_a?(Torch::NN::Linear)
          mod.weight.data.normal!(mean: 0.0, std: @config.initializer_range)
          if !mod.bias.nil?
            mod.bias.data.zero!
          end
        elsif mod.is_a?(Torch::NN::Embedding)
          mod.weight.data.normal!(mean: 0.0, std: @config.initializer_range)
          if !mod.instance_variable_get(:@padding_idx).nil?
            mod.weight.data[mod.instance_variable_get(:@padding_idx)].zero!
          end
        elsif mod.is_a?(Torch::NN::LayerNorm)
          mod.bias.data.zero!
          mod.weight.data.fill!(1.0)
        end
      end
    end

    class BertModel < BertPreTrainedModel
      def initialize(config, add_pooling_layer: true)
        super(config)
        @config = config

        @embeddings = BertEmbeddings.new(config)
        @encoder = BertEncoder.new(config)

        @pooler = add_pooling_layer ? BertPooler.new(config) : nil

        @attn_implementation = config._attn_implementation
        @position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        post_init
      end

      def _prune_heads(heads_to_prune)
        heads_to_prune.each do |layer, heads|
          @encoder.layer[layer].attention.prune_heads(heads)
        end
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
        past_key_values: nil,
        use_cache: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        output_attentions = !output_attentions.nil? ? output_attentions : @config.output_attentions
        output_hidden_states = (
          !output_hidden_states.nil? ? output_hidden_states : @config.output_hidden_states
        )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        if @config.is_decoder
          use_cache = !use_cache.nil? ? use_cache : @config.use_cache
        else
          use_cache = false
        end

        if !input_ids.nil? && !inputs_embeds.nil?
          raise ArgumentError, "You cannot specify both input_ids and inputs_embeds at the same time"
        elsif !input_ids.nil?
          # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
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
          if @embeddings.token_type_ids
            buffered_token_type_ids = @embeddings.token_type_ids[0.., 0...seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
          else
            token_type_ids = Torch.zeros(input_shape, dtype: Torch.long, device: device)
          end
        end

        embedding_output = @embeddings.(
          input_ids: input_ids,
          position_ids: position_ids,
          token_type_ids: token_type_ids,
          inputs_embeds: inputs_embeds,
          past_key_values_length: past_key_values_length
        )

        if attention_mask.nil?
          attention_mask = Torch.ones([batch_size, seq_length + past_key_values_length], device: device)
        end

        use_sdpa_attention_masks = (
          @attn_implementation == "sdpa" &&
          @position_embedding_type == "absolute" &&
          head_mask.nil? &&
          !output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks
          raise Todo
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

          if use_sdpa_attention_masks
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
              encoder_attention_mask, embedding_output.dtype, tgt_len: seq_length
            )
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

        encoder_outputs = @encoder.(
          embedding_output,
          attention_mask: extended_attention_mask,
          head_mask: head_mask,
          encoder_hidden_states: encoder_hidden_states,
          encoder_attention_mask: encoder_extended_attention_mask,
          past_key_values: past_key_values,
          use_cache: use_cache,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          return_dict: return_dict
        )
        sequence_output = encoder_outputs[0]
        pooled_output = !@pooler.nil? ? @pooler.(sequence_output) : nil

        if !return_dict
          raise Todo
        end

        BaseModelOutputWithPoolingAndCrossAttentions.new(
          last_hidden_state: sequence_output,
          pooler_output: pooled_output,
          past_key_values: encoder_outputs.past_key_values,
          hidden_states: encoder_outputs.hidden_states,
          attentions: encoder_outputs.attentions,
          cross_attentions: encoder_outputs.cross_attentions
        )
      end
    end

    class BertForMaskedLM < BertPreTrainedModel
      def initialize(config)
        super(config)

        if config.is_decoder
          Transformers.logger.warn(
            "If you want to use `BertForMaskedLM` make sure `config.is_decoder: false` for " +
            "bi-directional self-attention."
          )
        end

        @bert = BertModel.new(config, add_pooling_layer: false)
        @cls = BertOnlyMLMHead.new(config)
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
        return_dict = !return_dict.nil? ? return_dict : config.use_return_dict

        outputs = @bert.(
          input_ids: input_ids,
          attention_mask: attention_mask,
          token_type_ids: token_type_ids,
          position_ids: position_ids,
          head_mask: head_mask,
          inputs_embeds: inputs_embeds,
          encoder_hidden_states: encoder_hidden_states,
          encoder_attention_mask: encoder_attention_mask,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          return_dict: return_dict
        )

        sequence_output = outputs[0]
        prediction_scores = @cls.(sequence_output)

        masked_lm_loss = nil
        if !labels.nil?
          raise Todo
        end

        if !return_dict
          raise Todo
        end

        MaskedLMOutput.new(
          loss: masked_lm_loss,
          logits: prediction_scores,
          hidden_states: outputs.hidden_states,
          attentions: outputs.attentions
        )
      end
    end

    class BertForTokenClassification < BertPreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels

        @bert = BertModel.new(config, add_pooling_layer: false)
        classifier_dropout = (
          !config.classifier_dropout.nil? ? config.classifier_dropout : config.hidden_dropout_prob
        )
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

        outputs = @bert.(
          input_ids: input_ids,
          attention_mask: attention_mask,
          token_type_ids: token_type_ids,
          position_ids: position_ids,
          head_mask: head_mask,
          inputs_embeds: inputs_embeds,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          return_dict: return_dict
        )

        sequence_output = outputs[0]

        sequence_output = @dropout.(sequence_output)
        logits = @classifier.(sequence_output)

        loss = nil
        if !labels.nil?
          loss_fct = CrossEntropyLoss.new
          loss = loss_fct.(logits.view(-1,@num_labels), labels.view(-1))
        end

        if !return_dict
          raise Todo
        end

        TokenClassifierOutput.new(
          loss: loss,
          logits: logits,
          hidden_states: outputs.hidden_states,
          attentions: outputs.attentions
        )
      end
    end
  end

  BertModel = Bert::BertModel
  BertForTokenClassification = Bert::BertForTokenClassification
end
