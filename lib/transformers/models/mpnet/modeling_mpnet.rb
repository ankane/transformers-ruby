# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
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
  module Mpnet
    class MPNetPreTrainedModel < PreTrainedModel
      self.config_class = MPNetConfig
      self.base_model_prefix = "mpnet"

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

    class MPNetEmbeddings < Torch::NN::Module
      def initialize(config)
        super()
        @padding_idx = 1
        @word_embeddings = Torch::NN::Embedding.new(config.vocab_size, config.hidden_size, padding_idx: @padding_idx)
        @position_embeddings = Torch::NN::Embedding.new(config.max_position_embeddings, config.hidden_size, padding_idx: @padding_idx)

        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
        register_buffer("position_ids", Torch.arange(config.max_position_embeddings).expand([1, -1]), persistent: false)
      end

      def forward(input_ids: nil, position_ids: nil, inputs_embeds: nil, **kwargs)
        if position_ids.nil?
          if !input_ids.nil?
            position_ids = create_position_ids_from_input_ids(input_ids, @padding_idx)
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

        if position_ids.nil?
          position_ids = @position_ids[0.., ...seq_length]
        end

        if inputs_embeds.nil?
          inputs_embeds = @word_embeddings.(input_ids)
        end
        position_embeddings = @position_embeddings.(position_ids)

        embeddings = inputs_embeds + position_embeddings
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

      def create_position_ids_from_input_ids(input_ids, padding_idx)
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int
        incremental_indices = Torch.cumsum(mask, dim: 1).type_as(mask) * mask
        incremental_indices.long + padding_idx
      end
    end

    class MPNetSelfAttention < Torch::NN::Module
      def initialize(config)
        super()
        if config.hidden_size % config.num_attention_heads != 0 && !config.instance_variable_defined?(:@embedding_size)
          raise ArgumentError, "The hidden size (#{config.hidden_size}) is not a multiple of the number of attention heads (#{config.num_attention_heads})"
        end

        @num_attention_heads = config.num_attention_heads
        @attention_head_size = (config.hidden_size / config.num_attention_heads).to_i
        @all_head_size = @num_attention_heads * @attention_head_size

        @q = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
        @k = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
        @v = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
        @o = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)

        @dropout = Torch::NN::Dropout.new(p: config.attention_probs_dropout_prob)
      end

      def transpose_for_scores(x)
        new_x_shape = x.size[...-1] + [@num_attention_heads, @attention_head_size]
        x = x.view(*new_x_shape)
        x.permute(0, 2, 1, 3)
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        position_bias: nil,
        output_attentions: false,
        **kwargs
      )
        q = @q.(hidden_states)
        k = @k.(hidden_states)
        v = @v.(hidden_states)

        q = transpose_for_scores(q)
        k = transpose_for_scores(k)
        v = transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = Torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / Math.sqrt(@attention_head_size)

        # Apply relative position embedding (precomputed in MPNetEncoder) if provided.
        if !position_bias.nil?
          attention_scores += position_bias
        end

        if !attention_mask.nil?
          attention_scores = attention_scores + attention_mask
        end

        # Normalize the attention scores to probabilities.
        attention_probs = Torch::NN::Functional.softmax(attention_scores, dim: -1)

        attention_probs = @dropout.(attention_probs)

        if !head_mask.nil?
          attention_probs = attention_probs * head_mask
        end

        c = Torch.matmul(attention_probs, v)

        c = c.permute(0, 2, 1, 3).contiguous
        new_c_shape = c.size[...-2] + [@all_head_size]
        c = c.view(*new_c_shape)

        o = @o.(c)

        outputs = output_attentions ? [o, attention_probs] : [o]
        outputs
      end
    end

    class MPNetAttention < Torch::NN::Module
      def initialize(config)
        super()
        @attn = MPNetSelfAttention.new(config)
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)

        @pruned_heads = Set.new
      end

      def prune_heads(heads)
        if heads.length == 0
          return
        end
        heads, index = TorchUtils.find_pruneable_heads_and_indices(heads, @attn.num_attention_heads, @attn.attention_head_size, @pruned_heads)

        @q = TorchUtils.prune_linear_layer(@attn.q, index)
        @k = TorchUtils.prune_linear_layer(@attn.k, index)
        @v = TorchUtils.prune_linear_layer(@attn.v, index)
        @o = TorchUtils.prune_linear_layer(@attn.o, index, dim: 1)

        @num_attention_heads = @attn.num_attention_heads - heads.length
        @all_head_size = @attn.attention_head_size * @attn.num_attention_heads
        @pruned_heads = @pruned_heads.union(heads)
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        position_bias: nil,
        output_attentions: false,
        **kwargs
      )
        self_outputs = @attn.(hidden_states, attention_mask: attention_mask, head_mask: head_mask, position_bias: position_bias, output_attentions: output_attentions)
        attention_output = @LayerNorm.(@dropout.(self_outputs[0]) + hidden_states)
        outputs = [attention_output] + self_outputs[1..]
        outputs
      end
    end

    class MPNetIntermediate < Torch::NN::Module
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

    class MPNetOutput < Torch::NN::Module
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

    class MPNetLayer < Torch::NN::Module
      def initialize(config)
        super()
        @attention = MPNetAttention.new(config)
        @intermediate = MPNetIntermediate.new(config)
        @output = MPNetOutput.new(config)
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        position_bias: nil,
        output_attentions: false,
        **kwargs
      )
        self_attention_outputs = @attention.(hidden_states, attention_mask: attention_mask, head_mask: head_mask, position_bias: position_bias, output_attentions: output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1..]

        intermediate_output = @intermediate.(attention_output)
        layer_output = @output.(intermediate_output, attention_output)
        outputs = [layer_output] + outputs
        outputs
      end
    end

    class MPNetEncoder < Torch::NN::Module
      def initialize(config)
        super()
        @config = config
        @n_heads = config.num_attention_heads
        @layer = Torch::NN::ModuleList.new(config.num_hidden_layers.times.map { |_| MPNetLayer.new(config) })
        @relative_attention_bias = Torch::NN::Embedding.new(config.relative_attention_num_buckets, @n_heads)
      end

      def forward(
        hidden_states,
        attention_mask: nil,
        head_mask: nil,
        output_attentions: false,
        output_hidden_states: false,
        return_dict: false,
        **kwargs
      )
        position_bias = compute_position_bias(hidden_states)
        all_hidden_states = output_hidden_states ? [] : nil
        all_attentions = output_attentions ? [] : nil
        @layer.each_with_index do |layer_module, i|
          if output_hidden_states
            all_hidden_states = all_hidden_states + [hidden_states]
          end

          layer_outputs = layer_module.(hidden_states, attention_mask: attention_mask, head_mask: head_mask[i], position_bias: position_bias, output_attentions: output_attentions, **kwargs)
          hidden_states = layer_outputs[0]

          if output_attentions
            all_attentions = all_attentions + [layer_outputs[1]]
          end
        end

        # Add last layer
        if output_hidden_states
          all_hidden_states = all_hidden_states + [hidden_states]
        end

        if !return_dict
          return Array([hidden_states, all_hidden_states, all_attentions].select { |v| !v.nil? })
        end
        BaseModelOutput.new(last_hidden_state: hidden_states, hidden_states: all_hidden_states, attentions: all_attentions)
      end

      def compute_position_bias(x, position_ids: nil, num_buckets: 32)
        bsz, qlen, klen = [x.size(0), x.size(1), x.size(1)]
        if !position_ids.nil?
          context_position = position_ids[0.., 0.., nil]
          memory_position = position_ids[0.., nil, 0..]
        else
          context_position = Torch.arange(qlen, dtype: Torch.long)[0.., nil]
          memory_position = Torch.arange(klen, dtype: Torch.long)[nil, 0..]
        end

        relative_position = memory_position - context_position

        rp_bucket = self.class.relative_position_bucket(relative_position, num_buckets: num_buckets)
        rp_bucket = rp_bucket.to(x.device)
        values = @relative_attention_bias.(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand([bsz, -1, qlen, klen]).contiguous
        values
      end

      def self.relative_position_bucket(relative_position, num_buckets: 32, max_distance: 128)
        ret = 0
        n = -relative_position

        num_buckets /= 2
        ret += n.lt(0).to(Torch.long) * num_buckets
        n = Torch.abs(n)

        max_exact = num_buckets / 2
        is_small = n.lt(max_exact)

        val_if_large = max_exact + (
          Torch.log(n.float / max_exact) / Math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(Torch.long)

        val_if_large = Torch.min(val_if_large, Torch.full_like(val_if_large, num_buckets - 1))
        ret += Torch.where(is_small, n, val_if_large)
        ret
      end
    end

    class MPNetPooler < Torch::NN::Module
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

    class MPNetModel < MPNetPreTrainedModel
      def initialize(config, add_pooling_layer: true)
        super(config)
        @config = config

        @embeddings = MPNetEmbeddings.new(config)
        @encoder = MPNetEncoder.new(config)
        @pooler = add_pooling_layer ? MPNetPooler.new(config) : nil

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
        input_ids: nil,
        attention_mask: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil,
        **kwargs
      )
        output_attentions = !output_attentions.nil? ? output_attentions : @config.output_attentions
        output_hidden_states = !output_hidden_states.nil? ? output_hidden_states : @config.output_hidden_states
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

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

        device = !input_ids.nil? ? input_ids.device : inputs_embeds.device

        if attention_mask.nil?
          attention_mask = Torch.ones(input_shape, device: device)
        end
        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape)

        head_mask = get_head_mask(head_mask, @config.num_hidden_layers)
        embedding_output = @embeddings.(input_ids: input_ids, position_ids: position_ids, inputs_embeds: inputs_embeds)
        encoder_outputs = @encoder.(embedding_output, attention_mask: extended_attention_mask, head_mask: head_mask, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = !@pooler.nil? ? @pooler.(sequence_output) : nil

        if !return_dict
          return [sequence_output, pooled_output] + encoder_outputs[1..]
        end

        BaseModelOutputWithPooling.new(last_hidden_state: sequence_output, pooler_output: pooled_output, hidden_states: encoder_outputs.hidden_states, attentions: encoder_outputs.attentions)
      end
    end

    class MPNetForMaskedLM < MPNetPreTrainedModel
      self._tied_weights_keys = ["lm_head.decoder"]

      def initialize(config)
        super(config)

        @mpnet = MPNetModel.new(config, add_pooling_layer: false)
        @lm_head = MPNetLMHead.new(config)

        # Initialize weights and apply final processing
        post_init
      end

      def get_output_embeddings
        @lm_head.decoder
      end

      def set_output_embeddings(new_embeddings)
        @decoder = new_embeddings
        @bias = new_embeddings.bias
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @mpnet.(input_ids, attention_mask: attention_mask, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        sequence_output = outputs[0]
        prediction_scores = @lm_head.(sequence_output)

        masked_lm_loss = nil
        if !labels.nil?
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

    class MPNetLMHead < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        @layer_norm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)

        @decoder = Torch::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
        @bias = Torch::NN::Parameter.new(Torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        @bias = @bias
      end

      def _tie_weights
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
    end

    class MPNetForSequenceClassification < MPNetPreTrainedModel
      def initialize(config)
        super(config)

        @num_labels = config.num_labels
        @mpnet = MPNetModel.new(config, add_pooling_layer: false)
        @classifier = MPNetClassificationHead.new(config)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @mpnet.(input_ids, attention_mask: attention_mask, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)
        sequence_output = outputs[0]
        logits = @classifier.(sequence_output)

        loss = nil
        if !labels.nil?
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

    class MPNetForMultipleChoice < MPNetPreTrainedModel
      def initialize(config)
        super(config)

        @mpnet = MPNetModel.new(config)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
        @classifier = Torch::NN::Linear.new(config.hidden_size, 1)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict
        num_choices = !input_ids.nil? ? input_ids.shape[1] : inputs_embeds.shape[1]

        flat_input_ids = !input_ids.nil? ? input_ids.view(-1, input_ids.size(-1)) : nil
        flat_position_ids = !position_ids.nil? ? position_ids.view(-1, position_ids.size(-1)) : nil
        flat_attention_mask = !attention_mask.nil? ? attention_mask.view(-1, attention_mask.size(-1)) : nil
        flat_inputs_embeds = !inputs_embeds.nil? ? inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) : nil

        outputs = @mpnet.(flat_input_ids, position_ids: flat_position_ids, attention_mask: flat_attention_mask, head_mask: head_mask, inputs_embeds: flat_inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)
        pooled_output = outputs[1]

        pooled_output = @dropout.(pooled_output)
        logits = @classifier.(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = nil
        if !labels.nil?
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

    class MPNetForTokenClassification < MPNetPreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels

        @mpnet = MPNetModel.new(config, add_pooling_layer: false)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
        @classifier = Torch::NN::Linear.new(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        position_ids: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @mpnet.(input_ids, attention_mask: attention_mask, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        sequence_output = outputs[0]

        sequence_output = @dropout.(sequence_output)
        logits = @classifier.(sequence_output)

        loss = nil
        if !labels.nil?
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

    class MPNetClassificationHead < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        @dropout = Torch::NN::Dropout.new(p: config.hidden_dropout_prob)
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

    class MPNetForQuestionAnswering < MPNetPreTrainedModel
      def initialize(config)
        super(config)

        @num_labels = config.num_labels
        @mpnet = MPNetModel.new(config, add_pooling_layer: false)
        @qa_outputs = Torch::NN::Linear.new(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
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

        outputs = @mpnet.(input_ids, attention_mask: attention_mask, position_ids: position_ids, head_mask: head_mask, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

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

  MPNetForMaskedLM = Mpnet::MPNetForMaskedLM
end
