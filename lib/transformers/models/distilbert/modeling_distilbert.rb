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
    class Embeddings < Torch::NN::Module
      def initialize(config)
        super()
        @word_embeddings = Torch::NN::Embedding.new(config.vocab_size, config.dim, padding_idx: config.pad_token_id)
        @position_embeddings = Torch::NN::Embedding.new(config.max_position_embeddings, config.dim)

        @LayerNorm = Torch::NN::LayerNorm.new(config.dim, eps: 1e-12)
        @dropout = Torch::NN::Dropout.new(p: config.dropout)
        register_buffer(
          "position_ids", Torch.arange(config.max_position_embeddings).expand([1, -1]), persistent: false
        )
      end

      def forward(input_ids, input_embeds)
        if !input_ids.nil?
          input_embeds = @word_embeddings.(input_ids)  # (bs, max_seq_length, dim)
        end

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if @position_ids
          position_ids = @position_ids[0.., 0...seq_length]
        else
          position_ids = Torch.arange(seq_length, dtype: :long, device: input_ids.device)  # (max_seq_length)
          position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        end

        position_embeddings = @position_embeddings.(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = @LayerNorm.(embeddings)  # (bs, max_seq_length, dim)
        embeddings = @dropout.(embeddings)  # (bs, max_seq_length, dim)
        embeddings
      end
    end

    class MultiHeadSelfAttention < Torch::NN::Module
      def initialize(config)
        super()
        @config = config

        @n_heads = config.n_heads
        @dim = config.dim
        @dropout = Torch::NN::Dropout.new(p: config.attention_dropout)
        @is_causal = false

        # Have an even number of multi heads that divide the dimensions
        if @dim % @n_heads != 0
          # Raise value errors for even multi-head attention nodes
          raise ArgumentError, "self.n_heads: #{@n_heads} must divide self.dim: #{@dim} evenly"
        end

        @q_lin = Torch::NN::Linear.new(config.dim, config.dim)
        @k_lin = Torch::NN::Linear.new(config.dim, config.dim)
        @v_lin = Torch::NN::Linear.new(config.dim, config.dim)
        @out_lin = Torch::NN::Linear.new(config.dim, config.dim)

        @pruned_heads = Set.new
        @attention_head_size = @dim.div(@n_heads)
      end

      def prune_heads(heads)
        if heads.length == 0
          return
        end
        raise Todo
      end

      def forward(
        query:,
        key:,
        value:,
        mask:,
        head_mask: nil,
        output_attentions: false
      )
        bs, _q_length, dim = query.size
        k_length = key.size(1)
        if dim != @dim
          raise "Dimensions do not match: #{dim} input vs #{@dim} configured"
        end
        if key.size != value.size
          raise Todo
        end

        dim_per_head = @dim.div(@n_heads)

        mask_reshp = [bs, 1, 1, k_length]

        shape = lambda do |x|
          x.view(bs, -1, @n_heads, dim_per_head).transpose(1, 2)
        end

        unshape = lambda do |x|
          x.transpose(1, 2).contiguous.view(bs, -1, @n_heads * dim_per_head)
        end

        q = shape.(@q_lin.(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape.(@k_lin.(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape.(@v_lin.(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / Math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = Torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask.eq(0)).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores =
          scores.masked_fill(
            # TODO use Torch.finfo
            mask, Torch.tensor(0)
          )  # (bs, n_heads, q_length, k_length)

        weights = Torch::NN::Functional.softmax(scores, dim: -1)  # (bs, n_heads, q_length, k_length)
        weights = @dropout.(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if !head_mask.nil?
          weights = weights * head_mask
        end

        context = Torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape.(context)  # (bs, q_length, dim)
        context = @out_lin.(context)  # (bs, q_length, dim)

        if output_attentions
          [context, weights]
        else
          [context]
        end
      end
    end

    class DistilBertFlashAttention2 < MultiHeadSelfAttention
    end

    class FFN < Torch::NN::Module
      def initialize(config)
        super()
        @dropout = Torch::NN::Dropout.new(p: config.dropout)
        @chunk_size_feed_forward = config.chunk_size_feed_forward
        @seq_len_dim = 1
        @lin1 = Torch::NN::Linear.new(config.dim, config.hidden_dim)
        @lin2 = Torch::NN::Linear.new(config.hidden_dim, config.dim)
        @activation = Activations.get_activation(config.activation)
      end

      def forward(input)
        TorchUtils.apply_chunking_to_forward(method(:ff_chunk), @chunk_size_feed_forward, @seq_len_dim, input)
      end

      def ff_chunk(input)
        x = @lin1.(input)
        x = @activation.(x)
        x = @lin2.(x)
        x = @dropout.(x)
        x
      end
    end

    DISTILBERT_ATTENTION_CLASSES = {
      "eager" => MultiHeadSelfAttention,
      "flash_attention_2" => DistilBertFlashAttention2
    }

    class TransformerBlock < Torch::NN::Module
      def initialize(config)
        super()

        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0
          raise ArgumentError, "config.n_heads #{config.n_heads} must divide config.dim #{config.dim} evenly"
        end

        @attention = DISTILBERT_ATTENTION_CLASSES[config._attn_implementation].new(config)
        @sa_layer_norm = Torch::NN::LayerNorm.new(config.dim, eps: 1e-12)

        @ffn = FFN.new(config)
        @output_layer_norm = Torch::NN::LayerNorm.new(config.dim, eps: 1e-12)
      end

      def forward(
        x:,
        attn_mask: nil,
        head_mask: nil,
        output_attentions: false
      )
        # Self-Attention
        sa_output =
          @attention.(
            query: x,
            key: x,
            value: x,
            mask: attn_mask,
            head_mask: head_mask,
            output_attentions: output_attentions,
          )
        if output_attentions
          sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
          if !sa_output.is_a?(Array)
            raise TypeError, "sa_output must be an array but it is #{sa_output.class.name} type"
          end

          sa_output = sa_output[0]
        end
        sa_output = @sa_layer_norm.(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = @ffn.(sa_output)  # (bs, seq_length, dim)
        ffn_output = @output_layer_norm.(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = [ffn_output]
        if output_attentions
          output = [sa_weights] + output
        end
        output
      end
    end

    class Transformer < Torch::NN::Module
      def initialize(config)
        super()
        @n_layers = config.n_layers
        @layer = Torch::NN::ModuleList.new(config.n_layers.times.map { TransformerBlock.new(config) })
        @gradient_checkpointing = false
      end

      def forward(
        x:,
        attn_mask: nil,
        head_mask: nil,
        output_attentions: false,
        output_hidden_states: false,
        return_dict: nil
      )
        all_hidden_states = output_hidden_states ? [] : nil
        all_attentions = output_attentions ? [] : nil

        hidden_state = x
        @layer.each_with_index do |layer_module, i|
          if output_hidden_states
            all_hidden_states = all_hidden_states + [hidden_state]
          end

          if @gradient_checkpointing && training
            layer_outputs =
              _gradient_checkpointing_func(
                layer_module.__call__,
                hidden_state,
                attn_mask,
                head_mask[i],
                output_attentions,
              )
          else
            layer_outputs =
              layer_module.(
                x: hidden_state,
                attn_mask: attn_mask,
                head_mask: head_mask[i],
                output_attentions: output_attentions
              )
          end

          hidden_state = layer_outputs[-1]

          if output_attentions
            if layer_outputs.length != 2
              raise ArgumentError, "The length of the layer_outputs should be 2, but it is #{layer_outputs.length}"
            end

            attentions = layer_outputs[0]
            all_attentions = all_attentions + [attentions]
          else
            if layer_outputs.length != 1
              raise ArgumentError, "The length of the layer_outputs should be 1, but it is #{layer_outputs.length}"
            end
          end
        end

        # Add last layer
        if output_hidden_states
          all_hidden_states = all_hidden_states + [hidden_state]
        end

        if !return_dict
          raise Todo
        end
        BaseModelOutput.new(
          last_hidden_state: hidden_state, hidden_states: all_hidden_states, attentions: all_attentions
        )
      end
    end

    class DistilBertPreTrainedModel < PreTrainedModel
      self.config_class = DistilBertConfig
      self.base_model_prefix = "distilbert"

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
        elsif mod.is_a?(Embeddings) && @config.sinusoidal_pos_embds
          create_sinusoidal_embeddings(
            @config.max_position_embeddings, @config.dim, mod.position_embeddings.weight
          )
        end
      end

      private

      def create_sinusoidal_embeddings(n_pos, dim, out)
        # TODO
      end
    end

    class DistilBertModel < DistilBertPreTrainedModel
      def initialize(config)
        super(config)

        @embeddings = Embeddings.new(config)  # Embeddings
        @transformer = Transformer.new(config)  # Encoder
        @use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # Initialize weights and apply final processing
        post_init
      end

      def get_position_embeddings
        @embeddings.position_embeddings
      end

      def get_input_embeddings
        @embeddings.word_embeddings
      end

      def _prune_heads(heads_to_prune)
        heads_to_prune.each do |layer, heads|
          @transformer.layer[layer].attention.prune_heads(heads)
        end
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        head_mask: nil,
        inputs_embeds: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        output_attentions = !output_attentions.nil? ? output_attentions : @config.output_attentions
        output_hidden_states = (
          !output_hidden_states.nil? ? output_hidden_states : @config.output_hidden_states
        )
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

        # Prepare head mask if needed
        head_mask = get_head_mask(head_mask, @config.num_hidden_layers)

        embeddings = @embeddings.(input_ids, inputs_embeds)  # (bs, seq_length, dim)

        if @use_flash_attention_2
          raise Todo
        else
          if attention_mask.nil?
            attention_mask = Torch.ones(input_shape, device: device)  # (bs, seq_length)
          end
        end

        @transformer.(
          x: embeddings,
          attn_mask: attention_mask,
          head_mask: head_mask,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          return_dict: return_dict
        )
      end
    end

    class DistilBertForMaskedLM < DistilBertPreTrainedModel
      self._tied_weights_keys = ["vocab_projector.weight"]

      def initialize(config)
        super(config)

        @activation = get_activation(config.activation)

        @distilbert = DistilBertModel.new(config)
        @vocab_transform = Torch::NN::Linear.new(config.dim, config.dim)
        @vocab_layer_norm = Torch::NN::LayerNorm.new(config.dim, eps: 1e-12)
        @vocab_projector = Torch::NN::Linear.new(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        post_init

        @mlm_loss_fct = Torch::NN::CrossEntropyLoss.new
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        dlbrt_output = @distilbert.(
          input_ids: input_ids,
          attention_mask: attention_mask,
          head_mask: head_mask,
          inputs_embeds: inputs_embeds,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          return_dict: return_dict
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = @vocab_transform.(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = @activation.(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = @vocab_layer_norm.(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = @vocab_projector.(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = nil
        if !labels.nil?
          mlm_loss = @mlm_loss_fct.(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        end

        if !return_dict
          raise Todo
        end

        MaskedLMOutput.new(
          loss: mlm_loss,
          logits: prediction_logits,
          hidden_states: dlbrt_output.hidden_states,
          attentions: dlbrt_output.attentions
        )
      end
    end

    class DistilBertForSequenceClassification < DistilBertPreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels
        @config = config

        @distilbert = DistilBertModel.new(config)
        @pre_classifier = Torch::NN::Linear.new(config.dim, config.dim)
        @classifier = Torch::NN::Linear.new(config.dim, config.num_labels)
        @dropout = Torch::NN::Dropout.new(p: config.seq_classif_dropout)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        head_mask: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        distilbert_output =
          @distilbert.(
            input_ids: input_ids,
            attention_mask: attention_mask,
            head_mask: head_mask,
            inputs_embeds: inputs_embeds,
            output_attentions: output_attentions,
            output_hidden_states: output_hidden_states,
            return_dict: return_dict
          )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[0.., 0]  # (bs, dim)
        pooled_output = @pre_classifier.(pooled_output)  # (bs, dim)
        pooled_output = Torch::NN::ReLU.new.(pooled_output)  # (bs, dim)
        pooled_output = @dropout.(pooled_output)  # (bs, dim)
        logits = @classifier.(pooled_output)  # (bs, num_labels)

        loss = nil
        if !labels.nil?
          raise Todo
        end

        if !return_dict
          raise Todo
        end

        SequenceClassifierOutput.new(
          loss: loss,
          logits: logits,
          hidden_states: distilbert_output.hidden_states,
          attentions: distilbert_output.attentions
        )
      end
    end

    class DistilBertForQuestionAnswering < DistilBertPreTrainedModel
      def initialize(config)
        super(config)

        @distilbert = DistilBertModel.new(config)
        @qa_outputs = Torch::NN::Linear.new(config.dim, config.num_labels)
        if config.num_labels != 2
          raise ArgumentError, "config.num_labels should be 2, but it is #{config.num_labels}"
        end

        @dropout = Torch::NN::Dropout.new(p: config.qa_dropout)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        head_mask: nil,
        inputs_embeds: nil,
        start_positions: nil,
        end_positions: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        distilbert_output = @distilbert.(
          input_ids: input_ids,
          attention_mask: attention_mask,
          head_mask: head_mask,
          inputs_embeds: inputs_embeds,
          output_attentions: output_attentions,
          output_hidden_states: output_hidden_states,
          return_dict: return_dict
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        hidden_states = @dropout.(hidden_states)  # (bs, max_query_len, dim)
        logits = @qa_outputs.(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim: -1)
        start_logits = start_logits.squeeze(-1).contiguous  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous  # (bs, max_query_len)

        total_loss = nil
        if !start_positions.nil? && !end_positions.nil?
          raise Todo
        end

        if !return_dict
          raise Todo
        end

        QuestionAnsweringModelOutput.new(
          loss: total_loss,
          start_logits: start_logits,
          end_logits: end_logits,
          hidden_states: distilbert_output.hidden_states,
          attentions: distilbert_output.attentions
        )
      end
    end
  end

  DistilBertForMaskedLM = Distilbert::DistilBertForMaskedLM
  DistilBertForSequenceClassification = Distilbert::DistilBertForSequenceClassification
  DistilBertForQuestionAnswering = Distilbert::DistilBertForQuestionAnswering
end
