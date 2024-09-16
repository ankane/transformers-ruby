# Copyright 2020 Microsoft and the Hugging Face Inc. team.
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
    class ContextPooler < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.pooler_hidden_size, config.pooler_hidden_size)
        @dropout = StableDropout.new(config.pooler_dropout)
        @config = config
      end

      def forward(hidden_states)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[0.., 0]
        context_token = @dropout.(context_token)
        pooled_output = @dense.(context_token)
        pooled_output = ACT2FN[@config.pooler_hidden_act].(pooled_output)
        pooled_output
      end

      def output_dim
        @config.hidden_size
      end
    end

    # TODO Torch::Autograd::Function
    class XSoftmax
      def self.apply(input, mask, dim)
        @dim = dim
        rmask = mask.to(Torch.bool).bitwise_not

        # TODO use Torch.finfo
        output = input.masked_fill(rmask, Torch.tensor(-3.40282e+38))
        output = Torch.softmax(output, @dim)
        output.masked_fill!(rmask, 0)
        # ctx.save_for_backward(output)
        output
      end
    end

    class DropoutContext
      def initialize
        @dropout = 0
        @mask = nil
        @scale = 1
        @reuse_mask = true
      end
    end

    def get_mask(input, local_context)
      if !local_context.is_a?(DropoutContext)
        dropout = local_context
        mask = nil
      else
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.reuse_mask ? local_context.mask : nil
      end

      if dropout > 0 && mask.nil?
        mask = (1 - Torch.empty_like(input).bernoulli!(1 - dropout)).to(Torch.bool)
      end

      if local_context.is_a?(DropoutContext)
        if local_context.mask.nil?
          @mask = mask
        end
      end

      [mask, dropout]
    end

    # TODO Torch::Autograd::Function
    class XDropout
      def self.apply(input, local_ctx)
        mask, dropout = get_mask(input, local_ctx)
        @scale = 1.0 / (1 - dropout)
        if dropout > 0
          # ctx.save_for_backward(mask)
          input.masked_fill(mask, 0) * ctx.scale
        else
          input
        end
      end
    end

    class StableDropout < Torch::NN::Module
      def initialize(drop_prob)
        super()
        @drop_prob = drop_prob
        @count = 0
        @context_stack = nil
      end

      def forward(x)
        if @training && @drop_prob > 0
          return XDropout.apply(x, get_context)
        end
        x
      end

      def clear_context
        @count = 0
        @context_stack = nil
      end

      def init_context(reuse_mask: true, scale: 1)
        if @context_stack.nil?
          @context_stack = []
        end
        @count = 0
        @context_stack.each do |c|
          @reuse_mask = reuse_mask
          @scale = scale
        end
      end

      def get_context
        if !@context_stack.nil?
          if @count >= @context_stack.length
            @context_stack << DropoutContext.new
          end
          ctx = @context_stack.fetch(@count)
          @dropout = @drop_prob
          @count += 1
          ctx
        else
          @drop_prob
        end
      end
    end

    class DebertaV2SelfOutput < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.hidden_size, config.hidden_size)
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = StableDropout.new(config.hidden_dropout_prob)
      end

      def forward(hidden_states, input_tensor)
        hidden_states = @dense.(hidden_states)
        hidden_states = @dropout.(hidden_states)
        hidden_states = @LayerNorm.(hidden_states + input_tensor)
        hidden_states
      end
    end

    class DebertaV2Attention < Torch::NN::Module
      def initialize(config)
        super()
        @self = DisentangledSelfAttention.new(config)
        @output = DebertaV2SelfOutput.new(config)
        @config = config
      end

      def forward(
        hidden_states,
        attention_mask,
        output_attentions: false,
        query_states: nil,
        relative_pos: nil,
        rel_embeddings: nil
      )
        self_output = @self.(hidden_states, attention_mask, output_attentions:, query_states: query_states, relative_pos: relative_pos, rel_embeddings: rel_embeddings)
        if output_attentions
          self_output, att_matrix = self_output
        end
        if query_states.nil?
          query_states = hidden_states
        end
        attention_output = @output.(self_output, query_states)

        if output_attentions
          [attention_output, att_matrix]
        else
          attention_output
        end
      end
    end

    class DebertaV2Intermediate < Torch::NN::Module
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

    class DebertaV2Output < Torch::NN::Module
      def initialize(config)
        super()
        @dense = Torch::NN::Linear.new(config.intermediate_size, config.hidden_size)
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = StableDropout.new(config.hidden_dropout_prob)
        @config = config
      end

      def forward(hidden_states, input_tensor)
        hidden_states = @dense.(hidden_states)
        hidden_states = @dropout.(hidden_states)
        hidden_states = @LayerNorm.(hidden_states + input_tensor)
        hidden_states
      end
    end

    class DebertaV2Layer < Torch::NN::Module
      def initialize(config)
        super()
        @attention = DebertaV2Attention.new(config)
        @intermediate = DebertaV2Intermediate.new(config)
        @output = DebertaV2Output.new(config)
      end

      def forward(
        hidden_states,
        attention_mask,
        query_states: nil,
        relative_pos: nil,
        rel_embeddings: nil,
        output_attentions: false
      )
        attention_output = @attention.(hidden_states, attention_mask, output_attentions: output_attentions, query_states: query_states, relative_pos: relative_pos, rel_embeddings: rel_embeddings)
        if output_attentions
          attention_output, att_matrix = attention_output
        end
        intermediate_output = @intermediate.(attention_output)
        layer_output = @output.(intermediate_output, attention_output)
        if output_attentions
          [layer_output, att_matrix]
        else
          layer_output
        end
      end
    end

    class ConvLayer < Torch::NN::Module
      def initialize(config)
        super()
        kernel_size = config.getattr("conv_kernel_size", 3)
        groups = config.getattr("conv_groups", 1)
        @conv_act = config.getattr("conv_act", "tanh")
        @conv = Torch::NN::Conv1d.new(config.hidden_size, config.hidden_size, kernel_size, padding: (kernel_size - 1) / 2, groups: groups)
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = StableDropout.new(config.hidden_dropout_prob)
        @config = config
      end

      def forward(hidden_states, residual_states, input_mask)
        out = @conv.(hidden_states.permute(0, 2, 1).contiguous).permute(0, 2, 1).contiguous
        rmask = (1 - input_mask).bool
        out.masked_fill!(rmask.unsqueeze(-1).expand(out.size), 0)
        out = ACT2FN[@conv_act].(@dropout.(out))

        layer_norm_input = residual_states + out
        output = @LayerNorm.(layer_norm_input).to(layer_norm_input)

        if input_mask.nil?
          output_states = output
        elsif input_mask.dim != layer_norm_input.dim
          if input_mask.dim == 4
            input_mask = input_mask.squeeze(1).squeeze(1)
          end
          input_mask = input_mask.unsqueeze(2)
        end

        output_states
      end
    end

    class DebertaV2Encoder < Torch::NN::Module
      def initialize(config)
        super()

        @layer = Torch::NN::ModuleList.new(config.num_hidden_layers.times.map { |_| DebertaV2Layer.new(config) })
        @relative_attention = config.getattr("relative_attention", false)

        if @relative_attention
          @max_relative_positions = config.getattr("max_relative_positions", -1)
          if @max_relative_positions < 1
            @max_relative_positions = config.max_position_embeddings
          end

          @position_buckets = config.getattr("position_buckets", -1)
          pos_ebd_size = @max_relative_positions * 2

          if @position_buckets > 0
            pos_ebd_size = @position_buckets * 2
          end

          @rel_embeddings = Torch::NN::Embedding.new(pos_ebd_size, config.hidden_size)
        end

        @norm_rel_ebd = config.getattr("norm_rel_ebd", "none").downcase.split("|").map { |x| x.strip }

        if @norm_rel_ebd.include?("layer_norm")
          @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps, elementwise_affine: true)
        end

        @conv = config.getattr("conv_kernel_size", 0) > 0 ? ConvLayer.new(config) : nil
        @gradient_checkpointing = false
      end

      def get_rel_embedding
        rel_embeddings = @relative_attention ? @rel_embeddings.weight : nil
        if !rel_embeddings.nil? && @norm_rel_ebd.include?("layer_norm")
          rel_embeddings = @LayerNorm.(rel_embeddings)
        end
        rel_embeddings
      end

      def get_attention_mask(attention_mask)
        if attention_mask.dim <= 2
          extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
          attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elsif attention_mask.dim == 3
          attention_mask = attention_mask.unsqueeze(1)
        end

        attention_mask
      end

      def get_rel_pos(hidden_states, query_states: nil, relative_pos: nil)
        if @relative_attention && relative_pos.nil?
          q = !query_states.nil? ? query_states.size(-2) : hidden_states.size(-2)
          relative_pos = DebertaV2.build_relative_position(q, hidden_states.size(-2), bucket_size: @position_buckets, max_position: @max_relative_positions, device: hidden_states.device)
        end
        relative_pos
      end

      def forward(
        hidden_states,
        attention_mask,
        output_hidden_states: true,
        output_attentions: false,
        query_states: nil,
        relative_pos: nil,
        return_dict: true
      )
        if attention_mask.dim <= 2
          input_mask = attention_mask
        else
          input_mask = attention_mask.sum(-2) > 0
        end
        attention_mask = get_attention_mask(attention_mask)
        relative_pos = get_rel_pos(hidden_states, query_states:, relative_pos:)

        all_hidden_states = output_hidden_states ? [] : nil
        all_attentions = output_attentions ? [] : nil

        if hidden_states.is_a?(Array)
          next_kv = hidden_states[0]
        else
          next_kv = hidden_states
        end
        rel_embeddings = get_rel_embedding
        output_states = next_kv
        @layer.each_with_index do |layer_module, i|
          if output_hidden_states
            all_hidden_states = all_hidden_states + [output_states]
          end

          if @gradient_checkpointing && @training
            output_states = _gradient_checkpointing_func(layer_module.__call__, next_kv, attention_mask, query_states, relative_pos, rel_embeddings, output_attentions)
          else
            output_states = layer_module.(next_kv, attention_mask, query_states: query_states, relative_pos: relative_pos, rel_embeddings: rel_embeddings, output_attentions: output_attentions)
          end

          if output_attentions
            output_states, att_m = output_states
          end

          if i == 0 && !@conv.nil?
            output_states = @conv.(hidden_states, output_states, input_mask)
          end

          if !query_states.nil?
            query_states = output_states
            if hidden_states.is_a?(Array)
              next_kv = i + 1 < @layer.length ? hidden_states[i + 1] : nil
            end
          else
            next_kv = output_states
          end

          if output_attentions
            all_attentions = all_attentions + [att_m]
          end
        end

        if output_hidden_states
          all_hidden_states = all_hidden_states + [output_states]
        end

        if !return_dict
          return Array([output_states, all_hidden_states, all_attentions].select { |v| !v.nil? })
        end
        BaseModelOutput.new(last_hidden_state: output_states, hidden_states: all_hidden_states, attentions: all_attentions)
      end
    end

    def self.make_log_bucket_position(relative_pos, bucket_size, max_position)
      sign = Torch.sign(relative_pos)
      mid = bucket_size / 2
      abs_pos = Torch.where(relative_pos.lt(mid) & relative_pos.gt(-mid), Torch.tensor(mid - 1).type_as(relative_pos), Torch.abs(relative_pos))
      log_pos = Torch.ceil((Torch.log(abs_pos / mid) / Torch.log(Torch.tensor((max_position - 1) / mid))) * (mid - 1)) + mid
      bucket_pos = Torch.where(abs_pos.le(mid), relative_pos.type_as(log_pos), log_pos * sign)
      bucket_pos
    end

    def self.build_relative_position(query_size, key_size, bucket_size: -1, max_position: -1, device: nil)
      q_ids = Torch.arange(0, query_size, device: device)
      k_ids = Torch.arange(0, key_size, device: device)
      rel_pos_ids = q_ids[0.., nil] - k_ids[nil, 0..]
      if bucket_size > 0 && max_position > 0
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
      end
      rel_pos_ids = rel_pos_ids.to(Torch.long)
      rel_pos_ids = rel_pos_ids[...query_size, 0..]
      rel_pos_ids = rel_pos_ids.unsqueeze(0)
      rel_pos_ids
    end

    def self.c2p_dynamic_expand(c2p_pos, query_layer, relative_pos)
      c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])
    end

    def self.p2c_dynamic_expand(c2p_pos, query_layer, key_layer)
      c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])
    end

    def self.pos_dynamic_expand(pos_index, p2c_att, key_layer)
      pos_index.expand(p2c_att.size[...2] + [pos_index.size(-2), key_layer.size(-2)])
    end

    class DisentangledSelfAttention < Torch::NN::Module
      def initialize(config)
        super()
        if config.hidden_size % config.num_attention_heads != 0
          raise ArgumentError, "The hidden size (#{config.hidden_size}) is not a multiple of the number of attention heads (#{config.num_attention_heads})"
        end
        @num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size / config.num_attention_heads
        @attention_head_size = config.getattr("attention_head_size", _attention_head_size)
        @all_head_size = @num_attention_heads * @attention_head_size
        @query_proj = Torch::NN::Linear.new(config.hidden_size, @all_head_size, bias: true)
        @key_proj = Torch::NN::Linear.new(config.hidden_size, @all_head_size, bias: true)
        @value_proj = Torch::NN::Linear.new(config.hidden_size, @all_head_size, bias: true)

        @share_att_key = config.getattr("share_att_key", false)
        @pos_att_type = !config.pos_att_type.nil? ? config.pos_att_type : []
        @relative_attention = config.getattr("relative_attention", false)

        if @relative_attention
          @position_buckets = config.getattr("position_buckets", -1)
          @max_relative_positions = config.getattr("max_relative_positions", -1)
          if @max_relative_positions < 1
            @max_relative_positions = config.max_position_embeddings
          end
          @pos_ebd_size = @max_relative_positions
          if @position_buckets > 0
            @pos_ebd_size = @position_buckets
          end

          @pos_dropout = StableDropout.new(config.hidden_dropout_prob)

          if !@share_att_key
            if @pos_att_type.include?("c2p")
              @pos_key_proj = Torch::NN::Linear.new(config.hidden_size, @all_head_size, bias: true)
            end
            if @pos_att_type.include?("p2c")
              @pos_query_proj = Torch::NN::Linear.new(config.hidden_size, @all_head_size)
            end
          end
        end

        @dropout = StableDropout.new(config.attention_probs_dropout_prob)
      end

      def transpose_for_scores(x, attention_heads)
        new_x_shape = x.size[...-1] + [attention_heads, -1]
        x = x.view(new_x_shape)
        x.permute(0, 2, 1, 3).contiguous.view(-1, x.size(1), x.size(-1))
      end

      def forward(
        hidden_states,
        attention_mask,
        output_attentions: false,
        query_states: nil,
        relative_pos: nil,
        rel_embeddings: nil
      )
        if query_states.nil?
          query_states = hidden_states
        end
        query_layer = transpose_for_scores(@query_proj.(query_states), @num_attention_heads)
        key_layer = transpose_for_scores(@key_proj.(hidden_states), @num_attention_heads)
        value_layer = transpose_for_scores(@value_proj.(hidden_states), @num_attention_heads)

        rel_att = nil
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if @pos_att_type.include?("c2p")
          scale_factor += 1
        end
        if @pos_att_type.include?("p2c")
          scale_factor += 1
        end
        scale = Torch.sqrt(Torch.tensor(query_layer.size(-1), dtype: Torch.float) * scale_factor)
        attention_scores = Torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype: query_layer.dtype))
        if @relative_attention
          rel_embeddings = @pos_dropout.(rel_embeddings)
          rel_att = disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        end

        if !rel_att.nil?
          attention_scores = attention_scores + rel_att
        end
        attention_scores = attention_scores
        attention_scores = attention_scores.view(-1, @num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))

        # bsz x height x length x dimension
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = @dropout.(attention_probs)
        context_layer = Torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, @num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous
        new_context_layer_shape = context_layer.size[...-2] + [-1]
        context_layer = context_layer.view(new_context_layer_shape)
        if output_attentions
          [context_layer, attention_probs]
        else
          context_layer
        end
      end

      def disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        if relative_pos.nil?
          q = query_layer.size(-2)
          relative_pos = DebertaV2.build_relative_position(q, key_layer.size(-2), bucket_size: @position_buckets, max_position: @max_relative_positions, device: query_layer.device)
        end
        if relative_pos.dim == 2
          relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elsif relative_pos.dim == 3
          relative_pos = relative_pos.unsqueeze(1)
        elsif relative_pos.dim != 4
          raise ArgumentError, "Relative position ids must be of dim 2 or 3 or 4. #{relative_pos.dim}"
        end

        att_span = @pos_ebd_size
        relative_pos = relative_pos.long.to(query_layer.device)

        rel_embeddings = rel_embeddings[0...att_span * 2, 0..].unsqueeze(0)
        if @share_att_key
          pos_query_layer = transpose_for_scores(@query_proj.(rel_embeddings), @num_attention_heads).repeat(query_layer.size(0) / @num_attention_heads, 1, 1)
          pos_key_layer = transpose_for_scores(@key_proj.(rel_embeddings), @num_attention_heads).repeat(query_layer.size(0) / @num_attention_heads, 1, 1)
        elsif @pos_att_type.include?("c2p")
          pos_key_layer = transpose_for_scores(@pos_key_proj.(rel_embeddings), @num_attention_heads).repeat(query_layer.size(0) / @num_attention_heads, 1, 1)
        end

        score = 0
        # content->position
        if @pos_att_type.include?("c2p")
          scale = Torch.sqrt(Torch.tensor(pos_key_layer.size(-1), dtype: Torch.float) * scale_factor)
          c2p_att = Torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
          c2p_pos = Torch.clamp(relative_pos + att_span, 0, (att_span * 2) - 1)
          c2p_att = Torch.gather(c2p_att, dim: -1, index: c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
          score += c2p_att / scale.to(dtype: c2p_att.dtype)
        end

        # position->content
        if @pos_att_type.include?("p2c")
          scale = Torch.sqrt(Torch.tensor(pos_query_layer.size(-1), dtype: Torch.float) * scale_factor)
          if key_layer.size(-2) != query_layer.size(-2)
            r_pos = DebertaV2.build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size: @position_buckets, max_position: @max_relative_positions, device: query_layer.device)
            r_pos = r_pos.unsqueeze(0)
          else
            r_pos = relative_pos
          end

          p2c_pos = Torch.clamp(-r_pos + att_span, 0, (att_span * 2) - 1)
          p2c_att = Torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
          p2c_att = Torch.gather(p2c_att, dim: -1, index: p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1, -2)
          score += p2c_att / scale.to(dtype: p2c_att.dtype)
        end

        score
      end
    end

    class DebertaV2Embeddings < Torch::NN::Module
      def initialize(config)
        super()
        pad_token_id = config.getattr("pad_token_id", 0)
        @embedding_size = config.getattr("embedding_size", config.hidden_size)
        @word_embeddings = Torch::NN::Embedding.new(config.vocab_size, @embedding_size, padding_idx: pad_token_id)

        @position_biased_input = config.getattr("position_biased_input", true)
        if !@position_biased_input
          @position_embeddings = nil
        else
          @position_embeddings = Torch::NN::Embedding.new(config.max_position_embeddings, @embedding_size)
        end

        if config.type_vocab_size > 0
          @token_type_embeddings = Torch::NN::Embedding.new(config.type_vocab_size, @embedding_size)
        end

        if @embedding_size != config.hidden_size
          @embed_proj = Torch::NN::Linear.new(@embedding_size, config.hidden_size, bias: false)
        end
        @LayerNorm = Torch::NN::LayerNorm.new(config.hidden_size, eps: config.layer_norm_eps)
        @dropout = StableDropout.new(config.hidden_dropout_prob)
        @config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        register_buffer("position_ids", Torch.arange(config.max_position_embeddings).expand([1, -1]), persistent: false)
      end

      def forward(input_ids: nil, token_type_ids: nil, position_ids: nil, mask: nil, inputs_embeds: nil)
        if !input_ids.nil?
          input_shape = input_ids.size
        else
          input_shape = inputs_embeds.size[...-1]
        end

        seq_length = input_shape[1]

        if position_ids.nil?
          position_ids = @position_ids[0.., ...seq_length]
        end

        if token_type_ids.nil?
          token_type_ids = Torch.zeros(input_shape, dtype: Torch.long, device: @position_ids.device)
        end

        if inputs_embeds.nil?
          inputs_embeds = @word_embeddings.(input_ids)
        end

        if !@position_embeddings.nil?
          position_embeddings = @position_embeddings.(position_ids.long)
        else
          position_embeddings = Torch.zeros_like(inputs_embeds)
        end

        embeddings = inputs_embeds
        if @position_biased_input
          embeddings += position_embeddings
        end
        if @config.type_vocab_size > 0
          token_type_embeddings = @token_type_embeddings.(token_type_ids)
          embeddings += token_type_embeddings
        end

        if @embedding_size != @config.hidden_size
          embeddings = @embed_proj.(embeddings)
        end

        embeddings = @LayerNorm.(embeddings)

        if !mask.nil?
          if mask.dim != embeddings.dim
            if mask.dim == 4
              mask = mask.squeeze(1).squeeze(1)
            end
            mask = mask.unsqueeze(2)
          end
          mask = mask.to(embeddings.dtype)

          embeddings = embeddings * mask
        end

        embeddings = @dropout.(embeddings)
        embeddings
      end
    end

    class DebertaV2PreTrainedModel < PreTrainedModel
      self.config_class = DebertaV2Config
      self.base_model_prefix = "deberta"
      # self._keys_to_ignore_on_load_unexpected = ["position_embeddings"]
      # self.supports_gradient_checkpointing = true

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
        end
      end
    end

    class DebertaV2Model < DebertaV2PreTrainedModel
      def initialize(config)
        super(config)

        @embeddings = DebertaV2Embeddings.new(config)
        @encoder = DebertaV2Encoder.new(config)
        @z_steps = 0
        @config = config
        # Initialize weights and apply final processing
        post_init
      end

      def get_input_embeddings
        @embeddings.word_embeddings
      end

      def set_input_embeddings(new_embeddings)
        @word_embeddings = new_embeddings
      end

      def _prune_heads(heads_to_prune)
        raise NotImplementedError, "The prune function is not implemented in DeBERTa model."
      end

      def forward(
        input_ids,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        inputs_embeds: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
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
        if token_type_ids.nil?
          token_type_ids = Torch.zeros(input_shape, dtype: Torch.long, device: device)
        end

        embedding_output = @embeddings.(input_ids: input_ids, token_type_ids: token_type_ids, position_ids: position_ids, mask: attention_mask, inputs_embeds: inputs_embeds)

        encoder_outputs = @encoder.(embedding_output, attention_mask, output_hidden_states: true, output_attentions: output_attentions, return_dict: return_dict)
        encoded_layers = encoder_outputs[1]

        if @z_steps > 1
          hidden_states = encoded_layers[-2]
          layers = @z_steps.times.map { |_| @encoder.layer[-1] }
          query_states = encoded_layers[-1]
          rel_embeddings = @encoder.get_rel_embedding
          attention_mask = @encoder.get_attention_mask(attention_mask)
          rel_pos = @encoder.get_rel_pos(embedding_output)
          layers[1..].each do |layer|
            query_states = layer(hidden_states, attention_mask, output_attentions: false, query_states: query_states, relative_pos: rel_pos, rel_embeddings: rel_embeddings)
            encoded_layers << query_states
          end
        end

        sequence_output = encoded_layers[-1]

        if !return_dict
          return [sequence_output] + encoder_outputs[output_hidden_states ? 1 : 2..]
        end

        BaseModelOutput.new(last_hidden_state: sequence_output, hidden_states: output_hidden_states ? encoder_outputs.hidden_states : nil, attentions: encoder_outputs.attentions)
      end
    end

    class DebertaV2ForMaskedLM < DebertaV2PreTrainedModel
      self._tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

      def initialize(config)
        super(config)

        @deberta = DebertaV2Model.new(config)
        @cls = DebertaV2OnlyMLMHead.new(config)

        # Initialize weights and apply final processing
        post_init
      end

      def get_output_embeddings
        @cls.predictions.decoder
      end

      def set_output_embeddings(new_embeddings)
        @decoder = new_embeddings
        @bias = new_embeddings.bias
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @deberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        sequence_output = outputs[0]
        prediction_scores = @cls.(sequence_output)

        masked_lm_loss = nil
        if !labels.nil?
          loss_fct = Torch::NN::CrossEntropyLoss.new
          masked_lm_loss = loss_fct.(prediction_scores.view(-1, @config.vocab_size), labels.view(-1))
        end

        if !return_dict
          output = [prediction_scores] + outputs[1..]
          return !masked_lm_loss.nil? ? [masked_lm_loss] + output : output
        end

        MaskedLMOutput.new(loss: masked_lm_loss, logits: prediction_scores, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class DebertaV2PredictionHeadTransform < Torch::NN::Module
      def initialize(config)
        super()
        @embedding_size = config.getattr("embedding_size", config.hidden_size)

        @dense = Torch::NN::Linear.new(config.hidden_size, @embedding_size)
        if config.hidden_act.is_a?(String)
          @transform_act_fn = ACT2FN[config.hidden_act]
        else
          @transform_act_fn = config.hidden_act
        end
        @LayerNorm = Torch::NN::LayerNorm.new(@embedding_size, eps: config.layer_norm_eps)
      end

      def forward(hidden_states)
        hidden_states = @dense.(hidden_states)
        hidden_states = @transform_act_fn.(hidden_states)
        hidden_states = @LayerNorm.(hidden_states)
        hidden_states
      end
    end

    class DebertaV2LMPredictionHead < Torch::NN::Module
      def initialize(config)
        super()
        @transform = DebertaV2PredictionHeadTransform.new(config)

        @embedding_size = config.getattr("embedding_size", config.hidden_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        @decoder = Torch::NN::Linear.new(@embedding_size, config.vocab_size, bias: false)

        @bias = Torch::NN::Parameter.new(Torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        @bias = @bias
      end

      def _tie_weights
        @bias = @bias
      end

      def forward(hidden_states)
        hidden_states = @transform.(hidden_states)
        hidden_states = @decoder.(hidden_states)
        hidden_states
      end
    end

    class DebertaV2OnlyMLMHead < Torch::NN::Module
      def initialize(config)
        super()
        @predictions = DebertaV2LMPredictionHead.new(config)
      end

      def forward(sequence_output)
        prediction_scores = @predictions.(sequence_output)
        prediction_scores
      end
    end

    class DebertaV2ForSequenceClassification < DebertaV2PreTrainedModel
      def initialize(config)
        super(config)

        num_labels = config.getattr("num_labels", 2)
        @num_labels = num_labels

        @deberta = DebertaV2Model.new(config)
        @pooler = ContextPooler.new(config)
        output_dim = @pooler.output_dim

        @classifier = Torch::NN::Linear.new(output_dim, num_labels)
        drop_out = config.getattr("cls_dropout", nil)
        drop_out = drop_out.nil? ? @config.hidden_dropout_prob : drop_out
        @dropout = StableDropout.new(drop_out)

        # Initialize weights and apply final processing
        post_init
      end

      def get_input_embeddings
        @deberta.get_input_embeddings
      end

      def set_input_embeddings(new_embeddings)
        @deberta.set_input_embeddings(new_embeddings)
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )
        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @deberta.(input_ids, token_type_ids: token_type_ids, attention_mask: attention_mask, position_ids: position_ids, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        encoder_layer = outputs[0]
        pooled_output = @pooler.(encoder_layer)
        pooled_output = @dropout.(pooled_output)
        logits = @classifier.(pooled_output)

        loss = nil
        if !labels.nil?
          if @config.problem_type.nil?
            if @num_labels == 1
              # regression task
              loss_fn = Torch::NN::MSELoss.new
              logits = logits.view(-1).to(labels.dtype)
              loss = loss_fn.(logits, labels.view(-1))
            elsif labels.dim == 1 || labels.size(-1) == 1
              label_index = (labels >= 0).nonzero
              labels = labels.long
              if label_index.size(0) > 0
                labeled_logits = Torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                labels = Torch.gather(labels, 0, label_index.view(-1))
                loss_fct = Torch::NN::CrossEntropyLoss.new
                loss = loss_fct.(labeled_logits.view(-1, @num_labels).float, labels.view(-1))
              else
                loss = Torch.tensor(0).to(logits)
              end
            else
              log_softmax = Torch::NN::LogSoftmax.new(-1)
              loss = -(log_softmax.(logits) * labels).sum(-1).mean
            end
          elsif @config.problem_type == "regression"
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
          output = [logits] + outputs[1..]
          return !loss.nil? ? [loss] + output : output
        end

        SequenceClassifierOutput.new(loss: loss, logits: logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class DebertaV2ForTokenClassification < DebertaV2PreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels

        @deberta = DebertaV2Model.new(config)
        @dropout = Torch::NN::Dropout.new(config.hidden_dropout_prob)
        @classifier = Torch::NN::Linear.new(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        inputs_embeds: nil,
        labels: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )

        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @deberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        sequence_output = outputs[0]

        sequence_output = @dropout.(sequence_output)
        logits = @classifier.(sequence_output)

        loss = nil
        if !labels.nil?
          loss_fct = Torch::NN::CrossEntropyLoss.new
          loss = loss_fct.(logits.view(-1, @num_labels), labels.view(-1))
        end

        if !return_dict
          output = [logits] + outputs[1..]
          return !loss.nil? ? [loss] + output : output
        end

        TokenClassifierOutput.new(loss: loss, logits: logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class DebertaV2ForQuestionAnswering < DebertaV2PreTrainedModel
      def initialize(config)
        super(config)
        @num_labels = config.num_labels

        @deberta = DebertaV2Model.new(config)
        @qa_outputs = Torch::NN::Linear.new(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        post_init
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
        inputs_embeds: nil,
        start_positions: nil,
        end_positions: nil,
        output_attentions: nil,
        output_hidden_states: nil,
        return_dict: nil
      )

        return_dict = !return_dict.nil? ? return_dict : @config.use_return_dict

        outputs = @deberta.(input_ids, attention_mask: attention_mask, token_type_ids: token_type_ids, position_ids: position_ids, inputs_embeds: inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

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
          output = [start_logits, end_logits] + outputs[1..]
          return !total_loss.nil? ? [total_loss] + output : output
        end

        QuestionAnsweringModelOutput.new(loss: total_loss, start_logits: start_logits, end_logits: end_logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end

    class DebertaV2ForMultipleChoice < DebertaV2PreTrainedModel
      def initialize(config)
        super(config)

        num_labels = config.getattr("num_labels", 2)
        @num_labels = num_labels

        @deberta = DebertaV2Model.new(config)
        @pooler = ContextPooler.new(config)
        output_dim = @pooler.output_dim

        @classifier = Torch::NN::Linear.new(output_dim, 1)
        drop_out = config.getattr("cls_dropout", nil)
        drop_out = drop_out.nil? ? @config.hidden_dropout_prob : drop_out
        @dropout = StableDropout.new(drop_out)

        init_weights
      end

      def get_input_embeddings
        @deberta.get_input_embeddings
      end

      def set_input_embeddings(new_embeddings)
        @deberta.set_input_embeddings(new_embeddings)
      end

      def forward(
        input_ids: nil,
        attention_mask: nil,
        token_type_ids: nil,
        position_ids: nil,
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
        flat_token_type_ids = !token_type_ids.nil? ? token_type_ids.view(-1, token_type_ids.size(-1)) : nil
        flat_attention_mask = !attention_mask.nil? ? attention_mask.view(-1, attention_mask.size(-1)) : nil
        flat_inputs_embeds = !inputs_embeds.nil? ? inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) : nil

        outputs = @deberta.(flat_input_ids, position_ids: flat_position_ids, token_type_ids: flat_token_type_ids, attention_mask: flat_attention_mask, inputs_embeds: flat_inputs_embeds, output_attentions: output_attentions, output_hidden_states: output_hidden_states, return_dict: return_dict)

        encoder_layer = outputs[0]
        pooled_output = @pooler.(encoder_layer)
        pooled_output = @dropout.(pooled_output)
        logits = @classifier.(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = nil
        if !labels.nil?
          loss_fct = Torch::NN::CrossEntropyLoss.new
          loss = loss_fct.(reshaped_logits, labels)
        end

        if !return_dict
          output = [reshaped_logits] + outputs[1..]
          return !loss.nil? ? [loss] + output : output
        end

        MultipleChoiceModelOutput.new(loss: loss, logits: reshaped_logits, hidden_states: outputs.hidden_states, attentions: outputs.attentions)
      end
    end
  end

  DebertaV2ForSequenceClassification = DebertaV2::DebertaV2ForSequenceClassification
end
