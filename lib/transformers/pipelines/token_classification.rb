module Transformers
  class TokenClassificationArgumentHandler < ArgumentHandler
  end

  class AggregationStrategy < ExplicitEnum
    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"
  end

  class TokenClassificationPipeline < ChunkPipeline
    extend ClassAttribute

    class_attribute :default_input_names, "sequences"

    def initialize(*args, args_parser: TokenClassificationArgumentHandler.new, **kwargs)
      super(*args, **kwargs)
      check_model_type(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)

      @basic_tokenizer = Bert::BertTokenizer::BasicTokenizer.new(do_lower_case: false)
      @args_parser = args_parser
    end

    def _sanitize_parameters(
      ignore_labels: nil,
      grouped_entities: nil,
      ignore_subwords: nil,
      aggregation_strategy: nil,
      offset_mapping: nil,
      stride: nil
    )
      preprocess_params = {}
      if !offset_mapping.nil?
        preprocess_params[:offset_mapping] = offset_mapping
      end

      postprocess_params = {}
      if !grouped_entities.nil? || !ignore_subwords.nil?
        if grouped_entities && ignore_subwords
          aggregation_strategy = AggregationStrategy::FIRST
        elsif grouped_entities && !ignore_subwords
          aggregation_strategy = AggregationStrategy::SIMPLE
        else
          aggregation_strategy = AggregationStrategy::NONE
        end

        if !grouped_entities.nil?
          warn(
            "`grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to" +
            " `aggregation_strategy=\"#{aggregation_strategy}\"` instead."
          )
        end
        if !ignore_subwords.nil?
          warn(
            "`ignore_subwords` is deprecated and will be removed in version v5.0.0, defaulted to" +
            " `aggregation_strategy=\"#{aggregation_strategy}\"` instead."
          )
        end
      end

      if !aggregation_strategy.nil?
        if aggregation_strategy.is_a?(String)
          aggregation_strategy = AggregationStrategy.new(aggregation_strategy.downcase).to_s
        end
        if (
          [AggregationStrategy::FIRST, AggregationStrategy::MAX, AggregationStrategy::AVERAGE].include?(aggregation_strategy) &&
          !@tokenizer.is_fast
        )
          raise ArgumentError,
            "Slow tokenizers cannot handle subwords. Please set the `aggregation_strategy` option" +
            ' to `"simple"` or use a fast tokenizer.'
        end
        postprocess_params[:aggregation_strategy] = aggregation_strategy
      end
      if !ignore_labels.nil?
        postprocess_params[:ignore_labels] = ignore_labels
      end
      if !stride.nil?
        if stride >= @tokenizer.model_max_length
          raise ArgumentError,
            "`stride` must be less than `tokenizer.model_max_length` (or even lower if the tokenizer adds special tokens)"
        end
        if aggregation_strategy == AggregationStrategy::NONE
          raise ArgumentError,
            "`stride` was provided to process all the text but `aggregation_strategy=" +
            "\"#{aggregation_strategy}\"`, please select another one instead."
        else
          if @tokenizer.is_fast
            tokenizer_params = {
              return_overflowing_tokens: true,
              padding: true,
              stride: stride
            }
            preprocess_params[:tokenizer_params] = tokenizer_params
          else
            raise ArgumentError,
              "`stride` was provided to process all the text but you're using a slow tokenizer." +
              " Please use a fast tokenizer."
          end
        end
      end
      [preprocess_params, {}, postprocess_params]
    end

    def preprocess(sentence, offset_mapping: nil, **preprocess_params)
      tokenizer_params = preprocess_params.delete(:tokenizer_params) { {} }
      truncation = @tokenizer.model_max_length && @tokenizer.model_max_length > 0
      inputs = @tokenizer.(
        sentence,
        return_tensors: @framework,
        truncation: truncation,
        return_special_tokens_mask: true,
        return_offsets_mapping: @tokenizer.is_fast,
        **tokenizer_params
      )
      inputs.delete(:overflow_to_sample_mapping)
      num_chunks = inputs[:input_ids].length

      num_chunks.times do |i|
        if @framework == "tf"
          raise Todo
        else
          model_inputs = inputs.to_h { |k, v| [k, v[i].unsqueeze(0)] }
        end
        if !@offset_mapping.nil?
          model_inputs[:offset_mapping] = offset_mapping
        end
        model_inputs[:sentence] = i == 0 ? sentence : nil
        model_inputs[:is_last] = (i == num_chunks - 1)

        yield model_inputs
      end
    end

    def _forward(model_inputs)
      # Forward
      special_tokens_mask = model_inputs.delete(:special_tokens_mask)
      offset_mapping = model_inputs.delete(:offset_mapping)
      sentence = model_inputs.delete(:sentence)
      is_last = model_inputs.delete(:is_last)
      if @framework == "tf"
        logits = @model.(**model_inputs)[0]
      else
        output = @model.(**model_inputs)
        logits = output.is_a?(Hash) ? output[:logits] : output[0]
      end

      {
        logits: logits,
        special_tokens_mask: special_tokens_mask,
        offset_mapping: offset_mapping,
        sentence: sentence,
        is_last: is_last,
        **model_inputs
      }
    end

    def postprocess(all_outputs, aggregation_strategy: AggregationStrategy::NONE, ignore_labels: nil)
      if ignore_labels.nil?
        ignore_labels = ["O"]
      end
      all_entities = []
      all_outputs.each do |model_outputs|
        logits = model_outputs[:logits][0].numo
        sentence = all_outputs[0][:sentence]
        input_ids = model_outputs[:input_ids][0]
        offset_mapping = (
          !model_outputs[:offset_mapping].nil? ? model_outputs[:offset_mapping][0] : nil
        )
        special_tokens_mask = model_outputs[:special_tokens_mask][0].numo

        maxes = logits.max(axis: -1).expand_dims(-1)
        shifted_exp = Numo::NMath.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis: -1).expand_dims(-1)

        if @framework == "tf"
          raise Todo
        end

        pre_entities = gather_pre_entities(
          sentence, input_ids, scores, offset_mapping, special_tokens_mask, aggregation_strategy
        )
        grouped_entities = aggregate(pre_entities, aggregation_strategy)
        # Filter anything that is in self.ignore_labels
        entities =
          grouped_entities.select do |entity|
            !ignore_labels.include?(entity[:entity]) && !ignore_labels.include?(entity[:entity_group])
          end
        all_entities.concat(entities)
      end
      num_chunks = all_outputs.length
      if num_chunks > 1
        all_entities = aggregate_overlapping_entities(all_entities)
      end
      all_entities
    end

    def gather_pre_entities(
      sentence,
      input_ids,
      scores,
      offset_mapping,
      special_tokens_mask,
      aggregation_strategy
    )
      pre_entities = []
      scores.each_over_axis(0).with_index do |token_scores, idx|
        # Filter special_tokens
        if special_tokens_mask[idx] != 0
          next
        end

        word = @tokenizer.convert_ids_to_tokens(input_ids[idx].to_i)
        if !offset_mapping.nil?
          start_ind, end_ind = offset_mapping[idx].to_a
          if !start_ind.is_a?(Integer)
            if @framework == "pt"
              start_ind = start_ind.item
              end_ind = end_ind.item
            end
          end
          word_ref = sentence[start_ind...end_ind]
          if @tokenizer.instance_variable_get(:@tokenizer).respond_to?(:continuing_subword_prefix)
            # This is a BPE, word aware tokenizer, there is a correct way
            # to fuse tokens
            is_subword = word.length != word_ref.length
          else
            is_subword = start_ind > 0 && !sentence[(start_ind - 1)...(start_ind + 1)].include?(" ")
          end

          if input_ids[idx].to_i == @tokenizer.unk_token_id
            word = word_ref
            is_subword = false
          end
        else
          start_ind = nil
          end_ind = nil
          is_subword = nil
        end

        pre_entity = {
          word: word,
          scores: token_scores,
          start: start_ind,
          end: end_ind,
          index: idx,
          is_subword: is_subword
        }
        pre_entities << pre_entity
      end
      pre_entities
    end

    def aggregate(pre_entities, aggregation_strategy)
      if [AggregationStrategy::NONE, AggregationStrategy::SIMPLE].include?(aggregation_strategy)
        entities = []
        pre_entities.each do |pre_entity|
          entity_idx = pre_entity[:scores].argmax
          score = pre_entity[:scores][entity_idx]
          entity = {
            entity: @model.config.id2label[entity_idx],
            score: score,
            index: pre_entity[:index],
            word: pre_entity[:word],
            start: pre_entity[:start],
            end: pre_entity[:end]
          }
          entities << entity
        end
      else
        entities = aggregate_words(pre_entities, aggregation_strategy)
      end

      if aggregation_strategy == AggregationStrategy::NONE
        return entities
      end
      group_entities(entities)
    end

    def aggregate_word(entities, aggregation_strategy)
      raise Todo
    end

    def aggregate_words(entities, aggregation_strategy)
      raise Todo
    end

    def group_sub_entities(entities)
      # Get the first entity in the entity group
      entity = entities[0][:entity].split("-", 2)[-1]
      scores = entities.map { |entity| entity[:score] }
      tokens = entities.map { |entity| entity[:word] }

      entity_group = {
        entity_group: entity,
        score: scores.sum / scores.count.to_f,
        word: @tokenizer.convert_tokens_to_string(tokens),
        start: entities[0][:start],
        end: entities[-1][:end]
      }
      entity_group
    end

    def get_tag(entity_name)
      if entity_name.start_with?("B-")
        bi = "B"
        tag = entity_name[2..]
      elsif entity_name.start_with?("I-")
        bi = "I"
        tag = entity_name[2..]
      else
        # It's not in B-, I- format
        # Default to I- for continuation.
        bi = "I"
        tag = entity_name
      end
      [bi, tag]
    end

    def group_entities(entities)
      entity_groups = []
      entity_group_disagg = []

      entities.each do |entity|
        if entity_group_disagg.empty?
          entity_group_disagg << entity
          next
        end

        # If the current entity is similar and adjacent to the previous entity,
        # append it to the disaggregated entity group
        # The split is meant to account for the "B" and "I" prefixes
        # Shouldn't merge if both entities are B-type
        bi, tag = get_tag(entity[:entity])
        _last_bi, last_tag = get_tag(entity_group_disagg[-1][:entity])

        if tag == last_tag && bi != "B"
          # Modify subword type to be previous_type
          entity_group_disagg << entity
        else
          # If the current entity is different from the previous entity
          # aggregate the disaggregated entity group
          entity_groups << group_sub_entities(entity_group_disagg)
          entity_group_disagg = [entity]
        end
      end
      if entity_group_disagg.any?
        # it's the last entity, add it to the entity groups
        entity_groups << group_sub_entities(entity_group_disagg)
      end

      entity_groups
    end
  end
end
