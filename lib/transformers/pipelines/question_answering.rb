module Transformers
  class QuestionAnsweringArgumentHandler < ArgumentHandler
    def normalize(item)
      if item.is_a?(SquadExample)
        return item
      elsif item.is_a?(Hash)
        [:question, :context].each do |k|
          if !item.include?(k)
            raise KeyError, "You need to provide a dictionary with keys {question:..., context:...}"
          elsif item[k].nil?
            raise ArgumentError, "`#{k}` cannot be nil"
          elsif item[k].is_a?(String) && item[k].length == 0
            raise ArgumentError, "`#{k}` cannot be empty"
          end
        end

        return QuestionAnsweringPipeline.create_sample(**item)
      end
      raise ArgumentError, "#{item} argument needs to be of type (SquadExample, dict)"
    end

    def call(*args, **kwargs)
      # Detect where the actual inputs are
      if args.any?
        if args.length == 1
          inputs = args[0]
        elsif args.length == 2 && args.all? { |el| el.is_a?(String) }
          inputs = [{question: args[0], context: args[1]}]
        else
          inputs = args.to_a
        end
      elsif kwargs.include?(:question) && kwargs.include?(:context)
        if kwargs[:question].is_a?(Array) && kwargs[:context].is_a?(String)
          inputs = kwargs[:question].map { |q| {question: q, context: kwargs[:context]} }
        elsif kwargs[:question].is_a?(Array) && kwargs[:context].is_a?(Array)
          if kwargs[:question].length != kwargs[:context].length
            raise ArgumentError, "Questions and contexts don't have the same lengths"
          end

          inputs = kwargs[:question].zip(kwargs[:context]).map { |q, c| {question: q, context: c} }
        elsif kwargs[:question].is_a?(String) && kwargs[:context].is_a?(String)
          inputs = [{question: kwargs[:question], context: kwargs[:context]}]
        else
          raise ArgumentError, "Arguments can't be understood"
        end
      else
        raise ArgumentError, "Unknown arguments #{kwargs}"
      end

      # Normalize inputs
      if inputs.is_a?(Hash)
        inputs = [inputs]
      elsif inputs.is_a?(Enumerable)
        # Copy to avoid overriding arguments
        inputs = inputs.to_a.dup
      else
        raise ArgumentError, "Invalid arguments #{kwargs}"
      end

      inputs.each_with_index do |item, i|
        inputs[i] = normalize(item)
      end

      inputs
    end
  end

  class QuestionAnsweringPipeline < ChunkPipeline
    extend ClassAttribute

    class_attribute :default_input_names, "question,context"
    class_attribute :handle_impossible_answer, false

    def initialize(
      model,
      tokenizer:,
      modelcard: nil,
      framework: nil,
      task: "",
      **kwargs
    )
      super(
        model,
        tokenizer: tokenizer,
        modelcard: modelcard,
        framework: framework,
        task: task,
        **kwargs
      )

      @args_parser = QuestionAnsweringArgumentHandler.new
      check_model_type(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES)
    end

    def self.create_sample(
      question:, context:
    )
      if question.is_a?(Array)
        question.zip(context).map { |q, c| SquadExample.new(nil, q, c, nil, nil, nil) }
      else
        SquadExample.new(nil, question, context, nil, nil, nil)
      end
    end

    def _sanitize_parameters(
      padding: nil,
      topk: nil,
      top_k: nil,
      doc_stride: nil,
      max_answer_len: nil,
      max_seq_len: nil,
      max_question_len: nil,
      handle_impossible_answer: nil,
      align_to_words: nil,
      **kwargs
    )
      # Set defaults values
      preprocess_params = {}
      if !padding.nil?
        preprocess_params[:padding] = padding
      end
      if !doc_stride.nil?
        preprocess_params[:doc_stride] = doc_stride
      end
      if !max_question_len.nil?
        preprocess_params[:max_question_len] = max_question_len
      end
      if !max_seq_len.nil?
        preprocess_params[:max_seq_len] = max_seq_len
      end

      postprocess_params = {}
      if !topk.nil? && top_k.nil?
        warn("topk parameter is deprecated, use top_k instead")
        top_k = topk
      end
      if !top_k.nil?
        if top_k < 1
          raise ArgumentError, "top_k parameter should be >= 1 (got #{top_k})"
        end
        postprocess_params[:top_k] = top_k
      end
      if !max_answer_len.nil?
        if max_answer_len < 1
          raise ArgumentError, "max_answer_len parameter should be >= 1 (got #{max_answer_len})"
        end
      end
      if !max_answer_len.nil?
        postprocess_params[:max_answer_len] = max_answer_len
      end
      if !handle_impossible_answer.nil?
        postprocess_params[:handle_impossible_answer] = handle_impossible_answer
      end
      if !align_to_words.nil?
        postprocess_params[:align_to_words] = align_to_words
      end
      [preprocess_params, {}, postprocess_params]
    end

    def call(*args, **kwargs)
      examples = @args_parser.(*args, **kwargs)
      if examples.is_a?(Array) && examples.length == 1
        return super(examples[0], **kwargs)
      end
      super(examples, **kwargs)
    end

    def preprocess(example, padding: "do_not_pad", doc_stride: nil, max_question_len: 64, max_seq_len: nil)
      # XXX: This is specal, args_parser will not handle anything generator or dataset like
      # For those we expect user to send a simple valid example either directly as a SquadExample or simple dict.
      # So we still need a little sanitation here.
      if example.is_a?(Hash)
        example = SquadExample.new(nil, example[:question], example[:context], nil, nil, nil)
      end

      if max_seq_len.nil?
        max_seq_len = [@tokenizer.model_max_length, 384].min
      end
      if doc_stride.nil?
        doc_stride = [max_seq_len.div(2), 128].min
      end

      if doc_stride > max_seq_len
        raise ArgumentError, "`doc_stride` (#{doc_stride}) is larger than `max_seq_len` (#{max_seq_len})"
      end

      if !@tokenizer.is_fast
        features = squad_convert_examples_to_features(
          examples: [example],
          tokenizer: @tokenizer,
          max_seq_length: max_seq_len,
          doc_stride: doc_stride,
          max_query_length: max_question_len,
          padding_strategy: PaddingStrategy::MAX_LENGTH,
          is_training: false,
          tqdm_enabled: false
        )
      else
        # Define the side we want to truncate / pad and the text/pair sorting
        question_first = @tokenizer.padding_side == "right"

        encoded_inputs = @tokenizer.(
          question_first ? example.question_text : example.context_text,
          text_pair: question_first ? example.context_text : example.question_text,
          padding: padding,
          truncation: question_first ? "only_second" : "only_first",
          max_length: max_seq_len,
          stride: doc_stride,
          return_token_type_ids: true,
          return_overflowing_tokens: true,
          return_offsets_mapping: true,
          return_special_tokens_mask: true\
        )
        # When the input is too long, it's converted in a batch of inputs with overflowing tokens
        # and a stride of overlap between the inputs. If a batch of inputs is given, a special output
        # "overflow_to_sample_mapping" indicate which member of the encoded batch belong to which original batch sample.
        # Here we tokenize examples one-by-one so we don't need to use "overflow_to_sample_mapping".
        # "num_span" is the number of output samples generated from the overflowing tokens.
        num_spans = encoded_inputs[:input_ids].length

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
        p_mask =
          num_spans.times.map do |span_id|
            encoded_inputs.sequence_ids(span_id).map { |tok| tok != (question_first ? 1 : 0) }
          end

        features = []
        num_spans.times do |span_idx|
          input_ids_span_idx = encoded_inputs[:input_ids][span_idx]
          attention_mask_span_idx = (
            encoded_inputs.include?(:attention_mask) ? encoded_inputs[:attention_mask][span_idx] : nil
          )
          token_type_ids_span_idx = (
            encoded_inputs.include?(:token_type_ids) ? encoded_inputs[:token_type_ids][span_idx] : nil
          )
          # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
          if !@tokenizer.cls_token_id.nil?
            cls_indices = (Numo::NArray.cast(input_ids_span_idx).eq(@tokenizer.cls_token_id)).where
            cls_indices.each do |cls_index|
              p_mask[span_idx][cls_index] = false
            end
          end
          submask = p_mask[span_idx]
          features <<
            SquadFeatures.new(
              input_ids: input_ids_span_idx,
              attention_mask: attention_mask_span_idx,
              token_type_ids: token_type_ids_span_idx,
              p_mask: submask,
              encoding: encoded_inputs[span_idx],
              # We don't use the rest of the values - and actually
              # for Fast tokenizer we could totally avoid using SquadFeatures and SquadExample
              cls_index: nil,
              token_to_orig_map: {},
              example_index: 0,
              unique_id: 0,
              paragraph_len: 0,
              token_is_max_context: 0,
              tokens: [],
              start_position: 0,
              end_position: 0,
              is_impossible: false,
              qas_id: nil
            )
        end
      end

      features.each_with_index do |feature, i|
        fw_args = {}
        others = {}
        model_input_names = @tokenizer.model_input_names + ["p_mask", "token_type_ids"]

        feature.instance_variables.each do |k|
          v = feature.instance_variable_get(k)
          k = k[1..]
          if model_input_names.include?(k)
            if @framework == "tf"
              raise Todo
            elsif @framework == "pt"
              tensor = Torch.tensor(v)
              if tensor.dtype == Torch.int32
                tensor = tensor.long
              end
              fw_args[k.to_sym] = tensor.unsqueeze(0)
            end
          else
            others[k.to_sym] = v
          end
        end

        is_last = i == features.length - 1
        yield({example: example, is_last: is_last}.merge(fw_args).merge(others))
      end
    end

    def _forward(inputs)
      example = inputs[:example]
      model_inputs = @tokenizer.model_input_names.to_h { |k| [k.to_sym, inputs[k.to_sym]] }
      # `XXXForSequenceClassification` models should not use `use_cache=True` even if it's supported
      # model_forward = @model.forward if self.framework == "pt" else self.model.call
      # if "use_cache" in inspect.signature(model_forward).parameters.keys():
      #   model_inputs[:use_cache] = false
      # end
      output = @model.(**model_inputs)
      if output.is_a?(Hash)
        {start: output[:start_logits], end: output[:end_logits], example: example}.merge(inputs)
      else
        start, end_ = output[...2]
        {start: start, end: end_, example: example}.merge(inputs)
      end
    end

    def postprocess(
      model_outputs,
      top_k: 1,
      handle_impossible_answer: false,
      max_answer_len: 15,
      align_to_words: true
    )
      min_null_score = 1000000  # large and positive
      answers = []
      model_outputs.each do |output|
        start_ = output[:start]
        end_ = output[:end]
        example = output[:example]
        p_mask = output[:p_mask]
        attention_mask = (
          !output[:attention_mask].nil? ? output[:attention_mask].numo : nil
        )

        starts, ends, scores, min_null_score = select_starts_ends(
          start_, end_, p_mask, attention_mask, min_null_score, top_k, handle_impossible_answer, max_answer_len
        )

        if !@tokenizer.is_fast
          raise Todo
        else
          # Convert the answer (tokens) back to the original text
          # Score: score from the model
          # Start: Index of the first character of the answer in the context string
          # End: Index of the character following the last character of the answer in the context string
          # Answer: Plain text of the answer
          question_first = @tokenizer.padding_side == "right"
          enc = output[:encoding]

          # Encoding was *not* padded, input_ids *might*.
          # It doesn't make a difference unless we're padding on
          # the left hand side, since now we have different offsets
          # everywhere.
          if @tokenizer.padding_side == "left"
            offset = output[:input_ids].eq(@tokenizer.pad_token_id).numo.sum
          else
            offset = 0
          end

          # Sometimes the max probability token is in the middle of a word so:
          # - we start by finding the right word containing the token with `token_to_word`
          # - then we convert this word in a character span with `word_to_chars`
          sequence_index = question_first ? 1 : 0
          starts.to_a.zip(ends.to_a, scores.to_a) do |s, e, score|
            s = s - offset
            e = e - offset

            start_index, end_index = get_indices(enc, s, e, sequence_index, align_to_words)

            answers <<
              {
                score: score[0],
                start: start_index,
                end: end_index,
                answer: example.context_text[start_index...end_index]
              }
          end
        end
      end

      if handle_impossible_answer
        answers << {score: min_null_score, start: 0, end: 0, answer: ""}
      end
      answers = answers.sort_by { |x| -x[:score] }[...top_k]
      if answers.length == 1
        return answers[0]
      end
      answers
    end

    def get_indices(
      enc, s, e, sequence_index, align_to_words
    )
      if align_to_words
        begin
          start_word = enc.token_to_word(s)
          end_word = enc.token_to_word(e)
          start_index = enc.word_to_chars(start_word, sequence_index)[0]
          end_index = enc.word_to_chars(end_word, sequence_index)[1]
        rescue
          # TODO
          raise
          # Some tokenizers don't really handle words. Keep to offsets then.
          start_index = enc.offsets[s][0]
          end_index = enc.offsets[e][1]
        end
      else
        start_index = enc.offsets[s][0]
        end_index = enc.offsets[e][1]
      end
      [start_index, end_index]
    end

    def decode_spans(
      start, end_, topk, max_answer_len, undesired_tokens
    )
      # Ensure we have batch axis
      if start.ndim == 1
        start = start[nil]
      end

      if end_.ndim == 1
        end_ = end_[nil]
      end

      # Compute the score of each tuple(start, end) to be the real answer
      outer = start.expand_dims(-1).dot(end_.expand_dims(1))

      # Remove candidate with end < start and end - start > max_answer_len
      candidates = outer.triu.tril(max_answer_len - 1)

      # Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
      scores_flat = candidates.flatten
      if topk == 1
        idx_sort = [scores_flat.argmax]
      elsif scores_flat.length < topk
        raise Todo
      else
        raise Todo
      end

      starts, ends = unravel_index(idx_sort, candidates.shape)[1..]
      desired_spans = isin(starts, undesired_tokens.where) & isin(ends, undesired_tokens.where)
      starts = starts[desired_spans]
      ends = ends[desired_spans]
      scores = candidates[0, starts, ends]

      [starts, ends, scores]
    end

    def unravel_index(indices, shape)
      indices = Numo::NArray.cast(indices)
      result = []
      factor = 1
      shape.size.times do |i|
        result.unshift(indices / factor % shape[-1 - i])
        factor *= shape[-1 - i]
      end
      result
    end

    def isin(element, test_elements)
      test_elements = test_elements.to_a
      Numo::Bit.cast(element.to_a.map { |e| test_elements.include?(e) })
    end

    def select_starts_ends(
      start,
      end_,
      p_mask,
      attention_mask,
      min_null_score = 1000000,
      top_k = 1,
      handle_impossible_answer = false,
      max_answer_len = 15
    )
      # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
      undesired_tokens = ~p_mask.numo

      if !attention_mask.nil?
        undesired_tokens = undesired_tokens & attention_mask
      end

      # Generate mask
      undesired_tokens_mask = undesired_tokens.eq(0)

      # Make sure non-context indexes in the tensor cannot contribute to the softmax
      start = start.numo
      end_ = end_.numo
      start[undesired_tokens_mask] = -10000.0
      end_[undesired_tokens_mask] = -10000.0

      # Normalize logits and spans to retrieve the answer
      start = Numo::NMath.exp(start - start.max(axis: -1, keepdims: true))
      start = start / start.sum

      end_ = Numo::NMath.exp(end_ - end_.max(axis: -1, keepdims: true))
      end_ = end_ / end_.sum

      if handle_impossible_answer
        min_null_score = [min_null_score, (start[0, 0] * end_[0, 0]).item].min
      end

      # Mask CLS
      start[0, 0] = end_[0, 0] = 0.0

      starts, ends, scores = decode_spans(start, end_, top_k, max_answer_len, undesired_tokens)
      [starts, ends, scores, min_null_score]
    end
  end
end
