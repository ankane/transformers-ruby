module Transformers
  class TextClassificationPipeline < Pipeline
    def initialize(*args, **kwargs)
      super

      check_model_type(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)
    end

    private

    def _sanitize_parameters(return_all_scores: nil, function_to_apply: nil, top_k: "", **tokenizer_kwargs)
      # Using "" as default argument because we're going to use `top_k=None` in user code to declare
      # "No top_k"
      preprocess_params = tokenizer_kwargs

      postprocess_params = {}
      if @model.config.respond_to?(:return_all_scores) && return_all_scores.nil?
        return_all_scores = @model.config.return_all_scores
      end

      if top_k.is_a?(Integer) || top_k.nil?
        postprocess_params[:top_k] = top_k
        postprocess_params[:_legacy] = false
      elsif !return_all_scores.nil?
        warn(
          "`return_all_scores` is now deprecated, if want a similar functionality use `top_k: nil` instead of" +
          " `return_all_scores: true` or `top_k: 1` instead of `return_all_scores: false`.",
        )
        if return_all_scores
          postprocess_params[:top_k] = nil
        else
          postprocess_params[:top_k] = 1
        end
      end

      if function_to_apply.is_a?(String)
        function_to_apply = ClassificationFunction.new(function_to_apply.upcase).to_s
      end

      if !function_to_apply.nil?
        postprocess_params[:function_to_apply] = function_to_apply
      end
      [preprocess_params, {}, postprocess_params]
    end

    def preprocess(inputs, **tokenizer_kwargs)
      return_tensors = @framework
      if inputs.is_a?(Hash)
        return @tokenizer.(**inputs, return_tensors: return_tensors, **tokenizer_kwargs)
      elsif inputs.is_a?(Array) && inputs.length == 1 && inputs[0].is_a?(Array) && inputs[0].length == 2
        # It used to be valid to use a list of list of list for text pairs, keeping this path for BC
        return @tokenizer.(
          inputs[0][0], text_pair: inputs[0][1], return_tensors: return_tensors, **tokenizer_kwargs
        )
      elsif inputs.is_a?(Array)
        # This is likely an invalid usage of the pipeline attempting to pass text pairs.
        raise ArgumentError,
          "The pipeline received invalid inputs, if you are trying to send text pairs, you can try to send a" +
          ' dictionary `{"text": "My text", "text_pair": "My pair"}` in order to send a text pair.'
      end
      @tokenizer.(inputs, return_tensors: return_tensors, **tokenizer_kwargs)
    end

    def _forward(model_inputs)
      @model.(**model_inputs.to_h)
    end

    def postprocess(model_outputs, function_to_apply: nil, top_k: 1, _legacy: true)
      if function_to_apply.nil?
        if @model.config.problem_type == "multi_label_classification" || @model.config.num_labels == 1
          function_to_apply = ClassificationFunction::SIGMOID
        elsif @model.config.problem_type == "single_label_classification" || @model.config.num_labels > 1
          function_to_apply = ClassificationFunction::SOFTMAX
        elsif @model.config.instance_variable_defined?(:@function_to_apply) && function_to_apply.nil?
          function_to_apply = @model.config.function_to_apply
        else
          function_to_apply = ClassificationFunction::NONE
        end
      end

      outputs = model_outputs["logits"][0]
      outputs = outputs.numo

      if function_to_apply == ClassificationFunction::SIGMOID
        scores = sigmoid(outputs)
      elsif function_to_apply == ClassificationFunction::SOFTMAX
        scores = softmax(outputs)
      elsif function_to_apply == ClassificationFunction::NONE
        scores = outputs
      else
        raise ArgumentError, "Unrecognized `function_to_apply` argument: #{function_to_apply}"
      end

      if top_k == 1 && _legacy
        return {label: @model.config.id2label[scores.argmax], score: scores.max}
      end

      dict_scores =
        scores.to_a.map.with_index do |score, i|
          {label: @model.config.id2label[i], score: score}
        end
      if !_legacy
        dict_scores.sort_by! { |x| -x[:score] }
        if !top_k.nil?
          dict_scores = dict_scores.first(top_k)
        end
      end
      dict_scores
    end

    private

    def sigmoid(_outputs)
      1.0 / (1.0 + Numo::NMath.exp(-_outputs))
    end

    def softmax(_outputs)
      maxes = _outputs.max(axis: -1, keepdims: true)
      shifted_exp = Numo::NMath.exp(_outputs - maxes)
      shifted_exp / shifted_exp.sum(axis: -1, keepdims: true)
    end
  end
end
