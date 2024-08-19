module Transformers
  class FeatureExtractionPipeline < Pipeline
    def _sanitize_parameters(truncation: nil, tokenize_kwargs: nil, return_tensors: nil, **kwargs)
      if tokenize_kwargs.nil?
        tokenize_kwargs = {}
      end

      if !truncation.nil?
        if tokenize_kwargs.include?(:truncation)
          raise ArgumentError,
            "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
        end
        tokenize_kwargs[:truncation] = truncation
      end

      preprocess_params = tokenize_kwargs

      postprocess_params = {}
      if !return_tensors.nil?
        postprocess_params[:return_tensors] = return_tensors
      end

      [preprocess_params, {}, postprocess_params]
    end

    def preprocess(inputs, **tokenize_kwargs)
      model_inputs = @tokenizer.(inputs, return_tensors: @framework, **tokenize_kwargs)
      model_inputs
    end

    def _forward(model_inputs)
      model_outputs = @model.(**model_inputs)
      model_outputs
    end

    def postprocess(model_outputs, return_tensors: false)
      # [0] is the first available tensor, logits or last_hidden_state.
      if return_tensors
        model_outputs[0]
      elsif @framework == "pt"
        model_outputs[0].to_a
      elsif @framework == "tf"
        raise Todo
      end
    end
  end
end
