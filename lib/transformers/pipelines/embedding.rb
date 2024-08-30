module Transformers
  class EmbeddingPipeline < Pipeline
    def _sanitize_parameters(**kwargs)
      [{}, {}, kwargs]
    end

    def preprocess(inputs)
      @tokenizer.(inputs, return_tensors: @framework)
    end

    def _forward(model_inputs)
      {
        last_hidden_state: @model.(**model_inputs)[0],
        attention_mask: model_inputs[:attention_mask]
      }
    end

    def postprocess(model_outputs, pooling: "mean", normalize: true)
      output = model_outputs[:last_hidden_state]

      case pooling
      when "none"
        # do nothing
      when "mean"
        output = mean_pooling(output, model_outputs[:attention_mask])
      when "cls"
        output = output[0.., 0]
      else
        raise Error, "Pooling method '#{pooling}' not supported."
      end

      if normalize
        output = Torch::NN::Functional.normalize(output, p: 2, dim: 1)
      end

      output[0].to_a
    end

    private

    def mean_pooling(output, attention_mask)
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(output.size).float
      Torch.sum(output * input_mask_expanded, 1) / Torch.clamp(input_mask_expanded.sum(1), min: 1e-9)
    end
  end
end
