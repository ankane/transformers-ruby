module Transformers
  class RerankingPipeline < Pipeline
    def _sanitize_parameters(**kwargs)
      [{}, {}, kwargs]
    end

    def preprocess(inputs)
      @tokenizer.(
        [inputs[:query]] * inputs[:documents].length,
        text_pair: inputs[:documents],
        return_tensors: @framework
      )
    end

    def _forward(model_inputs)
      model_outputs = @model.(**model_inputs)
      model_outputs
    end

    def call(query, documents)
      super({query: query, documents: documents})
    end

    def postprocess(model_outputs)
       model_outputs[0]
        .sigmoid
        .squeeze
        .to_a
        .map.with_index { |s, i| {index: i, score: s} }
        .sort_by { |v| -v[:score] }
    end
  end
end
