module Transformers
  # TODO remove in 0.2.0
  class SentenceTransformer
    def initialize(model_id)
      @model_id = model_id
      @model = Transformers.pipeline("embedding", model_id)
    end

    def encode(sentences)
      # TODO check modules.json
      if [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
      ].include?(@model_id)
        @model.(sentences)
      else
        @model.(sentences, pooling: "cls", normalize: false)
      end
    end
  end
end
