module Transformers
  class SentenceTransformer
    def initialize(model_id)
      @model_id = model_id
      @tokenizer = Transformers::AutoTokenizer.from_pretrained(model_id)
      @model = Transformers::AutoModel.from_pretrained(model_id)
    end

    def encode(sentences)
      singular = sentences.is_a?(String)
      sentences = [sentences] if singular

      input = @tokenizer.(sentences, padding: true, truncation: true, return_tensors: "pt")
      output = Torch.no_grad { @model.(**input) }[0]

      # TODO check modules.json
      if [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
      ].include?(@model_id)
        output = mean_pooling(output, input[:attention_mask])
        output = Torch::NN::Functional.normalize(output, p: 2, dim: 1).to_a
      else
        output = output[0.., 0].to_a
      end

      singular ? output[0] : output
    end

    private

    def mean_pooling(output, attention_mask)
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(output.size).float
      Torch.sum(output * input_mask_expanded, 1) / Torch.clamp(input_mask_expanded.sum(1), min: 1e-9)
    end
  end
end
