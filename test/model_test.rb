require_relative "test_helper"

class ModelTest < Minitest::Test
  def setup
    skip if ci?
  end

  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  def test_all_mini_lm
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = Transformers.pipeline("embedding", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.(sentences)

    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end

  # https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
  def test_multi_qa_minilm
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]

    model = Transformers.pipeline("embedding", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    query_embedding = model.(query)
    doc_embeddings = model.(docs)
    scores = doc_embeddings.map { |e| e.zip(query_embedding).sum { |d, q| d * q } }
    doc_score_pairs = docs.zip(scores).sort_by { |d, s| -s }

    assert_equal "Around 9 Million people live in London", doc_score_pairs[0][0]
    assert_in_delta 0.9156, doc_score_pairs[0][1]
    assert_equal "London is known for its financial district", doc_score_pairs[1][0]
    assert_in_delta 0.4948, doc_score_pairs[1][1]
  end

  # https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
  def test_mxbai_embed
    query_prefix = "Represent this sentence for searching relevant passages: "

    input = [
      query_prefix + "puppy",
      "The dog is barking",
      "The cat is purring"
    ]

    model = Transformers.pipeline("embedding", "mixedbread-ai/mxbai-embed-large-v1")
    embeddings = model.(input, pooling: "cls", normalize: false)

    assert_elements_in_delta [-0.00624076, 0.12864432, 0.5248165], embeddings[0][..2]
    assert_elements_in_delta [-0.61227727, 1.4060247, -0.04079155], embeddings[-1][..2]
  end

  # https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1
  def test_opensearch
    docs = ["The dog is barking", "The cat is purring", "The bear is growling"]

    model_id = "opensearch-project/opensearch-neural-sparse-encoding-v1"
    model = Transformers::AutoModelForMaskedLM.from_pretrained(model_id)
    tokenizer = Transformers::AutoTokenizer.from_pretrained(model_id)
    special_token_ids = tokenizer.special_tokens_map.map { |_, token| tokenizer.vocab[token] }

    feature = tokenizer.(docs, padding: true, truncation: true, return_tensors: "pt", return_token_type_ids: false)
    output = model.(**feature)[0]

    values, _ = Torch.max(output * feature[:attention_mask].unsqueeze(-1), dim: 1)
    values = Torch.log(1 + Torch.relu(values))
    values[0.., special_token_ids] = 0
    embeddings = values.to_a

    assert_equal 74, embeddings[0].count { |v| v != 0 }
    assert_equal 77, embeddings[1].count { |v| v != 0 }
    assert_equal 102, embeddings[2].count { |v| v != 0 }
  end
end
