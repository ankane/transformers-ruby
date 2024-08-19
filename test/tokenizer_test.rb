require_relative "test_helper"

class TokenizerTest < Minitest::Test
  def test_auto_tokenizer
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = Transformers::AutoTokenizer.from_pretrained(model_name)

    encoding = tokenizer.("We are very happy to show you the 🤗 Transformers library.")
    assert_equal [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102], encoding[:input_ids]
    assert_equal [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], encoding[:attention_mask]
  end
end
