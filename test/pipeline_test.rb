require_relative "test_helper"

class PipelineTest < Minitest::Test
  def test_ner
    ner = Transformers.pipeline("ner")
    result = ner.("Ruby is a programming language created by Matz")
    assert_equal 3, result.size
    assert_equal "I-MISC", result[0][:entity]
    assert_in_delta 0.96, result[0][:score]
    assert_equal 1, result[0][:index]
    assert_equal "Ruby", result[0][:word]
    assert_equal 0, result[0][:start]
    assert_equal 4, result[0][:end]
  end

  def test_ner_aggregation_strategy
    ner = Transformers.pipeline("ner", aggregation_strategy: "simple")
    result = ner.("Ruby is a programming language created by Matz")
    assert_equal 2, result.size

    assert_equal "MISC", result[0][:entity_group]
    assert_in_delta 0.9608, result[0][:score]
    assert_equal "Ruby", result[0][:word]
    assert_equal 0, result[0][:start]
    assert_equal 4, result[0][:end]

    assert_equal "PER", result[1][:entity_group]
    assert_in_delta 0.9496, result[1][:score]
    assert_equal "Matz", result[1][:word]
    assert_equal 42, result[1][:start]
    assert_equal 46, result[1][:end]
  end

  def test_sentiment_analysis
    classifier = Transformers.pipeline("sentiment-analysis")
    result = classifier.("We are very happy to show you the 🤗 Transformers library.")
    assert_equal "POSITIVE", result[:label]
    assert_in_delta 0.9998, result[:score]

    result = classifier.(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
    assert_equal "POSITIVE", result[0][:label]
    assert_in_delta 0.9998, result[0][:score]
    assert_equal "NEGATIVE", result[1][:label]
    assert_in_delta 0.5309, result[1][:score]
  end

  def test_question_answering
    qa = Transformers.pipeline("question-answering")
    result = qa.(question: "Who invented Ruby?", context: "Ruby is a programming language created by Matz")
    assert_in_delta 0.998, result[:score]
    assert_equal 42, result[:start]
    assert_equal 46, result[:end]
    assert_equal "Matz", result[:answer]

    result = qa.("Who invented Ruby?", "Ruby is a programming language created by Matz")
    assert_equal "Matz", result[:answer]
  end

  def test_feature_extraction
    fe = Transformers.pipeline("feature-extraction")
    result = fe.("We are very happy to show you the 🤗 Transformers library.")
    assert_in_delta 0.454, result[0][0][0]
  end

  def test_embedding
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embed = Transformers.pipeline("embedding")
    embeddings = embed.(sentences)
    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end

  def test_reranking
    query = "How many people live in London?"
    docs = ["Around 9 Million people live in London", "London is known for its financial district"]
    rerank = Transformers.pipeline("reranking")
    result = rerank.(query, docs)
    assert_equal 2, result.size
    assert_equal 0, result[0][:index]
    assert_in_delta 0.984, result[0][:score]
    assert_equal 1, result[1][:index]
    assert_in_delta 0.139, result[1][:score]
  end

  def test_image_classification
    classifier = Transformers.pipeline("image-classification")
    result = classifier.("test/support/pipeline-cat-chonk.jpeg")
    assert_equal "lynx, catamount", result[0][:label]
    assert_in_delta 0.433, result[0][:score], 0.01
    assert_equal "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor", result[1][:label]
    assert_in_delta 0.035, result[1][:score], 0.01
  end

  def test_image_feature_extraction
    fe = Transformers.pipeline("image-feature-extraction")
    result = fe.("test/support/pipeline-cat-chonk.jpeg")
    assert_in_delta 0.868, result[0][0][0], 0.01
  end

  def test_device
    skip unless mac?

    sentences = ["This is an example sentence", "Each sentence is converted"]
    embed = Transformers.pipeline("embedding", device: "mps")
    embeddings = embed.(sentences)
    assert_elements_in_delta [0.067657, 0.063496, 0.048713], embeddings[0][..2]
    assert_elements_in_delta [0.086439, 0.10276, 0.0053946], embeddings[1][..2]
  end

  def test_pipeline_input_works_with_more_than_ten
    embedding = Transformers.pipeline("embedding")
    11.times do
      result = embedding.("Ruby is a programming language created by Matz")
      assert_instance_of(Array, result)
    end
  end
end
