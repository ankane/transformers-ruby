# dependencies
require "numo/narray"
require "safetensors"
require "tokenizers"
require "torch-rb"

# stdlib
require "cgi"
require "fileutils"
require "io/console"
require "json"
require "logger"
require "net/http"
require "pathname"
require "securerandom"
require "set"
require "uri"

# modules
require_relative "transformers/ruby_utils"
require_relative "transformers/utils/generic"
require_relative "transformers/activations"
require_relative "transformers/dynamic_module_utils"
require_relative "transformers/configuration_utils"
require_relative "transformers/convert_slow_tokenizer"
require_relative "transformers/feature_extraction_utils"
require_relative "transformers/image_utils"
require_relative "transformers/image_processing_base"
require_relative "transformers/image_processing_utils"
require_relative "transformers/image_transforms"
require_relative "transformers/modeling_outputs"
require_relative "transformers/modeling_utils"
require_relative "transformers/sentence_transformer"
require_relative "transformers/tokenization_utils_base"
require_relative "transformers/tokenization_utils"
require_relative "transformers/tokenization_utils_fast"
require_relative "transformers/torch_utils"
require_relative "transformers/version"

# data
require_relative "transformers/data/processors/squad"

# hub
require_relative "transformers/hf_hub/constants"
require_relative "transformers/hf_hub/errors"
require_relative "transformers/hf_hub/file_download"
require_relative "transformers/hf_hub/utils/_errors"
require_relative "transformers/hf_hub/utils/_headers"

# models auto
require_relative "transformers/models/auto/auto_factory"
require_relative "transformers/models/auto/configuration_auto"
require_relative "transformers/models/auto/feature_extraction_auto"
require_relative "transformers/models/auto/image_processing_auto"
require_relative "transformers/models/auto/modeling_auto"
require_relative "transformers/models/auto/tokenization_auto"

# models bert
require_relative "transformers/models/bert/configuration_bert"
require_relative "transformers/models/bert/modeling_bert"
require_relative "transformers/models/bert/tokenization_bert"
require_relative "transformers/models/bert/tokenization_bert_fast"

# models deberta-v2
require_relative "transformers/models/deberta_v2/configuration_deberta_v2"
require_relative "transformers/models/deberta_v2/modeling_deberta_v2"
require_relative "transformers/models/deberta_v2/tokenization_deberta_v2_fast"

# models distilbert
require_relative "transformers/models/distilbert/configuration_distilbert"
require_relative "transformers/models/distilbert/modeling_distilbert"
require_relative "transformers/models/distilbert/tokenization_distilbert"
require_relative "transformers/models/distilbert/tokenization_distilbert_fast"

# models mpnet
require_relative "transformers/models/mpnet/configuration_mpnet"
require_relative "transformers/models/mpnet/modeling_mpnet"
require_relative "transformers/models/mpnet/tokenization_mpnet_fast"

# models vit
require_relative "transformers/models/vit/configuration_vit"
require_relative "transformers/models/vit/image_processing_vit"
require_relative "transformers/models/vit/modeling_vit"

# pipelines
require_relative "transformers/pipelines/base"
require_relative "transformers/pipelines/feature_extraction"
require_relative "transformers/pipelines/embedding"
require_relative "transformers/pipelines/image_classification"
require_relative "transformers/pipelines/image_feature_extraction"
require_relative "transformers/pipelines/pt_utils"
require_relative "transformers/pipelines/question_answering"
require_relative "transformers/pipelines/text_classification"
require_relative "transformers/pipelines/token_classification"
require_relative "transformers/pipelines/_init"

# utils
require_relative "transformers/utils/_init"
require_relative "transformers/utils/import_utils"
require_relative "transformers/utils/hub"
require_relative "transformers/utils/logging"

module Transformers
  class Error < StandardError; end

  class Todo < Error
    def message
      "not implemented yet"
    end
  end

  class << self
    # experimental
    attr_accessor :fast_init
  end
  self.fast_init = false
end
