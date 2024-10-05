module Transformers
  module HfHub
    # Possible values for env variables

    ENV_VARS_TRUE_VALUES = ["1", "ON", "YES", "TRUE"]
    ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES + ["AUTO"]

    def self._is_true(value)
      if value.nil?
        return false
      end
      ENV_VARS_TRUE_VALUES.include?(value.upcase)
    end

    def self._as_int(value)
      if value.nil?
        return nil
      end
      value.to_i
    end

    # Constants for file downloads

    DEFAULT_ETAG_TIMEOUT = 10

    # Git-related constants

    DEFAULT_REVISION = "main"

    ENDPOINT = ENV["HF_ENDPOINT"] || "https://huggingface.co"

    HUGGINGFACE_CO_URL_TEMPLATE = ENDPOINT + "/%{repo_id}/resolve/%{revision}/%{filename}"
    HUGGINGFACE_HEADER_X_REPO_COMMIT = "x-repo-commit"
    HUGGINGFACE_HEADER_X_LINKED_ETAG = "x-linked-etag"
    HUGGINGFACE_HEADER_X_LINKED_SIZE = "x-linked-size"

    REPO_ID_SEPARATOR = "--"
    # ^ this substring is not allowed in repo_ids on hf.co
    # and is the canonical one we use for serialization of repo ids elsewhere.

    REPO_TYPE_DATASET = "dataset"
    REPO_TYPE_SPACE = "space"
    REPO_TYPE_MODEL = "model"
    REPO_TYPES = [nil, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]

    REPO_TYPES_URL_PREFIXES = {
      REPO_TYPE_DATASET => "datasets/",
      REPO_TYPE_SPACE => "spaces/"
    }

    # default cache
    DEFAULT_HOME = File.join(ENV.fetch("HOME"), ".cache")
    HF_HOME =
      File.expand_path(
        ENV.fetch(
          "HF_HOME",
          File.join(ENV.fetch("XDG_CACHE_HOME", DEFAULT_HOME), "huggingface")
        )
      )

    # New env variables
    HF_HUB_CACHE = ENV["HF_HUB_CACHE"] || File.join(HF_HOME, "hub")

    HF_HUB_OFFLINE = _is_true(ENV["HF_HUB_OFFLINE"] || ENV["TRANSFORMERS_OFFLINE"])

    # Disable sending the cached token by default is all HTTP requests to the Hub
    HF_HUB_DISABLE_IMPLICIT_TOKEN = _is_true(ENV["HF_HUB_DISABLE_IMPLICIT_TOKEN"])

    HF_HUB_ENABLE_HF_TRANSFER = _is_true(ENV["HF_HUB_ENABLE_HF_TRANSFER"])
  end
end
