module Transformers
  module HfHub
    class Error < StandardError; end

    # Raised if local token is required but not found.
    class LocalTokenNotFoundError < Error; end

    # Raised when a request is made but `HF_HUB_OFFLINE=1` is set as environment variable.
    class OfflineModeIsEnabled < Error; end
  end
end
