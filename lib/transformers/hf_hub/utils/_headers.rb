# Copyright 2022-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module Transformers
  module HfHub
    class << self
      def build_hf_headers(
        token: nil,
        is_write_action: false,
        library_name: nil,
        library_version: nil,
        user_agent: nil,
        headers: nil
      )
        # Get auth token to send
        token_to_send = get_token_to_send(token)
        _validate_token_to_send(token_to_send, is_write_action)

        # Combine headers
        hf_headers = {
          "user-agent" => _http_user_agent(
            library_name: library_name,
            library_version: library_version,
            user_agent: user_agent
          )
        }
        if !token_to_send.nil?
          hf_headers["authorization"] = "Bearer #{token_to_send}"
        end
        if headers
          hf_headers.merge!(headers)
        end
        hf_headers
      end

      def get_token_to_send(token)
        # Case token is explicitly provided
        if token.is_a?(String)
          return token
        end

        # Case token is explicitly forbidden
        if token == false
          return nil
        end

        # Token is not provided: we get it from local cache
        cached_token = nil # get_token

        # Case token is explicitly required
        if token == true
          if cached_token.nil?
            raise LocalTokenNotFoundError,
              "Token is required (`token: true`), but no token found. You" +
              " need to provide a token or be logged in to Hugging Face with" +
              " `huggingface-cli login` or `huggingface_hub.login`. See" +
              " https://huggingface.co/settings/tokens."
          end
          return cached_token
        end

        # Case implicit use of the token is forbidden by env variable
        if HF_HUB_DISABLE_IMPLICIT_TOKEN
          return nil
        end

        # Otherwise: we use the cached token as the user has not explicitly forbidden it
        cached_token
      end

      def _validate_token_to_send(token, is_write_action)
        if is_write_action
          if token.nil?
            raise ArgumentError,
              "Token is required (write-access action) but no token found. You need" +
              " to provide a token or be logged in to Hugging Face with" +
              " `huggingface-cli login` or `huggingface_hub.login`. See" +
              " https://huggingface.co/settings/tokens."
          end
        end
      end

      def _http_user_agent(
        library_name: nil,
        library_version: nil,
        user_agent: nil
      )
        if !library_name.nil?
          ua = "#{library_name}/#{library_version}"
        else
          ua = "unknown/None"
        end
        ua += "; ruby/#{RUBY_VERSION.to_f}"
        ua
      end
    end
  end
end
