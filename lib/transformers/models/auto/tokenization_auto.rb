# Copyright 2018 The HuggingFace Inc. team.
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
  TOKENIZER_MAPPING_NAMES = {
    "bert" => ["BertTokenizer", "BertTokenizerFast"],
    "distilbert" => ["DistilBertTokenizer", "DistilBertTokenizerFast"]
  }

  TOKENIZER_MAPPING = LazyAutoMapping.new(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

  class AutoTokenizer
    class << self
      def from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        config = kwargs.delete(:config)
        kwargs[:_from_auto] = true

        use_fast = kwargs.delete(:use_fast) { true }
        tokenizer_type = kwargs.delete(:tokenizer_type) { nil }
        trust_remote_code = kwargs.delete(:trust_remote_code)

        if !tokenizer_type.nil?
          raise Todo
        end

        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        if tokenizer_config.include?("_commit_hash")
          kwargs[:_commit_hash] = tokenizer_config["_commit_hash"]
        end
        config_tokenizer_class = tokenizer_config["tokenizer_class"]
        _tokenizer_auto_map = nil
        if tokenizer_config["auto_map"]
          raise Todo
        end

        # If that did not work, let's try to use the config.
        if config_tokenizer_class.nil?
          if !config.is_a?(PretrainedConfig)
            config = AutoConfig.from_pretrained(
              pretrained_model_name_or_path, trust_remote_code: trust_remote_code, **kwargs
            )
            config_tokenizer_class = config.tokenizer_class
            # if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
            #     tokenizer_auto_map = config.auto_map["AutoTokenizer"]
          end
        end

        if !config_tokenizer_class.nil?
          tokenizer_class = nil
          if use_fast && !config_tokenizer_class.end_with?("Fast")
            tokenizer_class_candidate = "#{config_tokenizer_class}Fast"
            tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
          end
          if tokenizer_class.nil?
            tokenizer_class_candidate = config_tokenizer_class
            tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
          end
          if tokenizer_class.nil?
            raise ArgumentError, "Tokenizer class #{tokenizer_class_candidate} does not exist or is not currently imported."
          end
          return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        end

        model_type = config_class_to_model_type(config.class.name.split("::").last)
        if !model_type.nil?
          tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[config.class.name.split("::").last]
          if tokenizer_class_fast && (use_fast || tokenizer_class_py.nil?)
            return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
          else
            if !tokenizer_class_py.nil?
              return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else
              raise ArgumentError, "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed in order to use this tokenizer."
            end
          end
        end

        raise ArgumentError, "Unrecognized configuration class #{config.class.name} to build an AutoTokenizer."
      end

      private

      def tokenizer_class_from_name(class_name)
        if class_name == "PreTrainedTokenizerFast"
          return PreTrainedTokenizerFast
        end

        TOKENIZER_MAPPING_NAMES.each do |module_name, tokenizers|
          if tokenizers.include?(class_name)
            cls = Transformers.const_get(module_name.capitalize).const_get(class_name)
            raise Error, "Invalid tokenizer class: #{class_name}" unless cls < PreTrainedTokenizer || cls < PreTrainedTokenizerFast
            return cls
          end
        end

        raise Todo
      end

      def get_tokenizer_config(
        pretrained_model_name_or_path,
        cache_dir: nil,
        force_download: false,
        resume_download: false,
        proxies: nil,
        token: nil,
        revision: nil,
        local_files_only: false,
        subfolder: "",
        **kwargs
      )
        commit_hash = kwargs[:_commit_hash]
        resolved_config_file = Utils::Hub.cached_file(
          pretrained_model_name_or_path,
          TOKENIZER_CONFIG_FILE,
          cache_dir: cache_dir,
          force_download: force_download,
          resume_download: resume_download,
          proxies: proxies,
          token: token,
          revision: revision,
          local_files_only: local_files_only,
          subfolder: subfolder,
          _raise_exceptions_for_gated_repo: false,
          _raise_exceptions_for_missing_entries: false,
          _raise_exceptions_for_connection_errors: false,
          _commit_hash: commit_hash
        )
        if resolved_config_file.nil?
          Transformers.logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
          return {}
        end
        commit_hash = Utils::Hub.extract_commit_hash(resolved_config_file, commit_hash)

        result = JSON.load_file(resolved_config_file)
        result["_commit_hash"] = commit_hash
        result
      end

      def config_class_to_model_type(config)
        CONFIG_MAPPING_NAMES.each do |key, cls|
          if cls == config
            return key
          end
        end
        nil
      end
    end
  end
end
