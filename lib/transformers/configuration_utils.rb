# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
  class PretrainedConfig
    extend ClassAttribute

    class_attribute :model_type, ""
    class_attribute :attribute_map, {}

    # TODO support setter
    def method_missing(m, *args, **kwargs)
      if self.class.attribute_map.include?(m)
        instance_variable_get("@#{self.class.attribute_map[m]}")
      else
        super
      end
    end

    # TODO support setter
    def respond_to_missing?(m, include_private = true)
      self.class.attribute_map.include?(m) || super
    end

    attr_reader :output_hidden_states, :output_attentions, :pruned_heads, :tie_word_embeddings, :tokenizer_class,
      :chunk_size_feed_forward, :pad_token_id, :is_decoder, :add_cross_attention,
      :problem_type, :id2label, :architectures, :is_encoder_decoder, :tie_encoder_decoder, :_commit_hash

    def initialize(**kwargs)
      @return_dict = kwargs.delete(:return_dict) { true }
      @output_hidden_states = kwargs.delete(:output_hidden_states) { false }
      @output_attentions = kwargs.delete(:output_attentions) { false }
      @pruned_heads = kwargs.delete(:pruned_heads) { {} }
      @tie_word_embeddings = kwargs.delete(:tie_word_embeddings) { true }
      @chunk_size_feed_forward = kwargs.delete(:chunk_size_feed_forward) { 0 }

      # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
      @is_encoder_decoder = kwargs.delete(:is_encoder_decoder) { false }
      @is_decoder = kwargs.delete(:is_decoder) { false }
      @cross_attention_hidden_size = kwargs.delete(:cross_attention_hidden_size)
      @add_cross_attention = kwargs.delete(:add_cross_attention) { false }
      @tie_encoder_decoder = kwargs.delete(:tie_encoder_decoder) { false }

      # Fine-tuning task arguments
      @architectures = kwargs.delete(:architectures)
      @finetuning_task = kwargs.delete(:finetuning_task)
      @id2label = kwargs.delete(:id2label)
      @label2id = kwargs.delete(:label2id)
      if !@label2id.nil? && !@label2id.is_a?(Hash)
        raise ArgumentError, "Argument label2id should be a dictionary."
      end
      if !@id2label.nil?
        if !@id2label.is_a?(Hash)
          raise ArgumentError, "Argument id2label should be a dictionary."
        end
        num_labels = kwargs.delete(:num_labels)
        if !num_labels.nil? && id2label.length != num_labels
          raise Todo
        end
        @id2label = @id2label.transform_keys(&:to_i)
        # Keys are always strings in JSON so convert ids to int here.
      else
        self.num_labels = kwargs.delete(:num_labels) { 2 }
      end

      # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
      @tokenizer_class = kwargs.delete(:tokenizer_class)
      @prefix = kwargs.delete(:prefix)
      @bos_token_id = kwargs.delete(:bos_token_id)
      @pad_token_id = kwargs.delete(:pad_token_id)
      @eos_token_id = kwargs.delete(:eos_token_id)
      @sep_token_id = kwargs.delete(:sep_token_id)

      # regression / multi-label classification
      @problem_type = kwargs.delete(:problem_type)

      # Name or path to the pretrained checkpoint
      @name_or_path = kwargs.delete(:name_or_path).to_s
      # Config hash
      @commit_hash = kwargs.delete(:_commit_hash)

      # Attention implementation to use, if relevant.
      @attn_implementation_internal = kwargs.delete(:attn_implementation)

      # Drop the transformers version info
      @transformers_version = kwargs.delete(:transformers_version)

      # Deal with gradient checkpointing
      # if kwargs[:gradient_checkpointing] == false
      #   warn(
      #     "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 " +
      #     "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the " +
      #     "`Trainer` API, pass `gradient_checkpointing: true` in your `TrainingArguments`."
      #   )
      # end

      kwargs.each do |k, v|
        instance_variable_set("@#{k}", v)
      end
    end

    def name_or_path
      @name_or_path
    end

    def name_or_path=(value)
      @name_or_path = value.to_s
    end

    def num_labels
      @id2label.length
    end

    def num_labels=(num_labels)
      if @id2label.nil? || @id2label.length != num_labels
        @id2label = num_labels.times.to_h { |i| [i, "LABEL_#{i}"] }
        @label2id =  @id2label.invert
      end
    end

    def _attn_implementation
      # This property is made private for now (as it cannot be changed and a PreTrainedModel.use_attn_implementation method needs to be implemented.)
      if instance_variable_defined?(:@attn_implementation_internal)
        if instance_variable_get(:@attn_implementation_internal).nil?
          # `config.attn_implementation` should never be None, for backward compatibility.
          "eager"
        else
          @attn_implementation_internal
        end
      else
        "eager"
      end
    end

    def use_return_dict
      @return_dict
    end

    def to_s
      "#{self.class.name} #{to_json_string}"
    end

    def to_diff_dict
      config_dict = to_dict

      # get the default config dict
      default_config_dict = PretrainedConfig.new.to_dict

      serializable_config_dict = {}

      config_dict.each do |key, value|
        key = :_name_or_path if key == :name_or_path
        if !default_config_dict.include?(key) || value != default_config_dict[key] || key == :transformers_version
          serializable_config_dict[key] = value
        end
      end

      serializable_config_dict
    end

    def _dict
      instance_variables.to_h { |k| [k[1..].to_sym, instance_variable_get(k)] }
    end

    def to_dict
      output = Copy.deepcopy(_dict)
      output[:model_type] = self.class.model_type
      output.delete(:_auto_class)
      output.delete(:_commit_hash)
      output.delete(:_attn_implementation_internal)

      # Transformers version when serializing the model
      output[:transformers_version] = VERSION

      output
    end

    def to_json_string(use_diff: true)
      if use_diff == true
        config_dict = to_diff_dict
      else
        config_dict = to_dict
      end
      JSON.pretty_generate(config_dict.sort_by { |k, _| k }.to_h) + "\n"
    end

    def getattr(key, default)
      if respond_to?(key)
        public_send(key)
      elsif instance_variable_defined?("@#{key}")
        instance_variable_get("@#{key}")
      else
        default
      end
    end

    def hasattr(key)
      respond_to?(key) || instance_variable_defined?("@#{key}")
    end

    class << self
      def from_pretrained(
        pretrained_model_name_or_path,
        cache_dir: nil,
        force_download: false,
        local_files_only: false,
        token: nil,
        revision: "main",
        **kwargs
      )
        config_dict, kwargs = get_config_dict(pretrained_model_name_or_path, **kwargs)

        from_dict(config_dict, **kwargs)
      end

      def from_dict(config_dict, **kwargs)
        return_unused_kwargs = kwargs.delete(:return_unused_kwargs) { false }

        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if kwargs.include?(:_commit_hash) && config_dict.include?(:_commit_hash)
          kwargs[:_commit_hash] = config_dict[:_commit_hash]
        end

        config = new(**config_dict)

        to_remove = []
        kwargs.each do |key, value|
          if config.respond_to?("#{key}=")
            config.public_send("#{key}=", value)
          end
          if key != :torch_dtype
            to_remove << key
          end
        end
        to_remove.each do |key|
          kwargs.delete(key)
        end

        Transformers.logger.info("Model config #{config}")
        if return_unused_kwargs
          [config, kwargs]
        else
          config
        end
      end

      def get_config_dict(pretrained_model_name_or_path, **kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = _get_config_dict(pretrained_model_name_or_path, **kwargs)

        [config_dict, kwargs]
      end

      private

      def _get_config_dict(pretrained_model_name_or_path, **kwargs)
        cache_dir = kwargs.delete(:cache_dir)
        force_download = kwargs.delete(:force_download) { false }
        resume_download = kwargs.delete(:resume_download) { false }
        proxies = kwargs.delete(:proxies)
        token = kwargs.delete(:token)
        local_files_only = kwargs.delete(:local_files_only) { false }
        revision = kwargs.delete(:revision)
        _trust_remote_code = kwargs.delete(:trust_remote_code)
        subfolder = kwargs.delete(:subfolder) { "" }
        _from_pipeline = kwargs.delete(:_from_pipeline)
        from_auto_class = kwargs.delete(:_from_auto) { false }
        commit_hash = kwargs.delete(:_commit_hash)

        user_agent = {file_type: "config", from_auto_class: from_auto_class}

        is_local = Dir.exist?(pretrained_model_name_or_path)
        configuration_file = kwargs.delete(:_configuration_file) || CONFIG_NAME

        resolved_config_file = Utils::Hub.cached_file(
          pretrained_model_name_or_path,
          configuration_file,
          cache_dir: cache_dir,
          force_download: force_download,
          proxies: proxies,
          resume_download: resume_download,
          local_files_only: local_files_only,
          token: token,
          user_agent: user_agent,
          revision: revision,
          subfolder: subfolder,
          _commit_hash: commit_hash
        )
        commit_hash = Utils::Hub.extract_commit_hash(resolved_config_file, commit_hash)

        config_dict = _dict_from_json_file(resolved_config_file)
        config_dict[:_commit_hash] = commit_hash

        if is_local
          Transformers.logger.info("loading configuration file #{resolved_config_file}")
        else
          Transformers.logger.info("loading configuration file #{configuration_file} from cache at #{resolved_config_file}")
        end

        [config_dict, kwargs]
      end

      def _dict_from_json_file(json_file)
        JSON.load_file(json_file).transform_keys(&:to_sym)
      end
    end
  end
end
