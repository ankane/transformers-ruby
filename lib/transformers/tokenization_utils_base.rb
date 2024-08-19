# Copyright 2020 The HuggingFace Inc. team.
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
  VERY_LARGE_INTEGER = 1e30.to_i  # This is used to set the max input length for a model with infinite size input
  LARGE_INTEGER = 1e20.to_i  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

  # Slow tokenizers used to be saved in three separated files
  SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
  ADDED_TOKENS_FILE = "added_tokens.json"
  TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

  # Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
  FULL_TOKENIZER_FILE = "tokenizer.json"

  class TruncationStrategy < ExplicitEnum
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"
  end

  class BatchEncoding
    def initialize(
      data: nil,
      encoding: nil,
      tensor_type: nil,
      prepend_batch_axis: false,
      n_sequences: nil
    )
      @data = data

      @encodings = encoding

      convert_to_tensors(tensor_type: tensor_type, prepend_batch_axis: prepend_batch_axis)
    end

    def convert_to_tensors(tensor_type: nil, prepend_batch_axis: false)
      if tensor_type.nil?
        return self
      end

      if !tensor_type.is_a?(TensorType)
        tensor_type = TensorType.new(tensor_type)
      end

      is_tensor = Torch.method(:tensor?)

      as_tensor = lambda do |value, dtype: nil|
        if value.is_a?(Array) && value[0].is_a?(Numo::NArray)
          return Torch.tensor(Numo::NArray.cast(value))
        end
        Torch.tensor(value)
      end

      items.each do |key, value|
        if prepend_batch_axis
          value = [value]
        end

        if !is_tensor.(value)
          tensor = as_tensor.(value)
          @data[key] = tensor
        end
      end
    end

    def [](item)
      if item.is_a?(String)
        @data[item]
      elsif item.is_a?(Symbol)
        @data[item.to_s]
      elsif !@encodings.nil?
        @encodings[item]
      elsif item.is_a?(Range)
        @data.keys.to_h { |key| [key, @data[key][item]] }
      else
        raise KeyError, "Invalid key. Only three types of key are available: (1) string, (2) integers for backend Encoding, and (3) ranges for data subsetting."
      end
    end

    def include?(item)
      @data.include?(item.to_s)
    end

    def delete(item)
      @data.delete(item.to_s)
    end

    def items
      @data
    end

    def encodings
      @encodings
    end

    def sequence_ids(batch_index = 0)
      if !@encodings
        raise ArgumentError,
          "sequence_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`" +
          " class)."
      end
      @encodings[batch_index].sequence_ids
    end

    def to_h
      @data.transform_keys(&:to_sym)
    end
    alias_method :to_hash, :to_h
  end

  module SpecialTokensMixin
    SPECIAL_TOKENS_ATTRIBUTES = [
      :bos_token,
      :eos_token,
      :unk_token,
      :sep_token,
      :pad_token,
      :cls_token,
      :mask_token,
      :additional_special_tokens
    ]
    attr_reader(*SPECIAL_TOKENS_ATTRIBUTES)

    def initialize(**kwargs)
      SPECIAL_TOKENS_ATTRIBUTES.each do |k|
        instance_variable_set("@#{k}", kwargs[k])
      end
    end

    def bos_token_id
      if @bos_token.nil?
        return nil
      end
      convert_tokens_to_ids(@bos_token)
    end

    def eos_token_id
      if @eos_token.nil?
        return nil
      end
      convert_tokens_to_ids(@eos_token)
    end

    def unk_token_id
      if @unk_token.nil?
        return nil
      end
      convert_tokens_to_ids(@unk_token)
    end

    def sep_token_id
      if @sep_token.nil?
        return nil
      end
      convert_tokens_to_ids(@sep_token)
    end

    def pad_token_id
      if @pad_token.nil?
        return nil
      end
      convert_tokens_to_ids(@pad_token)
    end

    def cls_token_id
      if @cls_token.nil?
        return nil
      end
      convert_tokens_to_ids(@cls_token)
    end

    def special_tokens_map
      set_attr = {}
      SPECIAL_TOKENS_ATTRIBUTES.each do |attr|
        attr_value = send(attr)
        if attr_value
          set_attr[attr] = attr_value
        end
      end
      set_attr
    end
  end

  class PreTrainedTokenizerBase
    include SpecialTokensMixin
    extend ClassAttribute

    class_attribute :vocab_files_names, {}

    class_attribute :model_input_names, ["input_ids", "token_type_ids", "attention_mask"]
    class_attribute :padding_side, "right"
    class_attribute :truncation_side, "right"
    class_attribute :slow_tokenizer_class

    attr_reader :init_kwargs, :model_max_length

    def initialize(**kwargs)
      # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
      @init_inputs = []
      @init_kwargs = kwargs.dup # copy.deepcopy(kwargs)
      @name_or_path = kwargs.delete(:name_or_path) { "" }
      @processor_class = kwargs.delete(:processor_class)

      # For backward compatibility we fallback to set model_max_length from max_len if provided
      model_max_length = kwargs.delete(:model_max_length) { kwargs.delete(:max_len) }
      @model_max_length = !model_max_length.nil? ? model_max_length : VERY_LARGE_INTEGER

      # Padding and truncation side are right by default and overridden in subclasses. If specified in the kwargs, it
      # is changed.
      @padding_side = kwargs.delete(:padding_side) { self.class.padding_side }
      if !["right", "left"].include?(@padding_side)
        raise ArgumentError, "Padding side should be selected between 'right' and 'left', current value: #{@padding_side}"
      end

      @truncation_side = kwargs.delete(:truncation_side) { self.class.truncation_side }
      if !["right", "left"].include?(@truncation_side)
        raise ArgumentError, "Truncation side should be selected between 'right' and 'left', current value: #{@truncation_side}"
      end

      @model_input_names = kwargs.delete(:model_input_names) { self.class.model_input_names }

      # By default, cleaning tokenization spaces for both fast and slow tokenizers
      @clean_up_tokenization_spaces = kwargs.delete(:clean_up_tokenization_spaces) { true }

      # By default, do not split special tokens for both fast and slow tokenizers
      @split_special_tokens = kwargs.delete(:split_special_tokens) { false }

      @deprecation_warnings = {}
      @in_target_context_manager = false

      # Stores a Jinja template that formats chat histories into tokenizable strings
      @chat_template = kwargs.delete(:chat_template)
      if @chat_template.is_a?(Array)
        # Chat templates are stored as lists of dicts with fixed key names,
        # we reconstruct that into a single dict while loading them.
        @chat_template = @chat_template.to_h { |template| [template["name"], template["template"]] }
      end

      super
    end

    def _eventual_warn_about_too_long_sequence(ids, max_length, verbose)
      if max_length.nil? && ids.length > @model_max_length && verbose
        raise Todo
      end
    end

    def call(
      text,
      text_pair: nil,
      text_target: nil,
      text_pair_target: nil,
      add_special_tokens: true,
      padding: false,
      truncation: nil,
      max_length: nil,
      stride: 0,
      is_split_into_words: false,
      pad_to_multiple_of: nil,
      return_tensors: nil,
      return_token_type_ids: nil,
      return_attention_mask: nil,
      return_overflowing_tokens: false,
      return_special_tokens_mask: false,
      return_offsets_mapping: false,
      return_length: false,
      verbose: true,
      **kwargs
    )
      # To avoid duplicating
      all_kwargs = {
        add_special_tokens: add_special_tokens,
        padding: padding,
        truncation: truncation,
        max_length: max_length,
        stride: stride,
        is_split_into_words: is_split_into_words,
        pad_to_multiple_of: pad_to_multiple_of,
        return_tensors: return_tensors,
        return_token_type_ids: return_token_type_ids,
        return_attention_mask: return_attention_mask,
        return_overflowing_tokens: return_overflowing_tokens,
        return_special_tokens_mask: return_special_tokens_mask,
        return_offsets_mapping: return_offsets_mapping,
        return_length: return_length,
        verbose: verbose
      }
      all_kwargs.merge!(kwargs)
      if text.nil? && text_target.nil?
        raise ArgumentError, "You need to specify either `text` or `text_target`."
      end
      if !text.nil?
        # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
        # input mode in this case.
        if !@in_target_context_manager
          _switch_to_input_mode
        end
        encodings = _call_one(text: text, text_pair: text_pair, **all_kwargs)
      end
      if !text_target.nil?
        _switch_to_target_mode
        target_encodings = _call_one(text: text_target, text_pair: text_pair_target, **all_kwargs)
      end
      # Leave back tokenizer in input mode
      _switch_to_input_mode

      if text_target.nil?
        encodings
      elsif text.nil?
        target_encodings
      else
        encodings["labels"] = target_encodings["input_ids"]
        encodings
      end
    end

    protected

    def _switch_to_input_mode
    end

    def _switch_to_target_mode
    end

    private

    def _call_one(
      text:,
      text_pair: nil,
      add_special_tokens: true,
      padding: false,
      truncation: nil,
      max_length: nil,
      stride: 0,
      is_split_into_words: false,
      pad_to_multiple_of: nil,
      return_tensors: nil,
      return_token_type_ids: nil,
      return_attention_mask: nil,
      return_overflowing_tokens: false,
      return_special_tokens_mask: false,
      return_offsets_mapping: false,
      return_length: false,
      verbose: true,
      **kwargs
    )
      # Input type checking for clearer error
      _is_valid_text_input = lambda do |t|
        if t.is_a?(String)
          # Strings are fine
          true
        elsif t.is_a?(Array)
          # List are fine as long as they are...
          if t.length == 0
            # ... empty
            true
          elsif t[0].is_a?(String)
            # ... list of strings
            true
          elsif t[0].is_a?(Array)
            # ... list with an empty list or with a list of strings
            t[0].length == 0 || t[0][0].is_a?(String)
          else
            false
          end
        else
          false
        end
      end

      if !_is_valid_text_input.(text)
        raise ArgumentError, "text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples)."
      end

      if !text_pair.nil? && !_is_valid_text_input.(text_pair)
        raise ArgumentError, "text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples)."
      end

      if is_split_into_words
        is_batched = text.is_a?(Array) && text[0].is_a?(Array)
      else
        is_batched = text.is_a?(Array)
      end

      if is_batched
        if text_pair.is_a?(String)
          raise TypeError, "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`."
        end
        if !text_pair.nil? && text.length != text_pair.length
          raise ArgumentError, "batch length of `text`: #{text.length} does not match batch length of `text_pair`: #{text_pair.length}."
        end
        batch_text_or_text_pairs = !text_pair.nil? ? text.zip(text_pair).to_a : text
        batch_encode_plus(
          batch_text_or_text_pairs: batch_text_or_text_pairs,
          add_special_tokens: add_special_tokens,
          padding: padding,
          truncation: truncation,
          max_length: max_length,
          stride: stride,
          is_split_into_words: is_split_into_words,
          pad_to_multiple_of: pad_to_multiple_of,
          return_tensors: return_tensors,
          return_token_type_ids: return_token_type_ids,
          return_attention_mask: return_attention_mask,
          return_overflowing_tokens: return_overflowing_tokens,
          return_special_tokens_mask: return_special_tokens_mask,
          return_offsets_mapping: return_offsets_mapping,
          return_length: return_length,
          verbose: verbose,
          **kwargs
        )
      else
        encode_plus(
          text: text,
          text_pair: text_pair,
          add_special_tokens: add_special_tokens,
          padding: padding,
          truncation: truncation,
          max_length: max_length,
          stride: stride,
          is_split_into_words: is_split_into_words,
          pad_to_multiple_of: pad_to_multiple_of,
          return_tensors: return_tensors,
          return_token_type_ids: return_token_type_ids,
          return_attention_mask: return_attention_mask,
          return_overflowing_tokens: return_overflowing_tokens,
          return_special_tokens_mask: return_special_tokens_mask,
          return_offsets_mapping: return_offsets_mapping,
          return_length: return_length,
          verbose: verbose,
          **kwargs
        )
      end
    end

    def encode_plus(
      text:,
      text_pair: nil,
      add_special_tokens: true,
      padding: false,
      truncation: nil,
      max_length: nil,
      stride: 0,
      is_split_into_words: false,
      pad_to_multiple_of: nil,
      return_tensors: nil,
      return_token_type_ids: nil,
      return_attention_mask: nil,
      return_overflowing_tokens: false,
      return_special_tokens_mask: false,
      return_offsets_mapping: false,
      return_length: false,
      verbose: true,
      **kwargs
    )
      # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
      padding_strategy, truncation_strategy, max_length, kwargs =
        _get_padding_truncation_strategies(
          padding: padding,
          truncation: truncation,
          max_length: max_length,
          pad_to_multiple_of: pad_to_multiple_of,
          verbose: verbose,
          **kwargs
        )

      _encode_plus(
        text: text,
        text_pair: text_pair,
        add_special_tokens: add_special_tokens,
        padding_strategy: padding_strategy,
        truncation_strategy: truncation_strategy,
        max_length: max_length,
        stride: stride,
        is_split_into_words: is_split_into_words,
        pad_to_multiple_of: pad_to_multiple_of,
        return_tensors: return_tensors,
        return_token_type_ids: return_token_type_ids,
        return_attention_mask: return_attention_mask,
        return_overflowing_tokens: return_overflowing_tokens,
        return_special_tokens_mask: return_special_tokens_mask,
        return_offsets_mapping: return_offsets_mapping,
        return_length: return_length,
        verbose: verbose,
        **kwargs
      )
    end

    def batch_encode_plus(
      batch_text_or_text_pairs:,
      add_special_tokens: true,
      padding: false,
      truncation: nil,
      max_length: nil,
      stride: 0,
      is_split_into_words: false,
      pad_to_multiple_of: nil,
      return_tensors: nil,
      return_token_type_ids: nil,
      return_attention_mask: nil,
      return_overflowing_tokens: false,
      return_special_tokens_mask: false,
      return_offsets_mapping: false,
      return_length: false,
      verbose: true,
      **kwargs
    )
      # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
      padding_strategy, truncation_strategy, max_length, kwargs =
        _get_padding_truncation_strategies(
          padding: padding,
          truncation: truncation,
          max_length: max_length,
          pad_to_multiple_of: pad_to_multiple_of,
          verbose: verbose,
          **kwargs
        )

      _batch_encode_plus(
        batch_text_or_text_pairs,
        add_special_tokens: add_special_tokens,
        padding_strategy: padding_strategy,
        truncation_strategy: truncation_strategy,
        max_length: max_length,
        stride: stride,
        is_split_into_words: is_split_into_words,
        pad_to_multiple_of: pad_to_multiple_of,
        return_tensors: return_tensors,
        return_token_type_ids: return_token_type_ids,
        return_attention_mask: return_attention_mask,
        return_overflowing_tokens: return_overflowing_tokens,
        return_special_tokens_mask: return_special_tokens_mask,
        return_offsets_mapping: return_offsets_mapping,
        return_length: return_length,
        verbose: verbose,
        **kwargs
      )
    end

    def _get_padding_truncation_strategies(
      padding: false,
      truncation: nil,
      max_length: nil,
      pad_to_multiple_of: nil,
      verbose: true,
      **kwargs
    )
      padding_strategy = PaddingStrategy::DO_NOT_PAD
      truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE

      old_truncation_strategy = kwargs.delete(:truncation_strategy) || "do_not_truncate"
      old_pad_to_max_length = kwargs.delete(:pad_to_max_length) || false

      # Backward compatibility for previous behavior, maybe we should deprecate it:
      # If you only set max_length, it activates truncation for max_length
      if !max_length.nil? && padding == false && truncation.nil?
        raise Todo
      end

      # Get padding strategy
      if padding == false && old_pad_to_max_length
          if verbose
            raise Todo
          end
          if max_length.nil?
            padding_strategy = PaddingStrategy::LONGEST
          else
            padding_strategy = PaddingStrategy::MAX_LENGTH
          end
      elsif padding != false
        if padding == true
          if verbose
            # raise Todo
          end
          padding_strategy = PaddingStrategy::LONGEST  # Default to pad to the longest sequence in the batch
        elsif !padding.is_a?(PaddingStrategy)
          padding_strategy = PaddingStrategy.new(padding)
        elsif padding.is_a?(PaddingStrategy)
          padding_strategy = padding
        end
      else
        padding_strategy = PaddingStrategy::DO_NOT_PAD
      end

      # Get truncation strategy
      if truncation.nil? && old_truncation_strategy != "do_not_truncate"
        if verbose
          raise Todo
        end
        truncation_strategy = TruncationStrategy.new(old_truncation_strategy).to_s
      elsif truncation != false && !truncation.nil?
        if truncation == true
          truncation_strategy = (
            TruncationStrategy::LONGEST_FIRST
          )  # Default to truncate the longest sequences in pairs of inputs
        elsif !truncation.is_a?(TruncationStrategy)
          truncation_strategy = TruncationStrategy.new(truncation).to_s
        else
          truncation_strategy = truncation
        end
      else
        truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE
      end

      # Set max length if needed
      if max_length.nil?
        if padding_strategy == PaddingStrategy::MAX_LENGTH
          if @model_max_length > LARGE_INTEGER
            if verbose
              raise Todo
            end
            padding_strategy = PaddingStrategy::DO_NOT_PAD
          else
            max_length = @model_max_length
          end
        end

        if truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE
          if @model_max_length > LARGE_INTEGER
            if verbose
              raise Todo
            end
            truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE
          else
            max_length = @model_max_length
          end
        end
      end

      # Test if we have a padding token
      if padding_strategy != PaddingStrategy::DO_NOT_PAD && (@pad_token.nil? || pad_token_id < 0)
        raise ArgumentError,
          "Asking to pad but the tokenizer does not have a padding token. " +
          "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` " +
          "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
      end

      # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
      if (
        truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE &&
        padding_strategy != PaddingStrategy::DO_NOT_PAD &&
        !pad_to_multiple_of.nil? &&
        !max_length.nil? &&
        (max_length % pad_to_multiple_of != 0)
      )
        raise ArgumentError,
          "Truncation and padding are both activated but " +
          "truncation length (#{max_length}) is not a multiple of pad_to_multiple_of (#{pad_to_multiple_of})."
      end

      [padding_strategy, truncation_strategy, max_length, kwargs]
    end

    class << self
      def from_pretrained(
        pretrained_model_name_or_path,
        *init_inputs,
        cache_dir: nil,
        force_download: false,
        local_files_only: false,
        token: nil,
        revision: "main",
        trust_remote_code: false,
        **kwargs
      )
        resume_download = kwargs.delete(:resume_download) { false }
        proxies = kwargs.delete(:proxies)
        subfolder = kwargs.delete(:subfolder)
        from_pipeline = kwargs.delete(:_from_pipeline)
        from_auto_class = kwargs.delete(:_from_auto) { false }
        commit_hash = kwargs.delete(:_commit_hash)

        user_agent = {file_type: "tokenizer", from_auto_class: from_auto_class, is_fast: name.include?("Fast")}
        if !from_pipeline.nil?
          user_agent[:using_pipeline] = from_pipeline
        end

        if Utils::Hub.is_offline_mode && !local_files_only
          Transformers.logger.info("Offline mode: forcing local_files_only: true")
          local_files_only = true
        end

        pretrained_model_name_or_path = pretrained_model_name_or_path.to_s
        vocab_files = {}
        init_configuration = {}

        is_local = Dir.exist?(pretrained_model_name_or_path)
        single_file_id = nil
        if File.exist?(pretrained_model_name_or_path)
          raise Todo
        end

        # At this point pretrained_model_name_or_path is either a directory or a model identifier name
        additional_files_names = {
          added_tokens_file: ADDED_TOKENS_FILE, # kept only for legacy
          special_tokens_map_file: SPECIAL_TOKENS_MAP_FILE, # kept only for legacy
          tokenizer_config_file: TOKENIZER_CONFIG_FILE,
          # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
          tokenizer_file: FULL_TOKENIZER_FILE
        }
        vocab_files = vocab_files_names.merge(additional_files_names)
        if vocab_files[:tokenizer_file]
          # Try to get the tokenizer config to see if there are versioned tokenizer files.
          fast_tokenizer_file = FULL_TOKENIZER_FILE
          resolved_config_file =
            Utils::Hub.cached_file(
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
              user_agent: user_agent,
              _raise_exceptions_for_gated_repo: false,
              _raise_exceptions_for_missing_entries: false,
              _raise_exceptions_for_connection_errors: false,
              _commit_hash: commit_hash
            )
          commit_hash = Utils::Hub.extract_commit_hash(resolved_config_file, commit_hash)
          if !resolved_config_file.nil?
            tokenizer_config = JSON.load_file(resolved_config_file)
            if tokenizer_config["fast_tokenizer_files"]
              fast_tokenizer_file = get_fast_tokenizer_file(tokenizer_config["fast_tokenizer_files"])
            end
          end
          vocab_files[:tokenizer_file] = fast_tokenizer_file
        end

        # Get files from url, cache, or disk depending on the case
        resolved_vocab_files = {}
        unresolved_files = []
        vocab_files.each do |file_id, file_path|
          if file_path.nil?
            resolved_vocab_files[file_id] = nil
          elsif single_file_id == file_id
            if File.exist?(file_path)
              resolved_vocab_files[file_id] = file_path
            else
              raise Todo
            end
          else
            resolved_vocab_files[file_id] =
              Utils::Hub.cached_file(
                pretrained_model_name_or_path,
                file_path,
                cache_dir: cache_dir,
                force_download: force_download,
                proxies: proxies,
                resume_download: resume_download,
                local_files_only: local_files_only,
                token: token,
                user_agent: user_agent,
                revision: revision,
                subfolder: subfolder,
                _raise_exceptions_for_gated_repo: false,
                _raise_exceptions_for_missing_entries: false,
                _raise_exceptions_for_connection_errors: false,
                _commit_hash: commit_hash,
              )
            commit_hash = Utils::Hub.extract_commit_hash(resolved_vocab_files[file_id], commit_hash)
          end
        end

        # not used?
        if unresolved_files.length > 0
          raise Todo
        end

        vocab_files.each do |file_id, file_path|
          if !resolved_vocab_files.include?(file_id)
            next
          end

          if is_local
            Transformers.logger.info("loading file #{file_path}")
          else
            Transformers.logger.info("loading file #{file_path} from cache at #{resolved_vocab_files[file_id] || "nil"}")
          end
        end

        _from_pretrained(
          resolved_vocab_files,
          pretrained_model_name_or_path,
          init_configuration,
          *init_inputs,
          token: token,
          cache_dir: cache_dir,
          local_files_only: local_files_only,
          _commit_hash: commit_hash,
          _is_local: is_local,
          trust_remote_code: trust_remote_code,
          **kwargs
        )
      end

      def _from_pretrained(
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token: nil,
        cache_dir: nil,
        local_files_only: false,
        _commit_hash: nil,
        _is_local: false,
        trust_remote_code: false,
        **kwargs
      )
        # We instantiate fast tokenizers based on a slow tokenizer if we don't have access to the tokenizer.json
        # file or if `from_slow` is set to True.
        from_slow = kwargs.delete(:from_slow) { false }
        has_tokenizer_file = !resolved_vocab_files[:tokenizer_file].nil?
        if (from_slow || !has_tokenizer_file) && !slow_tokenizer_class.nil?
          slow_tokenizer =
            slow_tokenizer_class._from_pretrained(
              Copy.deepcopy(resolved_vocab_files),
              pretrained_model_name_or_path,
              Copy.deepcopy(init_configuration),
              *init_inputs,
              token: token,
              cache_dir: cache_dir,
              local_files_only: local_files_only,
              _commit_hash: _commit_hash,
              **Copy.deepcopy(kwargs)
            )
        else
          slow_tokenizer = nil
        end

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.delete(:tokenizer_config_file)
        if !tokenizer_config_file.nil?
          init_kwargs = JSON.load_file(tokenizer_config_file).transform_keys(&:to_sym)
          # First attempt. We get tokenizer_class from tokenizer_config to check mismatch between tokenizers.
          config_tokenizer_class = init_kwargs[:tokenizer_class]
          init_kwargs.delete(:tokenizer_class)
          if !has_tokenizer_file
            init_kwargs.delete(:tokenizer_file)
          end
          saved_init_inputs = init_kwargs.delete(:init_inputs) { [] }
          if init_inputs.empty?
            init_inputs = saved_init_inputs
          end
        else
          config_tokenizer_class = nil
          init_kwargs = init_configuration
        end

        if config_tokenizer_class.nil?
          config =
            AutoConfig.from_pretrained(
              pretrained_model_name_or_path,
              token: token,
              cache_dir: cache_dir,
              local_files_only: local_files_only,
              trust_remote_code: trust_remote_code,
              _commit_hash: _commit_hash,
            )
          config_tokenizer_class = config.tokenizer_class

          if config_tokenizer_class.nil?
            # Third attempt. If we have not yet found the original type of the tokenizer,
            # we are loading we see if we can infer it from the type of the configuration file
            if config.class.model_type
              model_type = config.class.model_type
            else
              # Fallback: use pattern matching on the string.
              model_type = nil
              TOKENIZER_MAPPING_NAMES.each_key do |pattern|
                if pretrained_model_name_or_path.to_s.include?(pattern)
                  model_type = pattern
                  break
                end
              end
            end

            if !model_type.nil?
              config_tokenizer_class, config_tokenizer_class_fast =
                TOKENIZER_MAPPING_NAMES.fetch(model_type, [nil, nil])

              if config_tokenizer_class.nil?
                config_tokenizer_class = config_tokenizer_class_fast
              end
            end
          end
        end

        if !config_tokenizer_class.nil?
          if name.split("::").last.gsub("Fast", "") != config_tokenizer_class.gsub("Fast", "")
            raise Todo
          end
        end

        # Update with newly provided kwargs
        init_kwargs.merge!(kwargs)

        # Merge resolved_vocab_files arguments in init_kwargs.
        _added_tokens_file = resolved_vocab_files.delete(:added_tokens_file)
        _special_tokens_map_file = resolved_vocab_files.delete(:special_tokens_map_file)
        resolved_vocab_files.each do |args_name, file_path|
          if !init_kwargs.include?(args_name)
            init_kwargs[args_name] = file_path
          end
        end
        _tokenizer_file = resolved_vocab_files.delete(:tokenizer_file)

        if !slow_tokenizer.nil?
          init_kwargs[:__slow_tokenizer] = slow_tokenizer
        end
        init_kwargs[:name_or_path] = pretrained_model_name_or_path

        # Instantiate the tokenizer.
        tokenizer = new(*init_inputs, **init_kwargs)

        tokenizer
      end
    end
  end
end
