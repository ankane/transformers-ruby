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
  class PreTrainedTokenizerFast < PreTrainedTokenizerBase
    def initialize(*args, **kwargs)
      tokenizer_object = kwargs.delete(:tokenizer_object)
      slow_tokenizer = kwargs.delete(:__slow_tokenizer)
      fast_tokenizer_file = kwargs.delete(:tokenizer_file)
      from_slow = kwargs.delete(:from_slow) { false }
      _added_tokens_decoder = kwargs.delete(:added_tokens_decoder)

      if !tokenizer_object.nil?
        fast_tokenizer = Copy.deepcopy(tokenizer_object)
      elsif !fast_tokenizer_file.nil? && !from_slow
        # We have a serialization from tokenizers which let us directly build the backend
        fast_tokenizer = Tokenizers::Tokenizer.from_file(fast_tokenizer_file)
      elsif !slow_tokenizer.nil?
        # We need to convert a slow tokenizer to build the backend
        fast_tokenizer = ConvertSlowTokenizer.convert_slow_tokenizer(slow_tokenizer)
      elsif !@slow_tokenizer_class.nil?
        # We need to create and convert a slow tokenizer to build the backend
        slow_tokenizer = @slow_tokenizer_class.new(*args, **kwargs)
        fast_tokenizer = ConvertSlowTokenizer.convert_slow_tokenizer(slow_tokenizer)
      else
        raise ArgumentError, <<~MSG
          Couldn't instantiate the backend tokenizer from one of:
          (1) a `tokenizers` library serialization file,
          (2) a slow tokenizer instance to convert or
          (3) an equivalent slow tokenizer class to instantiate and convert.
          You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
        MSG
      end

      @tokenizer = fast_tokenizer

      if !slow_tokenizer.nil?
        kwargs.merge!(slow_tokenizer.init_kwargs)
      end

      @decode_use_source_tokenizer = false

      _truncation = @tokenizer.truncation

      if !_truncation.nil?
        _truncation = _truncation.transform_keys(&:to_sym)
        @tokenizer.enable_truncation(_truncation[:max_length], **_truncation.except(:max_length))
        kwargs[:max_length] ||= _truncation[:max_length]
        kwargs[:truncation_side] ||= _truncation[:direction]
        kwargs[:stride] ||= _truncation[:stride]
        kwargs[:truncation_strategy] ||= _truncation[:strategy]
      else
        @tokenizer.no_truncation
      end

      _padding = @tokenizer.padding
      if !_padding.nil?
        _padding = _padding.transform_keys(&:to_sym)
        @tokenizer.enable_padding(**_padding)
        kwargs[:pad_token] ||= _padding[:pad_token]
        kwargs[:pad_token_type_id] ||= _padding[:pad_token_type_id]
        kwargs[:padding_side] ||= _padding[:direction]
        kwargs[:max_length] ||= _padding[:length]
        kwargs[:pad_to_multiple_of] ||= _padding[:pad_to_multiple_of]
      end

      # We call this after having initialized the backend tokenizer because we update it.
      super(**kwargs)
    end

    def is_fast
      true
    end

    def get_vocab
      @tokenizer.vocab(with_added_tokens: true)
    end

    def vocab
      get_vocab
    end

    def backend_tokenizer
      @tokenizer
    end

    def convert_tokens_to_ids(tokens)
      if tokens.nil?
        return nil
      end

      if tokens.is_a?(String)
        return _convert_token_to_id_with_added_voc(tokens)
      end

      ids = []
      tokens.each do |token|
        ids << _convert_token_to_id_with_added_voc(token)
      end
      ids
    end

    def _convert_token_to_id_with_added_voc(token)
      index = @tokenizer.token_to_id(token)
      if index.nil?
        return unk_token_id
      end
      index
    end

    def convert_ids_to_tokens(ids, skip_special_tokens: false)
      if ids.is_a?(Integer)
        return @tokenizer.id_to_token(ids)
      end
      tokens = []
      ids.each do |index|
        index = index.to_i
        if skip_special_tokens && @all_special_ids.include?(index)
          next
        end
        tokens << @tokenizer.id_to_token(index)
      end
      tokens
    end

    def convert_tokens_to_string(tokens)
      backend_tokenizer.decoder.decode(tokens)
    end

    private

    def set_truncation_and_padding(
      padding_strategy:,
      truncation_strategy:,
      max_length:,
      stride:,
      pad_to_multiple_of:
    )
      _truncation = @tokenizer.truncation
      _padding = @tokenizer.padding
      # Set truncation and padding on the backend tokenizer
      if truncation_strategy == TruncationStrategy::DO_NOT_TRUNCATE
        if !_truncation.nil?
          @tokenizer.no_truncation
        end
      else
        target = {
          max_length: max_length,
          stride: stride,
          strategy: truncation_strategy,
          direction: @truncation_side
        }

        # _truncation might contain more keys that the target `transformers`
        # supports. Use only the target keys to trigger `enable_truncation`.
        # This should enable this code to works on various `tokenizers`
        # targets.
        if _truncation.nil?
          current = nil
        else
          current = target.to_h { |k, _| [k, _truncation[k]] }
        end

        if current != target
          @tokenizer.enable_truncation(target.delete(:max_length), **target)
        end
      end

      if padding_strategy == PaddingStrategy::DO_NOT_PAD
        if !_padding.nil?
          @tokenizer.no_padding
        end
      else
        length = padding_strategy == PaddingStrategy::MAX_LENGTH ? max_length : nil
        target = {
          length: length,
          direction: @padding_side,
          pad_id: @pad_token_id,
          pad_token: @pad_token,
          pad_type_id: @pad_token_type_id,
          pad_to_multiple_of: pad_to_multiple_of
        }
        if _padding != target
          @tokenizer.enable_padding(**target)
        end
      end
    end

    def _batch_encode_plus(
      batch_text_or_text_pairs,
      add_special_tokens: true,
      padding_strategy: PaddingStrategy::DO_NOT_PAD,
      truncation_strategy: TruncationStrategy::DO_NOT_TRUNCATE,
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
      verbose: true
    )
      if !batch_text_or_text_pairs.is_a?(Array)
        raise TypeError, "batch_text_or_text_pairs has to be an array (got #{batch_text_or_text_pairs.class.name})"
      end

      # Set the truncation and padding strategy and restore the initial configuration
      set_truncation_and_padding(
        padding_strategy: padding_strategy,
        truncation_strategy: truncation_strategy,
        max_length: max_length,
        stride: stride,
        pad_to_multiple_of: pad_to_multiple_of
      )

      encodings =
        @tokenizer.encode_batch(
          batch_text_or_text_pairs,
          add_special_tokens: add_special_tokens,
          is_pretokenized: is_split_into_words,
        )

      # Convert encoding to dict
      # `Tokens` has type: Tuple[
      #                       List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]],
      #                       List[EncodingFast]
      #                    ]
      # with nested dimensions corresponding to batch, overflows, sequence length
      tokens_and_encodings =
        encodings.map do |encoding|
          _convert_encoding(
            encoding: encoding,
            return_token_type_ids: return_token_type_ids,
            return_attention_mask: return_attention_mask,
            return_overflowing_tokens: return_overflowing_tokens,
            return_special_tokens_mask: return_special_tokens_mask,
            return_offsets_mapping: return_offsets_mapping,
            return_length: return_length,
            verbose: verbose
          )
        end

      # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
      # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
      # (we say ~ because the number of overflow varies with the example in the batch)
      #
      # To match each overflowing sample with the original sample in the batch
      # we add an overflow_to_sample_mapping array (see below)
      sanitized_tokens = {}
      tokens_and_encodings[0][0].each_key do |key|
        stack = tokens_and_encodings.map { |item, _| item[key][0] }
        sanitized_tokens[key] = stack
      end
      sanitized_encodings = tokens_and_encodings.map { |_, item| item[0] }

      # If returning overflowing tokens, we need to return a mapping
      # from the batch idx to the original sample
      if return_overflowing_tokens
        overflow_to_sample_mapping = []
        tokens_and_encodings.each_with_index do |(toks, _), i|
          overflow_to_sample_mapping += [i] * toks["input_ids"].length
        end
        sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping
      end

      sanitized_tokens["input_ids"].each do |input_ids|
        _eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
      end

      BatchEncoding.new(data: sanitized_tokens, encoding: sanitized_encodings, tensor_type: return_tensors)
    end

    def _convert_encoding(
      encoding:,
      return_token_type_ids: nil,
      return_attention_mask: nil,
      return_overflowing_tokens: false,
      return_special_tokens_mask: false,
      return_offsets_mapping: false,
      return_length: false,
      verbose: true
    )
      if return_token_type_ids.nil?
        return_token_type_ids = self.class.model_input_names.include?("token_type_ids")
      end
      if return_attention_mask.nil?
        return_attention_mask = self.class.model_input_names.include?("attention_mask")
      end

      if return_overflowing_tokens && !encoding.overflowing.nil?
        encodings = [encoding] + encoding.overflowing
      else
        encodings = [encoding]
      end

      encoding_dict = Hash.new { |h, k| h[k] = [] }
      encodings.each do |e|
        encoding_dict["input_ids"] << e.ids

        if return_token_type_ids
          encoding_dict["token_type_ids"] << e.type_ids
        end
        if return_attention_mask
          encoding_dict["attention_mask"] << e.attention_mask
        end
        if return_special_tokens_mask
          encoding_dict["special_tokens_mask"] << e.special_tokens_mask
        end
        if return_offsets_mapping
          encoding_dict["offset_mapping"] << e.offsets
        end
        if return_length
          encoding_dict["length"] << e.ids.length
        end
      end

      [encoding_dict, encodings]
    end

    def _encode_plus(
      text:,
      text_pair: nil,
      add_special_tokens: true,
      padding_strategy: PaddingStrategy::DO_NOT_PAD,
      truncation_strategy: TruncationStrategy::DO_NOT_TRUNCATE,
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
      batched_input = text_pair ? [[text, text_pair]] : [text]
      batched_output =
        _batch_encode_plus(
          batched_input,
          is_split_into_words: is_split_into_words,
          add_special_tokens: add_special_tokens,
          padding_strategy: padding_strategy,
          truncation_strategy: truncation_strategy,
          max_length: max_length,
          stride: stride,
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

      # Return tensor is None, then we can remove the leading batch axis
      # Overflowing tokens are returned as a batch of output so we keep them in this case
      if return_tensors.nil? && !return_overflowing_tokens
        batched_output =
          BatchEncoding.new(
            data: batched_output.items.to_h { |key, value|
              [key, value.length > 0 && value[0].is_a?(Array) ? value[0] : value]
            },
            encoding: batched_output.encodings
          )
      end

      _eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

      batched_output
    end
  end
end
