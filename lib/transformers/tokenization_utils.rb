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
  class PreTrainedTokenizer < PreTrainedTokenizerBase
    def initialize(**kwargs)
      # 2. init `_added_tokens_decoder` if child class did not
      if !instance_variable_defined?(:@added_tokens_decoder)
        @added_tokens_decoder = {}
      end

      # 3. if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite
      @added_tokens_decoder.merge!(kwargs.delete(:added_tokens_decoder) { {} })
      @added_tokens_encoder = @added_tokens_decoder.to_h { |k, v| [k.content, v] }

      # 4 init the parent class
      super(**kwargs)
    end

    def is_fast
      false
    end

    def vocab_size
      raise NotImplementedError
    end

    def tokenize(text, **kwargs)
      raise Todo
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
      get_input_ids = lambda do |text|
        if text.is_a?(String)
          tokens = tokenize(text, **kwargs)
          convert_tokens_to_ids(tokens)
        elsif text.is_a?(Array) && text.length > 0 && text[0].is_a?(String)
          if is_split_into_words
            raise Todo
          else
            convert_tokens_to_ids(text)
          end
        elsif text.is_a?(Array) && text.length > 0 && text[0].is_a?(Integer)
          text
        else
          if is_split_into_words
            raise ArgumentError,
              "Input #{text} is not valid. Should be a string or a list/tuple of strings when" +
              " `is_split_into_words=True`."
          else
            raise ArgumentError,
              "Input #{text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of" +
              " integers."
          end
        end
      end

      if return_offsets_mapping
        raise RuntimeError,
          "return_offset_mapping is not available when using Ruby tokenizers. " +
          "To use this feature, change your tokenizer to one deriving from " +
          "Transformers::PreTrainedTokenizerFast. " +
          "More information on available tokenizers at " +
          "https://github.com/huggingface/transformers/pull/2674"
      end

      first_ids = get_input_ids.(text)
      second_ids = !text_pair.nil? ? get_input_ids.(text_pair) : nil

      prepare_for_model(
        first_ids,
        pair_ids: second_ids,
        add_special_tokens: add_special_tokens,
        padding: padding_strategy,
        truncation: truncation_strategy,
        max_length: max_length,
        stride: stride,
        pad_to_multiple_of: pad_to_multiple_of,
        return_tensors: return_tensors,
        prepend_batch_axis: true,
        return_attention_mask: return_attention_mask,
        return_token_type_ids: return_token_type_ids,
        return_overflowing_tokens: return_overflowing_tokens,
        return_special_tokens_mask: return_special_tokens_mask,
        return_length: return_length,
        verbose: verbose
      )
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
      if token.nil?
        return nil
      end

      if @added_tokens_encoder.include?(token)
        return @added_tokens_encoder[token]
      end
      _convert_token_to_id(token)
    end

    def _convert_token_to_id(token)
      raise NotImplementedError
    end
  end
end
