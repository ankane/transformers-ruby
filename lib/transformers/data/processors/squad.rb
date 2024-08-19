# Copyright 2020 The HuggingFace Team. All rights reserved.
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
  class SquadExample
    attr_reader :question_text, :context_text

    def initialize(
      qas_id,
      question_text,
      context_text,
      answer_text,
      start_position_character,
      title,
      answers: [],
      is_impossible: false
    )
      @qas_id = qas_id
      @question_text = question_text
      @context_text = context_text
      @answer_text = answer_text
      @title = title
      @is_impossible = is_impossible
      @answers = answers

      @start_position, @end_position = 0, 0

      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = true

      # Split on whitespace so that different tokens may be attributed to their original position.
      @context_text.each_char do |c|
        if _is_whitespace(c)
          prev_is_whitespace = true
        else
          if prev_is_whitespace
            doc_tokens << c
          else
            doc_tokens[-1] += c
          end
          prev_is_whitespace = false
        end
        char_to_word_offset << (doc_tokens.length - 1)
      end

      @doc_tokens = doc_tokens
      @char_to_word_offset = char_to_word_offset

      # Start and end positions only has a value during evaluation.
      if !start_position_character.nil? && !is_impossible
        @start_position = char_to_word_offset[start_position_character]
        @end_position = char_to_word_offset[
          [start_position_character + answer_text.length - 1, char_to_word_offset.length - 1].min
        ]
      end
    end

    def _is_whitespace(c)
      c == " " || c == "\t" || c == "\r" || c == "\n" || c.ord == 0x202F
    end
  end

  class SquadFeatures
    def initialize(
      input_ids:,
      attention_mask:,
      token_type_ids:,
      cls_index:,
      p_mask:,
      example_index:,
      unique_id:,
      paragraph_len:,
      token_is_max_context:,
      tokens:,
      token_to_orig_map:,
      start_position:,
      end_position:,
      is_impossible:,
      qas_id: nil,
      encoding: nil
    )
      @input_ids = input_ids
      @attention_mask = attention_mask
      @token_type_ids = token_type_ids
      @cls_index = cls_index
      @p_mask = p_mask

      @example_index = example_index
      @unique_id = unique_id
      @paragraph_len = paragraph_len
      @token_is_max_context = token_is_max_context
      @tokens = tokens
      @token_to_orig_map = token_to_orig_map

      @start_position = start_position
      @end_position = end_position
      @is_impossible = is_impossible
      @qas_id = qas_id

      @encoding = encoding
    end
  end
end
