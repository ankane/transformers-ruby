# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
  module Bert
    class BertTokenizer < PreTrainedTokenizer
      class BasicTokenizer
        attr_reader :do_lower_case, :tokenize_chinese_chars, :never_split, :strip_accents, :do_split_on_punc

        def initialize(
          do_lower_case: true,
          never_split: nil,
          tokenize_chinese_chars: true,
          strip_accents: nil,
          do_split_on_punc: true
        )
          if never_split.nil?
            never_split = []
          end
          @do_lower_case = do_lower_case
          @never_split = Set.new(never_split)
          @tokenize_chinese_chars = tokenize_chinese_chars
          @strip_accents = strip_accents
          @do_split_on_punc = do_split_on_punc
        end
      end

      class WordpieceTokenizer
        def initialize(vocab:, unk_token:, max_input_chars_per_word: 100)
          @vocab = vocab
          @unk_token = unk_token
          @max_input_chars_per_word = max_input_chars_per_word
        end
      end

      attr_reader :vocab, :basic_tokenizer

      def initialize(
        vocab_file:,
        do_lower_case: true,
        do_basic_tokenize: true,
        never_split: nil,
        unk_token: "[UNK]",
        sep_token: "[SEP]",
        pad_token: "[PAD]",
        cls_token: "[CLS]",
        mask_token: "[MASK]",
        tokenize_chinese_chars: true,
        strip_accents: nil,
        **kwargs
      )
        if !File.exist?(vocab_file)
          raise ArgumentError,
            "Can't find a vocabulary file at path '#{vocab_file}'. To load the vocabulary from a Google pretrained" +
            " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
        end
        @vocab = load_vocab(vocab_file)
        @ids_to_tokens = @vocab.invert
        @do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize
          @basic_tokenizer =
            BasicTokenizer.new(
              do_lower_case: do_lower_case,
              never_split: never_split,
              tokenize_chinese_chars: tokenize_chinese_chars,
              strip_accents: strip_accents
            )
        end

        @wordpiece_tokenizer = WordpieceTokenizer.new(vocab: @vocab, unk_token: unk_token.to_s)

        super(
          do_lower_case: do_lower_case,
          do_basic_tokenize: do_basic_tokenize,
          never_split: never_split,
          unk_token: unk_token,
          sep_token: sep_token,
          pad_token: pad_token,
          cls_token: cls_token,
          mask_token: mask_token,
          tokenize_chinese_chars: tokenize_chinese_chars,
          strip_accents: strip_accents,
          **kwargs
        )
      end

      def _convert_token_to_id(token)
        @vocab.fetch(token, @vocab.fetch(@unk_token))
      end

      private

      def load_vocab(vocab_file)
        vocab = {}
        tokens = File.readlines(vocab_file)
        tokens.each_with_index do |token, index|
          token = token.chomp
          vocab[token] = index
        end
        vocab
      end
    end
  end
end
