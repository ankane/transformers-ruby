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
  module ConvertSlowTokenizer
    class Converter
      def initialize(original_tokenizer)
        @original_tokenizer = original_tokenizer
      end

      def converted
        raise NotImplementedError
      end
    end

    class BertConverter < Converter
      def converted
        vocab = @original_tokenizer.vocab
        tokenizer = Tokenizers::Tokenizer.new(Tokenizers::Models::WordPiece.new(vocab: vocab, unk_token: @original_tokenizer.unk_token.to_s))

        tokenize_chinese_chars = false
        strip_accents = false
        do_lower_case = false
        if @original_tokenizer.basic_tokenizer
          tokenize_chinese_chars = @original_tokenizer.basic_tokenizer.tokenize_chinese_chars
          strip_accents = @original_tokenizer.basic_tokenizer.strip_accents
          do_lower_case = @original_tokenizer.basic_tokenizer.do_lower_case
        end

        tokenizer.normalizer =
          Tokenizers::Normalizers::BertNormalizer.new(
            clean_text: true,
            handle_chinese_chars: tokenize_chinese_chars,
            strip_accents: strip_accents,
            lowercase: do_lower_case,
          )
        tokenizer.pre_tokenizer = Tokenizers::PreTokenizers::BertPreTokenizer.new

        cls = @original_tokenizer.cls_token.to_s
        sep = @original_tokenizer.sep_token.to_s
        cls_token_id = @original_tokenizer.cls_token_id
        sep_token_id = @original_tokenizer.sep_token_id

        tokenizer.post_processor =
          Tokenizers::Processors::TemplateProcessing.new(
            single: "#{cls}:0 $A:0 #{sep}:0",
            pair: "#{cls}:0 $A:0 #{sep}:0 $B:1 #{sep}:1",
            special_tokens: [
              [cls, cls_token_id],
              [sep, sep_token_id]
            ]
          )
        tokenizer.decoder = Tokenizers::Decoders::WordPiece.new(prefix: "##")

        tokenizer
      end
    end

    SLOW_TO_FAST_CONVERTERS = {
      "BertTokenizer" => BertConverter,
      "DistilBertTokenizer" => BertConverter
    }

    def self.convert_slow_tokenizer(transformer_tokenizer)
      tokenizer_class_name = transformer_tokenizer.class.name.split("::").last

      if !SLOW_TO_FAST_CONVERTERS.include?(tokenizer_class_name)
        raise ArgumentError,
          "An instance of tokenizer class #{tokenizer_class_name} cannot be converted in a Fast tokenizer instance." +
          " No converter was found. Currently available slow->fast convertors:" +
          " #{SLOW_TO_FAST_CONVERTERS.keys}"
      end

      converter_class = SLOW_TO_FAST_CONVERTERS.fetch(tokenizer_class_name)

      converter_class.new(transformer_tokenizer).converted
    end
  end
end
