# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
# limitations under the License

module Transformers
  module XlmRoberta
    class XLMRobertaTokenizerFast < PreTrainedTokenizerFast
      VOCAB_FILES_NAMES = {vocab_file: "sentencepiece.bpe.model", tokenizer_file: "tokenizer.json"}

      self.vocab_files_names = VOCAB_FILES_NAMES
      self.model_input_names = ["input_ids", "attention_mask"]
      # self.slow_tokenizer_class = XLMRobertaTokenizer

      def initialize(
        vocab_file: nil,
        tokenizer_file: nil,
        bos_token: "<s>",
        eos_token: "</s>",
        sep_token: "</s>",
        cls_token: "<s>",
        unk_token: "<unk>",
        pad_token: "<pad>",
        mask_token: "<mask>",
        **kwargs
      )
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = mask_token.is_a?(String) ? Tokenizers::AddedToken.new(mask_token, lstrip: true, rstrip: false) : mask_token

        super(vocab_file, tokenizer_file: tokenizer_file, bos_token: bos_token, eos_token: eos_token, sep_token: sep_token, cls_token: cls_token, unk_token: unk_token, pad_token: pad_token, mask_token: mask_token, **kwargs)

        @vocab_file = vocab_file
      end

      def can_save_slow_tokenizer
        @vocab_file ? File.exist?(@vocab_file) : false
      end

      def build_inputs_with_special_tokens(token_ids_0, token_ids_1: nil)
        if token_ids_1.nil?
          return [@cls_token_id] + token_ids_0 + [@sep_token_id]
        end
        cls = [@cls_token_id]
        sep = [@sep_token_id]
        cls + token_ids_0 + sep + sep + token_ids_1 + sep
      end

      def create_token_type_ids_from_sequences(token_ids_0, token_ids_1: nil)
        sep = [@sep_token_id]
        cls = [@cls_token_id]

        if token_ids_1.nil?
          return (cls + token_ids_0 + sep).length * [0]
        end
        (cls + token_ids_0 + sep + sep + token_ids_1 + sep).length * [0]
      end
    end
  end
end
