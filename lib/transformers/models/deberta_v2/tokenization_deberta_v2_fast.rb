# Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
  module DebertaV2
    class DebertaV2TokenizerFast < PreTrainedTokenizerFast
      VOCAB_FILES_NAMES = {vocab_file: "spm.model", tokenizer_file: "tokenizer.json"}

      self.vocab_files_names = VOCAB_FILES_NAMES
      # self.slow_tokenizer_class = DebertaV2Tokenizer

      def initialize(
        vocab_file: nil,
        tokenizer_file: nil,
        do_lower_case: false,
        split_by_punct: false,
        bos_token: "[CLS]",
        eos_token: "[SEP]",
        unk_token: "[UNK]",
        sep_token: "[SEP]",
        pad_token: "[PAD]",
        cls_token: "[CLS]",
        mask_token: "[MASK]",
        **kwargs
      )
        super(vocab_file, tokenizer_file: tokenizer_file, do_lower_case: do_lower_case, bos_token: bos_token, eos_token: eos_token, unk_token: unk_token, sep_token: sep_token, pad_token: pad_token, cls_token: cls_token, mask_token: mask_token, split_by_punct: split_by_punct, **kwargs)

        @do_lower_case = do_lower_case
        @split_by_punct = split_by_punct
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
        cls + token_ids_0 + sep + token_ids_1 + sep
      end

      def get_special_tokens_mask(token_ids_0, token_ids_1: nil, already_has_special_tokens: false)
        if already_has_special_tokens
          return super(token_ids_0: token_ids_0, token_ids_1: token_ids_1, already_has_special_tokens: true)
        end

        if !token_ids_1.nil?
          return [1] + ([0] * token_ids_0.length) + [1] + ([0] * token_ids_1.length) + [1]
        end
        [1] + ([0] * token_ids_0.length) + [1]
      end

      def create_token_type_ids_from_sequences(token_ids_0, token_ids_1: nil)
        sep = [@sep_token_id]
        cls = [@cls_token_id]
        if token_ids_1.nil?
          return (cls + token_ids_0 + sep).length * [0]
        end
        ((cls + token_ids_0 + sep).length * [0]) + ((token_ids_1 + sep).length * [1])
      end
    end
  end
end
