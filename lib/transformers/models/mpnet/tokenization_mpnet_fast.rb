# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
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
  module Mpnet
    class MPNetTokenizerFast < PreTrainedTokenizerFast
      VOCAB_FILES_NAMES = {"vocab_file" => "vocab.txt", "tokenizer_file" => "tokenizer.json"}

      self.vocab_files_names = VOCAB_FILES_NAMES
      # self.slow_tokenizer_class = MPNetTokenizer
      self.model_input_names = ["input_ids", "attention_mask"]

      def initialize(
        vocab_file: nil,
        tokenizer_file: nil,
        do_lower_case: true,
        bos_token: "<s>",
        eos_token: "</s>",
        sep_token: "</s>",
        cls_token: "<s>",
        unk_token: "[UNK]",
        pad_token: "<pad>",
        mask_token: "<mask>",
        tokenize_chinese_chars: true,
        strip_accents: nil,
        **kwargs
      )
        bos_token = bos_token.is_a?(String) ? Tokenizers::AddedToken.new(bos_token, lstrip: false, rstrip: false) : bos_token
        eos_token = eos_token.is_a?(String) ? Tokenizers::AddedToken.new(eos_token, lstrip: false, rstrip: false) : eos_token
        sep_token = sep_token.is_a?(String) ? Tokenizers::AddedToken.new(sep_token, lstrip: false, rstrip: false) : sep_token
        cls_token = cls_token.is_a?(String) ? Tokenizers::AddedToken.new(cls_token, lstrip: false, rstrip: false) : cls_token
        unk_token = unk_token.is_a?(String) ? Tokenizers::AddedToken.new(unk_token, lstrip: false, rstrip: false) : unk_token
        pad_token = pad_token.is_a?(String) ? Tokenizers::AddedToken.new(pad_token, lstrip: false, rstrip: false) : pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = mask_token.is_a?(String) ? Tokenizers::AddedToken.new(mask_token, lstrip: true, rstrip: false) : mask_token

        super(vocab_file, tokenizer_file: tokenizer_file, do_lower_case: do_lower_case, bos_token: bos_token, eos_token: eos_token, sep_token: sep_token, cls_token: cls_token, unk_token: unk_token, pad_token: pad_token, mask_token: mask_token, tokenize_chinese_chars: tokenize_chinese_chars, strip_accents: strip_accents, **kwargs)

        # TODO support
        # pre_tok_state = JSON.parse(backend_tokenizer.normalizer.__getstate__)
        # if (pre_tok_state["lowercase"] || do_lower_case) != do_lower_case || (pre_tok_state["strip_accents"] || strip_accents) != strip_accents
        #   pre_tok_class = getattr(normalizers, pre_tok_state.delete("type"))
        #   pre_tok_state["lowercase"] = do_lower_case
        #   pre_tok_state["strip_accents"] = strip_accents
        #   @normalizer = pre_tok_class(**pre_tok_state)
        # end

        @do_lower_case = do_lower_case
      end

      def mask_token
        if @mask_token.nil?
          if @verbose
            Transformers.logger.error("Using mask_token, but it is not set yet.")
          end
          return nil
        end
        @mask_token.to_s
      end

      def mask_token=(value)
        # Mask token behave like a normal word, i.e. include the space before it
        # So we set lstrip to True
        value = value.is_a?(String) ? Tokenizers::AddedToken.new(value, lstrip: true, rstrip: false) : value
        @mask_token = value
      end

      def build_inputs_with_special_tokens(token_ids_0, token_ids_1: nil)
        output = [@bos_token_id] + token_ids_0 + [@eos_token_id]
        if token_ids_1.nil?
          return output
        end

        output + [@eos_token_id] + token_ids_1 + [@eos_token_id]
      end

      def create_token_type_ids_from_sequences(token_ids_0, token_ids_1: nil)
        sep = [@sep_token_id]
        cls = [@cls_token_id]

        if token_ids_1.nil?
          return (cls + token_ids_0 + sep).length * [0]
        end
        (cls + token_ids_0 + sep + sep + token_ids_1 + sep).length * [0]
      end

      def save_vocabulary(save_directory, filename_prefix: nil)
        files = @tokenizer.model.save(save_directory, name: filename_prefix)
        Array(files)
      end
    end
  end
end
