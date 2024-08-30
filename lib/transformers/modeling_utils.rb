# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
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
  module ModuleUtilsMixin
    def get_extended_attention_mask(
      attention_mask,
      input_shape,
      device: nil,
      dtype: nil
    )
      if dtype.nil?
        dtype = @dtype
      end

      if !(attention_mask.dim == 2 && @config.is_decoder)
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if !device.nil?
          raise Todo
        end
      end
      # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
      # ourselves in which case we just need to make it broadcastable to all heads.
      if attention_mask.dim == 3
        raise Todo
      elsif attention_mask.dim == 2
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if @config.is_decoder
          raise Todo
        else
          extended_attention_mask = attention_mask[0.., nil, nil, 0..]
        end
      else
        raise Todo
      end

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and the dtype's smallest value for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      extended_attention_mask = extended_attention_mask.to(dtype: dtype)  # fp16 compatibility
      # TODO use Torch.finfo
      extended_attention_mask = (1.0 - extended_attention_mask) * -3.40282e+38
      extended_attention_mask
    end

    def get_head_mask(head_mask, num_hidden_layers, is_attention_chunked: false)
      if !head_mask.nil?
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked == true
          head_mask = head_mask.unsqueeze(-1)
        end
      else
        head_mask = [nil] * num_hidden_layers
      end

      head_mask
    end
  end

  class PreTrainedModel < Torch::NN::Module
    extend ClassAttribute
    include ModuleUtilsMixin

    class_attribute :config_class
    class_attribute :base_model_prefix, ""
    class_attribute :main_input_name, "input_ids"
    class_attribute :model_tags

    class_attribute :_tied_weights_keys

    attr_reader :config

    def dummy_inputs
      raise Todo
    end

    def framework
      "pt"
    end

    def initialize(config, *inputs, **kwargs)
      super()
      @config = config
    end

    def post_init
      init_weights
      _backward_compatibility_gradient_checkpointing
    end

    def dequantize
      raise Todo
    end

    def _backward_compatibility_gradient_checkpointing
      # TODO
    end

    def base_model
      instance_variable_get("@#{self.class.base_model_prefix}") || self
    end

    def can_generate
      # TODO improve
      false
    end

    def get_input_embeddings
      raise Todo
    end

    def set_input_embeddings(value)
      raise Todo
    end

    def get_output_embeddings
      nil  # Overwrite for models with output embeddings
    end

    def _init_weights(mod)
      # pass
    end

    def _initialize_weights(mod)
      _init_weights(mod)
    end

    def tie_weights
      if @config.tie_word_embeddings != false
        output_embeddings = get_output_embeddings
        if !output_embeddings.nil?
          raise Todo
        end
      end

      if @config.is_encoder_decoder && @config.tie_encoder_decoder
        raise Todo
      end

      modules.each do |mod|
        if mod.respond_to?(:_tie_weights)
          mod._tie_weights
        end
      end
    end

    def init_weights
      # Prune heads if needed
      if @config.pruned_heads
        prune_heads(@config.pruned_heads)
      end

      if true
        # Initialize weights
        apply(method(:_initialize_weights))

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        tie_weights
      end
    end

    def prune_heads(heads_to_prune)
      # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
      heads_to_prune.each do |layer, heads|
        union_heads = Set.new(@config.pruned_heads.fetch(layer, [])) | Set.new(heads)
        @config.pruned_heads[layer] = union_heads.to_a # Unfortunately we have to store it as list for JSON
      end

      base_model._prune_heads(heads_to_prune)
    end

    def warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
      if !attention_mask.nil? || @config.pad_token_id.nil?
        return
      end

      # Check only the first and last input IDs to reduce overhead.
      if input_ids[0.., [-1, 0]].include?(@config.pad_token_id)
        raise Todo
      end
    end

    class << self
      def from_pretrained(
        pretrained_model_name_or_path,
        *model_args,
        config: nil,
        cache_dir: nil,
        ignore_mismatched_sizes: false,
        force_download: false,
        local_files_only: false,
        token: nil,
        revision: "main",
        use_safetensors: nil,
        **kwargs
      )
        state_dict = kwargs.delete(:state_dict)
        from_tf = kwargs.delete(:from_tf) { false }
        from_flax = kwargs.delete(:from_flax) { false }
        resume_download = kwargs.delete(:resume_download) { false }
        proxies = kwargs.delete(:proxies)
        output_loading_info = kwargs.delete(:output_loading_info) { false }
        _use_auth_token = kwargs.delete(:use_auth_token)
        trust_remote_code = kwargs.delete(:trust_remote_code)
        _ = kwargs.delete(:mirror)
        from_pipeline = kwargs.delete(:_from_pipeline)
        from_auto_class = kwargs.delete(:_from_auto) { false }
        _fast_init = kwargs.delete(:_fast_init) { true }
        torch_dtype = kwargs.delete(:torch_dtype)
        low_cpu_mem_usage = kwargs.delete(:low_cpu_mem_usage)
        device_map = kwargs.delete(:device_map)
        _max_memory = kwargs.delete(:max_memory)
        offload_folder = kwargs.delete(:offload_folder)
        offload_state_dict = kwargs.delete(:offload_state_dict) { false }
        load_in_8bit = kwargs.delete(:load_in_8bit) { false }
        load_in_4bit = kwargs.delete(:load_in_4bit) { false }
        quantization_config = kwargs.delete(:quantization_config)
        subfolder = kwargs.delete(:subfolder) { "" }
        commit_hash = kwargs.delete(:_commit_hash)
        variant = kwargs.delete(:variant)
        _adapter_kwargs = kwargs.delete(:adapter_kwargs) { {} }
        _adapter_name = kwargs.delete(:adapter_name) { "default" }
        _use_flash_attention_2 = kwargs.delete(:use_flash_attention_2) { false }

        if use_safetensors.nil? && !is_safetensors_available
          use_safetensors = false
        end
        if trust_remote_code
          Transformers.logger.warn(
            "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is" +
            " ignored."
          )
        end

        if commit_hash.nil?
          if !config.is_a?(PretrainedConfig)
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file =
              Utils::Hub.cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                cache_dir: cache_dir,
                force_download: force_download,
                resume_download: resume_download,
                proxies: proxies,
                local_files_only: local_files_only,
                token: token,
                revision: revision,
                subfolder: subfolder,
                _raise_exceptions_for_gated_repo: false,
                _raise_exceptions_for_missing_entries: false,
                _raise_exceptions_for_connection_errors: false,
              )
            commit_hash = Utils::Hub.extract_commit_hash(resolved_config_file, commit_hash)
          else
            commit_hash = config._commit_hash
          end
        end

        if !device_map.nil?
          raise Todo
        end

        # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
        if load_in_4bit || load_in_8bit
          raise Todo
        end

        from_pt = !(from_tf || from_flax)

        user_agent = {file_type: "model", framework: "pytorch", from_auto_class: from_auto_class}
        if !from_pipeline.nil?
          user_agent[:using_pipeline] = from_pipeline
        end

        if Utils::Hub.is_offline_mode && !local_files_only
          Transformers.logger.info "Offline mode: forcing local_files_only: true"
          local_files_only = true
        end

        # Load config if we don't provide a configuration
        if !config.is_a?(PretrainedConfig)
          config_path = !config.nil? ? config : pretrained_model_name_or_path
          config, model_kwargs =
            config_class.from_pretrained(
              config_path,
              cache_dir: cache_dir,
              return_unused_kwargs: true,
              force_download: force_download,
              resume_download: resume_download,
              proxies: proxies,
              local_files_only: local_files_only,
              token: token,
              revision: revision,
              subfolder: subfolder,
              _from_auto: from_auto_class,
              _from_pipeline: from_pipeline,
              **kwargs
            )
        else
          # In case one passes a config to `from_pretrained` + "attn_implementation"
          # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
          # Please see: https://github.com/huggingface/transformers/issues/28038

          # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
          # we pop attn_implementation from the kwargs but this handles the case where users
          # passes manually the config to `from_pretrained`.
          config = Copy.deepcopy(config)

          kwarg_attn_imp = kwargs.delete(:attn_implementation)
          if !kwarg_attn_imp.nil? && config._attn_implementation != kwarg_attn_imp
            config._attn_implementation = kwarg_attn_imp
          end
          model_kwargs = kwargs
        end

        pre_quantized = false # !config.quantization_config.nil?
        if pre_quantized || !quantization_config.nil?
          raise Todo
        else
          hf_quantizer = nil
        end

        if !hf_quantizer.nil?
          raise Todo
        end

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = false
        sharded_metadata = nil
        # Load model
        _loading_info = nil

        # Keep in fp32 modules
        keep_in_fp32_modules = nil
        _use_keep_in_fp32_modules = false

        resolved_archive_file = nil
        if !pretrained_model_name_or_path.nil?
          pretrained_model_name_or_path = pretrained_model_name_or_path.to_s
          is_local = Dir.exist?(pretrained_model_name_or_path)
          if is_local
            raise Todo
          elsif File.exist?(File.join(subfolder, pretrained_model_name_or_path))
            _archive_file = pretrained_model_name_or_path
            is_local = true
          else
            # set correct filename
            if use_safetensors != false
              filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else
              filename = _add_variant(WEIGHTS_NAME, variant)
            end

            # Load from URL or cache if already cached
            cached_file_kwargs = {
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
              _commit_hash: commit_hash
            }
            resolved_archive_file = Utils::Hub.cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

            # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
            # result when internet is up, the repo and revision exist, but the file does not.
            if resolved_archive_file.nil? && filename == _add_variant(SAFE_WEIGHTS_NAME, variant)
              # Maybe the checkpoint is sharded, we try to grab the index name in this case.
              resolved_archive_file = Utils::Hub.cached_file(
                pretrained_model_name_or_path,
                _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                **cached_file_kwargs,
              )
              if !resolved_archive_file.nil?
                is_sharded = true
              elsif use_safetensors
                raise Todo
              else
                # This repo has no safetensors file of any kind, we switch to PyTorch.
                filename = _add_variant(WEIGHTS_NAME, variant)
                resolved_archive_file = Utils::Hub.cached_file(
                  pretrained_model_name_or_path, filename, **cached_file_kwargs
                )
              end
            end
            if resolved_archive_file.nil? && filename == _add_variant(WEIGHTS_NAME, variant)
              # Maybe the checkpoint is sharded, we try to grab the index name in this case.
              resolved_archive_file = Utils::Hub.cached_file(
                pretrained_model_name_or_path,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
                **cached_file_kwargs
              )
              if !resolved_archive_file.nil?
                is_sharded = true
              end
            end
            if !local_files_only && !Utils::Hub.is_offline_mode
              if !resolved_archive_file.nil?
                if [WEIGHTS_NAME, WEIGHTS_INDEX_NAME].include?(filename)
                  # If the PyTorch file was found, check if there is a safetensors file on the repository
                  # If there is no safetensors file on the repositories, start an auto conversion
                  _safe_weights_name = is_sharded ? SAFE_WEIGHTS_INDEX_NAME : SAFE_WEIGHTS_NAME
                  has_file_kwargs = {
                    revision: revision,
                    proxies: proxies,
                    token: token,
                    cache_dir: cache_dir,
                    local_files_only: local_files_only
                  }
                  cached_file_kwargs = {
                    cache_dir: cache_dir,
                    force_download: force_download,
                    resume_download: resume_download,
                    local_files_only: local_files_only,
                    user_agent: user_agent,
                    subfolder: subfolder,
                    _raise_exceptions_for_gated_repo: false,
                    _raise_exceptions_for_missing_entries: false,
                    _commit_hash: commit_hash,
                    **has_file_kwargs
                  }
                  # skip auto conversion
                  # if !Utils::Hub.has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs)
                  # end
                end
              else
                raise Todo
              end
            end

            if is_local
              Transformers.logger.info("loading weights file #{archive_file}")
              resolved_archive_file = archive_file
            else
              Transformers.logger.info("loading weights file #{filename} from cache at #{resolved_archive_file}")
            end
          end
        else
          resolved_archive_file = nil
        end

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded
          raise Todo
        end

        metadata = nil
        if is_safetensors_available && resolved_archive_file.is_a?(String) && resolved_archive_file.end_with?(".safetensors")
          Safetensors.safe_open(resolved_archive_file, framework: "pt") do |f|
            metadata = f.metadata
          end

          if metadata["format"] == "pt"
            # do nothing
          else
            raise ArgumentError,
              "Incompatible safetensors file. File metadata is not ['pt'] but #{metadata["format"]}"
          end
        end

        from_pt = !(from_tf || from_flax)

        # load pt weights early so that we know which dtype to init the model under
        if from_pt
          if !is_sharded && state_dict.nil?
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_archive_file)
          end

          # set dtype to instantiate the model under:
          # 1. If torch_dtype is not None, we use that dtype
          # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
          #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
          # we also may have config.torch_dtype available, but we won't rely on it till v5
          dtype_orig = nil

          if !torch_dtype.nil?
            raise Todo
          end

          if is_sharded
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
          else
            loaded_state_dict_keys = state_dict.keys
          end
        end

        config.name_or_path = pretrained_model_name_or_path

        model_kwargs = {}
        model = new(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        if device_map.is_a?(String)
          raise Todo
        elsif !device_map.nil?
          raise Todo
        end

        if from_pt
          # restore default dtype
          if !dtype_orig.nil?
            Torch.set_default_dtype(dtype_orig)
          end

          model, _missing_keys, _unexpected_keys, _mismatched_keys, _offload_index, _error_msgs =
            _load_pretrained_model(
              model,
              state_dict,
              loaded_state_dict_keys,  # XXX: rename?
              resolved_archive_file,
              pretrained_model_name_or_path,
              ignore_mismatched_sizes: ignore_mismatched_sizes,
              sharded_metadata: sharded_metadata,
              _fast_init: _fast_init,
              low_cpu_mem_usage: low_cpu_mem_usage,
              device_map: device_map,
              offload_folder: offload_folder,
              offload_state_dict: offload_state_dict,
              dtype: torch_dtype,
              hf_quantizer: hf_quantizer,
              keep_in_fp32_modules: keep_in_fp32_modules
            )
        end

        # make sure token embedding weights are still tied if needed
        model.tie_weights

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate && !pretrained_model_name_or_path.nil?
          raise Todo
        end

        # Dispatch model with hooks on all devices if necessary
        if !device_map.nil?
          raise Todo
        end

        if !hf_quantizer.nil?
          raise Todo
        end

        if output_loading_info
          raise Todo
        end

        model
      end

      private

      def _load_pretrained_model(
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes: false,
        sharded_metadata: nil,
        _fast_init: true,
        low_cpu_mem_usage: false,
        device_map: nil,
        offload_folder: nil,
        offload_state_dict: nil,
        dtype: nil,
        hf_quantizer: nil,
        keep_in_fp32_modules: nil
      )
        is_safetensors = false

        _is_sharded_safetensors = is_safetensors && !sharded_metadata.nil?

        # tie the model weights before retrieving the state_dict
        model.tie_weights

        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict
        expected_keys = model_state_dict.keys
        prefix = model.class.base_model_prefix

        _fix_key = lambda do |key|
          if key.include?("beta")
            key.gsub("beta", "bias")
          end
          if key.include?("gamma")
            key.gsub("gamma", "weight")
          else
            key
          end
        end

        original_loaded_keys = loaded_keys
        loaded_keys = loaded_keys.map { |key| _fix_key.(key) }

        if prefix.length > 0
          has_prefix_module = loaded_keys.any? { |s| s.start_with?(prefix) }
          expects_prefix_module = expected_keys.any? { |s| s.start_with?(prefix) }
        else
          has_prefix_module = false
          expects_prefix_module = false
        end

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = !has_prefix_module && expects_prefix_module
        add_prefix_to_model = has_prefix_module && !expects_prefix_module

        if remove_prefix_from_model
          _prefix = "#{prefix}."
          expected_keys_not_prefixed = expected_keys.select { |s| !s.start_with?(_prefix) }
          expected_keys = expected_keys.map { |s| s.start_with?(_prefix) ? s[_prefix.length..] : s }
        elsif add_prefix_to_model
          expected_keys = expected_keys.map { |s| [prefix, s].join(".") }
        end

        missing_keys = (Set.new(expected_keys) - Set.new(loaded_keys)).sort
        unexpected_keys = Set.new(loaded_keys) - Set.new(expected_keys)
        # Remove nonpersistent buffers from unexpected keys: they are not in the state dict but will be in the model
        # buffers
        model_buffers = model.named_buffers(recurse: true).keys
        if remove_prefix_from_model
          raise Todo
        elsif add_prefix_to_model
          model_buffers = model_buffers.map { |key| [prefix, key].join(".") }
        end
        unexpected_keys = (unexpected_keys - model_buffers).sort

        model.tie_weights
        if device_map.nil?
          ptrs = Hash.new { |hash, key| hash[key] = [] }

          model.state_dict.each do |name, tensor|
            # TODO fix
            id_tensor = tensor.object_id # id_tensor_storage(tensor)
            ptrs[id_tensor] << name
          end

          # These are all the pointers of shared tensors.
          tied_params = ptrs.select { |_, names| names.length > 1 }.values
        else
          raise Todo
        end

        tied_params.each do |group|
          if remove_prefix_from_model
            group = group.map { |key| key.delete_prefix(_prefix) }
          elsif add_prefix_to_model
            group = group.map { |key| [prefix, key].join(".") }
          end
          missing_in_group = missing_keys.select { |k| group.include?(k) }
          if missing_in_group.length > 0 && missing_in_group.length < group.length
            missing_keys = missing_keys.select { |k| !missing_in_group.include?(k) }
          end
        end

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if base_model_prefix.length > 0 && !model.instance_variable_defined?("@#{base_model_prefix}") && has_prefix_module
          start_prefix = base_model_prefix + "."
        end
        if base_model_prefix.length > 0 && model.instance_variable_defined?("@#{base_model_prefix}") && !has_prefix_module
          model_to_load = model.instance_variable_get("@#{base_model_prefix}")
          base_model_expected_keys = model_to_load.state_dict.keys
          if loaded_keys.any? { |key| expected_keys_not_prefixed.include?(key) && !base_model_expected_keys.include?(key) }
            raise ArgumentError, "The state dictionary of the model you are trying to load is corrupted. Are you sure it was properly saved?"
          end
          if !device_map.nil?
            raise Todo
          end
        end

        _find_mismatched_keys = lambda do |state_dict, model_state_dict, loaded_keys, add_prefix_to_model, remove_prefix_from_model, ignore_mismatched_sizes|
          mismatched_keys = []
          if ignore_mismatched_sizes
            loaded_keys.each do |checkpoint_key|
              # If the checkpoint is sharded, we may not have the key here.
              if !state_dict.include?(checkpoint_key)
                next
              end
              model_key = checkpoint_key
              if remove_prefix_from_model
                # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                model_key = "#{prefix}.#{checkpoint_key}"
              elsif add_prefix_to_model
                # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                model_key = checkpoint_key.split(".")[1..].join(".")
              end

              if model_state_dict.include?(model_key) && state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                raise Todo
              end
            end
          end
          mismatched_keys
        end

        if !resolved_archive_file.nil?
          _folder = File.dirname(resolved_archive_file)
        else
          _folder = nil
        end

        if !device_map.nil? && is_safetensors
          raise Todo
        end

        if !state_dict.nil?
          # Whole checkpoint
          mismatched_keys =
            _find_mismatched_keys.(
              state_dict,
              model_state_dict,
              original_loaded_keys,
              add_prefix_to_model,
              remove_prefix_from_model,
              ignore_mismatched_sizes
            )
          error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
          offload_index = nil
        else
          raise Todo
        end

        if error_msgs.length > 0
          raise Todo
        end

        if unexpected_keys.length > 0
          archs = model.config.architectures.nil? ? [] : model.config.architectures
          warner = archs.include?(model.class.name.split("::").last) ? Transformers.logger.method(:warn) : Transformers.logger.method(:info)
          warner.(
            "Some weights of the model checkpoint at #{pretrained_model_name_or_path} were not used when" +
            " initializing #{model.class.name}: #{unexpected_keys}\n- This IS expected if you are" +
            " initializing #{model.class.name} from the checkpoint of a model trained on another task or" +
            " with another architecture (e.g. initializing a BertForSequenceClassification model from a" +
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing" +
            " #{model.class.name} from the checkpoint of a model that you expect to be exactly identical" +
            " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
          )
        else
          Transformers.logger.info("All model checkpoint weights were used when initializing #{model.class.name}.\n")
        end
        if missing_keys.length > 0
          Transformers.logger.info("Some weights of #{model.class.name} were not initialized from the model checkpoint at #{pretrained_model_name_or_path} and are newly initialized: #{missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")
        elsif mismatched_keys.length == 0
          Transformers.logger.info(
            "All the weights of #{model.class.name} were initialized from the model checkpoint at" +
            " #{pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint" +
            " was trained on, you can already use #{model.class.name} for predictions without further" +
            " training."
          )
        end
        if mismatched_keys.length > 0
          raise Todo
        end

        [model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs]
      end

      def _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        state_dict.each_key do |key|
          new_key = nil
          if key.include?("gamma")
            new_key = key.gsub("gamma", "weight")
          end
          if key.include?("beta")
            new_key = key.gsub("beta", "bias")
          end
          if new_key
            old_keys << key
            new_keys << new_key
          end
        end
        old_keys.zip(new_keys) do |old_key, new_key|
          state_dict[new_key] = state_dict.delete(old_key)
        end

        # copy state_dict so _load_from_state_dict can modify it
        metadata = nil #getattr(state_dict, "_metadata", None)
        state_dict = state_dict.dup
        if !metadata.nil?
          state_dict._metadata = metadata
        end

        error_msgs = []

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        load = lambda do |mod, state_dict, prefix|
          local_metadata = metadata.nil? ? {} : metadata.fetch(prefix[...-1], {})
          args = [state_dict, prefix, local_metadata, true, [], [], error_msgs]
          # Parameters of module and children will start with prefix. We can exit early if there are none in this
          # state_dict
          if state_dict.any? { |key, _| key.start_with?(prefix) }
            mod.send(:load_from_state_dict, *args)
          end

          mod.named_children.each do |name, child|
            if !child.nil?
              load.(child, state_dict, prefix + name + ".")
            end
          end
        end

        load.(model_to_load, state_dict, start_prefix)

        error_msgs
      end

      def is_safetensors_available
        defined?(Safetensors)
      end

      def load_state_dict(checkpoint_file)
        if checkpoint_file.end_with?(".safetensors") && is_safetensors_available
          # Check format of the archive
          metadata = nil
          Safetensors.safe_open(checkpoint_file, framework: "pt") do |f|
            metadata = f.metadata
          end
          if !["pt", "tf", "flax"].include?(metadata["format"])
            raise OSError, "The safetensors archive passed at #{checkpoint_file} does not contain the valid metadata. Make sure you save your model with the `save_pretrained` method."
          end
          return Safetensors::Torch.load_file(checkpoint_file)
        end
        begin
          _map_location = "cpu"
          _extra_args = {}
          _weights_only_kwarg = {weights_only: true}
          Torch.load(
            checkpoint_file,
            # Torch.rb does not currently support additional options
            # map_location: map_location,
            # **weights_only_kwarg,
            # **extra_args
          )
        rescue => e
          # TODO improve
          raise e
        end
      end

      def _add_variant(weights_name, variant)
        if !variant.nil?
          splits = weights_name.split(".")
          splits = splits[...-1] + [variant] + splits[-1..]
          weights_name = splits.join(".")
        end

        weights_name
      end
    end
  end
end
