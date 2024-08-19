# Copyright 2021 The HuggingFace Inc. team.
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
  class BaseAutoModelClass
    extend ClassAttribute

    class_attribute :_model_mapping

    class << self
      def from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        config = kwargs.delete(:config)
        trust_remote_code = kwargs.delete(:trust_remote_code)
        hub_kwargs_names = [
          :cache_dir,
          :force_download,
          :local_files_only,
          :proxies,
          :resume_download,
          :revision,
          :subfolder,
          :use_auth_token,
          :token
        ]
        hub_kwargs = hub_kwargs_names.select { |k| kwargs.key?(k) }.to_h { |name| [name, kwargs.delete(name)] }
        code_revision = kwargs.delete(:code_revision)
        commit_hash = kwargs.delete(:_commit_hash)

        if commit_hash.nil?
          if !config.is_a?(PretrainedConfig)
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file = Utils::Hub.cached_file(
              pretrained_model_name_or_path,
              CONFIG_NAME,
              _raise_exceptions_for_gated_repo: false,
              _raise_exceptions_for_missing_entries: false,
              _raise_exceptions_for_connection_errors: false,
              **hub_kwargs
            )
            commit_hash = Utils::Hub.extract_commit_hash(resolved_config_file, commit_hash)
          else
            raise Todo
          end
        end

        if !config.is_a?(PretrainedConfig)
          config, kwargs =
            AutoConfig.from_pretrained(
              pretrained_model_name_or_path,
              return_unused_kwargs: true,
              trust_remote_code: trust_remote_code,
              code_revision: code_revision,
              _commit_hash: commit_hash,
              **hub_kwargs,
              **kwargs
            )
        end

        model_class = _get_model_class(config, _model_mapping)
        model_class.from_pretrained(
          pretrained_model_name_or_path, *model_args, config: config, **hub_kwargs, **kwargs
        )
      end

      private

      def _get_model_class(config, model_mapping)
        supported_models = model_mapping[config.class.name.split("::").last]
        if !supported_models.is_a?(Array)
          return supported_models
        end

        raise Todo
      end
    end
  end

  class LazyAutoMapping
    def initialize(config_mapping, model_mapping)
      @config_mapping = config_mapping
      @reverse_config_mapping = config_mapping.invert
      @model_mapping = model_mapping
      @modules = {}
    end

    def [](key)
      model_type = @reverse_config_mapping[key]
      if @model_mapping[model_type]
        model_name = @model_mapping[model_type]
        return _load_attr_from_module(model_type, model_name)
      end

      raise KeyError, key
    end

    def include?(key)
      self[key]
      true
    rescue KeyError
      false
    end

    private

    def _load_attr_from_module(model_type, attr)
      module_name = model_type_to_module_name(model_type)
      if !@modules.include?(module_name)
        @modules[module_name] = Transformers.const_get(module_name.capitalize)
      end
      getattribute_from_module(@modules[module_name], attr)
    end

    def getattribute_from_module(mod, attr)
      if attr.nil?
        nil
      elsif attr.is_a?(Array)
        attr.map { |a| mod.const_get(a) }
      else
        mod.const_get(attr)
      end
    end

    def model_type_to_module_name(key)
      key
    end
  end
end
