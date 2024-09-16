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
  CONFIG_MAPPING_NAMES = {
    "bert" => "BertConfig",
    "deberta-v2" => "DebertaV2Config",
    "distilbert" => "DistilBertConfig",
    "mpnet" => "MPNetConfig",
    "vit" => "ViTConfig",
    "xlm-roberta" => "XLMRobertaConfig"
  }

  class LazyConfigMapping
    def initialize(mapping)
      @mapping = mapping
      @extra_content = {}
      @modules = {}
    end

    def [](key)
      value = @mapping.fetch(key)
      module_name = model_type_to_module_name(key)
      if !@modules.include?(module_name)
        @modules[module_name] = Transformers.const_get(module_name.split("-").map(&:capitalize).join)
      end
      @modules[module_name].const_get(value)
    end

    def model_type_to_module_name(key)
      key
    end
  end

  CONFIG_MAPPING = LazyConfigMapping.new(CONFIG_MAPPING_NAMES)

  class AutoConfig
    def self.from_pretrained(pretrained_model_name_or_path, **kwargs)
      kwargs[:_from_auto] = true
      kwargs[:name_or_path] = pretrained_model_name_or_path
      _trust_remote_code = kwargs.delete(:trust_remote_code)
      _code_revision = kwargs.delete(:code_revision)

      config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
      if config_dict[:model_type]
        config_class = CONFIG_MAPPING[config_dict[:model_type]]
        return config_class.from_dict(config_dict, **unused_kwargs)
      else
        raise Todo
      end
    end
  end
end
