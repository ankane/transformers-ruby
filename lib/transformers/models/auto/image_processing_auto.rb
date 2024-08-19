# Copyright 2022 The HuggingFace Inc. team.
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
  IMAGE_PROCESSOR_MAPPING_NAMES = {
    "vit" => ["ViTImageProcessor"]
  }

  IMAGE_PROCESSOR_MAPPING = LazyAutoMapping.new(CONFIG_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING_NAMES)

  class AutoImageProcessor
    def self.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
      config = kwargs.delete(:config)
      use_fast = kwargs.delete(:use_fast)
      trust_remote_code = kwargs.delete(:trust_remote_code)
      kwargs[:_from_auto] = true

      config_dict, _ = ImageProcessingMixin.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
      image_processor_class = config_dict[:image_processor_type]
      image_processor_auto_map = nil
      if (config_dict[:auto_map] || {}).include?("AutoImageProcessor")
        image_processor_auto_map = config_dict[:auto_map]["AutoImageProcessor"]
      end

      # If we still don't have the image processor class, check if we're loading from a previous feature extractor config
      # and if so, infer the image processor class from there.
      if image_processor_class.nil? && image_processor_auto_map.nil?
        feature_extractor_class = config_dict.delete(:feature_extractor_type)
        if !feature_extractor_class.nil?
          image_processor_class = feature_extractor_class.sub("FeatureExtractor", "ImageProcessor")
        end
        if (config_dict[:auto_map] || {}).include?("AutoFeatureExtractor")
          feature_extractor_auto_map = config_dict[:auto_map]["AutoFeatureExtractor"]
          image_processor_auto_map = feature_extractor_auto_map.sub("FeatureExtractor", "ImageProcessor")
        end
      end

      # If we don't find the image processor class in the image processor config, let's try the model config.
      if image_processor_class.nil? && image_processor_auto_map.nil?
        if !config.is_a?(PretrainedConfig)
          config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        end
        # It could be in `config.image_processor_type``
        image_processor_class = config.instance_variable_get(:@image_processor_type)
      end

      if !image_processor_class.nil?
        raise Todo
      end

      has_remote_code = !image_processor_auto_map.nil?
      has_local_code = !image_processor_class.nil? || IMAGE_PROCESSOR_MAPPING.include?(config.class.name.split("::").last)
      trust_remote_code = DynamicModuleUtils.resolve_trust_remote_code(
        trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
      )

      if !image_processor_auto_map.nil? && !image_processor_auto_map.is_a?(Array)
        raise Todo
      end

      if has_remote_code && trust_remote_code
        raise Todo
      elsif !image_processor_class.nil?
        return image_processor_class.from_dict(config_dict, **kwargs)
      # Last try: we use the IMAGE_PROCESSOR_MAPPING.
      elsif IMAGE_PROCESSOR_MAPPING.include?(config.class.name.split("::").last)
        image_processor_tuple = IMAGE_PROCESSOR_MAPPING[config.class.name.split("::").last]

        image_processor_class_py, image_processor_class_fast = image_processor_tuple

        if !use_fast && !image_processor_class_fast.nil?
          _warning_fast_image_processor_available(image_processor_class_fast)
        end

        if image_processor_class_fast && (use_fast || image_processor_class_py.nil?)
          return image_processor_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        else
          if !image_processor_class_py.nil?
            return image_processor_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
          else
            raise ArgumentError,
              "This image processor cannot be instantiated. Please make sure you have `Pillow` installed."
          end
        end
      end

      raise ArgumentError,
        "Unrecognized image processor in #{pretrained_model_name_or_path}. Should have a " +
        "`image_processor_type` key in its #{IMAGE_PROCESSOR_NAME} of #{CONFIG_NAME}, or one of the following " +
        "`model_type` keys in its #{CONFIG_NAME}: #{IMAGE_PROCESSOR_MAPPING_NAMES.keys.join(", ")}"
    end
  end
end
