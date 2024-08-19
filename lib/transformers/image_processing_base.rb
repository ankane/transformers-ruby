# Copyright 2020 The HuggingFace Inc. team.
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
  class ImageProcessingMixin
    def self.from_pretrained(
      pretrained_model_name_or_path,
      cache_dir: nil,
      force_download: false,
      local_files_only: false,
      token: nil,
      revision: "main",
      **kwargs
    )
      kwargs[:cache_dir] = cache_dir
      kwargs[:force_download] = force_download
      kwargs[:local_files_only] = local_files_only
      kwargs[:revision] = revision

      if !token.nil?
        kwargs[:token] = token
      end

      image_processor_dict, kwargs = get_image_processor_dict(pretrained_model_name_or_path, **kwargs)

      from_dict(image_processor_dict, **kwargs)
    end

    def self.get_image_processor_dict(
      pretrained_model_name_or_path, **kwargs
    )
      cache_dir = kwargs.delete(:cache_dir)
      force_download = kwargs.delete(:force_download) { false }
      resume_download = kwargs.delete(:resume_download)
      proxies = kwargs.delete(:proxies)
      token = kwargs.delete(:token)
      _use_auth_token = kwargs.delete(:use_auth_token)
      local_files_only = kwargs.delete(:local_files_only) { false }
      revision = kwargs.delete(:revision)
      subfolder = kwargs.delete(:subfolder) { "" }

      from_pipeline = kwargs.delete(:_from_pipeline)
      from_auto_class = kwargs.delete(:_from_auto) { false }

      user_agent = {file_type: "image processor", from_auto_class: from_auto_class}
      if !from_pipeline.nil?
        user_agent[:using_pipeline] = from_pipeline
      end

      if Utils::Hub.is_offline_mode && !local_files_only
        Transformers.logger.info("Offline mode: forcing local_files_only: true")
        local_files_only = true
      end

      pretrained_model_name_or_path = pretrained_model_name_or_path.to_s
      is_local = Dir.exist?(pretrained_model_name_or_path)
      if Dir.exist?(pretrained_model_name_or_path)
        image_processor_file = File.join(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME)
      end
      if File.exist?(pretrained_model_name_or_path)
        resolved_image_processor_file = pretrained_model_name_or_path
        is_local = true
      elsif Utils::Hub.is_remote_url(pretrained_model_name_or_path)
        raise Todo
      else
        image_processor_file = IMAGE_PROCESSOR_NAME
        begin
          # Load from local folder or from cache or download from model Hub and cache
          resolved_image_processor_file = Utils::Hub.cached_file(
            pretrained_model_name_or_path,
            image_processor_file,
            cache_dir: cache_dir,
            force_download: force_download,
            proxies: proxies,
            resume_download: resume_download,
            local_files_only: local_files_only,
            token: token,
            user_agent: user_agent,
            revision: revision,
            subfolder: subfolder
          )
        rescue EnvironmentError
          # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
          # the original exception.
          raise
        rescue
          # For any other exception, we throw a generic error.
          raise EnvironmentError,
            "Can't load image processor for '#{pretrained_model_name_or_path}'. If you were trying to load" +
            " it from 'https://huggingface.co/models', make sure you don't have a local directory with the" +
            " same name. Otherwise, make sure '#{pretrained_model_name_or_path}' is the correct path to a" +
            " directory containing a #{IMAGE_PROCESSOR_NAME} file"
        end
      end

      begin
        image_processor_dict = JSON.load_file(resolved_image_processor_file).transform_keys(&:to_sym)
      rescue JSON::ParserError
        raise EnvironmentError,
          "It looks like the config file at '#{resolved_image_processor_file}' is not a valid JSON file."
      end

      if is_local
        Transformers.logger.info("loading configuration file #{resolved_image_processor_file}")
      else
        Transformers.logger.info(
          "loading configuration file #{image_processor_file} from cache at #{resolved_image_processor_file}"
        )
      end

      if !is_local
        if image_processor_dict.include?("auto_map")
          raise Todo
        end
        if image_processor_dict.include?("custom_pipelines")
          raise Todo
        end
      end
      [image_processor_dict, kwargs]
    end

    def self.from_dict(image_processor_dict, **kwargs)
      image_processor_dict = image_processor_dict.dup
      return_unused_kwargs = kwargs.delete(:return_unused_kwargs) { false }

      # The `size` parameter is a dict and was previously an int or tuple in feature extractors.
      # We set `size` here directly to the `image_processor_dict` so that it is converted to the appropriate
      # dict within the image processor and isn't overwritten if `size` is passed in as a kwarg.
      if kwargs.include?(:size) && image_processor_dict.include?(:size)
        image_processor_dict[:size] = kwargs.delete(:size)
      end
      if kwargs.include?(:crop_size) && image_processor_dict.include?(:crop_size)
        image_processor_dict[:crop_size] = kwargs.delete(:crop_size)
      end

      image_processor = new(**image_processor_dict)

      # Update image_processor with kwargs if needed
      to_remove = []
      kwargs.each do |key, value|
        if image_processor.instance_variable_defined?("@#{key}")
          image_processor.instance_variable_set("@#{key}", value)
          to_remove << key
        end
      end
      to_remove.each do |key|
        kwargs.delete(key)
      end

      Transformers.logger.info("Image processor #{image_processor}")
      if return_unused_kwargs
        [image_processor, kwargs]
      else
        image_processor
      end
    end
  end
end
