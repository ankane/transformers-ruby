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
  class ChannelDimension < ExplicitEnum
    FIRST = "channels_first"
    LAST = "channels_last"
  end

  module ImageUtils
    def self.load_image(image, timeout: nil)
      Utils.requires_backends(__method__, ["vision"])
      if image.is_a?(URI)
        require "open-uri"

        image = Vips::Image.new_from_buffer(image.read(open_timeout: timeout, read_timeout: timeout), "")
      elsif image.is_a?(String) && File.exist?(image)
        image = Vips::Image.new_from_file(image)
      elsif image.is_a?(Vips::Image)
        image = image
      else
        raise ArgumentError, "Incorrect format used for image"
      end
      image
    end

    def self.validate_preprocess_arguments(
      do_rescale: nil,
      rescale_factor: nil,
      do_normalize: nil,
      image_mean: nil,
      image_std: nil,
      do_pad: nil,
      size_divisibility: nil,
      do_center_crop: nil,
      crop_size: nil,
      do_resize: nil,
      size: nil,
      resample:  nil
    )
      if do_rescale && rescale_factor.nil?
        raise ArgumentError, "`rescale_factor` must be specified if `do_rescale` is `true`."
      end

      if do_pad && size_divisibility.nil?
        # Here, size_divisor might be passed as the value of size
        raise ArgumentError, "Depending on the model, `size_divisibility`, `size_divisor`, `pad_size` or `size` must be specified if `do_pad` is `true`."
      end

      if do_normalize && (image_mean.nil? || image_std.nil?)
        raise ArgumentError, "`image_mean` and `image_std` must both be specified if `do_normalize` is `true`."
      end

      if do_center_crop && crop_size.nil?
        raise ArgumentError, "`crop_size` must be specified if `do_center_crop` is `true`."
      end

      if do_resize && (size.nil? || resample.nil?)
        raise ArgumentError, "`size` and `resample` must be specified if `do_resize` is `true`."
      end
    end

    def self.make_list_of_images(images, expected_ndims: 3)
      # TODO improve
      images.is_a?(Array) ? images : [images]
    end

    def self.to_numo_array(img)
      Numo::UInt8.from_binary(img.write_to_memory, [img.height, img.width, img.bands])
    end

    def self.infer_channel_dimension_format(
      image, num_channels: nil
    )
      num_channels = !num_channels.nil? ? num_channels : [1, 3]
      num_channels = num_channels.is_a?(Integer) ? [num_channels] : num_channels

      if image.ndim == 3
        first_dim, last_dim = 0, 2
      elsif image.ndim == 4
        first_dim, last_dim = 1, 3
      else
        raise ArgumentError, "Unsupported number of image dimensions: #{image.ndim}"
      end

      if num_channels.include?(image.shape[first_dim]) && num_channels.include?(image.shape[last_dim])
        Transformers.logger.warn(
          "The channel dimension is ambiguous. Got image shape #{image.shape}. Assuming channels are the first dimension."
        )
        return ChannelDimension::FIRST
      elsif num_channels.include?(image.shape[first_dim])
        return ChannelDimension::FIRST
      elsif num_channels.include?(image.shape[last_dim])
        return ChannelDimension::LAST
      end
      raise ArgumentError, "Unable to infer channel dimension format"
    end

    def self.get_channel_dimension_axis(
      image, input_data_format: nil
    )
      if input_data_format.nil?
        input_data_format = infer_channel_dimension_format(image)
      end
      if input_data_format == ChannelDimension::FIRST
        return image.ndim - 3
      elsif input_data_format == ChannelDimension::LAST
        return image.ndim - 1
      end
      raise ArgumentError, "Unsupported data format: #{input_data_format}"
    end

    def self.is_vips_image(img)
      Utils.is_vision_available && img.is_a?(Vips::Image)
    end

    def self.is_valid_image(img)
      is_vips_image(img) || is_numo_array(img) || is_torch_tensor(img)
    end

    def self.valid_images(imgs)
      # If we have an list of images, make sure every image is valid
      if imgs.is_a?(Array)
        imgs.each do |img|
          if !valid_images(img)
            return false
          end
        end
      # If not a list of tuple, we have been given a single image or batched tensor of images
      elsif !is_valid_image(imgs)
        return false
      end
      true
    end

    def self.is_scaled_image(image)
      if image.is_a?(Numo::UInt8)
        return false
      end

      # It's possible the image has pixel values in [0, 255] but is of floating type
      image.min >= 0 && image.max <= 1
    end

    def self.validate_kwargs(valid_processor_keys:, captured_kwargs:)
      unused_keys = Set.new(captured_kwargs).difference(Set.new(valid_processor_keys))
      if unused_keys.any?
        unused_key_str = unused_keys.join(", ")
        # TODO raise a warning here instead of simply logging?
        Transformers.logger.warn("Unused or unrecognized kwargs: #{unused_key_str}.")
      end
    end
  end
end
