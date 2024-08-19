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
  module ImageTransforms
    def self.to_channel_dimension_format(
      image,
      channel_dim,
      input_channel_dim: nil
    )
      if !image.is_a?(Numo::NArray)
        raise ArgumentError, "Input image must be of type Numo::NArray, got #{image.class.name}"
      end

      if input_channel_dim.nil?
        input_channel_dim = infer_channel_dimension_format(image)
      end

      target_channel_dim = ChannelDimension.new(channel_dim).to_s
      if input_channel_dim == target_channel_dim
        return image
      end

      if target_channel_dim == ChannelDimension::FIRST
        image = image.transpose(2, 0, 1)
      elsif target_channel_dim == ChannelDimension::LAST
        image = image.transpose(1, 2, 0)
      else
        raise ArgumentError, "Unsupported channel dimension format: #{channel_dim}"
      end

      image
    end

    def self.rescale(
      image,
      scale,
      data_format: nil,
      dtype: Numo::SFloat,
      input_data_format: nil
    )
      if !image.is_a?(Numo::NArray)
        raise ArgumentError, "Input image must be of type Numo::NArray, got #{image.class.name}"
      end

      rescaled_image = image * scale
      if !data_format.nil?
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)
      end

      rescaled_image = rescaled_image.cast_to(dtype)

      rescaled_image
    end

    def self.resize(
      image,
      size,
      resample: nil,
      reducing_gap: nil,
      data_format: nil,
      return_numpy: true,
      input_data_format: nil
    )
      resample = !resample.nil? ? resample : nil # PILImageResampling.BILINEAR

      if size.length != 2
        raise ArgumentError, "size must have 2 elements"
      end

      # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
      # The resized image from PIL will always have channels last, so find the input format first.
      if input_data_format.nil?
        input_data_format = ImageUtils.infer_channel_dimension_format(image)
      end
      data_format = data_format.nil? ? input_data_format : data_format

      # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
      # the pillow library to resize the image and then convert back to numpy
      do_rescale = false
      if !image.is_a?(Vips::Image)
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(image, do_rescale: do_rescale, input_data_format: input_data_format)
      end
      height, width = size
      # TODO support resample
      resized_image = image.thumbnail_image(width, height: height, size: :force)

      if return_numpy
        resized_image = ImageUtils.to_numo_array(resized_image)
        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        resized_image = resized_image.ndim == 2 ? resized_image.expand_dims(-1) : resized_image
        # The image is always in channels last format after converting from a PIL image
        resized_image = to_channel_dimension_format(
          resized_image, data_format, input_channel_dim: ChannelDimension::LAST
        )
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        resized_image = do_rescale ? rescale(resized_image, 1 / 255.0) : resized_image
      end
      resized_image
    end

    def self.normalize(
      image,
      mean,
      std,
      data_format: nil,
      input_data_format: nil
    )
      if !image.is_a?(Numo::NArray)
        raise ArgumentError, "image must be a numpy array"
      end

      if input_data_format.nil?
        input_data_format = infer_channel_dimension_format(image)
      end

      channel_axis = ImageUtils.get_channel_dimension_axis(image, input_data_format: input_data_format)
      num_channels = image.shape[channel_axis]

      # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
      # We preserve the original dtype if it is a float type to prevent upcasting float16.
      if !image.is_a?(Numo::SFloat) && !image.is_a?(Numo::DFloat)
        image = image.cast_to(Numo::SFloat)
      end

      if mean.is_a?(Enumerable)
        if mean.length != num_channels
          raise ArgumentError, "mean must have #{num_channels} elements if it is an iterable, got #{mean.length}"
        end
      else
        mean = [mean] * num_channels
      end
      mean = Numo::DFloat.cast(mean)

      if std.is_a?(Enumerable)
        if std.length != num_channels
          raise ArgumentError, "std must have #{num_channels} elements if it is an iterable, got #{std.length}"
        end
      else
        std = [std] * num_channels
      end
      std = Numo::DFloat.cast(std)

      if input_data_format == ChannelDimension::LAST
        image = (image - mean) / std
      else
        image = ((image.transpose - mean) / std).transpose
      end

      image = !data_format.nil? ? to_channel_dimension_format(image, data_format, input_data_format) : image
      image
    end

    def self.to_pil_image(
      image,
      do_rescale: nil,
      input_data_format: nil
    )
      if image.is_a?(Vips::Image)
        return image
      end

      # Convert all tensors to numo arrays before converting to Vips image
      if !image.is_a?(Numo::NArray)
        raise ArgumentError, "Input image type not supported: #{image.class.name}"
      end

      # If the channel has been moved to first dim, we put it back at the end.
      image = to_channel_dimension_format(image, ChannelDimension::LAST, input_channel_dim: input_data_format)

      # If there is a single channel, we squeeze it, as otherwise PIL can't handle it.
      # image = image.shape[-1] == 1 ? image.squeeze(-1) : image

      # Rescale the image to be between 0 and 255 if needed.
      do_rescale = do_rescale.nil? ? _rescale_for_pil_conversion(image) : do_rescale

      if do_rescale
        image = rescale(image, 255)
      end

      image = image.cast_to(Numo::UInt8)
      Vips::Image.new_from_memory(image.to_binary, image.shape[1], image.shape[0], image.shape[2], :uchar)
    end

    def self._rescale_for_pil_conversion(image)
      if image.is_a?(Numo::UInt8)
        do_rescale = false
      else
        raise Todo
      end
      do_rescale
    end
  end
end
