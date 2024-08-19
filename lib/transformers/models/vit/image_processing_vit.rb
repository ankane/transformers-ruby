# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
  module Vit
    class ViTImageProcessor < BaseImageProcessor
      def initialize(
        do_resize: true,
        size: nil,
        resample: :bilinear,
        do_rescale: true,
        rescale_factor: 1 / 255.0,
        do_normalize: true,
        image_mean: nil,
        image_std: nil,
        **kwargs
      )
        super(**kwargs)
        size = !size.nil? ? size : {height: 224, width: 224}
        size = ImageProcessingUtils.get_size_dict(size)
        @do_resize = do_resize
        @do_rescale = do_rescale
        @do_normalize = do_normalize
        @size = size
        @resample = resample
        @rescale_factor = rescale_factor
        @image_mean = !image_mean.nil? ? image_mean : IMAGENET_STANDARD_MEAN
        @image_std = !image_std.nil? ? image_std : IMAGENET_STANDARD_STD
        @valid_processor_keys = [
          :images,
          :do_resize,
          :size,
          :resample,
          :do_rescale,
          :rescale_factor,
          :do_normalize,
          :image_mean,
          :image_std,
          :return_tensors,
          :data_format,
          :input_data_format
        ]
      end

      def resize(
        image,
        size,
        resample: :bilinear,
        data_format: nil,
        input_data_format: nil,
        **kwargs
      )
        size = ImageProcessingUtils.get_size_dict(size)
        if !size.include?(:height) || !size.include?(:width)
          raise ArgumentError, "The `size` dictionary must contain the keys `height` and `width`. Got #{size.keys}"
        end
        output_size = [size[:height], size[:width]]
        ImageTransforms.resize(
          image,
          output_size,
          resample: resample,
          data_format: data_format,
          input_data_format: input_data_format,
          **kwargs
        )
      end

      def preprocess(
        images,
        do_resize: nil,
        size: nil,
        resample: nil,
        do_rescale: nil,
        rescale_factor: nil,
        do_normalize: nil,
        image_mean: nil,
        image_std: nil,
        return_tensors: nil,
        data_format: ChannelDimension::FIRST,
        input_data_format: nil,
        **kwargs
      )
        do_resize = !do_resize.nil? ? do_resize : @do_resize
        do_rescale = !do_rescale.nil? ? do_rescale : @do_rescale
        do_normalize = !do_normalize.nil? ? do_normalize : @do_normalize
        resample = !resample.nil? ? resample : @resample
        rescale_factor = !rescale_factor.nil? ? rescale_factor : @rescale_factor
        image_mean = !image_mean.nil? ? image_mean : @image_mean
        image_std = !image_std.nil? ? image_std : @image_std

        size = !size.nil? ? size : @size
        size_dict = ImageProcessingUtils.get_size_dict(size)

        images = ImageUtils.make_list_of_images(images)

        ImageUtils.validate_kwargs(captured_kwargs: kwargs.keys, valid_processor_keys: @valid_processor_keys)

        if !ImageUtils.valid_images(images)
          raise ArgumentError,
            "Invalid image type. Must be of type Vips::Image, Numo::NArray, or Torch::Tensor."
        end
        ImageUtils.validate_preprocess_arguments(
          do_rescale: do_rescale,
          rescale_factor: rescale_factor,
          do_normalize: do_normalize,
          image_mean: image_mean,
          image_std: image_std,
          do_resize: do_resize,
          size: size,
          resample: resample
        )

        # All transformations expect numo arrays.
        images = images.map { |image| ImageUtils.to_numo_array(image) }

        if ImageUtils.is_scaled_image(images[0]) && do_rescale
          Transformers.logger.warn(
            "It looks like you are trying to rescale already rescaled images. If the input" +
            " images have pixel values between 0 and 1, set `do_rescale: false` to avoid rescaling them again."
          )
        end

        if input_data_format.nil?
          # We assume that all images have the same channel dimension format.
          input_data_format = ImageUtils.infer_channel_dimension_format(images[0])
        end

        if do_resize
          images =
            images.map do |image|
              resize(image, size_dict, resample: resample, input_data_format: input_data_format)
            end
        end

        if do_rescale
          images =
            images.map do |image|
              rescale(image, rescale_factor, input_data_format: input_data_format)
            end
        end

        if do_normalize
          images =
            images.map do |image|
              normalize(image, image_mean, image_std, input_data_format: input_data_format)
            end
        end

        images =
          images.map do |image|
            ImageTransforms.to_channel_dimension_format(image, data_format, input_channel_dim: input_data_format)
          end

        data = {pixel_values: images}
        BatchFeature.new(data: data, tensor_type: return_tensors)
      end
    end
  end
end
