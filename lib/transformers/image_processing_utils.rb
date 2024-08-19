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
  class BaseImageProcessor < ImageProcessingMixin
    def initialize(**kwargs)
      super(**kwargs)
    end

    def call(images, **kwargs)
      preprocess(images, **kwargs)
    end

    def preprocess(images, **kwargs)
      raise NotImplementedError, "Each image processor must implement its own preprocess method"
    end

    def rescale(
      image,
      scale,
      data_format: nil,
      input_data_format: nil,
      **kwargs
    )
      ImageTransforms.rescale(image, scale, data_format: data_format, input_data_format: input_data_format, **kwargs)
    end

    def normalize(
      image,
      mean,
      std,
      data_format: nil,
      input_data_format: nil,
      **kwargs
    )
      ImageTransforms.normalize(
        image, mean, std, data_format: data_format, input_data_format: input_data_format, **kwargs
      )
    end
  end

  module ImageProcessingUtils
    def self.get_size_dict(size)
      if !size.is_a?(Hash)
        size_dict = {height: size, width: size}
      else
        size_dict = size
      end
      size_dict
    end
  end
end
