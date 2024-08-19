# Copyright 2022 The HuggingFace Team. All rights reserved.
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
  class ModelOutput
    def self.attributes
      @attributes ||= []
    end

    def self.attribute(attribute)
      attributes << attribute.to_sym

      define_method(attribute) do
        self[attribute]
      end
    end

    def initialize(**kwargs)
      @data = kwargs
    end

    def [](k)
      if k.is_a?(String) || k.is_a?(Symbol)
        @data[k.to_sym]
      else
        to_tuple[k]
      end
    end

    def to_tuple
      self.class.attributes.map { |k| @data[k] }.compact
    end
  end

  class ExplicitEnum
    def initialize(value)
      expected = self.class.constants.map { |k| self.class.const_get(k) }
      unless expected.include?(value)
        raise ArgumentError, "#{value} is not a valid #{self.class.name}, please select one of #{expected.inspect}"
      end
      @value = value
    end

    def to_s
      @value
    end
  end

  class PaddingStrategy < ExplicitEnum
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"
  end

  class TensorType < ExplicitEnum
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
    MLX = "mlx"
  end

  module Utils
    def self.infer_framework(model_class)
      if model_class < Torch::NN::Module
        "pt"
      else
        raise TypeError, "Could not infer framework from class #{model_class}."
      end
    end

    def self._is_numo(x)
      x.is_a?(Numo::NArray)
    end

    def self.is_numo_array(x)
      _is_numo(x)
    end

    def self._is_torch(x)
      x.is_a?(Torch::Tensor)
    end

    def self.is_torch_tensor(x)
      _is_torch(x)
    end

    def self._is_torch_device(x)
      x.is_a?(Torch::Device)
    end

    def self.is_torch_device(x)
      _is_torch_device(x)
    end
  end
end
