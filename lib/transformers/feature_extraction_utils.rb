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
  class BatchFeature
    def initialize(data:, tensor_type:)
      @data = data
      convert_to_tensors(tensor_type: tensor_type)
    end

    def to_h
      @data
    end
    alias_method :to_hash, :to_h

    def [](item)
      @data[item]
    end

    def keys
      @data.keys
    end

    def values
      @data.values
    end

    def items
      @data
    end

    def _get_is_as_tensor_fns(tensor_type: nil)
      if tensor_type.nil?
        return [nil, nil]
      end

      as_tensor = lambda do |value|
        if value.is_a?(Array) && value.length > 0 && value[0].is_a?(Numo::NArray)
          value = Numo::NArray.cast(value)
        end
        Torch.tensor(value)
      end

      is_tensor = Torch.method(:tensor?)

      [is_tensor, as_tensor]
    end

    def convert_to_tensors(tensor_type: nil)
      if tensor_type.nil?
        return self
      end

      is_tensor, as_tensor = _get_is_as_tensor_fns(tensor_type: tensor_type)

      # Do the tensor conversion in batch
      items.each do |key, value|
        begin
          if !is_tensor.(value)
            tensor = as_tensor.(value)

            @data[key] = tensor
          end
        rescue
          if key == :overflowing_values
            raise ArgumentError, "Unable to create tensor returning overflowing values of different lengths."
          end
          raise ArgumentError,
            "Unable to create tensor, you should probably activate padding " +
            "with 'padding: true' to have batched tensors with the same length."
        end
      end

      self
    end

    def to(*args, **kwargs)
      new_data = {}
      device = kwargs[:device]
      # Check if the args are a device or a dtype
      if device.nil? && args.length > 0
        raise Todo
      end
      # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
      items.each do |k, v|
        # check if v is a floating point
        if Torch.floating_point?(v)
          # cast and send to device
          new_data[k] = v.to(*args, **kwargs)
        elsif !device.nil?
          new_data[k] = v.to(device)
        else
          new_data[k] = v
        end
      end
      @data = new_data
      self
    end
  end
end
