# Copyright 2020 The HuggingFace Team. All rights reserved.
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
  class GELUActivation < Torch::NN::Module
    def initialize(use_gelu_python: false)
      super()
      if use_gelu_python
        @act = _gelu_python
      else
        @act = Torch::NN::Functional.method(:gelu)
      end
    end

    def _gelu_python(input)
      input * 0.5 * (1.0 + Torch.erf(input / Math.sqrt(2.0)))
    end

    def forward(input)
      @act.(input)
    end
  end

  class ClassInstantier
    def initialize(data)
      @data = data
    end

    def [](key)
      content = @data.fetch(key)
      cls, kwargs = content.is_a?(Array) ? content : [content, {}]
      cls.new(**kwargs)
    end
  end

  ACT2CLS = {
    "gelu" => GELUActivation
  }
  ACT2FN = ClassInstantier.new(ACT2CLS)

  module Activations
    def self.get_activation(activation_string)
      ACT2FN[activation_string]
    end
  end
end
