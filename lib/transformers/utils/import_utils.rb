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
  module Utils
    ENV_VARS_TRUE_VALUES = ["1", "ON", "YES", "TRUE"]

    def self.requires_backends(obj, backends)
      if !backends.is_a?(Array)
        backends = [backends]
      end

      name = obj.is_a?(Symbol) ? obj : obj.class.name

      checks = backends.map { |backend| BACKENDS_MAPPING.fetch(backend) }
      failed = checks.filter_map { |available, msg| format(msg, name) if !available.call }
      if failed.any?
        raise Error, failed.join("")
      end
    end

    def self.is_vision_available
      defined?(Vips)
    end

    VISION_IMPORT_ERROR = <<~MSG
    %s requires the `ruby-vips` gem
    MSG

    BACKENDS_MAPPING = {
      "vision" => [singleton_method(:is_vision_available), VISION_IMPORT_ERROR]
    }
  end
end
