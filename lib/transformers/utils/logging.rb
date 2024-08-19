# Copyright 2020 Optuna, Hugging Face
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
  class << self
    attr_accessor :logger
  end

  # TODO add detail
  LOG_LEVELS = {
    "debug" => Logger::DEBUG,
    "info" => Logger::INFO,
    "warning" => Logger::WARN,
    "error" => Logger::ERROR,
    "critical" => Logger::FATAL
  }

  DEFAULT_LOG_LEVEL = Logger::WARN

  def self._get_default_logging_level
    env_level_str = ENV["TRANSFORMERS_VERBOSITY"]
    if env_level_str
      if LOG_LEVELS.include?(env_level_str)
        return LOG_LEVELS[env_level_str]
      else
        warn(
          "Unknown option TRANSFORMERS_VERBOSITY=#{env_level_str}, " +
          "has to be one of: #{LOG_LEVELS.keys.join(", ")}"
        )
      end
    end
    DEFAULT_LOG_LEVEL
  end

  self.logger = begin
    logger = Logger.new(STDERR)
    logger.level = _get_default_logging_level
    logger.formatter = proc { |severity, datetime, progname, msg| "#{msg}\n" }
    logger
  end
end
