require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"

unless ENV["TRANSFORMERS_VERBOSITY"]
  Transformers.logger.level = Logger::ERROR
end

Transformers.fast_init = true

class Minitest::Test
  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def ci?
    ENV["CI"]
  end
end
