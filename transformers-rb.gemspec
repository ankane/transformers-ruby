require_relative "lib/transformers/version"

Gem::Specification.new do |spec|
  spec.name          = "transformers-rb"
  spec.version       = Transformers::VERSION
  spec.summary       = "State-of-the-art transformers for Ruby"
  spec.homepage      = "https://github.com/ankane/transformers-ruby"
  spec.license       = "Apache-2.0"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib,licenses}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 3.1"

  spec.add_dependency "numo-narray", ">= 0.9.2"
  spec.add_dependency "safetensors", ">= 0.1.1"
  spec.add_dependency "tokenizers", ">= 0.5"
  spec.add_dependency "torch-rb", ">= 0.17.1"
end
