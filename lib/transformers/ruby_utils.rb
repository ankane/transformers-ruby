module Transformers
  module ClassAttribute
    def class_attribute(name, default = nil)
      singleton_class.attr_writer name
      var = "@#{name}"
      instance_variable_set(var, default)
      singleton_class.define_method(name) do
        # ancestors includes current module
        ancestors.find { |c| c.instance_variable_defined?(var) }.instance_variable_get(var)
      end
      define_method(name) do
        self.class.send(name)
      end
    end
  end

  module Copy
    def self.deepcopy(value, memo = {})
      key = value.object_id
      if !memo.key?(key)
        copy = value.dup
        memo[key] = copy
        if value.is_a?(Hash)
          copy.transform_keys! { |k| deepcopy(k, memo) }
          copy.transform_values! { |v| deepcopy(v, memo) }
        elsif value.is_a?(Array)
          copy.map! { |v| deepcopy(v, memo) }
        end
      end
      memo[key]
    end
  end
end
