# Copyright 2018 The HuggingFace Inc. team.
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
  module Pipelines
    def self.get_default_model_and_revision(targeted_task, framework, task_options)
      defaults = targeted_task["default"]

      if defaults.key?("model")
        default_models = targeted_task["default"]["model"]
      end

      if framework.nil?
        framework = "pt"
      end

      default_models[framework]
    end

    def self.infer_framework_load_model(
      model,
      config,
      model_classes: nil,
      task: nil,
      framework: nil,
      **model_kwargs
    )
      if model.is_a?(String)
        model_kwargs[:_from_pipeline] = task
        class_tuple = []
        look_pt = true

        if model_classes
          if look_pt
            class_tuple = class_tuple + model_classes.fetch("pt", AutoModel)
          end
        end
        if config.architectures
          classes = []
          config.architectures.each do |architecture|
            if look_pt
              _class = Transformers.const_get(architecture)
              if !_class.nil?
                classes << _class
              end
            end
          end
          class_tuple = class_tuple + classes
        end

        if class_tuple.length == 0
          raise ArgumentError, "Pipeline cannot infer suitable model classes from #{model}"
        end

        class_tuple.each do |model_class|
          raise Error, "Invalid auto model class: #{model_class}" unless model_class < BaseAutoModelClass
          kwargs = model_kwargs.dup

          begin
            model = model_class.from_pretrained(model, **kwargs)
            if model.respond_to?(:eval)
              model = model.eval
            end
            break
          rescue
            # TODO
            raise
          end
        end
      end

      if framework.nil?
        framework = Utils.infer_framework(model.class)
      end
      [framework, model]
    end
  end

  class ArgumentHandler
  end

  class Pipeline
    def initialize(
      model,
      tokenizer: nil,
      feature_extractor: nil,
      image_processor: nil,
      modelcard: nil,
      framework: nil,
      task: "",
      device: "cpu",
      **kwargs
    )
      if framework.nil?
        raise Todo
      end

      @task = task
      @model = model
      @tokenizer = tokenizer
      @feature_extractor = feature_extractor
      @image_processor = image_processor
      @modelcard = modelcard
      @framework = framework
      @device = device
      @device = Torch::Device.new(device) if device.is_a?(String)

      if device.nil?
        if Torch::CUDA.available? || Torch::Backends::MPS.available?
          Transformers.logger.warn(
            "Hardware accelerator e.g. GPU is available in the environment, but no `device` argument" +
            " is passed to the `Pipeline` object. Model will be on CPU."
          )
        end
      end

      @call_count = 0
      @batch_size = kwargs.delete(:batch_size)
      @num_workers = kwargs.delete(:num_workers)
      @preprocess_params, @forward_params, @postprocess_params = _sanitize_parameters(**kwargs)
    end

    def torch_dtype
      @model.dtype
    end

    def check_model_type(supported_models)
      if !supported_models.is_a?(Array)
        supported_models_names = []
        supported_models.each do |_, model_name|
          # Mapping can now contain tuples of models for the same configuration.
          if model_name.is_a?(Array)
            supported_models_names.concat(model_name)
          else
            supported_models_names << model_name
          end
        end
        supported_models = supported_models_names
      end
      if !supported_models.include?(@model.class.name.split("::").last)
        Transformers.logger.error(
          "The model '#{@model.class.name}' is not supported for #{@task}. Supported models are" +
          " #{supported_models}."
        )
      end
    end

    def get_iterator(
      inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
    )
      if inputs.respond_to?(:size)
        dataset = PipelineDataset.new(inputs, method(:preprocess), preprocess_params)
      else
        if num_workers > 1
          Transformers.logger.warn(
            "For iterable dataset using num_workers>1 is likely to result" +
            " in errors since everything is iterable, setting `num_workers: 1`" +
            " to guarantee correctness."
          )
          num_workers = 1
        end
        dataset = PipelineIterator.new(inputs, method(:preprocess), preprocess_params)
      end

      # TODO hack by collating feature_extractor and image_processor
      feature_extractor = !@feature_extractor.nil? ? @feature_extractor : @image_processor
      collate_fn = batch_size == 1 ? method(:no_collate_fn) : pad_collate_fn(@tokenizer, feature_extractor)
      dataloader = Torch::Utils::Data::DataLoader.new(dataset, batch_size: batch_size, collate_fn: collate_fn) # num_workers: num_workers,
      model_iterator = PipelineIterator.new(dataloader, method(:forward), forward_params, loader_batch_size: batch_size)
      final_iterator = PipelineIterator.new(model_iterator, method(:postprocess), postprocess_params)
      final_iterator
    end

    def call(inputs, *args, num_workers: nil, batch_size: nil, **kwargs)
      if args.any?
        Transformers.logger.warn("Ignoring args : #{args}")
      end

      if num_workers.nil?
        if @num_workers.nil?
          num_workers = 0
        else
          num_workers = @num_workers
        end
      end
      if batch_size.nil?
        if @batch_size.nil?
          batch_size = 1
        else
          batch_size = @batch_size
        end
      end

      preprocess_params, forward_params, postprocess_params = _sanitize_parameters(**kwargs)

      preprocess_params = @preprocess_params.merge(preprocess_params)
      forward_params = @forward_params.merge(forward_params)
      postprocess_params = @postprocess_params.merge(postprocess_params)

      @call_count += 1
      if @call_count > 10 && @framework == "pt" && @device.type == "cuda"
        Transformers.logger.warn(
          "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a" +
          " dataset"
        )
      end

      is_dataset = inputs.is_a?(Torch::Utils::Data::Dataset)
      is_generator = inputs.is_a?(Enumerable)
      is_list = inputs.is_a?(Array)

      _is_iterable = is_dataset || is_generator || is_list

      # TODO make the get_iterator work also for `tf` (and `flax`).
      can_use_iterator = @framework == "pt" && (is_dataset || is_generator || is_list)

      if is_list
        if can_use_iterator
          final_iterator = get_iterator(
            inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
          )
          outputs = final_iterator.to_a
          outputs
        else
          run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        end
      else
        run_single(inputs, preprocess_params, forward_params, postprocess_params)
      end
    end

    private

    def _sanitize_parameters(**kwargs)
      raise NotImplementedError, "_sanitize_parameters not implemented"
    end

    def forward(model_inputs, **forward_params)
      _forward(model_inputs, **forward_params)
    end

    def run_single(inputs, preprocess_params, forward_params, postprocess_params)
      model_inputs = preprocess(inputs, **preprocess_params)
      model_outputs = forward(model_inputs, **forward_params)
      outputs = postprocess(model_outputs, **postprocess_params)
      outputs
    end

    def no_collate_fn(items)
      if items.length != 1
        raise ArgumentError, "This collate_fn is meant to be used with batch_size=1"
      end
      items[0]
    end
  end

  class ChunkPipeline < Pipeline
    def run_single(inputs, preprocess_params, forward_params, postprocess_params)
      all_outputs = []
      preprocess(inputs, **preprocess_params) do |model_inputs|
        model_outputs = forward(model_inputs, **forward_params)
        all_outputs << model_outputs
      end
      outputs = postprocess(all_outputs, **postprocess_params)
      outputs
    end
  end

  class PipelineRegistry
    def initialize(supported_tasks:, task_aliases:)
      @supported_tasks = supported_tasks
      @task_aliases = task_aliases
    end

    def get_supported_tasks
      supported_task = @supported_tasks.keys + @task_aliases.keys
      supported_task.sort
    end

    def check_task(task)
      if @task_aliases[task]
        task = @task_aliases[task]
      end
      if @supported_tasks[task]
        targeted_task = @supported_tasks[task]
        return task, targeted_task, nil
      end

      raise KeyError, "Unknown task #{task}, available tasks are #{get_supported_tasks}"
    end
  end
end
