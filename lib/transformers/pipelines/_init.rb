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
  TASK_ALIASES = {
    "sentiment-analysis" => "text-classification",
    "ner" => "token-classification"
  }

  SUPPORTED_TASKS = {
    "feature-extraction" => {
      "impl" => FeatureExtractionPipeline,
      "pt" => [AutoModel],
      "default" => {
        "model" => {
          "pt" => ["distilbert/distilbert-base-cased", "6ea8117"]
        }
      },
      "type" => "multimodal"
    },
    "text-classification" => {
      "impl" => TextClassificationPipeline,
      "pt" => [AutoModelForSequenceClassification],
      "default" => {
        "model" => {
          "pt" => ["distilbert/distilbert-base-uncased-finetuned-sst-2-english", "714eb0f"]
        }
      },
      "type" => "text"
    },
    "token-classification" => {
      "impl" => TokenClassificationPipeline,
      "pt" => [AutoModelForTokenClassification],
      "default" => {
        "model" => {
          "pt" => ["dbmdz/bert-large-cased-finetuned-conll03-english", "4c53496"]
        }
      },
      "type" => "text"
    },
    "question-answering" => {
      "impl" => QuestionAnsweringPipeline,
      "pt" => [AutoModelForQuestionAnswering],
      "default" => {
        "model" => {
          "pt" => ["distilbert/distilbert-base-cased-distilled-squad", "564e9b5"]
        }
      },
      "type" => "text"
    },
    "image-classification" => {
      "impl" => ImageClassificationPipeline,
      "pt" => [AutoModelForImageClassification],
      "default" => {
        "model" => {
          "pt" => ["google/vit-base-patch16-224", "3f49326"]
        }
      },
      "type" => "image"
    },
    "image-feature-extraction" => {
      "impl" => ImageFeatureExtractionPipeline,
      "pt" => [AutoModel],
      "default" => {
        "model" => {
          "pt" => ["google/vit-base-patch16-224", "3f49326"]
        }
      },
      "type" => "image"
    },
    "embedding" => {
      "impl" => EmbeddingPipeline,
      "pt" => [AutoModel],
      "default" => {
        "model" => {
          "pt" => ["sentence-transformers/all-MiniLM-L6-v2", "8b3219a"]
        }
      },
      "type" => "text"
    },
    "reranking" => {
      "impl" => RerankingPipeline,
      "pt" => [AutoModelForSequenceClassification],
      "default" => {
        "model" => {
          "pt" => ["mixedbread-ai/mxbai-rerank-base-v1", "03241da"]
        }
      },
      "type" => "text"
    }
  }

  PIPELINE_REGISTRY = PipelineRegistry.new(supported_tasks: SUPPORTED_TASKS, task_aliases: TASK_ALIASES)

  class << self
    def pipeline(
      task,
      model_arg = nil,
      model: nil,
      config: nil,
      tokenizer: nil,
      feature_extractor: nil,
      image_processor: nil,
      framework: nil,
      revision: nil,
      use_fast: true,
      token: nil,
      device: nil,
      device_map: nil,
      torch_dtype: nil,
      trust_remote_code: nil,
      model_kwargs: nil,
      pipeline_class: nil,
      **kwargs
    )
      if !model_arg.nil?
        if !model.nil?
          raise ArgumentError, "Cannot pass multiple models"
        end
        model = model_arg
      end

      model_kwargs ||= {}
      # Make sure we only pass use_auth_token once as a kwarg (it used to be possible to pass it in model_kwargs,
      # this is to keep BC).
      use_auth_token = model_kwargs.delete(:use_auth_token)
      if !use_auth_token.nil?
        raise Todo
      end

      code_revision = kwargs.delete(:code_revision)
      commit_hash = kwargs.delete(:_commit_hash)

      hub_kwargs = {
        revision: revision,
        token: token,
        trust_remote_code: trust_remote_code,
        _commit_hash: commit_hash
      }

      if task.nil? && model.nil?
        raise RuntimeError,
          "Impossible to instantiate a pipeline without either a task or a model " +
          "being specified. " +
          "Please provide a task class or a model"
      end

      if model.nil? && !tokenizer.nil?
        raise RuntimeError,
          "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer" +
          " may not be compatible with the default model. Please provide a PreTrainedModel class or a" +
          " path/identifier to a pretrained model when providing tokenizer."
      end
      if model.nil? && !feature_extractor.nil?
        raise RuntimeError,
          "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided" +
          " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class" +
          " or a path/identifier to a pretrained model when providing feature_extractor."
      end
      if model.is_a?(Pathname)
        model = model.to_s
      end

      if commit_hash.nil?
        pretrained_model_name_or_path = nil
        if config.is_a?(String)
          pretrained_model_name_or_path = config
        elsif config.nil? && model.is_a?(String)
          pretrained_model_name_or_path = model
        end

        if !config.is_a?(PretrainedConfig) && !pretrained_model_name_or_path.nil?
          # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
          resolved_config_file = Utils::Hub.cached_file(
            pretrained_model_name_or_path,
            CONFIG_NAME,
            _raise_exceptions_for_gated_repo: false,
            _raise_exceptions_for_missing_entries: false,
            _raise_exceptions_for_connection_errors: false,
            cache_dir: model_kwargs[:cache_dir],
            **hub_kwargs
          )
          hub_kwargs[:_commit_hash] = Utils::Hub.extract_commit_hash(resolved_config_file, commit_hash)
        else
          hub_kwargs[:_commit_hash] = nil # getattr(config, "_commit_hash", None)
        end
      end

      # Config is the primordial information item.
      # Instantiate config if needed
      if config.is_a?(String)
        raise Todo
      elsif config.nil? && model.is_a?(String)
        config = AutoConfig.from_pretrained(
          model, _from_pipeline: task, code_revision: code_revision, **hub_kwargs, **model_kwargs
        )
        hub_kwargs[:_commit_hash] = config._commit_hash
      end

      custom_tasks = {}
      if !config.nil? && (config.instance_variable_get(:@custom_pipelines) || {}).length > 0
        raise Todo
      end

      if task.nil? && !model.nil?
        raise Todo
      end

      # Retrieve the task
      if custom_tasks.include?(task)
        raise Todo
      else
        _normalized_task, targeted_task, task_options = check_task(task)
        if pipeline_class.nil?
          pipeline_class = targeted_task["impl"]
        end
      end

      # Use default model/config/tokenizer for the task if no model is provided
      if model.nil?
        # At that point framework might still be undetermined
        model, default_revision = Pipelines.get_default_model_and_revision(targeted_task, framework, task_options)
        revision = !revision.nil? ? revision : default_revision
        Transformers.logger.warn(
          "No model was supplied, defaulted to #{model} and revision" +
          " #{revision} (#{Utils::Hub::HUGGINGFACE_CO_RESOLVE_ENDPOINT}/#{model}).\n" +
          "Using a pipeline without specifying a model name and revision in production is not recommended."
        )
        hub_kwargs[:revision] = revision
        if config.nil? && model.is_a?(String)
          config = AutoConfig.from_pretrained(model, _from_pipeline: task, **hub_kwargs, **model_kwargs)
          hub_kwargs[:_commit_hash] = config._commit_hash
        end
      end

      if !device_map.nil?
        raise Todo
      end
      if !torch_dtype.nil?
        raise Todo
      end

      model_name = model.is_a?(String) ? model : nil

      # Load the correct model if possible
      # Infer the framework from the model if not already defined
      if model.is_a?(String) || framework.nil?
        model_classes = {"tf" => targeted_task["tf"], "pt" => targeted_task["pt"]}
        framework, model =
          Pipelines.infer_framework_load_model(
            model,
            config,
            model_classes: model_classes,
            framework: framework,
            task: task,
            **hub_kwargs,
            **model_kwargs
          )
      end

      model_config = model.config
      hub_kwargs[:_commit_hash] = model.config._commit_hash
      model_config_type = model_config.class.name.split("::").last
      load_tokenizer = TOKENIZER_MAPPING.include?(model_config_type) || !model_config.tokenizer_class.nil?
      load_feature_extractor = FEATURE_EXTRACTOR_MAPPING.include?(model_config_type) || !feature_extractor.nil?
      load_image_processor = IMAGE_PROCESSOR_MAPPING.include?(model_config_type) || !image_processor.nil?

      if load_tokenizer
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer.nil?
          if model_name.is_a?(String)
            tokenizer = model_name
          elsif config.is_a?(String)
            tokenizer = config
          else
            # Impossible to guess what is the right tokenizer here
            raise "Impossible to guess which tokenizer to use. Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
          end
        end

        # Instantiate tokenizer if needed
        if tokenizer.is_a?(String) || tokenizer.is_a?(Array)
          if tokenizer.is_a?(Array)
            # For array we have [tokenizer name, {kwargs}]
            use_fast = tokenizer[1].delete(:use_fast) { use_fast }
            tokenizer_identifier = tokenizer[0]
            tokenizer_kwargs = tokenizer[1]
          else
            tokenizer_identifier = tokenizer
            tokenizer_kwargs = model_kwargs.dup
            tokenizer_kwargs.delete(:torch_dtype)
          end

          tokenizer =
            AutoTokenizer.from_pretrained(
              tokenizer_identifier, use_fast: use_fast, _from_pipeline: task, **hub_kwargs, **tokenizer_kwargs
            )
        end
      end

      if load_image_processor
        # Try to infer image processor from model or config name (if provided as str)
        if image_processor.nil?
          if model_name.is_a?(String)
            image_processor = model_name
          elsif config.is_a?(String)
            image_processor = config
          # Backward compatibility, as `feature_extractor` used to be the name
          # for `ImageProcessor`.
          elsif !feature_extractor.nil? && feature_extractor.is_a?(BaseImageProcessor)
            image_processor = feature_extractor
          else
            # Impossible to guess what is the right image_processor here
            raise RuntimeError,
              "Impossible to guess which image processor to use. " +
              "Please provide a PreTrainedImageProcessor class or a path/identifier " +
              "to a pretrained image processor."
          end
        end

        # Instantiate image_processor if needed
        if image_processor.is_a?(String) || image_processor.is_a?(Array)
          image_processor = AutoImageProcessor.from_pretrained(
            image_processor, _from_pipeline: task, **hub_kwargs, **model_kwargs
          )
        end
      end

      if load_feature_extractor
        raise Todo
      end

      if task == "translation" && model.config.task_specific_params
        raise Todo
      end

      if !tokenizer.nil?
        kwargs[:tokenizer] = tokenizer
      end

      if !feature_extractor.nil?
        kwargs[:feature_extractor] = feature_extractor
      end

      if !torch_dtype.nil?
        kwargs[:torch_dtype] = torch_dtype
      end

      if !image_processor.nil?
        kwargs[:image_processor] = image_processor
      end

      if !device.nil?
        kwargs[:device] = device
      end

      pipeline_class.new(model, framework: framework, task: task, **kwargs)
    end

    private

    def check_task(task)
      PIPELINE_REGISTRY.check_task(task)
    end
  end
end
