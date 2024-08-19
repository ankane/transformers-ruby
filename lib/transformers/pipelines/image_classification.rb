module Transformers
  class ClassificationFunction < ExplicitEnum
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"
  end

  class ImageClassificationPipeline < Pipeline
    extend ClassAttribute

    class_attribute :function_to_apply, ClassificationFunction::NONE

    def initialize(*args, **kwargs)
      super(*args, **kwargs)
      Utils.requires_backends(self, "vision")
      check_model_type(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES)
    end

    def _sanitize_parameters(top_k: nil, function_to_apply: nil, timeout: nil)
      preprocess_params = {}
      if !timeout.nil?
        preprocess_params[:timeout] = timeout
      end
      postprocess_params = {}
      if !top_k.nil?
        postprocess_params[:top_k] = top_k
      end
      if function_to_apply.is_a?(String)
        function_to_apply = ClassificationFunction.new(function_to_apply.downcase).to_s
      end
      if !function_to_apply.nil?
        postprocess_params[:function_to_apply] = function_to_apply
      end
      [preprocess_params, {}, postprocess_params]
    end

    def preprocess(image, timeout: nil)
      image = ImageUtils.load_image(image, timeout: timeout)
      model_inputs = @image_processor.(image, return_tensors: @framework)
      if @framework == "pt"
        # TODO
        # model_inputs = model_inputs.to(torch_dtype)
      end
      model_inputs
    end

    def _forward(model_inputs)
      model_outputs = @model.(**model_inputs.to_h)
      model_outputs
    end

    def postprocess(model_outputs, function_to_apply: nil, top_k: 5)
      if function_to_apply.nil?
        if @model.config.problem_type == "multi_label_classification" || @model.config.num_labels == 1
          function_to_apply = ClassificationFunction::SIGMOID
        elsif @model.config.problem_type == "single_label_classification" || @model.config.num_labels > 1
          function_to_apply = ClassificationFunction::SOFTMAX
        elsif @model.config.instance_variable_defined?(:@function_to_apply) && function_to_apply.nil?
          function_to_apply = @model.config.function_to_apply
        else
          function_to_apply = ClassificationFunction::NONE
        end
      end

      if top_k > @model.config.num_labels
        top_k = @model.config.num_labels
      end

      outputs = model_outputs[:logits][0]
      if @framework == "pt" && [Torch.bfloat16, Torch.float16].include?(outputs.dtype)
        outputs = outputs.to(Torch.float32).numo
      else
        outputs = outputs.numo
      end

      if function_to_apply == ClassificationFunction::SIGMOID
        scores = sigmoid(outputs)
      elsif function_to_apply == ClassificationFunction::SOFTMAX
        scores = softmax(outputs)
      elsif function_to_apply == ClassificationFunction::NONE
        scores = outputs
      else
        raise ArgumentError, "Unrecognized `function_to_apply` argument: #{function_to_apply}"
      end

      dict_scores =
        scores.to_a.map.with_index do |score, i|
          {label: @model.config.id2label[i], score: score}
        end
      dict_scores.sort_by! { |x| -x[:score] }
      if !top_k.nil?
        dict_scores = dict_scores[...top_k]
      end

      dict_scores
    end

    private

    def sigmoid(_outputs)
      1.0 / (1.0 + Numo::NMath.exp(-_outputs))
    end

    def softmax(_outputs)
      maxes = _outputs.max(axis: -1, keepdims: true)
      shifted_exp = Numo::NMath.exp(_outputs - maxes)
      shifted_exp / shifted_exp.sum(axis: -1, keepdims: true)
    end
  end
end
