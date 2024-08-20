module Transformers
  class ImageFeatureExtractionPipeline < Pipeline
    def _sanitize_parameters(image_processor_kwargs: nil, return_tensors: nil, pool: nil, **kwargs)
      preprocess_params = image_processor_kwargs.nil? ? {} : image_processor_kwargs

      postprocess_params = {}
      if !pool.nil?
        postprocess_params[:pool] = pool
      end
      if !return_tensors.nil?
        postprocess_params[:return_tensors] = return_tensors
      end

      if kwargs.include?(:timeout)
        preprocess_params[:timeout] = kwargs[:timeout]
      end

      [preprocess_params, {}, postprocess_params]
    end

    def preprocess(image, timeout: nil, **image_processor_kwargs)
      image = ImageUtils.load_image(image, timeout: timeout)
      model_inputs = @image_processor.(image, return_tensors: @framework, **image_processor_kwargs)
      if @framework == "pt"
        # TODO
        # model_inputs = model_inputs.to(torch_dtype)
      end
      model_inputs
    end

    def _forward(model_inputs)
      model_outputs = @model.(**model_inputs)
      model_outputs
    end

    def postprocess(model_outputs, pool: nil, return_tensors: false)
      pool = !pool.nil? ? pool : false

      if pool
        raise Todo
      else
        # [0] is the first available tensor, logits or last_hidden_state.
        outputs = model_outputs[0]
      end

      if return_tensors
        return outputs
      end
      if @framework == "pt"
        outputs.to_a
      else
        raise Todo
      end
    end
  end
end
