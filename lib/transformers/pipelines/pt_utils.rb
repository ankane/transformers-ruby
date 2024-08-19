module Transformers
  class PipelineDataset < Torch::Utils::Data::Dataset
    def initialize(dataset, process, params)
      @dataset = dataset
      @process = process
      @params = params
    end

    def size
      @dataset.size
    end

    def [](i)
      item = @dataset[i]
      processed = @process.(item, **@params)
      processed
    end
  end

  class PipelineIterator < Torch::Utils::Data::IterableDataset
    def initialize(loader, infer, params, loader_batch_size: nil)
      @loader = loader
      @infer = infer
      @params = params
      if loader_batch_size == 1
        # Let's spare some time by deactivating altogether
        loader_batch_size = nil
      end
      @loader_batch_size = loader_batch_size

      # Internal bookkeeping
      @loader_batch_index = nil
      @loader_batch_data = nil
    end

    def size
      @loader.size
    end

    def [](i)
      @infer.(@loader[i], **@params)
    end

    def each
      @iterator = @loader

      @iterator.each do |item|
        processed = @infer.(item, **@params)
        yield processed
      end
    end
  end
end
