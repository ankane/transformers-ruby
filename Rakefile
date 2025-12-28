require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new do |t|
  t.pattern = FileList["test/**/*_test.rb"].exclude("test/model_test.rb")
end

task default: :test

def download_file(url)
  require "open-uri"

  file = File.basename(url)
  puts "Downloading #{file}..."
  dest = "test/support/#{file}"
  File.binwrite(dest, URI.parse(url).read)
  puts "Saved #{dest}"
end

namespace :download do
  task :files do
    Dir.mkdir("test/support") unless Dir.exist?("test/support")

    download_file("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
  end
end
