# Copyright 2020 The HuggingFace Team. All rights reserved.
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
  module Utils
    module Hub
      IS_OFFLINE_MODE = HfHub::HF_HUB_OFFLINE

      PYTORCH_PRETRAINED_BERT_CACHE = ENV.fetch("PYTORCH_PRETRAINED_BERT_CACHE", HfHub::HF_HUB_CACHE)
      PYTORCH_TRANSFORMERS_CACHE = ENV.fetch("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
      TRANSFORMERS_CACHE = ENV.fetch("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

      DEFAULT_ENDPOINT = "https://huggingface.co"
      HUGGINGFACE_CO_RESOLVE_ENDPOINT = ENV.fetch("HF_ENDPOINT", DEFAULT_ENDPOINT)

      class << self
        def is_offline_mode
          IS_OFFLINE_MODE
        end

        def is_remote_url(url_or_filename)
          url_or_filename.is_a?(URI)
        end

        def http_user_agent(user_agent = nil)
          ua = "transformers.rb/#{Transformers::VERSION}; ruby/#{RUBY_VERSION.to_f}"
          if user_agent.is_a?(Hash)
            ua += "; " + user_agent.map { |k, v| "#{k}/#{v}" }.join("; ")
          elsif user_agent.is_a?(String)
            ua += "; " + user_agent
          end
          ua
        end

        def extract_commit_hash(resolved_file, commit_hash)
          if resolved_file.nil? || !commit_hash.nil?
            return commit_hash
          end
          search = /snapshots\/([^\/]+)/.match(resolved_file)
          if search.nil?
            return nil
          end
          commit_hash = search[1]
          HfHub::REGEX_COMMIT_HASH.match(commit_hash) ? commit_hash : nil
        end

        def cached_file(
          path_or_repo_id,
          filename,
          cache_dir: nil,
          force_download: false,
          resume_download: false,
          proxies: nil,
          token: nil,
          revision: nil,
          local_files_only: false,
          subfolder: "",
          repo_type: nil,
          user_agent: nil,
          _raise_exceptions_for_gated_repo: true,
          _raise_exceptions_for_missing_entries: true,
          _raise_exceptions_for_connection_errors: true,
          _commit_hash: nil,
          **deprecated_kwargs
        )
          if is_offline_mode && !local_files_only
            Transformers.logger.info "Offline mode: forcing local_files_only: true"
            local_files_only = true
          end
          if subfolder.nil?
            subfolder = ""
          end

          path_or_repo_id = path_or_repo_id.to_s
          full_filename = File.join(subfolder, filename)
          if Dir.exist?(path_or_repo_id)
            raise Todo
          end

          if cache_dir.nil?
            cache_dir = TRANSFORMERS_CACHE
          end
          if cache_dir.is_a?(Pathname)
            cache_dir = cache_dir.to_s
          end

          if !_commit_hash.nil? && !force_download
            # If the file is cached under that commit hash, we return it directly.
            resolved_file =
              HfHub.try_to_load_from_cache(
                path_or_repo_id, full_filename, cache_dir: cache_dir, revision: _commit_hash, repo_type: repo_type
              )
            if !resolved_file.nil?
              if resolved_file != HfHub::CACHED_NO_EXIST
                return resolved_file
              elsif !_raise_exceptions_for_missing_entries
                return nil
              else
                raise EnvironmentError, "Could not locate #{full_filename} inside #{path_or_repo_id}."
              end
            end
          end

          user_agent = http_user_agent(user_agent)

          resolved_file = nil
          begin
            resolved_file =
              HfHub.hf_hub_download(
                path_or_repo_id,
                filename,
                subfolder: subfolder.length == 0 ? nil : subfolder,
                repo_type: repo_type,
                revision: revision,
                cache_dir: cache_dir,
                user_agent: user_agent,
                force_download: force_download,
                proxies: proxies,
                resume_download: resume_download,
                token: token,
                local_files_only: local_files_only
              )
          rescue => e
            raise e if _raise_exceptions_for_missing_entries
          end
          resolved_file
        end

        def has_file(
          path_or_repo,
          filename,
          revision: nil,
          proxies: nil,
          token: nil,
          local_files_only: false,
          cache_dir: nil,
          repo_type: nil,
          **deprecated_kwargs
        )
          # If path to local directory, check if the file exists
          if Dir.exist?(path_or_repo)
            return File.exist?(File.join(path_or_repo, filename))
          end

          # Else it's a repo => let's check if the file exists in local cache or on the Hub

          # Check if file exists in cache
          # This information might be outdated so it's best to also make a HEAD call (if allowed).
          cached_path = HfHub.try_to_load_from_cache(
            path_or_repo,
            filename,
            revision: revision,
            repo_type: repo_type,
            cache_dir: cache_dir
          )
          has_file_in_cache = cached_path.is_a?(String)

          # If local_files_only, don't try the HEAD call
          if local_files_only
            return has_file_in_cache
          end

          # Check if the file exists
          begin
            HfHub._request_wrapper(
              "HEAD",
              HfHub.hf_hub_url(path_or_repo, filename, revision: revision, repo_type: repo_type),
              headers: HfHub.build_hf_headers(token: token, user_agent: http_user_agent),
              allow_redirects: false,
              proxies: proxies,
              timeout: 10
            )
            return true
          rescue HfHub::OfflineModeIsEnabled
            return has_file_in_cache
          rescue HfHub::GatedRepoError => e
            Transformers.logger.error(e)
            raise EnvironmentError,
              "#{path_or_repo} is a gated repository. Make sure to request access at " +
              "https://huggingface.co/#{path_or_repo} and pass a token having permission to this repo either by " +
              "logging in with `huggingface-cli login` or by passing `token=<your_token>`."
          rescue HfHub::RepositoryNotFoundError => e
            Transformers.logger.error(e)
            raise EnvironmentError,
              "#{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'."
          rescue HfHub::RevisionNotFoundError => e
            Transformers.logger.error(e)
            raise EnvironmentError,
              "#{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this " +
              "model name. Check the model page at 'https://huggingface.co/#{path_or_repo}' for available revisions."
          rescue HfHub::EntryNotFoundError
            return false  # File does not exist
          end
        end
      end
    end
  end
end
