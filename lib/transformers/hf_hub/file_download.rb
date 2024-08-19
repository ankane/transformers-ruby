module Transformers
  module HfHub
    # Return value when trying to load a file from cache but the file does not exist in the distant repo.
    CACHED_NO_EXIST = Object.new

    # Regex to get filename from a "Content-Disposition" header for CDN-served files
    HEADER_FILENAME_PATTERN = /filename="(?<filename>.*?)";/

    # Regex to check if the revision IS directly a commit_hash
    REGEX_COMMIT_HASH = /^[0-9a-f]{40}$/

    class HfFileMetadata
      attr_reader :commit_hash, :etag, :location, :size

      def initialize(commit_hash:, etag:, location:, size:)
        @commit_hash = commit_hash
        @etag = etag
        @location = location
        @size = size
      end
    end

    class << self
      def hf_hub_url(
        repo_id,
        filename,
        subfolder: nil,
        repo_type: nil,
        revision: nil,
        endpoint: nil
      )
        if subfolder == ""
          subfolder = nil
        end
        if !subfolder.nil?
          filename = "#{subfolder}/#{filename}"
        end

        if !REPO_TYPES.include?(repo_type)
          raise ArgumentError, "Invalid repo type"
        end

        if REPO_TYPES_URL_PREFIXES.include?(repo_type)
          repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id
        end

        if revision.nil?
          revision = DEFAULT_REVISION
        end
        url =
          HUGGINGFACE_CO_URL_TEMPLATE %
          {repo_id: repo_id, revision: CGI.escape(revision), filename: CGI.escape(filename)}
        # Update endpoint if provided
        if !endpoint.nil? && url.start_with?(ENDPOINT)
          url = endpoint + url[ENDPOINT.length..]
        end
        url
      end

      def _request_wrapper(method, url, follow_relative_redirects: false, redirects: 0, **params)
        # Recursively follow relative redirects
        if follow_relative_redirects
          if redirects > 10
            raise "Too many redirects"
          end

          response = _request_wrapper(
            method,
            url,
            follow_relative_redirects: false,
            **params
          )

          # If redirection, we redirect only relative paths.
          # This is useful in case of a renamed repository.
          if response.is_a?(Net::HTTPRedirection)
            parsed_target = URI.parse(response["Location"])
            if netloc(parsed_target) == ""
              # This means it is a relative 'location' headers, as allowed by RFC 7231.
              # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
              # We want to follow this relative redirect !
              #
              # Highly inspired by `resolve_redirects` from requests library.
              # See https://github.com/psf/requests/blob/main/requests/sessions.py#L159
              next_url = URI.parse(url)
              next_url.path = parsed_target.path
              return _request_wrapper(method, next_url, follow_relative_redirects: true, redirects: redirects + 1, **params)
            end
          end
          return response
        end

        # Perform request and return if status_code is not in the retry list.
        uri = URI.parse(url)

        http_options = {use_ssl: true}
        if params[:timeout]
          http_options[:open_timeout] = params[:timeout]
          http_options[:read_timeout] = params[:timeout]
          http_options[:write_timeout] = params[:timeout]
        end
        response =
          Net::HTTP.start(uri.host, uri.port, **http_options) do |http|
            http.send_request(method, uri.path, nil, params[:headers])
          end
        response.uri ||= uri
        hf_raise_for_status(response)
        response
      end

      def http_get(
        url,
        temp_file,
        proxies: nil,
        resume_size: 0,
        headers: nil,
        expected_size: nil,
        displayed_filename: nil,
        _nb_retries: 5
      )
        uri = URI.parse(url)

        if resume_size > 0
          headers["range"] = "bytes=%d-" % [resume_size]
        end

        size = resume_size
        Net::HTTP.start(uri.host, uri.port, use_ssl: true) do |http|
          request = Net::HTTP::Get.new(uri)
          headers.each do |k, v|
            request[k] = v
          end
          http.request(request) do |response|
            case response
            when Net::HTTPSuccess
              if displayed_filename.nil?
                displayed_filename = url
                content_disposition = response["content-disposition"]
                if !content_disposition.nil?
                  match = HEADER_FILENAME_PATTERN.match(content_disposition)
                  if !match.nil?
                    # Means file is on CDN
                    displayed_filename = match["filename"]
                  end
                end
              end

              stream = STDERR
              tty = stream.tty?
              width = tty ? stream.winsize[1] : 80

              response.read_body do |chunk|
                temp_file.write(chunk)
                size += chunk.bytesize

                if tty
                  stream.print "\r#{display_progress(displayed_filename, width, size, expected_size)}"
                end
              end

              if tty
                stream.puts
              else
                stream.puts display_progress(displayed_filename, width, size, expected_size)
              end
            else
              hf_raise_for_status(response)
            end
          end
        end
      end

      def _normalize_etag(etag)
        if etag.nil?
          return nil
        end
        etag.sub(/\A\W/, "").delete('"')
      end

      def _create_symlink(src, dst, new_blob: false)
        begin
          FileUtils.rm(dst)
        rescue Errno::ENOENT
          # do nothing
        end

        # abs_src = File.absolute_path(File.expand_path(src))
        # abs_dst = File.absolute_path(File.expand_path(dst))
        # abs_dst_folder = File.dirname(abs_dst)

        FileUtils.symlink(src, dst)
      end

      def _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)
        if revision != commit_hash
          ref_path = Pathname.new(storage_folder) / "refs" / revision
          ref_path.parent.mkpath
          if !ref_path.exist? || commit_hash != ref_path.read
            # Update ref only if has been updated. Could cause useless error in case
            # repo is already cached and user doesn't have write access to cache folder.
            # See https://github.com/huggingface/huggingface_hub/issues/1216.
            ref_path.write(commit_hash)
          end
        end
      end

      def repo_folder_name(repo_id:, repo_type:)
        # remove all `/` occurrences to correctly convert repo to directory name
        parts = ["#{repo_type}s"] + repo_id.split("/")
        parts.join(REPO_ID_SEPARATOR)
      end

      def _check_disk_space(expected_size, target_dir)
        # TODO
      end

      def hf_hub_download(
        repo_id,
        filename,
        subfolder: nil,
        repo_type: nil,
        revision: nil,
        library_name: nil,
        library_version: nil,
        cache_dir: nil,
        local_dir: nil,
        local_dir_use_symlinks: "auto",
        user_agent: nil,
        force_download: false,
        force_filename: nil,
        proxies: nil,
        etag_timeout: DEFAULT_ETAG_TIMEOUT,
        resume_download: false,
        token: nil,
        local_files_only: false,
        legacy_cache_layout: false,
        endpoint: nil
      )
        if cache_dir.nil?
          cache_dir = HF_HUB_CACHE
        end
        if revision.nil?
          revision = DEFAULT_REVISION
        end

        if subfolder == ""
          subfolder = nil
        end
        if !subfolder.nil?
          # This is used to create a URL, and not a local path, hence the forward slash.
          filename = "#{subfolder}/#{filename}"
        end

        if repo_type.nil?
          repo_type = "model"
        end
        if !REPO_TYPES.include?(repo_type)
          raise ArgumentError, "Invalid repo type: #{repo_type}. Accepted repo types are: #{REPO_TYPES}"
        end

        headers =
          build_hf_headers(
            token: token,
            library_name: library_name,
            library_version: library_version,
            user_agent: user_agent
          )

        if !local_dir.nil?
          raise Todo
        else
          _hf_hub_download_to_cache_dir(
            # Destination
            cache_dir: cache_dir,
            # File info
            repo_id: repo_id,
            filename: filename,
            repo_type: repo_type,
            revision: revision,
            # HTTP info
            endpoint: endpoint,
            etag_timeout: etag_timeout,
            headers: headers,
            proxies: proxies,
            token: token,
            # Additional options
            local_files_only: local_files_only,
            force_download: force_download
          )
        end
      end

      def _hf_hub_download_to_cache_dir(
        cache_dir:,
        # File info
        repo_id:,
        filename:,
        repo_type:,
        revision:,
        # HTTP info
        endpoint:,
        etag_timeout:,
        headers:,
        proxies:,
        token:,
        # Additional options
        local_files_only:,
        force_download:
      )
        _locks_dir = File.join(cache_dir, ".locks")
        storage_folder = File.join(cache_dir, repo_folder_name(repo_id: repo_id, repo_type: repo_type))

        # cross platform transcription of filename, to be used as a local file path.
        relative_filename = File.join(*filename.split("/"))

        # if user provides a commit_hash and they already have the file on disk, shortcut everything.
        if REGEX_COMMIT_HASH.match?(revision)
          pointer_path = _get_pointer_path(storage_folder, revision, relative_filename)
          if File.exist?(pointer_path) && !force_download
            return pointer_path
          end
        end

        # Try to get metadata (etag, commit_hash, url, size) from the server.
        # If we can't, a HEAD request error is returned.
        url_to_download, etag, commit_hash, expected_size, head_call_error = _get_metadata_or_catch_error(
          repo_id: repo_id,
          filename: filename,
          repo_type: repo_type,
          revision: revision,
          endpoint: endpoint,
          proxies: proxies,
          etag_timeout: etag_timeout,
          headers: headers,
          token: token,
          local_files_only: local_files_only,
          storage_folder: storage_folder,
          relative_filename: relative_filename
        )

        # etag can be None for several reasons:
        # 1. we passed local_files_only.
        # 2. we don't have a connection
        # 3. Hub is down (HTTP 500 or 504)
        # 4. repo is not found -for example private or gated- and invalid/missing token sent
        # 5. Hub is blocked by a firewall or proxy is not set correctly.
        # => Try to get the last downloaded one from the specified revision.
        #
        # If the specified revision is a commit hash, look inside "snapshots".
        # If the specified revision is a branch or tag, look inside "refs".
        if !head_call_error.nil?
          # Couldn't make a HEAD call => let's try to find a local file
          if !force_download
            commit_hash = nil
            if REGEX_COMMIT_HASH.match(revision)
              commit_hash = revision
            else
              ref_path = File.join(storage_folder, "refs", revision)
              if File.exist?(ref_path)
                commit_hash = File.read(ref_path)
              end
            end

            # Return pointer file if exists
            if !commit_hash.nil?
              pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)
              if File.exist?(pointer_path) && !force_download
                return pointer_path
              end
            end
          end

          # Otherwise, raise appropriate error
          _raise_on_head_call_error(head_call_error, force_download, local_files_only)
        end

        # From now on, etag and commit_hash are not None.
        raise "etag must have been retrieved from server" if etag.nil?
        raise "commit_hash must have been retrieved from server" if commit_hash.nil?
        raise "file location must have been retrieved from server" if url_to_download.nil?
        raise "expected_size must have been retrieved from server" if expected_size.nil?
        blob_path = File.join(storage_folder, "blobs", etag)
        pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)

        FileUtils.mkdir_p(File.dirname(blob_path))
        FileUtils.mkdir_p(File.dirname(pointer_path))

        # if passed revision is not identical to commit_hash
        # then revision has to be a branch name or tag name.
        # In that case store a ref.
        _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)

        if !force_download
          if File.exist?(pointer_path)
            return pointer_path
          end

          if File.exist?(blob_path)
            # we have the blob already, but not the pointer
            _create_symlink(blob_path, pointer_path, new_blob: false)
            return pointer_path
          end
        end

        # Prevent parallel downloads of the same file with a lock.
        # etag could be duplicated across repos,
        # lock_path = File.join(locks_dir, repo_folder_name(repo_id: repo_id, repo_type: repo_type), "#{etag}.lock")

        _download_to_tmp_and_move(
          incomplete_path: Pathname.new(blob_path + ".incomplete"),
          destination_path: Pathname.new(blob_path),
          url_to_download: url_to_download,
          proxies: proxies,
          headers: headers,
          expected_size: expected_size,
          filename: filename,
          force_download: force_download
        )
        _create_symlink(blob_path, pointer_path, new_blob: true)

        pointer_path
      end

      def try_to_load_from_cache(
        repo_id,
        filename,
        cache_dir: nil,
        revision: nil,
        repo_type: nil
      )
        if revision.nil?
          revision = "main"
        end
        if repo_type.nil?
          repo_type = "model"
        end
        if !REPO_TYPES.include?(repo_type)
          raise ArgumentError, "Invalid repo type: #{repo_type}. Accepted repo types are: #{REPO_TYPES}"
        end
        if cache_dir.nil?
          cache_dir = HF_HUB_CACHE
        end

        object_id = repo_id.gsub("/", "--")
        repo_cache = File.join(cache_dir, "#{repo_type}s--#{object_id}")
        if !Dir.exist?(repo_cache)
          # No cache for this model
          return nil
        end

        refs_dir = File.join(repo_cache, "refs")
        snapshots_dir = File.join(repo_cache, "snapshots")
        no_exist_dir = File.join(repo_cache, ".no_exist")

        # Resolve refs (for instance to convert main to the associated commit sha)
        if Dir.exist?(refs_dir)
          revision_file = File.join(refs_dir, revision)
          if File.exist?(revision_file)
            revision = File.read(revision_file)
          end
        end

        # Check if file is cached as "no_exist"
        if File.exist?(File.join(no_exist_dir, revision, filename))
          return CACHED_NO_EXIST
        end

        # Check if revision folder exists
        if !Dir.exist?(snapshots_dir)
          return nil
        end
        cached_shas = Dir.glob("*", base: snapshots_dir)
        if !cached_shas.include?(revision)
          # No cache for this revision and we won't try to return a random revision
          return nil
        end

        # Check if file exists in cache
        cached_file = File.join(snapshots_dir, revision, filename)
        File.exist?(cached_file) ? cached_file : nil
      end

      def get_hf_file_metadata(
        url,
        token: nil,
        proxies: nil,
        timeout: DEFAULT_REQUEST_TIMEOUT,
        library_name: nil,
        library_version: nil,
        user_agent: nil,
        headers: nil
      )
        headers =
          build_hf_headers(
            token: token,
            library_name: library_name,
            library_version: library_version,
            user_agent: user_agent,
            headers: headers
          )
        headers["Accept-Encoding"] = "identity" # prevent any compression => we want to know the real size of the file

        # Retrieve metadata
        r =
          _request_wrapper(
            "HEAD",
            url,
            headers: headers,
            allow_redirects: false,
            follow_relative_redirects: true,
            proxies: proxies,
            timeout: timeout
          )
        hf_raise_for_status(r)

        # Return
        HfFileMetadata.new(
          commit_hash: r[HUGGINGFACE_HEADER_X_REPO_COMMIT],
          # We favor a custom header indicating the etag of the linked resource, and
          # we fallback to the regular etag header.
          etag: _normalize_etag(r[HUGGINGFACE_HEADER_X_LINKED_ETAG] || r["etag"]),
          # Either from response headers (if redirected) or defaults to request url
          # Do not use directly `url`, as `_request_wrapper` might have followed relative
          # redirects.
          location: r["location"] || r.uri.to_s,
          size: _int_or_none(r[HUGGINGFACE_HEADER_X_LINKED_SIZE] || r["content-length"])
        )
      end

      def _get_metadata_or_catch_error(
        repo_id:,
        filename:,
        repo_type:,
        revision:,
        endpoint:,
        proxies:,
        etag_timeout:,
        headers:,  # mutated inplace!
        token:,
        local_files_only:,
        relative_filename: nil,  # only used to store `.no_exists` in cache
        storage_folder: nil  # only used to store `.no_exists` in cache
      )
        if local_files_only
          return [
            nil,
            nil,
            nil,
            nil,
            OfflineModeIsEnabled.new(
              "Cannot access file since 'local_files_only: true' as been set. (repo_id: #{repo_id}, repo_type: #{repo_type}, revision: #{revision}, filename: #{filename})"
            )
          ]
        end

        url = hf_hub_url(repo_id, filename, repo_type: repo_type, revision: revision, endpoint: endpoint)
        url_to_download = url
        etag = nil
        commit_hash = nil
        expected_size = nil
        head_error_call = nil

        if !local_files_only
          metadata = nil
          begin
            metadata =
              get_hf_file_metadata(
                url,
                proxies: proxies,
                timeout: etag_timeout,
                headers: headers,
                token: token
              )
          rescue => e
            raise e
            raise Todo
          end

          # Commit hash must exist
          commit_hash = metadata.commit_hash
          if commit_hash.nil?
            raise Todo
          end

           # Etag must exist
          etag = metadata.etag
          if etag.nil?
            raise Todo
          end

          # Expected (uncompressed) size
          expected_size = metadata.size
          if expected_size.nil?
            raise Todo
          end

          if metadata.location != url
            url_to_download = metadata.location
            if netloc(URI.parse(url)) != netloc(URI.parse(metadata.location))
              # Remove authorization header when downloading a LFS blob
              headers.delete("authorization")
            end
          end
        end

        if !(local_files_only || !etag.nil? || !head_call_error.nil?)
          raise "etag is empty due to uncovered problems"
        end

        [url_to_download, etag, commit_hash, expected_size, head_error_call]
      end

      def _raise_on_head_call_error(head_call_error, force_download, local_files_only)
        # No head call => we cannot force download.
        if force_download
          if local_files_only
            raise ArgumentError, "Cannot pass 'force_download: true' and 'local_files_only: true' at the same time."
          elsif head_call_error.is_a?(OfflineModeIsEnabled)
            raise ArgumentError, "Cannot pass 'force_download: true' when offline mode is enabled."
          else
            raise ArgumentError, "Force download failed due to the above error."
          end
        end

        # If we couldn't find an appropriate file on disk, raise an error.
        # If files cannot be found and local_files_only=True,
        # the models might've been found if local_files_only=False
        # Notify the user about that
        if local_files_only
          raise LocalEntryNotFoundError,
            "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable" +
            " hf.co look-ups and downloads online, set 'local_files_only' to false."
        elsif head_call_error.is_a?(RepositoryNotFoundError) || head_call_error.is_a?(GatedRepoError)
          # Repo not found or gated => let's raise the actual error
          raise head_call_error
        else
          # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
          raise LocalEntryNotFoundError,
            "An error happened while trying to locate the file on the Hub and we cannot find the requested files" +
            " in the local cache. Please check your connection and try again or make sure your Internet connection" +
            " is on."
        end
      end

      def _download_to_tmp_and_move(
        incomplete_path:,
        destination_path:,
        url_to_download:,
        proxies:,
        headers:,
        expected_size:,
        filename:,
        force_download:
      )
        if destination_path.exist? && !force_download
          # Do nothing if already exists (except if force_download=True)
          return
        end

        if incomplete_path.exist? && (force_download || (HF_HUB_ENABLE_HF_TRANSFER && !proxies))
          # By default, we will try to resume the download if possible.
          # However, if the user has set `force_download=True` or if `hf_transfer` is enabled, then we should
          # not resume the download => delete the incomplete file.
          message = "Removing incomplete file '#{incomplete_path}'"
          if force_download
            message += " (force_download: true)"
          elsif HF_HUB_ENABLE_HF_TRANSFER && !proxies
            message += " (hf_transfer: true)"
          end
          Transformers.logger.info(message)
          incomplete_path.unlink #(missing_ok=True)
        end

        incomplete_path.open("ab") do |f|
          f.seek(0, IO::SEEK_END)
          resume_size = f.tell
          message = "Downloading '#{filename}' to '#{incomplete_path}'"
          if resume_size > 0 && !expected_size.nil?
            message += " (resume from #{resume_size}/#{expected_size})"
          end
          Transformers.logger.info(message)

          if !expected_size.nil?  # might be None if HTTP header not set correctly
            # Check disk space in both tmp and destination path
            _check_disk_space(expected_size, incomplete_path.parent)
            _check_disk_space(expected_size, destination_path.parent)
          end

          http_get(
            url_to_download,
            f,
            proxies: proxies,
            resume_size: resume_size,
            headers: headers,
            expected_size: expected_size,
          )
        end

        Transformers.logger.info("Download complete. Moving file to #{destination_path}")
        _chmod_and_move(incomplete_path, destination_path)
      end

      def _int_or_none(value)
        value&.to_i
      end

      def _chmod_and_move(src, dst)
        tmp_file = dst.parent.parent / "tmp_#{SecureRandom.uuid}"
        begin
          FileUtils.touch(tmp_file)
          cache_dir_mode = Pathname.new(tmp_file).stat.mode
          src.chmod(cache_dir_mode)
        ensure
          begin
            tmp_file.unlink
          rescue Errno::ENOENT
            # fails if `tmp_file.touch()` failed => do nothing
            # See https://github.com/huggingface/huggingface_hub/issues/2359
          end
        end

        FileUtils.move(src.to_s, dst.to_s)
      end

      def _get_pointer_path(storage_folder, revision, relative_filename)
        snapshot_path = File.join(storage_folder, "snapshots")
        pointer_path = File.join(snapshot_path, revision, relative_filename)
        if !parents(Pathname.new(File.absolute_path(pointer_path))).include?(Pathname.new(File.absolute_path(snapshot_path)))
          raise ArgumentError,
            "Invalid pointer path: cannot create pointer path in snapshot folder if" +
            " `storage_folder: #{storage_folder.inspect}`, `revision: #{revision.inspect}` and" +
            " `relative_filename: #{relative_filename.inspect}`."
        end
        pointer_path
      end

      # additional methods

      def netloc(uri)
        [uri.host, uri.port].compact.join(":")
      end

      def parents(path)
        parents = []
        100.times do
          if path == path.parent
            break
          end
          path = path.parent
          parents << path
        end
        parents
      end

      def display_progress(filename, width, size, expected_size)
        bar_width = width - (filename.length + 3)
        progress = size / expected_size.to_f
        done = (progress * bar_width).round
        not_done = bar_width - done
        "#{filename} |#{"█" * done}#{" " * not_done}|"
      end
    end
  end
end
