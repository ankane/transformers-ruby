module Transformers
  module HfHub
    class HfHubHTTPError < Error
      def initialize(message, response = nil)
        super(message)
      end
    end

    class RepositoryNotFoundError < HfHubHTTPError; end

    class GatedRepoError < RepositoryNotFoundError; end

    class DisabledRepoError < HfHubHTTPError; end

    class RevisionNotFoundError < HfHubHTTPError; end

    class EntryNotFoundError < HfHubHTTPError; end

    class LocalEntryNotFoundError < EntryNotFoundError; end

    class BadRequestError < HfHubHTTPError; end

    class << self
      def hf_raise_for_status(response, endpoint_name: nil)
        begin
          response.value unless response.is_a?(Net::HTTPRedirection)
        rescue
          error_code = response["X-Error-Code"]
          error_message = response["X-Error-Message"]

          if error_code == "RevisionNotFound"
            message = "#{response.code} Client Error." + "\n\n" + "Revision Not Found for url: #{response.uri}."
            raise RevisionNotFoundError.new(message, response)

          elsif error_code == "EntryNotFound"
            message = "#{response.code} Client Error." + "\n\n" + "Entry Not Found for url: #{response.uri}."
            raise EntryNotFoundError.new(message, response)

          elsif error_code == "GatedRepo"
            message = (
              "#{response.code} Client Error." + "\n\n" + "Cannot access gated repo for url #{response.uri}."
            )
            raise GatedRepoError.new(message, response)

          elsif error_message == "Access to this resource is disabled."
            message = (
              "#{response.code} Client Error." +
              "\n\n" +
              "Cannot access repository for url #{response.uri}." +
              "\n" +
              "Access to this resource is disabled."
            )
            raise DisabledRepoError.new(message, response)

          elsif error_code == "RepoNotFound"
            # 401 is misleading as it is returned for:
            #    - private and gated repos if user is not authenticated
            #    - missing repos
            # => for now, we process them as `RepoNotFound` anyway.
            # See https://gist.github.com/Wauplin/46c27ad266b15998ce56a6603796f0b9
            message = (
              "#{response.code} Client Error." +
              "\n\n" +
              "Repository Not Found for url: #{response.uri}." +
              "\nPlease make sure you specified the correct `repo_id` and" +
              " `repo_type`.\nIf you are trying to access a private or gated repo," +
              " make sure you are authenticated."
            )
            raise RepositoryNotFoundError.new(message, response)

          elsif response.code.to_i == 400
            message = (
              !endpoint_name.nil? ? "\n\nBad request for #{endpoint_name} endpoint:" : "\n\nBad request:"
            )
            raise BadRequestError.new(message, response)

          elsif response.code.to_i == 403
            message = (
              "\n\n{response.code} Forbidden: #{error_message}." +
              "\nCannot access content at: #{response.uri}." +
              "\nIf you are trying to create or update content, " +
              "make sure you have a token with the `write` role."
            )
            raise HfHubHTTPError.new(message, response)
          end

          # Convert `HTTPError` into a `HfHubHTTPError` to display request information
          # as well (request id and/or server error message)
          raise HfHubHTTPError.new(e.to_s, response)
        end
      end
    end
  end
end
