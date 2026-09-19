#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kVersion{"0.1.0"};

struct ServerOptions {
  std::string host{"127.0.0.1"};
  unsigned short port{8080};
};

struct Response {
  int status_code;
  std::string_view reason;
  std::string_view body;
  std::string_view extra_headers{};
};

[[nodiscard]] std::optional<unsigned short> parse_port(std::string_view value) {
  unsigned int parsed = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() || parsed > 65535) {
    return std::nullopt;
  }
  return static_cast<unsigned short>(parsed);
}

void print_usage(std::ostream& output) {
  output << "Usage: veritas-inference [--host IPV4_ADDRESS] [--port PORT]\n"
         << "       veritas-inference --version\n";
}

[[nodiscard]] std::optional<ServerOptions> parse_options(int argc, char* argv[]) {
  ServerOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help") {
      print_usage(std::cout);
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--version") {
      if (argc != 2) {
        return std::nullopt;
      }
      std::cout << "veritas-inference " << kVersion << '\n';
      std::exit(EXIT_SUCCESS);
    }
    if ((argument != "--host" && argument != "--port") || index + 1 >= argc) {
      return std::nullopt;
    }

    const std::string_view value{argv[++index]};
    if (argument == "--host") {
      options.host = value;
    } else {
      const auto port = parse_port(value);
      if (!port.has_value()) {
        return std::nullopt;
      }
      options.port = *port;
    }
  }
  return options;
}

[[nodiscard]] int create_listener(const ServerOptions& options, unsigned short& bound_port) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(options.port);
  if (inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
    std::cerr << "Invalid IPv4 host: " << options.host << '\n';
    return -1;
  }

  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) {
    std::perror("Unable to create HTTP listener");
    return -1;
  }

  const int enabled = 1;
  if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0 ||
      bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
      listen(listener, SOMAXCONN) != 0) {
    std::perror("Unable to start HTTP listener");
    close(listener);
    return -1;
  }

  sockaddr_in bound_address{};
  socklen_t address_length = sizeof(bound_address);
  if (getsockname(listener, reinterpret_cast<sockaddr*>(&bound_address), &address_length) != 0) {
    std::perror("Unable to determine HTTP listener port");
    close(listener);
    return -1;
  }
  bound_port = ntohs(bound_address.sin_port);
  return listener;
}

[[nodiscard]] Response route_request(std::string_view request) {
  const auto line_end = request.find("\r\n");
  const std::string_view request_line = request.substr(0, line_end);
  const auto first_space = request_line.find(' ');
  const auto second_space = request_line.find(' ', first_space + 1);
  if (first_space == std::string_view::npos || second_space == std::string_view::npos) {
    return {400, "Bad Request", R"({"error":{"code":"invalid_request","message":"A valid HTTP request line is required."}})"};
  }

  const std::string_view method = request_line.substr(0, first_space);
  const std::string_view path = request_line.substr(first_space + 1, second_space - first_space - 1);
  if (path == "/v1/infer") {
    if (method != "POST") {
      return {405, "Method Not Allowed", R"({"error":{"code":"method_not_allowed","message":"Only POST is supported by this endpoint."}})", "Allow: POST\r\n"};
    }
    return {503, "Service Unavailable", R"({"error":{"code":"model_unavailable","message":"The synthetic portrait classifier is not loaded."}})"};
  }
  if (method != "GET") {
    return {405, "Method Not Allowed", R"({"error":{"code":"method_not_allowed","message":"Only GET is supported by this endpoint."}})", "Allow: GET\r\n"};
  }

  if (path == "/health" || path == "/healthz") {
    return {200, "OK", R"({"status":"ok","service":"veritas-face-inference","version":"0.1.0"})"};
  }
  if (path == "/v1/model-info") {
    return {200,
            "OK",
            R"({"service":"veritas-face-inference","version":"0.1.0","model":{"id":"synthetic-portrait-classifier","version":"not_loaded","status":"unavailable"},"input":{"color_space":"RGB","width":224,"height":224,"channels":3,"layout":"HWC","value_range":"0_to_255","normalization":"not_configured"}})"};
  }
  return {404, "Not Found", R"({"error":{"code":"not_found","message":"The requested endpoint does not exist."}})"};
}

void write_response(int client, const Response& response) {
  std::string headers = "HTTP/1.1 " + std::to_string(response.status_code) + " " +
                        std::string(response.reason) + "\r\n"
                        "Content-Type: application/json\r\n"
                        "Cache-Control: no-store\r\n"
                        "Connection: close\r\n" +
                        std::string(response.extra_headers) + "Content-Length: " +
                        std::to_string(response.body.size()) + "\r\n\r\n";
  headers += response.body;

  std::size_t sent = 0;
  while (sent < headers.size()) {
    const ssize_t count = send(client, headers.data() + sent, headers.size() - sent, 0);
    if (count <= 0) {
      return;
    }
    sent += static_cast<std::size_t>(count);
  }
}

void serve_client(int client) {
  char buffer[8192];
  const ssize_t received = recv(client, buffer, sizeof(buffer), 0);
  if (received > 0) {
    write_response(client, route_request(std::string_view{buffer, static_cast<std::size_t>(received)}));
  }
  close(client);
}

}  // namespace

int main(int argc, char* argv[]) {
  const auto options = parse_options(argc, argv);
  if (!options.has_value()) {
    print_usage(std::cerr);
    return 64;
  }

  std::signal(SIGPIPE, SIG_IGN);
  unsigned short bound_port = 0;
  const int listener = create_listener(*options, bound_port);
  if (listener < 0) {
    return EXIT_FAILURE;
  }

  std::cout << "veritas-inference listening on http://" << options->host << ':' << bound_port << '\n'
            << std::flush;
  while (true) {
    const int client = accept(listener, nullptr, nullptr);
    if (client < 0) {
      if (errno == EINTR) {
        continue;
      }
      std::perror("Unable to accept HTTP connection");
      close(listener);
      return EXIT_FAILURE;
    }
    serve_client(client);
  }
}
