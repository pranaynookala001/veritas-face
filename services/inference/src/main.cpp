#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "face_preprocessor.hpp"

namespace {

constexpr std::string_view kVersion{"0.1.0"};
constexpr std::size_t kMaximumHeaderBytes = 16 * 1024;
constexpr std::size_t kMaximumRequestBodyBytes = 512 * 1024;

struct ModelOptions {
  std::string path;
  std::string id{"synthetic-portrait-classifier"};
  std::string version;
  unsigned int threads{1};
};

struct ServerOptions {
  std::string host{"127.0.0.1"};
  unsigned short port{8080};
  std::optional<ModelOptions> model;
};

struct Response {
  int status_code;
  std::string_view reason;
  std::string body;
  std::string extra_headers{};
};

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
};

[[nodiscard]] std::optional<unsigned int> parse_unsigned(std::string_view value,
                                                          unsigned int maximum) {
  if (value.empty()) {
    return std::nullopt;
  }
  unsigned int parsed = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() || parsed > maximum) {
    return std::nullopt;
  }
  return parsed;
}

[[nodiscard]] std::optional<unsigned short> parse_port(std::string_view value) {
  const auto parsed = parse_unsigned(value, 65535);
  if (!parsed.has_value()) {
    return std::nullopt;
  }
  return static_cast<unsigned short>(*parsed);
}

void print_usage(std::ostream& output) {
  output << "Usage: veritas-inference [--host IPV4_ADDRESS] [--port PORT]\n"
         << "       veritas-inference --model MODEL.onnx --model-version VERSION "
            "[--model-id ID] [--threads COUNT] [--host IPV4_ADDRESS] [--port PORT]\n"
         << "       veritas-inference --version\n";
}

[[nodiscard]] std::optional<ServerOptions> parse_options(int argc, char* argv[]) {
  ServerOptions options;
  std::optional<std::string> model_path;
  std::optional<std::string> model_version;
  std::optional<std::string> model_id;
  unsigned int model_threads = 1;

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
    if ((argument != "--host" && argument != "--port" && argument != "--model" &&
         argument != "--model-id" && argument != "--model-version" && argument != "--threads") ||
        index + 1 >= argc) {
      return std::nullopt;
    }

    const std::string_view value{argv[++index]};
    if (value.empty()) {
      return std::nullopt;
    }
    if (argument == "--host") {
      options.host = value;
    } else if (argument == "--port") {
      const auto port = parse_port(value);
      if (!port.has_value()) {
        return std::nullopt;
      }
      options.port = *port;
    } else if (argument == "--model") {
      if (model_path.has_value()) {
        return std::nullopt;
      }
      model_path = value;
    } else if (argument == "--model-version") {
      if (model_version.has_value()) {
        return std::nullopt;
      }
      model_version = value;
    } else if (argument == "--model-id") {
      if (model_id.has_value()) {
        return std::nullopt;
      }
      model_id = value;
    } else {
      const auto threads = parse_unsigned(value, 128);
      if (!threads.has_value() || *threads == 0) {
        return std::nullopt;
      }
      model_threads = *threads;
    }
  }

  if (model_path.has_value() != model_version.has_value() ||
      (!model_path.has_value() && (model_id.has_value() || model_threads != 1))) {
    return std::nullopt;
  }
  if (model_path.has_value()) {
    options.model = ModelOptions{
        .path = std::move(*model_path),
        .id = model_id.value_or("synthetic-portrait-classifier"),
        .version = std::move(*model_version),
        .threads = model_threads,
    };
  }
  return options;
}

[[nodiscard]] std::string json_quote(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char character : value) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20U) {
          constexpr std::array<char, 17> kHex{"0123456789abcdef"};
          escaped += "\\u00";
          escaped.push_back(kHex[(static_cast<unsigned char>(character) >> 4U) & 0x0FU]);
          escaped.push_back(kHex[static_cast<unsigned char>(character) & 0x0FU]);
        } else {
          escaped.push_back(character);
        }
    }
  }
  escaped.push_back('"');
  return escaped;
}

[[nodiscard]] Response error_response(int status_code, std::string_view reason, std::string_view code,
                                       std::string_view message, std::string_view allow = {}) {
  std::string body = "{\"error\":{\"code\":" + json_quote(code) +
                     ",\"message\":" + json_quote(message) + "}}";
  std::string headers;
  if (!allow.empty()) {
    headers = "Allow: " + std::string(allow) + "\r\n";
  }
  return {status_code, reason, std::move(body), std::move(headers)};
}

class JsonValue final {
 public:
  enum class Type { string, number, object };

  Type type{};
  std::string string_value;
  std::uint64_t number_value{};
  std::map<std::string, JsonValue, std::less<>> object_value;
};

class JsonParser final {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  [[nodiscard]] std::optional<JsonValue> parse() {
    skip_whitespace();
    const auto value = parse_value(0);
    if (!value.has_value()) {
      return std::nullopt;
    }
    skip_whitespace();
    if (position_ != input_.size()) {
      return std::nullopt;
    }
    return value;
  }

 private:
  void skip_whitespace() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\n' || input_[position_] == '\r' ||
            input_[position_] == '\t')) {
      ++position_;
    }
  }

  [[nodiscard]] std::optional<JsonValue> parse_value(unsigned int depth) {
    if (depth > 4 || position_ >= input_.size()) {
      return std::nullopt;
    }
    if (input_[position_] == '{') {
      return parse_object(depth + 1);
    }
    if (input_[position_] == '"') {
      const auto string = parse_string();
      if (!string.has_value()) {
        return std::nullopt;
      }
      return JsonValue{.type = JsonValue::Type::string, .string_value = std::move(*string)};
    }
    return parse_number();
  }

  [[nodiscard]] std::optional<JsonValue> parse_object(unsigned int depth) {
    ++position_;  // Opening brace.
    skip_whitespace();
    JsonValue object{.type = JsonValue::Type::object};
    if (position_ < input_.size() && input_[position_] == '}') {
      ++position_;
      return object;
    }

    while (position_ < input_.size()) {
      if (input_[position_] != '"') {
        return std::nullopt;
      }
      const auto key = parse_string();
      if (!key.has_value()) {
        return std::nullopt;
      }
      skip_whitespace();
      if (position_ >= input_.size() || input_[position_++] != ':') {
        return std::nullopt;
      }
      skip_whitespace();
      const auto value = parse_value(depth);
      if (!value.has_value() || !object.object_value.emplace(*key, std::move(*value)).second) {
        return std::nullopt;
      }
      skip_whitespace();
      if (position_ >= input_.size()) {
        return std::nullopt;
      }
      const char delimiter = input_[position_++];
      if (delimiter == '}') {
        return object;
      }
      if (delimiter != ',') {
        return std::nullopt;
      }
      skip_whitespace();
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string> parse_string() {
    if (position_ >= input_.size() || input_[position_++] != '"') {
      return std::nullopt;
    }
    std::string output;
    while (position_ < input_.size()) {
      const char character = input_[position_++];
      if (character == '"') {
        return output;
      }
      if (static_cast<unsigned char>(character) < 0x20U) {
        return std::nullopt;
      }
      if (character != '\\') {
        output.push_back(character);
        continue;
      }
      if (position_ >= input_.size()) {
        return std::nullopt;
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          output.push_back(escaped);
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        default:
          return std::nullopt;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<JsonValue> parse_number() {
    const std::size_t start = position_;
    while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
      ++position_;
    }
    if (start == position_ || (position_ - start > 1 && input_[start] == '0')) {
      return std::nullopt;
    }
    std::uint64_t number = 0;
    const auto [end, error] = std::from_chars(input_.data() + start, input_.data() + position_, number);
    if (error != std::errc{} || end != input_.data() + position_) {
      return std::nullopt;
    }
    return JsonValue{.type = JsonValue::Type::number, .number_value = number};
  }

  std::string_view input_;
  std::size_t position_{};
};

[[nodiscard]] const JsonValue* object_field(const JsonValue& object, std::string_view name) {
  if (object.type != JsonValue::Type::object) {
    return nullptr;
  }
  const auto field = object.object_value.find(name);
  return field == object.object_value.end() ? nullptr : &field->second;
}

[[nodiscard]] std::optional<std::string> string_field(const JsonValue& object, std::string_view name) {
  const JsonValue* field = object_field(object, name);
  if (field == nullptr || field->type != JsonValue::Type::string) {
    return std::nullopt;
  }
  return field->string_value;
}

[[nodiscard]] std::optional<std::size_t> size_field(const JsonValue& object, std::string_view name) {
  const JsonValue* field = object_field(object, name);
  if (field == nullptr || field->type != JsonValue::Type::number ||
      field->number_value > std::numeric_limits<std::size_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(field->number_value);
}

[[nodiscard]] int base64_value(char character) {
  if (character >= 'A' && character <= 'Z') {
    return character - 'A';
  }
  if (character >= 'a' && character <= 'z') {
    return character - 'a' + 26;
  }
  if (character >= '0' && character <= '9') {
    return character - '0' + 52;
  }
  if (character == '+') {
    return 62;
  }
  if (character == '/') {
    return 63;
  }
  return -1;
}

[[nodiscard]] std::optional<std::vector<std::uint8_t>> decode_base64(std::string_view encoded) {
  if (encoded.empty() || encoded.size() % 4 != 0) {
    return std::nullopt;
  }
  std::vector<std::uint8_t> decoded;
  decoded.reserve((encoded.size() / 4) * 3);
  for (std::size_t index = 0; index < encoded.size(); index += 4) {
    const int first = base64_value(encoded[index]);
    const int second = base64_value(encoded[index + 1]);
    const bool third_padding = encoded[index + 2] == '=';
    const bool fourth_padding = encoded[index + 3] == '=';
    const int third = third_padding ? 0 : base64_value(encoded[index + 2]);
    const int fourth = fourth_padding ? 0 : base64_value(encoded[index + 3]);
    const bool final_group = index + 4 == encoded.size();
    if (first < 0 || second < 0 || third < 0 || fourth < 0 ||
        (!final_group && (third_padding || fourth_padding)) || (third_padding && !fourth_padding)) {
      return std::nullopt;
    }
    decoded.push_back(static_cast<std::uint8_t>((first << 2) | (second >> 4)));
    if (!third_padding) {
      decoded.push_back(static_cast<std::uint8_t>(((second & 0x0F) << 4) | (third >> 2)));
    }
    if (!fourth_padding) {
      decoded.push_back(static_cast<std::uint8_t>(((third & 0x03) << 6) | fourth));
    }
  }
  return decoded;
}

enum class CropParseResult { valid, invalid_request, invalid_crop };

[[nodiscard]] CropParseResult parse_face_crop(std::string_view body,
                                               veritas::inference::FaceCrop& crop) {
  const auto root = JsonParser(body).parse();
  if (!root.has_value() || root->type != JsonValue::Type::object) {
    return CropParseResult::invalid_request;
  }
  const JsonValue* face_crop = object_field(*root, "face_crop");
  if (face_crop == nullptr || face_crop->type != JsonValue::Type::object) {
    return CropParseResult::invalid_request;
  }
  const auto color_space = string_field(*face_crop, "color_space");
  const auto width = size_field(*face_crop, "width");
  const auto height = size_field(*face_crop, "height");
  const auto channels = size_field(*face_crop, "channels");
  const auto layout = string_field(*face_crop, "layout");
  const auto value_range = string_field(*face_crop, "value_range");
  const auto pixels_base64 = string_field(*face_crop, "pixels_base64");
  if (!color_space.has_value() || !width.has_value() || !height.has_value() ||
      !channels.has_value() || !layout.has_value() || !value_range.has_value() ||
      !pixels_base64.has_value()) {
    return CropParseResult::invalid_request;
  }
  const auto pixels = decode_base64(*pixels_base64);
  if (!pixels.has_value()) {
    return CropParseResult::invalid_crop;
  }
  crop = veritas::inference::FaceCrop{
      .color_space = *color_space,
      .width = *width,
      .height = *height,
      .channels = *channels,
      .layout = *layout,
      .value_range = *value_range,
      .pixels = std::move(*pixels),
  };
  return CropParseResult::valid;
}

class ModelRunner final {
 public:
  explicit ModelRunner(ModelOptions options)
      : options_(std::move(options)),
        environment_(ORT_LOGGING_LEVEL_WARNING, "veritas-face-inference"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    session_options_.SetIntraOpNumThreads(static_cast<int>(options_.threads));
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(environment_, options_.path.c_str(), session_options_);
    validate_model_contract();
  }

  [[nodiscard]] const ModelOptions& options() const { return options_; }

  [[nodiscard]] std::pair<float, double> infer(
      veritas::inference::PreprocessedFaceCrop& preprocessed) const {
    if (preprocessed.nchw_pixels.size() != veritas::inference::kFaceCropByteCount) {
      throw std::invalid_argument("The preprocessed tensor has an invalid length.");
    }
    constexpr std::array<int64_t, 4> kInputShape{1, 3,
                                                   static_cast<int64_t>(veritas::inference::kFaceCropHeight),
                                                   static_cast<int64_t>(veritas::inference::kFaceCropWidth)};
    auto input = Ort::Value::CreateTensor<float>(memory_info_,
                                                 preprocessed.nchw_pixels.data(),
                                                 preprocessed.nchw_pixels.size(), kInputShape.data(),
                                                 kInputShape.size());
    const std::array<const char*, 1> input_names{input_name_.c_str()};
    const std::array<const char*, 1> output_names{output_name_.c_str()};
    const auto started = std::chrono::steady_clock::now();
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input, input_names.size(),
                                 output_names.data(), output_names.size());
    const auto finished = std::chrono::steady_clock::now();
    if (outputs.size() != 1 || !outputs.front().IsTensor()) {
      throw std::runtime_error("The model did not return its required probability tensor.");
    }
    const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementCount() != 1) {
      throw std::runtime_error("The model did not return one float32 synthetic probability.");
    }
    const float probability = *outputs.front().GetTensorData<float>();
    if (!std::isfinite(probability) || probability < 0.0F || probability > 1.0F) {
      throw std::runtime_error("The model returned a synthetic probability outside the 0 to 1 range.");
    }
    const double elapsed_milliseconds =
        std::chrono::duration<double, std::milli>(finished - started).count();
    return {probability, elapsed_milliseconds};
  }

 private:
  void validate_model_contract() {
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
      throw std::runtime_error("The model must have exactly one input and one output.");
    }
    const Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> input_shape = input_info.GetShape();
    constexpr std::array<int64_t, 4> kExpectedInputShape{1, 3,
                                                           static_cast<int64_t>(veritas::inference::kFaceCropHeight),
                                                           static_cast<int64_t>(veritas::inference::kFaceCropWidth)};
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_shape.size() != kExpectedInputShape.size() ||
        !std::equal(input_shape.begin(), input_shape.end(), kExpectedInputShape.begin())) {
      throw std::runtime_error("The model input must be one float32 NCHW tensor shaped 1x3x224x224.");
    }
    const Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
    const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementCount() != 1) {
      throw std::runtime_error("The model output must contain one float32 synthetic probability.");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session_->GetInputNameAllocated(0, allocator);
    const auto output_name = session_->GetOutputNameAllocated(0, allocator);
    if (!input_name || !output_name || input_name.get()[0] == '\0' || output_name.get()[0] == '\0') {
      throw std::runtime_error("The model input and output must each have a name.");
    }
    input_name_ = input_name.get();
    output_name_ = output_name.get();
  }

  ModelOptions options_;
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
};

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

[[nodiscard]] std::string lowercase_ascii(std::string_view value) {
  std::string lowercase;
  lowercase.reserve(value.size());
  for (const char character : value) {
    if (character >= 'A' && character <= 'Z') {
      lowercase.push_back(static_cast<char>(character - 'A' + 'a'));
    } else {
      lowercase.push_back(character);
    }
  }
  return lowercase;
}

[[nodiscard]] std::optional<HttpRequest> read_request(int client) {
  std::string received;
  received.reserve(4096);
  std::array<char, 4096> buffer{};
  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    const ssize_t count = recv(client, buffer.data(), buffer.size(), 0);
    if (count <= 0 || received.size() + static_cast<std::size_t>(count) > kMaximumHeaderBytes) {
      return std::nullopt;
    }
    received.append(buffer.data(), static_cast<std::size_t>(count));
    header_end = received.find("\r\n\r\n");
  }
  const std::size_t body_start = header_end + 4;
  const std::string_view header_view{received.data(), header_end};
  const std::size_t request_line_end = header_view.find("\r\n");
  const std::string_view request_line = header_view.substr(0, request_line_end);
  const std::size_t first_space = request_line.find(' ');
  const std::size_t second_space = request_line.find(' ', first_space + 1);
  if (first_space == std::string_view::npos || second_space == std::string_view::npos ||
      request_line.substr(second_space + 1) != "HTTP/1.1") {
    return std::nullopt;
  }
  HttpRequest request{
      .method = std::string(request_line.substr(0, first_space)),
      .path = std::string(request_line.substr(first_space + 1, second_space - first_space - 1)),
  };

  bool has_content_length = false;
  std::size_t content_length = 0;
  std::size_t line_start =
      request_line_end == std::string_view::npos ? header_view.size() : request_line_end + 2;
  while (line_start < header_view.size()) {
    const std::size_t line_end = header_view.find("\r\n", line_start);
    const std::string_view line = header_view.substr(line_start, line_end - line_start);
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos) {
      return std::nullopt;
    }
    const std::string_view name = line.substr(0, colon);
    std::string_view value = line.substr(colon + 1);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.remove_prefix(1);
    }
    if (lowercase_ascii(name) == "content-length") {
      if (has_content_length) {
        return std::nullopt;
      }
      std::uint64_t parsed_length = 0;
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), parsed_length);
      if (error != std::errc{} || end != value.data() + value.size() ||
          parsed_length > kMaximumRequestBodyBytes) {
        return std::nullopt;
      }
      has_content_length = true;
      content_length = static_cast<std::size_t>(parsed_length);
    }
    if (line_end == std::string_view::npos) {
      break;
    }
    line_start = line_end + 2;
  }
  if (request.method == "POST" && !has_content_length) {
    return std::nullopt;
  }
  while (received.size() - body_start < content_length) {
    const ssize_t count = recv(client, buffer.data(), buffer.size(), 0);
    if (count <= 0 || received.size() + static_cast<std::size_t>(count) >
                          kMaximumHeaderBytes + kMaximumRequestBodyBytes) {
      return std::nullopt;
    }
    received.append(buffer.data(), static_cast<std::size_t>(count));
  }
  request.body = received.substr(body_start, content_length);
  return request;
}

[[nodiscard]] std::string unavailable_model_info() {
  return R"({"service":"veritas-face-inference","version":"0.1.0","model":{"id":"synthetic-portrait-classifier","version":"not_loaded","status":"unavailable"},"input":{"color_space":"RGB","width":224,"height":224,"channels":3,"layout":"HWC","value_range":"0_to_255","normalization":"not_configured"}})";
}

[[nodiscard]] std::string ready_model_info(const ModelRunner& model) {
  const ModelOptions& options = model.options();
  return "{\"service\":\"veritas-face-inference\",\"version\":\"0.1.0\",\"model\":{\"id\":" +
         json_quote(options.id) + ",\"version\":" + json_quote(options.version) +
         ",\"status\":\"ready\",\"runtime\":\"onnxruntime-cpu\"},\"input\":{\"color_space\":\"RGB\",\"width\":224,\"height\":224,\"channels\":3,\"layout\":\"HWC\",\"value_range\":\"0_to_255\",\"normalization\":\"imagenet_rgb_v1\"}}";
}

[[nodiscard]] Response route_request(const HttpRequest& request, const ModelRunner* model) {
  if (request.path == "/v1/infer") {
    if (request.method != "POST") {
      return error_response(405, "Method Not Allowed", "method_not_allowed",
                            "Only POST is supported by this endpoint.", "POST");
    }
    if (model == nullptr) {
      return error_response(503, "Service Unavailable", "model_unavailable",
                            "The synthetic portrait classifier is not loaded.");
    }
    veritas::inference::FaceCrop crop;
    const CropParseResult parse_result = parse_face_crop(request.body, crop);
    if (parse_result == CropParseResult::invalid_request) {
      return error_response(400, "Bad Request", "invalid_inference_request",
                            "The request must contain one valid face_crop object.");
    }
    if (parse_result == CropParseResult::invalid_crop) {
      return error_response(422, "Unprocessable Content", "invalid_face_crop",
                            "The face crop must contain valid base64-encoded RGB pixel bytes.");
    }
    veritas::inference::PreprocessedFaceCrop preprocessed;
    std::string preprocessing_error;
    if (!veritas::inference::preprocess_face_crop(crop, preprocessed, preprocessing_error)) {
      return error_response(422, "Unprocessable Content", "invalid_face_crop", preprocessing_error);
    }
    try {
      const auto [probability, latency_milliseconds] = model->infer(preprocessed);
      std::ostringstream latency;
      latency << std::fixed << std::setprecision(3) << latency_milliseconds;
      std::ostringstream score;
      score << std::fixed << std::setprecision(7) << probability;
      const ModelOptions& options = model->options();
      return {200,
              "OK",
              "{\"synthetic_probability\":" + score.str() + ",\"detector\":{\"id\":" +
                  json_quote(options.id) + ",\"version\":" + json_quote(options.version) +
                  "},\"latency_ms\":" + latency.str() + "}"};
    } catch (const std::exception& exception) {
      std::cerr << "ONNX inference failed: " << exception.what() << '\n';
      return error_response(503, "Service Unavailable", "inference_failed",
                            "The synthetic portrait classifier could not complete this request.");
    }
  }
  if (request.method != "GET") {
    return error_response(405, "Method Not Allowed", "method_not_allowed",
                          "Only GET is supported by this endpoint.", "GET");
  }
  if (request.path == "/health" || request.path == "/healthz") {
    return {200, "OK", R"({"status":"ok","service":"veritas-face-inference","version":"0.1.0"})"};
  }
  if (request.path == "/v1/model-info") {
    return {200, "OK", model == nullptr ? unavailable_model_info() : ready_model_info(*model)};
  }
  return error_response(404, "Not Found", "not_found", "The requested endpoint does not exist.");
}

void write_response(int client, const Response& response) {
  std::string headers = "HTTP/1.1 " + std::to_string(response.status_code) + " " +
                        std::string(response.reason) + "\r\n"
                        "Content-Type: application/json\r\n"
                        "Cache-Control: no-store\r\n"
                        "Connection: close\r\n" +
                        response.extra_headers + "Content-Length: " +
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

void serve_client(int client, const ModelRunner* model) {
  const auto request = read_request(client);
  if (!request.has_value()) {
    write_response(client, error_response(400, "Bad Request", "invalid_request",
                                          "A valid HTTP request is required."));
  } else {
    write_response(client, route_request(*request, model));
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

  std::unique_ptr<ModelRunner> model;
  if (options->model.has_value()) {
    try {
      model = std::make_unique<ModelRunner>(*options->model);
    } catch (const std::exception& exception) {
      std::cerr << "Unable to load configured ONNX model: " << exception.what() << '\n';
      return EXIT_FAILURE;
    }
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
    serve_client(client, model.get());
  }
}
