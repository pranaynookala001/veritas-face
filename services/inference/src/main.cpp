#include <iostream>
#include <string_view>

int main(int argc, char* argv[]) {
  constexpr std::string_view kVersion{"0.1.0"};
  if (argc == 2 && std::string_view{argv[1]} == "--version") {
    std::cout << "veritas-inference " << kVersion << '\n';
    return 0;
  }

  std::cout << "veritas-inference scaffold " << kVersion
            << ": ONNX HTTP service is scheduled for Milestone 2.\n";
  return 0;
}
