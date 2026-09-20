#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace veritas::inference {

inline constexpr std::size_t kFaceCropWidth = 224;
inline constexpr std::size_t kFaceCropHeight = 224;
inline constexpr std::size_t kFaceCropChannels = 3;
inline constexpr std::size_t kFaceCropByteCount =
    kFaceCropWidth * kFaceCropHeight * kFaceCropChannels;

struct FaceCrop final {
  std::string color_space;
  std::size_t width{};
  std::size_t height{};
  std::size_t channels{};
  std::string layout;
  std::string value_range;
  std::vector<std::uint8_t> pixels;
};

struct PreprocessedFaceCrop final {
  std::vector<float> nchw_pixels;
};

// Converts a row-major RGB uint8 HWC face crop to a float32 NCHW tensor. The
// fixed ImageNet RGB normalization is part of the model input contract, not a
// detector result: (channel / 255 - mean[channel]) / stddev[channel].
[[nodiscard]] bool preprocess_face_crop(const FaceCrop& crop,
                                        PreprocessedFaceCrop& output,
                                        std::string& error);

}  // namespace veritas::inference
