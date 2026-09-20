#include "face_preprocessor.hpp"

#include <array>
#include <utility>

namespace veritas::inference {
namespace {

constexpr std::array<float, kFaceCropChannels> kMeans{0.485F, 0.456F, 0.406F};
constexpr std::array<float, kFaceCropChannels> kStandardDeviations{0.229F, 0.224F, 0.225F};
constexpr float kByteScale = 1.0F / 255.0F;

}  // namespace

bool preprocess_face_crop(const FaceCrop& crop, PreprocessedFaceCrop& output, std::string& error) {
  if (crop.color_space != "RGB" || crop.layout != "HWC" || crop.value_range != "0_to_255") {
    error = "The face crop must use RGB, HWC, and the 0_to_255 value range.";
    return false;
  }
  if (crop.width != kFaceCropWidth || crop.height != kFaceCropHeight ||
      crop.channels != kFaceCropChannels) {
    error = "The face crop must be exactly 224 by 224 pixels with three channels.";
    return false;
  }
  if (crop.pixels.size() != kFaceCropByteCount) {
    error = "The decoded face-crop pixel buffer has an invalid length.";
    return false;
  }

  output.nchw_pixels.resize(kFaceCropByteCount);
  const std::size_t plane_size = kFaceCropWidth * kFaceCropHeight;
  for (std::size_t row = 0; row < kFaceCropHeight; ++row) {
    for (std::size_t column = 0; column < kFaceCropWidth; ++column) {
      const std::size_t pixel_index = row * kFaceCropWidth + column;
      const std::size_t source_index = pixel_index * kFaceCropChannels;
      for (std::size_t channel = 0; channel < kFaceCropChannels; ++channel) {
        const float scaled = static_cast<float>(crop.pixels[source_index + channel]) * kByteScale;
        output.nchw_pixels[channel * plane_size + pixel_index] =
            (scaled - kMeans[channel]) / kStandardDeviations[channel];
      }
    }
  }

  error.clear();
  return true;
}

}  // namespace veritas::inference
