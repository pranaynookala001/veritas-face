"""Content-based validation for one supported portrait upload."""

from __future__ import annotations

from io import BytesIO
import warnings

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000

SUPPORTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MEDIA_TYPE_BY_IMAGE_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class UploadValidationError(Exception):
    """A validation failure that the API can turn into a stable error response."""

    def __init__(self, status_code: int, code: str, message: str, *, field: str = "portrait") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.field = field


async def validate_upload(portrait: UploadFile) -> str:
    """Read and validate a supported encoded image without retaining its bytes.

    The declared MIME type is checked when supplied, but eligibility is determined
    from the decoded image format so a renamed or spoofed file cannot pass intake.
    """
    declared_media_type = (portrait.content_type or "").lower()
    if declared_media_type == "application/octet-stream":
        declared_media_type = ""
    if declared_media_type and declared_media_type not in SUPPORTED_MEDIA_TYPES:
        raise UploadValidationError(
            415,
            "unsupported_media_type",
            "Upload a JPEG, PNG, or WebP image.",
        )

    content = await portrait.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise UploadValidationError(400, "empty_upload", "Upload a non-empty image file.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            413,
            "upload_too_large",
            "Upload an image no larger than 10 MiB.",
        )

    detected_media_type = _detect_media_type(content)
    if declared_media_type and declared_media_type != detected_media_type:
        raise UploadValidationError(
            415,
            "media_type_mismatch",
            "The declared image type does not match the uploaded image content.",
        )
    return detected_media_type


def _detect_media_type(content: bytes) -> str:
    """Verify the image encoding and return its normalized MIME type."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise UploadValidationError(
                        422,
                        "image_too_large",
                        "The image dimensions exceed the supported safety limit.",
                    )
                if getattr(image, "is_animated", False):
                    raise UploadValidationError(
                        422,
                        "animated_image_not_supported",
                        "Upload a still JPEG, PNG, or WebP image.",
                    )
                image_format = image.format
                image.verify()
            with Image.open(BytesIO(content)) as image:
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise UploadValidationError(
            422,
            "image_too_large",
            "The image dimensions exceed the supported safety limit.",
        ) from None
    except (OSError, SyntaxError, UnidentifiedImageError):
        raise UploadValidationError(
            422,
            "invalid_image",
            "The upload is not a valid JPEG, PNG, or WebP image.",
        ) from None

    detected_media_type = MEDIA_TYPE_BY_IMAGE_FORMAT.get(image_format or "")
    if detected_media_type is None:
        raise UploadValidationError(
            415,
            "unsupported_image_format",
            "Upload a JPEG, PNG, or WebP image.",
        )
    return detected_media_type
