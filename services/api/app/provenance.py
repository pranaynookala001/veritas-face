"""Normalize local image metadata and offline C2PA verification results.

The adapter deliberately returns only safe presence and validation facts.  It
does not expose EXIF/XMP values, manifest contents, signer identities, or
remote manifest locations in a public report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Protocol

from PIL import Image


PROVENANCE_ADAPTER_VERSION = "1.0"
C2PA_SDK_VERSION = "c2pa-python-0.37.10"


class C2paVerificationStatus(str, Enum):
    """Publicly safe outcomes from C2PA manifest verification."""

    VERIFIED = "verified"
    INVALID = "invalid"
    NOT_PRESENT = "not_present"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ImageMetadata:
    """Presence-only metadata facts extracted from a decoded image."""

    format_name: str
    width: int
    height: int
    exif_present: bool
    xmp_present: bool
    other_metadata_present: bool

    @property
    def embedded_metadata_present(self) -> bool:
        """Whether the decoded image carries any embedded metadata."""
        return self.exif_present or self.xmp_present or self.other_metadata_present


@dataclass(frozen=True)
class C2paVerification:
    """A privacy-preserving C2PA verification summary."""

    status: C2paVerificationStatus
    version: str = C2PA_SDK_VERSION


@dataclass(frozen=True)
class ProvenanceFindings:
    """Normalized local provenance facts used by the report builder."""

    metadata: ImageMetadata
    c2pa: C2paVerification


class C2paVerifier(Protocol):
    """Small seam that keeps C2PA SDK behavior deterministic in tests."""

    def verify(self, content: bytes, media_type: str) -> C2paVerification:
        """Verify the C2PA manifest embedded in one accepted image."""


class C2paSdkVerifier:
    """Use the official SDK without fetching external manifests for uploads."""

    def verify(self, content: bytes, media_type: str) -> C2paVerification:
        try:
            import c2pa
        except ImportError:
            return C2paVerification(C2paVerificationStatus.UNAVAILABLE, version="c2pa-python-unavailable")

        try:
            # User uploads must not cause the worker to retrieve remote manifest
            # stores. Embedded Content Credentials are still fully checked.
            settings = c2pa.Settings.from_dict({"verify": {"remote_manifest_fetch": False}})
            with c2pa.Context(settings) as context:
                with c2pa.Reader(media_type, BytesIO(content), context=context) as reader:
                    validation_state = reader.get_validation_state()
        except Exception as error:
            # The SDK uses ManifestNotFound for files without embedded credentials.
            # Do not include an SDK error because malformed user-controlled data may
            # be reflected in it.
            if "ManifestNotFound" in str(error):
                return C2paVerification(C2paVerificationStatus.NOT_PRESENT, _sdk_version(c2pa))
            return C2paVerification(C2paVerificationStatus.UNAVAILABLE, _sdk_version(c2pa))

        if str(validation_state).casefold() in {"valid", "trusted"}:
            return C2paVerification(C2paVerificationStatus.VERIFIED, _sdk_version(c2pa))
        return C2paVerification(C2paVerificationStatus.INVALID, _sdk_version(c2pa))


def inspect_provenance(
    content: bytes,
    media_type: str,
    *,
    c2pa_verifier: C2paVerifier | None = None,
) -> ProvenanceFindings:
    """Extract presence-only EXIF/XMP facts and verify embedded credentials."""
    with Image.open(BytesIO(content)) as image:
        image.load()
        metadata = ImageMetadata(
            format_name=(image.format or media_type).upper(),
            width=image.width,
            height=image.height,
            exif_present=bool(image.getexif()),
            xmp_present=_has_xmp(image),
            other_metadata_present=_has_other_metadata(image),
        )

    verifier = c2pa_verifier or C2paSdkVerifier()
    return ProvenanceFindings(metadata=metadata, c2pa=verifier.verify(content, media_type))


def metadata_detail(metadata: ImageMetadata) -> str:
    """Render non-sensitive metadata facts without returning any values."""
    metadata_presence = "embedded metadata is present" if metadata.embedded_metadata_present else "no embedded metadata is present"
    return (
        f"Decoded {metadata.format_name} image, {metadata.width} × {metadata.height} pixels; "
        f"EXIF metadata is {_presence_word(metadata.exif_present)}; "
        f"XMP metadata is {_presence_word(metadata.xmp_present)}; "
        f"{metadata_presence}. Metadata presence or absence is not an authenticity signal."
    )


def c2pa_detail(verification: C2paVerification) -> str:
    """Render a safe explanation of a C2PA result without manifest contents."""
    if verification.status is C2paVerificationStatus.VERIFIED:
        return (
            "An embedded C2PA/Content Credentials manifest verified with local-only checks. "
            "It verifies declared provenance, not whether this portrait is authentic or synthetic."
        )
    if verification.status is C2paVerificationStatus.INVALID:
        return (
            "An embedded C2PA/Content Credentials manifest did not validate. "
            "An invalid credential is not proof that this portrait is authentic or synthetic."
        )
    if verification.status is C2paVerificationStatus.NOT_PRESENT:
        return (
            "No embedded C2PA/Content Credentials manifest was found. "
            "Missing provenance is not evidence that this portrait is authentic or synthetic."
        )
    return (
        "C2PA/Content Credentials verification could not complete for this image. "
        "An unavailable verification result is not evidence that this portrait is authentic or synthetic."
    )


def _has_xmp(image: Image.Image) -> bool:
    """Recognize Pillow's XMP fields without parsing user-provided XML values."""
    return "xmp" in image.info or "XML:com.adobe.xmp" in image.info


def _has_other_metadata(image: Image.Image) -> bool:
    """Keep the existing generic metadata-presence signal without exposing it."""
    return any(key not in {"exif", "xmp", "XML:com.adobe.xmp"} for key in image.info)


def _presence_word(value: bool) -> str:
    return "present" if value else "not present"


def _sdk_version(c2pa_module: object) -> str:
    version = getattr(c2pa_module, "__version__", None)
    return f"c2pa-python-{version}" if version else C2PA_SDK_VERSION
