# Face selection and quality gates

Before any provenance or detector score is interpreted, Veritas Face selects one
dominant face using OpenCV's local Haar cascade. The cascade produces candidate
rectangles; the largest valid rectangle is selected only when it is clearly larger
than every other detected face.

The assessment is marked `inconclusive` rather than treated as evidence for either
origin when any of these gates apply:

- no face is detected;
- two similarly sized faces make the primary subject ambiguous;
- the primary face is smaller than 96 pixels in either dimension;
- Laplacian sharpness variance indicates a blurry crop; or
- mean grayscale brightness indicates an extremely dark or bright crop.

These are eligibility checks, not authenticity signals. The detector runs locally
on validated temporary artifacts, and tests generate image arrays in memory rather
than storing portrait fixtures in the repository.
