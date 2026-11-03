"""Smooth noise with weighted-average (convolution) filters.

Run it from the top folder of the workshop repository with:

    uv run src/smoothing_filters.py

The script applies three smoothing filters to Nucleus_cropped_noise.tif.  A
smoothing filter replaces every pixel by a weighted average of the pixels in its
neighborhood: each neighbor is multiplied by its weight in the "kernel", the
products are added up, and the sum is divided by the sum of the weights
(normalization).  Because of the normalization, the average brightness of the
image does not change.  Three images are saved next to the original in testImages/:

    Nucleus_cropped_noise_smooth_3x3.tif          3x3 kernel, all weights 1
    Nucleus_cropped_noise_smooth_3x3_center4.tif  3x3 kernel, all weights 1 except the center, which is 4
    Nucleus_cropped_noise_smooth_7x7.tif          7x7 kernel, all weights 1
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# The kernels.  Each number is the weight given to that neighbor.
# The pixel being calculated is always in the middle of the kernel.
KERNELS = {
    "3x3": np.ones((3, 3)),
    "3x3_center4": np.array([[1, 1, 1],
                             [1, 4, 1],
                             [1, 1, 1]], dtype=float),
    "7x7": np.ones((7, 7)),
}

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the filtered images have the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels, mean intensity {image.mean():.2f}\n")

for name, kernel in KERNELS.items():
    # Normalize: divide by the sum so the weights add up to 1.
    weights = kernel / kernel.sum()
    print(f"Kernel {name} (weights add up to {kernel.sum():g}, so each is divided by {kernel.sum():g}):")
    print(kernel.astype(int))

    # Slide the kernel over the image: at every position multiply the neighbors
    # by the weights and add them up.  We calculate with decimal numbers, because
    # the averages are not whole numbers.
    smoothed = ndimage.convolve(image.astype(float), weights)

    # An average of 16-bit values stays inside the 16-bit range, so we only have
    # to round to whole numbers and go back to the input's number type.
    smoothed = np.rint(smoothed).astype(image.dtype)
    print(f"Mean intensity after smoothing: {smoothed.mean():.2f}")

    output = INPUT.with_name(f"{INPUT.stem}_smooth_{name}.tif")
    tifffile.imwrite(output, smoothed, imagej=True,
                     resolution=resolution, metadata={"unit": unit})
    print(f"Saved {output.name}\n")
