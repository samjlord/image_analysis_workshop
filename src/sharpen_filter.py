"""Sharpen an image with weighted (convolution) filters.

Run it from the top folder of the workshop repository with:

    uv run src/sharpen_filter.py

The script applies two sharpening filters to Actin_cropped.tif.  Every pixel is
replaced by a weighted sum of the pixels in its 3x3 neighborhood.  In a sharpening
kernel the middle weight is large and the neighbors have negative weights, so a
pixel that is brighter than its neighbors becomes even brighter, and a pixel that
is darker becomes even darker.  This makes edges look crisper (it also makes
noise stronger).  Each kernel is divided by the sum of its weights (normalization),
so the average brightness of the image does not change.  Two images are saved next
to the original in testImages/:

    Actin_cropped_sharpen_8_neighbors.tif  center 12, all 8 neighbors -1 (stronger)
    Actin_cropped_sharpen_4_neighbors.tif  center 5, the 4 edge neighbors -1, corners 0 (milder)
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"

# The kernels.  Each number is the weight given to that neighbor.
# The pixel being calculated is always in the middle of the kernel.
KERNELS = {
    "8_neighbors": np.array([[-1, -1, -1],
                             [-1, 12, -1],
                             [-1, -1, -1]], dtype=float),
    "4_neighbors": np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=float),
}

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the sharpened images have the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels, mean intensity {image.mean():.2f}\n")

# The largest number the image type can hold (65535 for 16-bit images).
largest = np.iinfo(image.dtype).max

for name, kernel in KERNELS.items():
    # Normalize: divide by the sum so the weights add up to 1.
    weights = kernel / kernel.sum()
    print(f"Kernel {name} (weights add up to {kernel.sum():g}, so each is divided by {kernel.sum():g}):")
    print(kernel.astype(int))
    print("Normalized weights:")
    print(weights)

    # Slide the kernel over the image: at every position multiply the neighbors
    # by the weights and add them up.  We calculate with decimal numbers, because
    # the results are not whole numbers and can be negative.
    sharpened = ndimage.convolve(image.astype(float), weights)

    # The negative weights can push values below 0 or above the largest number
    # the image type can hold.  Clip them back into the range of the input's type.
    n_clipped = np.count_nonzero((sharpened < 0) | (sharpened > largest))
    sharpened = np.rint(np.clip(sharpened, 0, largest)).astype(image.dtype)
    print(f"{n_clipped} pixels were clipped to the allowed range (0 to {largest})")
    print(f"Mean intensity after sharpening: {sharpened.mean():.2f}")

    output = INPUT.with_name(f"{INPUT.stem}_sharpen_{name}.tif")
    tifffile.imwrite(output, sharpened, imagej=True,
                     resolution=resolution, metadata={"unit": unit})
    print(f"Saved {output.name}\n")
