"""Sharpen an image with an unsharp mask: blur it, then subtract the blur.

Run it from the top folder of the workshop repository with:

    uv run src/unsharp_mask.py

The script does the following to Actin_cropped.tif, once with a 5x5 and once
with a 21x21 Gaussian kernel:

    1. Blur the image with the Gaussian kernel.
    2. Subtract the blurred image from the original.  What is left is the fine
       detail: the parts of the image that the blur removed.
    3. Add that detail back to the original, so the fine detail is boosted.

A Gaussian kernel gives the most weight to the center pixel and less and less
to pixels farther away, following a bell curve.  The smaller 5x5 kernel blurs
only a little, so the detail is the finest structures.  The larger 21x21 kernel
blurs much more, so the detail includes larger structures as well.

Four images are saved next to the original in testImages/:

    Actin_cropped_unsharp_5x5_detail.tif      original - blurred, 5x5 (signed decimal numbers)
    Actin_cropped_unsharp_5x5_sharpened.tif   original + detail, 5x5
    Actin_cropped_unsharp_21x21_detail.tif    original - blurred, 21x21 (signed decimal numbers)
    Actin_cropped_unsharp_21x21_sharpened.tif original + detail, 21x21
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"

# Kernel sizes (width and height in pixels).  They must be odd, so that the
# pixel being calculated is exactly in the middle.
SIZES = [5, 21]

# How much of the detail is added back to the original.  1 = all of it.
AMOUNT = 1.0


def gaussian_kernel(size):
    """Return a size x size Gaussian kernel whose weights add up to 1."""
    radius = size // 2
    # The kernel reaches out to 3 sigma from the center, where the bell curve is
    # almost zero.  A bigger kernel therefore also means a wider bell curve.
    sigma = radius / 3
    offsets = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(offsets, offsets)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # Normalize: divide by the sum so the weights add up to 1.
    return kernel / kernel.sum(), sigma


with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the output images have the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels, mean intensity {image.mean():.2f}\n")

# The largest number the image type can hold (65535 for 16-bit images).
largest = np.iinfo(image.dtype).max
original = image.astype(float)

for size in SIZES:
    kernel, sigma = gaussian_kernel(size)
    name = f"{size}x{size}"
    print(f"Gaussian kernel {name}: sigma = {sigma:.2f} pixels")
    print(f"Center weight {kernel[size // 2, size // 2]:.4f}, "
          f"corner weight {kernel[0, 0]:.6f}, weights add up to {kernel.sum():.4f}")

    # Slide the kernel over the image: at every position multiply the neighbors
    # by the weights and add them up.  This is the blurred image.
    blurred = ndimage.convolve(original, kernel)

    # Subtract the blur from the original.  The result is positive where the
    # original is brighter than its surroundings and negative where it is darker,
    # so it is saved as signed decimal numbers (32-bit float).
    detail = original - blurred
    print(f"Detail ranges from {detail.min():.1f} to {detail.max():.1f}, "
          f"mean {detail.mean():.3f}")

    detail_output = INPUT.with_name(f"{INPUT.stem}_unsharp_{name}_detail.tif")
    tifffile.imwrite(detail_output, detail.astype(np.float32), imagej=True,
                     resolution=resolution, metadata={"unit": unit})
    print(f"Saved {detail_output.name}")

    # Add the detail back to the original.  The sum can fall outside the range of
    # the image type, so clip it, then round and go back to the input's type.
    sharpened = original + AMOUNT * detail
    n_clipped = np.count_nonzero((sharpened < 0) | (sharpened > largest))
    sharpened = np.rint(np.clip(sharpened, 0, largest)).astype(image.dtype)
    print(f"{n_clipped} pixels were clipped to the allowed range (0 to {largest})")

    sharpened_output = INPUT.with_name(f"{INPUT.stem}_unsharp_{name}_sharpened.tif")
    tifffile.imwrite(sharpened_output, sharpened, imagej=True,
                     resolution=resolution, metadata={"unit": unit})
    print(f"Saved {sharpened_output.name}\n")
