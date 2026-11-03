"""Remove noise with median filters of different shapes and sizes.

Run it from the top folder of the workshop repository with:

    uv run src/median_filters.py

The script applies six median filters to Nucleus_cropped_noise.tif.  A median
filter replaces every pixel by the median (the middle value, after sorting) of
the pixels in its neighborhood.  The "kernel" says which neighbors count.
Six images are saved next to the original in testImages/:

    Nucleus_cropped_noise_median_3x3_no_corners.tif  3x3 kernel without its four corners (5 pixels)
    Nucleus_cropped_noise_median_3x3.tif             3x3 kernel (square, 9 pixels)
    Nucleus_cropped_noise_median_5x5_plus.tif        5x5 kernel, 2x2 block cut from each corner (thin plus, 9 pixels)
    Nucleus_cropped_noise_median_5x5_diamond.tif     5x5 kernel, 3 pixels cut from each corner (diamond, 13 pixels)
    Nucleus_cropped_noise_median_5x5_no_corners.tif  5x5 kernel without its four corners (21 pixels)
    Nucleus_cropped_noise_median_5x5.tif             5x5 kernel (square, 25 pixels)
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# The kernels.  1 = this neighbor is used for the median, 0 = it is ignored.
# The pixel being calculated is always in the middle of the kernel.
KERNELS = {
    "3x3_no_corners": np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=bool),
    "3x3": np.ones((3, 3), dtype=bool),
    "5x5_plus": np.array([[0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [1, 1, 1, 1, 1],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0]], dtype=bool),
    "5x5_diamond": np.array([[0, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0]], dtype=bool),
    "5x5_no_corners": np.array([[0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0]], dtype=bool),
    "5x5": np.ones((5, 5), dtype=bool),
}

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the filtered images have the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels\n")

for name, kernel in KERNELS.items():
    print(f"Kernel {name} ({kernel.sum()} pixels used):")
    print(kernel.astype(int), "\n")

    # Slide the kernel over the image and take the median at every position.
    # A median is always one of the original values, so the result has the same
    # number type (16-bit) as the input and no rescaling is needed.
    filtered = ndimage.median_filter(image, footprint=kernel)

    output = INPUT.with_name(f"{INPUT.stem}_median_{name}.tif")
    tifffile.imwrite(output, filtered, imagej=True,
                     resolution=resolution, metadata={"unit": unit})
    print(f"Saved {output.name}\n")
