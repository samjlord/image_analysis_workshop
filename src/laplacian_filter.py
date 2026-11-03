"""Find edges, lines and spots with a Laplacian filter.

Run it from the top folder of the workshop repository with:

    uv run src/laplacian_filter.py

The Laplacian compares every pixel with its neighbors: the neighbors are added up
and the pixel itself, multiplied by the number of neighbors, is subtracted.  In a
flat region (or where the brightness changes at a constant rate) the pixel is
equal to the average of its neighbors, so the result is 0.  Where a pixel differs
from its neighbors, as at an edge, on a thin line or in a spot, the result is
not 0.  The Laplacian measures how much the brightness "bends".

The script applies two versions of the Laplacian kernel to Actin_cropped.tif:
one that uses the 4 direct neighbors, and one that uses all 8 neighbors,
including the diagonal ones.  Two images are saved next to the original in
testImages/:

    Actin_cropped_laplacian_4_neighbors.tif  4 neighbors (signed decimal numbers)
    Actin_cropped_laplacian_8_neighbors.tif  8 neighbors (signed decimal numbers)
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"

# The kernels.  Each number is the weight given to that neighbor.
# The pixel being calculated is always in the middle of the kernel.
# The weights add up to 0, so a flat region gives 0.  That also means these
# kernels cannot be normalized by their sum (we would divide by zero), so they
# are used as they are.
KERNELS = {
    "4_neighbors": np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=float),
    "8_neighbors": np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]], dtype=float),
}

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the output images have the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels, mean intensity {image.mean():.2f}\n")

for name, kernel in KERNELS.items():
    print(f"Kernel {name} (weights add up to {kernel.sum():g}):")
    print(kernel.astype(int))

    # Slide the kernel over the image: at every position multiply the neighbors
    # by the weights and add them up.  The kernels are symmetric, so it makes no
    # difference whether the kernel is flipped first.  We calculate with decimal
    # numbers, because the results can be negative.
    laplacian = ndimage.convolve(image.astype(float), kernel)

    # With these kernels (negative in the middle) a spot that is brighter than its
    # surroundings gives a negative number and a spot that is darker gives a
    # positive number.  The result is therefore saved as signed decimal numbers
    # (32-bit float), not clipped to the range of the 16-bit input.
    print(f"Result: from {laplacian.min():.1f} to {laplacian.max():.1f}, "
          f"mean {laplacian.mean():.3f}")

    output = INPUT.with_name(f"{INPUT.stem}_laplacian_{name}.tif")
    tifffile.imwrite(output, laplacian.astype(np.float32), imagej=True,
                     resolution=resolution, metadata={"unit": unit})
    print(f"Saved {output.name}\n")
