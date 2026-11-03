"""Find edges with the Sobel kernels and the gradient magnitude.

Run it from the top folder of the workshop repository with:

    uv run src/sobel_filters.py

An edge is a place where the brightness changes quickly.  The Sobel x kernel
measures how fast the brightness changes from left to right, so it finds
vertical edges.  The Sobel y kernel measures how fast it changes from top to
bottom, so it finds horizontal edges.  To find edges in every direction the two
results are combined into the "gradient magnitude":

    gradient magnitude = square root of (Sobel x squared + Sobel y squared)

Noise also makes the brightness change quickly, so the gradient magnitude picks
up noise as well as real edges.  To see the effect, the script calculates the
gradient magnitude twice: on Actin_cropped.tif itself, and on Actin_cropped.tif
after blurring it with a 5x5 Gaussian kernel.

Five images are saved next to the original in testImages/:

    Actin_cropped_sobel_x.tif                       Sobel x (positive and negative numbers)
    Actin_cropped_sobel_y.tif                       Sobel y (positive and negative numbers)
    Actin_cropped_sobel_magnitude.tif               gradient magnitude of the original
    Actin_cropped_gaussian_5x5.tif                  the original blurred with a 5x5 Gaussian
    Actin_cropped_gaussian_5x5_sobel_magnitude.tif  gradient magnitude of the blurred image
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"

# The Sobel kernels.  Each is a smoothing part (1 2 1) along the edge, times a
# difference part (-1 0 1) across the edge.  The weights add up to 0, so a flat
# region gives 0.  That also means these kernels cannot be normalized by their
# sum (we would divide by zero), so they are used as they are.
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=float)

GAUSSIAN_SIZE = 5


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
    return kernel / kernel.sum()


def sobel(image):
    """Return Sobel x, Sobel y and the gradient magnitude of a decimal-number image."""
    # The Sobel kernels are not symmetric.  ndimage.convolve flips the kernel
    # before sliding it over the image, which would reverse the sign of the
    # result.  ndimage.correlate uses the kernel exactly as written above, so a
    # positive value means that the brightness increases to the right (x) or
    # downward (y).
    gx = ndimage.correlate(image, SOBEL_X)
    gy = ndimage.correlate(image, SOBEL_Y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return gx, gy, magnitude


def save(array, suffix, number_type):
    """Save an array next to the input, with the pixel size of the input."""
    output = INPUT.with_name(f"{INPUT.stem}_{suffix}.tif")
    tifffile.imwrite(output, array.astype(number_type), imagej=True,
                     resolution=resolution, metadata={"unit": unit})
    print(f"Saved {output.name}")


def describe(name, array):
    print(f"{name}: from {array.min():.1f} to {array.max():.1f}, mean {array.mean():.2f}")


with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the output images have the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels, mean intensity {image.mean():.2f}\n")

print(f"Sobel x kernel (weights add up to {SOBEL_X.sum():g}):")
print(SOBEL_X.astype(int))
print(f"Sobel y kernel (weights add up to {SOBEL_Y.sum():g}):")
print(SOBEL_Y.astype(int), "\n")

# The original image.  The results are signed decimal numbers (32-bit float),
# because Sobel x and y are negative where the brightness decreases.
original = image.astype(float)
gx, gy, magnitude = sobel(original)
describe("Sobel x", gx)
describe("Sobel y", gy)
describe("Gradient magnitude", magnitude)
save(gx, "sobel_x", np.float32)
save(gy, "sobel_y", np.float32)
save(magnitude, "sobel_magnitude", np.float32)

# The blurred image.  A weighted average stays inside the range of the image, so
# for saving we only round to whole numbers and go back to the input's type.
print(f"\nBlurring with a {GAUSSIAN_SIZE}x{GAUSSIAN_SIZE} Gaussian kernel")
blurred = ndimage.convolve(original, gaussian_kernel(GAUSSIAN_SIZE))
_, _, blurred_magnitude = sobel(blurred)
describe("Gradient magnitude of the blurred image", blurred_magnitude)
save(np.rint(blurred), f"gaussian_{GAUSSIAN_SIZE}x{GAUSSIAN_SIZE}", image.dtype)
save(blurred_magnitude,
     f"gaussian_{GAUSSIAN_SIZE}x{GAUSSIAN_SIZE}_sobel_magnitude", np.float32)

print(f"\nThe mean gradient magnitude of the blurred image is "
      f"{blurred_magnitude.mean() / magnitude.mean():.0%} of the original's.")
