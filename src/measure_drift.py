"""Measure drift between two images with normalized cross-correlation.

Run it from the top folder of the workshop repository with:

    uv run src/measure_drift.py

To test the method, the script makes a copy of Mito_Cropped.tif in which every
pixel is moved down by 7 pixels, like an image of a sample that drifted during
the experiment.  Then it measures that movement:

    1. The small image Mito_template.tif is slid over the original image, and
       over the shifted image.  At every position the normalized
       cross-correlation is calculated (see normalized_cross_correlation.py).
       Where the template matches the image well, the result is close to 1.
    2. In each result the 5 highest peaks are found.  They mark the 5 places
       that look most like the template.
    3. Because both images show the same structures, the highest peak of the
       original belongs to the highest peak of the shifted image, the second
       highest to the second highest, and so on.  The difference between the
       positions of a pair is the movement of that structure.

If the method works, every pair moves down by 7 pixels and sideways by 0.

Two images are saved next to the original in testImages/:

    Mito_Cropped_shifted_down_7.tif                              the shifted image
    Mito_Cropped_shifted_down_7_normalized_crosscorrelation.tif  normalized cross-correlation of the shifted image
"""

from pathlib import Path

import numpy as np
import tifffile
from skimage.feature import match_template, peak_local_max

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE = REPO / "testImages" / "Mito_template.tif"

# How many pixels the image is moved down.
SHIFT = 7
# How many of the highest peaks are used.
N_PEAKS = 5
# Peaks must be at least this many pixels apart.
MIN_PEAK_DISTANCE = 4

with tifffile.TiffFile(IMAGE) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the output images have the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

with tifffile.TiffFile(TEMPLATE) as tif:
    template = tif.asarray()

rows, cols = image.shape
t_rows, t_cols = template.shape
print(f"Image:    {IMAGE.name}, {rows} x {cols} pixels")
print(f"Template: {TEMPLATE.name}, {t_rows} x {t_cols} pixels\n")

# Move every pixel down by SHIFT pixels: row 0 of the original becomes row SHIFT
# of the shifted image.  The bottom SHIFT rows of the original fall off the image,
# and the top SHIFT rows of the shifted image are empty (0).
shifted = np.zeros_like(image)
shifted[SHIFT:, :] = image[:-SHIFT, :]

shifted_output = IMAGE.with_name(f"{IMAGE.stem}_shifted_down_{SHIFT}.tif")
tifffile.imwrite(shifted_output, shifted, imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"Moved the image down by {SHIFT} pixels and saved {shifted_output.name}\n")


def find_peaks(picture):
    """Return the N_PEAKS highest peaks of the normalized cross-correlation of the
    template with the picture, as an array of (row, column, value), highest first."""
    # The normalized cross-correlation, at every position where the template fits
    # completely inside the picture.  It is a little smaller than the picture.
    valid = match_template(picture, template)

    # Put the result at the position of the center of the template, in an array of
    # the size of the picture, so that the positions are positions in the picture.
    # The border, where the template would hang over the edge, stays 0.
    normalized = np.zeros(picture.shape, dtype=np.float32)
    normalized[t_rows // 2:t_rows // 2 + valid.shape[0],
               t_cols // 2:t_cols // 2 + valid.shape[1]] = valid

    # peak_local_max keeps a pixel only if it is the highest in a square window
    # that reaches min_distance pixels in every direction, so with min_distance = 3
    # two peaks are always at least 4 pixels apart.  The peaks are returned sorted
    # from the highest to the lowest, as (row, column) positions.
    peaks = peak_local_max(normalized, min_distance=MIN_PEAK_DISTANCE - 1,
                           num_peaks=N_PEAKS, exclude_border=False)
    values = normalized[peaks[:, 0], peaks[:, 1]]
    return np.column_stack([peaks, values]), normalized


original_peaks, _ = find_peaks(image)
shifted_peaks, shifted_normalized = find_peaks(shifted)

shifted_normalized_output = IMAGE.with_name(
    f"{IMAGE.stem}_shifted_down_{SHIFT}_normalized_crosscorrelation.tif")
tifffile.imwrite(shifted_normalized_output, shifted_normalized, imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"Saved {shifted_normalized_output.name}\n")

# The i-th highest peak of the original belongs to the i-th highest peak of the
# shifted image.  The movement is the position in the shifted image minus the
# position in the original.  Rows run downward, so a positive row movement is
# a movement down, and a positive column movement is a movement to the right.
print(f"The {N_PEAKS} highest peaks of the normalized cross-correlation:")
print("        original image            shifted image          movement")
print("     row  col   value        row  col   value      down  right")
moves = []
for i, (orig, shift) in enumerate(zip(original_peaks, shifted_peaks), start=1):
    move_row, move_col = shift[0] - orig[0], shift[1] - orig[1]
    moves.append((move_row, move_col))
    print(f"{i}   {orig[0]:4.0f} {orig[1]:4.0f}  {orig[2]:6.4f}      "
          f"{shift[0]:4.0f} {shift[1]:4.0f}  {shift[2]:6.4f}    "
          f"{move_row:+5.0f}  {move_col:+5.0f}")

moves = np.array(moves)
print(f"\nMeasured movement: {moves[:, 0].mean():.2f} pixels down "
      f"(standard deviation {moves[:, 0].std():.2f}) and "
      f"{moves[:, 1].mean():.2f} pixels to the right "
      f"(standard deviation {moves[:, 1].std():.2f})")
print(f"The image was moved {SHIFT} pixels down and 0 pixels to the right.")
