"""Find a small template in an image with cross-correlation.

Run it from the top folder of the workshop repository with:

    uv run src/cross_correlation.py

The small image Mito_template.tif is placed on top of Mito_Cropped.tif and moved
("rolled") across it, one pixel at a time.  At every position the pixels of the
template are multiplied by the pixels of the image underneath, and the products
are added up.  That sum is written to the output image at the position of the
center of the template.  Where the template and the image look alike, bright
with bright and dark with dark, the sum is large.

This is a plain cross-correlation: the result is not normalized.  Because of
that, bright regions of the image give large values even if they do not look
like the template.

One image is saved next to the original in testImages/:

    Mito_Cropped_crosscorrelation.tif  cross-correlation (decimal numbers)

The script also prints where the highest value is.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE = REPO / "testImages" / "Mito_template.tif"

with tifffile.TiffFile(IMAGE) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the output has the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

with tifffile.TiffFile(TEMPLATE) as tif:
    template = tif.asarray()
    template_resolution = (tif.pages[0].tags["XResolution"].value,
                           tif.pages[0].tags["YResolution"].value)

rows, cols = image.shape
t_rows, t_cols = template.shape
print(f"Image:    {IMAGE.name}, {rows} x {cols} pixels")
print(f"Template: {TEMPLATE.name}, {t_rows} x {t_cols} pixels")
if template_resolution != resolution:
    print("Warning: the template and the image do not have the same pixel size!")

# Slide the template over the image.  Every output pixel is the sum of
# (template pixel x image pixel under it).  We calculate with decimal numbers,
# because the sums are much larger than a 16-bit number can hold.
#
# ndimage.correlate uses the template as it is.  (ndimage.convolve would flip the
# template first, which is not what we want for cross-correlation.)  Where the
# template hangs over the edge of the image, the missing pixels count as 0
# (mode="constant"), so the output image has the same size as the input image.
correlation = ndimage.correlate(image.astype(float), template.astype(float),
                                mode="constant", cval=0.0)

# Find the highest value.  Positions are counted from 0, starting at the top left.
row, col = np.unravel_index(np.argmax(correlation), correlation.shape)
print(f"\nHighest value in the output image: {correlation[row, col]:.0f}")
print(f"  at row {row}, column {col} (x = {col}, y = {row}), "
      f"the position of the center of the template")
print(f"  the template covers rows {row - t_rows // 2} to {row + t_rows // 2} "
      f"and columns {col - t_cols // 2} to {col + t_cols // 2} of the image")
print(f"Output image: from {correlation.min():.0f} to {correlation.max():.0f}, "
      f"mean {correlation.mean():.0f}")

output = IMAGE.with_name(f"{IMAGE.stem}_crosscorrelation.tif")
tifffile.imwrite(output, correlation.astype(np.float32), imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"\nSaved {output.name}")
