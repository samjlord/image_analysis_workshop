"""Cut a small piece (a "template") out of Mito_Cropped.tif for cross-correlation.

Run it from the top folder of the workshop repository with:

    uv run src/make_mito_template.py

The result is saved as testImages/Mito_template.tif.
"""

from pathlib import Path

import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Mito_Cropped.tif"
OUTPUT = REPO / "testImages" / "Mito_template.tif"

# The center of the piece (row, column) and how far it extends on each side.
# This spot is a single, squiggly mitochondrion near the top of the image.
# A half-width of 9 gives a 19 x 19 pixel piece: big enough to hold the whole
# squiggle, so it is not mistaken for other short bits of mitochondria.
# Careful: Fiji shows positions as (x, y) = (column, row), but numpy wants
# (row, column).  So Fiji's x=144, y=37 is row 37, column 144 here.
CENTER_ROW, CENTER_COL = 37, 144
HALF_WIDTH = 9

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the template has the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

# "start:stop" picks rows (or columns) start up to, but not including, stop.
top, left = CENTER_ROW - HALF_WIDTH, CENTER_COL - HALF_WIDTH
template = image[top:CENTER_ROW + HALF_WIDTH + 1, left:CENTER_COL + HALF_WIDTH + 1]

tifffile.imwrite(OUTPUT, template, imagej=True, resolution=resolution,
                 metadata={"unit": unit})

print(f"Cut rows {top} - {top + template.shape[0] - 1}, "
      f"columns {left} - {left + template.shape[1] - 1} "
      f"({template.shape[0]} x {template.shape[1]} pixels)")
print(f"Saved {OUTPUT}")
