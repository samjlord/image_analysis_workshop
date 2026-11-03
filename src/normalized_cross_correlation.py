"""Find a small template in an image with normalized cross-correlation.

Run it from the top folder of the workshop repository with:

    uv run src/normalized_cross_correlation.py

The small image Mito_template.tif is placed on top of Mito_Cropped.tif and moved
("rolled") across it, one pixel at a time.  At every position:

    1. The template is normalized: its average is subtracted from every pixel and
       the result is divided by its spread (standard deviation).  The normalized
       pixels are called z-scores.
    2. The piece of the image under the template (the "patch") is normalized in
       the same way, with its own average and spread.
    3. The z-scores of the template and of the patch are multiplied pixel by pixel,
       and the average of the products is written to the output image at the
       position of the center of the template.

The normalization removes the overall brightness (the average) and the contrast
(the spread) of the patch, so only the pattern of brighter and darker pixels is
compared.  The result is always between -1 and 1:

     1  the patch has the same pattern as the template
     0  no relationship
    -1  the patch is the template turned inside out (bright where it is dark)

The script then looks for the places where the template matches well: all peaks of
the result with a value above 0.8 that are at least 4 pixels apart.  They are
printed and shown as circles on top of Mito_Cropped.tif in napari: a green circle
for the highest peak and red circles for the others.  Close the napari window to
end the script.

One image is saved next to the original in testImages/:

    Mito_Cropped_normalized_crosscorrelation.tif  normalized cross-correlation (decimal numbers)
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from skimage.feature import match_template, peak_local_max

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE = REPO / "testImages" / "Mito_template.tif"

# A peak must have a value above this threshold ...
THRESHOLD = 0.8
# ... and peaks must be at least this many pixels apart.
MIN_PEAK_DISTANCE = 4

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

# Slide the template over the image and calculate the normalized cross-correlation
# at every position where the template fits completely inside the image.  The
# result is a little smaller than the image: (rows - t_rows + 1) x (cols - t_cols + 1).
valid = match_template(image, template)

# Put the result at the position of the center of the template, in an output image
# of the same size as the input image, so that it lines up with the input pixel
# for pixel.  The border, where the template would hang over the edge, stays 0.
normalized = np.zeros(image.shape, dtype=np.float32)
normalized[t_rows // 2:t_rows // 2 + valid.shape[0],
           t_cols // 2:t_cols // 2 + valid.shape[1]] = valid
print(f"\nNormalized cross-correlation: from {normalized.min():.4f} to {normalized.max():.4f}")

output = IMAGE.with_name(f"{IMAGE.stem}_normalized_crosscorrelation.tif")
tifffile.imwrite(output, normalized, imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"Saved {output.name}")

# Find the peaks: the pixels that are higher than all of their neighbors and above
# the threshold.  peak_local_max keeps a pixel only if it is the highest in a square
# window that reaches min_distance pixels in every direction, so with
# min_distance = 3 two peaks are always at least 4 pixels apart.  The peaks are
# returned sorted from the highest to the lowest, as (row, column) positions.
peaks = peak_local_max(normalized, min_distance=MIN_PEAK_DISTANCE - 1,
                       threshold_abs=THRESHOLD, exclude_border=False)

print(f"\nFound {len(peaks)} peaks above {THRESHOLD} that are at least "
      f"{MIN_PEAK_DISTANCE} pixels apart:")
print("   row  column   value")
for peak_row, peak_col in peaks:
    print(f"  {peak_row:4d}  {peak_col:6d}  {normalized[peak_row, peak_col]:7.4f}")

# Show the image in napari with a circle around every peak.  The circles are as
# large as the template, so they show which part of the image matched.  The
# highest peak gets a green circle, all the others a red one.  The positions are
# (row, column), which is the order that napari uses.
colors = ["red"] * len(peaks)
if len(peaks) > 0:
    values = normalized[peaks[:, 0], peaks[:, 1]]
    colors[np.argmax(values)] = "lime"

viewer = napari.Viewer()
viewer.add_image(image, name=IMAGE.stem)
viewer.add_points(peaks, name=f"peaks above {THRESHOLD}", size=max(t_rows, t_cols),
                  symbol="disc", face_color="transparent",
                  border_color=colors, border_width=0.1)
napari.run()
