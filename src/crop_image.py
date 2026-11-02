"""Crop a rectangular region (ROI) out of an image and save it as a new file.

Run it from the top folder of the workshop repository with:

    uv run src/crop_image.py

The script always crops the file
testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif
and asks you for the position of two opposite corners of the region, typed
exactly as Napari shows them (y first, then x, in micrometers).

It works for simple 2D images as well as multi-channel / multi-slice / time-lapse
images (such as OME-TIFF).  Every channel, slice and time point is cropped the
same way.  The result is saved next to the original, with "_cropped" added to
the name, and keeps the pixel size and channel names of the original.

Tip: open the image in Napari first.  Hover the mouse over a corner of the
region you want; the status bar at the bottom shows the position as
[y x] (row first!) in micrometers, with [0 0] at the top-left.  Type those two
numbers in the same order.  The script converts them to pixels for you.
"""

from pathlib import Path

from bioio import BioImage
from bioio.writers import OmeTiffWriter


# The one image this script works on.
IMAGE_PATH = Path("testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif")


def ask_corner(label, width, height, pixel_size_x, pixel_size_y):
    """Ask for one corner as shown in Napari ('y x' in micrometers); return pixel (x, y)."""
    max_x = (width - 1) * pixel_size_x
    max_y = (height - 1) * pixel_size_y
    while True:
        text = input(f"{label} corner as 'y x' in micrometers (y: 0-{max_y:.1f}, x: 0-{max_x:.1f}): ")
        # Accept "10 20" as well as "10, 20", and decimals such as "10.5 20".
        parts = text.replace(",", " ").split()
        try:
            y_um, x_um = (float(p) for p in parts)
        except ValueError:
            print("  Please type exactly two numbers, for example: 120.5 300")
            continue
        # Micrometers -> pixels: divide by the size of one pixel, then round.
        x = round(x_um / pixel_size_x)
        y = round(y_um / pixel_size_y)
        if 0 <= x < width and 0 <= y < height:
            return x, y
        print("  That point is outside the image.  Please try again.")


path = IMAGE_PATH
if not path.is_file():
    raise SystemExit(f"Cannot find '{path}'.  Run this script from the top folder of the repository.")

# BioImage opens many formats (plain TIFF, OME-TIFF, ND2, ...) and always hands
# the pixels back in the same layout.  If the file holds several images
# ("scenes", e.g. several stage positions), the first one is used.
img = BioImage(path)

# "TCZYX" = Time, Channel, Z-slice, Y (rows), X (columns).  Asking for this
# order always gives a 5-dimensional array, even for a plain 2D image (where
# T, C and Z each have size 1), so one piece of code handles every file type.
n_t, n_c, n_z, height, width = img.shape
print(f"\nOpened {path.name}")
print(f"  {width} x {height} pixels, {n_c} channel(s), {n_z} slice(s), {n_t} time point(s)")
print(f"  number type: {img.dtype}\n")

# Napari shows positions in micrometers (pixel size x number of pixels), so we
# need the size of one pixel to convert them back to pixel numbers.
pixel_size_x = img.physical_pixel_sizes.X
pixel_size_y = img.physical_pixel_sizes.Y
if not pixel_size_x or not pixel_size_y:
    raise SystemExit("This image has no pixel size stored, so micrometers cannot be converted to pixels.")
print(f"  pixel size: {pixel_size_x} x {pixel_size_y} micrometers\n")

x1, y1 = ask_corner("First", width, height, pixel_size_x, pixel_size_y)
x2, y2 = ask_corner("Opposite", width, height, pixel_size_x, pixel_size_y)

# The two corners may be given in any order, so sort them.  Both corners are
# included in the crop (a slice stops one before its end, hence the "+ 1").
x_min, x_max = sorted((x1, x2))
y_min, y_max = sorted((y1, y2))

# Slicing "[..., y_min:y_max + 1, x_min:x_max + 1]" cuts the same rectangle out
# of every channel, slice and time point at once.  The "..." means "leave all
# the other dimensions as they are".
cropped = img.get_image_dask_data("TCZYX")[..., y_min : y_max + 1, x_min : x_max + 1]
cropped = cropped.compute()  # now actually read the pixels from the file

# Name the output after the input: "image.ome.tif" -> "image_cropped.ome.tif".
stem = path.name
for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
    if stem.lower().endswith(suffix):
        stem = stem[: -len(suffix)]
        break
else:
    stem = path.stem
output = path.with_name(f"{stem}_cropped.ome.tif")

# The pixel size (micrometers per pixel) does not change when cropping, so it
# is copied unchanged to the new file.  The channel names are kept as well.
OmeTiffWriter.save(
    cropped,
    output,
    dim_order="TCZYX",
    physical_pixel_sizes=img.physical_pixel_sizes,
    channel_names=[str(name) for name in img.channel_names],
)

print(f"\nCropped x {x_min}-{x_max}, y {y_min}-{y_max}  ->  {cropped.shape[-1]} x {cropped.shape[-2]} pixels")
print(f"Saved: {output}")
