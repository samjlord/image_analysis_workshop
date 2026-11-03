"""Calculate a 3D point spread function (PSF) of an ideal microscope objective.

Run it from the top folder of the workshop repository.  Without settings, it
makes the PSF for the Cy3 channel of the BPAE z-stack (100x / 1.40 oil objective):

    uv run src/deconvolution/make_airy_psf.py

Give any of the settings to make a PSF for other data, for example:

    uv run src/deconvolution/make_airy_psf.py --wavelength 520 --output PSF_FITC_520nm.tif
    uv run src/deconvolution/make_airy_psf.py --na 0.75 --n-immersion 1.0 --pixel-size 0.325 --z-step 1

To see all settings and their default values:

    uv run src/deconvolution/make_airy_psf.py --help

The model: an ideal (aberration-free) objective.  In the focal plane this PSF is
exactly the Airy disk:

    intensity(r) = [ 2 J1(v) / v ]^2      with v = 2 pi NA r / wavelength

Above and below focus the same model gives the familiar "hourglass" shape.
Think of it as adding up all the light waves that the objective collects
(one for each angle up to the NA), each with its own delay when you move
out of focus.  In focus the waves are all in step and you get the Airy disk.

Use the pixel size and z-step of the images you want to deconvolve, and the
emission wavelength of the dye (the center of the emission filter).

The PSF is saved as a 32-bit z-stack (pixel values add up to 1) with the
pixel size and z-step in the file, plus a picture (same name, ending in .png)
showing the PSF from above (XY) and from the side (XZ).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import special

parser = argparse.ArgumentParser(
    description="Calculate a 3D PSF of an ideal objective (Airy disk in focus).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--na", type=float, default=1.40,
                    help="numerical aperture of the objective")
parser.add_argument("--n-immersion", type=float, default=1.515,
                    help="refractive index of the immersion medium (oil 1.515, "
                         "water 1.33, air 1.0)")
parser.add_argument("--pixel-size", type=float, default=0.0651,
                    help="pixel size in the image, in microns")
parser.add_argument("--z-step", type=float, default=0.26,
                    help="distance between the planes of the z-stack, in microns")
parser.add_argument("--wavelength", type=float, default=605,
                    help="emission wavelength, in nanometers")
parser.add_argument("--size-xy", type=float, default=4.0,
                    help="width and height of the PSF image, in microns")
parser.add_argument("--size-z", type=float, default=8.0,
                    help="depth of the PSF stack, in microns")
parser.add_argument("--output", type=Path, default=None,
                    help="file to save the PSF in (default: PSF_<wavelength>nm.tif "
                         "in the current folder)")
args = parser.parse_args()

if args.na >= args.n_immersion:
    parser.error("the NA must be smaller than the refractive index of the immersion medium")


def odd_size(length_um, step_um):
    """Number of pixels covering the length, odd so that there is a center pixel."""
    n = int(round(length_um / step_um))
    return n + 1 if n % 2 == 0 else n


def airy_psf_3d(wavelength_um, na, n_immersion, xy_um, z_um, nxy, nz):
    """3D PSF of an ideal objective; the plane in focus is the Airy disk."""
    k = 2 * np.pi / wavelength_um
    # Each wave entering the objective at a different angle is described by
    # s = n * sin(angle), which runs from 0 (straight on) up to the NA.
    s = np.linspace(0, na, 400)
    ds = s[1] - s[0]

    # Distance of every pixel to the center of the image, in microns.
    coords = (np.arange(nxy) - nxy // 2) * xy_um
    r = np.hypot(*np.meshgrid(coords, coords, indexing="ij"))
    # Many pixels have the same distance; only calculate each distance once.
    r_unique, r_index = np.unique(r, return_inverse=True)
    bessel = special.j0(k * np.outer(r_unique, s)) * s   # spread over the ring

    psf = np.empty((nz, nxy, nxy))
    for i, z in enumerate((np.arange(nz) - nz // 2) * z_um):
        # Out of focus, waves at steeper angles get a different delay.
        delay = np.exp(1j * k * z * np.sqrt(n_immersion**2 - s**2))
        field = (bessel * delay).sum(axis=1) * ds
        psf[i] = (np.abs(field) ** 2)[r_index].reshape(r.shape)
    return psf / psf.sum()


wavelength_um = args.wavelength / 1000
xy_um, z_um = args.pixel_size, args.z_step
nxy, nz = odd_size(args.size_xy, xy_um), odd_size(args.size_z, z_um)
output = args.output or Path(f"PSF_{args.wavelength:g}nm.tif")

print(f"NA {args.na}, immersion n = {args.n_immersion}, wavelength {args.wavelength:g} nm")
print(f"Pixel size {xy_um} um, z-step {z_um} um -> PSF of {nz} x {nxy} x {nxy} pixels")
# Size of the Airy disk: radius of the first dark ring, 0.61 * wavelength / NA.
airy_radius = 0.61 * wavelength_um / args.na
print(f"Airy disk radius {airy_radius:.3f} um = {airy_radius / xy_um:.1f} pixels")
if airy_radius / xy_um < 2:
    print("  Warning: the Airy disk is only a few pixels wide; the pixels are too "
          "large to show the PSF well.")

psf = airy_psf_3d(wavelength_um, args.na, args.n_immersion, xy_um, z_um, nxy, nz)
tifffile.imwrite(output, psf.astype(np.float32), imagej=True,
                 resolution=(1 / xy_um, 1 / xy_um),
                 metadata={"axes": "ZYX", "spacing": z_um, "unit": "micron"})
print(f"Saved {output}")

# Show the PSF from above (focal plane) and from the side (through the center).
half_xy, half_z = nxy * xy_um / 2, nz * z_um / 2
fig, axes = plt.subplots(2, 1, figsize=(4, 7), height_ratios=[nxy * xy_um, nz * z_um])
axes[0].imshow(psf[nz // 2], cmap="magma", extent=[-half_xy, half_xy, -half_xy, half_xy])
axes[0].set_title(f"{args.wavelength:g} nm, NA {args.na}: XY in focus")
axes[0].set_ylabel("y (µm)")
# Square root makes the faint parts above and below focus easier to see.
axes[1].imshow(np.sqrt(psf[:, nxy // 2, :]), cmap="magma", aspect="auto",
               extent=[-half_xy, half_xy, -half_z, half_z])
axes[1].set_title("XZ (square root of intensity)")
axes[1].set_xlabel("x (µm)")
axes[1].set_ylabel("z (µm)")
fig.tight_layout()
picture = output.with_suffix(".png")
fig.savefig(picture, dpi=120)
print(f"Saved {picture}")
plt.show()
