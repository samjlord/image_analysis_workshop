"""Deconvolve the Cy3 channel of the BPAE z-stack with the PSF from make_airy_psf.py.

Make the PSF first (once), then run this from the top folder of the repository:

    uv run src/deconvolution/make_airy_psf.py
    uv run src/deconvolution/deconvolve_gradient_consensus.py

The first command saves PSF_605nm.tif in the top folder, where this script
looks for it.  If your copy of the data is somewhere else, give the folder
(and, if you like, a different PSF file):

    uv run src/deconvolution/deconvolve_gradient_consensus.py "D:/my data/BPAE_100x_ZStack_3Crop"
    uv run src/deconvolution/deconvolve_gradient_consensus.py "D:/my data/BPAE_100x_ZStack_3Crop" PSF_605nm.tif

The method is Richardson-Lucy deconvolution, adapted from the Colab notebook by
James Manton (MRC LMB) and colleagues (deconvolution/gradientConsensusDeconvolution.py).
Richardson-Lucy improves a guess of the "true" image step by step:

  1. blur the guess with the PSF: this is what the microscope would have shown
  2. compare with the recorded image (divide recorded by predicted)
  3. use that ratio to correct the guess, and repeat

Plain Richardson-Lucy never says when to stop, and if you go on too long it
starts to "deconvolve" the noise.  This version decides by itself:

  - Every step, the recorded image is split at random into two half images
    (each photon goes to one half or the other).  Both halves show the same
    sample but have different noise.
  - The guess is only corrected where both halves agree on the direction of
    the correction ("gradient consensus").  Where they disagree, the
    correction is caused by noise and is skipped.
  - When the prediction stops getting closer to the half images, more steps
    would only add noise, so it stops.
  - "Biggs-Andrews acceleration" takes larger steps in the direction the guess
    was already moving, so fewer steps are needed.

The calculations run on the graphics card (GPU) if there is one, which is much
faster.  Without a GPU it works too, but slowly: set CROP_XY below to try it on
the center of the image first.

The result is saved next to the data as a 32-bit z-stack with the same pixel
size, and maximum intensity projections of before and after are shown.
"""

import os
import re
import sys
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

# Let Apple GPUs hand calculations they don't support to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

DATA_DIR = Path("Z:/ImageAnalysisCourse/ZStack_100x_BPAE/BPAE_100x_ZStack_3Crop")
IMAGE_NAME = "BPAE_100x_ZStack_3_MMStack_Pos0.ome.tif"
CHANNEL = 0                       # Cy3 is the first channel in the stack
PSF_FILE = Path("PSF_605nm.tif")  # made by make_airy_psf.py (for Cy3: 605 nm)

# Use only the center of the image (e.g. CROP_XY = 512) to try things out quickly.
CROP_XY = None

# Stop as soon as one of the half images gets worse (True), or only when both
# do (False).  True is the safer choice and gives fewer artifacts.
PLAY_IT_SAFE = True
SEED = 0                          # makes the random splits the same every run


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psf_as_image(psf_small, shape):
    """Put the small PSF in an image of zeros of the given size, with the PSF's
    center pixel at position (0, 0, 0), as the Fourier transform needs."""
    psf = np.zeros(shape, dtype=np.float32)
    psf[tuple(slice(0, n) for n in psf_small.shape)] = psf_small
    psf = np.roll(psf, [-(n // 2) for n in psf_small.shape], axis=(0, 1, 2))
    return psf / psf.sum()


def fftconv(x, H, shape):
    """Convolve x with a kernel, using its Fourier transform H."""
    return torch.fft.irfftn(torch.fft.rfftn(x) * H, s=shape)


def kldiv(p, q):
    """Kullback-Leibler divergence: how different two images are (0 = the same)."""
    p = p + 1e-4
    q = q + 1e-4
    p = p / p.sum()
    q = q / q.sum()
    kld = p * (torch.log(p) - torch.log(q))
    return torch.nan_to_num(kld).sum().item()


def show_mips(image, recon):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image.max(axis=0), cmap="gray")
    plt.axis("off")
    plt.title("Original image MIP")
    plt.subplot(1, 2, 2)
    plt.imshow(recon.max(axis=0), cmap="gray")
    plt.axis("off")
    plt.title("Deconvolved image MIP")
    plt.tight_layout()
    plt.show()


data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
psf_file = Path(sys.argv[2]) if len(sys.argv) > 2 else PSF_FILE

# --- Load the image and the PSF -------------------------------------------------
with tifffile.TiffFile(data_dir / IMAGE_NAME) as tif:
    image_np = tif.series[0].asarray()[CHANNEL]           # axes: z, y, x
    # Pixel size and z-step (in microns), to copy to the result.
    ome = tif.ome_metadata
    xy_um = float(re.search(r'PhysicalSizeX="([\d.eE+-]+)"', ome).group(1))
    z_um = float(re.search(r'PhysicalSizeZ="([\d.eE+-]+)"', ome).group(1))
with tifffile.TiffFile(psf_file) as tif:
    psf_small = tif.asarray()
    # TIFF stores the resolution as pixels per micron, written as a fraction.
    pixels, microns = tif.pages[0].tags["XResolution"].value
    psf_xy_um = microns / pixels
    psf_z_um = tif.imagej_metadata.get("spacing", 1.0)

# The PSF must have the same pixel size and z-step as the image, or the
# deconvolution uses a PSF that is too large or too small.
if not (np.isclose(psf_xy_um, xy_um, rtol=0.01) and np.isclose(psf_z_um, z_um, rtol=0.01)):
    print(f"Warning: the PSF has pixel size {psf_xy_um:.4f} um and z-step {psf_z_um} um, "
          f"but the image has {xy_um} um and {z_um} um.\n"
          f"Make a new PSF with: --pixel-size {xy_um} --z-step {z_um}")

if CROP_XY:
    y0 = (image_np.shape[1] - CROP_XY) // 2
    x0 = (image_np.shape[2] - CROP_XY) // 2
    image_np = image_np[:, y0:y0 + CROP_XY, x0:x0 + CROP_XY]

device = pick_device()
print(f"Image: {image_np.shape[0]} x {image_np.shape[1]} x {image_np.shape[2]} pixels "
      f"(z, y, x), PSF: {psf_file.name}, calculating on: {device}")

# --- Prepare the calculation --------------------------------------------------------
image = torch.tensor(image_np.astype(np.float32), device=device)
psf = torch.tensor(psf_as_image(psf_small, image_np.shape), device=device)
shape = image.shape

# Fourier transform of the PSF (the "OTF"), its mirror image, and the two combined
otf = torch.fft.rfftn(psf)
otfT = torch.conj(otf)
otfotfT = otf * otfT
del psf

generator = torch.Generator(device=device).manual_seed(SEED)
half = torch.full_like(image, 0.5)

# Start with a flat guess: every pixel has the average brightness of the image.
recon = torch.full_like(image, image.mean().item())
previous_recon = recon

num_iters = 0
prev_kld1 = np.inf
prev_kld2 = np.inf
start_time = timeit.default_timer()

# Biggs-Andrews acceleration: the last two changes of the guess
g2 = torch.zeros_like(recon)
g1 = torch.zeros_like(recon)

# --- Richardson-Lucy iterations -------------------------------------------------------
while True:
    iter_start_time = timeit.default_timer()

    # Split recorded image into 50:50 images
    split1 = torch.binomial(image, half, generator=generator)
    split2 = image - split1

    # Biggs-Andrews acceleration: step further in the direction we were moving
    alpha = ((g1 * g2).sum() / (g2 * g2).sum()).item()
    alpha = 0.0 if np.isnan(alpha) else min(max(alpha, 0.0), 1.0)
    y = recon + alpha * (recon - previous_recon)

    # Calculate prediction: what the microscope would show for this guess
    Hu = fftconv(y, otf, shape)

    # Calculate KL divergences and stop iterations if they have increased
    kldim = kldiv(Hu, image)
    kld1 = kldiv(Hu, split1)
    kld2 = kldiv(Hu, split2)
    if PLAY_IT_SAFE:
        stop = kld1 > prev_kld1 or kld2 > prev_kld2
    else:
        stop = kld1 > prev_kld1 and kld2 > prev_kld2
    if stop:
        recon = previous_recon
        print(f"Optimum result obtained after {num_iters - 1} iterations with a total "
              f"time of {timeit.default_timer() - start_time:.1f} seconds.")
        break
    prev_kld1 = kld1
    prev_kld2 = kld2

    # Biggs-Andrews acceleration
    previous_recon = recon
    recon = y

    # Calculate updates for split images and full images (H^T (d / Hu))
    HTratio1 = fftconv(split1 / (0.5 * (Hu + 1e-12)), otfT, shape)
    del split1
    HTratio2 = fftconv(split2 / (0.5 * (Hu + 1e-12)), otfT, shape)
    del split2
    # Average of the two (the Colab code adds them, which makes the result
    # twice as bright and stops the consensus test below from working).
    HTratio = 0.5 * (HTratio1 + HTratio2)
    del Hu

    # Calculate gradient consensus and blur with the interaction kernel, i.e. H^T H
    consensus_map = fftconv((HTratio1 - 1) * (HTratio2 - 1), otfotfT, shape)
    del HTratio1, HTratio2

    # Update our reconstruction only where the blurred consensus map says we
    # should (and don't allow negative brightness)
    recon = torch.where(consensus_map < 0, recon, recon * HTratio).clamp(min=0)
    del consensus_map

    # Biggs-Andrews acceleration
    g2 = g1
    g1 = recon - y
    del y

    min_HTratio = HTratio.min().item()
    max_HTratio = HTratio.max().item()
    max_relative_delta = ((recon - previous_recon) / recon.max()).max().item()
    del HTratio

    num_iters += 1
    print(f"Iteration {num_iters:03d} completed in "
          f"{timeit.default_timer() - iter_start_time:.3f} s. "
          f"KLDs = {kldim:.4f} (image), {kld1:.4f} (split 1), {kld2:.4f} (split 2). "
          f"Update range: {min_HTratio:.2f} to {max_HTratio:.2f}. "
          f"Largest relative delta = {max_relative_delta:.5f}.")

# --- Save the result with the same pixel size as the original ---------------------------
recon_np = recon.cpu().numpy().astype(np.float32)
output = data_dir / f"{Path(IMAGE_NAME).name.removesuffix('.ome.tif')}_Cy3_deconvolved.tif"
tifffile.imwrite(output, recon_np, imagej=True, resolution=(1 / xy_um, 1 / xy_um),
                 metadata={"axes": "ZYX", "spacing": z_um, "unit": "micron"})
print(f"Saved {output}")

show_mips(image_np, recon_np)
