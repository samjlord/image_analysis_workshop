# Deconvolution

Microscope images are produced by convolution of an object with the Point Spread Function of the microscope.  A Point Spread Function (PSF) defines how the microscope images a very small point (significantly smaller than the resolution of the miroscope).  The PSF depends very much on the objective lens used - especially on the numerical aperture of the objective lens - but also on other parameters, such as the magnification to the camera, pixel size of the camera, the refractive index of the sample, and how deep in the sample the point is located.  Since the image you acquire is a convolution of the object with the PSF, it seems worthwhile to think about deconvolving the image with the PSF to better approximate the object that you are interested in.

We learned that convolution in real space is multiplication in Fourier (frequency) space, hence deconvolution is division of the image with the PSF in frequency space.  Since real images (and the PSF) contain noise, divisions in frequency space lead to wildly exploding noise.  To get around this problem, deconvolution algorithms often start with a guess about the true object, convolve this guess with the microscope PSF to find out what the microscope would see with this object, and compare the image to the actual image.  The algorithm then improves its estimate of the object, convolves this with the PSF, and iteratively constructs the best guess for the object through error minimization.  Often, some real-world physical rules, such as a positivity constrains (there is no such thing as negative light intensity) are used.




