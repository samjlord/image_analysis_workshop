# Segmentation (again)

## Illumination correction
Something that we glossed over in the previous section is illumination (or flatfield) correction.  Even in the best adjusted microscope, the illumination intensity (and the dection efficiency) is different in different parts of the image.  Most often, things are bright in the center and dimmer at the edges.  You can often see this by taking an image with your object of interest in the center, estimating (or measuring) its intenisty, the move it to a corner and repeat the estimate.  

This is obviously disastrous when you quantify things in your image.  They better have the same intensity independent of where in the image they are.  

To improve things we often apply "flatfield correction".  This relies on having perfect knowlegde of the combination of illumination and detection efficiency at each pixel.  In the CALM, we create such "flatfield" image using the best possible approximation of a uniform, thin, fluorescent sample.  These consist of highly concentrated fluorescent dye solution, so concentrated that the absorb most of the excitation light within the first micron or less of the sample, hence only a thin film is fluoresceing.  This approach was first suggested by [Michael Model](http://www.ncbi.nlm.nih.gov/pubmed/11500847). For a full protocol, see [the CALM website](https://calm.ucsf.edu/how-acquire-flat-field-correction-images).

When we consider this flatfield image (FI), the background image (BI the "dark" image of the camera), the measured image ('MI") and the "ideal", corrected image (CI) that we would like to have, we can write the following equation:

> MI = (CI * FI) + BI

Rewriting this to solve for CI, we get:

> CI = (MI - BI) / FI


When doing this calculation, it is nice to normalize the FI image to 1 (i.e. convert it to a floating point image and divide the image by its mean).  That way the mean intensity of the CI stays the same as that of the MI.  

* Ask your AI coding assistant to analyze the nuclei in the DAPI channel of the image(s) 3Channel_Rw_25Pos_1, with and without Flatfield correction with the files available in the folder Flatfield (both of these are in our shared Box folder "image_analysis_workshop_2026).  Ask for a histogram of the background corrected summed DAPi intensity.  Is it better to calculate the Otsu threshold per image or one threshold for the complete dataset? Also, you can ask for local background subtraction, for instance by using the mdeian value of a ring 10-20 pixels out of the nuclei (taking care to avoid nearby nuclei and their rings).  Which approach gives you a histogram where you can discern G1 and G2 nuclei most faithfully? 

Another approach is to use a "White Top-Hat" filter.  This procedure works by erosion followed by dilation of the original greyscale image to remove objects smaller than the filter size (therefore, you need to carefully consider the needed filter size).  The resulting image represents a good estimate of the background and can be used to subtract from the orginal image. It works best when objects are not that large. 


## Segmentation using trained networks
During the previous session you have seen how you can combine convolution filters to create an image that can be easily segmented into the objects that you are ineterested in.  Choosing the correct (sequencee) of filters and selecting a threshold is laborious and often not transferable from one image to the next.  Computer vision scientist have long worked on methods to automate this process so that a single operation can produce the desired segmentation.

To do so, researchers used annotated data sets, consisting of pairs of input images and the desired (segmented) result.  Initially, gigantic datasets were needed to produce the desired results.  Those algorithms were often fragile and did not work with images slightly different from what was in the training set.  A big breakthrough was the design of a ["U-Net"](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) architecture.  In a U-Net, the image is convoluted and downsamples in a couple of steps to a compressed, smaller image, and then convluted and upsampled.  A fuller and much better description of this architecture can be found [here](https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/) and in the original paper.  A huge advantage of this architecture is that the size of the training set can be greatly reduced, and training sets of 10s of images can be enough to give useful results.

Many deep learning tools based on U-Net have been released.  Possibly the most widely used one is [Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v2.full) which not only excelled at segmenting cells from microscope images it had never seen before, but also was released with a nice user interface including tools to fine-train Cellpose for images on which it did not directly yield the desired results. 

Deep learning tools such as Cellpose run much faster on a GPU than a CPU, and GPU usage is therefore highly desirable.  However, this causes extra complications, as the software needed to use GPU accelaration will depend on your computer and OS, the GPU in the computer, and the needs of the deep learning tool (i.e. which underlying library they use, such as torch - PyTorch, or Tensorflow).  

* Ask your AI assistant if GPU acceleration of Cellpose is possible on your computer, and follow its instruction to get AI acceleration (or let it carry out the installation, but please do be aware of what hets installed). This may not be needed on a Mac, but why not confirm?

* Ask your AI assistant to write a script that uses Cellpose to segment the image testimages/3CChannel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif.  Use the DAPI channel to get the nucleic, the FITC channel to segment the cytoplams.  Show the original image the the two masks in napari.

* Note: Cellpose is both a library and an application.  You can run the Cellpose GUI by typing: `uv run cellpose`.  This GUI is especially helpful if you want to finetune existing models to work better on your data.
