# PEL (Pygame Effect Library)
### Under active development 


## What is PEL, 
```
PEL is an open source project (MIT License) written in python containing 
special effects, image processing tools and game methods to be use in 
addition to PYGAME library.

Amongst the image processing tools, PEL contains the following methods:
Sobel, Feldman, Canny filter, Prewit algorithms, Gaussian blur and image 
sharpening tools, Sepia, Grayscale, Hue shift, control over the luminescence
and saturation.

By default, PEL will works with the following file extensions (if pygame is
built with full image support):
JPG, PNG, GIF (non-animated), BMP, PCX, 
TGA (uncompressed), TIFF, LBM, PBM, XPM 

Saving images only supports a limited set of formats:
BMP, TGA, PNG, JPEG

PEL was originally written in python then ported into CYTHON and C programming
language to increase overall performances to allow real time rendering. 

Most of PEL algorithms are iterating over all the surface’s pixels to apply 
transformations (raster type images). 
As a result, processing large image sizes will have a significant impact on
the total processing time.
 
You can boost the overall performances by setting a variable to use 
multiprocessing before compiling the project. This will enable full potential
of multiprocessing (OPENMP, open multiprocessing programming) when using PEL 
tools.
It is highly recommended to use the multi-processing option if your CPU has 
at least 8 threads (most of the threads will be used intensively during image
processing) leaving your system slightly un-responsive if the number of threads
is not high enough. 

If you are using PEL for image processing you can safely set the variable 
to use multiprocessing capabilities to modify images as quick as possible.
```
## In addition, PEL provides the following methods:
```
Add/remove transparency to image
Surface scrolling (horizontal/vertical, right/left)
Create gradient surface
Blending textures
Transition effect,
Alpha blending,
Image inversion, 
Image wave effect
Wobbly texture effect
Swirl effect
Water ripple effect
Plasma effect
Light effect
Image RGB filtering,
Greyscale methods, 
Dithering, 
Bilinear filter,
Image rescaling,
RGB split channel, 
Heat flux effect, 
Median filter, 
Bilateral filter,
Colour reduction, 
Glitch effect, 
Fisheye, 
```
## And soon to be added:
```
Alien writing, 
tv turned off, 
half tone filter, 
BOID algorithm, 
dolly zoom effect, 
long exposure effect, 
lens effect, zoom in, zoom out, 
elastic collision algorithm
PROGRESSIVE ZOOM 
LENS EFFECT on texture
CAMERA LENS EFFECT
PIXEL EFFECT
CROP IMAGE 
ELECTRICITY EFFECT
WELL COLLAPSING EFFECT
TELEPORTATION EFFECT
WIND EFFECT
RAIN EFFECT
FRACTAL
PERLIN NOISE
NEON
Fire effect, 
Moiré effect, 
Tunnel effect, 
Snow effect, 
Icon glowing effect, 
Cartoonish effect, 
DOG difference of Gaussian,
Lateral / vertical scan effect, 
```
## Requirements
```
PYTHON 3, 
PYGAME version 1.8 
CYTHON
OPENCV
NUMPY
And a C compiler.
```

### Transparency methods
```
def make_transparent(image_: Surface, alpha_: int)->Surface
def make_transparent_buffer(image_: Surface, alpha_: int)->Surface
def make_array_transparent(rgb_array_: ndarray, alpha_array_: ndarray, alpha_: int)->Surface
def transparent_mask(image: pygame.Surface, mask_alpha: numpy.ndarray)
```
### Opacity methods
```
def make_opaque(image_:Surface, alpha_: int) -> Surface
def make_opaque_buffer(image_:Surface, alpha_: int) -> Surface
def make_array_opaque(rgb_array_:ndarray, alpha_array_:ndarray, alpha_: int) -> Surface
```
### Blink surface 
```
def blink32(image_: Surface, alpha_: int) -> Surface
def blink32_mask(image_: Surface, alpha_: int) -> Surface
```
### Filtering RGB values 
```
def low_th_alpha(surface_: Surface, new_alpha_: int, threshold_: int) -> Surface
def high_th_alpha(surface_: Surface, new_alpha_: int, threshold_: int) -> Surface
```
### Greyscale methods
```
# Conserve lightness
def greyscale_light_alpha(image: Surface)->Surface
def greyscale_light(image: Surface)->Surface

# Conserve luminosity 
def greyscale_lum_alpha(image: Surface)->Surface
def greyscale_lum(image: Surface)->Surface

# Average values
def make_greyscale_32(image: Surface)->Surface
def make_greyscale_24(image: Surface)->Surface
```
