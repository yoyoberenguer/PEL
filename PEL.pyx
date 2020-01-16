###cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

"""
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


"""

# LINKS http://seancode.com/demofx/
# https://tympanus.net/codrops/2016/05/03/animated-heat-distortion-effects-webgl/
# https://tympanus.net/codrops/2017/10/10/liquid-distortion-effects/
# fire effect -> http://lodev.org

# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, dstack, full, ones,\
    asarray, ascontiguousarray
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")
    raise SystemExit

# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    print("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")
    raise SystemExit

cimport numpy as np

# OPENCV IS REQUIRED
try:
    import cv2
except ImportError:
    print("\n<cv2> library is missing on your system."
          "\nTry: \n   C:\\pip install opencv-python on a window command prompt.")
    raise SystemExit


# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d
    from pygame.image import frombuffer

except ImportError:
    print("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")
    raise SystemExit

cimport numpy as np
import random
from random import randint
import math

from libc.math cimport sin, sqrt, cos, atan2, pi, round, floor, fmax, fmin, pi, tan, exp, ceil, fmod
from libc.stdio cimport printf
from libc.stdlib cimport srand, rand, RAND_MAX, qsort, malloc, free, abs


from RippleEffect import droplet_float, droplet_int, droplet_grad

__author__ = "Yoann Berenguer"
__credits__ = ["Yoann Berenguer"]
__version__ = "1.0.0 untested"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"

DEF OPENMP = True


# num_threads – The num_threads argument indicates how many threads the team should consist of.
# If not given, OpenMP will decide how many threads to use.
# Typically this is the number of cores available on the machine. However,
# this may be controlled through the omp_set_num_threads() function,
# or through the OMP_NUM_THREADS environment variable.
if OPENMP == True:
    DEF THREAD_NUMBER = 8
else:
    DEF THREAD_NUMNER = 1


# static:
# If a chunksize is provided, iterations are distributed to all threads ahead of
# time in blocks of the given chunksize. If no chunksize is given, the iteration
# space is divided into chunks that are approximately equal in size,
# and at most one chunk is assigned to each thread in advance.
# This is most appropriate when the scheduling overhead matters and the problem can be
# cut down into equally sized chunks that are known to have approximately the same runtime.

# dynamic:
# The iterations are distributed to threads as they request them, with a default chunk size of 1.
# This is suitable when the runtime of each chunk differs and is not known in advance and
# therefore a larger number of smaller chunks is used in order to keep all threads busy.

# guided:
# As with dynamic scheduling, the iterations are distributed to threads as they request them,
# but with decreasing chunk size. The size of each chunk is proportional to the number of
# unassigned iterations divided by the number of participating threads,
# decreasing to 1 (or the chunksize if provided).
# This has an advantage over pure dynamic scheduling when it turns out that the last chunks
# take more time than expected or are otherwise being badly scheduled, so that
# most threads start running idle while the last chunks are being worked on by
# only a smaller number of threads.

# runtime:
# The schedule and chunk size are taken from the runtime scheduling variable,
# which can be set through the openmp.omp_set_schedule() function call,
# or the OMP_SCHEDULE environment variable. Note that this essentially
# disables any static compile time optimisations of the scheduling code itself
# and may therefore show a slightly worse performance than when the same scheduling
# policy is statically configured at compile time. The default schedule is implementation defined.
# For more information consult the OpenMP specification [1].
DEF SCHEDULE = 'static'

DEF HALF = 1.0/2.0
DEF ONE_THIRD = 1.0/3.0
DEF ONE_FOURTH = 1.0/4.0
DEF ONE_FIFTH = 1.0/5.0
DEF ONE_SIXTH = 1.0/6.0
DEF ONE_SEVENTH = 1.0/7.0
DEF ONE_HEIGHT = 1.0/8.0
DEF ONE_NINTH = 1.0/9.0
DEF ONE_TENTH = 1.0/10.0
DEF ONE_ELEVENTH = 1.0/11.0
DEF ONE_TWELVE = 1.0/12.0
DEF ONE_255 = 1.0/255.0
DEF ONE_360 = 1.0/360.0
DEF TWO_THIRD = 2.0/3.0

DEF DEG_TO_RAD = 3.14159265359 / 180.0
DEF RAD_TO_DEG = 180.0 / 3.14159265359

cdef:
    float [360] MSIN = numpy.zeros(360, dtype=numpy.float32)
    float [360] MCOS = numpy.zeros(360, dtype=numpy.float32)
    int i = 0

# Pre-calculate sin and cos for angle [0...360] degrees
for i in range(360):
    MCOS[i] = <float>(cos(1 * DEG_TO_RAD))
    MSIN[i] = <float>(sin(1 * DEG_TO_RAD))


cdef struct color_tuple:
    unsigned char red
    unsigned char green
    unsigned char blue
ctypedef color_tuple t_color



cdef extern from 'library.c' nogil:
    double distance (double x1, double y1, double x2, double y2)
    double gaussian (double v, double sigma)
    int * quickSort(int arr[], int low, int high)
    double * rgb_to_hsv(double red, double green, double blue)
    double * hsv_to_rgb(double h, double s, double v)
    double * rgb_to_hls(double r, double g, double b)
    double * hls_to_rgb(double h, double l, double s)
    double fmax_rgb_value(double red, double green, double blue)
    double fmin_rgb_value(double red, double green, double blue)
    unsigned char max_rgb_value(unsigned char red, unsigned char green, unsigned char blue);
    unsigned char min_rgb_value(unsigned char red, unsigned char green, unsigned char blue);
    unsigned char umax_(unsigned char a, unsigned char b);



#***********************************************
#**********  METHOD HSV TO RGB   ***************
#***********************************************
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# CYTHON
# cython version of hsv to rgb and rgb to hsv conversion model,
# are offering much better performances than
# colorsys methods.
# The C version of those two technics are offering by far the best
# performances.
def hsv2rgb(h: float, s: float, v: float):
    cdef float *rgb
    rgb = hsv2rgb_c(h, s, v)
    return rgb[0], rgb[1], rgb[2]
# CYTHON
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rgb2hsv(r: float, g: float, b: float):
    cdef float *hsv
    hsv = rgb2hsv_c(r, g, b)
    return hsv[0], hsv[1], hsv[2]
# C VERSION
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rgb_to_hsv_c(r, g, b):
    cdef double *hsv
    hsv = rgb_to_hsv(r, g, b)
    return hsv[0], hsv[1], hsv[2]
# C VERSION
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def hsv_to_rgb_c(r: float, g: float, b: float):
    cdef double *hsv
    hsv = hsv_to_rgb(r, g, b)
    return hsv[0], hsv[1], hsv[2]


#***********************************************
#**********  METHOD RGB TO HLS   ***************
#***********************************************
# CYTHON
# cython version of hls to rgb and rgb to hls conversion model,
# are offering much better performances than
# colorsys methods.
# The C version of those two technics are offering by far the best
# performances.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rgb2hls(r: float, g: float, b: float):
    cdef float *hls
    hls = rgb2hls_c(r, g, b)
    return hls[0], hls[1], hls[2]
# CYTHON
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def hls2rgb(h:float , l: float, s: float):
    cdef float *rgb
    rgb = hls2rgb_c(h, l, s)
    return rgb[0], rgb[1], rgb[2]
# C VERSION
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rgb_to_hls_c(r: float, g: float, b: float):
    cdef double *hls
    hls = rgb_to_hls(r, g, b)
    return hls[0], hls[1], hls[2]
# C VERSION
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def hls_to_rgb_c(h:float , l: float, s: float):
     cdef double *rgb
     rgb = hls_to_rgb(h, l, s)
     return rgb[0], rgb[1], rgb[2]

#***********************************************
#**********  METHOD TRANSPARENCY ***************
#***********************************************
# Create transparent/opaque texture 
# 3 methods for controlling transparency and opacity of an image
# 1) All pixels.
# 2) Value (threshold) control pixels that needs to be adjusted.
# 3) Use a mask to determine parts of the image that will be affected 


# ADD TRANSPARENCY -----------------------------
def make_transparent(image_: Surface, alpha_: int)->Surface:
    return make_transparent_c(image_, alpha_)

# Use a buffer instead of numpy arrays (faster)
# image/texture has to be converted with convert_alpha()
# Return an BGRA texture if conversion is omitted.
def make_transparent_buffer(image_: Surface, alpha_: int)->Surface:
    return make_transparent_b(image_, alpha_)

def make_array_transparent(rgb_array_: ndarray, alpha_array_: ndarray, alpha_: int)->Surface:
    return make_array_transparent_c(rgb_array_, alpha_array_, alpha_)

def transparent_mask(image: pygame.Surface, mask_alpha: numpy.ndarray):
    return transparent_mask_c(image, mask_alpha)


# ADD OPACITY ----------------------------------
def make_opaque(image_:Surface, alpha_: int) -> Surface:
    return make_opaque_c(image_, alpha_)

# Image/texture have to be converted with convert_alpha()
# If the conversion is omitted, output image will be BGRA format.
def make_opaque_buffer(image_:Surface, alpha_: int) -> Surface:
    return make_opaque_b(image_, alpha_)

def make_array_opaque(rgb_array_:ndarray, alpha_array_:ndarray, alpha_: int) -> Surface:
    return make_array_opaque_c(rgb_array_, alpha_array_, alpha_)

# Change pixels with alpha value = 255 only (preserve transparency mask).
# Value are not cap to [0..255] allowing value to shift from 255 to 0 and vice versa creating 
# an illusion of blinking when animated.
# Compatible only with 32-bit surface (use convert_alpha) prior processing.
def blink32(image_: Surface, alpha_: int) -> Surface:
    return blink32_b(image_, alpha_)

# change alpha value for pixels RGBA with alpha != 255
# This method allow to change only the transparency part of an
# image (transparency mask) and not the visible pixels.
# This method is the exact opposite of blink32
def blink32_mask(image_: Surface, alpha_: int) -> Surface:
    return blink32_mask_b(image_, alpha_)

# FILTERING RGB VALUES -------------------------
def low_th_alpha(surface_: Surface, new_alpha_: int, threshold_: int) -> Surface:
    return low_threshold_alpha_c(surface_, new_alpha_, threshold_)

def high_th_alpha(surface_: Surface, new_alpha_: int, threshold_: int) -> Surface:
    return high_threshold_alpha_c(surface_, new_alpha_, threshold_)


#***********************************************
#**********  METHOD GREYSCALE ******************
#***********************************************

# CONSERVE LIGHTNESS ----------------------------
def greyscale_light_alpha(image: Surface)->Surface:
    return greyscale_lightness_alpha_c(image)

def greyscale_light(image: Surface)->Surface:
    return greyscale_lightness_c(image)


# CONSERVE LUMINOSITY --------------------------
# compatible 32-bit 
def greyscale_lum_alpha(image: Surface)->Surface:
    return greyscale_luminosity_alpha_c(image)
# compatible 24-bit
def greyscale_lum(image: Surface)->Surface:
    return greyscale_luminosity_c(image)
 
# AVERAGE VALUES --------------------------------
# compatible 32-bit
def make_greyscale_32(image: Surface)->Surface:
    return make_greyscale_32_c(image)
# compatible 24-bit
def make_greyscale_24(image: Surface)->Surface:
    return make_greyscale_24_c(image)

# GREYSCALE ARRAYS ------------------------------
def make_greyscale_altern(image: Surface)->Surface:
    return make_greyscale_altern_c(image)

# 3d array to surface
# in : RGB array shape (width, height, 3)
# out: Greyscale pygame surface 
def greyscale_arr2surf(array_: ndarray)->Surface:
    return greyscale_arr2surf_c(array_)

# in : RGB array shape (width, height, 3)
# out: greyscale array (width, height, 3)
def greyscale_array(array_: ndarray)->ndarray:
    return greyscale_array_c(array_)

# in : RGB array shape (width, height, 3)
# out: greyscale 2d array shape (width, height)
def greyscale_3d_to_2d(array_: ndarray)->ndarray:
    return greyscale_3d_to_2d_c(array_)

# in : 2d array shape (width, height)
# out: greyscale 3d array shape (width, height, 3)
def greyscale_2d_to_3d(array_: ndarray)->ndarray:
    return greyscale_2d_to_3d_c(array_)

# TODO: need testing
def greyscale_mask(image: pygame.Surface, mask_array: numpy.ndarray):
    raise NotImplemented

# TODO NOT TESTED
# in : pygame Surface (color)
# out: tuple (greyscale surface, 1d buffer )
def buffer_greyscale(image: pygame.Surface)->tuple:
    return buffer_greyscale_c(image)


#-----------------------------------------------
#***********************************************
#*******  BLACK & WHITE TRANSFORM  *************
#***********************************************
# compatible 24-bit
def bw_surface24(image: pygame.Surface)->tuple:
    return bw_surface24_c(image)

# compatible 32-bit
def bw_surface32(image: pygame.Surface)->tuple:
    return bw_surface32_c(image)

# in : 3d or 2d array (RGB colors or greyscale values)
# out: Black and white 3d or 2d array (Depends on input array)
def bw_array(array: numpy.ndarray)->numpy.ndarray:
    return bw_array_c(array)             

#-----------------------------------------------
#***********************************************
#**********  METHOD COLORIZE  ******************
#***********************************************

# --------- functions uses buffer --------------------
def redscale_buffer(image: Surface)->Surface:
    return redscale_b(image)

# compatible 32-bit
def redscale_alpha_buffer(image: Surface)->Surface:
    return redscale_alpha_b(image)

# compatible 24-bit
def greenscale_buffer(image: Surface)->Surface:
    return greenscale_b(image)

# compatible 32-bit
def greenscale_alpha_buffer(image: Surface)->Surface:
    return greenscale_alpha_b(image)

# compatible 24-bit
def bluescale_buffer(image: Surface)->Surface:
    return bluescale_b(image)

# compatible 32-bit
def bluescale_alpha_buffer(image: Surface)->Surface:
    return bluescale_alpha_b(image)

# ---- Same functions but use arrays --------------

def redscale(image: Surface)->Surface:
    return redscale_c(image)
 
def redscale_alpha(image: Surface)->Surface:
    return redscale_alpha_c(image)

def greenscale(image: Surface)->Surface:
    return greenscale_c(image)

def greenscale_alpha(image: Surface)->Surface:
    return greenscale_alpha_c(image)

def bluescale(image: Surface)->Surface:
    return bluescale_c(image)

def bluescale_alpha(image: Surface)->Surface:
    return bluescale_alpha_c(image)

#***********************************************
#*********  METHOD LOADING PER-PIXEL  **********
#***********************************************
def load_per_pixel(file: str)->Surface:
    return load_per_pixel_c(file)

def load_image32(path: str)->Surface:
    return load_image32_c(path)


#***********************************************
#********* METHOD LOAD SPRITE SHEET ************
#***********************************************
def spritesheet_per_pixel(file_: str, chunk_: int,
                          colums_: int, rows_: int)->list:
    return spritesheet_per_pixel_c(file_, chunk_, colums_, rows_)

def spritesheet_per_pixel_fs8(file: str, chunk: int,
                              columns_: int, rows_: int, tweak_:bool=False, *args)->list:
    ...
def spritesheet_alpha(file: str, chunk: int, columns_: int,
                      rows_: int, tweak_:bool=False, *args)->list:
    ...
def spritesheet(file, int chunk, int columns_, int rows_,
                tweak_: bool = False, *args)->list:
    ...
def spritesheet_fs8(file: str, chunk: int, columns_: int,
                    rows_: int, tweak_: bool=False, *args) -> list:
    ...

def spritesheet_new(file_: str, chunk_: int, columns_: int, rows_: int):
    return spritesheet_new_c(file_, chunk_, columns_, rows_)

#***********************************************
#**********  METHOD SHADOW *********************
#***********************************************
def shadow32(image: Surface, attenuation: float)->Surface:
    return shadow_32c(image, attenuation)

def shadow32buffer(image: Surface, attenuation: float)->Surface:
    return shadow_32b(image, attenuation)

#***********************************************
#**********  METHOD MAKE_ARRAY  ****************
#***********************************************
# create a 3d array shape (w, h, 4) from RGB and ALPHA array values
# in : RGB shape (w, h, 3) and alpha (w, h) shape
# out: numpy.ndarray shape (w, h, 4). 
# All stacked values are copied into a new array keeping the exact same
# indexing (stride) and homogeneity.
def make_array(rgb_array_: ndarray, alpha_:ndarray):
    return make_array_c_code(rgb_array_, alpha_)

# Create a 3d array shape (w, h, 4) from RGB and ALPHA array values
# in : RGB shape (w, h, 3) and alpha (w, h) shape
# out: numpy.ndarray shape (w, h, 4). numpy.dstack equivalent
# All values are transpose into a new array (this method is used when the
# array need to be flipped before creating the pygame surface with pygame.frombuffer
# *pygame.frombuffer cannot flipped the surface.
def make_array_trans(rgb_array_: ndarray, alpha_:ndarray):
    return make_array_c_transpose(rgb_array_, alpha_)

# Create a 3d array shape (w, h, 4) from a bufferproxy object
# in : BufferProxy
# out: numpy.ndarray shape (w, h, 4)
def make_array_from_buffer(buffer_: BufferProxy, size_: tuple)->ndarray:
    return make_array_from_buffer_c(buffer_, size_)

#***********************************************
#**********  METHOD MAKE_SURFACE ***************
#***********************************************

def make_surface(rgba_array: ndarray) -> Surface:
    return make_surface_c(rgba_array)

def make_surface_c1(buffer_: BufferProxy, w, h)->Surface:
    return make_surface_c_1(buffer_, w, h)

def make_surface_c2(rgba_array_: ndarray)->Surface:
    return make_surface_c_2(rgba_array_)

def make_surface_c4(rgba_array_: ndarray)->Surface:
    return make_surface_c_4(rgba_array_)

def make_surface_c5(rgba_array_: ndarray)->Surface:
    return make_surface_c_5(rgba_array_)

def make_surface_c6(rgba_array_: ndarray)->Surface:
    return make_surface_c_6(rgba_array_)

#*********************************************
#**********  METHOD SPLIT RGB ****************
#*********************************************

def rgb_split(surface_: Surface)->tuple:
    return rgb_split_c(surface_)

def rgb_split_buffer(surface_: Surface)-> tuple:
    return rgb_split_b(surface_)

def rgb_split32(surface_: Surface):
    return rgb_split32_c(surface_)

def rgb_split32_buffer(surface_: Surface):
    return rgb_split32_b(surface_)

def red_channel(surface_: Surface)->Surface:
    return red_channel_c(surface_)

def green_channel(surface_: Surface)->Surface:
    return green_channel_c(surface_)

def blue_channel(surface_: Surface)->Surface:
    return blue_channel_c(surface_)


def red_channel_buffer(surface_: Surface)->Surface:
    return red_channel_b(surface_)

def green_channel_buffer(surface_: Surface)->Surface:
    return green_channel_b(surface_)

def blue_channel_buffer(surface_: Surface)->Surface:
    return blue_channel_b(surface_)

#*********************************************
#**********  SWAP CHANNELS   *****************
#*********************************************
def swap_channels(surface: Surface, model: str):
    return swap_channels_c(surface, model)

#*********************************************
#**********  METHOD FISHEYE  *****************
#*********************************************

def fish_eye(image)->Surface:
    return fish_eye_c(image)

def fish_eye_32(image)->Surface:
    return fish_eye_32c(image)

#*********************************************
#**********  METHOD ROTATE  ******************
#*********************************************
def rotate_inplace(image: Surface, angle: int)->Surface:
    return rotate_inplace_c(image, angle)

def rotate_24(image: Surface, angle: int)->Surface:
    return rotate_c24(image, angle)

def rotate_32(image: Surface, angle: int)->Surface:
    return rotate_c32(image, angle)

#*********************************************
#**********  METHOD HUE SHIFT  ***************
#*********************************************

# TODO CREATE SIMILAR METHOD WITH BUFFER
def hue_surface_24(surface_: Surface, float shift_)->Surface:
    return hue_surface_24c(surface_, shift_)

def hue_surface_32(surface_: Surface, float shift_)->Surface:
    return hue_surface_32c(surface_, shift_)

# HUE A SURFACE WITH A GIVEN MASK
# TODO NOT TESTED
def hue_mask(surface_: Surface, shift_: float, mask_array=None)->Surface:
    return hue_mask_c(surface_, shift_, mask_array)

# HUE GIVEN COLOR VALUE FROM A SURFACE 24BIT
def hue_surface_24_color(surface_: Surface, float shift_, red, green, blue)->Surface:
    return hue_surface_24_color_c(surface_, shift_, red, green, blue)

# HUE GIVEN COLOR VALUE FROM A SURFACE 32BIT
def hue_surface_32_color(surface_: Surface, float shift_, red, green, blue)->Surface:
    raise NotImplemented

# HUE PIXEL MEAN AVG VALUE OVER OR EQUAL TO A GIVEN THRESHOLD VALUE
def hsah(surface_: Surface, threshold_: int, shift_: float)->Surface:
    return hsah_c(surface_, threshold_, shift_)

# HUE PIXEL MEAN AVG VALUE BELOW OR EQUAL TO A GIVEN THRESHOLD VALUE
def hsal(surface_: Surface, threshold_: int, shift_: float)->Surface:
    return hsal_c(surface_, threshold_, shift_)

# HUE ARRAY (RED CHANNEL)
def hue_array_red(array_: ndarray, shift_: float)->ndarray:
    return hue_array_red_c(array_, shift_)
# HUE ARRAY (GREEN CHANNEL)
def hue_array_green(array: ndarray, shift_: float)->ndarray:
    return hue_array_green_c(array, shift_)
# HUE ARRAY (BLUE CHANNEL)
def hue_array_blue(array: ndarray, shift_: float)->ndarray:
    return hue_array_blue_c(array, shift_)

# --------------- BUFFERS -----------------------------
# RED CHANNEL
def hue_red24(array_: ndarray, shift_: float)->Surface:
    return hue_red24_b(array_, shift_)
def hue_red32(array_: ndarray, shift_: float)->Surface:
    return hue_red32_b(array_, shift_)

# GREEN CHANNEL
def hue_green24(array_: ndarray, shift_: float)->Surface:
    return hue_green24_b(array_, shift_)
def hue_green32(array_: ndarray, shift_: float)->Surface:
    return hue_green32_b(array_, shift_)

# BLUE CHANNEL
def hue_blue24(array_: ndarray, shift_: float)->Surface:
    return hue_blue24_b(array_, shift_)
def hue_blue32(array_: ndarray, shift_: float)->Surface:
    return hue_blue32_b(array_, shift_)


#*********************************************
#**********  METHOD BRIGHTNESS  **************
#*********************************************

def brightness_24(surface_: Surface, shift_: float)->Surface:
    return brightness_24c(surface_, shift_)

def brightness_32(surface_: Surface, shift_: float)->Surface:
    return brightness_32c(surface_, shift_)

def brightness_24_i(surface_: Surface, shift_: float)->Surface:
    return brightness_24_fast(surface_, shift_)

def brightness_32_i(surface_: Surface, shift_: float)->Surface:
    return brightness_32_fast(surface_, shift_)
# TODO NOT TESTED
def brightness_mask_32(surface_:Surface, shift_: float, mask_array=None):
    return brightness_mask_32c(surface_, shift_, mask_array)
# TODO NOT TESTED
def brightness_mask_24(surface_:Surface, shift_: float, mask_array=None):
    return brightness_mask_24c(surface_, shift_, mask_array)

#*********************************************
#**********  METHOD SATURATION  **************
#*********************************************

def saturation_24(surface_: Surface, shift_: float)->Surface:
    return saturation_24_c(surface_, shift_)

def saturation_32(surface_: Surface, shift_: float)->Surface:
    return saturation_32_c(surface_, shift_)

def saturation_mask_24(surface_: Surface, shift_: float,
                       mask_array: numpy.ndarray)->Surface:
    return saturation_mask_24_c(surface_, shift_, mask_array)

def saturation_mask_32(surface_: Surface, shift_: float,
                       mask_array: numpy.ndarray)->Surface:
    return saturation_mask_32_c(surface_, shift_, mask_array)

#*********************************************
#**********  METHOD ROLL/SCROLL  *************
#*********************************************

# ROLL ARRAY 3D TYPE (W, H, 3) NUMPY.UINT8
def scroll_array(array: ndarray, dy: int=0, dx: int=0) -> ndarray:
    return scroll_array_c(array, dy, dx)

# ROLL ARRAY 3D TYPE (W, H, 4) NUMPY.UINT8
def scroll_array_32(array: ndarray, dy: int=0, dx: int=0) -> ndarray:
    return scroll_array_32_c(array, dy, dx)

# USE NUMPY LIBRARY (NUMPY.ROLL)
def scroll_surface_org(array: ndarray, dx: int=0, dy: int=0)->tuple:
    ...

# Roll the value of an entire array (lateral and vertical)
# Identical algorithm (scroll_array) but returns a tuple (surface, array)
def scroll_surface(surface: pygame.Surface, dy: int=0, dx: int=0)->tuple:
    return scroll_surface_c(surface, dy, dx)

# ROLL IMAGE TRANSPARENCY INSTEAD
def scroll_surface_alpha(surface: pygame.Surface, dy: int=0, dx: int=0) -> tuple:
    return scroll_surface_alpha_c(surface, dy, dx)


#*********************************************
#**********  METHOD GRADIENT  ****************
#*********************************************
# Create an horizontal array filled with gradient color, shape(w, h) and w > h
def gradient_horizarray(width: int, height: int, start_color: tuple, end_color: tuple)->ndarray:
    return gradient_horizarray_c(width, height, start_color, end_color)
# Create a vertical array filled with gradient color shape(w, h) and h > w
def gradient_vertarray(width: int, height: int, top_value: tuple, bottom_value: tuple)-> ndarray:
    return gradient_vertarray_c(width, height, top_value, bottom_value)

def gradient_horiz_2darray(width: int, height: int, start_value: int, end_value: int)-> ndarray:
    return gradient_horiz_2darray_c(width, height, start_value, end_value)

def gradient_vert_2darray(width: int, height: int, top_value: int, bottom_value: int)-> ndarray:
    return gradient_vert_2darray_c(width, height, top_value, bottom_value)

#*********************************************
#**********  METHOD BLENDING  ****************
#*********************************************
def blend_texture(surface_: Surface, max_steps: int,
                  final_color_:(tuple, Color), goto_value: int) -> Surface:
    return blend_texture_c(surface_, max_steps, final_color_, goto_value)

def blend_2_textures(source_: Surface,
                     destination_: Surface,
                     steps: int,
                     lerp_to_step: int) -> Surface:
    return blend_2_textures_c(source_, destination_, steps, lerp_to_step)

def alpha_blending(surface1: Surface, surface2: Surface) -> Surface:
    return alpha_blending_c(surface1, surface2)

def alpha_blending_static(surface1: Surface, surface2: Surface, float a1, float a2)->Surface:
    return alpha_blending_static_c(surface1, surface2, a1, a2)

#*********************************************
#**********  METHOD INVERT  ******************
#*********************************************
def invert_surface_24bit(image: Surface)->Surface:
    return invert_surface_24bit_c(image)

def invert_surface_32bit(image: Surface)->Surface:
    return invert_surface_32bit_c(image)

def invert_array_24(array_: ndarray)->ndarray:
    return invert_array_24_c(array_)

def invert_array_32(array_: ndarray)->ndarray:
    return invert_array_32_c(array_)
# --------------- BUFFERS --------------------
def invert_surface_24bit_buffer(image: Surface)->Surface:
    return invert_surface_24bit_b(image)

def invert_surface_32bit_buffer(image: Surface)->Surface:
    return invert_surface_32bit_b(image)

#*********************************************
#**********  KERNEL OPERATION  *************** 
#*********************************************
def kernel_deviation(sigma: float, kernel_size: int)->ndarray:
    return deviation(sigma, kernel_size)

# Sharp filter
# Sharpening an image increases the contrast between bright and dark regions
# to bring out features. The sharpening process is basically the application
# of a high pass filter to an image. The following array is a kernel for a
# common high pass filter used to sharpen an image.
# This method cannot use a defined kernel.
# The border pixels are convolved with the adjacent pixels.
# The final image will have the exact same size than the original 
def sharpen3x3(image:Surface)-> ndarray:
    return sharpen3x3_c(image)

def sharpen3x3_mask():
    raise NotImplemented

def sharpen5x5(image: Surface)-> ndarray:
    raise NotImplemented

# Blur effect
def guaussian_boxblur3x3(image: Surface)->ndarray:
    return guaussian_boxblur3x3_c(image)

def guaussian_boxblur3x3_approx(image: Surface)->ndarray:
    return guaussian_boxblur3x3_capprox(image)

def guaussian_blur3x3(image: Surface)->ndarray:
    return guaussian_blur3x3_c(image)

def gaussian_blur5x5(rgb_array: ndarray):
    return gaussian_blur5x5_c(rgb_array)

# blur mask 
def gaussian_blur5x5_mask(rgb_array: ndarray, mask: ndarray):
    raise gaussian_blur5x5_mask_c(rgb_array, mask)

# Edge detection
def edge_detection3x3(image: Surface)->ndarray:
    return edge_detection3x3_c(image)

def edge_detection3x3_alter(image: Surface)->ndarray:
    return edge_detection3x3_c1(image)

def edge_detection3x3_fast(image: Surface)->ndarray:
    return edge_detection3x3_c2(image)

def edge_detection5x5(image: Surface)->ndarray:
    raise NotImplemented 

# http//www.sciencedirect.com
# High pass filter
# a high-pass filter enhances the high-frequency parts of an image be reducing
# the low frequency components.This type of filter can be used to sharpen image.
def highpass_3x3(image:Surface)->ndarray:
    raise highpass_3x3_c(image)

def highpass_5x5(image: Surface)->ndarray:
    raise highpass_5x5_c(image)
# Laplacien Filter
# The Laplacien filter enhances discontinuities, it outputs brighter pixel values
# as it passes over parts of the image, that have abrupt changes in intensity,
# and outputs darker values where the image in not changing rapidly.
def laplacien_3x3(image: Surface)->ndarray:
    raise laplacien_3x3_c(image)

def laplacien_5x5(image: Surface)->ndarray:
    raise laplacien_5x5(image)

# Gradient detection (Embossing)
# Changes in value over 3 pixels can be detected using kernels called gradient
# masks or prewitt masks. The filter detects changes in gradient along limited
# directions, named after points of the compass (with north equal to the up
# direction on the screen)
def emboss_3x3(image: Surface)->ndarray:
    raise NotImplemented

def emboss_5x5(image: Surface)->ndarray:
    raise NotImplemented

def motion_blur(image: Surface)->ndarray:
    """
    https://lodev.org/cgtutor/filtering.html
    double filter[filterHeight][filterWidth] =
    {
      1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1,
    };
    """
    raise NotImplemented 

#*********************************************
#**********  WATER DROP EFFECT  **************
#*********************************************

def water_effect_f(texture, frames: int, dropx: int, dropy: int, intensity: int, drops_number: int):
    return water_ripple_effect_cf(texture, frames, dropx, dropy, intensity, drops_number)

def water_effect_i(texture, frames: int, dropx: int, dropy: int, intensity: int, drops_number: int):
    return water_ripple_effect_ci(texture, frames, dropx, dropy, intensity, drops_number)

def water_ripple_effect_rand(texture, frames: int, intensity: int, drops_number: int):
    return water_ripple_effect_crand(texture, frames, intensity, drops_number)


#*********************************************
#**************  HPF/LPF  ********************
#*********************************************

def lpf(rgb_array: ndarray)->ndarray:
    ...

def hpf(rgb_array: ndarray)->ndarray:
    ...

#*********************************************
#**************  WOBBLY IMAGE ****************
#*********************************************

def wobbly_array(rgb_array: ndarray, alpha_array: ndarray, f: float)->Surface:
    return wobbly_array_c(rgb_array, alpha_array, f)

def wobbly_surface(surface: Surface, f: float)->Surface:
    return wobbly_surface_c(surface, f)

#*********************************************
#**************  SWIRL IMAGE ****************
#*********************************************

def swirl_surface(surface: Surface, degrees: float)->Surface:
    return swirl_surface_c(surface, degrees)

def swirl_surf2surf(surface: Surface, degrees: float)->Surface:
    return swirl_surf2surf_c(surface, degrees)

#*********************************************
#*************  LIGHT EFFECT  ****************
#*********************************************

# Create realistic light effect on texture/surface 
def light_area(x: int, y: int, background_rgb: ndarray, mask_alpha: ndarray)->Surface:
    return light_area_c(x, y, background_rgb, mask_alpha)

def light(rgb: ndarray, alpha: ndarray,
          intensity: float, color: ndarray)->Surface:
    return light_c(rgb, alpha, intensity, color)

# TODO TEST BELOW AND IMPLEMENT THE METHOD TO THE REST 
# def light_buffer(rgb_buffer_: BufferProxy, alpha_buffer_: BufferProxy,
#           intensity: float, color: ndarray, int w, int h)->Surface:
#     return light_b(rgb_buffer_, alpha_buffer_, intensity, color, w, h)

def light_volume(x: int, y: int, background_rgb: ndarray,
                mask_alpha: ndarray, volume: ndarray, magnitude=1e-6)->Surface:
    return light_volume_c(x, y, background_rgb, mask_alpha, volume, magnitude)

def light_volumetric(rgb: ndarray, alpha: ndarray, intensity: float,
                     color: ndarray, volume: ndarray)->Surface:
    return light_volumetric_c(rgb, alpha, intensity, color, volume)

#*********************************************
#****************  SEPIA   *******************
#*********************************************

def sepia24(surface_: Surface)->Surface:
    return sepia24_c(surface_)

# TODO TEST BELOW AND IMPLEMENT THE SAME FOR 32bit
def sepia24_buffer(surface_: Surface)->Surface:
    return sepia24_b(surface_)

def sepia32(surface_: Surface)->Surface:
    return sepia32_c(surface_)

def sepia_mask_24(surface_: Surface, mask: numpy.ndarray)-> Surface:
    return sepia24_mask_c(surface_, mask)

def sepia_mask_32(surface_: Surface, mask: numpy.ndarray)-> Surface:
    return sepia32_mask_c(surface_, mask)

# def unsepia24(surface_: Surface)->Surface:
#     return unsepia24_c(surface_)

#*********************************************
#****************  PLASMA   *******************
#*********************************************

def plasma(width: int, height: int, frame: int):
    return plasma_c(width, height, frame)

#*********************************************
#*******  COLOR REDUCTION/ OILIFY   **********
#*********************************************

def color_reduction24(surface_: Surface, factor: int)->Surface:
    return color_reduction24_c(surface_, factor)

def color_reduction32(surface_: Surface, factor: int)->Surface:
    return color_reduction32_c(surface_, factor)

#*********************************************
#**************** DITHERING  *****************
#*********************************************

def dithering24(surface_: Surface, factor: int)->Surface:
    return dithering24_c(surface_, factor)

def dithering32(surface_: Surface, factor: int)->Surface:
    return dithering32_c(surface_, factor)

#*********************************************
#************* MEDIAN FILTER *****************
#*********************************************

def median_filter24(image_: Surface, kernel_size: int)->Surface:
    return median_filter24_c(image_, kernel_size)

def median_filter32(surface_: Surface, size: int)->Surface:
    return median_filter32_c(surface_, size)

def median_filter_greyscale(path: str, size: int) ->Surface:
    return median_filter_greyscale_c(path, size)

#*********************************************
#************* BILATERAL FILTER **************
#*********************************************

def bilateral_filter24(image: Surface, sigma_s: float, sigma_i: float)->Surface:
    return bilateral_filter24_c(image, sigma_s, sigma_i)

def bilateral_filter32(image: Surface, sigma_s: float, sigma_i: float)->Surface:
    raise NotImplemented

def bilateral_filter_greyscale(path: str, sigma_s: float, sigma_i: float)->Surface:
    return bilateral_greyscale_c(path, sigma_s, sigma_i)

#*********************************************
#************** RGBA TO BGRA  ****************
#*********************************************

# ARRAY
def array_rgba_to_bgra(rgba_array: ndarray)->ndarray:
    return array_rgba2bgra_c(rgba_array)

def array_rgb_to_bgr(rgb_array: ndarray)->ndarray:
    return array_rgb2bgr_c(rgb_array)

def array_bgra_to_rgba(bgra_array: ndarray)->ndarray:
    return array_bgra2rgba_c(bgra_array)

def array_bgr_to_rgb(bgr_array: ndarray)->ndarray:
    return array_bgr2rgb_c(bgr_array)

# BUFFERS
def buffer_bgra_to_rgba(bgra_buffer: pygame.BufferProxy) -> numpy.ndarray:
    return buffer_bgra_to_rgba_c(bgra_buffer)

def buffer_rgba_to_bgra(rgba_buffer: pygame.BufferProxy) -> numpy.ndarray:
    return buffer_rgba_to_bgra_c(rgba_buffer)

def buffer_bgr_to_rgb(bgr_buffer: pygame.BufferProxy) -> numpy.ndarray:
    return buffer_bgr_to_rgb_c(bgr_buffer)

def buffer_rgb_to_bgr(rgb_buffer: pygame.BufferProxy) -> numpy.ndarray:
    return buffer_rgb_to_bgr_c(rgb_buffer)

#*********************************************
#**************** RESCALE ********************
#*********************************************

def surface_rescale(surface: pygame.Surface, w2: int, h2: int)->Surface:
    return surface_rescale_c(surface, w2, h2)

def surface_rescale_alpha(surface: pygame.Surface, w2: int, h2: int):
    return surface_rescale_alphac(surface, w2, h2)

def array_rescale24(surface: pygame.Surface, w2: int, h2: int):
    return array_rescale_24c(surface, w2, h2)

def array_rescale32(surface: pygame.Surface, w2: int, h2: int):
    return array_rescale_32c(surface, w2, h2)

#*********************************************
#********** ZOOM X2 BILINEAIRE ***************
#*********************************************

def zoomx2_bilineare_alpha(surface_: Surface)->Surface:
    return zoomx2_bilineare_alphac(surface_)

def zoomx2_bilineare(surface_: Surface)->Surface:
    return zoomx2_bilineare_c(surface_)

def zoomx2_bilineaire_grey(surface_: Surface)->Surface:
    return zoomx2_bilineaire_greyc(surface_)

#*********************************************
#************* GLITCH EFFECT *****************
#*********************************************

def horizontal_glitch(texture_: Surface, rad1_:float,
                      frequency_:float, amplitude_:float)->Surface:
    return horizontal_glitch_c(texture_, rad1_, frequency_, amplitude_)


#*********************************************
#************* WAVE EFFECT *****************
#*********************************************

def wave_xy(texture:Surface, rad: float, size: int)->Surface:
    return wave_xy_c(texture, rad, size)

def wave_xy_alpha(texture:Surface, rad: float, size: int)->Surface:
    return wave_xy_alphac(texture, rad, size)

def wave_x(texture:Surface, rad: float, size: int)->Surface:
    return wave_x_c(texture, rad, size)

def wave_y(texture:Surface, rad: float, size: int)->Surface:
    return wave_y_c(texture, rad, size)


#*********************************************
#******** HEAT WAVE (DISPLACEMENT ************
#*********************************************
# heatwave_vertical(im, grey, r, 0.0095 + (r % 2) / 1000.0, attenuation=0.10)
def heatwave_vertical(texture: Surface, mask:numpy.ndarray,
                      frequency: float, amplitude: float, attenuation: float)->Surface:
    return heatwave_vertical_c1(texture, mask, frequency, amplitude, attenuation)

def heatwave_horizontal(texture: Surface, mask:numpy.ndarray,
                        frequency: float, amplitude: float, attenuation: float)->Surface:
    return heatwave_horizontal_c1(texture, mask, frequency, amplitude, attenuation)

#**********************************************
#************* COLOR MAPPING ******************
# *********************************************

def rgb2int(red :int, green: int, blue: int):
    return rgb_to_int(red, green, blue)

# FIND WHAT IS WRONG BELOW
# def int2rgb(n):
#     cdef unsigned int *rgb
#     rgb = int_to_rgb(n)
#     return rgb

def rgba2int(red :int, green: int, blue: int, alpha: int):
    return rgba_to_int(red, green, blue, alpha)

def int2rgba(n: int):
    cdef unsigned char *rgba
    rgba = int_to_rgba(n)
    return rgba

#*********************************************
#***************** FIRE EFFECT ***************
#*********************************************
def fire_effect24(w: int, h: int, frame: int,
                  factor: float, palette: numpy.ndarray, mask: numpy.ndarray)->Surface:
    return fire_texture24(w, h, frame, factor, palette, mask)

def fire_effect32(w: int, h: int, frame: int,
                  factor: float, palette: numpy.ndarray, mask: numpy.ndarray)->Surface:
    return fire_texture32(w, h, frame, factor, palette, mask)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_transparent_c(image_, int alpha_):
    """
    The pygame surface must be a 32-bit image containing per-pixel.
    
    Add transparency to a pygame surface.
    <alpha_> must be an integer in range[0 ... 255] otherwise raise a value error.
    <alpha_> = 255, the output surface will be 100% transparent,  <alpha_> = 0, no change.
    
    The code will raise a value error if the the surface is not encoded with per-pixel transparency. 
    ValueError: Surface without per-pixel information.
    
    EXAMPLE: 
        image = pygame.image.load('your image')
        output = make_transparent(image, 100) 

    :param image_: pygame.surface, 32-bit format image containing per-pixel  
    :param alpha_: integer value for alpha channel
    :return: Return a 32-bit pygame surface containing per-pixel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |  0.0019634586
        2 FAIL MODE: | 32-bit  | convert()        |  Surface without per-pixel information.
        3 PASS MODE: | 32-bit  | convert_alpha()  |  0.001923408
        
        4 FAIL MODE: | 24-bit  |                  |  Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |  Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |  0.0018333181 
        
        7 FAIL MODE: | 8-bit   |                  |  Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |  Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |  0.0017797554 
        
    """
    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)

    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    if not image_.get_bitsize() == 32:
        raise TypeError("\nSurface without per-pixel information.")

    try:
        rgb = pixels3d(image_)
    except (pygame.error, ValueError):
        raise ValueError('\nInvalid surface.')

    try:
        alpha = pixels_alpha(image_)
    except (pygame.error, ValueError):
        raise ValueError('\nSurface without per-pixel information.')

    cdef int w, h
    w, h = image_.get_size()

    if w==0 or h==0:
        raise ValueError(
            'Image with incorrect shape, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, ::1] new_array = numpy.empty((h, w, 4), dtype=numpy.uint8)
        unsigned char [:, :] alpha_array = alpha
        unsigned char [:, :, :] rgb_array = rgb
        int i=0, j=0, a

    with nogil:

        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                new_array[j, i, 0] = rgb_array[i, j, 0]
                new_array[j, i, 1] = rgb_array[i, j, 1]
                new_array[j, i, 2] = rgb_array[i, j, 2]
                a = alpha_array[i, j] - alpha_
                if a < 0:
                    a = 0
                new_array[j, i, 3] = a

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_transparent_b(image_, int alpha_):
    """
    The pygame surface must be a 32-bit image containing per-pixel 
    and converted with pygame method convert_alpha.
    This method is much faster than make_transparent_c as it process
    alpha values from 1D buffer (loop step every 4 pixels).  
   
    All pixels will be adjusted (decremented) with the new alpha_ value. 
    If you wish not to alter the mask alpha, prefer the method set_alpha_mask
    See TEST results.
    
    EXAMPLE: 
        image = pygame.image.load('your image').convert_alpha()
        output = make_transparent(image, 100) 

    :param image_: pygame.surface, 32-bit format image containing per-pixel  
    :param alpha_: integer value for alpha channel
    :return: Return a 32-bit pygame surface containing per-pixel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |  0.0001  DO NOT USE (BGRA format pixel)
        2 PASS MODE: | 32-bit  | convert()        |  0.00083 Image lost transparency (mask alpha). A new layer
                                                     is created with default alpha value 0. Adjusting transparency 
                                                     will have no effect, already at zero. 
                                                     DO NOT USE, Image will be fully transparent
                                                       
        3 PASS MODE: | 32-bit  | convert_alpha()  |  0.00083 OK 
        
        4 FAIL MODE: | 24-bit  |                  |  FAIL Surface without per-pixel information.
        
        5 PASS MODE: | 24-bit  | convert()        |  0.00069. DO NOT USE Image lost transparency (mask alpha). 
                                                     A new layer is created with default alpha value 0. 
                                                     Adjusting transparency will have no effect (already at minimum).
                                                      
                                                     
        6 PASS MODE: | 24-bit  | convert_alpha()  |  0.00069. Image with no per-pixel transparency, A new layer alpha
                                                     is build and all values are set to 255 (fully opaque). Adjusting
                                                     transparency will affect all pixels.
        
        7 FAIL MODE: | 8-bit   |                  |  FAIL Surface without per-pixel information.
        
        8 PASS MODE: | 8-bit   | convert()        |  0.00059. DO NOT USE, Adjusting transparency will have no effect.
                                            
        9 PASS MODE: | 8-bit   | convert_alpha()  |  0.00059. Image with no per-pixel transparency, A new layer alpha
                                                     is build and all values are set to 255 (fully opaque). Adjusting
                                                     transparency will affect all pixels.
        
    """
    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)

    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    if not image_.get_bitsize() == 32:
        raise TypeError("\nSurface without per-pixel information.")

    try:
        rgba_buffer_ = image_.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef int w, h
    w, h = image_.get_size()

    if w==0 or h==0:
        raise ValueError('Image with incorrect shape, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        int b_length = rgba_buffer_.length
        unsigned char [:] rgba_buffer = numpy.frombuffer(rgba_buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, dtype=numpy.uint8)
         
        int i=0, a
        unsigned char r, g, b

    with nogil:

        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            r = rgba_buffer[i]
            g = rgba_buffer[i + 1]
            b = rgba_buffer[i + 2]

            new_buffer[i] = b
            new_buffer[i + 1] = g
            new_buffer[i + 2] = r

            a = rgba_buffer[i + 3] - alpha_
            if a < 0:
                a = 0
            new_buffer[i + 3] = a

    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_array_transparent_c(np.ndarray[np.uint8_t, ndim=3] rgb_array_,
                              np.ndarray[np.uint8_t, ndim=2] alpha_array_, int alpha_):
    """
    Increase transparency of <alpha_array_> (alpha channel - alpha_ value).
    The output image is built from RGB and ALPHA values (faster than stacking both arrays).
    <alpha_> = 255, output surface will be 100% transparent.
    <alpha_> = 0, output is identical to the source image
    
    <rgb_array_> must be a numpy.ndarray (w, h, 3) of numpy.uint8 values (RGB values of a surface)
    <alpha_array_> must must numpy.array (w, h) of numpy.uint8 values (ALPHA values of a surface).
    <alpha_> must be an integer in range [0..255] otherwise raise a value error 
    
    EXAMPLE: 
        im1 = pygame.image.load('path to your image here')
        rgb = pygame.surfarray.pixels3d(im1)
        alpha = pygame.surfarray.pixels_alpha()
        output = make_array_transparent(rgb, alpha, 100) 
    
    :param rgb_array_: numpy.ndarray containing RGB values, (w, h, 3) numpy.uint8
    :param alpha_array_: numpy.ndarray of ALPHA values (w, h) numpy.uint8
    :param alpha_: integer in range [0..250] 
    :return: Returns a 32-bit pygame surface containing per-pixel information. 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0016227643
        2 PASS MODE: | 32-bit  | convert()        |   0.0018104685  
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0014597103
        
        4 PASS MODE: | 24-bit  |                  |   0.0013248745
        5 PASS MODE: | 24-bit  | convert()        |   0.0013554132
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0012235402 
        
        7 FAIL MODE: | 8-bit   |                  |   unsupported bit depth 8 for 3D reference array
        8 PASS MODE: | 8-bit   | convert()        |   0.0017410738
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0017312480
        
        
    """
    assert isinstance(rgb_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument rgb_array_ got %s ' % type(rgb_array_)
    assert isinstance(alpha_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument alpha_ got %s ' % type(alpha_)

    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)
    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    cdef int w, h
    try:
        w, h = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('Array shape not understood...')

    if w == 0 or h == 0:
        raise ValueError('rgb_array_ with incorrect shape, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, :] rgb_array_c = rgb_array_                       # non-contiguous array
        unsigned char [:, ::1] alpha_array_c = alpha_array_                    # contiguous values
        unsigned char [: , :, ::1] new_array_c = empty((h, w, 4), dtype=uint8) # output array with contiguous values
        int i=0, j=0, a=0
    with nogil:

        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                a = alpha_array_c[i, j]
                new_array_c[j, i, 0] = rgb_array_c[i, j, 0]
                new_array_c[j, i, 1] = rgb_array_c[i, j, 1]
                new_array_c[j, i, 2] = rgb_array_c[i, j, 2]
                a = alpha_array_c[i, j] - alpha_
                if a < 0:
                    a=0
                new_array_c[j, i, 3] = a

    return pygame.image.frombuffer(new_array_c, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef transparent_mask_c(image, mask_alpha):
    """
    The pygame surface must be a 24-32-bit image (image transparency will be replaced by
    the given mask_alpha)
    
    EXAMPLE:
    transparent_mask(image, mask_alpha)

    :param image: pygame.surface, 24-32-bit format image   
    :param mask_alpha: numpy.ndarray shape (w, h) 
    :return: Return a 32-bit pygame surface containing per-pixel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0019915131
        2 PASS MODE: | 32-bit  | convert()        |   0.0016703276
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0017840079
        
        4 PASS MODE: | 24-bit  |                  |   0.0017265877
        5 PASS MODE: | 24-bit  | convert()        |   0.0016682274
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0024571839
        
        7 FAIL MODE: | 8-bit   |                  |   unsupported bit depth 8 for 3D reference array
        8 PASS MODE: | 8-bit   | convert()        |   0.0024846367
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0019488871
        
    """
    assert isinstance(image, Surface), \
        'Expecting Surface for positional argument image got %s ' % type(image)
    assert isinstance(mask_alpha, numpy.ndarray), \
        'Expecting a numpy.ndarray for positional argument mask_alpha got %s ' % type(mask_alpha)

    try:
        rgb = pixels3d(image)
    except (pygame.error, ValueError) as e:
        raise ValueError('Surface Incompatible, %s' % e)

    cdef int w, h, w2, h2
    w, h = image.get_size()

    try:
        w2, h2 = mask_alpha.shape
    except (ValueError, pygame.error) as e:
        raise ValueError('\nmask shape not understood.')

    if w==0 or h==0:
        raise ValueError(
            'Image with incorrect dimensions must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    if w != w2 or h != h2:
        raise ValueError('\nSurface and mask size mismatch, '
                         'surface(w:%s, h:%s), mask(w:%s, h:%s)' % (w, h, w2, h2))

    cdef:
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, :, :] new_array = numpy.empty((h, w, 4), dtype=numpy.uint8)
        unsigned char [:, :] mask_a = mask_alpha
        int i=0, j=0
        
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                new_array[j, i, 0] = rgb_array[i, j, 0]
                new_array[j, i, 1] = rgb_array[i, j, 1]
                new_array[j, i, 2] = rgb_array[i, j, 2]
                new_array[j, i, 3] = mask_a[i, j]
                
    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_opaque_c(image_, int alpha_):
    """
    Increase opacity of a texture by adding a value to the entire alpha channel.
    <image_>must be a 32-bit pygame surface containing per-pixel information
    ValueError: unsupported color masks for alpha reference array.
    
    <alpha_> must be an integer in range [0...255]
    <alpha_> = 0 output image is identical to the source
    <alpha_> = 255 output image is 100% opaque.
    
    EXAMPLE: 
        image = pygame.image.load('your image')
        output = make_opaque(image, 100) 
    
    :param image_: pygame surface
    :param alpha_: integer in range [0...255]
    :return: Return a 32-bit pygame surface containing per-pixel alpha transparency.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0020762239  
        2 FAIL MODE: | 32-bit  | convert()        |   Surface without per-pixel information.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0018502498
        
        4 FAIL MODE: | 24-bit  |                  |   Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |   Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00152471300
        
        7 FAIL MODE: | 8-bit   |                  |   Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |   Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00149244380
    """
    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)

    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    if not image_.get_bitsize() == 32:
        raise ValueError('Surface without per-pixel information.')

    cdef int w, h
    w, h = image_.get_size()

    if w==0 or h==0:
        raise ValueError('Surface with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    try:
        rgb = pixels3d(image_)
    except (pygame.error, ValueError) as e:
        raise ValueError('\nSurface incompatible, %s.' % e)

    try:

       alpha = pixels_alpha(image_)
    except (pygame.error, ValueError):
        raise ValueError('\nSurface without per-pixel information.')

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, :] alpha_array = alpha
        int i=0, j=0, a

    with nogil:

        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                a = alpha_array[i, j] + alpha_
                if a > 255:
                    a = 255

                new_array[j, i, 0] = rgb_array[i, j, 0]
                new_array[j, i, 1] = rgb_array[i, j, 1]
                new_array[j, i, 2] = rgb_array[i, j, 2]
                new_array[j, i, 3] = a
    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_opaque_b(image_, int alpha_):
    """
    Increase opacity of a texture (only compatible with 32-bit format converted with 
    convert_alpha() prior processing the image).
    <image_>must be a 32-bit pygame surface containing per-pixel information
    Alpha values cap at 255
    
    EXAMPLE: 
        image = pygame.image.load('your image').convert_alpha()
        output = make_opaque_buffer(image, 100) 
    
    :param image_: pygame surface
    :param alpha_: integer in range [0...255]
    :return: Return a 32-bit pygame surface containing per-pixel alpha transparency.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0010 DO NOT USE (BGRA FORMAT)
        2 FAIL MODE: | 32-bit  | convert()        |   0.0007 Image per-pixel is removed and all alpha values are 
                                                      set to default 0. Opacity will increase on all pixels 
                                                      and not only on original transparency mask.
                                                      Incrementing opacity will affect all pixels.
                                                      alpha_ value will set the transpareny/opacity (255 fully opaque, 
                                                      zero fully transparent)
                                                      
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0069 OK
        
        4 FAIL MODE: | 24-bit  |                  |   FAIL Surface without per-pixel information.
        
        5 FAIL MODE: | 24-bit  | convert()        |   0.00068 Same here transparency mask is lost, all pixels alpha
                                                      are 0 by default. Incrementing opacity will affect all pixels.
                                                      alpha_ value will set the transpareny/opacity (255 fully opaque, 
                                                      zero fully transparent)
                                                      
                                                        
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00074 image doesn't have transparency mask.
                                                      A new layer is added to the image but all pixel alpha values are
                                                      set to default 255. Increasing opacity will have no effect 
        
        7 FAIL MODE: | 8-bit   |                  |   FAIL Surface without per-pixel information.
        
        8 FAIL MODE: | 8-bit   | convert()        |   0.00063 alpha = 0, lost transparency mask (same than above)
                                                      alpha_ value will set the transpareny/opacity (255 fully opaque, 
                                                      zero fully transparent)
                                                      
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00062 All values already at max 255. Increasing opacity will
                                                      have no effect
                                                      
    """
    
    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)

    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    if not image_.get_bitsize() == 32:
        raise ValueError('Surface without per-pixel information.')

    cdef int w, h
    w, h = image_.get_size()

    if w==0 or h==0:
        raise ValueError('Surface with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    try:
        rgba_buffer_ = image_.get_view('2')
        
    except (pygame.error, ValueError) as e:
        raise ValueError('\nSurface incompatible, %s.' % e)

    cdef:
        int b_length = rgba_buffer_.length
        unsigned char [:] rgba_buffer = numpy.frombuffer(rgba_buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, numpy.uint8)
        int i=0, a
        unsigned char r, g, b

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            r = rgba_buffer[i]
            g = rgba_buffer[i + 1]
            b = rgba_buffer[i + 2]

            new_buffer[i] = b
            new_buffer[i + 1] = g
            new_buffer[i + 2] = r

            a = rgba_buffer[i + 3] + alpha_
            if a > 255:
                a = 255
            new_buffer[i + 3] = a
            
    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_array_opaque_c(rgb_array_, alpha_array_, int alpha_):
    """
    Increase opacity of <alpha_array_> (alpha channel + alpha_ value).
    The output image is built from RGB and ALPHA values (faster than stacking both arrays).
    <alpha_> = 255, output surface will be 100% opaque.
    <alpha_> = 0, output is identical to the source image
    
    <rgb_array_> must be a numpy.ndarray (w, h, 3) of numpy.uint8 values (RGB values of a surface)
    <alpha_array_> must must numpy.array (w, h) of numpy.uint8 values (ALPHA values of a surface).
    <alpha_> must be an integer in range [0..255] otherwise raise a value error 
    
    EXAMPLE: 
        im1 = pygame.image.load('your image')
        rgb = pygame.surfarray.pixels3d(im1)
        alpha = pygame.surfarray.pixels_alpha()
        output = make_array_opaque(rgb, alpha, 100) 
    
    :param rgb_array_: numpy.ndarray (w, h, 3) (uint8) RGB array values 
    :param alpha_array_: numpy.ndarray (w, h) uint8, ALPHA values
    :param alpha_: integer in range [0...255], Value to add to alpha array
    :return: Returns 32-bit pygame surface with per-pixel information
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00231134380
        2 PASS MODE: | 32-bit  | convert()        |   0.00189980439
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00189980430
        
        4 FAIL MODE: | 24-bit  |                  |   0.00164420030
        5 FAIL MODE: | 24-bit  | convert()        |   0.001671595400
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00146796530
        
        7 FAIL MODE: | 8-bit   |                  |   unsupported bit depth 8 for 3D reference array
        8 FAIL MODE: | 8-bit   | convert()        |   0.00136076789
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00158335620
    """

    assert isinstance(rgb_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument rgb_array_ got %s ' % type(rgb_array_)
    assert isinstance(alpha_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument alpha_ got %s ' % type(alpha_)
    # below lines will slow down the algorithm (converting c variable into python object)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)

    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    cdef int w, h, w2, h2

    try:
        w, h = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('rgb_array_ shape not understood, %s...' % e)

    try:
        w2, h2 = (<object> alpha_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('alpha_array_ shape not understood, %s...' % e)

    if w == 0 or h == 0:
        raise ValueError('rgb_array_ with incorrect '
                         'dimensions must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    if w2 != w or h2 != h:
        raise ValueError('rgb array and alpha size mismatch '
                         'surface(w:%s, h:%s), alpha(w:%s, h:%s)' % (w, h, w2, h2))

    cdef:
        unsigned char [:, :, :] rgb_array_c = rgb_array_                        # non-contiguous values
        unsigned char [:, ::1] alpha_array_c = alpha_array_                     # contiguous values
        unsigned char [: , :, ::1] new_array_c = empty((h, w, 4), dtype=uint8)  # contiguous values
        int i=0, j=0, a=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                a = alpha_array_c[i, j] + alpha_
                if a > 255:
                    a = 255
                new_array_c[j, i, 0] = rgb_array_c[i, j, 0]
                new_array_c[j, i, 1] = rgb_array_c[i, j, 1]
                new_array_c[j, i, 2] = rgb_array_c[i, j, 2]
                new_array_c[j, i, 3] =  a

    return pygame.image.frombuffer(new_array_c, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blink32_b(image_, int alpha_):
    """
    Image is a pygame.Surface 32-bit format with per-pixel information, converted with 
    pygame convert_alpha() method prior processing.
    This function conserve the per-pixel transparency mask (only pixels with alpha value 255 will be modified). 
    Incompatible with 8 - 24 bit if not converted.
    Image 32-bit without conversion (with convert_alpha) will return an BGRA surface.
        
    :param image_: pygame surface 32-bit with per-pixel transparency (preferably converted with 
    the pygame method convert_alpha()). 
    :param alpha_: integer in range [0...255]
    :return: Return a pygame.Surface 32-bit with per-pixel transparency
                                             
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0008328 DO NOT USE,(BGRA FORMAT)
        2 PASS MODE: | 32-bit  | convert()        |   0.0006609 DO NOT USE, Entire layer alpha = 0 
                                                      (no transparency mask)
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006578 OK
        
        4 FAIL MODE: | 24-bit  |                  |   Surface without per-pixel information.
        5 PASS MODE: | 24-bit  | convert()        |   0.0006617 DO NOT USE, Entire layer alpha = 0 
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0006416 Opaque layer (value 255), transparency mask is 
                                                      lost and all pixels will be incremented.
        
        7 FAIL MODE: | 8-bit   |                  |   Surface without per-pixel information.
        8 PASS MODE: | 8-bit   | convert()        |   0.0006875 DO NOT USE, Entire layer alpha = 0 
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0006225 Opaque layer (value 255), transparency mask is 
                                                      lost and all pixels will be incremented.     
    """
    
    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)

    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    if not ((image_.get_bitsize() == 32) and bool(image_.get_flags() & pygame.SRCALPHA)):
        raise ValueError('Surface without per-pixel information.')

    cdef int w, h
    w, h = image_.get_size()

    if w==0 or h==0:
        raise ValueError('Surface with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    try:
        rgba_buffer_ = image_.get_view('2')
        
    except (pygame.error, ValueError) as e:
        raise ValueError('\nSurface incompatible, %s.' % e)

    cdef:
        int b_length = rgba_buffer_.length
        unsigned char [:] rgba_buffer = numpy.frombuffer(rgba_buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, numpy.uint8)
        int i=0
        unsigned char r, g, b

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            new_buffer[i] = rgba_buffer[i + 2]
            new_buffer[i + 1] = rgba_buffer[i + 1]
            new_buffer[i + 2] = rgba_buffer[i]
            if rgba_buffer[i + 3] == 255:
                new_buffer[i + 3] = <unsigned char>((rgba_buffer[i + 3] + alpha_) & 255)

    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blink32_mask_b(image_, int alpha_):
    """
    Image is a pygame.Surface 32-bit format with per-pixel information, converted with 
    pygame convert_alpha() method prior processing.
    This function conserve the per-pixel transparency mask (only pixels with alpha value <255 will be modified). 
    Incompatible with 8 - 24 bit without conversion.
    Image 32-bit without conversion (with convert_alpha) will return an BGRA surface.
        
    :param image_: pygame surface 32-bit with per-pixel transparency (preferably converted with 
    the pygame method convert_alpha()). 
    :param alpha_: integer in range [0...255]
    :return: Return a pygame.Surface 32-bit with per-pixel transparency
                                             
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0008328 DO NOT USE,(BGRA FORMAT)
        2 PASS MODE: | 32-bit  | convert()        |   0.0006609 DO NOT USE, Entire layer alpha = 0 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006578 OK
        
        4 FAIL MODE: | 24-bit  |                  |   FAIL: Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |   FAIL: Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0006416 Opaque layer (value 255). Image will be fully opaque
        
        7 FAIL MODE: | 8-bit   |                  |   FAIL: Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |   FAIL: Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0006225 Opaque layer (value 255). Image will be fully opaque
                                                      
    """

    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)

    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    if not ((image_.get_bitsize() == 32) and bool(image_.get_flags() & pygame.SRCALPHA)):
        raise ValueError('Surface without per-pixel information.')

    cdef int w, h
    w, h = image_.get_size()

    if w==0 or h==0:
        raise ValueError('Surface with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    try:
        rgba_buffer_ = image_.get_view('2')

    except (pygame.error, ValueError) as e:
        raise ValueError('\nSurface incompatible, %s.' % e)

    cdef:
        int b_length = rgba_buffer_.length
        unsigned char [:] rgba_buffer = numpy.frombuffer(rgba_buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, numpy.uint8)
        int i=0
        unsigned char r, g, b

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            new_buffer[i] = rgba_buffer[i + 2]
            new_buffer[i + 1] = rgba_buffer[i + 1]
            new_buffer[i + 2] = rgba_buffer[i]
            if rgba_buffer[i + 3] < 255:
                new_buffer[i + 3] = <unsigned char>((rgba_buffer[i + 3] + alpha_) & 255)
            else:
                new_buffer[i + 3] = rgba_buffer[i + 3]

    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef low_threshold_alpha_c(surface_: Surface, int new_alpha_, int threshold_):
    """
    Filter all pixels (R, G, B) from a given surface (Re-assign alpha values 
    of all pixels (with sum of RGB values) < <threshold_> to 0)
    
    Compatible with 24 - 32 bit surface with per-pixel transparency 
      
    All AVG pixel color strictly inferior to a threshold value will be set to a new
    alpha value (fully transparent or partially transparent, depends on <new_alpha_> value).
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        # Re-assign alpha values of all pixels (with RGB sum) < 50 to 0
        # in other words, make invisible every pixels < 50 (sum of RGB)  
        output = low_th_alpha(im1, 0, 50)   
    
    :param surface_: pygame surface 24-32 bit format
    :param new_alpha_: integer, new alpha value for the pixel in range [0...255] 
    :param threshold_: integer, threshold value in range [0...255]
    :return: return a new 32-bit filtered surface with per-pixel transparency.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0021239385
        2 FAIL MODE: | 32-bit  | convert()        |   Surface without per-pixel information.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00192622640
        
        4 FAIL MODE: | 24-bit  |                  |   Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |   Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0023752522
        
        7 FAIL MODE: | 8-bit   |                  |   Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |   Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00228327360
    """
    assert isinstance(surface_, Surface), \
        'Expecting Surface for argument surface_, got %s ' % type(surface_)
    assert isinstance(new_alpha_, int), \
        'Expecting numpy.array for argument new_alpha, got %s ' % type(new_alpha_)
    assert isinstance(threshold_, int), 'Expecting int for argument threshold_ got %s ' % type(threshold_)

    if not 0 <= new_alpha_ <= 255:
        raise ValueError('\n[-] invalid value for argument new_alpha_, range [0..255] got %s ' % new_alpha_)
    if not 0 <= threshold_ <= 255:
        raise ValueError('\n[-] invalid value for argument threshold_, range [0..255] got %s ' % threshold_)

    if not surface_.get_bitsize() == 32:
        raise ValueError('Surface without per-pixel information.')

    cdef int w, h
    w, h = surface_.get_size()

    if w==0 or h==0:
        raise ValueError('Surface with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    try:
        rgb = pixels3d(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nSurface incompatible.')

    try:
        alpha = pixels_alpha(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nSurface without per-pixel information.')

    cdef:
        unsigned char [:, :, ::1] source_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, :] alpha_array = alpha
        int i=0, j=0, v
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                source_array[j, i, 0] = rgb_array[i, j, 0]
                source_array[j, i, 1] = rgb_array[i, j, 1]
                source_array[j, i, 2] = rgb_array[i, j, 2]
                source_array[j, i, 3] = alpha_array[i, j]
                if alpha_array[i, j] != new_alpha_:
                    # SUM(R + G + B) / 3
                    v = <int>((source_array[j, i, 0] + source_array[j, i, 1] + source_array[j, i, 2]) * ONE_THIRD)
                    if v < threshold_:
                        source_array[j, i, 3] = new_alpha_

    return pygame.image.frombuffer(source_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef high_threshold_alpha_c(surface_: Surface, int new_alpha_, int threshold_):
    """
    Filter all pixels (R, G, B) from a given surface (Re-assign alpha values 
    of all pixels (with sum of RGB values) > <threshold_> to 0)
    
    Compatible with 24 - 32 bit surface with per-pixel transparency 
      
    All AVG pixel color strictly superior to a threshold value will be set to a new
    alpha value (fully transparent or partially transparent, depends on <new_alpha_> value).
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        # Re-assign alpha values of all pixels (with RGB sum) > 50 to 0
        # in other words, make invisible every pixels with sum > 50 
        output = high_th_alpha_c(im1, 0, 50)   
    
    :param surface_: pygame surface 24-32 bit format
    :param new_alpha_: integer, new alpha value for the pixel in range [0...255] 
    :param threshold_: integer, threshold value in range [0...255]
    :return: return a new 32-bit filtered surface with per-pixel transparency.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0013937027
        2 FAIL MODE: | 32-bit  | convert()        |   Surface without per-pixel information.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0014077361
        
        4 FAIL MODE: | 24-bit  |                  |   Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |   Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0014364098
        
        7 FAIL MODE: | 8-bit   |                  |   Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |   Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0014431168
    """
    assert isinstance(surface_, Surface), \
        'Expecting Surface for argument surface_, got %s ' % type(surface_)
    assert isinstance(new_alpha_, int), \
        'Expecting numpy.array for argument new_alpha, got %s ' % type(new_alpha_)
    assert isinstance(threshold_, int), 'Expecting int for argument threshold_ got %s ' % type(threshold_)

    if not 0 <= new_alpha_ <= 255:
        raise ValueError('\n[-] invalid value for argument new_alpha_, range [0..255] got %s ' % new_alpha_)
    if not 0 <= threshold_ <= 255:
        raise ValueError('\n[-] invalid value for argument threshold_, range [0..255] got %s ' % threshold_)

    if not surface_.get_bitsize() == 32:
        raise ValueError('Surface without per-pixel information.')

    cdef int w, h
    w, h = surface_.get_size()

    if w==0 or h==0:
        raise ValueError('Surface with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    try:
        rgb = pixels3d(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nSurface incompatible.')

    try:
        alpha = pixels_alpha(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nSurface without per-pixel information.')

    cdef:
        unsigned char [:, :, ::1] source_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, :] alpha_array = alpha
        int i=0, j=0, v

    with nogil:

        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                source_array[j, i, 0] = rgb_array[i, j, 0]
                source_array[j, i, 1] = rgb_array[i, j, 1]
                source_array[j, i, 2] = rgb_array[i, j, 2]
                source_array[j, i, 3] = alpha_array[i, j]
                if alpha_array[i, j] != new_alpha_:
                    # SUM(R + G + B) / 3
                    v = <int>((source_array[j, i, 0] + source_array[j, i, 1] + source_array[j, i, 2]) * ONE_THIRD)
                    if v > threshold_:
                        source_array[j, i, 3] = new_alpha_

    return pygame.image.frombuffer(source_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_lightness_alpha_c(image):
    """
    Transform an image into greyscale (conserve lightness).
    
    The image must have per-pixel information otherwise
    a ValueError will be raised (8, 24, 32 format image convert with
    convert_alpha() method will works)
    greyscale formula lightness = (max(RGB) + min(RGB))//2
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = greyscale_light_alpha(im1)   
    
    :param image: pygame 8, 24, 32-bit format surface (image must have per-pixel information or,
    needs to be converted with pygame methods convert_alpha() 
    :return: Return Greyscale Surface 32-bit with alpha channel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.001406339
        2 FAIL MODE: | 32-bit  | convert()        |   Surface without per-pixel information.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.001016601
        
        4 FAIL MODE: | 24-bit  |                  |   Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |   Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0009943713, Fully opaque 
        
        7 FAIL MODE: | 8-bit   |                  |   Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |   Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0009552216, Fully opaque
    """
    # TODO try cv2.imread
    assert isinstance(image, Surface), \
        'Argument image is not a valid Surface got %s ' % type(image)

    # Image must be 32-bit format and must have per-pixel transparency
    if not ((image.get_bitsize() == 32) and bool(image.get_flags() & pygame.SRCALPHA)):
        raise ValueError('Surface without per-pixel information.')

    try:
        array_ = pixels3d(image)
        alpha_ = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef:
        unsigned char [:, :, :] pixels = array_  # non-contiguous values
        unsigned char [:, :] alpha = alpha_    # contiguous values
        int w, h

    w, h = image.get_size()
    if w==0 or h==0:
        raise ValueError('Image with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, ::1] grey_c = empty((h, w, 4), dtype=uint8)  # contiguous values
        int i=0, j=0, lightness
        unsigned char r, g, b

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                r = pixels[i, j, 0]
                g = pixels[i, j, 1]
                b = pixels[i, j, 2]
                lightness = (<int>(max_rgb_value(r, g, b) + min_rgb_value(r, g, b))) >> 1
                grey_c[j, i, 0] = lightness
                grey_c[j, i, 1] = lightness
                grey_c[j, i, 2] = lightness
                grey_c[j, i, 3] = alpha[i, j]

    return pygame.image.frombuffer(grey_c, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_lightness_c(image):
    """
    Transform a pygame surface into a greyscale (conserve lightness).
    Compatible with 8, 24-32 bit format image with or without
    alpha channel.
    greyscale formula lightness = (max(RGB) + min(RGB))//2
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = greyscale_light(im1)   
        
    :param image: pygame surface 8, 24-32 bit format 
    :return: a greyscale Surface without alpha channel 24-bit
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0008169703
        2 PASS MODE: | 32-bit  | convert()        |   0.0009996107
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0009860283
        
        4 PASS MODE: | 24-bit  |                  |   0.0010736795
        5 PASS MODE: | 24-bit  | convert()        |   0.0008199265
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0007966573
        
        7 FAIL MODE: | 8-bit   |                  |   Incompatible image.
        8 PASS MODE: | 8-bit   | convert()        |   0.0008080083
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0008132392
    """

    assert isinstance(image, Surface),\
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    # acquires a buffer object for the pixels of the Surface.
    cdef int w_, h_
    w_, h_ = image.get_size()

    if w_==0 or h_==0:
        raise ValueError('Image with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array_                           # non-contiguous values
        unsigned char[:, :, ::1] rgb_out = empty((h, w, 3), dtype=uint8)    # contiguous values
        int red, green, blue, grey, lightness
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                lightness = (max(red, green, blue) + min(red, green, blue)) >> 1
                rgb_out[j, i, 0], rgb_out[j, i, 1], rgb_out[j, i, 2] =  lightness, lightness, lightness

    # Use the pygame method convert_alpha() in order to restore
    # the per-pixel transparency, otherwise use set_colorkey() or set_alpha()
    # methods to restore alpha transparency before blit.
    return pygame.image.frombuffer(rgb_out, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_luminosity_alpha_c(image):
    """
    Transform an image into greyscale (conserve luminosity)
    The image must have per-pixel information otherwise
    a ValueError will be raised (8, 24, 32 format image converted with
    convert_alpha() method will works)
    
    greyscale formula luminosity = R * 0.2126, G * 0.7152, B * 0.0722
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = greyscale_lum_alpha(im1)   
    
    :param image: pygame surface with alpha channel 
    :return: Return Greyscale 32-bit surface with alpha channel 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0012228521
        2 FAIL MODE: | 32-bit  | convert()        |   Surface without per-pixel information.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0009706860
        
        4 FAIL MODE: | 24-bit  |                  |   Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |   Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0009394852
        
        7 FAIL MODE: | 8-bit   |                  |   Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |   Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0009436715
    """
    # TODO CAN GAIN PERFORMANCES WITH cv2.imread
    assert isinstance(image, Surface), \
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    try:
        alpha_ = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('Surface without per-pixel information.')


    cdef:
        unsigned char [:, :, :] pixels = array_ # non-contiguous values
        unsigned char [:, :] alpha = alpha_   # not contiguous
        int w, h

    w, h = image.get_size()
    if w==0 or h==0:
        raise ValueError('Image with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, ::1] grey_c = empty((h, w, 4), dtype=uint8)    # contiguous values
        int i=0, j=0, luminosity
        unsigned char r, g, b

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                r = pixels[i, j, 0]
                g = pixels[i, j, 1]
                b = pixels[i, j, 2]
                luminosity = <unsigned char>(r * 0.2126 + g * 0.7152 + b * 0.072)
                grey_c[j, i, 0], grey_c[j, i, 1], grey_c[j, i, 2], \
                    grey_c[j, i, 3] = luminosity, luminosity, luminosity, alpha[i, j]

    return pygame.image.frombuffer(grey_c, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_luminosity_c(image):
    """
    Transform a Surface into a greyscale (conserve lightness).
    Compatible with 8, 24-32 bit format image with or without
    alpha channel.
    
    greyscale formula luminosity = R * 0.2126, G * 0.7152, B * 0.0722
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = greyscale_lum(im1)   
    
    :param image: Surface 8, 24-32 bit format
    :return: Returns a greyscale 24-bit Surface without alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.001128739
        2 PASS MODE: | 32-bit  | convert()        |   0.001133377
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.001084092
        
        4 PASS MODE: | 24-bit  |                  |   0.001066166
        5 PASS MODE: | 24-bit  | convert()        |   0.001092046
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.001092046
        
        7 FAIL MODE: | 8-bit   |                  |   Incompatible image.
        8 PASS MODE: | 8-bit   | convert()        |   0.0010136184
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0010175665
    """

    assert isinstance(image, Surface),\
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
        
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef int w_, h_
    w_, h_ = image.get_size()

    if w_==0 or h_==0:
        raise ValueError('Image with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array_                           # non-contiguous values
        unsigned char[:, :, ::1] rgb_out = empty((h, w, 3), dtype=uint8)    # contiguous values
        int red, green, blue, grey, luminosity
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                luminosity = <unsigned char>(red * 0.2126 + green * 0.7152 +  blue * 0.0722)
                rgb_out[j, i, 0], rgb_out[j, i, 1], rgb_out[j, i, 2] =  luminosity, luminosity, luminosity

    # Use the pygame method convert_alpha() in order to restore
    # the per-pixel transparency, otherwise use set_colorkey() or set_alpha()
    # methods to restore alpha transparency before blit.
    return pygame.image.frombuffer(rgb_out, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_greyscale_32_c(image):
    """
    Transform an image into greyscale.
    The image must have per-pixel information encoded otherwise
    a ValueError will be raised (8, 24, 32 format image converted with
    pygame convert_alpha() method will works fine).
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = make_greyscale_32(im1)  
        
    :param image: pygame surface with alpha channel 
    :return: Return Greyscale 32-bit Surface with alpha channel 
     
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00237895459
        2 FAIL MODE: | 32-bit  | convert()        |   Surface without per-pixel information.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0016946426999999997
        
        4 FAIL MODE: | 24-bit  |                  |   Surface without per-pixel information.
        5 FAIL MODE: | 24-bit  | convert()        |   Surface without per-pixel information.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0015179608999999995
        
        7 FAIL MODE: | 8-bit   |                  |   Surface without per-pixel information.
        8 FAIL MODE: | 8-bit   | convert()        |   Surface without per-pixel information.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0013761324000000003
    """

    assert isinstance(image, Surface), \
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    try:
        alpha_ = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('Surface without per-pixel information.')

    cdef:
        unsigned char [:, :, :] pixels = array_ # non-contiguous values
        unsigned char [:, :] alpha = alpha_     # contiguous values
        int w, h

    w, h = image.get_size()

    if w==0 or h==0:
        raise ValueError('Image with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, ::1] grey_c = empty((h, w, 4), dtype=uint8)    # contiguous values
        int i=0, j=0
        unsigned char grey_value
        double c1 = ONE_THIRD
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                grey_value = <unsigned char>((pixels[i, j, 0] + pixels[i, j, 1] + pixels[i, j, 2]) * c1)
                grey_c[j, i, 0], grey_c[j, i, 1], grey_c[j, i, 2], \
                    grey_c[j, i, 3] = grey_value, grey_value, grey_value, 255 # alpha[i, j]

    return pygame.image.frombuffer(grey_c, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_greyscale_24_c(image):
    """
    Transform a pygame surface into a greyscale.
    Compatible with 8, 24-32 bit format image with or without
    alpha channel.
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = make_greyscale_24(im1)  
        
    :param image: pygame surface 8, 24-32 bit format
    :return: Returns a greyscale 24-bit surface without alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0008723759
        2 PASS MODE: | 32-bit  | convert()        |   0.0008968581
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0008402743
        
        4 PASS MODE: | 24-bit  |                  |   0.0008633314
        5 PASS MODE: | 24-bit  | convert()        |   0.0008686535
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0008565447
        
        7 FAIL MODE: | 8-bit   |                  |   Incompatible image.
        8 PASS MODE: | 8-bit   | convert()        |   0.00086224349
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0008768265
    
    """

    assert isinstance(image, Surface),\
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef int w_, h_
    # acquires a buffer object for the pixels of the Surface.
    w_, h_ = image.get_size()

    if w_==0 or h_==0:
        raise ValueError('Image with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))
    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array_                        # non-contiguous
        unsigned char[:, :, ::1] rgb_out = empty((h, w, 3), dtype=uint8) # contiguous
        int red, green, blue, grey
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                grey = <int>((red + green + blue) * 0.33)
                rgb_out[j, i, 0], rgb_out[j, i, 1], rgb_out[j, i, 2] =  grey, grey, grey

    # Use the pygame method convert_alpha() in order to restore
    # the per-pixel transparency, otherwise use set_colorkey() or set_alpha()
    # methods to restore alpha transparency before blit.
    return pygame.image.frombuffer(rgb_out, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef make_greyscale_altern_c(image):
    """
    Transform an image into a greyscale image containing per-pixel information.
    Image must be converted with method convert_alpha() or contains
    per-pixels information otherwise a value error will be raised.
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = make_greyscale_altern(im1)  
        
    :param image: pygame surface containing alpha channel 
    :return: Return a greyscale 32-bit surface with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0028212997
        2 FAIL MODE: | 32-bit  | convert()        |   Incompatible image.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0020116723
        
        4 FAIL MODE: | 24-bit  |                  |   Incompatible image.
        5 FAIL MODE: | 24-bit  | convert()        |   Incompatible image.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00213248959
        
        7 FAIL MODE: | 8-bit   |                  |   Incompatible image.
        8 FAIL MODE: | 8-bit   | convert()        |   Incompatible image.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0020254984
    """

    assert isinstance(image, Surface),\
        'Argument image is not a valid Surface, got %s ' % type(image)

    cdef int width, height
    width, height = image.get_size()

    try:
        rgb = pixels3d(image)
        alpha = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef:
        int w = width
        int h = height
        # create memoryview of a 3d numpy array referencing the RGB values, non-contiguous
        unsigned char [:, :, :] rgb_array = rgb
        # create memoryview of a 2d numpy array referencing alpha channel values, contiguous
        unsigned char [:, :] alpha_array = alpha
        unsigned char [:, :, ::1] grayscale_array  = empty((h, w, 4), dtype=uint8)  # contiguous
        int i=0, j=0
        unsigned char gray
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                gray = <int>(rgb_array[i, j, 0] * 0.299 + rgb_array[i, j, 1] * 0.587 + rgb_array[i, j, 2] * 0.114)
                grayscale_array[j, i, 0], grayscale_array[j, i, 1], grayscale_array[j, i, 2] = gray, gray, gray
                grayscale_array[j, i, 3] = alpha_array[i, j]

    return pygame.image.frombuffer(grayscale_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_arr2surf_c(array):
    """
    Transform a 3d numpy array shape (width, height, 3) with RGB values into a
    24-bit greyscale Surface (no per-pixel information).
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        array = pygame.surfarray.pixels3d(im1)
        output = greyscale_arr2surf(array)  
    
    :param array: numpy.ndarray (w, h, 3) uint8 with RGB values 
    :return: Return a greyscale pygame surface (24-bit) without per-pixel information
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0007547003
        2 PASS MODE: | 32-bit  | convert()        |   0.0005987907
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006095613
        
        4 PASS MODE: | 24-bit  |                  |   0.000660638
        5 PASS MODE: | 24-bit  | convert()        |   0.000665111
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.000622179
        
        7 PASS MODE: | 8-bit   |                  |   0.0006077221
        8 PASS MODE: | 8-bit   | convert()        |   0.0005943223
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0006720782 
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)
    cdef int w_, h_
    try:
        w_, h_ = array.shape[:2]
    except (ValueError, pygame.error):
        raise ValueError('Array shape not understood.')

    if w_==0 or h_==0:
        raise ValueError('Array with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array
        unsigned char[:, :, ::1] rgb_out = empty((h, w, 3), dtype=uint8)
        int red, green, blue, grey
        int i=0, j=0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                grey = <int>((red + green + blue) * 0.33)
                rgb_out[j, i, 0], rgb_out[j, i, 1], rgb_out[j, i, 2] =  grey, grey, grey

    # Use the pygame method convert_alpha() in order to restore
    # the per-pixel transparency, otherwise use set_colorkey() or set_alpha()
    # methods to restore alpha transparency before blit.
    return pygame.image.frombuffer(rgb_out, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_array_c(array):
    """
    Transform an RGB color array (w, h, 3) into a greyscale array same size (w, h, 3).
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        array = pygame.surfarray.pixels3d(im1)
        greyscale_array = greyscale_array(array)  
    
    :param array: numpy.ndarray (w, h, 3) uint8 containing RGB values 
    :return: Returns a numpy.ndarray (w, h, 3) uint8 with greyscale values 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0007897191
        2 PASS MODE: | 32-bit  | convert()        |   0.0007631486
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0008085894
        
        4 PASS MODE: | 24-bit  |                  |   0.0007664494
        5 PASS MODE: | 24-bit  | convert()        |   0.0007992083
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0009343332
        
        7 FAIL MODE: | 8-bit   |                  |   Incompatible
        8 PASS MODE: | 8-bit   | convert()        |   0.0010794872
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0010362500
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)

    cdef int w_, h_
    try:
        w_, h_ = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('Array shape not understood...')

    if w_==0 or h_==0:
        raise ValueError('Array with incorrect dimensions, '
                         'must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array
        unsigned char[:, :, ::1] rgb_out = empty((w, h, 3), dtype=uint8)
        int red, green, blue, grey
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                grey = <int>((red + green + blue) * 0.33)
                rgb_out[i, j, 0], rgb_out[i, j, 1], rgb_out[i, j, 2] =  grey, grey, grey

    return asarray(rgb_out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_3d_to_2d_c(array):
    """
    Transform an RGB color array (w, h, 3) into a greyscale (w, h) array 
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        array = pygame.surfarray.pixels3d(im1)
        greyscale_array = greyscale_3d_to_2d(array) # <- return a 2d array  
    
    :param array: numpy.ndarray (w, h, 3) uint8 with RGB values 
    :return: return a numpy.ndarray (w, h) uint8 with greyscale values
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0004365798
        2 PASS MODE: | 32-bit  | convert()        |   0.0003817847
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0003798564
        
        4 PASS MODE: | 24-bit  |                  |   0.0003565184
        5 PASS MODE: | 24-bit  | convert()        |   0.0003645961
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0004135035
        
        7 PASS MODE: | 8-bit   |                  |   0.0003438466
        8 PASS MODE: | 8-bit   | convert()        |   0.0003677791
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0003829953
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)
    cdef int w_, h_
    try:
        w_, h_ = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    if w_==0 or h_==0:
        raise ValueError('array with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array
        unsigned char[:, ::1] rgb_out = empty((w, h), dtype=uint8)
        int red, green, blue
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb_out[i, j] =  <int>((red + green + blue) * 0.3333)

    return asarray(rgb_out)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_2d_to_3d_c(array):
    """
    Transform a greyscale array (w, h) into a greyscale array (w, h, 3)
    
    EXAMPLE:
        greyscale_array = greyscale_2d_to_3d(2d_array)  
    
    :param array: numpy.ndarray (w, h) uint8 with greyscale values
    :return: Returns a numpy.ndarray (w, h, 3) uint 8 with greyscale values.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0004189829
        2 PASS MODE: | 32-bit  | convert()        |   0.0004441290
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0004055626
        
        4 PASS MODE: | 24-bit  |                  |   0.0004644079
        5 PASS MODE: | 24-bit  | convert()        |   0.0004916378
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0004254949
        
        7 PASS MODE: | 8-bit   |                  |   0.0004191546
        8 PASS MODE: | 8-bit   | convert()        |   0.0003937079
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0004289279
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)

    cdef int w_, h_
    try:
        w_, h_ = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    if w_==0 or h_==0:
        raise ValueError('array with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :] rgb_array = array
        unsigned char[:, :, ::1] rgb_out = empty((w, h, 3), dtype=uint8)
        int grey
        int i=0, j=0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                grey = rgb_array[i, j]
                rgb_out[i, j, 0], rgb_out[i, j, 1], rgb_out[i, j, 2] =  grey, grey, grey

    return asarray(rgb_out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef buffer_greyscale_c(image):
    """
    Create a greyscale image and returns a tuple (greyscale, array)
    Method using a buffer to retrieve pixels information.
    :param image: Color image (pygame.Surface)
    :returns: return a 32-bit greyscale pygame.Surface with per-pixel transparency
              and equivalent 2darray (width, height) uint8
    """

    cdef:
        int w, h
        
    w, h = image.get_size()
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)
    try:
        buffer_ = image.get_view('2') 
    except (pygame.error, ValueError):
        raise ValueError('Incompatible pygame surface')
    cdef:

        unsigned char [::1] c_buffer  = numpy.frombuffer(buffer_, numpy.uint8)
        int i = 0, l = buffer_.length
        unsigned char avg

    with nogil:
        for i in prange(0, l, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            avg = <int>((c_buffer[i] + c_buffer[i + 1] + c_buffer[i+2]) * 0.33)
            c_buffer[i] = avg
            c_buffer[i + 1] = avg
            c_buffer[i + 2] = avg

    return pygame.image.frombuffer(c_buffer, (w, h), 'RGBA'), numpy.asarray(c_buffer)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bw_surface24_c(image):
    """
    Transform a pygame surface (24, 32-bit RGB format) into a 32-bit format black and white (BW) surface. 
    The final surface will have full opacity. 
    
    :param image: pygame.Surface (24, 32 bit format with RGB values).
    Source alpha channel will be substitute to full opacity (255 values).
    :return: Returns a 24 bit BW pygame surface with alpha channel 
    (full opacity) and its equivalent BW buffer.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0005112683
        2 PASS MODE: | 32-bit  | convert()        |   0.0004727313 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006143273
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.0004119966 
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0003773526 
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size 
        8 PASS MODE: | 8-bit   | convert()        |   0.0004628787  
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0004385520 
    """
    cdef:
        int w, h

    w, h = image.get_size()
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    try:
        # '2' returns a (surface-width, surface-height) array of raw pixels.
        # The pixels are surface-bytesize-d unsigned integers.
        # The pixel format is surface specific.
        # The 3 byte unsigned integers of 24 bit surfaces are
        # unlikely accepted by anything other than other pygame functions.
        buffer_ = image.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Incompatible pygame surface')

    cdef:
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0, length_ =  buffer_.length

    with nogil:
        for i in prange(0, length_, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            if  (c_buffer[i] + c_buffer[i + 1] + c_buffer[i + 2]) > 381:
                # force full opacity alpha = 255
                c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], c_buffer[i + 3]= 255, 255, 255, 255
            else:
                # force full opacity alpha = 255
                c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], c_buffer[i + 3]= 0, 0, 0, 255
    # return a 32-bit format surface with full opacity
    return pygame.image.frombuffer(c_buffer, (w, h), 'RGBA'), numpy.asarray(c_buffer)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bw_surface32_c(image):
    """
    Transform a pygame surface (24, 32-bit RGB format) into a 32-bit format black and white (BW) pygame surface
    If the source image has per-pixel transparency (surface 32-bit) the final surface will be identical (transparent)
    If the surface is  24-bit surface the final surface will have full opacity (see TEST section).
    
    :param image: pygame.Surface (24, 32 bit format with RGB values).
    Source alpha channel will be unchanged.
    :return: Returns a 32 bit BW pygame surface and its equivalent BW buffer.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0002112683  TRANSPARENT
        2 PASS MODE: | 32-bit  | convert()        |   0.0002727313  ** no alpha channel, DO NOT USE
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0002143273  TRANSPARENT 
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size 
        5 PASS MODE: | 24-bit  | convert()        |   0.0002119966  ** no alpha channel, DO NOT USE
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0002773526  ** FULL OPACITY 
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 PASS MODE: | 8-bit   | convert()        |   0.0002628787  ** no alpha channel, DO NOT USE 
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0002385520  ** FULL OPACITY 
    """
    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for arguement image, got %s " % type(image)

    # uncomment the line below to test pixel fornat 8-24 bit
    if not image.get_bitsize() == 32:
        raise ValueError('Surface without per-pixel information.')
    
    cdef:
        int w, h
    w, h = image.get_size()

    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    try:
        # '2' returns a (surface-width, surface-height) array of raw pixels.
        # The pixels are surface-bytesize-d unsigned integers.
        # The pixel format is surface specific.
        # The 3 byte unsigned integers of 24 bit surfaces are
        # unlikely accepted by anything other than other pygame functions.
        buffer_ = image.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Incompatible pygame surface')

    cdef:
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0, length_ =  buffer_.length

    with nogil:
        for i in prange(0, length_, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            if  (c_buffer[i] + c_buffer[i + 1] + c_buffer[i + 2]) > 381:
                c_buffer[i], c_buffer[i + 1], c_buffer[i + 2]= 255, 255, 255
            else:
                c_buffer[i], c_buffer[i + 1], c_buffer[i + 2]= 0, 0, 0
    # return a 32-bit format surface with per-pixel transparency
    return pygame.image.frombuffer(c_buffer, (w, h), 'RGBA'), numpy.asarray(c_buffer)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef  bw_array_c(array):
    """
    Transform/convert an array shapes (w, h, 3) containing RGB values or greyscale
    values into a black and white surface (BW).
    Returns a 3d or 2d black and white array (depends on argument array)

    :param array: numpy.array shape (w, h, 3) containing RGB values uint8
    or greyscale values uint8
    :return: a BW 2d array shape (w, h).
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00072454
        2 PASS MODE: | 32-bit  | convert()        |   0.00055107 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00048434
        
        4 PASS MODE: | 24-bit  |                  |   0.00050048
        5 PASS MODE: | 24-bit  | convert()        |   0.00045421
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00055422  
        
        7 PASS MODE: | 8-bit   |                  |   0.00050695
        8 PASS MODE: | 8-bit   | convert()        |   0.00042287  
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00033910   
    """
    
    assert isinstance(array, numpy.ndarray), \
           "Argument array should be a numpy.ndarray, got %s " % type(array)
    
    cdef:
        int w, h

    try:
        # assume (w, h, dim) type array
        w, h = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood. Only array shape (w, h, 3) are compatible.')
        
    assert w>0 and h>0,\
        'Incorrect array dimensions, width & hight should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    cdef:
        unsigned char [:, :, :] array3d = array
        unsigned char [:, :] array2d = numpy.empty((w, h), dtype=numpy.uint8)
        int i = 0, j = 0

    # convert array type (w, h, n) into a 2d array
    # shape (w, h)
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                if (array3d[i, j, 0] + array3d[i, j, 1] + array3d[i, j, 2]) > 381:
                    array2d[i, j] = 255
                else:
                    array2d[i, j] = 0

    return numpy.asarray(array2d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef redscale_b(surface_:Surface):
    """
    Create a red scale image with (perceptual luminance-preserving),
    compatible with 8, 24-32 bit format image
    see TEST section for more details.
    
    :param surface_: pygame surface 8, 24-32 bit format
    :return: Return a redscale pygame surface 32-bit format with per-pixel transparency.
             Alpha channel values are set to 255. If you do not want to keep the per-pixel
             transparency, use the pygame method convert() to remove it.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.000581
        2 PASS MODE: | 32-bit  | convert()        |   0.000546
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.000501
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.000493
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0005059
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 PASS MODE: | 8-bit   | convert()        |   0.0005067 
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0005067
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()
    

    try:
        buffer_ = surface_.get_view('2')
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0
        int luminosity

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            luminosity = <unsigned char>(c_buffer[i] * 0.2126
                                         + c_buffer[i + 1] * 0.7152 + c_buffer[i + 2] * 0.0722)
            c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], c_buffer[i + 3]= luminosity, 0, 0, 255

    # todo: try make_surface instead to see if returned surface is RGB instead of RGBA
    return pygame.image.frombuffer(c_buffer, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef redscale_alpha_b(surface_: Surface):
    """
    Create a redscale image with (perceptual luminance-preserving),
    compatible with 32-bit format image with per-pixel only
        
    :param surface_: pygame surface (32-bit with per-pixel information)
    :return: Return a redscale pygame 32-bit surface with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.000438
        2 PASS MODE: | 32-bit  | convert()        |   0.000438 alpha = 0 do not use 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.000452
        
        4 FAIL MODE: | 24-bit  |                  |   Surface pixel format is not 32 bit, got 24 
        5 PASS MODE: | 24-bit  | convert()        |   0.000439 alpha = 0 do not use 
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.000440 alpha = 255 do not use 
        
        7 FAIL MODE: | 8-bit   |                  |   Surface pixel format is not 32 bit, got 24 
        8 PASS MODE: | 8-bit   | convert()        |   0.000439 alpha = 0 do not use 
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.000443 alpha = 255 do not use 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    # Remove the asser statement to process other pixel format such as 8-24 bit

    if not surface_.get_bitsize() == 32:
        raise ValueError('Surface pixel format is not 32 bit, got %s ' % surface_.get_bitsize())

    cdef int width, height
    width, height = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0
        int luminosity

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            luminosity = <unsigned char>(c_buffer[i] * 0.2126
                                         + c_buffer[i + 1] * 0.7152 + c_buffer[i + 2] * 0.0722)
            c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], = luminosity, 0, 0

    return pygame.image.frombuffer(c_buffer, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greenscale_b(surface_:Surface):
    """
    Create a green scale image with (perceptual luminance-preserving),
    compatible with 8, 24-32 bit format image.
    see TEST section for more details.
    
    :param surface_: pygame surface 8, 24-32 bit format
    :return: Return a greenscale pygame surface 32-bit format pixel with per-pixel transparency.
    
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.000504
        2 PASS MODE: | 32-bit  | convert()        |   0.000493 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.000492
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.000494
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.000495
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 PASS MODE: | 8-bit   | convert()        |   0.000496  
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.000491
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0
        int luminosity

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            luminosity = <unsigned char>(c_buffer[i] * 0.2126
                                         + c_buffer[i + 1] * 0.7152 + c_buffer[i + 2] * 0.0722)
            c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], c_buffer[i + 3] = 0, luminosity, 0, 255

    return pygame.image.frombuffer(c_buffer, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greenscale_alpha_b(surface_: Surface):
    """
    Create a greenscale image with (perceptual luminance-preserving),
    compatible with 32-bit format image with per-pixel only.
    see TEST section for more details.
    
    :param surface_: pygame surface (32-bit with per-pixel information)
    :return: Return a greenscale pygame 32-bit surface with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.000429
        2 PASS MODE: | 32-bit  | convert()        |   0.000428 alpha = 0 do not use 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.000429
        
        4 FAIL MODE: | 24-bit  |                  |   Surface pixel format is not 32 bit, got 24 
        5 PASS MODE: | 24-bit  | convert()        |   0.000428 alpha = 0 do not use 
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.000430 alpha = 255 do not use 
        
        7 FAIL MODE: | 8-bit   |                  |   Surface pixel format is not 32 bit, got 24 
        8 PASS MODE: | 8-bit   | convert()        |   0.000437 alpha = 0 do not use 
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.000443 alpha = 255 do not use 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    # Remove the asser statement to process other pixel format such as 8-24 bit
    assert surface_.get_bitsize() is 32, \
        'Surface pixel format is not 32 bit, got %s ' % surface_.get_bitsize()

    cdef int width, height
    width, height = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0
        int luminosity

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            luminosity = <unsigned char>(c_buffer[i] * 0.2126
                                         + c_buffer[i + 1] * 0.7152 + c_buffer[i + 2] * 0.0722)
            c_buffer[i], c_buffer[i + 1], c_buffer[i + 2] = 0, luminosity, 0

    return pygame.image.frombuffer(c_buffer, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bluescale_b(surface_:Surface):
    """
    Create a blue scale image with (perceptual luminance-preserving),
    compatible with 8, 24-32 bit format image
    
    :param surface_: pygame surface 8, 24-32 bit format
    :return: Return a bluescale pygame surface 32-bit with alpha channel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0007
        2 PASS MODE: | 32-bit  | convert()        |   0.0006
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0005
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.0005
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0005
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 PASS MODE: | 8-bit   | convert()        |   0.0005  
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0005
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0
        int luminosity

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            luminosity = <unsigned char>(c_buffer[i] * 0.2126
                                         + c_buffer[i + 1] * 0.7152 + c_buffer[i + 2] * 0.0722)
            c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], c_buffer[i + 3] = 0, 0, luminosity, 255

    return pygame.image.frombuffer(c_buffer, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bluescale_alpha_b(surface_:Surface):
    """
    Create a bluescale with (perceptual luminance-preserving),
    compatible with 32-bit format image with per-pixel only.
    
    :param surface_: pygame surface (32-bit with per-pixel information)
    :return: Return a bluescale pygame 32-bit surface with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0004
        2 PASS MODE: | 32-bit  | convert()        |   0.0004 alpha = 0 do not use 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0004
        
        4 FAIL MODE: | 24-bit  |                  |   Surface pixel format is not 32 bit, got 24 
        5 PASS MODE: | 24-bit  | convert()        |   0.0004 alpha = 0 do not use 
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0004 alpha = 255 do not use 
        
        7 FAIL MODE: | 8-bit   |                  |   Surface pixel format is not 32 bit, got 24 
        8 PASS MODE: | 8-bit   | convert()        |   0.0004 alpha = 255 do not use 
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0004 alpha = 255 do not use 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    # Remove the asser statement to process other pixel format such as 8-24 bit
    assert surface_.get_bitsize() is 32, \
        'Surface pixel format is not 32 bit, got %s ' % surface_.get_bitsize()

    cdef int width, height
    width, height = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0
        int luminosity

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            luminosity = <unsigned char>(c_buffer[i] * 0.2126
                                         + c_buffer[i + 1] * 0.7152 + c_buffer[i + 2] * 0.0722)
            c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], = 0, 0, luminosity

    return pygame.image.frombuffer(c_buffer, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef redscale_c(surface_:Surface):
    """
    Create a redscale image from the given Surface (compatible with 8, 24-32 bit format image)
    Alpha channel will be ignored from image converted with the pygame method convert_alpha.
    
    :param surface_: Surface, loaded with pygame.image method
    :return: Return a redscale Surface without alpha channel.
        
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00068803760
        2 PASS MODE: | 32-bit  | convert()        |   0.00063437914
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006763685
        
        4 PASS MODE: | 24-bit  |                  |   0.00060436344
        5 PASS MODE: | 24-bit  | convert()        |   0.00060436344
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00060099638
        
        7 PASS MODE: | 8-bit   |                  |   0.0068593642  * Slower using array3d instead of referencing pixels 
        8 PASS MODE: | 8-bit   | convert()        |   0.0006055111
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0006126049 
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except(pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        int r, g, b, luminosity
    with nogil:
        for i in prange(width):
            for j in range(height):
                r = rgb_array[i, j, 0]
                g = rgb_array[i, j, 1]
                b = rgb_array[i, j, 2]
                luminosity = <unsigned char>(r * 0.2126 + g * 0.7152 + b * 0.0722)
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2] = luminosity, 0, 0

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef redscale_alpha_c(surface_:Surface):
    """
    Create a redscale image from the given Surface (compatible with 32-bit format image with per-pixel
    transparency and with 8-24bit format image if converted with pygame method convert_alpha().
    
    :param surface_: pygame surface (32-bit with per-pixel information)
    :return: Return a redscale pygame 32-bit surface with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0012311873
        2 FAIL MODE: | 32-bit  | convert()        |   Incompatible pixel format.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00099194689
        
        4 FAIL MODE: | 24-bit  |                  |   Surface pixel format is not 32 bit, got 24 
        5 FAIL MODE: | 24-bit  | convert()        |   Incompatible pixel format.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00162914297 FULL OPACITY
        
        7 FAIL MODE: | 8-bit   |                  |   Surface pixel format is not 32 bit, got 8 
        8 FAIL MODE: | 8-bit   | convert()        |   Incompatible pixel format.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00267167812 FULL OPACITY
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    assert surface_.get_bitsize() is 32, \
        'Surface pixel format is not 32 bit, got %s ' % surface_.get_bitsize()

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :] alpha_array = alpha_
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        int i=0, j=0
        int luminosity

    with nogil:
        for i in prange(width):
            for j in range(height):
                luminosity =\
                <unsigned char>(rgb_array[i, j, 0] * 0.2126 + rgb_array[i, j, 1]
                                * 0.7152 + rgb_array[i, j, 2] * 0.0722)
                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3] = luminosity, 0, 0, alpha_array[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greenscale_c(surface_:Surface):

    """
    
    Create a greenscale image from the given Surface (compatible with 8, 24-32 bit format image)
    Alpha channel will be ignored from image converted with the pygame method convert_alpha.
    
    :param surface_: pygame surface 8, 24-32 bit format
    :return: Return a greenscale Surface without alpha channel.
    
     TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0014173207000000001
        2 PASS MODE: | 32-bit  | convert()        |   0.0010779202000000003
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0011935574999999998
        
        4 PASS MODE: | 24-bit  |                  |   0.0011942349000000005
        5 PASS MODE: | 24-bit  | convert()        |   0.0012506221999999995
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0014203511999999999
        
        7 PASS MODE: | 8-bit   |                  |   0.008594837800000001
        8 PASS MODE: | 8-bit   | convert()        |   0.0016511288000000022
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0015795762999999993
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except:
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = zeros((height, width, 3), dtype=uint8)
        int i=0, j=0
        int luminosity

    with nogil:
        for i in prange(width):
            for j in range(height):
                luminosity =\
                <unsigned char>(rgb_array[i, j, 0] * 0.2126 +
                                rgb_array[i, j, 1] * 0.7152 + rgb_array[i, j, 2] * 0.0722)
                new_array[j, i, 1] = luminosity

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greenscale_alpha_c(surface_:Surface):
    """
    Create a greenscale image from the given Surface (compatible with 32-bit format image with per-pixel
    transparency and with 8-24bit format image if converted with pygame convert_alpha method.
    Output image will have alpha channel.
    
    :param surface_: pygame surface 32 bit format
    :return: Return a greenscale pygame surface with alpha channel
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.004664306
        2 FAIL MODE: | 32-bit  | convert()        |   Incompatible pixel format.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.004293414
        
        4 FAIL MODE: | 24-bit  |                  |   Surface pixel format is not 32 bit, got 8 
        5 FAIL MODE: | 24-bit  | convert()        |   Incompatible pixel format.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.003173919 FULL OPACITY
        
        7 FAIL MODE: | 8-bit   |                  |   Surface pixel format is not 32 bit, got 8 
        8 FAIL MODE: | 8-bit   | convert()        |   Incompatible pixel format.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.003384680 FULL OPACITY
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    assert surface_.get_bitsize() is 32, \
        'Surface pixel format is not 32 bit, got %s ' % surface_.get_bitsize()

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :] alpha_array = alpha_
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        int i=0, j=0
        int luminosity
    with nogil:
        for i in prange(width):
            for j in range(height):
                luminosity =\
                <unsigned char>(rgb_array[i, j, 0] * 0.2126
                                + rgb_array[i, j, 1] * 0.7152 + rgb_array[i, j, 2] * 0.0722)
                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3]= 0, luminosity, 0, alpha_array[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bluescale_c(surface_:Surface):
    """
    Create a bluescale image from the given Surface (compatible with 8, 24-32 bit format image)
    Alpha channel will be ignored from image converted with the pygame method convert_alpha.
    
    :param surface_: Surface, loaded with pygame.image method
    :return: Return a bluescale Surface without alpha channel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00151
        2 PASS MODE: | 32-bit  | convert()        |   0.00176
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00278
        
        4 PASS MODE: | 24-bit  |                  |   0.00186
        5 PASS MODE: | 24-bit  | convert()        |   0.00229
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00186
        
        7 PASS MODE: | 8-bit   |                  |   0.00081
        8 PASS MODE: | 8-bit   | convert()        |   0.00105
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00142
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
         try:
            rgb_ = array3d(surface_)
         except:
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        int r, g, b, luminosity
    with nogil:
        for i in prange(width):
            for j in range(height):
                r = rgb_array[i, j, 0]
                g = rgb_array[i, j, 1]
                b = rgb_array[i, j, 2]
                luminosity = <unsigned char>(r * 0.2126 + g * 0.7152 + b * 0.0722)
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2] = 0, 0, luminosity

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bluescale_alpha_c(surface_:Surface):
    """
    Create a bluescale image from the given Surface (compatible with 32-bit format image with per-pixel
    transparency and with 8-24bit format image if converted with pygame convert_alpha method.
    Output image will have alpha channel.
    
    :param surface_: Surface, loaded with pygame.image method
    :return: Return a bluescale Surface with alpha channel
    
     TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.001111767
        2 FAIL MODE: | 32-bit  | convert()        |   Incompatible pixel format.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.001141270
        
        4 FAIL MODE: | 24-bit  |                  |   Surface pixel format is not 32 bit, got 8 
        5 FAIL MODE: | 24-bit  | convert()        |   Incompatible pixel format.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.000989298 FULL OPACITY
        
        7 FAIL MODE: | 8-bit   |                  |   Surface pixel format is not 32 bit, got 8 
        8 FAIL MODE: | 8-bit   | convert()        |   Incompatible pixel format.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.000900542 FULL OPACITY
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert surface_.get_bitsize() is 32, \
        'Surface pixel format is not 32 bit, got %s ' % surface_.get_bitsize()

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :] alpha_array = alpha_
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        int i=0, j=0
        int luminosity

    with nogil:
        for i in prange(width):
            for j in range(height):
                luminosity =\
                <unsigned char>(rgb_array[i, j, 0] * 0.2126
                                + rgb_array[i, j, 1] * 0.7152 + rgb_array[i, j, 2] * 0.0722)
                new_array[j, i, 0], new_array[j, i, 1],\
                new_array[j, i, 2], new_array[j, i, 3] = 0, 0, luminosity, alpha_array[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef load_per_pixel_c(file):
    """
    Load an image with per-pixel transparency 

    :param file: string, path to load the image.
    :return: Returns a Surface with per-pixel information
    """

    assert isinstance(file, str), \
        'Expecting python str value for positional argument file, got %s: ' % type(file)

    cdef int width, height

    try:
        surface_ = pygame.image.load(file)
        width, height = surface_.get_size()

    except (pygame.error, ValueError):
        raise FileNotFoundError('\nFile %s not found ' % file)

    if width == 0 or height == 0:
        raise ValueError('Image with incorrect dimensions,'
                         ' must be (w>0, h>0) got (w:%s, h:%s) ' % (width, height))

    return make_surface_c1(surface_.get_view('2'), width, height)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# todo this method will not raise an error if the image does not have per-pixel infos
cdef load_image32_c(path):
    """
    Load an image with per-pixel transparency and returns its equivalent Surface and RGBA array
    
    :param path: String, Path to load the image
    The image must be encoded with per-pixel transparency otherwise the function will raise an error.
    :return: Returns a Surface with per-pixel information and a numpy.ndarray (w, h, 4) uint8 (RGBA values)
    """
    assert isinstance(path, str), \
        'Expecting str value for positional argument path, got %s: ' % type(path)

    cdef int width, height
    
    try:
        # load an image with alpha channel numpy.array (w, h, 4) uint 8
        # imread() decodes the image into a matrix with the color channels
        # stored in the following order :Blue, Green and Red respectively.
        bgra_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        width, height = bgra_array.shape[:2]
    except (pygame.error, ValueError):
        raise FileNotFoundError('\npImage %s is not found ' % path)

    if width == 0 or height == 0:
        raise ValueError('Image with incorrect dimensions, '
                         ' must be (w>0, h>0) got (w:%s, h:%s) ' % (width, height))
    # convert the BGRA array to RGBA
    rgba_array = array_bgra2rgba_c(bgra_array)
    return pygame.image.frombuffer(rgba_array, (height, width), 'RGBA'), rgba_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef spritesheet_per_pixel_c(file_, int chunk_, int columns_, int rows_):
    """
    Retrieve all sprites from a sprite sheets.
    All individual sprite will contains per-pixel transparency information,
    if the image has been encoded with alpha channel.
    Works only with 32bit surface with per-pixel transparency 
    
    :param file_: Str, full path to the texture/image
    :param chunk_: int, Size of a single sprite (size in pixels)
    :param columns_: int, number of columns
    :param rows_: int, number of rows
    :return: return a python list containing all the sprites
    """

    assert isinstance(file_, str),\
        'Argument file_ is not a python string type, got %s ' % type(file_)
    assert isinstance(chunk_, int),\
        'Argument chunk_ is not a python integer type, got %s ' % type(chunk_)
    assert isinstance(columns_, int),\
        'Argument columns_ is not a python integer type, got %s ' % type(columns_)
    assert isinstance(rows_, int),\
        'Argument rows_ is not a python integer type, got %s ' % type(rows_)

    try:
        surface = pygame.image.load(file_)

    except (pygame.error, ValueError):
        raise FileNotFoundError('\nFile %s not found ' % file_)

    cdef int w, h
    # todo try except for get_view
    buffer_ = surface.get_view('2')
    w, h = surface.get_size()
    if w==0 or h==0:
        raise ValueError('image with incorrect dimensions must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        np.ndarray[np.uint8_t, ndim=3] source_array = \
            numpy.frombuffer(buffer_, dtype=uint8).reshape((h, w, 4))
        np.ndarray[np.uint8_t, ndim=3] array1 = empty((chunk_, chunk_,4), uint8)
        int rows = 0
        int columns = 0

    animation = []
    for rows in range(rows_):

        for columns in range(columns_):

            array1 = source_array[rows * chunk_:(rows + 1) * chunk_,
                                  columns * chunk_:(columns + 1) * chunk_, :]
            surface_ = pygame.image.frombuffer(ascontiguousarray(array1), (chunk_, chunk_), 'RGBA')
            animation.append(surface_.convert(32, pygame.SWSURFACE | RLEACCEL | SRCALPHA))
    return animation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def spritesheet_per_pixel_fs8(file, int chunk, int columns_, int rows_, tweak_=False, *args):
    """
    Retrieve all sprites from a sprite sheets.
    All individual sprite will contains per-pixel transparency information only if the
    image has been encoded with per-pixel information.

    :param file: str,  full path to the texture
    :param chunk: int, size of a single image in bytes e.g 64x64 (equal
    :param rows_: int, number of rows
    :param columns_: int, number of column
    :param tweak_: bool, modify the chunk sizes (in bytes) in order to process
                   data with non equal width and height e.g 320x200
    :param args: tuple, used with theak_, args is a tuple containing the new chunk size,
                 e.g (320, 200)
    :return: list, Return textures (pygame surface) containing** per-pixel transparency into a
            python list. ** if original image had per-pixel information.
    """

    #TODO CHECK FOR 8 BIT TEXTURE, 8 BIT TEXTURE WILL HAVE NO PER-PIXEL TRANSPARENCY

    assert isinstance(file, str), \
        'Expecting string for argument file got %s: ' % type(file)
    assert isinstance(chunk, int), \
        'Expecting int for argument number got %s: ' % type(chunk)
    assert isinstance(rows_, int) and isinstance(columns_, int), \
        'Expecting int for argument rows_ and columns_ ' \
        'got %s, %s ' % (type(rows_), type(columns_))
    assert isinstance(tweak_, bool), \
        "Expecting boolean for argument tweak_ got %s " % type(tweak_)

    cdef int width, height

    try:
        image_ = pygame.image.load(file)
        width, height = image_.get_size()

    except (pygame.error, ValueError):
        raise FileNotFoundError('\nFile %s not found ' % file)

    if width==0 or height==0:
        raise ValueError('Surface dimensions is not correct, '
                         'must be: (w>0, h>0) got (w:%s, h:%s) ' % (width, height))
    try:
        # Reference pixels into a 3d array
        # pixels3d(Surface) -> array
        # Create a new 3D array that directly references the pixel values
        # in a Surface. Any changes to the array will affect the pixels in
        # the Surface. This is a fast operation since no data is copied.
        # This will only work on Surfaces that have 24-bit or 32-bit formats.
        # Lower pixel formats cannot be referenced.
        rgb_array_ = pixels3d(image_)
        alpha_array_ = pixels_alpha(image_)

    except (pygame.error, ValueError):
        try:
            # Copy pixels into a 3d array
            # array3d(Surface) -> array
            # Copy the pixels from a Surface into a 3D array.
            # The bit depth of the surface will control the size of the integer values,
            # and will work for any type of pixel format.
            # This function will temporarily lock the Surface as
            # pixels are copied (see the Surface.lock()
            # lock the Surface memory for pixel access
            # - lock the Surface memory for pixel access method).
            rgb_array_ = array3d(image_)
            # Copy the pixel alpha values (degree of transparency) from
            # a Surface into a 2D array. This will work for any type
            # of Surface format. Surfaces without a pixel alpha will
            # return an array with all opaque values.
            alpha_array_ = array_alpha(image_)

        except (pygame.error, ValueError):
            raise RuntimeError('\nCould not create a 3d numpy array from the file %s.\n'
                               'Check if the file exist on your file system.\n'
                               'Make sure the surface is not locked by another concurrent process.\n'
                               'The image must be 24-32 bits with alpha transparency.' % file)
    cdef:
        # create a numpy array 3d containing RGBA values.
        np.ndarray[np.uint8_t, ndim=3] rgba_array = make_array_c_transpose(rgb_array_, alpha_array_)
        np.ndarray[np.uint8_t, ndim=3] array1 = zeros((chunk, chunk, 4), dtype=uint8)
        int chunkx = 0
        int chunky = 0
        int rows = 0
        int columns = 0

    animation = []

    # modify the chunk size
    if tweak_ and args is not None:

        if isinstance(args, tuple):
            try:
                chunkx = args[0][0]
                chunky = args[0][1]
            except IndexError:
                raise IndexError('Parse argument not understood.')
            if chunkx==0 or chunky==0:
                raise ValueError('Chunkx and chunky cannot be equal to zero.')
            if (width % chunkx) != 0:
                raise ValueError('Chunkx size value is not a correct fraction of %s ' % width)
            if (height % chunky) != 0:
                raise ValueError('Chunky size value is not a correct fraction of %s ' % height)
        else:
            raise ValueError('Parse argument not understood.')
    else:
        chunkx, chunky = chunk, chunk

    # get all the sprites
    for rows in range(rows_):
        for columns in range(columns_):
            array1 = rgba_array[rows * chunky:(rows + 1) * chunky, columns * chunkx:(columns + 1) * chunkx, :]
            animation.append(pygame.image.frombuffer(
                ascontiguousarray(array1), (chunkx, chunky), 'RGBA'))
    return animation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def spritesheet_alpha(file, int chunk, int columns_, int rows_, tweak_=False, *args):
    """
    Extract all sprites from a sprite sheet.
    This method is using OpencCV to open the spritesheet (much faster than Pygame)

    All sprites will contain per-pixel transparency information,
    Surface without per-pixel information will raise a ValueError
    This will only work on Surfaces that have 24-bit or 32-bit formats.

    :param file: str,  full path to the texture
    :param chunk: int, size of a single image in bytes e.g 64x64 (equal
    :param rows_: int, number of rows
    :param columns_: int, number of column
    :param tweak_: bool, modify the chunk sizes (in bytes) in order to process
                   data with non equal width and height e.g 320x200
    :param args: tuple, used with theak_, args is a tuple containing the new chunk size,
                 e.g (320, 200)
    :return: list, Return textures (pygame surface) containing per-pixel transparency into a
            python list
    """

    assert isinstance(file, str), \
        'Expecting string for argument file got %s: ' % type(file)
    assert isinstance(chunk, int), \
        'Expecting int for argument number got %s: ' % type(chunk)
    assert isinstance(rows_, int) and isinstance(columns_, int), \
        'Expecting int for argument rows_ and columns_ ' \
        'got %s, %s ' % (type(rows_), type(columns_))
    assert isinstance(tweak_, bool), \
        "Expecting boolean for argument tweak_ got %s " % type(tweak_)

    try:
        # load an image with alpha channel numpy.array (w, h, 4) uint 8
        # imread() decodes the image into a matrix with the color channels
        # stored in the order of Blue, Green and Red respectively.

        bgra_array = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if bgra_array is None:
            raise ValueError
    except Exception:
        raise FileNotFoundError('\npImage %s is not found ' % file)

    cdef int width, height, dim
    try:
        width, height, dim = bgra_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    if width == 0 or height == 0 or dim!=4:
        raise ValueError('image with incorrect dimensions must'
                         ' be (w>0, h>0, 4) got (w:%s, h:%s, %s) ' % (width, height, dim))

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgba_array = asarray(array_bgra2rgba_c(bgra_array))
        np.ndarray [np.uint8_t, ndim=3] array1 = empty((chunk, chunk, 4), dtype=uint8)
        int chunkx = 0
        int chunky = 0
        int rows = 0
        int columns = 0

    animation = []

    # modify the chunk size
    if tweak_ and args is not None:

        if isinstance(args, tuple):
            try:
                chunkx = args[0][0]
                chunky = args[0][1]
            except IndexError:
                raise IndexError('Parse argument not understood.')
            if chunkx==0 or chunky==0:
                raise ValueError('Chunkx and chunky cannot be equal to zero.')
            if (width % chunkx) != 0:
                raise ValueError('Chunkx size value is not a correct fraction of %s ' % width)
            if (height % chunky) != 0:
                raise ValueError('Chunky size value is not a correct fraction of %s ' % height)
        else:
            raise ValueError('Parse argument not understood.')
    else:
        chunkx, chunky = chunk, chunk

    # get all the sprites
    for rows in range(rows_):
        for columns in range(columns_):
            array1 = rgba_array[rows * chunky:(rows + 1) * chunky, columns * chunkx:(columns + 1) * chunkx, :]
            animation.append(pygame.image.frombuffer(ascontiguousarray(array1), (chunkx, chunky), 'RGBA'))
    return animation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# TODO: NOT WORKING
def spritesheet(file, int chunk, int columns_, int rows_, tweak_: bool = False, *args):
    """
    Extract all the sprites from a sprite sheet.

    All output sprites will have transparency set by the colorkey value (default black)
    This will only work on Surfaces that have 24-bit or 32-bit formats.
    Spritesheet alpha transparency will be disregards.

    :param file: str,  full path to the texture
    :param chunk: int, size of a single image in bytes e.g 64x64 (equal
    :param rows_: int, number of rows
    :param columns_: int, number of column
    :param tweak_: bool, modify the chunk sizes (in bytes) in order to process
                   data with non equal width and height e.g 320x200
    :param args: tuple, used with theak_, args is a tuple containing the new chunk size,
                 e.g (320, 200)
    :return: list, Return textures (pygame surface) containing alpha transparency (set by colorkey) into a
            python list
    """
    assert isinstance(file, str), \
        'Expecting string for argument file got %s: ' % type(file)
    assert isinstance(chunk, int),\
        'Expecting int for argument number got %s: ' % type(chunk)
    assert isinstance(rows_, int) and isinstance(columns_, int), \
        'Expecting int for argument rows_ and columns_ ' \
        'got %s, %s ' % (type(rows_), type(columns_))

    try:
        # imread() decodes the image into a matrix with the color channels
        # stored in the order of Blue, Green and Red respectively.
        # It specifies to load a color image. Any transparency of image will be neglected.
        # It is the default flag. Alternatively, we can pass integer value 1 for this flag.
        bgr_array = cv2.imread(file, cv2.IMREAD_COLOR)

        if bgr_array is None:
            raise ValueError

    except Exception:
        raise FileNotFoundError('\npImage %s is not found ' % file)

    cdef int width, height, dim
    try:
        width, height, dim = bgr_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.') 

    if width == 0 or height == 0 or dim !=3:
        raise ValueError('image with incorrect dimensions'
                         ' must be (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % (width, height, dim))

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb_array = asarray(array_bgr2rgb_c(bgr_array)).transpose([1, 0, 2])
        np.ndarray[np.uint8_t, ndim=3] array1 = empty((chunk, chunk, 3), dtype=uint8)
        int chunkx
        int chunky
        int rows = 0
        int columns = 0

    # modify the chunk size
    if tweak_ and args is not None:

        if isinstance(args, tuple):
            try:
                chunkx = args[0][0]
                chunky = args[0][1]
            except IndexError:
                raise IndexError('Parse argument not understood.')
            if chunkx==0 or chunky==0:
                raise ValueError('Chunkx and chunky cannot be equal to zero.')
            if (width % chunkx) != 0:
                raise ValueError('Chunkx size value is not a correct fraction of %s ' % width)
            if (height % chunky) != 0:
                raise ValueError('Chunky size value is not a correct fraction of %s ' % height)
        else:
            raise ValueError('Parse argument not understood.')
    else:
        chunkx, chunky = chunk, chunk

    animation = []
    # split sprite-sheet into many sprites
    for rows in range(rows_):
        for columns in range(columns_):
            array1 = rgb_array[columns * chunkx:(columns + 1) * chunkx, rows * chunky:(rows + 1) * chunky, :]
            surface_ = pygame.pixelcopy.make_surface(array1).convert()
            surface_.set_colorkey((0, 0, 0, 0), RLEACCEL)
            animation.append(surface_)
    return animation

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def spritesheet_fs8(file, int chunk, int columns_, int rows_, tweak_: bool = False, *args):
    """
    Retrieve all sprites from a sprite sheets.
    All individual sprite will contains transparency set by the colorkey value (default black)
    This will only work on Surfaces that have 24-bit or 32-bit formats.

    :param file: str,  full path to the texture
    :param chunk: int, size of a single image in bytes e.g 64x64 (equal
    :param rows_: int, number of rows
    :param columns_: int, number of column
    :param tweak_: bool, modify the chunk sizes (in bytes) in order to process
                   data with non equal width and height e.g 320x200
    :param args: tuple, used with theak_, args is a tuple containing the new chunk size,
                 e.g (320, 200)
    :return: list, Return textures (pygame surface) containing per-pixel transparency into a
            python list
    """
    assert isinstance(file, str), \
        'Expecting string for argument file got %s: ' % type(file)
    assert isinstance(chunk, int),\
        'Expecting int for argument number got %s: ' % type(chunk)
    assert isinstance(rows_, int) and isinstance(columns_, int), \
        'Expecting int for argument rows_ and columns_ ' \
        'got %s, %s ' % (type(rows_), type(columns_))

    cdef int width, height
    
    try:
        image_ = pygame.image.load(file)
        width, height = image_.get_size()

    except (pygame.error, ValueError):
        raise FileNotFoundError('\nFile %s is not found ' % file)

    if width==0 or height==0:
        raise ValueError(
            'Surface dimensions is not correct, must be: (w>0, h>0) got (w:%s, h:%s) ' % (width, height))

    try:
        # Reference pixels into a 3d array
        # pixels3d(Surface) -> array
        # Create a new 3D array that directly references the pixel values
        # in a Surface. Any changes to the array will affect the pixels in
        # the Surface. This is a fast operation since no data is copied.
        # This will only work on Surfaces that have 24-bit or 32-bit formats.
        # Lower pixel formats cannot be referenced.
        rgb_array_ = pixels3d(image_)

    except (pygame.error, ValueError):
        # Copy pixels into a 3d array
        # array3d(Surface) -> array
        # Copy the pixels from a Surface into a 3D array.
        # The bit depth of the surface will control the size of the integer values,
        # and will work for any type of pixel format.
        # This function will temporarily lock the Surface as
        # pixels are copied (see the Surface.lock()
        # lock the Surface memory for pixel access
        # - lock the Surface memory for pixel access method).
        rgb_array_ = array3d(image_)


    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb_array = rgb_array_
        np.ndarray[np.uint8_t, ndim=3] array1 = empty((chunk, chunk, 3), dtype=uint8)
        int chunkx
        int chunky
        int rows = 0
        int columns = 0

    # modify the chunk size
    if tweak_ and args is not None:

        if isinstance(args, tuple):
            try:
                chunkx = args[0][0]
                chunky = args[0][1]
            except IndexError:
                raise IndexError('Parse argument not understood.')
            if chunkx==0 or chunky==0:
                raise ValueError('Chunkx and chunky cannot be equal to zero.')
            if (width % chunkx) != 0:
                raise ValueError('Chunkx size value is not a correct fraction of %s ' % width)
            if (height % chunky) != 0:
                raise ValueError('Chunky size value is not a correct fraction of %s ' % height)
        else:
            raise ValueError('Parse argument not understood.')
    else:
        chunkx, chunky = chunk, chunk

    animation = []
    # split sprite-sheet into many sprites
    for rows in range(rows_):
        for columns in range(columns_):
            array1 = rgb_array[columns * chunkx:(columns + 1) * chunkx, rows * chunky:(rows + 1) * chunky, :]
            surface_ = pygame.pixelcopy.make_surface(array1).convert()
            surface_.set_colorkey((0, 0, 0, 0), RLEACCEL)
            animation.append(surface_)
    return animation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef spritesheet_new_c(file_, int chunk_, int columns_, int rows_):
    """
    Extract all the sprites from a sprite sheet
    All sprites will be encoded with the flag RLEACCEL,
    Black pixels will be transparent.
    
    :param file_: python string, path to the surface to load 
    :param chunk_: integer, sprite size e.g 64x64 sprite -> enter 64
    :param columns_: integer, count of sprites added horizontally 
    :param rows_:  integer, count of sprites added vertically 
    :return: Return a python list containing all the sprites, black pixels will be transparent.
    """
    assert isinstance(file_, str), \
           "Argument file_ must be a python string type, got %s " % type(file_)
    assert isinstance(chunk_, int), \
           "Argument chunk_ must be a python int type, got %s " % type(chunk_)
    assert isinstance(columns_, int), \
           "Argument columns_ must be a python int type, got %s " % type(columns_)
    assert isinstance(rows_, int), \
           "Argument rows_ must be a python int type, got %s " % type(rows_)
    
    try:
        surface = pygame.image.load(file_)
    except (pygame.error, ValueError):
        raise FileNotFoundError('\nFile %s not found ' % file_)
    animation = []
    
    cdef:
        int rows = 0
        int columns = 0
        
    for rows in range(rows_):
        for columns in range(columns_):
            new_surface = Surface((chunk_, chunk_), flags=RLEACCEL)
            new_surface.blit(surface, (0, 0), (columns * chunk_, rows * chunk_, chunk_, chunk_))
            new_surface.set_colorkey((0, 0, 0, 0), RLEACCEL)
            animation.append(new_surface)
    return animation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_array_c_code(unsigned char[:, :, :] rgb_array_c, unsigned char[:, :] alpha_c):
    """
    Stack array RGB values with alpha channel.
    
    :param rgb_array_c: numpy.ndarray (w, h, 3) uint8 containing RGB values 
    :param alpha_c: numpy.ndarray (w, h) uint8 containing alpha values 
    :return: return a numpy.ndarray (w, h, 4) uint8, stack array of RGBA values
    The values are copied into a new array (out array is not transpose).
    """
    cdef int width, height
    try:
        width, height = (<object> rgb_array_c).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char[:, :, ::1] new_array =  empty((width, height, 4), dtype=uint8)
        int i=0, j=0
    # Equivalent to a numpy dstack
    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                new_array[i, j, 0], new_array[i, j, 1], new_array[i, j, 2], \
                new_array[i, j, 3] =  rgb_array_c[i, j, 0], rgb_array_c[i, j, 1], \
                                   rgb_array_c[i, j, 2], alpha_c[i, j]
    return asarray(new_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_array_c_transpose(unsigned char[:, :, :] rgb_array_c, unsigned char[:, :] alpha_c):
    """
    Stack RGB values an alpha channel together and transpose the array.
    (width and height) are transposed.
    
    :param rgb_array_c: numpy.ndarray (w, h, 3) uint8 containing RGB values 
    :param alpha_c: numpy.ndarray (w, h) uint8 containing alpha values 
    :return: return a numpy.ndarray (w, h, 4) uint8, stack array of RGBA values
    The values are transposed such as width becomes height (equivalent to numpy.transpose(1, 0, 2)).
    """
    cdef int width, height
    try:
        width, height = (<object> rgb_array_c).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    cdef:
        unsigned char[:, :, ::1] new_array =  empty((height, width, 4), dtype=uint8)
        int i=0, j=0

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2], \
                new_array[j, i, 3] =  rgb_array_c[i, j, 0], rgb_array_c[i, j, 1], \
                                   rgb_array_c[i, j, 2], alpha_c[i, j,]
    return asarray(new_array)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_array_from_buffer_c(buffer_, size_: tuple):
    """
    Create a RGBA array with shapes (w, h, 4) containing alpha channel 
    :param buffer_: pygame.BufferProxy, use get_view() to return a buffer from a given surface.
    e.g :
    IMAGE = pygame.image.load('Namiko1.png')
    buffer_ = IMAGE.get_view('2')
    :param size_: tuple, size width and height of a pygame surface e.g size_ = IMAGE.get_size()
    :return: return a 3d array with RGBA values (uint8)
    """
    return numpy.frombuffer(buffer_, dtype=uint8).reshape((*size_, 4))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_surface_c(rgba_array):
    """
    This function is used for 24-32 bit pygame surface with pixel alphas transparency layer
    make_surface(RGBA array) -> Surface
    Argument rgba_array is a 3d numpy array like (width, height, RGBA)
    This method create a 32 bit pygame surface that combines RGB values and alpha layer.
    :param rgba_array: 3D numpy array created with the method surface.make_array.
                       Combine RGB values and alpha values.
    :return:           Return a pixels alpha surface.This surface contains a transparency value
                       for each pixels.
    """
    return pygame.image.frombuffer((rgba_array.transpose([1, 0, 2])).tobytes(),
                                   (rgba_array.shape[0], rgba_array.shape[1]), 'RGBA')


# BUFFERPROXY, RETURN A PYGAME.SURFACE WITH PER-PIXEL
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline make_surface_c_1(buffer_, w, h):
    return pygame.image.frombuffer(buffer_, (w, h), 'RGBA')

# DIRECT, RETURN A PYGAME.SURFACE WITH PER-PIXEL
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_surface_c_2(rgba_array_):
    assert isinstance(rgba_array_, numpy.ndarray),\
        'Expecting numpy.ndarray for argument rgba_array, got %s ' % type(rgba_array_)
    cdef int w, h
    w, h = rgba_array_.shape[:2]
    return pygame.image.frombuffer(rgba_array_, (h, w), 'RGBA')

# TRANSFORM ARRAY TO BYTES STRING AND RETURN A PYGAME.SURFACE WITH PER-PIXEL
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_surface_c_4(rgba_array_):
    assert isinstance(rgba_array_, numpy.ndarray),\
        'Expecting numpy.ndarray for argument rgba_array, got %s ' % type(rgba_array_)
    cdef int w, h
    w, h = rgba_array_.shape[:2]
    return pygame.image.frombuffer(rgba_array_.tobytes(), (w, h), 'RGBA')

# ARRAY COPY CONTIGUOUS AND RETURN A PYGAME.SURFACE WITH PER-PIXEL
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_surface_c_5(rgba_array_):
    assert isinstance(rgba_array_, numpy.ndarray),\
        'Expecting numpy.ndarray for argument rgba_array, got %s ' % type(rgba_array_)
    cdef int w, h
    w, h = rgba_array_.shape[:2]
    return pygame.image.frombuffer(rgba_array_.copy('C'), (w, h), 'RGBA')

# TRANSPOSE ARRAY AND MAKE IT CONTIGUOUS IF NEEDED AND RETURN A PYGAME.SURFACE
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_surface_c_6(rgba_array_):
    assert isinstance(rgba_array_, numpy.ndarray),\
        'Expecting numpy.ndarray for argument rgba_array, got %s ' % type(rgba_array_)
    cdef int w, h
    w, h = rgba_array_.shape[:2]
    transpose_array = rgba_array_.transpose([1, 0, 2])
    if not transpose_array.flags['C_CONTIGUOUS']:
        transpose_array =  ascontiguousarray(transpose_array)
    return pygame.image.frombuffer(transpose_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef shadow_32c(image, float attenuation):
    """
    Create a greyscale from a given pygame surface and apply an attenuation to it.
    Compatible only with 32-bit format pixel (image containing per-pixel transparency) or
    image converted with the pygame method convert_alpha().
    The output image is a greyscale image format 32bit with transparency.
    This method is loading RGB and alpha values from the texture in order to create the final image. 
    
    :param image: Surface containing alpha channel
    :param attenuation: float; value can be in range [0 ... 1.0/2295.0]. Lower the greyscale intensity.
    :return: Return a greyscale pygame surface (32-bit format) with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   
        2 PASS MODE: | 32-bit  | convert()        |    
        3 PASS MODE: | 32-bit  | convert_alpha()  |   
        
        4 PASS MODE: | 24-bit  |                  |   
        5 PASS MODE: | 24-bit  | convert()        |  
        6 PASS MODE: | 24-bit  | convert_alpha()  |  
        
        7 PASS MODE: | 8-bit   |                  |   
        8 PASS MODE: | 8-bit   | convert()        |   
        9 PASS MODE: | 8-bit   | convert_alpha()  |   
    """
                         
    assert isinstance(image, Surface),\
        "Argument image must be a Surface, got %s " % type(image)
    assert image.get_bitsize() is 32, \
        'Surface pixel format is not 32 bit, got %s ' % image.get_bitsize()

    try:
        rgb = pixels3d(image)
        alpha = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('\Incompatible image.')
        
    cdef int width, height  
    width, height = image.get_size()
    
    cdef:
        int w = width
        int h = height
        unsigned char [:, :, :] rgb_array =  rgb
        unsigned char [:, ::1] alpha_array =  alpha
        unsigned char [:, :, ::1] greyscale_array = empty((h, w, 4), dtype=uint8)
        int gray, i=0, j=0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                gray = <int>((rgb_array[i, j, 0] + rgb_array[i, j, 1] + rgb_array[i, j, 2] ) *  attenuation)
                greyscale_array[j, i, 0], greyscale_array[j, i, 1], greyscale_array[j, i, 2] = gray, gray, gray
                greyscale_array[j, i, 3] = alpha_array[i, j]
    return pygame.image.frombuffer(greyscale_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef shadow_32b(surface_:Surface, float attenuation):
    """
    Create a greyscale from a given pygame surface and apply an attenuation to it.
    Compatible only with 32-bit format pixel (image containing per-pixel transparency) or
    image converted with the pygame method convert_alpha().
    The output image is a greyscale image format 32bit with transparency.
    This method is using a pygame bufferProxy object to access RGBA values
    
    :param image: Surface containing alpha channel
    :param attenuation: float; value can be in range [0 ... 1.0/2295.0] lower the greyscale intensity.
    :return: Return a greyscale pygame surface (32-bit format) with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   
        2 PASS MODE: | 32-bit  | convert()        |    
        3 PASS MODE: | 32-bit  | convert_alpha()  |   
        
        4 PASS MODE: | 24-bit  |                  |   
        5 PASS MODE: | 24-bit  | convert()        |  
        6 PASS MODE: | 24-bit  | convert_alpha()  |  
        
        7 PASS MODE: | 8-bit   |                  |   
        8 PASS MODE: | 8-bit   | convert()        |   
        9 PASS MODE: | 8-bit   | convert_alpha()  |   
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        int i = 0
        unsigned char grey

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            grey = <unsigned char>((c_buffer[i] + c_buffer[i + 1] + c_buffer[i + 2]) * attenuation)
            c_buffer[i], c_buffer[i + 1], c_buffer[i + 2], = grey, grey, grey

    return pygame.image.frombuffer(c_buffer, (width, height), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef rgb_split_c(surface_: Surface):
    """
    Split RGB channels from a given Surface
    Image can be converted to fast blit with convert() or convert_alpha()
    Return a tuple of RGB surface (without alpha channel)
    
    :param surface_: Surface 8, 24-32 bit format 
    :return: Return a tuple containing 3 surfaces (R, G, B) without alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00318
        2 PASS MODE: | 32-bit  | convert()        |   0.00235 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00239
        
        4 PASS MODE: | 24-bit  |                  |   0.00276
        5 PASS MODE: | 24-bit  | convert()        |   0.00235
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00243
        
        7 FAIL MODE: | 8-bit   |                  |   unsupported colormasks for red reference array
        8 PASS MODE: | 8-bit   | convert()        |   0.0035
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0036
    """

    cdef int width, height
    width, height = surface_.get_size()

    assert isinstance(surface_, Surface), \
        '\nPositional argument surface_ must be a Surface, got %s ' % type(surface_)
    
    if width == 0 or height == 0:
        raise ValueError('\nIncorrect pixel size or wrong format.'
                         '\nsurface_ dimensions (width, height) cannot be null.')
    try:
        rgb_array = pygame.surfarray.pixels3d(surface_)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible pixel format.')

    cdef:
        unsigned char [:, :, ::1] red_s = numpy.empty((height, width, 3), dtype=uint8)
        unsigned char [:, :, ::1] green_s = numpy.empty((height, width, 3), dtype=uint8)
        unsigned char [:, :, ::1] blue_s = numpy.empty((height, width, 3), dtype=uint8)
        unsigned char [:, :, :] rgb = rgb_array
        int i=0, j=0
        
    with nogil:
        for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                red_s[j, i, 0], red_s[j, i, 1], red_s[j, i, 2] = rgb[i, j, 0], 0, 0
                green_s[j, i, 1], green_s[j, i, 0], green_s[j, i, 2] = rgb[i, j, 1], 0, 0
                blue_s[j, i, 2], blue_s[j, i, 1], blue_s[j, i, 0] = rgb[i, j, 2], 0, 0

    return pygame.image.frombuffer(red_s, (width, height), 'RGB'),\
           pygame.image.frombuffer(green_s, (width, height), 'RGB'),\
           pygame.image.frombuffer(blue_s, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef rgb_split_b(surface_: Surface):
    """
    Image can be a 8 - 24bit format converted with pygame method (convert, convert_alpha) or 32 bit format
    see TEST for more details
    
    :param surface_: Surface 8, 24-32 bit format 
    :return: Return a tuple containing 3 surfaces (R, G, B) without alpha channel
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00265 DO NOT USE BGRA FORMAT
        2 PASS MODE: | 32-bit  | convert()        |   0.00180 OK 
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00139 OK
        
        4 FAIL MODE: | 24-bit  |                  |   Surface pixel format is not 32 bit, got 24 
        5 PASS MODE: | 24-bit  | convert()        |   0.00148 OK
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00132 OK 
        
        7 FAIL MODE: | 8-bit   |                  |   Surface pixel format is not 32 bit, got 8 
        8 PASS MODE: | 8-bit   | convert()        |   0.00118 OK 
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00117 OK
    """

    cdef int width, height
    width, height = surface_.get_size()

    assert isinstance(surface_, Surface), \
        '\nPositional argument surface_ must be a Surface, got %s ' % type(surface_)

    assert surface_.get_bitsize() == 32, \
         'Surface pixel format is not 32 bit, got %s ' % surface_.get_bitsize()

    
    if width == 0 or height == 0:
        raise ValueError('\nIncorrect pixel size or wrong format.'
                         '\nsurface_ dimensions (width, height) cannot be null.')
    try:
        rgb_buffer = surface_.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('Incompatible pixel format.')

    
    cdef:
        unsigned char [:, :, ::1] red_s = numpy.zeros((width, height, 3), dtype=uint8)
        unsigned char [:, :, ::1] green_s = numpy.zeros((width, height, 3), dtype=uint8)
        unsigned char [:, :, ::1] blue_s = numpy.zeros((width, height, 3), dtype=uint8)
        unsigned char [:] cbuffer = numpy.frombuffer(rgb_buffer, dtype=numpy.uint8)
        int b_length = rgb_buffer.length
        int i = 0, j = 0, ii=0, n =0
        
    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            n = i >> 2
            j = <int>(n / height)
            ii = n - (j * height)
            red_s[j, ii, 0] = cbuffer[i + 2]
            green_s[j, ii, 1] = cbuffer[i + 1]
            blue_s[j, ii, 2] = cbuffer[i]

    return pygame.image.frombuffer(red_s, (width, height), 'RGB'),\
           pygame.image.frombuffer(green_s, (width, height), 'RGB'),\
           pygame.image.frombuffer(blue_s, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef rgb_split32_c(surface_: Surface):
    """
    Split RGB channels from a given Surface
    Surface is a 32-bit format with per-pixel transparency.
    Return a tuple containing 3 surfaces (R, G, B) with per-pixel transparency.
    
    :param surface_: Surface 8, 24-32 bit format with alpha channel
    :return: Return a tuple containing 3 surfaces (R, G, B) with alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0031
        2 FAIL MODE: | 32-bit  | convert()        |   FAIL Incompatible pixel format.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0029
        
        4 FAIL MODE: | 24-bit  |                  |   FAIL Surface pixel format is not 32 bit, got 24 
        5 FAIL MODE: | 24-bit  | convert()        |   FAIL  Incompatible pixel format.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.029 FULL OPACITY 
        
        7 FAIL MODE: | 8-bit   |                  |   FAIL Surface pixel format is not 32 bit, got 8
        8 FAIL MODE: | 8-bit   | convert()        |   FAIL Incompatible pixel format.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.029 FULL OPACITY
    """

    assert isinstance(surface_, Surface), \
        '\nPositional argument surface_ must be a Surface, got %s ' % type(surface_)
    
    assert surface_.get_bitsize() == 32, \
        "\nSurface pixel format is not 32 bit, got %s " % surface_.get_bitsize()

    cdef int w, h
    w, h = surface_.get_size()
    
    if w == 0 or h == 0:
        raise ValueError('\nIncorrect pixel size or wrong format.'
                         '\nsurface_ dimensions (width, height) cannot be null.')
    try:
        alpha_array = pixels_alpha(surface_)
        
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    try:
        rgb_array = pixels3d(surface_)

    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    cdef:
        unsigned char [:, :, ::1] red_s = numpy.zeros((h, w, 4), dtype=uint8)
        unsigned char [:, :, ::1] green_s = numpy.zeros((h, w, 4), dtype=uint8)
        unsigned char [:, :, ::1] blue_s = numpy.zeros((h, w, 4), dtype=uint8)
        
        unsigned char [:, :, :] rgb = rgb_array
        unsigned char [:, :] alpha = alpha_array
        unsigned char a
        int i=0, j=0

    with nogil:
        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                a = alpha[i, j]
                red_s[j, i, 0], red_s[j, i, 3] = rgb[i, j, 0], a
                green_s[j, i, 1], green_s[j, i, 3] = rgb[i, j, 1], a
                blue_s[j, i, 2], blue_s[j, i, 3] = rgb[i, j, 2], a

    return pygame.image.frombuffer(red_s, (w, h), 'RGBA'),\
           pygame.image.frombuffer(green_s, (w, h), 'RGBA'),\
           pygame.image.frombuffer(blue_s, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef rgb_split32_b(surface_: Surface):
    """
    Split channels from a given surface (return a tuple of surfaces red, green, blue). 
    The surface must be a 32 bit format with per-pixel transparency in order to 
    extract the pixel correctly and return red, green, blue channels.
    This method is using a buffer instead of numpy ndarray to speed up the process.
    
    :param surface_: Surface 8, 24-32 bit format with alpha channel, see TEST for more information
    :return: Return a tuple containing 3 surfaces (R, G, B) with per-pixel transparency 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00216 DO NOT USE BGRA FORMAT
        2 PASS MODE: | 32-bit  | convert()        |   0.00203 DO NOT USE, missing per-pixel transparency (blueish image)  
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00208 OK 
        
        4 FAIL MODE: | 24-bit  |                  |   Invalid pixel format 
        5 FAIL MODE: | 24-bit  | convert()        |   Invalid pixel format
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00202 FULL OPACITY (channels are encoded 
                                                      in RGBA but alpha values are set to maximum opacity 255)
        
        7 FAIL MODE: | 8-bit   |                  |   Invalid pixel format 
        8 FAIL MODE: | 8-bit   | convert()        |   Invalid pixel format
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00209 FULL OPACITY (channels are encoded 
                                                      in RGBA but alpha values are set to maximum opacity 255)
    """

    assert isinstance(surface_, Surface), \
        '\nPositional argument surface_ must be a Surface, got %s ' % type(surface_)

    assert (surface_.get_bitsize() == 32) and (surface_.get_flags() & pygame.SRCALPHA), \
         'Invalide pixel format '

    cdef int w, h
    w, h = surface_.get_size()
    
    if w == 0 or h == 0:
        raise ValueError('\nIncorrect pixel size or wrong format.'
                         '\nsurface_ dimensions (width, height) cannot be null.')
    try:
        rgba_buffer = surface_.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('\nInvalid pixel format')

    cdef:
        unsigned char [:, :, ::1] red_s = numpy.zeros((w, h, 4), dtype=uint8)
        unsigned char [:, :, ::1] green_s = numpy.zeros((w, h, 4), dtype=uint8)
        unsigned char [:, :, ::1] blue_s = numpy.zeros((w, h, 4), dtype=uint8)
        unsigned char [:] cbuffer = numpy.frombuffer(rgba_buffer, dtype=numpy.uint8)
        int b_length = rgba_buffer.length
        int i = 0, j = 0, n = 0, ii = 0

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            n = i >> 2
            j = <int>(n / h)
            ii = n - (j * h)
            red_s[j, ii, 0], red_s[j, ii, 3] = cbuffer[i + 2], cbuffer[i + 3]
            green_s[j, ii, 1], green_s[j, ii, 3] = cbuffer[i + 1], cbuffer[i + 3]
            blue_s[j, ii, 2], blue_s[j, ii, 3] = cbuffer[i], cbuffer[i + 3]

    return pygame.image.frombuffer(red_s, (w, h), 'RGBA'),\
           pygame.image.frombuffer(green_s, (w, h), 'RGBA'),\
           pygame.image.frombuffer(blue_s, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef red_channel_c(surface_: Surface):
    """
    Extract red channel from a given surface
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a Surface (Red channel) without alpha channel.
    
     TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0007
        2 PASS MODE: | 32-bit  | convert()        |   0.0007
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006
        
        4 PASS MODE: | 24-bit  |                  |   0.0006
        5 PASS MODE: | 24-bit  | convert()        |   0.0006
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0006
        
        7 PASS MODE: | 8-bit   |                  |   0.007  *SLOW d/t array3d instead of pixels3d
        8 PASS MODE: | 8-bit   | convert()        |   0.0006
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0006
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    try:
        rgb = pixels3d(surface_)

    except (pygame.error, ValueError):
        try:
            rgb = array3d(surface_)

        except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')

    cdef int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, :, ::1] red_array = zeros((h, w, 3), dtype=uint8)
        int i = 0, j = 0
    with nogil:
        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red_array[j, i, 0] = rgb_array[i, j, 0]

    return pygame.image.frombuffer(red_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef red_channel_b(surface_: Surface):
    """
    Extract red channel from a given surface 
    see TEST for all compatible modes.
    The red channel will have per-pixel transparency information with alpha set to maximum opacity 255.
    If you do not wish to keep the per-pixel info, use pygame convert method post processing.    
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a 32 bit Surface (Red channel) with per-pixel transparency (RGBA FORMAT) 
    with alpha channel set to 255.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00015 DO NOT USE BGRA FORMAT PIXEL (red channel is in fact the 
                                                      the blue channel)
        2 PASS MODE: | 32-bit  | convert()        |   0.00015
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00015
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.00015
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00015
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size  
        8 PASS MODE: | 8-bit   | convert()        |   0.00015
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00015
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    try:
        rgb_buffer = surface_.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    cdef int w, h
    w, h = surface_.get_size()

    cdef:
        int b_length = rgb_buffer.length
        unsigned char [:] cbuffer = numpy.frombuffer(rgb_buffer, dtype=numpy.uint8)
        int i = 0
    
    with nogil:
        for i in prange(w, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            cbuffer[i + 2], cbuffer[i + 1], cbuffer[i + 3] = 0, 0, 255
                
    return pygame.image.frombuffer(cbuffer, (w, h), 'RGBA')
                                                     

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef green_channel_c(surface_: Surface):
    """
    Extract green channel from a given surface
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a Surface (green channel) without alpha channel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0007
        2 PASS MODE: | 32-bit  | convert()        |   0.0007
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006
        
        4 PASS MODE: | 24-bit  |                  |   0.0006
        5 PASS MODE: | 24-bit  | convert()        |   0.0006
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0006
        
        7 PASS MODE: | 8-bit   |                  |   0.007  *SLOW d/t array3d instead of pixels3d
        8 PASS MODE: | 8-bit   | convert()        |   0.0006
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0006
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    try:
        rgb = pixels3d(surface_)

    except (pygame.error, ValueError):
        try:
            rgb = array3d(surface_)

        except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')

    cdef int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, :, ::1] empty_array = zeros((h, w, 3), dtype=uint8)
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                empty_array[j, i, 1] = rgb_array[i, j, 1]
                
    return pygame.image.frombuffer(empty_array, (w, h), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef green_channel_b(surface_: Surface):
    """
    Extract green channel from a given surface 
    see TEST for all compatible modes.
    The green channel will have per-pixel transparency information with alpha set to maximum opacity 255.
    If you do not wish to keep the per-pixel info, use pygame convert method post processing.    
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a 32 bit Surface (green channel) with per-pixel transparency (RGBA FORMAT) 
    with alpha channel set to 255.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00015 
        2 PASS MODE: | 32-bit  | convert()        |   0.00015
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00015
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.00015
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00015
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size  
        8 PASS MODE: | 8-bit   | convert()        |   0.00015
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00015
    
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    try:
        rgb_buffer = surface_.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    cdef int w, h
    w, h = surface_.get_size()

    cdef:
        int b_length = rgb_buffer.length
        unsigned char [:] cbuffer = numpy.frombuffer(rgb_buffer, dtype=numpy.uint8)
        int i = 0
        
    with nogil:
        for i in prange(w, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            cbuffer[i], cbuffer[i + 2], cbuffer[i + 3] = 0, 0, 255
                
    return pygame.image.frombuffer(cbuffer, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef blue_channel_c(surface_: Surface):
    """
    Extract blue channel from a given surface
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a Surface (blue channel) without alpha channel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0007
        2 FAIL MODE: | 32-bit  | convert()        |   0.0007
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0006
        
        4 FAIL MODE: | 24-bit  |                  |   0.0006
        5 FAIL MODE: | 24-bit  | convert()        |   0.0006
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0006
        
        7 FAIL MODE: | 8-bit   |                  |   0.007  *SLOW d/t array3d instead of pixels3d
        8 FAIL MODE: | 8-bit   | convert()        |   0.0006
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0006
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    try:
        rgb = pixels3d(surface_)

    except (pygame.error, ValueError):
        try:
            rgb = array3d(surface_)

        except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')

    cdef int w, h
    w, h = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, :, ::1] empty_array = zeros((h, w, 3), dtype=uint8)
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                empty_array[j, i, 2] = rgb_array[i, j, 2]
                
    return pygame.image.frombuffer(empty_array, (w, h), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef blue_channel_b(surface_: Surface):
    """
    Extract blue channel from a given surface
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a Surface (Red channel) without alpha channel.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00015 DO NOT USE (INPUT BGRA FORMAT) 
        2 FAIL MODE: | 32-bit  | convert()        |   0.00015
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00015
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 FAIL MODE: | 24-bit  | convert()        |   0.00015
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00015
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 FAIL MODE: | 8-bit   | convert()        |   0.00015
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00015
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    try:
        rgb_buffer = surface_.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    cdef int w, h
    w, h = surface_.get_size()

    cdef:
        int b_length = rgb_buffer.length
        unsigned char [:] cbuffer = numpy.frombuffer(rgb_buffer, dtype=numpy.uint8)
        int i = 0
        
    with nogil:
        for i in prange(w, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            cbuffer[i], cbuffer[i + 1], cbuffer[i + 3] = 0, 0, 255
                
    return pygame.image.frombuffer(cbuffer, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fish_eye_c(image):
    """
    Transform an image into a fish eye lens model.
    
    :param image: Surface (8, 24-32 bit format) 
    :return: Return a Surface without alpha channel, fish eye lens model.
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0051
        2 PASS MODE: | 32-bit  | convert()        |   0.0052
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0051
        
        4 PASS MODE: | 24-bit  |                  |   0.0051
        5 PASS MODE: | 24-bit  | convert()        |   0.0051
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0051
        
        7 PASS MODE: | 8-bit   |                  |   0.0120 * SLOW (use array3d instead pixels3d)
        8 PASS MODE: | 8-bit   | convert()        |   0.0051
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0051
    """
    assert isinstance(image, Surface), \
        "\nArguement image is not a pygame.Surface type, got %s " % type(image)

    try:
        array = pixels3d(image)
    except (pygame.error, ValueError):
        try:
            array = array3d(image)
        except:
            raise ValueError('\nInvalid pixel format.')

    cdef double w, h
    w, h = image.get_size()
   
    assert (w!=0 and h!=0),\
        'Incorrect image format (w>0, h>0) got (w:%s h:%s) ' % (w, h)

    cdef:
        unsigned char [:, :, :] rgb_array = array
        int y=0, x=0, v
        double ny, ny2, nx, nx2, r, theta, nxn, nyn, nr
        int x2, y2
        double s = w * h
        double c1 = 2 / h
        double c2 = 2 / w
        double w2 = w / 2
        double h2 = h / 2
        unsigned char [:, :, ::1] rgb_empty = zeros((int(h), int(w), 3), dtype=uint8)

    with nogil:
        for y in prange(<int>h, schedule=SCHEDULE, num_threads=THREAD_NUMBER, chunksize=8):
            ny = y * c1 - 1
            ny2 = ny * ny
            for x in range(<int>w):
                nx = x * c2 - 1.0
                nx2 = nx * nx
                r = sqrt(nx2 + ny2)
                if 0.0 <= r <= 1.0:
                    nr = (r + 1.0 - sqrt(1.0 - (nx2 + ny2))) * 0.5
                    if nr <= 1.0:
                        theta = atan2(ny, nx)
                        nxn = nr * cos(theta)
                        nyn = nr * sin(theta)
                        x2 = <int>(nxn * w2 + w2)
                        y2 = <int>(nyn * h2 + h2)
                        v = <int>(y2 * w + x2)
                        if 0 <= v < s:
                            rgb_empty[y, x, 0], rgb_empty[y, x, 1], rgb_empty[y, x, 2] = rgb_array[x2, y2, 0],\
                            rgb_array[x2, y2, 1], rgb_array[x2, y2, 2]
    return pygame.image.frombuffer(rgb_empty, (<int>w, <int>h), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fish_eye_32c(image):
    """
    Transform an image into a fish eye lens model (compatible 32-bit)
    
    :param image: Surface (8, 24-32 bit format) with per-pixel or converted
                  with convet_alpha() method
    :return: Return a 32-bit Surface with alpha channel, fish eye lens model with per-pixel transparency
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0051
        2 PASS MODE: | 32-bit  | convert()        |   Incompatible pixel format.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0051
        
        4 FAIL MODE: | 24-bit  |                  |   Incompatible pixel format.
        5 FAIL MODE: | 24-bit  | convert()        |   Incompatible pixel format.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0051 input image has full opacity. Return 
        an image with full opacity into a eye lens model
                                                                        
        
        7 FAIL MODE: | 8-bit   |                  |   Incompatible pixel format.
        8 FAIL MODE: | 8-bit   | convert()        |   Incompatible pixel format.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0051 input image has full opacity. Return 
        an image with full opacity into a eye lens model
    """
    assert isinstance(image, Surface), \
        "\nArguement image is not a pygame.Surface type, got %s " % type(image)

    try:
        array = pixels3d(image)
        alpha = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    cdef double w, h
    w, h = image.get_size()
   
    assert (w!=0 and h!=0),\
        'Incorrect image format (w>0, h>0) got (w:%s h:%s) ' % (w, h)

    cdef:
        unsigned char [:, :, :] rgb_array = array
        unsigned char [:, :] alpha_array = alpha
        int y=0, x=0, v
        double ny, ny2, nx, nx2, r, theta, nxn, nyn, nr
        int x2, y2
        double s = w * h
        double c1 = 2 / h
        double c2 = 2 / w
        double w2 = w / 2
        double h2 = h / 2
        unsigned char [:, :, ::1] rgb_empty = zeros((int(h), int(w), 4), dtype=uint8)
    with nogil:
        for y in prange(<int>h, schedule=SCHEDULE, num_threads=THREAD_NUMBER, chunksize=8):
            ny = y * c1 - 1
            ny2 = ny * ny
            for x in range(<int>w):
                nx = x * c2 - 1.0
                nx2 = nx * nx
                r = sqrt(nx2 + ny2)
                if 0.0 <= r <= 1.0:
                    nr = (r + 1.0 - sqrt(1.0 - (nx2 + ny2))) * 0.5
                    if nr <= 1.0:
                        theta = atan2(ny, nx)
                        nxn = nr * cos(theta)
                        nyn = nr * sin(theta)
                        x2 = <int>(nxn * w2 + w2)
                        y2 = <int>(nyn * h2 + h2)
                        v = <int>(y2 * w + x2)
                        if 0 <= v < s:
                            rgb_empty[y, x, 0], rgb_empty[y, x, 1], \
                            rgb_empty[y, x, 2], rgb_empty[y, x, 3] = rgb_array[x2, y2, 0],\
                            rgb_array[x2, y2, 1], rgb_array[x2, y2, 2], rgb_array[x2, y2, 3]
    return pygame.image.frombuffer(rgb_empty, (<int>w, <int>h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef rotate_inplace_c(image: Surface, double angle):
    """
    Rotate an image inplace (centered)
    The final image will be smoothed/padded by adding neighboround pixels during rotation.
    
    :param image: Surface (8, 24-32 bit format)
    :param angle: pygame float, radian angle 
    :return: a rotated Surface without alpha channel
    """
    cdef int width, height, width2, height2
    width, height = image.get_size()
    width2, height2 = width >> 1, height >> 1
    try:
        array = pixels3d(image)
    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        raise ValueError('\nMake sure the surface_ contains per-pixel alpha transparency values.')

    cdef:
        float w_f = width
        float h_f = height
        float w_f_ = 2.0/w_f
        float h_f_ = 2.0/h_f
        int w = width
        int h = height
        float w2 = width2
        float h2 = height2
        unsigned char [:, :, ::1] rgb_empty = zeros((h, w, 3), dtype=uint8)
        unsigned char [:, :, :] rgb_array = array
        int ix, iy, x, y
        float nx, ny, radius, alpha
        float p = pi/180.0, cosi, sinu
        int r, g, b

    with nogil:    
        for ix in prange(0, w-1):
            nx = (ix * w_f_) - 1.0
            for iy in range(0, h-1):

                ny = (iy * h_f_) - 1.0
                radius = sqrt(nx * nx + ny * ny)

                alpha = -atan2(ny, nx) + angle * p
                cosi = cos(alpha) * radius
                sinu = sin(alpha) * radius

                x = <int>(cosi * w2 + w2)
                y = <int>(sinu * h2 + h2)

                x = min(x, w-1)
                x = max(x, 0)
                y = min(y, h-1)
                y = max(y, 0)
                ix = min(ix, w-1)
                ix = max(ix, 0)
                iy = min(iy, h-1)
                iy = max(iy, 0)

                rgb_empty[y, x, 0], rgb_empty[y, x, 1], rgb_empty[y, x, 2] = \
                    rgb_array[ix, iy, 0], rgb_array[ix, iy, 1], rgb_array[ix, iy, 2]
                if x + 1 < w-1:
                    rgb_empty[y, x + 1, 0], rgb_empty[y, x + 1, 1], rgb_empty[y, x + 1, 2] = \
                        rgb_array[ix, iy, 0], rgb_array[ix, iy, 1], rgb_array[ix, iy, 2]
                if x > 0:
                    rgb_empty[y, x - 1, 0], rgb_empty[y, x - 1, 1], rgb_empty[y, x - 1, 2] = \
                        rgb_array[ix, iy, 0], rgb_array[ix, iy, 1], rgb_array[ix,iy, 2]
                if y > 0:
                    rgb_empty[y - 1, x, 0], rgb_empty[y -1, x, 1], rgb_empty[y -1, x, 2] = \
                        rgb_array[ix, iy, 0], rgb_array[ix, iy, 1], rgb_array[ix, iy, 2]
                if y < h-1:
                    rgb_empty[y + 1, x, 0], rgb_empty[y + 1, x, 1], rgb_empty[y +1, x, 2] = \
                        rgb_array[ix, iy, 0], rgb_array[ix, iy, 1], rgb_array[ix, iy, 2]


    return pygame.image.frombuffer(rgb_empty, (w, h), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef rotate_c24(image: Surface, int angle):
    """
    Rotate an image (compatible 24-32 bit format)
    
    :param image: Surface(8, 24-32 bit format)
    :param angle: integer, angle in degrees
    :return: return a Surface without alpha channel
    """
    assert isinstance(image, Surface),\
        'Expecting a Surface for argument image got %s ' % type(image)
    assert isinstance(angle, int), \
        'Expecting an int for argument angle got %s ' % type(angle)

    cdef int w, h
    w, h = image.get_size()

    assert (w!=0 and h!=0), 'Incorrect image format got w:%s h:%s ' % (w, h)

    try:
        # todo this will not works for 24 bit use array3d instead
        array = pixels3d(image)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nThis will only work on Surfaces that have 24-bit format.')
    cdef:
        int w2 = w >> 1
        int h2 = h >> 1
        float [:] msin = empty(360, dtype=float32)
        float [:] mcos = empty(360, dtype=float32)
        float sinma
        float cosma
        int i, x, y, xt, yt, xs, ys
        float rad
        unsigned char [:, :, ::1] array_empty = zeros((h, w, 3), dtype=uint8)
        unsigned char [:, :, :] rgb_array = array
    # Pre-calculate sin and cos
    rad = <float>(pi / 180.0)

    for i in range(360):
        msin[i] = <float>(sin(-i * rad))
        mcos[i] = <float>(cos(-i * rad))

    with nogil:
        for x in prange(w):
            for y in range(h):
                xt = x - w2
                yt = y - h2
                sinma = msin[angle]
                cosma = mcos[angle]
                xs = <int>((cosma * xt - sinma * yt) + w2)
                ys = <int>((sinma * xt + cosma * yt) + h2)
                if 0 <= xs < w and 0 <= ys < h:
                    array_empty[y, x, 0], array_empty[y, x, 1], array_empty[y, x, 2] = \
                        rgb_array[xs, ys, 0], rgb_array[xs, ys, 1], rgb_array[xs, ys, 2]

    return pygame.image.frombuffer(array_empty, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef rotate_c32(image: Surface, int angle):
    """
    Rotate an image 32 bit format with per-pixel information
    
    :param image: Surface format 8, 24-32 bit containg per-pixel transparency
    :param angle: integer, angle in degrees 
    :return: a rotated image containing alpha channel
    """
    assert isinstance(image, Surface),\
        'Expecting a Surface for argument image got %s ' % type(image)
    assert isinstance(angle, int), \
        'Expecting an int for argument angle got %s ' % type(angle)

    cdef int w, h
    w, h = image.get_size()

    assert (w!=0 and h!=0), 'Incorrect image format got w:%s h:%s ' % (w, h)

    try:
        rgb_ = pixels3d(image)
        alpha_ = pixels_alpha(image)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nThis will only work on Surfaces that have 32-bit formats')

    cdef:
        int w2 = w >> 1
        int h2 = h >> 1
        float [:] msin = empty(360, dtype=float32)
        float [:] mcos = empty(360, dtype=float32)
        float sinma
        float cosma
        int i, x, y, xt, yt, xs, ys
        float rad
        unsigned char [:, :, ::1] array_empty = zeros((h, w, 4), dtype=uint8)
        # This can only work on 32-bit Surfaces with a per-pixel alpha value.
        unsigned char [:, ::1] alpha_array = alpha_
        unsigned char [:, :, :] rgb_array = rgb_

    # Pre-calculate sin and cos
    rad = <float>(pi / 180.0)
    for i in range(360):
        msin[i] = <float>(sin(-i * rad))
        mcos[i] = <float>(cos(-i * rad))
        
    with nogil:
        for x in prange(w):
            for y in range(h):
                xt = x - w2
                yt = y - h2
                sinma = msin[angle]
                cosma = mcos[angle]
                xs = int((cosma * xt - sinma * yt) + w2)
                ys = int((sinma * xt + cosma * yt) + h2)
                if 0 <= xs < w and 0 <= ys < h:
                    array_empty[y, x, 0], array_empty[y, x, 1], array_empty[y, x, 2], array_empty[y, x, 3] = \
                        rgb_array[xs, ys, 0], rgb_array[xs, ys, 1], rgb_array[xs, ys, 2], alpha_array[xs, ys]

    return pygame.image.frombuffer(array_empty, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_surface_24c(surface_: Surface, double shift_):
    """
    Rotate hue for a given pygame surface (compatible 8, 24 - 32 bit format pixel)
    
    :param surface_: Surface 8, 24-32 bit format 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: return a 24 - bit pygame Surface with no alpha channel (full opacity)
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.004
        2 PASS MODE: | 32-bit  | convert()        |   0.004
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.004
        
        4 PASS MODE: | 24-bit  |                  |   0.004
        5 PASS MODE: | 24-bit  | convert()        |   0.004
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.004                                                                       
        
        7 PASS MODE: | 8-bit   |                  |   0.004
        8 PASS MODE: | 8-bit   | convert()        |   0.004
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.004
        
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)
    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except:
            raise ValueError('\nInvalid pixel format.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        float r, g, b
        float h, s, v
        float rr, gg, bb, mx, mn
        float df, df_
        float f, p, q, t, ii

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER, chunksize=4):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                rr = r * ONE_255 # / 255.0
                gg = g * ONE_255 # / 255.0
                bb = b * ONE_255 # / 255.0
                mx = max(rr, gg, bb)
                mn = min(rr, gg, bb)
                df = mx-mn
                df_ = 1.0/df
                if mx == mn:
                    h = 0
                elif mx == rr:
                    h = (60 * ((gg-bb) * df_) + 360) % 360
                elif mx == gg:
                    h = (60 * ((bb-rr) * df_) + 120) % 360
                elif mx == bb:
                    h = (60 * ((rr-gg) * df_) + 240) % 360
                if mx == 0:
                    s = 0
                else:
                    s = df/mx
                v = mx
                h = (h * ONE_360) + shift_

                if s == 0.0:
                    r, g, b = v, v, v
                ii = <int>(h * 6.0)
                f = (h * 6.0) - ii
                p = v*(1.0 - s)
                q = v*(1.0 - s * f)
                t = v*(1.0 - s * (1.0 - f))
                ii = ii % 6

                if ii == 0:
                    r, g, b = v, t, p
                if ii == 1:
                    r, g, b = q, v, p
                if ii == 2:
                    r, g, b = p, v, t
                if ii == 3:
                    r, g, b = p, q, v
                if ii == 4:
                    r, g, b = t, p, v
                if ii == 5:
                    r, g, b = v, p, q

                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2] = <int>(r*255.0), <int>(g*255.0), <int>(b*255.0)

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_surface_32c(surface_: Surface, float shift_):
    """
    Rotate hue for a given pygame surface (compatible 32 bit with per-pixel transparency) or 
    8, 24 bit images converted with pygame methods convert_alpha.
    
    :param surface_: Surface 32 bit with per-pixel
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: return a 32 bit Surface with per-pixel information
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.004
        2 FAIL MODE: | 32-bit  | convert()        |   Compatible only for 32-bit format with per-pixel transparency.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.004
        
        4 FAIL MODE: | 24-bit  |                  |   Compatible only for 32-bit format with per-pixel transparency.
        5 FAIL MODE: | 24-bit  | convert()        |   Compatible only for 32-bit format with per-pixel transparency.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.004 FULL OPACITY (final image contains per-pixel but all alpha
                                                      values are set to maximum opacity 255)                                                                   
        
        7 FAIL MODE: | 8-bit   |                  |   Compatible only for 32-bit format with per-pixel transparency.
        8 FAIL MODE: | 8-bit   | convert()        |   Compatible only for 32-bit format with per-pixel transparency.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.004 FULL OPACITY (final image contains per-pixel but all alpha
                                                      values are set to maximum opacity 255)
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert 0.0 <= shift_ <= 1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
       raise ValueError('\nCompatible only for 32-bit format with per-pixel transparency.')

    cdef int width, height
    width, height = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :] alpha_array = alpha_
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        int i=0, j=0
        #float r, g, b
        #float h, s, v
        double r, g, b
        double h, s, v
        float rr, gg, bb, mx, mn
        float df, df_
        float f, p, q, t, ii
        double *hsv
        double *rgb

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                rr = r * ONE_255
                gg = g * ONE_255
                bb = b * ONE_255
                mx = max(rr, gg, bb)
                mn = min(rr, gg, bb)
                df = mx-mn
                df_ = 1.0/df
                if mx == mn:
                    h = 0
                elif mx == rr:
                    h = (60 * ((gg-bb) * df_) + 360) % 360
                elif mx == gg:
                    h = (60 * ((bb-rr) * df_) + 120) % 360
                elif mx == bb:
                    h = (60 * ((rr-gg) * df_) + 240) % 360
                if mx == 0:
                    s = 0
                else:
                    s = df/mx
                v = mx

                h = h * ONE_360 + shift_

                if s == 0.0:
                    r, g, b = v, v, v
                ii = <int>(h * 6.0)
                f = (h * 6.0) - ii
                p = v*(1.0 - s)
                q = v*(1.0 - s * f)
                t = v*(1.0 - s * (1.0 - f))
                ii = ii % 6

                if ii == 0:
                    r, g, b = v, t, p
                if ii == 1:
                    r, g, b = q, v, p
                if ii == 2:
                    r, g, b = p, v, t
                if ii == 3:
                    r, g, b = p, q, v
                if ii == 4:
                    r, g, b = t, p, v
                if ii == 5:
                    r, g, b = v, p, q
                # hsv = rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                # h = hsv[0]
                # s = hsv[1]
                # v = hsv[2]
                # h = h + shift_
                # rgb = hsv_to_rgb(h, s, v)
                # r = rgb[0]
                # g = rgb[1]
                # b = rgb[2]

                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3] = int(r*255.0), int(g*255.0), int(b*255.0), alpha_array[i, j]

    return pygame.image.frombuffer(new_array, (width, height ), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_mask_c(surface_, float shift_, mask_array):
    """
    Rotate hue value of a 24, 32-bit pygame Surface (with or without per-pixel info).
    You can pass a mask as argument in order to choose the pixels that will be 
    affected by the hue rotation. If mask_array is None, a default mask will 
    be created (same size than the texture) filled with 255 (255 values allow the hue rotation
    to the pixel while zero will copy the pixel to the surface without changing it). 
    The mask should be either a black and white surface converted into a 2d or 3d array.
    A greyscale surface converted into similar shapes is also allowed.
    C buffer are not supported in this version. 
    
    image must be encoded with per-pixel transparency or converted with method
    convert_alpha() otherwise an error message will be thrown.
    
    e.g:
        arr = pygame.surfarray.pixels3d(image)
        hue_mask(image, i / 360, mask_array=arr)
        if i > 360:
            i=0
    
    :param surface_: pygame.Surface 24-32 bit format with per-pixel transparency or surface converted 
                     with pygame method convert_alpha().
    :param mask_array: numpy.ndarray type (w, h) numpy.uint8 (black and white surface or greyscale). 
                       Black = no changes, white = changes
    :param shift_: float, amount of hue rotation range [0.0 ... 1.0] 
    :return: Returns a pygame surface 24-bit or 32-bit (if the source image contains per-pixels information)
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(mask_array, (numpy.ndarray, type(None))), \
           'Expecting numpy.ndarray for argument mask_array got %s ' % type(mask_array)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert 0.0 <= shift_ <= 1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    # Load the surface into a 3d array RGB values (uint8)
    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    cdef int width, height
    width, height = surface_.get_size()

    # load the per-pixel alpha transparency
    # alpha_ is None if per-pixel alpha cannot be extracted
    try:
        alpha_ = pixels_alpha(surface_)
        
    except (pygame.error, ValueError):
        print('\nUnsupported colormasks for alpha reference array.')
        alpha_ = None

    # No mask passed as argument,
    # Default 2d mask is created shapes (width, height) filled with 255 values (uint8)
    if mask_array is None:
        mask_array = numpy.full((width, height), 255, dtype=numpy.uint8)
        # print("\nDefault mask all values set to int 255 (width=%s, height=%s) " % (width, height))

    # check if the mask has the same dimensions than the texture
    cdef int w2, h2, dim
    try:

        w2, h2, dim = mask_array.shape[:3]
    except (ValueError, pygame.error) as e:
        try:
             w2, h2 = mask_array.shape[:2]
             dim = 0
        except (ValueError, pygame.error) as e:
            print('\nThis version does not support buffer for argument mask_array')
            print('\nError: %s ' % e)
            raise SystemExit

    if w2!=width and h2!=height:
            raise ValueError("\nMask and surface dimensions are not identical" \
                             " mask(w:%s, h:%s), surface (w:%s, h%s) " % (w2, h2, width, height))

    # convert 3d array (width, height, 3) in 2d array instead (width, height)
    cdef:
        unsigned char [:, ::1] new_mask = numpy.zeros((w2, h2), dtype=numpy.uint8)
        int i=0, j=0
    if dim == 3:
        for i in range(w2):
            for j in range(h2):
                new_mask[i, j] = mask_array[i, j, 0]
        mask_array = new_mask

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :] alpha_array = alpha_
        # RGBA array with per-pixel shape (height, width, 4)
        unsigned char [:, :, ::1] new_array4 = empty((height, width, 4), dtype=uint8)
        # RGB array without per-pixel alpha
        unsigned char [:, :, ::1] new_array3 = empty((height, width, 3), dtype=uint8)                     
        unsigned char [:, :] mask = mask_array
        double r, g, b
        double h, s, v
        float rr, gg, bb, mx, mn
        float df, df_
        float f, p, q, t, ii

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                if mask[i, j] > 0:
                    rr = r * ONE_255
                    gg = g * ONE_255
                    bb = b * ONE_255
                    mx = max(rr, gg, bb)
                    mn = min(rr, gg, bb)
                    df = mx-mn
                    df_ = 1.0/df
                    if mx == mn:
                        h = 0
                    elif mx == rr:
                        h = (60 * ((gg-bb) * df_) + 360) % 360
                    elif mx == gg:
                        h = (60 * ((bb-rr) * df_) + 120) % 360
                    elif mx == bb:
                        h = (60 * ((rr-gg) * df_) + 240) % 360
                    if mx == 0:
                        s = 0
                    else:
                        s = df/mx
                    v = mx

                    h = h * ONE_360 + shift_

                    if s == 0.0:
                        r, g, b = v, v, v
                    ii = <int>(h * 6.0)
                    f = (h * 6.0) - ii
                    p = v*(1.0 - s)
                    q = v*(1.0 - s * f)
                    t = v*(1.0 - s * (1.0 - f))
                    ii = ii % 6

                    if ii == 0:
                        r, g, b = v, t, p
                    if ii == 1:
                        r, g, b = q, v, p
                    if ii == 2:
                        r, g, b = p, v, t
                    if ii == 3:
                        r, g, b = p, q, v
                    if ii == 4:
                        r, g, b = t, p, v
                    if ii == 5:
                        r, g, b = v, p, q

                    if alpha_ is None:
                        new_array3[j, i, 0] = <unsigned char>(r * mask[i, j])
                        new_array3[j, i, 1] = <unsigned char>(g * mask[i, j])
                        new_array3[j, i, 2] = <unsigned char>(b * mask[i, j])
                    else:
                        new_array4[j, i, 0] = <unsigned char>(r * mask[i, j])
                        new_array4[j, i, 1] = <unsigned char>(g * mask[i, j])
                        new_array4[j, i, 2] = <unsigned char>(b * mask[i, j])
                        new_array4[j, i, 3] = alpha_array[i, j]
                # mask value is 0 (no change)
                else:
                    if alpha_ is None:
                        new_array3[j, i, 0], new_array3[j, i, 1], \
                        new_array3[j, i, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b
                    else:
                        new_array4[j, i, 0], new_array4[j, i, 1], \
                        new_array4[j, i, 2], new_array4[j, i, 3] = \
                            <unsigned char>r, <unsigned char>g, <unsigned char>b, alpha_array[i, j]
                                             
    if alpha_ is None:
        return pygame.image.frombuffer(new_array3, (width, height), 'RGB')
    else:
        return pygame.image.frombuffer(new_array4, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_surface_24_color_c(surface_, float shift_,
                            unsigned char red, unsigned char green, unsigned char blue):
    """
    Rotate hue for a specific color on a pygame surface (compatible 8, 24-32 bit format)
    The output surface is 24-bit without per-pixel info.

    :param surface_: pygame.Surface
    :param shift_: float; rotate hue range [0.0 ... 1.0]
    :param red: int; red value
    :param green: int; green value
    :param blue: int; blue value
    :returns: Return a 24-bit pygame.Surface without per-pixel info
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0005
        2 PASS MODE: | 32-bit  | convert()        |   0.0005
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0005
        
        4 PASS MODE: | 24-bit  |                  |   0.0005
        5 PASS MODE: | 24-bit  | convert()        |   0.0005
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0005                                                               
        
        7 PASS MODE: | 8-bit   |                  |   0.0010 * SLOW  
        8 PASS MODE: | 8-bit   | convert()        |   0.0005
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0005

    """
    

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)    
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)   
    assert 0.0 <= shift_ <= 1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'
    assert isinstance(red, int), \
           'Expecting int for argument red got %s ' % type(red)
    assert isinstance(green, int), \
           'Expecting int for argument green got %s ' % type(green)
    assert isinstance(blue, int), \
           'Expecting int for argument blue got %s ' % type(blue)

    # Load the surface into a 3d array RGB values (uint8)
    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')

    cdef int width, height, i, j
    width, height = surface_.get_size()

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        # RGB array without per-pixel alpha
        unsigned char [:, :, ::1] out = empty((height, width, 3), dtype=uint8)
        double r, g, b
        double h, s, v
        float rr, gg, bb, mx, mn
        float df, df_
        float f, p, q, t, ii

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                # apply hue only for a specific color
                if r == red and g == green and b == blue:
                    rr = r * ONE_255
                    gg = g * ONE_255
                    bb = b * ONE_255
                    mx = max(rr, gg, bb)
                    mn = min(rr, gg, bb)
                    df = mx-mn
                    df_ = 1.0/df
                    if mx == mn:
                        h = 0
                    elif mx == rr:
                        h = (60 * ((gg-bb) * df_) + 360) % 360
                    elif mx == gg:
                        h = (60 * ((bb-rr) * df_) + 120) % 360
                    elif mx == bb:
                        h = (60 * ((rr-gg) * df_) + 240) % 360
                    if mx == 0:
                        s = 0
                    else:
                        s = df/mx
                    v = mx

                    h = h * ONE_360 + shift_

                    if s == 0.0:
                        r, g, b = v, v, v
                    ii = <int>(h * 6.0)
                    f = (h * 6.0) - ii
                    p = v*(1.0 - s)
                    q = v*(1.0 - s * f)
                    t = v*(1.0 - s * (1.0 - f))
                    ii = ii % 6

                    if ii == 0:
                        r, g, b = v, t, p
                    if ii == 1:
                        r, g, b = q, v, p
                    if ii == 2:
                        r, g, b = p, v, t
                    if ii == 3:
                        r, g, b = p, q, v
                    if ii == 4:
                        r, g, b = t, p, v
                    if ii == 5:
                        r, g, b = v, p, q
                    out[j, i, 0] = <unsigned char>(r * 255)
                    out[j, i, 1] = <unsigned char>(g * 255)
                    out[j, i, 2] = <unsigned char>(b * 255)
                else:
                    out[j, i, 0] = <unsigned char>r
                    out[j, i, 1] = <unsigned char>g
                    out[j, i, 2] = <unsigned char>b
    return pygame.image.frombuffer(out, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float * rotate_hue(float shift_, float r, float g, float b)nogil:

    cdef:
        float h, s, v
        float rr, gg, bb, mx, mn
        float df, df_
        float f, p, q, t, ii
        float *rgb = <float *> malloc(3 * sizeof(float))

    rr = r * ONE_255
    gg = g * ONE_255
    bb = b * ONE_255
    mx = fmax_rgb_value(rr, gg, bb)
    mn = fmin_rgb_value(rr, gg, bb)
    df = mx-mn
    df_ = 1.0/df
    if mx == mn:
        h = 0
    elif mx == rr:
        h = (60 * ((gg-bb) * df_) + 360) % 360
    elif mx == gg:
        h = (60 * ((bb-rr) * df_) + 120) % 360
    elif mx == bb:
        h = (60 * ((rr-gg) * df_) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    h = (h * ONE_360) + shift_

    if s == 0.0:
        r, g, b = v, v, v
    ii = <int>(h * 6.0)
    f = (h * 6.0) - ii
    p = v*(1.0 - s)
    q = v*(1.0 - s * f)
    t = v*(1.0 - s * (1.0 - f))
    ii = ii % 6

    if ii == 0:
        rr, gg, bb = v, t, p
    if ii == 1:
        rr, gg, bb = q, v, p
    if ii == 2:
        rr, gg, bb = p, v, t
    if ii == 3:
        rr, gg, bb = p, q, v
    if ii == 4:
        rr, gg, bb = t, p, v
    if ii == 5:
        rr, gg, bb = v, p, q
    rgb[0] = rr
    rgb[1] = gg
    rgb[2] = bb
    free(rgb)
    return rgb

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hsah_c(surface_: Surface, int threshold_, double shift_):
    """
    Hue Surface Average high (HSAH)
    Rotate hue for pixels with average sum value >= threshold.
    e.g :
        threshold = 128
        RGB = 10, 20, 30 
        AVG = (10 + 20 + 30) / 3  = 20 
        AVG is not >= threshold (pixel value will not change)  
    
    :param surface_: Surface 8, 24-32 bit format
    :param threshold_: integer, threshold value in range [0 .. 255]
    :param shift_: pygame float,  hue rotation value in range [0.0 ... 1.0]
    :return: return a Surface with no alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.006
        2 PASS MODE: | 32-bit  | convert()        |   0.005
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.005
        
        4 PASS MODE: | 24-bit  |                  |   0.005
        5 PASS MODE: | 24-bit  | convert()        |   0.005
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.005                                                               
        
        7 PASS MODE: | 8-bit   |                  |   0.012 * SLOW  
        8 PASS MODE: | 8-bit   | convert()        |   0.005
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.005
        
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)
    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be in range[0.0 .. 1.0]'
    assert 0<= threshold_ <=255, 'Positional argument treshhold_ should be in range[0 ... 255]'
    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except (pygame.error, ValueError):
            raise ValueError('\nInvalid pixel format.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b, s
        float *rgb = <float *> malloc(3 * sizeof(float))

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                s = <unsigned char>((r + g + b) * ONE_THIRD)
                if s>=threshold_:
                    rgb = rotate_hue(shift_, r, g, b)
                    r = <unsigned char>(min(rgb[0] * 255.0, 255.0))
                    g = <unsigned char>(min(rgb[1] * 255.0, 255.0))
                    b = <unsigned char>(min(rgb[2] * 255.0, 255.0))
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2] = r, g, b
    # free(rgb)
    return pygame.image.frombuffer(new_array, (width, height), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hsal_c(surface_: Surface, int threshold_, double shift_):
    """
    Hue Surface Average Low (HSAL)
   
    Rotate hue for pixels with average sum value <= threshold.
    e.g :
        threshold = 128
        RGB = 10, 20, 30 
        AVG = (10 + 20 + 30) / 3  = 20 
        AVG is not <= threshold (pixel value will rotate hue)  
    
    :param surface_: Surface 8, 24-32 bit format
    :param threshold_: integer, threshold value in range [0 .. 255]
    :param shift_: pygame float,  hue rotation value in range [0.0 ... 1.0]
    :return: return a Surface with no alpha channel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.006
        2 PASS MODE: | 32-bit  | convert()        |   0.005
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.005
        
        4 PASS MODE: | 24-bit  |                  |   0.005
        5 PASS MODE: | 24-bit  | convert()        |   0.005
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.005                                                               
        
        7 PASS MODE: | 8-bit   |                  |   0.012 * SLOW  
        8 PASS MODE: | 8-bit   | convert()        |   0.005
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.005
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)
    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be in range[0.0 .. 1.0]'
    assert 0<= threshold_ <=255, 'Positional argument treshhold_ should be in range[0 ... 255]'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except (pygame.error, ValueError):
            raise ValueError('\nInvalid pixel format.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b, s
        float *rgb = <float *> malloc(3 * sizeof(float))
    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                s = <unsigned char>((r + g + b) * ONE_THIRD)
                if s<=threshold_:
                    rgb = rotate_hue(shift_, r, g, b)
                    r = <unsigned char>(min(rgb[0] * 255.0, 255.0))
                    g = <unsigned char>(min(rgb[1] * 255.0, 255.0))
                    b = <unsigned char>(min(rgb[2] * 255.0, 255.0))
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2] = r, g, b
    # free(rgb)
    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_array_red_c(array_, double shift_):
    """
    Rotate hue (red channel only).
    
    :param array_: numpy.ndarray (w, h, 3) uint8 colours values. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: numpy.ndarray (w, h, 3) uint8 with red colours shifted 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 PASS MODE: | 24-bit  |                  |   0.008
        5 PASS MODE: | 24-bit  | convert()        |   0.008
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008                                                              
        
        7 PASS MODE: | 8-bit   |                  |   0.008
        8 PASS MODE: | 8-bit   | convert()        |   0.008
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008
    
    """
    assert isinstance(array_, numpy.ndarray), \
           'Expecting numpy.ndarray for argument array_ got %s ' % type(array_)
    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)
    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    cdef int width, height
    try:
        width, height = array_.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] new_array = empty((width, height, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b
        float *rgb = <float *> malloc(3 * sizeof(float))
        float rr, gg, bb

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb = rotate_hue(shift_, r, g, b)
                new_array[i, j, 0], new_array[i, j, 1], \
                new_array[i, j, 2] = min(<unsigned char>(rgb[0]*255.0), 255), g, b
    # free(rgb)
    return asarray(new_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_red24_b(surface_, double shift_):
    """
    Rotate hue (red channel only).
    
    :param surface_: pygame.Surface compatible 8, 24-32 bit format pixel
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: 32-bit pygame.Surface with per-pixel transparency, alpha channel is set to maximum
    opacity 255
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008 BGR FORMAT
        2 PASS MODE: | 32-bit  | convert()        |   0.008
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.008
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008                                                              
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 PASS MODE: | 8-bit   | convert()        |   0.008
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008
    """
    assert isinstance(surface_, pygame.Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
                                                     
    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)
                                                     
    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    cdef int w, h
    w, h = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Invalid pixel format.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, dtype=numpy.uint8)
        int i=0
        unsigned char r, g, b, rr
        float *rgb = <float *> malloc(3 * sizeof(float))

    with nogil:

        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            r, g, b = cbuffer[i], cbuffer[i + 1], cbuffer[i + 2]
            rgb = rotate_hue(shift_, r, g, b)
            rr = <unsigned char>(rgb[0] * 255.0)
            if rr > 255:
                rr = 255
            new_buffer[i], new_buffer[i + 1], new_buffer[i + 2], new_buffer[i + 3] = b, g, rr, 255
    # free(rgb)
    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_red32_b(surface_, double shift_):
    """
    Rotate hue (red channel only).
    
    :param surface_: pygame.Surface compatible with 32 bit format pixel with per-pixel transparency. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: pygame.Surface 32 bit with per-pixel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008 DO NOT USE, alpha = 0
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 FAIL MODE: | 24-bit  |                  |   Compatible with 32 bit format pixel only.
        5 FAIL MODE: | 24-bit  | convert()        |   Compatible with 32 bit format pixel only.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008 Alpha = 255, image contains per-pixel transparency 
                                                      therefore all value are set to maximum opacity 255                                                           
        
        7 FAIL MODE: | 8-bit   |                  |   Compatible with 32 bit format pixel only.
        8 FAIL MODE: | 8-bit   | convert()        |   Compatible with 32 bit format pixel only.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008 Alpha = 255, image contains per-pixel transparency 
                                                      therefore all value are set to maximum opacity 255
    """
    assert isinstance(surface_, pygame.Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)

    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    assert surface_.get_bitsize() == 32 and surface_.get_flags() & pygame.SRCALPHA,\
        "Compatible with 32 bit format pixel only."

    cdef int w, h
    w, h = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Invalid pixel format.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, dtype=numpy.uint8)
        int i=0
        unsigned char r, g, b, rr
        float *rgb = <float *> malloc(3 * sizeof(float))

    with nogil:

        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            r, g, b = cbuffer[i], cbuffer[i + 1], cbuffer[i + 2]
            rgb = rotate_hue(shift_, r, g, b)
            rr = <unsigned char>(rgb[0] * 255.0)
            if rr > 255:
                rr =255
            new_buffer[i], new_buffer[i + 1], new_buffer[i + 2] = b, g,  rr
    # free(rgb)
    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_green24_b(surface_, double shift_):
    """
    Rotate hue (green channel only).
    
    :param surface_: pygame.Surface compatible 8, 24-32 bit format pixel
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: 32-bit pygame.Surface with per-pixel transparency, alpha channel is set to maximum
    opacity 255
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.008
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008                                                              
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 PASS MODE: | 8-bit   | convert()        |   0.008
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008
    """
    assert isinstance(surface_, pygame.Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)

    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    cdef int w, h
    w, h = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Invalid pixel format.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, dtype=numpy.uint8)
        int i=0
        unsigned char r, g, b, gg
        float *rgb = <float *> malloc(3 * sizeof(float))

    with nogil:

        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            r, g, b = cbuffer[i], cbuffer[i + 1], cbuffer[i + 2]
            rgb = rotate_hue(shift_, r, g, b)
            gg = <unsigned char>(rgb[1] * 255.0)
            if gg > 255:
                gg =255
            new_buffer[i], new_buffer[i + 1], new_buffer[i + 2], new_buffer[i + 3] = b, gg, r, 255
    # free(rgb)
    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_green32_b(surface_, double shift_):
    """
    Rotate hue (green channel only).
    
    :param surface_: pygame.Surface compatible with 32 bit format pixel with per-pixel transparency. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: pygame.Surface 32 bit with per-pixel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008 DO NOT USE, alpha = 0
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 FAIL MODE: | 24-bit  |                  |   Compatible with 32 bit format pixel only.
        5 FAIL MODE: | 24-bit  | convert()        |   Compatible with 32 bit format pixel only.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008 Alpha = 255, image contains per-pixel transparency 
                                                      therefore all value are set to maximum opacity 255                                                           
        
        7 FAIL MODE: | 8-bit   |                  |   Compatible with 32 bit format pixel only.
        8 FAIL MODE: | 8-bit   | convert()        |   Compatible with 32 bit format pixel only.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008 Alpha = 255, image contains per-pixel transparency 
                                                      therefore all value are set to maximum opacity 255
    """
    assert isinstance(surface_, pygame.Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)

    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    assert surface_.get_bitsize() == 32 and surface_.get_flags() & pygame.SRCALPHA,\
        "Compatible with 32 bit format pixel only."

    cdef int w, h
    w, h = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Invalid pixel format.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, dtype=numpy.uint8)
        int i=0
        unsigned char r, g, b, gg
        float *rgb = <float *> malloc(3 * sizeof(float))

    with nogil:

        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            r, g, b = cbuffer[i], cbuffer[i + 1], cbuffer[i + 2]
            rgb = rotate_hue(shift_, r, g, b)
            gg = <unsigned char>(rgb[0] * 255.0)
            if gg > 255:
                gg =255
            new_buffer[i], new_buffer[i + 1], new_buffer[i + 2] = b, gg, r
    # free(rgb)
    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_blue24_b(surface_, double shift_):
    """
    Rotate hue (blue channel only).
    
    :param surface_: pygame.Surface compatible 8, 24-32 bit format pixel
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: 32-bit pygame.Surface with per-pixel transparency, alpha channel is set to maximum
    opacity 255
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 FAIL MODE: | 24-bit  |                  |   Buffer length does not equal format and resolution size
        5 PASS MODE: | 24-bit  | convert()        |   0.008
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008                                                              
        
        7 FAIL MODE: | 8-bit   |                  |   Buffer length does not equal format and resolution size
        8 PASS MODE: | 8-bit   | convert()        |   0.008
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008
    """
    assert isinstance(surface_, pygame.Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)

    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    cdef int w, h
    w, h = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Invalid pixel format.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, dtype=numpy.uint8)
        int i=0
        unsigned char r, g, b, bb
        float *rgb = <float *> malloc(3 * sizeof(float))

    with nogil:

        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            r, g, b = cbuffer[i], cbuffer[i + 1], cbuffer[i + 2]
            rgb = rotate_hue(shift_, r, g, b)
            bb = <unsigned char>(rgb[2] * 255.0)
            if bb > 255:
                bb =255
            new_buffer[i], new_buffer[i + 1], new_buffer[i + 2], new_buffer[i + 3] = bb, g, r, 255
    # free(rgb)
    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_blue32_b(surface_, double shift_):
    """
    Rotate hue (blue channel only).
    
    :param surface_: pygame.Surface compatible with 32 bit format pixel with per-pixel transparency. 
    :param shift_: pygame float, hue rotation in range [0.0 ... 1.0]
    :return: pygame.Surface 32 bit with per-pixel
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008 DO NOT USE, alpha = 0
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 FAIL MODE: | 24-bit  |                  |   Compatible with 32 bit format pixel only.
        5 FAIL MODE: | 24-bit  | convert()        |   Compatible with 32 bit format pixel only.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008 Alpha = 255, image contains per-pixel transparency 
                                                      therefore all value are set to maximum opacity 255                                                           
        
        7 FAIL MODE: | 8-bit   |                  |   Compatible with 32 bit format pixel only.
        8 FAIL MODE: | 8-bit   | convert()        |   Compatible with 32 bit format pixel only.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008 Alpha = 255, image contains per-pixel transparency 
                                                      therefore all value are set to maximum opacity 255
    """
    assert isinstance(surface_, pygame.Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)

    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    assert surface_.get_bitsize() == 32 and surface_.get_flags() & pygame.SRCALPHA,\
        "Compatible with 32 bit format pixel only."

    cdef int w, h
    w, h = surface_.get_size()

    try:
        buffer_ = surface_.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Invalid pixel format.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] new_buffer = numpy.zeros(b_length, dtype=numpy.uint8)
        int i=0
        unsigned char r, g, b, bb
        float *rgb = <float *> malloc(3 * sizeof(float))

    with nogil:

        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            r, g, b = cbuffer[i], cbuffer[i + 1], cbuffer[i + 2]
            rgb = rotate_hue(shift_, r, g, b)
            bb = <unsigned char>(rgb[2] * 255.0)
            if bb > 255:
                bb =255
            new_buffer[i], new_buffer[i + 1], new_buffer[i + 2] = bb, g, r
    # free(rgb)
    return pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_array_green_c(array_, double shift_):
    """
    Rotate hue (green channel only)
    
    :param array_: numpy.ndarray (w, h, 3) uint8 colours values. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: numpy.ndarray (w, h, 3) uint8 with green colours shifted 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 PASS MODE: | 24-bit  |                  |   0.008
        5 PASS MODE: | 24-bit  | convert()        |   0.008
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008                                                              
        
        7 PASS MODE: | 8-bit   |                  |   0.008
        8 PASS MODE: | 8-bit   | convert()        |   0.008
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008
        
    """
    assert isinstance(array_, numpy.ndarray), \
           'Expecting numpy.ndarray for argument array_ got %s ' % type(array_)
    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)
    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    cdef int width, height
    width, height = array_.shape[:2]

    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] new_array = empty((width, height, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b
        float *rgb = <float *> malloc(3 * sizeof(float))
        float rr, gg, bb

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                rgb = rotate_hue(shift_, r, g, b)
                gg = rgb[1]*255.0
                new_array[i, j, 0], new_array[i, j, 1], \
                new_array[i, j, 2] = r, min(<unsigned char>gg, 255), b
    # free(rgb)
    return asarray(new_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_array_blue_c(array_, double shift_):
    """
    Rotate hue (blue channel only)
    
    :param array_: numpy.ndarray (w, h, 3) uint8 colours values. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: numpy.ndarray (w, h, 3) uint8 with blue colours shifted 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 PASS MODE: | 24-bit  |                  |   0.008
        5 PASS MODE: | 24-bit  | convert()        |   0.008
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008                                                              
        
        7 PASS MODE: | 8-bit   |                  |   0.008
        8 PASS MODE: | 8-bit   | convert()        |   0.008
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008
    """
    assert isinstance(array_, numpy.ndarray), \
           'Expecting numpy.ndarray for argument array_ got %s ' % type(array_)
    assert isinstance(shift_, float), \
            'Expecting double for argument shift_, got %s ' % type(shift_)
    assert 0.0<= shift_ <=1.0, 'Positional argument shift_ should be between[0.0 .. 1.0]'

    cdef int width, height
    try:
        width, height = array_.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] new_array = empty((width, height, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b
        float *rgb = <float *> malloc(3 * sizeof(float))
        float bb

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb = rotate_hue(shift_, r, g, b)
                bb =  rgb[2]*255.0
                if bb > 255.0:
                    bb =255.0
                new_array[i, j, 0] = r
                new_array[i, j, 1] = g
                new_array[i, j, 2] = <unsigned char>bb
    # free(rgb)
    return asarray(new_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef brightness_24c(surface_:Surface, float shift_):
    """
    Change the brightness level of a pygame.Surface (compatible 24-32 bit).
    Transform the RGB color model into HLS model and add the <shift_> value to the lightness. 
    :param surface_: pygame.Surface (24 - 32 bit format) 
    :param shift_: Value must be in range [-1.0 ... 1.0]
    :return: a pygame.Surface 24 bit without alpha channel 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.008
        2 PASS MODE: | 32-bit  | convert()        |   0.008
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.008
        
        4 PASS MODE: | 24-bit  |                  |   0.008
        5 PASS MODE: | 24-bit  | convert()        |   0.008
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.008                                                              
        
        7 PASS MODE: | 8-bit   |                  |   0.008
        8 PASS MODE: | 8-bit   | convert()        |   0.008
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.008
    """


    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except (pygame.error, ValueError):
            raise ValueError('\nInvalid pixel format.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        double r, g, b
        double h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]
        float maxc, minc, rc, gc, bc
        float high, low, high_

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                hls = rgb_to_hls(r * ONE_255, g * ONE_255, b * ONE_255)
                h = hls[0]
                l = hls[1]
                s = hls[2]
                l = min((l + shift_), 1.0)
                rgb = hls_to_rgb(h, l, s)
                r = min(rgb[0] * 255.0, 255.0)
                g = min(rgb[1] * 255.0, 255.0)
                b = min(rgb[2] * 255.0, 255.0)
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef brightness_32c(surface_:Surface, float shift_):
    """
    Change the brightness level of a pygame.Surface (compatible 32 bit format image)
    Transform the RGB color model into HLS model and add the <shift_> value to the lightness.
    :param surface_: pygame.Surface 32 bit format with per-pixel alpha transparency 
    :param shift_: Value must be in range [-1.0 ... 1.0]
    :return: a pygame.Surface 32 bit with per-pixel information 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.023
        2 FAIL MODE: | 32-bit  | convert()        |   Invalid pixel format.
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.025
        
        4 FAIL MODE: | 24-bit  |                  |   Invalid pixel format.
        5 FAIL MODE: | 24-bit  | convert()        |   Invalid pixel format.
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.031 FULL OPACITY                                                           
        
        7 FAIL MODE: | 8-bit   |                  |   Invalid pixel format.
        8 FAIL MODE: | 8-bit   | convert()        |   Invalid pixel format.
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.019 FULL OPACITY
        
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            raise ValueError('\nInvalid pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, :] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char [:, :] alpha_array = alpha_
        int i=0, j=0
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                hls = rgb_to_hls(r * ONE_255, g * ONE_255, b * ONE_255)
                h = hls[0]
                l = hls[1]
                s = hls[2]
                l = min((l + shift_), 1.0)
                rgb = hls_to_rgb(h, l, s)
                r = min(rgb[0] * 255.0, 255.0)
                g = min(rgb[1] * 255.0, 255.0)
                b = min(rgb[2] * 255.0, 255.0)
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3] = <int>r, <int>g, <int>b,alpha_array[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef brightness_24_fast(surface_:Surface, float shift_):
    """  
    Change the brightness level of a pygame.Surface (compatible with 8, 24-32 bit format image)
    Change the lightness of an image by decreasing/increasing the sum of RGB values.
    
    :param surface_: pygame.Surface 8, 24-32 bit format 
    :param shift_: Value must be in range [-1.0 ... 1.0]
    :return: a pygame.Surface 24 bit without per-pixel transparency 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.00140
        2 PASS MODE: | 32-bit  | convert()        |   0.00123
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.00162
        
        4 PASS MODE: | 24-bit  |                  |   0.00142
        5 PASS MODE: | 24-bit  | convert()        |   0.00136
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.00131                                                           
        
        7 PASS MODE: | 8-bit   |                  |   0.00900 SLOW
        8 PASS MODE: | 8-bit   | convert()        |   0.00122
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.00122
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
         try:
            rgb_ = array3d(surface_)
         except (pygame.error, ValueError):
            raise ValueError('\nInvalid pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        float r, g, b

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                r = min(r + shift_, 255.0)
                g = min(g + shift_, 255.0)
                b = min(b + shift_, 255.0)
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2] = <int>r, <int>g, <int>b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef brightness_32_fast(surface_:Surface, float shift_):
    """
    Change the brightness level of a pygame.Surface (compatible with 32-bit format image)
    Change the lightness of an image by decreasing/increasing the sum of RGB values.
    
    :param surface_: pygame.Surface 32 bit format 
    :param shift_: Value must be in range [-1.0 ... 1.0]
    :return: a pygame.Surface 32 bit with per-pixel information 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   0.0061
        2 FAIL MODE: | 32-bit  | convert()        |   Invalid pixel format
        3 PASS MODE: | 32-bit  | convert_alpha()  |   0.0031
        
        4 FAIL MODE: | 24-bit  |                  |   Invalid pixel format
        5 FAIL MODE: | 24-bit  | convert()        |   Invalid pixel format
        6 PASS MODE: | 24-bit  | convert_alpha()  |   0.0311                                                           
        
        7 FAIL MODE: | 8-bit   |                  |   Invalid pixel format
        8 FAIL MODE: | 8-bit   | convert()        |   Invalid pixel format
        9 PASS MODE: | 8-bit   | convert_alpha()  |   0.0032
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            raise ValueError('\nInvalid pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = zeros((height, width, 4), dtype=uint8)
        unsigned char [:, :] alpha_array = alpha_
        int i=0, j=0
        float r, g, b

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                r = min(r + shift_, 255.0)
                g = min(g + shift_, 255.0)
                b = min(b + shift_, 255.0)
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                new_array[j, i, 0], new_array[j, i, 1],\
                new_array[j, i, 2], new_array[j, i, 3] = <int>r, <int>g, <int>b, alpha_array[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef brightness_mask_32c(surface_:Surface, float shift_, mask_array):
    """
    Change the brightness level of a pygame.Surface (compatible 32 bit format image)
    Transform the RGB color model into HLS model and add the <shift_> value to the lightness.
    :param surface_: pygame.Surface 32 bit format with per-pixel alpha transparency 
    :param shift_: Value must be in range [-1.0 ... 1.0].
                   -1.0 to 0.0 decrease lightness, 0.0 to 1.0 increase lightness
    :param mask_array: numpy.ndarray, 2d or 3d array (numpy.uint8). 
                       Support 3d greyscale array (width, height, 3) uint8   
    :return: a pygame.Surface 32 bit with per-pixel information 
    
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   
        2 FAIL MODE: | 32-bit  | convert()        |   
        3 PASS MODE: | 32-bit  | convert_alpha()  |   
        
        4 FAIL MODE: | 24-bit  |                  |   
        5 FAIL MODE: | 24-bit  | convert()        |   
        6 PASS MODE: | 24-bit  | convert_alpha()  |                                                         
        
        7 FAIL MODE: | 8-bit   |                  |   
        8 FAIL MODE: | 8-bit   | convert()        |   
        9 PASS MODE: | 8-bit   | convert_alpha()  |   
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 ... 1.0].'

    cdef int width, height, w2, h2, dim
    width, height = surface_.get_size()

    if mask_array is not None:
        assert isinstance(mask_array, numpy.ndarray), \
               'Expecting numpy.ndarray for argument mask_array, got %s ' % type(mask_array)
        try:
            w2, h2, dim = mask_array.shape[:3]

        except (ValueError, pygame.error) as e:
            
            try:
                w2, h2 = mask_array.shape[:2]
                dim = 0
            except (ValueError, pygame.error) as e:
                raise ValueError("\nC buffer is not supported for argument mask_array.")

    else:
        # create a default mask (2d array shape width x height numpy uint8)
        mask_array = numpy.full((width, height), 255, dtype=numpy.uint8)
        w2 = width
        h2 = height
        dim = 0

    # check mask width an height
    if w2!= width or h2!=height:
        raise ValueError("\nMask does not match "
                         "texture dimensions, texture(width:%s, height%s) "
                         "vs mask(width%s, height%s)" % (width, height, w2, h2))
    cdef:
        int i=0, j=0
        unsigned char [:, ::1] new_mask = numpy.zeros((w2, h2), dtype=numpy.uint8)

    # convert a 3d mask into a 2d mask
    if dim > 0:
        for i in range(w2):
            for j in range(h2):
                new_mask[i, j] = mask_array[i, j, 0]

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
        
    except (pygame.error, ValueError):
            raise ValueError('\nThis will only work on Surfaces that have 32-bit format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, :] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char [:, :] alpha_array = alpha_
        unsigned char [:, :] mask  = new_mask if dim >0 else mask_array
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                if mask[i, j] > 0:
                    hls = rgb_to_hls(r * ONE_255, g *ONE_255, b * ONE_255)
                    h = hls[0]
                    l = hls[1]
                    s = hls[2]
                    l = min((l + shift_), 1.0)
                    rgb = hls_to_rgb(h, l, s)
                    r = min(rgb[0] * mask[i, j], 255.0)
                    g = min(rgb[1] * mask[i, j], 255.0)
                    b = min(rgb[2] * mask[i, j], 255.0)
                    if r < 0:
                        r = 0
                    if g < 0:
                        g = 0
                    if b < 0:
                        b = 0
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2], new_array[j, i, 3] = <int>r, <int>g, <int>b, alpha_array[i, j]
                else:
                    # no changes copy the pixel to the destination
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2], new_array[j, i, 3] = <int>r, <int>g, <int>b, alpha_array[i, j]
    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef brightness_mask_24c(surface_:Surface, float shift_, mask_array):
    """
    Change the brightness level of a pygame.Surface (compatible 24 bit format image)
    Transform the RGB color model into HLS model and add the <shift_> value to the lightness.
    :param surface_: pygame.Surface 32 bit format with per-pixel alpha transparency 
    :param shift_: Value must be in range [-1.0 ... 1.0].
                   -1.0 to 0.0 decrease lightness, 0.0 to 1.0 increase lightness
    :param mask_array: numpy.ndarray, 2d or 3d array (numpy.uint8). 
                       Support 3d greyscale array (width, height, 3) uint8   
    :return: a pygame.Surface 24 bit 
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 ... 1.0].'

    cdef int width, height, w2, h2, dim
    width, height = surface_.get_size()

    if mask_array is not None:
        assert isinstance(mask_array, numpy.ndarray), \
               'Expecting numpy.ndarray for argument mask_array, got %s ' % type(mask_array)
        try:
            w2, h2, dim = mask_array.shape[:3]

        except (ValueError, pygame.error) as e:
            try:
                w2, h2 = mask_array.shape[:2]
                dim = 0
            except (ValueError, pygame.error) as e:
                raise ValueError("\nC buffer is not supported for argument mask_array.")

    else:
        # create a default mask (2d array shape width x height numpy uint8)
        mask_array = numpy.full((width, height), 255, dtype=numpy.uint8)
        w2 = width
        h2 = height
        dim = 0

    # check mask width an height
    if w2!= width or h2!=height:
        raise ValueError("\nMask does not match "
                         "texture dimensions, texture(width:%s, height%s) "
                         "vs mask(width%s, height%s)" % (width, height, w2, h2))
    cdef:
        int i=0, j=0
        unsigned char [:, ::1] new_mask = numpy.zeros((w2, h2), dtype=numpy.uint8)

    # convert a 3d mask into a 2d mask
    if dim > 0:
        for i in range(w2):
            for j in range(h2):
                new_mask[i, j] = mask_array[i, j, 0]

    try:
        rgb_ = pixels3d(surface_)

    except (pygame.error, ValueError):
            raise ValueError('\nThis will only work on Surfaces that have 32-bit format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, :] new_array = empty((height, width, 3), dtype=uint8)
        unsigned char [:, :] mask  = new_mask if dim >0 else mask_array
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                if mask[i, j] > 0:
                    hls = rgb_to_hls(r * ONE_255, g *ONE_255, b * ONE_255)
                    h = hls[0]
                    l = hls[1]
                    s = hls[2]
                    l = min((l + shift_), 1.0)
                    rgb = hls_to_rgb(h, l, s)
                    r = min(rgb[0] * mask[i, j], 255.0)
                    g = min(rgb[1] * mask[i, j], 255.0)
                    b = min(rgb[2] * mask[i, j], 255.0)
                    if r < 0:
                        r = 0
                    if g < 0:
                        g = 0
                    if b < 0:
                        b = 0
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2] = <int>r, <int>g, <int>b
                else:
                    # no changes copy the pixel to the destination
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2]  = <int>r, <int>g, <int>b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_24_c(surface_:Surface, float shift_):
    """
    Change the saturation level of a pygame.Surface (compatible with 24-bit format image)
    Transform RGB model into HLS model and add <shift_> value to the saturation 
    
    :param surface_: pygame.Surface 24 bit format 
    :param shift_: Value must be in range [-1.0 ... 1.0]
    :return: a pygame.Surface 24-bit without per-pixel information 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nThis will only work on Surfaces that have 24-bit format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                hls = rgb_to_hls(r * ONE_255, g * ONE_255, b * ONE_255)
                h = hls[0]
                l = hls[1]
                s = hls[2]
                s = min((s + shift_), 0.5)
                s = max(s, 0.0)
                rgb = hls_to_rgb(h, l, s)
                r = rgb[0] * 255.0
                g = rgb[1] * 255.0
                b = rgb[2] * 255.0
                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_32_c(surface_:Surface, float shift_):
    """
    Change the saturation level of a pygame.Surface (compatible with 32-bit format image)
    Transform RGB model into HLS model and add <shift_> value to the saturation 
    
    :param surface_: pygame.Surface 32 bit format 
    :param shift_: Value must be in range [-1.0 ... 1.0]
    :return: a pygame.Surface 32-bit with per-pixel information 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nThis will only work on Surfaces that have 24-bit format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char [:, ::1] new_alpha = alpha_
        int i=0, j=0
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                hls = rgb_to_hls(r * ONE_255, g * ONE_255, b * ONE_255)
                h = hls[0]
                l = hls[1]
                s = hls[2]
                s = min((s + shift_), 1.0)
                s = max(s, 0.0)
                rgb = hls_to_rgb(h, l, s)
                r = rgb[0] * 255.0
                g = rgb[1] * 255.0
                b = rgb[2] * 255.0
                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b, new_alpha[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_mask_32_c(surface_:Surface, float shift_, mask_array):
    """
    Change the saturation level of a pygame.Surface (compatible with 24 converted
    with convert_alpha() method or 32-bit format).
    Transform RGB model into HLS model and <shift_> saturation value.
    Optional mask_array to determine area to be modified.
    The mask should be a 2d array (filled with 0 and 255 values) representing a black and white surface  
    
    :param surface_: pygame.Surface 24 converted with pygame convert_alpha() method or 32 bit pixel format 
    :param shift_: Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: numpy.ndarray (black and white mask) with shape (width, height, 3) or (width, height) 
    :return: a pygame.Surface 32-bit with per-pixel information 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height, w2, h2, dim
    width, height = surface_.get_size()

    if mask_array is not None:
        # check if mask is a numpy array
        assert isinstance(mask_array, numpy.ndarray), \
               'Expecting numpy.ndarray for argument mask_array, got %s ' % type(mask_array)
        try:
            # try array shape (width, height, 3)
            w2, h2, dim = mask_array.shape[:3]
        except (ValueError, pygame.error) as e:
            
            try:
                # try array shape (width, height)
                w2, h2 = mask_array.shape[:2]
                dim = 0
            except (ValueError, pygame.error) as e:
                # Buffer like array is not supported
                raise ValueError("\nmask_array shape not understood.")

    else:
        # No mask_array provided in the function call,
        # create a default mask (2d array shape width x height filled with 255)
        mask_array = numpy.full((width, height), 255, dtype=numpy.uint8)
        w2 = width
        h2 = height
        dim = 0

    # check mask width and height
    if w2!= width or h2!=height:
        raise ValueError("\nmask_array and texture size mismatch.")

    cdef:
        int i=0, j=0
        unsigned char [:, :] new_mask = numpy.zeros((width, height), dtype=numpy.uint8)

    # if mask is a 3d array, convert it to a 2d array
    if dim == 3:
        for i in range(width):
            for j in range(height):
                new_mask[i, j] = mask_array[i, j, 0]    
    elif dim == 0:
        # Already a 2d array 
        # copy array in case of shallow copy
        # referencing original image
        new_mask = mask_array.copy()
    else:
        # Anything else does not make any sense, dim is either 0 or 3
        raise ValueError("\nmask_array shape not understood.")

    try:
        rgb_ = pixels3d(surface_)           # RGB values 
        alpha_ = pixels_alpha(surface_)     # alpha values
        
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char [:, ::1] new_alpha = alpha_
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
                # load pixel RGB values
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                if new_mask[i, j] > 0:
                    # change saturation
                    hls = rgb_to_hls(r * ONE_255, g * ONE_255, b * ONE_255)
                    h = hls[0]
                    l = hls[1]
                    s = hls[2]
                    s = min((s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb = hls_to_rgb(h, l, s)
                    r = rgb[0] * 255.0
                    g = rgb[1] * 255.0
                    b = rgb[2] * 255.0                    
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2], new_array[j, i, 3] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b, new_alpha[i, j]
                else:
                    # no changes to the RGB values
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2], new_array[j, i, 3] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b, new_alpha[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_mask_24_c(surface_:Surface, float shift_, mask_array):
    """
    Change the saturation level of a pygame.Surface (compatible with 24 or 32-bit format).
    Transform RGB model into HLS model and <shift_> saturation value.
    Optional mask_array to determine area to be modified.
    The mask should be a 2d array (filled with 0 and 255 values) representing a black and white surface  
    
    :param surface_: pygame.Surface 24 or 32-bit pixel format 
    :param shift_: Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: numpy.ndarray (black and white mask) with shape (width, height, 3) or (width, height) 
    :return: a pygame.Surface 24-bit without per-pixel information 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height, w2, h2, dim
    width, height = surface_.get_size()

    if mask_array is not None:
        # check if mask is a numpy array
        assert isinstance(mask_array, numpy.ndarray), \
               'Expecting numpy.ndarray for argument mask_array, got %s ' % type(mask_array)
        try:
            # try array shape (width, height, 3)
            w2, h2, dim = mask_array.shape[:3]
        except (ValueError, pygame.error) as e:
            
            try:
                # try array shape (width, height)
                w2, h2 = mask_array.shape[:2]
                dim = 0
            except (ValueError, pygame.error) as e:
                # Buffer like array is not supported
                raise ValueError("\nmask_array shape not understood.")

    else:
        # No mask_array provided in the function call,
        # create a default mask (2d array shape width x height filled with 255)
        mask_array = numpy.full((width, height), 255, dtype=numpy.uint8)
        w2 = width
        h2 = height
        dim = 0

    # check mask width and height
    if w2!= width or h2!=height:
        raise ValueError("\nmask_array and texture size mismatch.")

    cdef:
        int i=0, j=0
        unsigned char [:, :] new_mask = numpy.zeros((width, height), dtype=numpy.uint8)

    # if mask is a 3d array, convert it to a 2d array
    if dim == 3:
        for i in range(width):
            for j in range(height):
                new_mask[i, j] = mask_array[i, j, 0]    
    elif dim == 0:
        # Already a 2d array 
        # copy array in case of shallow copy
        # referencing original image
        new_mask = mask_array.copy()
    else:
        # Anything else does not make any sense, dim is either 0 or 3
        raise ValueError("\nmask_array shape not understood.")

    try:
        rgb_ = pixels3d(surface_)           # RGB values      
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
        
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
                # load pixel RGB values
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                if new_mask[i, j] > 0:
                    # change saturation
                    hls = rgb_to_hls(r * ONE_255, g * ONE_255, b * ONE_255)
                    h = hls[0]
                    l = hls[1]
                    s = hls[2]
                    s = min((s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb = hls_to_rgb(h, l, s)
                    r = rgb[0] * 255.0
                    g = rgb[1] * 255.0
                    b = rgb[2] * 255.0                    
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b
                else:
                    # no changes to the RGB values
                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_array_c(array, int dy, int dx):
    """  
    Roll the value of an entire array (lateral and vertical)
    The roll effect can be apply to both direction (vertical and horizontal) at the same time
    dy scroll texture vertically (-dy scroll up, +dy scroll down)
    dx scroll texture horizontally (-dx scroll left, +dx scroll right,
    array must be a numpy.ndarray type (w, h, 3) uint8
    This method return a scrolled numpy.ndarray
    
    :param array: numpy.ndarray (w,h,3) uint8 (array to scroll)
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: a numpy.ndarray type (w, h, 3) numpy uint8
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array, a numpy.ndarray is required (got type %s)' % type(array))

    cdef int w, h, dim
    try:
        w, h, dim = (<object> array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    if w == 0 or h == 0:
        raise ValueError('Incorrect array dimension must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))
    if dim != 3:
        raise ValueError('Incompatible array, must be (w, h, 3) got (%s, %s, %s) ' % (w, h, dim))
    
    zero = zeros((w, h, 3), dtype=uint8)
    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, :] rgb_array = array
        unsigned char [:, :, ::1] empty_array = zero
        unsigned char [:, :, ::1] tmp_array = zero[:]

    if dx==0 and dy==0:
        tmp_array = array
    # print('dx=', dx, ' dy=', dy)
    if dx != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    ii = (i + dx) % w
                    if ii < 0:
                        ii = ii + w
                    # printf("\n%i %i %i", jj, w, h)
                    empty_array[ii, j, 0], empty_array[ii, j, 1], empty_array[ii, j, 2] = \
                        rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
            tmp_array = empty_array
    if dy != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    jj = (j + dy) % h
                    if jj < 0:
                        jj = jj + h
                    # printf("\n%i %i %i", jj, w, h)
                    tmp_array[i, jj, 0], tmp_array[i, jj, 1], tmp_array[i, jj, 2] = \
                        tmp_array[i, j, 0], tmp_array[i, j, 1], tmp_array[i, j, 2]
    return asarray(tmp_array)



def scroll_surface_org(array: numpy.ndarray, dx: int=0, dy: int=0)-> tuple:
    """
    Scroll pixels inside a 3d array (RGB values) and return a tuple (pygame surface, 3d array).
    Use dy to scroll up or down (move the image of dy pixels)
    Use dx to scroll left or right (move the image of dx pixels)
    :param dx: int, Use dx for scrolling right or left (move the image of dx pixels)
    :param dy: int, Use dy to scroll up or down (move the image of dy pixels)
    :param array: numpy.ndarray such as pixels3d(texture).
    This will only work on Surfaces that have 24-bit or 32-bit formats.
    Lower pixel formats cannot be referenced using this method.
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array, a numpy.ndarray is required (got type %s)' % type(array))
    if dx != 0:
        array = numpy.roll(array, dx, axis=1)
    if dy != 0:
        array = numpy.roll(array, dy, axis=0)
    return pygame.surfarray.make_surface(array), array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_surface_c(surface, int dy, int dx):
    """
    
    :param surface: pygame Surface 24, 32-bit format compatible.
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: Return a tuple (surface:Surface, array:numpy.ndarray) type (w, h, 3) numpy uint8
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(surface, pygame.Surface):
        raise TypeError('surface, a pygame.Surface is required (got type %s)' % type(surface))

    cdef int w, h, dim

    try:
        array = pixels3d(surface)
        alpha = pixels_alpha(surface)
    except (ValueError, pygame.error) as e:
        raise ValueError('\nIncompatible pixel format.')

    try:
        w, h, dim = (<object> array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    if w == 0 or h == 0:
        raise ValueError('Incorrect array dimension must be (w>0, h>0) got(w:%s, h:%s) ' % (w, h))
    if dim != 3:
        raise ValueError('Incompatible array, must be (w, h, 3) got (%s, %s %s) ' % (w, h, dim))

    zero = zeros((w, h, 3), dtype=uint8)
    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, :] rgb_array = array
        unsigned char [:, :, ::1] empty_array = zero
        unsigned char [:, :, ::1] tmp_array = zero[:]

    if dx==0 and dy==0:
        tmp_array = array
    if dx != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    ii = (i + dx) % w
                    if ii < 0:
                        ii = ii + w
                    empty_array[ii, j, 0], empty_array[ii, j, 1], empty_array[ii, j, 2] = \
                        rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
            tmp_array = empty_array
    if dy != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    jj = (j + dy) % h
                    if jj < 0:
                        jj = jj + h
                    tmp_array[i, jj, 0], tmp_array[i, jj, 1], tmp_array[i, jj, 2] = \
                        tmp_array[i, j, 0], tmp_array[i, j, 1], tmp_array[i, j, 2]

    return pygame.image.frombuffer(tmp_array, (w, h), 'RGB'), asarray(tmp_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_surface_alpha_c(surface, int dy, int dx):
    """
    Scroll surface channel alpha (lateral/vertical using optional dx, dy values)
    
    :param surface: 32 bit pygame surface only or 24-bit surface converted
    with convert_alpha() pygame method. An error will be thrown if the surface does not contains
    per-pixel transparency. 
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: Return a tuple (surface: 32 bit Surface with per-pixel info, array:3d array numpy.ndarray shape (w, h, 4)) 
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(surface, pygame.Surface):
        raise TypeError('surface, a pygame surface is required (got type %s)' % type(surface))

    cdef int w, h, dim

    try:
        array = pixels3d(surface)
        alpha = pixels_alpha(surface)
        
    except (ValueError, pygame.error) as e:
        raise ValueError('\nIncompatible pixel format.')

    try:
        w, h, dim = array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    if w == 0 or h == 0:
        raise ValueError('Incorrect array dimension must be (w>0, h>0) got(w:%s, h:%s) ' % (w, h))
    if dim != 3:
        raise ValueError('Incompatible array, must be (w, h, 3) got (%s, %s %s) ' % (w, h, dim))

    zero = zeros((w, h, 4), dtype=uint8)
    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, :] rgb_array = array
        unsigned char [:, :, ::1] empty_array = zero
        unsigned char [:, :, ::1] tmp_array = zero[:]
        unsigned char [:, :] alpha_array = alpha

    if dx==0 and dy==0:
        tmp_array = array

    if dx != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    ii = (i + dx) % w
                    if ii < 0:
                        ii = ii + w
                    empty_array[ii, j, 3] = alpha_array[i, j]
            tmp_array = empty_array
                         
    if dy != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    jj = (j + dy) % h
                    if jj < 0:
                        jj = jj + h                   
                    tmp_array[i, jj, 3] = alpha_array[i, j]

    return pygame.image.frombuffer(tmp_array, (w, h), 'RGBA'), asarray(tmp_array)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scroll_array_32_c(array, int dy, int dx):
    """  
    Roll the value of an entire array (lateral and vertical)
    The roll effect can be apply to both direction (vertical and horizontal) at the same time
    dy scroll texture vertically (-dy scroll up, +dy scroll down)
    dx scroll texture horizontally (-dx scroll left, +dx scroll right,
    array must be a numpy.ndarray type (w, h, 4) uint8
    This method return a scrolled numpy.ndarray
    
    :param array: numpy.ndarray (w, h, 4) uint8 (array to scroll)
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: a numpy.ndarray type (w, h, 4) numpy uint8
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array, a numpy.ndarray is required (got type %s)' % type(array))

    cdef int w, h, dim
    try:
        w, h, dim = (<object> array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    if w == 0 or h == 0:
        raise ValueError('Incorrect array dimension must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))
    if dim != 4:
        raise ValueError('Incompatible array, must be (w, h, 4) got (%s, %s, %s) ' % (w, h, dim))
    
    zero = zeros((w, h, 4), dtype=uint8)
    cdef:
        int i, j, ii=0, jj=0
        unsigned char [:, :, :] rgb_array = array
        unsigned char [:, :, ::1] empty_array = zero
        unsigned char [:, :, ::1] tmp_array = zero[:]

    if dx==0 and dy==0:
        tmp_array = array
    
    if dx != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    ii = (i + dx) % w
                    if ii < 0:
                        ii = ii + w                
                    empty_array[ii, j, 0] = rgb_array[i, j, 0]
                    empty_array[ii, j, 1] = rgb_array[i, j, 1]
                    empty_array[ii, j, 2] = rgb_array[i, j, 2]
                    empty_array[ii, j, 3] = rgb_array[i, j, 3]
            tmp_array = empty_array
    if dy != 0:
        with nogil:
            i=0
            j=0
            for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
                for j in range(h):
                    jj = (j + dy) % h
                    if jj < 0:
                        jj = jj + h                 
                    tmp_array[i, jj, 0] = tmp_array[i, j, 0]
                    tmp_array[i, jj, 1] = tmp_array[i, j, 1]
                    tmp_array[i, jj, 2] = tmp_array[i, j, 2]
                    tmp_array[i, jj, 3] = tmp_array[i, j, 3]    
    return asarray(tmp_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gradient_horizarray_c(int width, int height, start_color, end_color):

    """
    Create an horizontal RGB gradient array  (w, h, 3).
    First color (leftmost) is start_color
    Last color (rightmost) is end_color
    
    CREATE AN ARRAY SHAPE (W, H, 3) FILLED WITH GRADIENT COLOR (UINT8).
    e.g (FULL RED (255, 0, 0) TO FULL GREEN (0, 255, 0)).
    Index 0 gives the first value of the array (255, 0, 0)
    Last value correspond to (0, 255, 0)
    
    :param width: integer, Length/width of the array
    :param height: integer, height of the array
    :param start_color: python tuple, first color (left side of the gradient array)
    :param end_color:  python tuple, last color (right side of the gradient array)
    :return: returns an horizontal RGB gradient array type (w, h, 3) uint8.
    """
    assert width > 1, 'Positional argument width should be > 1'
    cdef:
        float [:] diff_ =  numpy.array(end_color, dtype=float32) - \
                            numpy.array(start_color, dtype=float32)
        float [::1] row = numpy.arange(width, dtype=float32) / (width - 1.0)
        unsigned char [:, :, ::1] rgb_gradient = empty((width, height, 3), dtype=uint8)
        float [3] start = numpy.array(start_color, dtype=numpy.float32)
        int i=0, j=0
    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
               rgb_gradient[i, j, 0] = <unsigned char>(start[0] + row[i] * diff_[0])
               rgb_gradient[i, j, 1] = <unsigned char>(start[1] + row[i] * diff_[1])
               rgb_gradient[i, j, 2] = <unsigned char>(start[2] + row[i] * diff_[2])

    return asarray(rgb_gradient)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gradient_vertarray_c(int width, int height, top_value, bottom_value):

    """    
    CREATE A VERTICAL ARRAY SHAPE (W, H, 3) FILLED WITH GRADIENT COLOR (UINT8).
    
    :param width: integer, Length/width of the array
    :param height: integer, height of the array
    :param top_value: python tuple, first color 
    :param bottom_value:  python tuple, last color 
    :return: returns a vertical RGB gradient array type (w, h, 3) uint8.
    """
    assert height > 1, 'Positional argument width should be > 1'
    cdef:
        float [:] diff_ =  numpy.array(bottom_value, dtype=float32) - numpy.array(top_value, dtype=float32)
        float [::1] row = numpy.arange(height, dtype=numpy.float32) / (height - 1.0)
        unsigned char [:, :, ::1] rgb_gradient = empty((width, height, 3), dtype=uint8)
        float [3] start = numpy.array(top_value, dtype=numpy.float32)
        int i=0, j=0
        
    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
               rgb_gradient[i, j, 0] = <unsigned char>(start[0] + row[i] * diff_[0])
               rgb_gradient[i, j, 1] = <unsigned char>(start[1] + row[i] * diff_[1])
               rgb_gradient[i, j, 2] = <unsigned char>(start[2] + row[i] * diff_[2])

    return asarray(rgb_gradient)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gradient_horiz_2darray_c(int width, int height, int start_value, int end_value):

    """
    Create an horizontal greyscale gradient array  shapes (w, h).
    First value (leftmost) is start_value
    Last value (rightmost) is end_value
    
    CREATE AN HORIZONTAL ARRAY SHAPE (W, H) FILLED WITH GRADIENT VALUES (UINT8).
    
    :param width: integer, Length/width of the array
    :param height: integer, height of the array
    :param start_value: python tuple, first color (left side of the gradient array)
    :param end_value:  python tuple, last color (right side of the gradient array)
    :return: returns an horizontal gradient array type (w, h) uint8.
    """
    assert width > 1, 'Positional argument width should be > 1'
    cdef:
        float diff_ =  end_value - start_value
        float [::1] row = numpy.arange(width, dtype=numpy.float32) / (width - 1.0)
        unsigned char [:, :, ::1] rgb_gradient = empty((width, height), dtype=uint8)      
        int i=0, j=0
        
    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
               rgb_gradient[i, j] = <unsigned char>(start_value + row[i] * diff_)
    return asarray(rgb_gradient)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gradient_vert_2darray_c(int width, int height, unsigned char top_value, unsigned char bottom_value):

    """
    Create a vertical greyscale gradient array  shapes (w, h).
    
    CREATE A VERTICAL ARRAY SHAPE (W, H) FILLED WITH GRADIENT VALUES (UINT8).
    
    :param width: integer, Length/width of the array
    :param height: integer, height of the array
    :param top_value: uint8 value, first color (top)
    :param bottom_value:  uint8 last color (bottom))
    :return: returns a vertical gradient array type (w, h) uint8.
    """
    assert height > 1, 'Positional argument height should be > 1'
    cdef:
        float diff_ =  bottom_value - top_value
        float [::1] row = numpy.arange(height, dtype=numpy.float32) / (height - 1.0)
        unsigned char [:, :] rgb_gradient = empty((width, height), dtype=uint8)
        int i=0, j=0
        
    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(height):
               rgb_gradient[i, j] = <unsigned char>(top_value + row[i] * diff_)
    return asarray(rgb_gradient)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gradient_color_c(int index, unsigned char [:, :, :] gradient):
    """
    Extract an RGB color from a gradient array (w, h, 3) uint8 
    
    :param index: integer value, position in the array (color choice)
    :param gradient: numpy.ndarray (w, h, 3) uint8 containing RGB values range [0..255]
    :return: Return a color tuple (RGB) values in range[0..255]
    """
    cdef int width, height
    try:
        width, height = (<object>gradient).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    assert 0 <= index <= width - 1, 'Positional argument index must be in range[0..%s] ' % (width - 1)
    return gradient[index, 0, 0], gradient[index, 0, 1], gradient[index, 0, 2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blend_texture_c(surface_, int steps, final_color_, int lerp_to_step):
    """
    
    Compatible with 32 bit with per-pixel alphas transparency.
    Blend a texture with a percentage of given color (linear lerp)
    The percentage is given by the formula (lerp_to_step/steps) * diff_ (max color deviation).
    e.g
    lerp_to_step/steps = 1, texture is blended with the maximum color deviation 100%
    lerp_to_step/steps = 0, texture remains unchanged, 0%
    when lerp_to_step/steps = 1/2, texture is blended with half of the color variation, 50% etc...
    **Source alpha channel will be transfer to the destination surface (no alteration
      of the alpha channel).

    :param surface_: pygame surface
    :param steps: integer, Steps between the texture to the given color.
    :param final_color_: Destination color. Can be a Color or a tuple
    :param lerp_to_step: integer, Lerp to the specific color (lerp_to_step/steps) * (diff) amount
    of color variation,
    :return: return a pygame.surface with per-pixels transparency only if the surface passed
                    as an argument has been created with convert_alpha() method.
                    Pixel transparency of the source array will be unchanged.
    """
    assert isinstance(steps, int), \
        'Argument steps must be a python int got %s ' % type(steps)
    assert isinstance(lerp_to_step, int), \
        'Argument lerp_to_step must be a python int got %s ' % type(lerp_to_step)
    assert isinstance(final_color_, (tuple, Color)), \
        'Argument final_color_ must be a tuple or a Color got %s ' % type(final_color_)
    assert steps != 0, 'Argument steps cannot be zero.'
    assert isinstance(surface_, Surface), \
        'Argument surface_ must be a Surface got %s ' % type(surface_)
    assert 0 <= lerp_to_step <= steps, 'Positional argument lerp_to_step must be in range [0..steps]'

    if lerp_to_step == 0:
        return surface_

    try:
        source_array = pixels3d(surface_)
        alpha_channel = pixels_alpha(surface_)
        
    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nMake sure the surface_ contains per-pixel alpha transparency values.')

    cdef:
        int w = source_array.shape[0]
        int h = source_array.shape[1]
        unsigned char [:, :, :] source = source_array
        unsigned char [:, :, ::1] final_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, :] alpha = alpha_channel
        unsigned char [:] f_color = numpy.array(final_color_[:3], dtype=uint8)  # take only rgb values
        int c1, c2, c3
        float c4 = 1.0 / steps # division by zero is checked above with assert statement
        int i=0, j=0

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(h):
                c1 = min(<int> (source[i, j, 0] + ((f_color[0] - source[i, j, 0]) * c4) * lerp_to_step), 255)
                c2 = min(<int> (source[i, j, 1] + ((f_color[1] - source[i, j, 0]) * c4) * lerp_to_step), 255)
                c3 = min(<int> (source[i, j, 2] + ((f_color[2] - source[i, j, 0]) * c4) * lerp_to_step), 255)
                if c1 < 0:
                    c1 = 0
                if c2 < 0:
                    c2 = 0
                if c3 < 0:
                    c3 = 0
                final_array[j, i, 0], final_array[j, i, 1], \
                final_array[j, i, 2], final_array[j, i, 3] = c1, c2, c3, alpha[i, j]

    return pygame.image.frombuffer(final_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blend_2_textures_c(source_, destination_, int steps, int lerp_to_step):
    """
    Compatible only with 32 bit with per-pixel alphas transparency.    
    Lerp a given surface toward a destination surface (both surface must have the same size).
    This method can be used for a transition effect
    
    **Source alpha channel will be transfer to the destination surface (no alteration
      of the alpha channel).

    :param source_: Surface
    :param destination_: Surface
    :param steps: integer, Steps between the texture to the given color.
    :param lerp_to_step: integer, Lerp to the specific color (lerp_to_step/steps) * (diff) amount
    of color variation,
    :return: return a pygame.surface with per-pixels transparency only if the surface passed
                    as an argument has been created with convert_alpha() method.
                    Pixel transparency of the source array will be unchanged.
    """
    assert isinstance(source_, Surface), \
        'Argument source_ must be a Surface got %s ' % type(source_)
    assert isinstance(destination_, Surface), \
        'Argument destination_ must be a Surface got %s ' % type(destination_)
    assert isinstance(steps, int), \
        'Argument steps must be a python int got %s ' % type(steps)
    assert isinstance(lerp_to_step, int), \
        'Argument lerp_to_step must be a python int got %s ' % type(lerp_to_step)
    assert steps != 0, 'Argument steps cannot be zero.'
    assert 0 <= lerp_to_step <= steps, 'Positional argument lerp_to_step must be in range [0..steps]'
    assert source_.get_size() == destination_.get_size(),\
        'Source and Destination surfaces must have same dimensions: ' \
        'Source (w:%s, h:%s), dest(w:%s, h:%s).' % (*source_.get_size(), *destination_.get_size())
    # no change
    if lerp_to_step == 0:
        return source_
    
    try:
        source_array = pixels3d(source_)
        alpha_channel = pixels_alpha(source_)
        destination_array = pixels3d(destination_)
        destination_alpha = pixels_alpha(destination_)
        
    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nMake sure the surface_ contains per-pixel alpha transparency values.')
    
    cdef:
        int w = source_array.shape[0]
        int h = source_array.shape[1]
        unsigned char [:, :, :] source = source_array
        unsigned char [:, :, ::1] destination = destination_array
        unsigned char [:, :, ::1] final_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, ::1] alpha = alpha_channel
        unsigned char [:, ::1] dest_alpha = destination_alpha
        int c1, c2, c3
        int i=0, j=0
        float c4 = 1.0/steps
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(h):

                c1 = min(<int> (source[i, j, 0] +
                                ((destination[i, j, 0] - source[i, j, 0]) * c4) * lerp_to_step), 255)
                c2 = min(<int> (source[i, j, 1] +
                                ((destination[i, j, 1] - source[i, j, 1]) * c4) * lerp_to_step), 255)
                c3 = min(<int> (source[i, j, 2] +
                                ((destination[i, j, 2] - source[i, j, 2]) * c4) * lerp_to_step), 255)
                if c1 < 0:
                    c1 = 0
                if c2 < 0:
                    c2 = 0
                if c3 < 0:
                    c3 = 0
                final_array[j, i, 0], final_array[j, i, 1], \
                final_array[j, i, 2], final_array[j, i, 3] = c1, c2, c3, alpha[i, j]

    return pygame.image.frombuffer(final_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef alpha_blending_c(surface1: Surface, surface2: Surface):
    """
    Alpha blending is the process of combining a translucent foreground color with a
    background color, thereby producing a new blended color.
    
    Alpha blending is the process of combining a translucent foreground color with a
    background color, thereby producing a new blended color. The degree of the foreground
    color's translucency may range from completely transparent to completely opaque.
    If the foreground color is completely transparent, the blended color will be the
    background color. Conversely, if it is completely opaque, the blended color will
    be the foreground color. The translucency can range between these extremes, in which
    case the blended color is computed as a weighted average of the foreground and background colors.
    Alpha blending is a convex combination of two colors allowing for transparency effects
    in computer graphics. The value of alpha in the color code ranges from 0.0 to 1.0, where 0.0
    represents a fully transparent color, and 1.0 represents a fully opaque color. This alpha
    value also corresponds to the ratio of "SRC over DST" in Porter and Duff equations.
    
    BOTH IMAGES MUST HAVE PER-PIXEL TRANSPARENCY.
    This technique is relatively slow

    :param surface1: (foreground image) that must contains per-pixel transparency
    :param surface2: (background image) that must contains per-pixel transparency
    :return: return a Surface with RGB values and alpha channel (per-pixel).
    """

    assert isinstance(surface1, Surface), \
        'Expecting Surface for argument surface got %s ' % type(surface1)
    assert isinstance(surface2, Surface), \
        'Expecting Surface for argument surface2 got %s ' % type(surface2)

    cdef int w, h, w2_, h2_
    # sizes
    w, h = surface1.get_size()
    w2_, h2_ = surface2.get_size()

    if w != w2_ or h != h2_:
        raise ValueError('Both surfaces must have exact same dimensions (width and height).')

    if w==0 or h==0:
        raise ValueError('Surface size is incorrect must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        float c1= 1.0 / 255.0
    try:
        buffer1 = pixels3d(surface1)
        buffer2 = pixels3d(surface2)
        buffer_alpha1 = pixels_alpha(surface1)
        buffer_alpha2 = pixels_alpha(surface2)

    except (pygame.error, ValueError) as e:
        raise ValueError('\nOnly compatible with surface with per-pixel transparency.')

    cdef:
        # Normalized every arrays / 255
        float [:, :, ::1] rgb1_normalized = numpy.array(buffer1 * c1, dtype=float32, order='C')
        float [:, :, ::1] rgb2_normalized = numpy.array(buffer2 * c1, dtype=float32, order='C')
        float [:, ::1] alpha1_normalized = numpy.array(buffer_alpha1 * c1, dtype=float32, order='C')
        float [:, ::1] alpha2_normalized = numpy.array(buffer_alpha2 * c1, dtype=float32, order='C')
        int i=0, j=0

    cdef:
        # create the outRGB
        unsigned char [:, :, ::1] outrgb = empty((h, w, 4), dtype=uint8, order='C')
        float a = 0
        unsigned char red, green, blue
    # ***********************************
    # Calculations for alpha & RGB values
    # outA = SrcA + DstA(1 - SrcA)
    # outRGB = SrcRGB + DstRGB(1 - SrcA)
    # ***********************************
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(h):
                a = (1.0 - alpha1_normalized[i, j])
                outrgb[j, i, 0] = min(<unsigned char>((rgb1_normalized[i, j, 0] +
                                                       rgb2_normalized[i, j, 0] * a) * 255.0), 255)
                outrgb[j, i, 1] = min(<unsigned char>((rgb1_normalized[i, j, 1] +
                                                       rgb2_normalized[i, j, 1] * a) * 255.0), 255)
                outrgb[j, i, 2] = min(<unsigned char>((rgb1_normalized[i, j, 2] +
                                                       rgb2_normalized[i, j, 2] * a) * 255.0), 255)
                outrgb[j, i, 3] = min(<unsigned char>((alpha1_normalized[i, j]
                                                       + alpha2_normalized[i, j] * a) * 255.0), 255)
    return pygame.image.frombuffer(outrgb, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef alpha_blending_static_c(surface1: Surface, surface2: Surface, float a1, float a2):
    """
    Alpha blending is the process of combining a translucent foreground color with a
    background color, thereby producing a new blended color.
    
    This method is almost identical to alpha_blending. 
    It uses two floats a1 and a2 to determine the blending composition between 
    the foreground and the background textures instead of the images alpha channels (per-pixel).
    This allow to blend two textures that does not have per-pixel alpha transparency and to set
    an arbitrary alpha values for each textures.
    This technique is relatively slow (46ms for blending two textures 320 x 720) 
    Compatible 24 - 32 bit surfaces (convert() or convert_alpha())
    
    :param surface1: (foreground image) that must contains per-pixel transparency 
    :param surface2: (background image) that must contains per-pixel transparency 
    :param a1: (foreground alpha value)
    :param a2: (background alpha value)
    :return: return a Surface with RGB values and alpha channel (per-pixel).
    """

    assert isinstance(surface1, Surface), \
        'Expecting Surface for argument surface got %s ' % type(surface1)
    assert isinstance(surface2, Surface), \
        'Expecting Surface for argument surface2 got %s ' % type(surface2)

    cdef int w, h, w2_, h2_
    w, h = surface1.get_width(), surface1.get_height()
    w2_, h2_ = surface2.get_width(), surface2.get_height()

    if w != w2_ or h != h2_:
        raise ValueError('Both surfaces must have exact same dimensions (width and height).')

    if w==0 or h==0:
        raise ValueError('Surface size is incorrect must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        float c1= 1.0 / 255.0
    try:
        buffer1 = pixels3d(surface1)
        buffer2 = pixels3d(surface2)
    except pygame.error as e:
        raise ValueError('\nOnly compatible with surface with per-pixel transparency.')

    cdef:
        # Normalized every arrays / 255
        float [:, :, :] rgb1_normalized = numpy.array(buffer1 * c1, dtype=float32)
        float [:, :, :] rgb2_normalized = numpy.array(buffer2 * c1, dtype=float32)
        np.ndarray[np.float32_t, ndim=2] alpha1 = numpy.full((h,w), a1, dtype=float32, order='C')
        np.ndarray[np.float32_t, ndim=2] alpha2 = numpy.full((h,w), a2, dtype=float32, order='C')
        int i=0, j=0
    cdef:
        # create the outRGB and outA arrays
        float [:, :, ::1] outrgb = empty((h, w, 3), dtype=float32, order='C')
        float a = 0, red, green, blue

    # ***********************************
    # Calculations for alpha & RGB values
    # outA = SrcA + DstA(1 - SrcA)
    # outRGB = SrcRGB + DstRGB(1 - SrcA)
    # ***********************************
    cdef float [:, ::1] alp = numpy.full((h, w), 1.0, dtype=float32, order='C') - alpha1[:, :]
    cdef float [:, ::1] outa = (alpha1[:, :] + alpha2[:, :] * alp) * 255.0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(h):
                a = (1.0 - alpha1[i, j])
                red = (rgb1_normalized[i, j, 0] + rgb2_normalized[i, j, 0] * a) * 255.0
                green = (rgb1_normalized[i, j, 1] + rgb2_normalized[i, j, 1] * a) * 255.0
                blue = (rgb1_normalized[i, j, 2] + rgb2_normalized[i, j, 2] * a) * 255.0
                outrgb[j, i, 0] = min(red, 255.0)
                outrgb[j, i, 1] = min(green, 255.0)
                outrgb[j, i, 2] = min(blue, 255.0)


    array = dstack((outrgb, outa)).astype(uint8)
    return pygame.image.frombuffer(array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef invert_surface_24bit_c(image):
    """
    Inverse RGB color values of an image
    
    Return an image with inverted colors (such as all pixels -> 255 - pixel color ). 
    Compatible with 24 bit image only.  
    :param image: image (Surface) to invert  
    :return: return a pygame Surface
    """
    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        try:
            array_ = array3d(image)
        except (pygame.error, ValueError):
            raise ValueError('Incompatible pixel format.')

    cdef int w, h, dim
    try:
        w, h, dim = array_.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % \
                (w, h, dim)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] inverted_array  = empty((h, w, 3), dtype=uint8)
        int i=0, j=0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                inverted_array[j, i, 0] = 255 -  rgb_array[i, j, 0]
                inverted_array[j, i, 1] = 255 -  rgb_array[i, j, 1]
                inverted_array[j, i, 2] = 255 -  rgb_array[i, j, 2]
    return pygame.image.frombuffer(inverted_array, (w, h), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef invert_surface_24bit_b(image):
    """
    
    
    Return an image with inverted colors (such as all pixels -> 255 - pixel color ). 
    Compatible with 24 bit image only.  
    :param image: image (Surface) to invert  
    :return: return a pygame Surface with per-pixel transparency (alpha set to 255)
    """

    try:
        rgb_buffer_ = image.get_view('2')

    except (pygame.error, ValueError):
        raise ValueError('Incompatible pixel format.')

    cdef int w, h
    try:
        w, h = image.get_size()
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h != 0, 'Image with incorrect dimensions (w>0, h>0) got (w:%s, h:%s) ' % (w, h)
    cdef:
        unsigned char [:] rgb_array = numpy.frombuffer(rgb_buffer_, dtype=numpy.uint8)
        int b_length = rgb_buffer_.length
        int i=0
        
    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            rgb_array[i] = 255 - rgb_array[i]
            rgb_array[i + 1] = 255 - rgb_array[i + 1]
            rgb_array[i + 2] = 255 - rgb_array[i + 2]
            rgb_array[i + 3] = 255
    return pygame.image.frombuffer(rgb_array, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef invert_surface_32bit_c(image):
    """
    Invert RGB color values of an image (alpha remains unchanged)
    
    Return an image with inverted colors (such as all pixels -> 255 - pixel color ). 
    Compatible with 24 - 32 bit images. The image must be encoded with alpha transparency. 
    Channel alpha is transfer to the final image without being modified. 
    :param image: image (Surface) to invert  
    :return: return a pygame Surface with per-pixel transparency
    """
    try:
        array_ = pixels3d(image)
        alpha_ = pixels_alpha(image)
        
    except (pygame.error, ValueError):
        raise ValueError('Incompatible pixel format.')

    cdef int w, h, dim
    try:
        w, h, dim = array_.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % \
                (w, h, dim)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, ::1 ] alpha_array = alpha_
        unsigned char [:, :, ::1] inverted_array  = empty((h, w, 4), dtype=uint8)
        int i=0, j=0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                inverted_array[j, i, 0] = 255 -  rgb_array[i, j, 0]
                inverted_array[j, i, 1] = 255 -  rgb_array[i, j, 1]
                inverted_array[j, i, 2] = 255 -  rgb_array[i, j, 2]
                inverted_array[j, i, 3] = alpha_array[i, j]
    return pygame.image.frombuffer(inverted_array, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef invert_surface_32bit_b(image):
    """
    
    
    Return an image with inverted colors (such as all pixels -> 255 - pixel color ). 
    Compatible with 24 - 32 bit images. The image must be encoded with alpha transparency. 
    Channel alpha is transfer to the final image without being modified. 
    :param image: image (Surface) to invert  
    :return: return a pygame Surface
    """
    try:
        rgba_buffer_ = image.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('Incompatible pixel format.')

    cdef int w, h
    try:
        w, h = image.get_size()
        
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0,\
            'Image with incorrect dimensions (w>0, h>0) got (w:%s, h:%s) ' % (w, h)
    cdef:
        unsigned char [:] rgba_buffer = numpy.frombuffer(rgba_buffer_, dtype=numpy.uint8)
        int b_length = rgba_buffer_.length
        int i=0, j = 0
    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):    
                rgba_buffer[i + 0] = 255 -  rgba_buffer[i + 0]
                rgba_buffer[i + 1] = 255 -  rgba_buffer[i + 1]
                rgba_buffer[i + 2] = 255 -  rgba_buffer[i + 2]
                rgba_buffer[i + 3] = 255
    return pygame.image.frombuffer(rgba_buffer, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef invert_array_24_c(array_):
    """
    Inverse RGB color values of an image
    
    Return a numpy.ndarray (w, h, 3) uint8 with inverted colors
    (such as all pixels -> 255 - pixel color ).  
    :param array_: numpy.ndarray type(w, h, 3) uint8 to invert  
    :return: return inverted numpy.array (w, h, 3) uint8 
    """
    cdef int w, h, dim
    w, h, dim = array_.shape[:3] 
    assert w != 0 or h !=0 or dim!=3,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % \
                (w, h, dim)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] inverted_array  = empty((h, w, 3), dtype=uint8)
        int i=0, j=0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(h):
                inverted_array[j, i, 0] = 255 -  rgb_array[i, j, 0]
                inverted_array[j, i, 1] = 255 -  rgb_array[i, j, 1]
                inverted_array[j, i, 2] = 255 -  rgb_array[i, j, 2]
    return asarray(inverted_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef invert_array_32_c(array_):
    """
    Inverse RGB color values (alpha layer remains unchanged)
    
    Return a numpy.ndarray (w, h, 4) uint8 with inverted colors (such as all pixels -> 255 - pixel color ).  
    :param array_: numpy.ndarray type(w, h, 4) uint8 to invert  
    :return: return inverted numpy.array (w, h, 4) uint8 with alpha transparency unchanged 
    """
    cdef int w, h, dim
    try:
        w, h, dim = array_.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0 or dim!=4,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % \
                (w, h, dim)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] inverted_array  = empty((h, w, 4), dtype=uint8)
        int i=0, j=0
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for j in range(h):
                inverted_array[j, i, 0] = 255 -  rgb_array[i, j, 0]
                inverted_array[j, i, 1] = 255 -  rgb_array[i, j, 1]
                inverted_array[j, i, 2] = 255 -  rgb_array[i, j, 2]
                inverted_array[j, i, 3] = rgb_array[i, j, 3]
    return asarray(inverted_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline deviation(double sigma, int kernel_size):
    """
    (1/(sqrt(2*pi)*sigma)) * e(-(x * x / 2 * sigma * sigma)) 
    
    Sample Gaussian matrix
    This is a sample matrix, produced by sampling the Gaussian filter kernel 
    (with σ = 0.84089642) at the midpoints of each pixel and then normalizing.
    Note that the center element (at [4, 4]) has the largest value, decreasing
     symmetrically as distance from the center increases.
    0.00000067	0.00002292	0.00019117	0.00038771	0.00019117	0.00002292	0.00000067
    0.00002292	0.00078633	0.00655965	0.01330373	0.00655965	0.00078633	0.00002292
    0.00019117	0.00655965	0.05472157	0.11098164	0.05472157	0.00655965	0.00019117
    0.00038771	0.01330373	0.11098164	0.22508352	0.11098164	0.01330373	0.00038771
    0.00019117	0.00655965	0.05472157	0.11098164	0.05472157	0.00655965	0.00019117
    0.00002292	0.00078633	0.00655965	0.01330373	0.00655965	0.00078633	0.00002292
    0.00000067	0.00002292	0.00019117	0.00038771	0.00019117	0.00002292	0.00000067
    
    :param sigma: sigma value 
    :param kernel_size: kernel size example 3x3, 5x5 etc 
    :return: a numpy.ndarray
    
    """

    assert isinstance(sigma, float), \
            'Positional argument sigma should be a python float, got %s ' % type(sigma)
    assert isinstance(kernel_size, int), \
           'Positional argument kernel_size should be an integer, got %s ' % type(kernel_size)
    if sigma==0.0:
        raise ValueError('Argument sigma cannot be equal to zero.')
    if kernel_size <=0:
        raise ValueError('Argument kernel_size cannot be <=0.')

    # In two dimensions, it is the product of two such Gaussian functions, one in each dimension:
    # 1 / (2 * math.pi * (sigma ** 2)) * math.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    cdef:
        double g1, g2
        double [:, :] kernel = zeros((kernel_size, kernel_size), dtype=float64)
        int half_kernel = kernel_size >> 1
        int x, y

    g1 = 1 / (2 * pi * (sigma * sigma))
    for x in range(-half_kernel, half_kernel+1):
            for y in range(-half_kernel, half_kernel+1):
                    g2 = exp(-((x * x + y *y) / (2 * sigma * sigma)))
                    g = g1 * g2
                    kernel[x + half_kernel, y + half_kernel] = g
    return asarray(kernel)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef sharpen3x3_c(image):
    """
    Sharpen image applying the below 3 x 3 kernel over every pixels.
    pixels convoluted outside image edges will be set to adjacent edge value
    [0 , -1,  0]
    [-1,  5, -1]
    [0 , -1,  0]
    
    :param image: Surface 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with no alpha channel
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)
    # kernel definition
    kernel = numpy.array(([0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0])).astype(dtype=float32, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(kernel)
    k_length = len(kernel)
    half_kernel = len(kernel) >> 1
    
    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
        
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        float kernel_weight = k_weight
        float [:, ::1] sharpen_kernel = \
            numpy.divide(kernel, k_weight, dtype=float32) if k_weight!=0 else kernel
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] sharp_array = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = sharpen_kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k

                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                sharp_array[x, y, 0], \
                sharp_array[x, y, 1], \
                sharp_array[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(sharp_array)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef guaussian_boxblur3x3_c(image):
    """
    Apply a Gaussian blur box filter using the below 3 x 3 kernel.
    pixels convoluted outside image edges will be set to adjacent edge value
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
    
    :param image: pygame.surface 8, 24-32 bit format 
    :return: numpy.ndarray (w, h, 3) uint8 
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)
    # kernel definition
    kernel = numpy.array(([1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1])).astype(dtype=float32, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(kernel)
    k_length = len(kernel)
    half_kernel = len(kernel) >> 1

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
        
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        float kernel_weight = k_weight
        float [:, ::1] sharpen_kernel = \
            numpy.divide(kernel, k_weight, dtype=float32) if k_weight!=0 else kernel
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] sharp_array = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = sharpen_kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k

                # todo check if below is necessary all value are x 1.0
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                sharp_array[x, y, 0], \
                sharp_array[x, y, 1], \
                sharp_array[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(sharp_array)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef guaussian_boxblur3x3_capprox(image):
    """
    Boxblur using kernel 3 x 3
    [1.0/9.0, 1.0/9.0, 1.0/9.0],
    [1.0/9.0, 1.0/9.0, 1.0/9.0],
    [1.0/9.0, 1.0/9.0, 1.0/9.0]
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)
    # kernel definition
    kernel = numpy.array(([1.0/9.0, 1.0/9.0, 1.0/9.0],
                          [1.0/9.0, 1.0/9.0, 1.0/9.0],
                          [1.0/9.0, 1.0/9.0, 1.0/9.0])).astype(dtype=float32, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(kernel)
    k_length = len(kernel)
    half_kernel = len(kernel) >> 1
    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        float kernel_weight = k_weight
        float [:, ::1] sharpen_kernel = \
            numpy.divide(kernel, k_weight, dtype=float32) if k_weight!=0 else kernel
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] sharp_array = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = sharpen_kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                # TODO REMOVE BELOW total kernel weight is 1
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                sharp_array[x, y, 0], \
                sharp_array[x, y, 1], \
                sharp_array[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(sharp_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef guaussian_blur3x3_c(image):
    """
    Gaussian blur using kernel 3 x 3
    [1.0/16.0, 2.0/16.0, 1.0/16.0],
    [2.0/16.0, 4.0/16.0, 2.0/16.0],
    [1.0/16.0, 2.0/16.0, 1.0/16.0]
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)
    # kernel definition
    kernel = numpy.array(([1.0/16.0, 2.0/16.0, 1.0/16.0],
                          [2.0/16.0, 4.0/16.0, 2.0/16.0],
                          [1.0/16.0, 2.0/16.0, 1.0/16.0])).astype(dtype=float32, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(kernel)
    k_length = len(kernel)
    half_kernel = len(kernel) >> 1

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        float kernel_weight = k_weight
        float [:, ::1] sharpen_kernel = \
            numpy.divide(kernel, k_weight, dtype=float32) if k_weight!=0 else kernel
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] sharp_array = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = sharpen_kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                # todo remove below, kernel weight is 1.0
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                sharp_array[x, y, 0], \
                sharp_array[x, y, 1], \
                sharp_array[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(sharp_array)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gaussian_blur5x5_c(rgb_array: numpy.ndarray):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgb_array: numpy.ndarray type (w, h, 3) uint8 
    :return: a numpy.ndarray type (w, h, 3) uint8
    """

    assert isinstance(rgb_array, numpy.ndarray),\
        'Positional arguement rgb_array must be a numpy.ndarray, got %s ' % type(rgb_array)

    cdef int w, h, dim
    try:
        w, h, dim = rgb_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w!=0 or h !=0, 'Array with incorrect shapes (w>0, h>0) got (%s, %s) ' % (w, h)

    zero = zeros((w, h, 3), dtype=uint8)
    # kernel 5x5 separable
    cdef:
        double [:] kernel_v = numpy.array(([1.0 / 16.0,
                                            4.0 / 16.0,
                                            6.0 / 16.0,
                                            4.0 / 16.0,
                                            1.0 / 16.0]), dtype=float64)  # vertical vector
        double [:] kernel_h = numpy.array(([1.0 / 16.0,
                                            4.0 / 16.0,
                                            6.0 / 16.0,
                                            4.0 / 16.0,
                                            1.0 / 16.0]), dtype=float64)  # horizontal vector
        short int kernel_half = 2
        unsigned char [:, :, :] convolve = zero
        unsigned char [:, :, :] convolved = zero[:]
        unsigned char [:, :, :] rgb_array_ = rgb_array
        short int kernel_length = len(kernel_h)
        int x, y, xx, yy
        double k, r, g, b
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel_h[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundarie.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                    # No need to cap the values
                    # (color brightness conservation see kernel values)
                    # if r > 255.0:
                    #     r = 255.0
                    # if g > 255.0:
                    #     g = 255.0
                    # if b > 255.0:
                    #     b = 255.0

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # return asarray(convolve)

        # Vertical convolution
        for x in prange(0,  w):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel_v[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[x, y, 0], convolved[x, y, 1], convolved[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(convolved)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gaussian_blur5x5_mask_c(rgb_array, mask):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgb_array: numpy.ndarray type (w, h, 3) uint8 
    :return: a numpy.ndarray type (w, h, 3) uint8
    """

    assert isinstance(rgb_array, numpy.ndarray),\
        'Positional arguement rgb_array must be a numpy.ndarray, got %s ' % type(rgb_array)
    assert isinstance(mask, numpy.ndarray),\
        'Positional arguement mask must be a numpy.ndarray, got %s ' % type(mask)

    cdef int w, h, dim, w2, h2, dim2
    try:
        w, h, dim = rgb_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not compatible.')

    try:
        w2, h2, dim2 = mask.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nMask array shape not compatible')
    
    assert w != 0 or h != 0, 'Array with incorrect shapes (w>0, h>0) got (%s, %s) ' % (w, h)
    assert w == w2 and h == h2, 'Mask array size mismatch.'

    zero = zeros((w, h, 3), dtype=uint8)
    # kernel 5x5 separable
    cdef:
        double [:] kernel_v = numpy.array(([1.0 / 16.0,
                                            4.0 / 16.0,
                                            6.0 / 16.0,
                                            4.0 / 16.0,
                                            1.0 / 16.0]), dtype=float64)  # vertical vector
        double [:] kernel_h = numpy.array(([1.0 / 16.0,
                                            4.0 / 16.0,
                                            6.0 / 16.0,
                                            4.0 / 16.0,
                                            1.0 / 16.0]), dtype=float64)  # horizontal vector
        short int kernel_half = 2
        unsigned char [:, :, :] convolve = zero
        unsigned char [:, :, :] convolved = zero[:]
        unsigned char [:, :, :] rgb_array_ = rgb_array
        unsigned char [:, :] mask_array = mask
        short int kernel_length = len(kernel_h)
        int x, y, xx, yy
        double k, r, g, b
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h):  
            for x in range(0, w):
                         
                if mask_array[x, y] > 0:

                    r, g, b = 0, 0, 0

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = kernel_h[kernel_offset + kernel_half]

                        xx = x + kernel_offset

                        # check boundarie.
                        # Fetch the edge pixel for the convolution
                        if xx < 0:
                            red, green, blue = rgb_array_[0, y, 0],\
                            rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                        elif xx > (w - 1):
                            red, green, blue = rgb_array_[w-1, y, 0],\
                            rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                        else:
                            red, green, blue = rgb_array_[xx, y, 0],\
                                rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                        r = r + red * k
                        g = g + green * k
                        b = b + blue * k

                        # No need to cap the values
                        # (color brightness conservation see kernel values)
                        # if r > 255.0:
                        #     r = 255.0
                        # if g > 255.0:
                        #     g = 255.0
                        # if b > 255.0:
                        #     b = 255.0

                    convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                        <unsigned char>g, <unsigned char>b
                else:
                    convolve[x, y, 0] = rgb_array_[x, y, 0]
                    convolve[x, y, 1] = rgb_array_[x, y, 1]
                    convolve[x, y, 2] = rgb_array_[x, y, 2]
                         

        # return asarray(convolve)

        # Vertical convolution
        for x in prange(0,  w):

            for y in range(0, h):

                if mask_array[x, y] > 0:
                    r, g, b = 0, 0, 0

                    for kernel_offset in range(-kernel_half, kernel_half + 1):

                        k = kernel_v[kernel_offset + kernel_half]
                        yy = y + kernel_offset

                        if yy < 0:
                            red, green, blue = convolve[x, 0, 0],\
                            convolve[x, 0, 1], convolve[x, 0, 2]
                        elif yy > (h -1):
                            red, green, blue = convolve[x, h-1, 0],\
                            convolve[x, h-1, 1], convolve[x, h-1, 2]
                        else:
                            red, green, blue = convolve[x, yy, 0],\
                                convolve[x, yy, 1], convolve[x, yy, 2]

                        r = r + red * k
                        g = g + green * k
                        b = b + blue * k

                    convolved[x, y, 0], convolved[x, y, 1], convolved[x, y, 2] = \
                        <unsigned char>r, <unsigned char>g, <unsigned char>b
                else:
                    convolve[x, y, 0] = rgb_array_[x, y, 0]
                    convolve[x, y, 1] = rgb_array_[x, y, 1]
                    convolve[x, y, 2] = rgb_array_[x, y, 2]

    return asarray(convolved)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef edge_detection3x3_c(image):
    """
    Edge detection filter, using kernel 3x3 
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
    
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)
    # kernel definition
    kernel = numpy.array(([-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1])).astype(dtype=float32, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(kernel)
    k_length = len(kernel)
    half_kernel = len(kernel) >> 1
    
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
        
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)

    cdef:
        float kernel_weight = k_weight
        float [:, ::1] sharpen_kernel = \
            numpy.divide(kernel, k_weight, dtype=float32) if k_weight!=0 else kernel
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] sharp_array = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = sharpen_kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                # TODO REMOVE BELOW CODE, kernel weight is 1.0
                if r < 0.0:
                    r = 0.0
                if g < 0.0:
                    g = 0.0
                if b < 0.0:
                    b = 0.
                if r > 255.0:
                    r = 255.0
                if g > 255.0:
                    g = 255.0
                if b > 255.0:
                    b = 255.0

                sharp_array[x, y, 0], \
                sharp_array[x, y, 1], \
                sharp_array[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(sharp_array)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef edge_detection3x3_c1(image):
    """
    Edge detection filter, using kernel 3x3 
    [0 , -1,  0]
    [-1,  4, -1]
    [0 , -1,  0]
    
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """


    assert isinstance(image, Surface), \
            'Argument image must be a valid Surface, got %s ' % type(image)

    # kernel definition
    kernel = numpy.array(([0, 1,  0],
                          [1, -4, 1],
                          [0, 1,  0])).astype(dtype=float32, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(kernel)
    k_length = len(kernel)
    half_kernel = len(kernel) >> 1

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
        
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    
    cdef:
        float kernel_weight = k_weight
        float [:, ::1] sharpen_kernel = \
            numpy.divide(kernel, k_weight, dtype=float32) if k_weight!=0 else kernel
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] sharp_array = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = sharpen_kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                # TODO REMOVE BELOW ? kernel weight 0
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                sharp_array[x, y, 0], \
                sharp_array[x, y, 1], \
                sharp_array[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(sharp_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef edge_detection3x3_c2(image):
    """
    Edge detection filter, using kernel 3x3 
    [0 , -1,  0]
    [-1,  4, -1]
    [0 , -1,  0]
    
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
        
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w == 0 or h ==0,\
            'Image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        int w_1 = w - 1
        int h_1 = h - 1
        unsigned char [:, :, ::1] sharp_array = zeros((w, h, 3), order='C', dtype=uint8)
        int x, y, xx, yy, red, green, blue,

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                red, green, blue = 0, 0, 0

                # check the edges. Use the first pixel (adjacent pixel)
                # for convolution when kernel is not fully covering
                # the image.
                if x < 0:
                    x = 0
                elif x > w_1:
                    x = w_1

                if y < 0:
                    y = 0
                elif y > h_1:
                    y = h_1

                red = rgb_array[x, y-1, 0] + rgb_array[x-1, y, 0] - 4 * rgb_array[x, y, 0] +\
                      rgb_array[x+1, y, 0] + rgb_array[x, y+1, 0]
                green = rgb_array[x, y-1, 1] + rgb_array[x-1, y, 1] - 4 * rgb_array[x, y, 1] +\
                      rgb_array[x+1, y, 1] + rgb_array[x, y+1, 1]
                blue = rgb_array[x, y-1, 2] + rgb_array[x-1, y, 2] - 4 * rgb_array[x, y, 2] +\
                      rgb_array[x+1, y, 2] + rgb_array[x, y+1, 2]
                # TODO CHECK IF BELOW IS NECESSARY
                if red < 0:
                    red = 0
                if green < 0:
                    green = 0
                if blue < 0:
                    blue = 0
                if red > 255:
                    red= 255
                if green > 255:
                    green = 255
                if blue > 255:
                    blue = 255

                sharp_array[x, y, 0], \
                sharp_array[x, y, 1], \
                sharp_array[x, y, 2] = <unsigned char>red, <unsigned char>green, <unsigned char>blue

    return asarray(sharp_array)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef highpass_3x3_c(image):
    """
    High-pass filter using kernel 3 x 3
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
    pixels convoluted outside image edges will be set to adjacent values
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)
    # kernel definition
    kernel_ = numpy.array(([-1, -1, -1],
                     [-1,  9, -1],
                     [-1, -1, -1])).astype(dtype=numpy.int8, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(kernel_)
    k_length = len(kernel_)
    half_kernel = len(kernel_) >> 1

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        int [:, :] kernel = kernel_
        float kernel_weight = k_weight
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] highpass = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                highpass[x, y, 0], \
                highpass[x, y, 1], \
                highpass[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(highpass)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef highpass_5x5_c(image):
    """
    High-pass filter using kernel 5 x 5
    [ 0, -1, -1, -1,   0],
    [-1,  2, -4,  2,  -1],
    [-1, -4, 13, -4,  -1],
    [-1,  2, -4,  2,  -1],
    [ 0, -1, -1, -1,   0]
    pixels convoluted outside image edges will be set to adjacent values
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid pygame surface, got %s ' % type(image)
    # kernel definition
    k_ = numpy.array(([ 0, -1, -1, -1,  0],
                          [-1,  2, -4,  2, -1],
                          [-1, -4, 13, -4, -1],
                          [-1,  2, -4,  2, -1],
                          [ 0, -1, -1, -1,  0])).astype(dtype=numpy.int8, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(k_)
    k_length = len(k_)
    half_kernel = len(k_) >> 1

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        int [:, :] kernel = k_
        float kernel_weight = k_weight
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] highpass = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                highpass[x, y, 0], \
                highpass[x, y, 1], \
                highpass[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(highpass)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef laplacien_3x3_c(image):
    """
    High-pass filter using kernel 3 x 3
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
    pixels convoluted outside image edges will be set to adjacent values
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)
    # kernel definition
    k_ = numpy.array(([ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0])).astype(dtype=numpy.int8, order='C')
    # kernel weight & kernel length
    k_weight = numpy.sum(k_)
    k_length = len(k_)
    half_kernel = len(k_) >> 1

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        int [:, :] kernel = k_
        float kernel_weight = k_weight
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] laplacien = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                laplacien[x, y, 0], \
                laplacien[x, y, 1], \
                laplacien[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(laplacien)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef laplacien_5x5_c(image):
    """
    High-pass filter using kernel 5 x 5
    [-1, -1, -1, -1, -1],
    [-1, -1, 24, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
    pixels convoluted outside image edges will be set to adjacent values
    
    :param image: Surface, 8, 24-32 bit format
    :return: return a numpy.ndarray (w, h, 3) uint8 with RGB values
    """

    assert isinstance(image, Surface), \
        'Argument image must be a valid pygame surface, got %s ' % type(image)

    # kernel definition
    k_ = numpy.array(([-1, -1, -1, -1, -1],
                     [-1, -1, 24, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1])).astype(dtype=numpy.int, order='C')

    # kernel weight & kernel length
    k_weight = numpy.sum(k_)
    k_length = len(k_)
    half_kernel = len(k_) >> 1

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        int [:, :] kernel = k_
        float kernel_weight = k_weight
        unsigned char [:, :, :] rgb_array = array_
        short kernel_half = half_kernel
        unsigned char [:, :, ::1] laplacien = zeros((w, h, 3), order='C', dtype=uint8)
        int kernel_length = k_length
        int x, y, xx, yy
        unsigned short red, green, blue,
        short kernel_offset_y, kernel_offset_x
        float r, g, b, k

    with nogil:

        for x in prange(0, w):

            for y in range(0, h):

                r, g, b = 0, 0, 0

                for kernel_offset_y in range(-kernel_half, kernel_half + 1):

                    for kernel_offset_x in range(-kernel_half, kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        # check the edges. Use the first pixel (adjacent pixel)
                        # for convolution when kernel is not fully covering
                        # the image.
                        if xx == -1:
                            xx = 0
                        elif xx == w + 1:
                            xx = w

                        if yy == -1:
                            yy = 0
                        elif yy == h + 1:
                            yy = h


                        red, green, blue = rgb_array[xx, yy, 0], rgb_array[xx, yy, 1],\
                            rgb_array[xx, yy, 2]
                        k = kernel[kernel_offset_y + kernel_half, kernel_offset_x + kernel_half]
                        r += red * k
                        g += green * k
                        b += blue * k
                
                if r < 0:
                    r = 0
                if g < 0:
                    g = 0
                if b < 0:
                    b = 0
                if r > 255:
                    r= 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                laplacien[x, y, 0], \
                laplacien[x, y, 1], \
                laplacien[x, y, 2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return asarray(laplacien)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef water_ripple_effect_cf(texture, int frames, int dropy,
                            int dropx, int intensity, int drops_number):
    """
    WATER RIPPLE EFFECT LOCALISED DROPLET(s) FLOAT
    Create a water ripple effect animation on the given texture / surface.
    texture: (Surface) image to use for animation (compatible with PNG, JPG, 24-32bit format.
    frames: (python integer) Number if images to be processed. Frame must be > 0.
    dropx: (python integer) x coordinate for the droplet (image width), dropx must be in range[0, width]
    dropy: (python integer) y coordinate for the droplet (image height), dropx must be in range[0, height]
    intensity: (python integer) intensity value must be > 0 and above 1024 to notice the wavelet on the texture

    :param texture: pygame Surface
    :param frames: Number of sprites/images for the animation 
    :param dropx: integer x coordinates for the droplet ( must be in range [0..image width])
    :param dropy: integer y coordinates for the water droplet (must be in range[0..image height])
    :param intensity: Ripple intensitty.
    :param drops_number: number of drops during animation
    """
    assert isinstance(texture, Surface), \
           "Argument texture must be a valid Surface, got %s " % type(texture)
    assert isinstance(frames, int),\
           "Positional argument frames must be a python integer got %s " % type(frames)
    assert drops_number!=0, 'Positional argument drops_number cannot be zero.'
    assert intensity!=0, 'Positional argument intensity cannot be zero.'

    cdef int w, h
    w, h = texture.get_size()
    assert w != 0 or h !=0,\
            'texture with incorrect size (w>0, h>0) got (w%s, h%s) ' % (w, h)
    sprites = []
    zero = zeros((w, h), dtype=float32, order='C')
    cdef:
        float [:, ::1] current = zero
        float [:, ::1] previous = zero[:]
        unsigned char [:, :, :] texture_array = array3d(texture)
        unsigned char [:, :, :] back_array = array3d(texture)
        int i=0, freq
    # water drop position
    previous[dropx, dropy] = intensity

    freq = <int>(max(frames / drops_number, 1.0))

    # new surface
    surface = Surface((w, h), RLEACCEL)

    for i in range(frames):

        if i % freq == 0:
            previous[dropx, dropy] = intensity


        previous, current, back_array =\
                   droplet_float(h, w, previous, current, texture_array, back_array)

        pygame.surfarray.blit_array(surface, asarray(back_array))
        sprites.append(surface.convert())

    return sprites


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef water_ripple_effect_ci(texture, int frames, int dropy,
                            int dropx, int intensity, int drops_number):
    """
    WATER RIPPLE EFFECT LOCALISED DROPLET(s) INTEGER
    Create a water ripple effect animation on the given texture / surface.
    texture: (Surface) image to use for animation (compatible with PNG, JPG, 24-32bit format.
    frames: (python integer) Number if images to be processed. Frame must be > 0.
    dropx: (python integer) x coordinate for the droplet (image width), dropx must be in range[0, width]
    dropy: (python integer) y coordinate for the droplet (image height), dropx must be in range[0, height]
    intensity: (python integer) intensity value must be > 0 and above 1024 to notice the wavelet on the texture

    
    :param texture: Surface
    :param frames: Number of sprites/images for the animation 
    :param dropx: integer x coordinates for the droplet ( must be in range [0..image width])
    :param dropy: integer y coordinates for the water droplet (must be in range[0..image height])
    :param intensity: Ripple intensitty.
    :param drops_number: number of drops during animation
    """
    assert isinstance(texture, Surface), \
           "Argument texture must be a valid Surface, got %s " % type(texture)
    assert isinstance(frames, int),\
           "Positional argument frames must be a python integer got %s " % type(frames)
    assert drops_number!=0, 'Positional argument drops_number cannot be zero.'
    assert intensity!=0, 'Positional argument intensity cannot be zero.'

    cdef int w, h 
    w, h = texture.get_size()
    assert w != 0 or h !=0,\
            'texture with incorrect size (w>0, h>0) got (w%s, h%s) ' % (w, h)
    sprites = []
    zero = zeros((w, h), dtype=int32, order='C')
    cdef:
        int [:, ::1] current = zero
        int [:, ::1] previous = zero[:]
        unsigned char [:, :, :] texture_array = array3d(texture)
        unsigned char [:, :, :] back_array = array3d(texture)
        int i=0, freq
    # water drop position
    previous[dropx, dropy] = intensity

    freq = <int>(max(frames / drops_number, 1.0))

    # new surface
    surface = Surface((w, h), RLEACCEL)

    for i in range(frames):

        if i % freq == 0:
            previous[dropx, dropy] = intensity


        previous, current, back_array =\
                   droplet_int(h, w, previous, current, texture_array, back_array)

        pygame.surfarray.blit_array(surface, asarray(back_array))
        sprites.append(surface.convert())

    return sprites


cdef water_ripple_effect_crand(texture, int frames, int intensity, int drops_number):
    """
    WATER RIPPLE EFFECT (RANDOM DROPLET(s))
    Create a water ripple effect animation on the given texture / surface.
    texture: (Surface) image to use for animation (compatible with PNG, JPG, 24-32bit format.
    frames: (python integer) Number if images to be processed. Frame must be > 0.
    intensity: (python integer) intensity value must be > 0 and above 1024 to notice the wavelet on the texture
    
    :param texture: Surface
    :param frames: Number of sprites/images for the animation 
    :param intensity: Ripple intensity.
    :param drops_number: number of drops during animation
    """
    assert isinstance(texture, Surface), \
           "Argument texture must be a valid Surface, got %s " % type(texture)
    assert isinstance(frames, int),\
           "Positional argument frames must be a python integer got %s " % type(frames)
    assert drops_number!=0, 'Positional argument drops_number cannot be zero.'
    assert intensity!=0, 'Positional argument intensity cannot be zero.'

    cdef int w, h
    w, h = texture.get_size()

    assert w != 0 or h !=0,\
            'texture with incorrect size (w>0, h>0) got (w%s, h%s) ' % (w, h)
    
    sprites = []
    zero = zeros((w, h), dtype=float32, order='C')
    cdef:
        float [:, ::1] current = zero
        float [:, ::1] previous = zero[:]
        unsigned char [:, :, :] texture_array = array3d(texture)
        unsigned char [:, :, :] back_array = array3d(texture)
        int i=0, freq
    # water drop position
    cdef int dropx = rand() % w
    cdef int dropy = rand() % h
    previous[dropx, dropy] = intensity
    
    freq = <int>(max(frames / drops_number, 1.0))

    surface = Surface((w, h), RLEACCEL)

    for i in range(frames):

        if i % freq == 0:
            dropx = rand() % w
            dropy = rand() % h
            previous[dropx, dropy] = intensity
        

        previous, current, back_array =\
                   droplet_float(h, w, previous, current, texture_array, back_array)
        
        # for blending effect 
        # surface = Surface((width, height), RLEACCEL)
        # surface.blit(pygame.surfarray.make_surface_c(asarray(back_array)), (0, 0))

        pygame.surfarray.blit_array(surface, asarray(back_array))
        sprites.append(surface.convert())
        
    return sprites


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def lpf(rgb_array: numpy.ndarray):
    """
    A low-pass filter (LPF) is a filter that passes signals 
    with a frequency lower than a selected cutoff frequency 
    and attenuates signals with frequencies higher than the 
    cutoff frequency. 

    rgb_array must be a greyscale 1d numpy.ndarray shape (w, h) uint8 or float
    returns a numpy.ndarray shape (w, h) float64
    The resulting array needs to be converted back to greyscale before being display as an image.
    don't forget to change the type from float64 to uint8.

    :param rgb_array: numpy.ndarray of type (w, h) containing grey colors uint8
    :return: a numpy.array shape (w, h) float64, this array needs to be converted to greyscale
    in order to be display
    """
    assert isinstance(rgb_array, numpy.ndarray), \
        'Argument rgb_array must be a numpy.ndarray, got %s ' % type(rgb_array)

    frequence = numpy.fft.fft2(rgb_array)
    freq_shift = numpy.fft.fftshift(frequence)
    try:
        rows, cols = rgb_array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    rows2, cols2 = rows >> 1, cols >> 1

    low_freq = freq_shift[rows2 - 30:rows2 + 30, cols2 - 30:cols2 + 30]
    freq_shift = zeros((rows, cols), dtype=numpy.complex_)
    freq_shift[rows2 - 30:rows2 + 30, cols2 - 30:cols2 + 30] = low_freq
    f_ishift = numpy.fft.ifftshift(freq_shift)
    array_ = numpy.fft.ifft2(f_ishift)
    return numpy.abs(array_)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def hpf(rgb_array: numpy.ndarray):
    """
    A high-pass filter (HPF) is an electronic filter that 
    passes signals with a frequency higher than a certain 
    cutoff frequency and attenuates signals with frequencies 
    lower than the cutoff frequency. The amount of attenuation
    for each frequency depends on the filter design. 
    A high-pass filter is usually modeled as a linear 
    time-invariant system.

    rgb_array must be a greyscale 1d numpy.ndarray shape (w, h) uint8 or float
    returns a numpy.ndarray shape (w, h) float64
    The resulting array needs to be converted back to greyscale before being display as an image.
    don't forget to change the type from float64 to uint8.

    :param rgb_array: numpy.ndarray of type (w, h) containing grey colors uint8
    :return: a numpy.array shape (w, h) float64, this array needs to be converted to greyscale
    in order to be display
    """

    assert isinstance(rgb_array, numpy.ndarray), \
        'Arguement rgb_array must be a numpy.ndarray, got %s ' % type(rgb_array)

    frequence = numpy.fft.fft2(rgb_array)
    freq_shift = numpy.fft.fftshift(frequence)
    try:
        rows, cols = rgb_array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    crow, ccol = rows >> 1, cols >> 1

    # Remove the low frequency from the domain
    freq_shift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = numpy.fft.ifftshift(freq_shift)
    array_back = numpy.fft.ifft2(f_ishift)
    return numpy.abs(array_back)



def damped_oscillation(t):
    return damped_oscillation_c(t)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double damped_oscillation_c(double t)nogil:
    return exp(-t) * cos(6.283185307179586 * t)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef wobbly_array_c(rgb_array_, alpha_array_, double f):
    """
    Create a wobbly effect when f is varying over the time 
    
    :param rgb_array_: numpy.ndarray (w, h, 3) uint8 containing RGB values
    :param alpha_array_: numpy.ndarray (w, h) uint8 containing alpha values
    :param f: float, value controlling the wobbly effect exp(-f) * cos(6.283185307179586 * f)
    :return: a pygame.Surface with per-pixel information.
    """

    assert isinstance(rgb_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument rgb_array_ got %s ' % type(rgb_array_)
    assert isinstance(alpha_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument alpha_ got %s ' % type(alpha_array_)

    cdef int w, h
    try:
        w, h = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    if w == 0 or h == 0:
        raise ValueError('Array rgb_array_ has incorrect shapes.')
    try:
        w, h = (<object> alpha_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    if w == 0 or h == 0:
        raise ValueError('Array alpha_array_ has incorrect shapes.')

    cdef:
        unsigned char [:, :, :] rgb_array_c = rgb_array_
        unsigned char [:, ::1] alpha_array_c = alpha_array_
        unsigned char [: , :, ::1] new_array_c = zeros((h, w, 4), dtype=uint8, order='C')
        int i=0, j=0, ii=0
        double a, r, g, b

    with nogil:
        for i in prange(w):
            for j in range(h):
                a = exp(-f) * cos(6.283185307179586 * f)
                ii = <int>(i + a)
                if ii < 0:
                    ii = 0
                elif ii > w - 1:
                    ii = w
                r = rgb_array_c[ii, j, 0]
                g = rgb_array_c[ii, j, 1]
                b = rgb_array_c[ii, j, 2]
                
                new_array_c[j, i, 0], new_array_c[j, i, 1], \
                new_array_c[j, i, 2], new_array_c[j, i, 3] =  \
                    <unsigned char>r, <unsigned char>g,  \
                    <unsigned char>b, alpha_array_c[ii, j]

    return pygame.image.frombuffer(new_array_c, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef wobbly_surface_c(surface: Surface, double f):
    """
    Create a surface wobbly effect when f is varying over the time 
    
    :param surface: pygame.Surface, Surface 32-bit with per-pixel information  
    :param f: float, value controlling the wobbly effect exp(-f) * cos(6.283185307179586 * f)
    :return: a pygame.Surface with per-pixel information.
    """

    assert isinstance(surface, Surface), \
        'Expecting a Surface for positional argument surface got %s ' % type(surface)

    try:
        rgb_array_ = pixels3d(surface)
        alpha_array_= pixels_alpha(surface)
        
    except (pygame.error, ValueError) as e:
        raise ValueError('Incompatible pixel format.')

    cdef int w, h
    try:
        w, h = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    if w == 0 or h == 0:
        raise ValueError('Array rgb_array_ has incorrect shapes.')

    try:
        w, h = (<object> alpha_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    if w == 0 or h == 0:
        raise ValueError('Array alpha_array_ has incorrect shapes.')

    cdef:
        unsigned char [:, :, :] rgb_array_c = rgb_array_
        unsigned char [:, ::1] alpha_array_c = alpha_array_
        unsigned char [: , :, ::1] new_array_c = zeros((h, w, 4), dtype=uint8, order='C')
        int i=0, j=0, ii=0
        double a, r, g, b

    with nogil:
        for i in prange(w):
            for j in range(h):
                a = exp(-f) * cos(6.283185307179586 * f)
                ii = <int>(i + a)
                if ii < 0:
                    ii = 0
                elif ii > w - 1:
                    ii = w
                r = rgb_array_c[ii, j, 0]
                g = rgb_array_c[ii, j, 1]
                b = rgb_array_c[ii, j, 2]

                new_array_c[j, i, 0], new_array_c[j, i, 1], \
                new_array_c[j, i, 2], new_array_c[j, i, 3] =  \
                    <unsigned char>r, <unsigned char>g,  \
                    <unsigned char>b, alpha_array_c[ii, j]

    return pygame.image.frombuffer(new_array_c, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef swirl_surface_c(surface_:Surface, float degrees):
    """
    Distort an image by whirling 
    
    :param surface_: pygame.Surface compatible 8, 24-32 bit format 
    :param degrees: intensity effect
    :return: returns a numpy array (w, h, 3) uint 8
    """
    cdef int w, h
    w, h = surface_.get_size()
    
    try:
        rgb_array = pixels3d(surface_)
                                                       
    except (pygame.error, ValueError) as e:
        raise ValueError('Incompatible pixel format.')

    cdef:
        int i, j, diffx, diffy
        float columns, rows, r, angle
        unsigned char [:, :, :] rgb = rgb_array
        unsigned char [:, :, ::1] new_array = zeros((w, h, 3), uint8, order='C')
        
    columns = 0.5 * (float(w)  - 1.0)
    rows = 0.5 * (float(h) - 1.0)
    for i in range(w):
        for j in range(h):
            di = float(i) - columns
            dj = float(j) - rows
            r = sqrt(di * di + dj * dj)
            angle = degrees * r
            diffx = <int>(di * cos(angle) - dj * sin(angle) + columns)
            diffy = <int>(di * sin(angle) + dj * cos(angle) + rows)
            # may be better to say r > 1 then cannot display the pixel
            if (diffx >= 0) and (diffx < w) and \
               (diffy >= 0) and (diffy < h):
                new_array[i, j, 0], new_array[i, j, 1],\
                    new_array[i, j, 2] = rgb[diffx, diffy, 0], rgb[diffx, diffy, 1], rgb[diffx, diffy, 2]
    return asarray(new_array)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef swirl_surf2surf_c(surface_:Surface, float degrees):
    """
    Distort an image by whirling
    
    :param surface_: pygame.Surface with per-pixel 
    :param degrees: whirl intensity
    :return: Returns a pygame.Surface 32 bit format with per-pixel information
    """
    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_array = pixels3d(surface_)
        alpha_array = pixels_alpha(surface_)
                                                       
    except (pygame.error, ValueError) as e:
        raise ValueError('Incompatible pixel format.')

    cdef:
        int i, j, diffx, diffy
        float columns, rows, r, angle, di, dj

        unsigned char [:, :, :] rgb = rgb_array
        unsigned char [:, ::1] alpha = alpha_array
        unsigned char [:, :, ::1] new_array = zeros((h, w, 4), uint8, order='C')

    columns = 0.5 * (float(w)  - 1.0)
    rows = 0.5 * (float(h) - 1.0)
    with nogil:
        for i in prange(w):
            for j in range(h):
                di = float(i) - columns
                dj = float(j) - rows
                r = sqrt(di * di + dj * dj)
                angle = degrees * r
                diffx = <int>(di * cos(angle) - dj * sin(angle) + columns)
                diffy = <int>(di * sin(angle) + dj * cos(angle) + rows)
                # may be better to say r > 1 then cannot display the pixel
                if (diffx >= 0) and (diffx < w) and \
                   (diffy >= 0) and (diffy < h):
                    new_array[j, i, 0], new_array[j, i, 1],\
                        new_array[j, i, 2], new_array[j, i, 3] = rgb[diffx, diffy, 0], \
                        rgb[diffx, diffy, 1], rgb[diffx, diffy, 2], alpha[diffx, diffy]

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef light_area_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
                np.ndarray[np.uint8_t, ndim=2] mask_alpha):
    """
    This function gets both blocks RGB and ALPHA values that needs to be transfer
     to the light algorithm for processing.
    The block size varies according to the light position (e.g full size when the 
    light is away from the border).
    This is due to the fact that a part of the lightning effect will be hidden 
    (outside of the screen) and the full size RGB matrix
    will be reduce to match the area flooded by the light effect.
    This method uses numpy array to slice RGB and ALPHA blocks of data from the background matrix.
    Output surface will be a pygame surface with per-pixel information
    The final image image can also be blend to the background using pygame
     special flag : pygame.BLEND_RGB_ADD to intensify
    the light effect.
    
    :param x: integer, light x coordinates (must be in range [0..max screen.size x] 
    :param y: integer, light y coordinates (must be in range [0..max screen size y]
    :param background_rgb: numpy.ndarray (w, h, 3) uint8, contains all background RGB values.   
    :param mask_alpha: numpy.ndarray (w, h) uint8, Represent the mask alpha 
    (see radial texture for more information
    regarding the linear gradient effect).
    :return: Return a Surface representing the lighting effect for a given light position (x, y)  
    """
    cdef int w, h, lx, ly
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        lx, ly = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    lx = lx >> 1
    ly = ly >> 1
    

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((w, h), SRCALPHA)

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = empty((lx, ly, 3), uint8, order='C')
        np.ndarray[np.uint8_t, ndim=2] alpha = empty((lx, ly), uint8, order='C')
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    # Change the block size (RGB and ALPHA) if
    # the light position is close to an edge
    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    # RGB block and ALPHA
    rgb = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    return light_c(rgb, alpha, 0.0001, numpy.array([200, 186, 205], uint8))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef light_volume_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
                np.ndarray[np.uint8_t, ndim=2] mask_alpha, np.ndarray[np.uint8_t, ndim=3] volume, float magnitude):
    """
    This function gets both blocks RGB and ALPHA values that needs to be transfer 
    to the light algorithm for processing. The block size varies according to the 
    light position (e.g full size when the light is away from the border).
    This is due to the fact that a part of the lightning effect will be hidden 
    (outside of the screen) and the full size RGB matrix
    will be reduce to match the area flooded by the light effect.
    This method uses numpy array to slice RGB and ALPHA blocks of data from the background matrix.
    Output surface will be a pygame surface with per-pixel information
    The final image image can also be blend to the background using pygame 
    special flag : pygame.BLEND_RGB_ADD to intensify the light effect.
    
    :param x: integer, light x coordinates (must be in range [0..max screen.size x] 
    :param y: integer, light y coordinates (must be in range [0..max screen size y]
    :param background_rgb: numpy.ndarray (w, h, 3) uint8, contains all background RGB values.   
    :param mask_alpha: numpy.ndarray (w, h) uint8, Represent the mask alpha 
    (see radial texture for more information
    :param volume: numpy.ndarray (w, h, 3) uint8, 2d volumetric texture to use  
    :param magnitude: float, light intensity (default 1e-6)
    regarding the linear gradient effect).
    :return: Return a Surface representing the lighting effect for a given light position (x, y)  
    """
    cdef int w, h, lx, ly
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        lx, ly = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    lx = lx >> 1
    ly = ly >> 1
    
    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((w, h), SRCALPHA)

    emp = empty((lx, ly, 3), uint8)
    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = emp
        np.ndarray[np.uint8_t, ndim=2] alpha = emp[:]
        np.ndarray[np.uint8_t, ndim=3] volumetric = emp[:]
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    # Change the block size (RGB and ALPHA) if
    # the light position is close to an edge
    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    # RGB block and ALPHA
    rgb = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]
    volumetric = volume[lx - w_low:lx + w_high, ly - h_low:ly + h_high, :]
    return light_volumetric_c(rgb, alpha, magnitude, numpy.array([200, 186, 205], uint8), volumetric)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef light_c(unsigned char[:, :, :] rgb, unsigned char[:, :] alpha, float intensity, unsigned char [:] color):
    """
    Create a volumetric light effect whom shapes and radial intensity are provided by the radial mask texture. 
    This algorithm gives a better light effect illusion than a pure texture (radial mask) blend using pygame
    special flags pygame.BLEND_RGBA_ADD or BLEND_RGBA_MAX.
    Intensity value can be a constant or variable in order to create a flickering or a light attenuation effect.
    Color can also be adjusted overtime to create a dynamic light source.
    
    Both arrays RGB and ALPHA must have the same shapes in order to be merged/multiply together rgb(w, h, 3) and 
    alpha(w, h).
    :param rgb: numpy.ndarray (w, h, 3) uint8 representing the screen portion being flood with light (RGB values) 
    :param alpha: numpy.ndarray (w, h) uint8 represents the light radial mask, see texture for more details)
    :param intensity: float, Light intensity factor. When the light intensity factor is high, the resulting RGB 
    array will be flooded with saturated values (255 being the maximum) and the light color will have no effect.
    The resulting lighting will be a white circular glow with slight transparency on the outskirt of the radial mask.
    When the intensity value is around 1e-4 the light colors will be mixed with background RGB colors and multiply with 
    the radial mask to create the illusion of real lighting (volume of light).
    As expected when intensity value is 0.0, the resulting RGB array will be flooded with zeros and the light effect
    will be imperceptible. Therefore, to save time, when the intensity value is equal zero, an empty Surface 
    with per-pixel transparency (same size of the light effect) will be returned to avoid un-necessary computations.
    The light intensity will be greater in the center and decrease gradually in intensity toward the end of the 
    volume (linear gradient). Refer to the radial mask texture to see the shape of the light 
    and gradient linear slope. 
       
    :param color: 1d numpy.array of RGB colors (uint8), Light color RGB values (int in range [0..255])
    :return: Returns a Surface containing per-pixel alpha transparency.  
    """

    cdef int w, h
    try:
        w, h = (<object>alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((w, h), SRCALPHA)

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 4), uint8)
        int i=0, j=0
        int r, g, b
        float f
    with nogil:
        for i in prange(w):
            for j in range(h):
                f = alpha[i, j] * intensity
                r = min(<int>(rgb[i, j, 0] * f * color[0]), 255)
                g = min(<int>(rgb[i, j, 1] * f * color[1]), 255)
                b = min(<int>(rgb[i, j, 2] * f * color[2]), 255)

                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3] = r, g, b, alpha[i, j]

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')


#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef light_b(rgb_buffer_, alpha_buffer_, float intensity, unsigned char [:] color, int w, int h):
#
#     # return an empty Surface when intensity = 0.0
#     if intensity == 0.0:
#         return Surface((w, h), SRCALPHA)
#
#     cdef:
#         int b_length = w * h * 4
#         unsigned char [:] new_array = empty(b_length, uint8)
#         unsigned char [:] rgb_buffer = rgb_buffer_
#         unsigned char [:] alpha_buffer = alpha_buffer_
#         int i=0, r, g, b, a
#         float f = 0
#
#     with nogil:
#         ii =0
#         for i in range(0, b_length, 4):
#                 a += 4
#                 f = alpha_buffer[i] * intensity
#                 r = min(<int>(rgb_buffer[i + a] * f * color[0]), 255)
#                 g = min(<int>(rgb_buffer[i + a + 1] * f * color[1]), 255)
#                 b = min(<int>(rgb_buffer[i + a + 2] * f * color[2]), 255)
#                 new_array[i], new_array[i + a + 1], new_array[i + 2], new_array[i + 3] = r, g, b, alpha_buffer[i]
#
#     return pygame.image.frombuffer(new_array, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef light_volumetric_c(unsigned char[:, :, :] rgb, unsigned char[:, :] alpha,
                        float intensity, unsigned char [:] color, unsigned char[:, :, :] volume):
    """
    
    :param rgb: numpy.ndarray (w, h, 3) uint8, array containing all the background RGB colors values
    :param alpha: numpy.ndarray (w, h) uint8 represent the light mask alpha transparency
    :param intensity: float, light intensity default value for volumetric effect is 1e-6, adjust the value to have
    the right light illumination.
    :param color: numpy.ndarray, Light color (RGB values)
    :param volume: numpy.ndarray, array containing the 2d volumetric texture to merge with the background RGB values
    The texture should be slightly transparent with white shades colors. Texture with black nuances
    will increase opacity
    :return: Surface, Returns a surface representing a 2d light effect with a 2d volumetric
    effect display the radial mask.
    """

    # todo assert volume same size than alpha sizes
    cdef int w, h, vol_width, vol_height
    try:
        w, h = (<object>alpha).shape[:2]
        vol_width, vol_height = (<object>volume).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    assert (vol_width != w or vol_height != h), \
           'Assertion error, Alpha (w:%s, h:%s) and Volume (w:%s, h:%s) arrays shapes are not identical.' \
           % (w, h, vol_width, vol_height)
        

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((w, h), SRCALPHA)

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 4), uint8)

        int i=0, j=0
        int r, g, b
        float f
    with nogil:
        for i in prange(w):
            for j in range(h):
                f = alpha[i, j] * intensity             
                r = min(<int>(rgb[i, j, 0] * f * color[0] * volume[i, j, 0]), 255)
                g = min(<int>(rgb[i, j, 1] * f * color[1] * volume[i, j, 1]), 255)
                b = min(<int>(rgb[i, j, 2] * f * color[2] * volume[i, j, 2]), 255)

                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3] = r, g, b, alpha[i, j]

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef sepia24_c(surface_:Surface):
    """
    Create a sepia image from the given Surface (compatible with 8, 24-32 bit format image)
    Alpha channel will be ignored from image converted with the pygame method convert_alpha.
    
    :param surface_: Surface, loaded with pygame.image method
    :return: Return a Surface in sepia ready to be display, the final image will not hold any per-pixel
    transparency layer
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=uint8)
        int i=0, j=0
        int r, g, b
    with nogil:
        for i in prange(w):
            for j in range(h):
                r = <int>(rgb_array[i, j, 0] * 0.393 +
                          rgb_array[i, j, 1] * 0.769 + rgb_array[i, j, 2] * 0.189)
                g = <int>(rgb_array[i, j, 0] * 0.349 +
                          rgb_array[i, j, 1] * 0.686 + rgb_array[i, j, 2] * 0.168)
                b = <int>(rgb_array[i, j, 0] * 0.272 +
                          rgb_array[i, j, 1] * 0.534 + rgb_array[i, j, 2] * 0.131)
                if r > 255:
                    r = 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255

                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2], = r, g, b

    return pygame.image.frombuffer(new_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef sepia24_b(surface_:Surface):
    """
    Create a sepia image from the given Surface (compatible with 8, 24-32 bit format image)
    Alpha channel will be ignored from image converted with the pygame method convert_alpha.
    
    :param surface_: Surface, loaded with pygame.image method
    :return: Return a Surface in sepia ready to be display, the final image will not hold any per-pixel
    transparency layer
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_buffer = surface_.get_view('3')
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:] cbuffer = rgb_buffer
        int b_length = rgb_buffer.length
        int i=0
        int r, g, b
                                                     
    with nogil:
        for i in prange(0, b_length, 3):
                r = <int>(cbuffer[i] * 0.393 + cbuffer[i + 1] * 0.769 + cbuffer[i + 2] * 0.189)
                g = <int>(cbuffer[i] * 0.349 + cbuffer[i + 1] * 0.686 + cbuffer[i + 2] * 0.168)
                b = <int>(cbuffer[i] * 0.272 + cbuffer[i + 1] * 0.534 + cbuffer[i + 2] * 0.131)
                if r > 255:
                    r = 255
                if g > 255:
                    g = 255
                if b > 255:
                    b = 255
                cbuffer[i], cbuffer[i + 1], cbuffer[i + 2], = r, g, b

    return pygame.image.frombuffer(cbuffer, (w, h), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef sepia32_c(surface_:Surface):
    """
    Create a sepia image from the given Surface (compatible with 8, 24-32 bit with
    per-pixel information (image converted to convert_alpha())
    Surface converted to fast blit with pygame method convert() will raise a ValueError (Incompatible pixel format.).
     
    :param surface_:  Pygame.Surface converted with pygame method convert_alpha. Surface without per-pixel transparency
    will raise a ValueError (Incompatible pixel format.).
    :return: Returns a pygame.Surface (surface with per-pixel transparency)
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, ::1] alpha_array = alpha_
        int i=0, j=0
        int r, g, b
    with nogil:
        for i in prange(w):
            for j in range(h):
                r = <int>(rgb_array[i, j, 0] * 0.393 +
                          rgb_array[i, j, 1] * 0.769 + rgb_array[i, j, 2] * 0.189)
                g = <int>(rgb_array[i, j, 0] * 0.349 +
                          rgb_array[i, j, 1] * 0.686 + rgb_array[i, j, 2] * 0.168)
                b = <int>(rgb_array[i, j, 0] * 0.272 +
                          rgb_array[i, j, 1] * 0.534 + rgb_array[i, j, 2] * 0.131)

                if r > 255:
                   r = 255
                if g > 255:
                   g = 255
                if b > 255:
                   b = 255

                new_array[j, i, 0], new_array[j, i, 1], \
                new_array[j, i, 2], new_array[j, i, 3] = r, g, b, alpha_array[i, j]

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef sepia32_mask_c(surface_, mask):
    """
    Create a sepia image from the given Surface (compatible with 8, 24-32 bit with
    per-pixel information (image converted with convert_alpha() method)
    Surface converted to fast blit with pygame method convert() will raise a ValueError (Incompatible pixel format.).
    Optional mask argument (must be a black and white 2d numpy.ndarray filled with 0 and 255 unsigned char values) and
    shape (w, h)
     
    :param surface_:  Pygame.Surface converted with pygame method convert_alpha. Surface without per-pixel transparency
    will raise a ValueError (Incompatible pixel format.).
    :param mask: 2d numpy.ndarray filled with 0 and 255 values (black and white surface converted into a 2d array shape (w, h)
    :return: Returns a pygame.Surface (surface with per-pixel transparency)
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(mask, numpy.ndarray), \
           'Expecting mask for argument mask got %s ' % type(mask)

    cdef int w, h, dim, w2, h2, dim2
    w, h = surface_.get_size()

    try:
        w2, h2 = mask.shape
    except (ValueError, pygame.error) as e:
        raise ValueError('\nMask array not compatible.')

    assert w == w2 and h == h2, '\nSurface and mask size mismatch.'

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((h, w, 4), dtype=uint8)
        unsigned char [:, ::1] alpha_array = alpha_
        unsigned char [:, :] mask_array = mask
        int i=0, j=0
        int r, g, b
    with nogil:
        for i in prange(w):
            for j in range(h):
                if mask_array[i, j] > 0:
                         
                    r = <int>(rgb_array[i, j, 0] * 0.393 +
                              rgb_array[i, j, 1] * 0.769 + rgb_array[i, j, 2] * 0.189)
                    g = <int>(rgb_array[i, j, 0] * 0.349 +
                              rgb_array[i, j, 1] * 0.686 + rgb_array[i, j, 2] * 0.168)
                    b = <int>(rgb_array[i, j, 0] * 0.272 +
                              rgb_array[i, j, 1] * 0.534 + rgb_array[i, j, 2] * 0.131)

                    if r > 255:
                       r = 255
                    if g > 255:
                       g = 255
                    if b > 255:
                       b = 255

                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2], new_array[j, i, 3] = r, g, b, alpha_array[i, j]
                else:
                    new_array[j, i, 0] = rgb_array[i, j, 0]
                    new_array[j, i, 1] = rgb_array[i, j, 1]
                    new_array[j, i, 2] = rgb_array[i, j, 2]
                    new_array[j, i, 3] = alpha_array[i, j] 

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')

# TODO CHECK BELOW DID NOT HAVE TIME 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef sepia24_mask_c(surface_, mask):
    """
    Create a sepia image from the given Surface (compatible with 8, 24-32 bit).
     
    :param surface_:  Pygame.Surface 24-32 bit format pixel
    :param mask: 2d numpy.ndarray filled with 0 and 255 values (black and white surface converted into a 2d array shape (w, h)
    :return: Returns a pygame.Surface (24 bit surface without per-pixel transparency)
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)
    assert isinstance(mask, numpy.ndarray), \
           'Expecting mask for argument mask got %s ' % type(mask)

    cdef int w, h, dim, w2, h2, dim2
    w, h = surface_.get_size()

    try:
        w2, h2 = mask.shape
    except (ValueError, pygame.error) as e:
        raise ValueError('\nMask array not compatible.')

    assert w == w2 and h == h2, '\nSurface and mask size mismatch.'

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=uint8)
        unsigned char [:, :] mask_array = mask
        int i=0, j=0
        int r, g, b
    with nogil:
        for i in prange(w):
            for j in range(h):
                if mask_array[i, j] > 0:
                         
                    r = <int>(rgb_array[i, j, 0] * 0.393 +
                              rgb_array[i, j, 1] * 0.769 + rgb_array[i, j, 2] * 0.189)
                    g = <int>(rgb_array[i, j, 0] * 0.349 +
                              rgb_array[i, j, 1] * 0.686 + rgb_array[i, j, 2] * 0.168)
                    b = <int>(rgb_array[i, j, 0] * 0.272 +
                              rgb_array[i, j, 1] * 0.534 + rgb_array[i, j, 2] * 0.131)

                    if r > 255:
                       r = 255
                    if g > 255:
                       g = 255
                    if b > 255:
                       b = 255

                    new_array[j, i, 0], new_array[j, i, 1], \
                    new_array[j, i, 2] = r, g, b, 
                else:
                    new_array[j, i, 0] = rgb_array[i, j, 0]
                    new_array[j, i, 1] = rgb_array[i, j, 1]
                    new_array[j, i, 2] = rgb_array[i, j, 2]

    return pygame.image.frombuffer(new_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline plasma_c(int width, int height, int frame):
    """
    Plasma effect for windowed mode or fullscreen

    :param width: integer for screen width
    :param height: integer for screen height
    :param frame: integer, frame number or incrementing number
    :return: returns a numpy array shape (w, h, 3) numpy uint8 representing the plasma effect.
    """
    cdef:
        float xx, yy, t
        float h, s, v
        unsigned char [:, :, ::1] pix = zeros([width, height, 3], dtype=uint8)
        int i = 0, x, y
        float f, p, q, t_
        float hue, r, g, b

    t = float(frame)
    with nogil:
        for x in prange(width):
            for y in range(height):
                xx = float(x * HALF)
                yy = float(y * HALF)

                hue = 4 + sin((xx * ONE_FOURTH + t) * ONE_TWELVE) + sin((yy + t) * ONE_TENTH) \
                      + sin((xx* HALF + yy * HALF) * ONE_TENTH) + sin(sqrt(xx * xx + yy * yy + t) * ONE_TWELVE)
                h, s, v = hue * ONE_SIXTH, hue * ONE_SIXTH, hue * ONE_HEIGHT


                # hue =  sin((xx/4.0 + t) * 0.05) + sin((yy + t) * 0.05) \
                #          + sin((xx/2.0 + yy/2.0) * 0.05)
                # hue = hue + sin(sqrt(((xx - width/4) * (xx - width/4) + (yy - height/4)*(yy - height/4))+ t ) * 1/12)
                #
                # h, s, v = hue/4, hue * 1/5, hue/3

                # h = 0
                i = <int>(h * 6.0)
                f = (h * 6.0) - i
                p = v*(1.0 - s)
                q = v*(1.0 - s * f)
                t_ = v*(1.0 - s * (1.0 - f))
                i = i % 6

                if i == 0:
                    r, g, b =  v, t, p
                if i == 1:
                     r, g, b = q, v, p
                if i == 2:
                     r, g, b = p, v, t
                if i == 3:
                     r, g, b = p, q, v
                if i == 4:
                     r, g, b = t_, p, v
                if i == 5:
                     r, g, b = v, p, q

                if s == 0.0:
                     r, g, b = v, v, v

                pix[x, y, 0], pix[x, y, 1], pix[x, y, 2] = <unsigned char>(r * 255.0),\
                                                           <unsigned char>(g * 255.0),\
                                                           <unsigned char>(b * 255.0)
    # return the array
    return asarray(pix)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef color_reduction24_c(surface_:Surface, int factor):
    """
    http://help.corel.com/paintshop-pro/v20/main/en/documentation/index.html#page/
    Corel_PaintShop_Pro%2FUnderstanding_color_reduction.html%23ww998934

    Error Diffusion — replaces the original color of a pixel with the most similar color in the palette,
    but spreads the discrepancy between the original and new colors to the surrounding pixels.
    As it replaces a color (working from the top left to the bottom right of the image),
    it adds the “error,” or discrepancy, to the next pixel, before selecting the most similar color.
    This method produces a natural-looking image and often works well for photos or complex graphics.
    With the Error Diffusion method, you select the Floyd-Steinberg, Burkes, or Stucki algorithm for
    the dithering pattern.

    :param surface_: Surface 8, 24-32 bit format (alpha transparency will be ignored).
    :param factor: integer, Number of possible color 2^n
    :return : Returns an image with color reduction using error diffusion method.
    The final image will be stripped out of its alpha channel.
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        float [:, :, :] rgb_array = rgb_.astype(float32)
        unsigned char [:, :, ::1] reduce = zeros((h, w, 3), uint8, order='C')
        int x=0, y=0

        float new_red, new_green, new_blue
        float oldr, oldg, oldb

    with nogil:
        for y in prange(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * ONE_255) * (255.0 / factor)
                new_green = round(factor * oldg * ONE_255) * (255.0 / factor)
                new_blue = round(factor * oldb *ONE_255) * (255.0 / factor)

                # rgb_array[x, y, 0], rgb_array[x, y, 1], rgb_array[x, y, 2] = new_red, new_green, new_blue
                reduce[y, x, 0] = <unsigned char>new_red
                reduce[y, x, 1] = <unsigned char>new_green
                reduce[y, x, 2] = <unsigned char>new_blue
    return pygame.image.frombuffer(reduce, (w, h), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef color_reduction32_c(surface_:Surface, int factor):
    """
    http://help.corel.com/paintshop-pro/v20/main/en/documentation/index.html#page/
    Corel_PaintShop_Pro%2FUnderstanding_color_reduction.html%23ww998934

    Error Diffusion — replaces the original color of a pixel with the most similar color in the palette,
    but spreads the discrepancy between the original and new colors to the surrounding pixels.
    As it replaces a color (working from the top left to the bottom right of the image),
    it adds the “error,” or discrepancy, to the next pixel, before selecting the most similar color.
    This method produces a natural-looking image and often works well for photos or complex graphics.
    With the Error Diffusion method, you select the Floyd-Steinberg, Burkes, or Stucki algorithm for
    the dithering pattern.

    :param surface_: Surface 8, 24-32 bit format (with alpha transparency).
    :param factor: integer, Number of possible color 2^n
    :return : Returns an image with color reduction using error diffusion method.
    The final image will have per-pixel transparency 
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        float [:, :, :] rgb_array = rgb_.astype(float32)
        unsigned char [:, ::1] alpha = alpha_
        unsigned char [:, :, ::1] reduce = zeros((h, w, 4), uint8, order='C')
        int x=0, y=0

        float new_red, new_green, new_blue
        float oldr, oldg, oldb

    with nogil:
        for y in prange(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * ONE_255) * (255.0 / factor)
                new_green = round(factor * oldg * ONE_255) * (255.0 / factor)
                new_blue = round(factor * oldb *ONE_255) * (255.0 / factor)

                reduce[y, x, 0] = <unsigned char>new_red
                reduce[y, x, 1] = <unsigned char>new_green
                reduce[y, x, 2] = <unsigned char>new_blue
                reduce[y, x, 3] = alpha[x, y]
    return pygame.image.frombuffer(reduce, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dithering24_c(surface_:Surface, int factor):
    """

    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of the colors
    within it (see color vision). Dithered images, particularly those with relatively few colors,
    can often be distinguished by a characteristic graininess or speckled appearance.

    :param surface_: Surface 8,24-32 bit format
    :param factor : integer, factor for reducing the amount of colors. factor = 1 (2 colors), factor = 2, (8 colors)
    :return : a dithered Surface same format than original image (no alpha channel).

    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        float [:, :, :] rgb_array = rgb_.astype(float32)
        int x=0, y=0

        float new_red, new_green, new_blue
        float quant_error_red, quant_error_green, quant_error_blue
        float c1 = 7.0/16.0
        float c2 = 3.0/16.0
        float c3 = 5.0/16.0
        float c4 = 1.0/16.0
        float oldr, oldg, oldb

    with nogil:
        for y in range(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * ONE_255) * (255.0 / factor)
                new_green = round(factor * oldg * ONE_255) * (255.0 / factor)
                new_blue = round(factor * oldb * ONE_255) * (255.0 / factor)

                rgb_array[x, y, 0], rgb_array[x, y, 1], rgb_array[x, y, 2] = new_red, new_green, new_blue
                quant_error_red = oldr - new_red
                quant_error_green = oldg - new_green
                quant_error_blue = oldb - new_blue

                rgb_array[x + 1, y, 0] = rgb_array[x + 1, y, 0] + quant_error_red * c1
                rgb_array[x + 1, y, 1] = rgb_array[x + 1, y, 1] + quant_error_green * c1
                rgb_array[x + 1, y, 2] = rgb_array[x + 1, y, 2] + quant_error_blue * c1

                rgb_array[x - 1, y + 1, 0] = rgb_array[x - 1, y + 1, 0] + quant_error_red * c2
                rgb_array[x - 1, y + 1, 1] = rgb_array[x - 1, y + 1, 1] + quant_error_green * c2
                rgb_array[x - 1, y + 1, 2] = rgb_array[x - 1, y + 1, 2] + quant_error_blue * c2

                rgb_array[x, y + 1, 0] = rgb_array[x, y + 1, 0] + quant_error_red * c3
                rgb_array[x, y + 1, 1] = rgb_array[x, y + 1, 1] + quant_error_green * c3
                rgb_array[x, y + 1, 2] = rgb_array[x, y + 1, 2] + quant_error_blue * c3

                rgb_array[x + 1, y + 1, 0] = rgb_array[x + 1, y + 1, 0] + quant_error_red * c4
                rgb_array[x + 1, y + 1, 1] = rgb_array[x + 1, y + 1, 1] + quant_error_green * c4
                rgb_array[x + 1, y + 1, 2] = rgb_array[x + 1, y + 1, 2] + quant_error_blue * c4


    return pygame.surfarray.make_surface(asarray(rgb_array, dtype=uint8))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dithering32_c(surface_:Surface, int factor):
    """

    Dithering is used in computer graphics to create the illusion of "color depth" in images with
    a limited color palette - a technique also known as color quantization. In a dithered image,
    colors that are not available in the palette are approximated by a diffusion of colored pixels
    from within the available palette. The human eye perceives the diffusion as a mixture of the colors
    within it (see color vision). Dithered images, particularly those with relatively few colors,
    can often be distinguished by a characteristic graininess or speckled appearance.

    :param surface_: Surface 8,24-32 bit format with alpha transparency
    :param factor : integer, factor for reducing the amount of colors. factor = 1 (2 colors), factor = 2, (8 colors)
    :return : a dithered Surface same format than original image with alpha transparency.

    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible pixel format.')
    cdef:
        float [:, :, :] rgb_array = rgb_.astype(float32)
        float[:, :, ::1] rgba_array = zeros((h, w, 4), uint8)
        unsigned char [:, ::1] alpha = alpha_
        int x=0, y=0

        float new_red, new_green, new_blue
        float quant_error_red, quant_error_green, quant_error_blue
        float c1 = 7.0/16.0
        float c2 = 3.0/16.0
        float c3 = 5.0/16.0
        float c4 = 1.0/16.0
        float oldr, oldg, oldb

    with nogil:
        for y in range(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * ONE_255) * (255.0 / factor)
                new_green = round(factor * oldg * ONE_255) * (255.0 / factor)
                new_blue = round(factor * oldb * ONE_255) * (255.0 / factor)

                rgb_array[x, y, 0], rgb_array[x, y, 1], rgb_array[x, y, 2] = new_red, new_green, new_blue
                quant_error_red = oldr - new_red
                quant_error_green = oldg - new_green
                quant_error_blue = oldb - new_blue

                rgba_array[y + 1, x, 0] = rgb_array[x + 1, y, 0] + quant_error_red * c1
                rgba_array[y + 1, x, 1] = rgb_array[x + 1, y, 1] + quant_error_green * c1
                rgba_array[y + 1, x, 2] = rgb_array[x + 1, y, 2] + quant_error_blue * c1

                rgba_array[y - 1, x + 1, 0] = rgb_array[x - 1, y + 1, 0] + quant_error_red * c2
                rgba_array[y - 1, x + 1, 1] = rgb_array[x - 1, y + 1, 1] + quant_error_green * c2
                rgba_array[y - 1, x + 1, 2] = rgb_array[x - 1, y + 1, 2] + quant_error_blue * c2

                rgba_array[y, x + 1, 0] = rgb_array[x, y + 1, 0] + quant_error_red * c3
                rgba_array[y, x + 1, 1] = rgb_array[x, y + 1, 1] + quant_error_green * c3
                rgba_array[y, x + 1, 2] = rgb_array[x, y + 1, 2] + quant_error_blue * c3

                rgba_array[y + 1, x + 1, 0] = rgb_array[x + 1, y + 1, 0] + quant_error_red * c4
                rgba_array[y + 1, x + 1, 1] = rgb_array[x + 1, y + 1, 1] + quant_error_green * c4
                rgba_array[y + 1, x + 1, 2] = rgb_array[x + 1, y + 1, 2] + quant_error_blue * c4

                rgba_array[y, x, 3] = alpha[x, y]

    return pygame.image.frombuffer(asarray(rgba_array, dtype=uint8), (w, h), 'RGBA')


cdef double distance_ (double x1, double y1, double x2, double y2)nogil:

  return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));


cdef double gaussian_ (double v, double sigma)nogil:

  return (1.0 / (2.0 * 3.14159265358 * (sigma * sigma))) * exp(-(v * v ) / (2.0 * sigma * sigma))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bilateral_filter24_c(image: Surface, double sigma_s, double sigma_i):
    """
    A bilateral filter is a non-linear, edge-preserving, and noise-reducing
    smoothing filter for images. It replaces the intensity of each pixel with a
    weighted average of intensity values from nearby pixels. This weight can be
    based on a Gaussian distribution.
    
    :param image: Surface 8, 24-32 bit format (alpha channel will be ignored)
    :param sigma_s: float sigma_s : Spatial extent of the kernel, size of the considered neightborhood
    :param sigma_i: float sigma_i range kernel, minimum amplitude of an edge.
    :return: return a filtered Surface
    """
    # todo check sigma for division by zeros
    assert isinstance(image, Surface), \
        'Argument image must be a valid Surface, got %s ' % type(image)

    # texture sizes
    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] bilateral = zeros((h, w, 3), order='C', dtype=uint8)
        int x, y, xx, yy
        int k = 4
        int kx, ky
        double gs, wr, wg, wb, ir, ig, ib , wpr, wpg, wpb
        int w_1 = w - 1
        int h_1 = h - 1


    with nogil:

        for x in range(0, w_1):

            for y in range(0, h_1):

                ir, ig, ib = 0, 0, 0
                wpr, wpg, wpb = 0, 0, 0

                for ky in range(-k, k + 1):
                    for kx in range(-k, k + 1):

                        xx = x + kx
                        yy = y + ky

                        if xx < 0:
                            xx = 0
                        elif xx > w_1:
                            xx = w_1
                        if yy < 0:
                            yy = 0
                        elif yy > h_1:
                            yy = h_1
                        gs = gaussian_(distance_(xx, yy, x, y), sigma_s)

                        wr = gaussian_(rgb_array[xx, yy, 0] - rgb_array[x, y, 0], sigma_i) * gs
                        wg = gaussian_(rgb_array[xx, yy, 1] - rgb_array[x, y, 1], sigma_i) * gs
                        wb = gaussian_(rgb_array[xx, yy, 2] - rgb_array[x, y, 2], sigma_i) * gs
                        ir = ir + rgb_array[xx, yy, 0] * wr
                        ig = ig + rgb_array[xx, yy, 1] * wg
                        ib = ib + rgb_array[xx, yy, 2] * wb
                        wpr = wpr + wr
                        wpg = wpg + wg
                        wpb = wpb + wb
                ir = ir / wpr
                ig = ig / wpg
                ib = ib / wpb
                bilateral[y, x, 0], bilateral[y, x, 1], bilateral[y, x, 2] = \
                             int(round(ir)), int(round(ig)), int(round(ib))

    return pygame.image.frombuffer(bilateral, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bilateral_greyscale_c(path, double sigma_s, double sigma_i):
    """
    A bilateral filter is a non-linear, edge-preserving, and noise-reducing
    smoothing filter for images. It replaces the intensity of each pixel with a
    weighted average of intensity values from nearby pixels. This weight can be
    based on a Gaussian distribution.
    
    :param path: string; path to load the greyscale image with opencv
    :param sigma_s: float sigma_s : Spatial extent of the kernel, size of the considered neightborhood
    :param sigma_i: float sigma_i range kenerl, minimum amplitude of an edge.
    :return: return a filtered greyscale Surface
    """

    assert isinstance(path, str), \
        'Argument path must be a valid python string path name, got %s ' % type(path)
    try:
        # load image and create an array (w x h)
        greyscale_array = cv2.imread(path, 0)

    except Exception as e:
        print('\nCould not load the image %s ' % path)

    # texture sizes
    cdef int w, h
    w, h = greyscale_array.get_size()

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        unsigned char [:, :] rgb_array = greyscale_array
        unsigned char [:, ::1] bilateral = zeros((w, h), order='C', dtype=uint8)
        int x, y, xx, yy
        int k = 2
        int kx, ky
        int w_1 = w - 1, h_1 = h - 1
        double gs, ww, i , wp


    with nogil:
        for x in prange(0, w_1):
            for y in range(0, h_1):
                i=0
                wp=0
                for ky in range(-k, k + 1):
                    for kx in range(-k, k + 1):

                        xx = x + kx
                        yy = y + ky

                        if xx < 0:
                            xx = 0
                        elif xx > w_1:
                            xx = w_1
                        if yy < 0:
                            yy = 0
                        elif yy > h_1:
                            yy = h_1
                        gs = gaussian(distance(xx, yy, x, y), sigma_s)
                        ww = gaussian(rgb_array[xx, yy] - rgb_array[x, y], sigma_i) * gs
                        i = i + rgb_array[xx, yy] * ww
                        wp = wp + ww
                i = i / wp
                bilateral[y, x] = int(round(i))

    return pygame.surfarray.make_surface(asarray(bilateral))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef median_filter_greyscale_c(path, int kernel_size):
    """
    :param path: String, Path to the image to load with cv2
    :param kernel_size: Kernel width 
    """

    assert isinstance(path, str), \
            'Argument path must be a python string, got %s ' % type(path)

    try:
        greyscale_array = cv2.imread(path, 0)
    except:
        raise ValueError('Cannot open the image %s ' % path)

    cdef int w, h
    try:
        w, h = greyscale_array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        unsigned char [:, :] grey_array = greyscale_array.astype(dtype=uint8)
        unsigned char [:, :, ::1] median_array = zeros((h, w, 3), dtype=uint8, order='C')
        int i, j, ky, kx, jj, ii
        int k = kernel_size >> 1
        int k_size = kernel_size * kernel_size
        # unsigned char [::1] tmp = empty(k_size, dtype=uint8)
        int index = 0
        int *tmp = <int *> malloc(k_size * sizeof(int))
        int w_1 = w - 1
        int h_1 = h - 1

    with nogil:
        for i in range(0, w_1):
            for j in range(0, h - 1):
                index = 0
                for kx in range(-k, k + 1):
                    for ky in range(-k, k + 1):
                        ii = i + kx
                        jj = j + ky
                        # substitute the pixel is close to the edge.
                        # below zero pixel will be pixel[0], over w, pixel will
                        # be pixel[w]
                        if ii < 0:
                            ii = 0
                        elif ii > w_1:
                            ii = w_1
                        if jj < 0:
                            jj = 0
                        elif jj > h_1:
                            jj = h_1
                        # add values to the memoryviews
                        tmp[index] = grey_array[ii, jj]
                        index = index + 1
                # median value will be in the middle of the array
                tmp = quickSort(tmp, 0, k_size)
                median_array[jj, ii, 0] = tmp[k + 1]
                median_array[jj, ii, 1] = tmp[k + 1]
                median_array[jj, ii, 2] = tmp[k + 1]
    # free(tmp)
    return pygame.surfarray.make_surface(asarray(median_array))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef median_filter24_c(image, int kernel_size):
    """
    :param image: Surface 8. 24-32 bit 
    :param kernel_size: Kernel width 
    """

    assert isinstance(image, Surface), \
            'Argument image must be a valid Surface, got %s ' % type(image)

    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] median_array = zeros((w, h, 3), dtype=uint8, order='C')
        int i=0, j=0, ky, kx, ii=0, jj=0
        int k = kernel_size >> 1
        int k_size = kernel_size * kernel_size
        # int [:] tmp_red = zeros(kernel_size * kernel_size, dtype=int32)
        # int [:] tmp_green = zeros(kernel_size * kernel_size, dtype=int32)
        # int [:] tmp_blue = zeros(kernel_size * kernel_size, dtype=int32)
        int *tmp_red   = <int *> malloc(k_size * sizeof(int))
        int *tmp_green = <int *> malloc(k_size * sizeof(int))
        int *tmp_blue  = <int *> malloc(k_size * sizeof(int))
        int *tmpr
        int *tmpg
        int *tmpb
        int index = 0
        int w_1 = w - 1, h_1 = h - 1
    with nogil:
        for i in range(0, w_1):
            for j in range(0, h_1):
                index = 0
                # tmp_red[...] = 0
                # tmp_green[...] = 0
                # tmp_blue[...] = 0
                for kx in range(-k, k + 1):
                    for ky in range(-k, k + 1):
                        ii = i + kx
                        jj = j + ky
                        # substitute the pixel is close to the edge.
                        # below zero, pixel will be pixel[0], over w, pixel will
                        # be pixel[w]
                        if ii < 0:
                            ii = 0
                        elif ii > w_1:
                            ii = w_1

                        if jj < 0:
                            jj = 0
                        elif jj > h_1:
                            jj = h_1

                        # add values to the memoryviews
                        tmp_red[index] = rgb_array[ii, jj, 0]
                        tmp_green[index] = rgb_array[ii, jj, 1]
                        tmp_blue[index] = rgb_array[ii, jj, 2]
                        index = index + 1
                # median value will be in the middle of the array,
                # also equivalent to the value half_kernel + 1

                # numpy.quicksort (slow)
                # ** Don't forget to remove the gil if you go with
                # the numpy method
                # asarray(tmp_red).sort(kind='quicksort')
                # asarray(tmp_green).sort(kind='quicksort')
                # asarray(tmp_blue).sort(kind='quicksort')
                # median_array[i, j, 0]=tmp_red[k + 1]
                # median_array[i, j, 1]=tmp_green[k + 1]
                # median_array[i, j, 2]=tmp_blue[k + 1]

                # External C quicksort
                tmpr = quickSort(tmp_red, 0, k_size)
                tmpg = quickSort(tmp_green, 0, k_size)
                tmpb = quickSort(tmp_blue, 0, k_size)
                median_array[i, j, 0] = tmpr[k + 1]
                median_array[i, j, 1] = tmpg[k + 1]
                median_array[i, j, 2] = tmpb[k + 1]
    #free(tmp_red)
    #free(tmp_green)
    #free(tmp_blue)
    return pygame.surfarray.make_surface(asarray(median_array))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef median_filter32_c(image, int kernel_size):
    """
    :param image: Surface 8. 24-32 bit with per-pixel transparency
    :param kernel_size: Kernel width e.g 3 for a matrix 3 x 3
    :return: Returns a Surface with no alpha channel
    """

    assert isinstance(image, Surface), \
            'Argument image must be a valid Surface, got %s ' % type(image)

    cdef int w, h
    w, h = image.get_size()

    try:
        array_ = pixels3d(image)
        alpha_ = pixels_alpha(image)

    except (pygame.error, ValueError):
        raise ValueError('\nTexture/image is not compatible.')

    assert w != 0 or h !=0,\
            'image with incorrect dimensions (w>0, h>0) got (%s, %s) ' % (w, h)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :] alpha = alpha_
        unsigned char [:, :, ::1] median_array = zeros((h, w, 4), dtype=uint8, order='C')
        int i=0, j=0, ky, kx, ii=0, jj=0
        int k = kernel_size >> 1
        int k_size = kernel_size * kernel_size
        # int [:] tmp_red = zeros(kernel_size * kernel_size, dtype=int32)
        # int [:] tmp_green = zeros(kernel_size * kernel_size, dtype=int32)
        # int [:] tmp_blue = numpy.zerosempty(kernel_size * kernel_size, dtype=int32)

        int *tmp_red   = <int *> malloc(k_size * sizeof(int))
        int *tmp_green = <int *> malloc(k_size * sizeof(int))
        int *tmp_blue  = <int *> malloc(k_size * sizeof(int))
        int *tmpr
        int *tmpg
        int *tmpb
        int index = 0
        int w_1 = w - 1, h_1 = h - 1

    # TODO make two passes horizontal and vertical
    with nogil:
        for i in range(0, w_1):
            for j in range(0, h_1):
                index = 0
                # tmp_red[...] = 0
                # tmp_green[...] = 0
                # tmp_blue[...] = 0
                for kx in range(-k, k + 1):
                    for ky in range(-k, k + 1):
                        ii = i + kx
                        jj = j + ky
                        # substitute the pixel is close to the edge.
                        # below zero, pixel will be pixel[0], over w, pixel will
                        # be pixel[w]
                        if ii < 0:
                            ii = 0
                        elif ii > w_1:
                            ii = w_1

                        if jj < 0:
                            jj = 0
                        elif jj > h_1:
                            jj = h_1

                        # add values to the memoryviews
                        tmp_red[index] = rgb_array[ii, jj, 0]
                        tmp_green[index] = rgb_array[ii, jj, 1]
                        tmp_blue[index] = rgb_array[ii, jj, 2]
                        index = index + 1
                # median value will be in the middle of the array,
                # also equivalent to the value half_kernel + 1

                # numpy.quicksort (slow)
                # asarray(tmp_red).sort(kind='quicksort')
                # asarray(tmp_green).sort(kind='quicksort')
                # asarray(tmp_blue).sort(kind='quicksort')
                # median_array[i, j, 0]=tmp_red[k + 1]
                # median_array[i, j, 1]=tmp_green[k + 1]
                # median_array[i, j, 2]=tmp_blue[k + 1]

                # External C quicksort
                tmpr = quickSort(tmp_red, 0, k_size)
                tmpg = quickSort(tmp_green, 0, k_size)
                tmpb = quickSort(tmp_blue, 0, k_size)
                median_array[j, i, 0] = tmpr[k + 1]
                median_array[j, i, 1] = tmpg[k + 1]
                median_array[j, i, 2] = tmpb[k + 1]
                median_array[j, i, 3] = alpha[i, j]

    #free(tmp_red)
    #free(tmp_green)
    #free(tmp_blue)
    return pygame.image.frombuffer(median_array, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef unsigned char[:, :, :] array_rgba2bgra_c(unsigned char[:, :, :] rgba_array):
    """
    Convert an RGBA color array into an BGRA array
    
    :param rgba_array: numpy.ndarray (w, h, 4) uint8 with RGBA values to convert into BGRA
    :return: Return a numpy.ndarray (w, h, 4) with BGRA values (uint8)
    """
    cdef int w, h
    try:
        w, h = (<object>rgba_array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    cdef:
        int i, j
        unsigned char [:, :, ::1] bgra_array = empty((w, h, 4), uint8)
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                bgra_array[i, j, 0] = rgba_array[i, j, 2]
                bgra_array[i, j, 1] = rgba_array[i, j, 1]
                bgra_array[i, j, 2] = rgba_array[i, j, 0]
                bgra_array[i, j, 3] = rgba_array[i, j, 3]
    return bgra_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef unsigned char[:, :, :] array_rgb2bgr_c(unsigned char[:, :, :] rgb_array):
    """
    Convert an RGB color array into an BGR array
    
    :param rgb_array: numpy.ndarray (w, h, 3) uint8 with RGB values to convert into BGR
    :return: Return a numpy.ndarray (w, h, 3) with BGR values (uint8)
    """
    cdef int w, h
    w, h = (<object>rgb_array).shape[:2]

    cdef:
        int i, j
        unsigned char [:, :, ::1] bgr_array = empty((w, h, 4), uint8)
        
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                bgr_array[i, j, 0] = rgb_array[i, j, 2]
                bgr_array[i, j, 1] = rgb_array[i, j, 1]
                bgr_array[i, j, 2] = rgb_array[i, j, 0]
                bgr_array[i, j, 4] = rgb_array[i, j, 4]
    return bgr_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef unsigned char[:, :, :] array_bgra2rgba_c(unsigned char[:, :, :] bgra_array):
    """
    Convert an BGRA color array into an RGBA
    
    :param bgra_array: numpy.ndarray (w, h, 4) uint8 with value BGRA to convert into RGBA
    :return: numpy.ndarray (w, h, 4) with RGBA values (uint8)
    """
    cdef int w, h
    
    try:
        w, h = (<object>bgra_array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        int i, j
        unsigned char [:, :, ::1] rgba_array = empty((w, h, 4), uint8)
        
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                rgba_array[i, j, 0] = bgra_array[i, j, 2]   # red
                rgba_array[i, j, 1] = bgra_array[i, j, 1]   # green
                rgba_array[i, j, 2] = bgra_array[i, j, 0]   # blue
                rgba_array[i, j, 3] = bgra_array[i, j, 3]   # alpha
    return rgba_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef unsigned char[:, :, :] array_bgr2rgb_c(unsigned char[:, :, :] bgr_array):
    """
    Convert an BGR color array into an RGB
    
    :param bgr_array: numpy.ndarray (w, h, 3) uint8 with value B, G, R to convert into RGB
    :return: numpy.ndarray (w, h, 3) with RGB values (uint8)
    """
    cdef int w, h
    w, h = (<object>bgr_array).shape[:2]

    cdef:
        int i, j
        unsigned char [:, :, ::1] rgb_array = empty((w, h, 3), uint8)
    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                rgb_array[i, j, 0] = bgr_array[i, j, 2]   # red
                rgb_array[i, j, 1] = bgr_array[i, j, 1]   # green
                rgb_array[i, j, 2] = bgr_array[i, j, 0]   # blue
    return rgb_array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef buffer_rgba_to_bgra_c(buffer_):
    """
    Convert a RGBA pygame bufferproxy into an BGRA equivalent model.
    :param buffer_: pygame.proxybuffer, represent an image buffer with pixels encoded in
    RGBA format
    buffer_ must be a bytes string format with uint8 values.
    To create a buffer from a pygame surface, use the method get_view('2') to extract
    RGBA values.
    :return : Returns a memoryslice of a BGRA pixel buffer, you can convert the buffer
    into a pygame surface using the method pygame.image.frombuffer(memoryslice, (width, height), 'RGBA')
    """
    
    assert isinstance(buffer_, pygame.bufferproxy), \
        'Expecting bufferproxy for argument buffer_ got %s ' % type(buffer_)

    cdef int b_length = buffer_.length

    assert b_length > 0, "bytes buffer_cannot be length zero."
        
    cdef:
        unsigned char [:] rgba_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] bgra_buffer = numpy.empty(buffer_.length, dtype=numpy.uint8)
        int i = 0          

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            bgra_buffer[i]     = rgba_buffer[i + 2]   
            bgra_buffer[i + 1] = rgba_buffer[i + 1]
            bgra_buffer[i + 2] = rgba_buffer[i]
            bgra_buffer[i + 3] = rgba_buffer[i + 3]
    return bgra_buffer

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef buffer_bgra_to_rgba_c(buffer_):
    """
    Convert a BGRA pygame bufferproxy into an RGBA equivalent model.
    :param buffer_: pygame.proxybuffer, represent an image buffer of raw pixels encoded in
    BGRA format.
    buffer_ must be a bytes string format with uint8 values.
    To create a buffer from a pygame surface, use the method get_view('2') to extract
    RGBA values.
    :return : Returns a memoryslice of a RGBA pixel buffer, you can convert the buffer
    into a pygame surface using the method pygame.image.frombuffer(memoryslice, (width, height), 'RGBA')
    """
    
    assert isinstance(buffer_, pygame.bufferproxy), \
        'Expecting bufferproxy for argument buffer_ got %s ' % type(buffer_)

    cdef:
        int b_length = buffer_.length
    assert b_length > 0, "bytes buffer_cannot be length zero."
        
    cdef:
        unsigned char [:] bgra_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] rgba_buffer = numpy.empty(buffer_.length, dtype=numpy.uint8)
        int i = 0          

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            rgba_buffer[i + 2]  = bgra_buffer[i]    
            rgba_buffer[i + 1] = bgra_buffer[i + 1] 
            rgba_buffer[i] = bgra_buffer[i + 2] 
            rgba_buffer[i + 3] = bgra_buffer[i + 3] 
    return rgba_buffer


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef buffer_rgb_to_bgr_c(buffer_):
    """
    Convert a RGB pygame bufferproxy into an BGR equivalent model.
    :param buffer_: pygame.proxybuffer, represent an image buffer with pixels encoded in
    RGB format. To create a buffer from a pygame surface, use the method get_view('3') to extract
    only RGB values.
    buffer_ must be a bytes string format with uint8 values.
    :return : Returns a memoryslice of a BGR pixel buffer, you can convert the buffer
    into a pygame surface using the method pygame.image.frombuffer(memoryslice, (width, height), 'RGB')
    """
    
    assert isinstance(buffer_, pygame.bufferproxy), \
        'Expecting bufferproxy for argument buffer_ got %s ' % type(buffer_)

    cdef:
        int b_length = buffer_.length

    assert b_length > 0, "bytes buffer_cannot be length zero."
        
    cdef:
        unsigned char [:] rgba_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] bgra_buffer = numpy.empty(buffer_.length, dtype=numpy.uint8)
        int i = 0          

    with nogil:
        for i in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            bgra_buffer[i]     = rgba_buffer[i + 2]   
            bgra_buffer[i + 1] = rgba_buffer[i + 1]
            bgra_buffer[i + 2] = rgba_buffer[i]           
    return bgra_buffer

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef buffer_bgr_to_rgb_c(buffer_):
    """
    Convert a BGR pygame bufferproxy into an RGB equivalent model.
    :param buffer_: pygame.proxybuffer, represent an image buffer of raw pixels encoded in
    BGR format.
    To create a buffer from a pygame surface, use the method get_view('3') to extract
    only RGB values.
    buffer_ must be a bytes string format with uint8 values
    :return : Returns a memoryslice of a RGB pixel buffer, you can convert the buffer
    into a pygame surface using the method pygame.image.frombuffer(memoryslice, (width, height), 'RGB')
    """
    
    assert isinstance(buffer_, pygame.bufferproxy), \
        'Expecting bufferproxy for argument buffer_ got %s ' % type(buffer_)
    cdef:
        int b_length = buffer_.length

    assert b_length > 0, "bytes buffer_cannot be length zero."
        
    cdef:
        unsigned char [:] bgra_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [:] rgba_buffer = numpy.empty(buffer_.length, dtype=numpy.uint8)
        int i = 0          

    with nogil:
        for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            rgba_buffer[i + 2]  = bgra_buffer[i]    
            rgba_buffer[i + 1] = bgra_buffer[i + 1] 
            rgba_buffer[i] = bgra_buffer[i + 2] 
    return rgba_buffer

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef surface_rescale_c(surface, int w2, int h2):
    """
    Rescale a given surface 
    
    :param surface: pygame.Surface to rescale, compatible 24-32 bit surface. 
        width and height must be >0 otherwise raise a value error.  
    :param w2: width for new surface
    :param h2: height for new surface 
    :return: return a rescale pygame.Surface 24-bit without per-pixel 
        transparency, dimensions (w2, h2). 
    """

    assert isinstance(surface, pygame.Surface),\
        '\nPositional argument surface must be a pygame.Surface type, got %s ' % type(surface)
    assert isinstance(w2, int), 'Argument w2 must be an integer got %s ' % type(w2)
    assert isinstance(h2, int), 'Argument h2 must be an integer got %s ' % type(h2)
    assert (w2>0 and h2>0), \
        "\nIncorrect value(s) for w2, h2 (w2>0, h2>0), got (w2:%s, h2:%s) " % (w2, h2)

    try:
        array = pygame.surfarray.pixels3d(surface)
        
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')
    
    cdef int w1, h1
    w1, h1 = surface.get_size()
    
    assert w1!=0 and h1!=0, '\nIncompatible surface dimensions (w>0, h>0) got (w:%s, h:%s) ' % (w1, h1)
    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((w2, h2, 3), numpy.uint8)
        unsigned char [:, :, :] rgb_array = array
        float fx = float(w1) / float(w2)
        float fy = float(h1) / float(h2)
        int x, y, xx, yy
        
    with nogil:
        for x in prange(w2, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for y in range(h2):
                xx = <int>(x * fx)
                yy = <int>(y * fy)
                new_array[x, y, 0] = rgb_array[xx, yy, 0]
                new_array[x, y, 1] = rgb_array[xx, yy, 1]
                new_array[x, y, 2] = rgb_array[xx, yy, 2]
    return pygame.surfarray.make_surface(numpy.asarray(new_array))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef surface_rescale_alphac(surface, int w2, int h2):
    """
    Rescale a given surface containing per-pixel alpha transparency
    
    :param surface: pygame.Surface to rescale, compatible 32 bit surface containing per-pixel information. 
    :param w2: new width
    :param h2: new height  
    :return: return a rescale pygame.Surface 32-bit with per-pixel information 
    """

    assert isinstance(surface, pygame.Surface),\
        '\nPositional argument surface must be a pygame.Surface type, got %s ' % type(surface)
    assert isinstance(w2, int), 'Argument w2 must be an integer got %s ' % type(w2)
    assert isinstance(h2, int), 'Argument h2 must be an integer got %s ' % type(h2)
    assert (w2>0 and h2>0), \
        "\nIncorrect value(s) for w2, h2 (w2>0, h2>0), got (w2:%s, h2:%s) " % (w2, h2)

    try:
        array_ = pygame.surfarray.pixels3d(surface)
        alpha_ = pygame.surfarray.pixels_alpha(surface)
        
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible pixel format.')

    cdef int w1, h1
    w1, h1 = surface.get_size()
    
    assert w1!=0 and h1!=0, '\nIncompatible surface dimensions (w>0, h>0) got (w:%s, h:%s) ' % (w1, h1)
    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((h2, w2, 4), numpy.uint8)
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, ::1] alpha = alpha_
        float fx = float(w1) / float(w2)
        float fy = float(h1) / float(h2)
        int x, y, xx, yy
    with nogil:
        for x in prange(w2, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for y in range(h2):
                xx = <int>(x * fx)
                yy = <int>(y * fy)
                new_array[y, x, 0] = rgb_array[xx, yy, 0]
                new_array[y, x, 1] = rgb_array[xx, yy, 1]
                new_array[y, x, 2] = rgb_array[xx, yy, 2]
                new_array[y, x, 3] = alpha[xx, yy]

    return pygame.image.frombuffer(new_array, (w2, h2), 'RGBA')




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef array_rescale_24c(np.ndarray[np.uint8_t, ndim=3] array, int w2, int h2):
    """
    Rescale a 24-bit format image from its given array 
    
    :param array: RGB numpy.ndarray, format (w, h, 3) numpy.uint8
    :param w2: new width 
    :param h2: new height
    :return: Return a pygame surface (format 24-bit without alpha channel) 
    """

    assert isinstance(array, numpy.ndarray),\
        '\nPositional argument array must be a numpy.ndarray type, got %s ' % type(array)
    assert isinstance(w2, int), 'Argument w2 must be an integer got %s ' % type(w2)
    assert isinstance(h2, int), 'Argument h2 must be an integer got %s ' % type(h2)
    assert (w2>0 and h2>0), \
        "\nIncorrect value(s) for w2, h2 (w2>0, h2>0), got (w2:%s, h2:%s) " % (w2, h2)

    cdef int w1, h1, s
    try:
        w1, h1, s = (<object>array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    assert w1!=0 and h1!=0 and s==3,\
        '\nIncompatible array dimensions (w>0, h>0, s=3) got (w:%s, h:%s, %s) ' % (w1, h1, s)
    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((w2, h2, 3), numpy.uint8)
        unsigned char [:, :, :] rgb_array = array
        float fx = float(w1) / float(w2)
        float fy = float(h1) / float(h2)
        int x, y, xx, yy
    with nogil:
        for x in prange(w2, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for y in range(h2):
                xx = <int>(x * fx)
                yy = <int>(y * fy)
                new_array[x, y, 0] = rgb_array[xx, yy, 0]
                new_array[x, y, 1] = rgb_array[xx, yy, 1]
                new_array[x, y, 2] = rgb_array[xx, yy, 2]

    return pygame.surfarray.make_surface(numpy.asarray(new_array))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef array_rescale_32c(np.ndarray[np.uint8_t, ndim=4] array, int w2, int h2):
    """
    Rescale a 32-bit format image from its given array 
    
    :param array: RGB numpy.ndarray, format (w, h, 4) numpy.uint8 with alpha channel
    :param w2: new width 
    :param h2: new height
    :return: Return a pygame surface (format 32-bit wit alpha channel) 
    """

    assert isinstance(array, numpy.ndarray),\
        '\nPositional argument array must be a numpy.ndarray type, got %s ' % type(array)
    assert isinstance(w2, int), 'Argument w2 must be an integer got %s ' % type(w2)
    assert isinstance(h2, int), 'Argument h2 must be an integer got %s ' % type(h2)
    assert (w2>0 and h2>0), \
        "\nIncorrect value(s) for w2, h2 (w2>0, h2>0), got (w2:%s, h2:%s) " % (w2, h2)

    cdef int w1, h1, s
    try:
        w1, h1, s = (<object>array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    assert w1!=0 and h1!=0 and s==4, \
        '\nIncompatible array dimensions (w>0, h>0, 4) got (w:%s, h:%s, %s) ' % (w1, h1, s)
    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((h2, w2, 4), numpy.uint8)
        unsigned char [:, :, :] rgb_array = array
        float fx = float(w1) / float(w2)
        float fy = float(h1) / float(h2)
        int x, y, xx, yy
    with nogil:
        for x in prange(w2, schedule=SCHEDULE, num_threads=THREAD_NUMBER): 
            for y in range(h2):
                xx = <int>(x * fx)
                yy = <int>(y * fy)
                new_array[y, x, 0] = rgb_array[xx, yy, 0]
                new_array[y, x, 1] = rgb_array[xx, yy, 1]
                new_array[y, x, 2] = rgb_array[xx, yy, 2]
                new_array[y, x, 3] = rgb_array[xx, yy, 3]

    return pygame.image.frombuffer(new_array, (w2, h2), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef zoomx2_bilineare_alphac(surface):
    """
    Scale x2 a given pygame.Surface using bilineaire approximation
    compatible with 32 bit format image or surface converted with convert_alpha method
    Final image is a 32bit format image with per-pixel transparency

    :param surface: pygame.Surface with alpha transparency (image must contains
    per-pixel alpha transparency otherwise raise a value error 'Incompatible image')
    :return : pygame.Surface, Returns a Surface with size x2 containing per-pixel alpha transparency.

    """


    assert isinstance(surface, pygame.Surface), \
           'Positional argument surface must be a pygame.Surface got %s ' % type(surface)


    try:
        rgb = pygame.surfarray.pixels3d(surface)
        alpha = pygame.surfarray.pixels_alpha(surface)
        
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef int w, h
    # determine size of the input array
    try:
        w, h = rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w!=0 and h!=0, 'Incompatible surface got (w:%s, h:%s) ' % (w, h)

    cdef:
        int i = 0, j = 0, row, col, row1, \
            col1, row2, col2, rr, cc, rr2, cc2, rr1, cc1
        int w2 = w * 2, h2 = h * 2
        unsigned char [:, :, ::1] out_array = numpy.zeros((h2, w2, 4), dtype=numpy.uint8)
        unsigned char [:, :, :] rgb_array = rgb
        unsigned char [:, ::1] alpha_array = alpha

    with nogil:
        for row in prange(w - 1):
            for col in range(h - 1):
                col1 = col + 1
                row1 = row + 1
                row2 = row + 2
                col2 = col + 2
                rr = row * 2
                cc = col * 2
                rr2 = rr + 2
                cc2 = cc + 2
                rr1 = rr + 1
                cc1 = cc + 1
                # Place the 4 known values
                # (topleft, topright, bottom left, bottom right)
                out_array[cc, rr, 0] = rgb_array[row, col, 0]
                out_array[cc, rr, 1] = rgb_array[row, col, 1]
                out_array[cc, rr, 2] = rgb_array[row, col, 2]
                out_array[cc, rr, 3] = alpha_array[row, col]
                # top right
                out_array[cc2, rr, 0] = rgb_array[row, col1, 0]
                out_array[cc2, rr, 1] = rgb_array[row, col1, 1]
                out_array[cc2, rr, 2] = rgb_array[row, col1, 2]
                out_array[cc2, rr, 3] = alpha_array[row, col1]
                # bottom left
                out_array[cc, rr2, 0] = rgb_array[row1, col, 0]
                out_array[cc, rr2, 1] = rgb_array[row1, col, 1]
                out_array[cc, rr2, 2] = rgb_array[row1, col, 2]
                out_array[cc, rr2, 3] = alpha_array[row1, col]
                # bottom right
                out_array[cc2, rr2, 0] = rgb_array[row1, col1, 0]
                out_array[cc2, rr2, 1] = rgb_array[row1, col1, 1]
                out_array[cc2, rr2, 2] = rgb_array[row1, col1, 2]
                out_array[cc2, rr2, 3] = alpha_array[row1, col1]
                if row == 0:
                    # top value
                    out_array[cc1, rr, 0] = (rgb_array[row, col1, 0] + rgb_array[row, col, 0]) >> 1
                    out_array[cc1, rr, 1] = (rgb_array[row, col1, 1] + rgb_array[row, col, 1]) >> 1
                    out_array[cc1, rr, 2] = (rgb_array[row, col1, 2] + rgb_array[row, col, 2]) >> 1
                    out_array[cc1, rr, 3] = (alpha_array[row, col1] + alpha_array[row, col]) >> 1
                if col == 0:
                    # left value
                    out_array[cc, rr1, 0] = (rgb_array[row1, col, 0] + rgb_array[row, col, 0]) >> 1
                    out_array[cc, rr1, 1] = (rgb_array[row1, col, 1] + rgb_array[row, col, 1]) >> 1
                    out_array[cc, rr1, 2] = (rgb_array[row1, col, 2] + rgb_array[row, col, 2]) >> 1
                    out_array[cc, rr1, 3] = (alpha_array[row1, col] + alpha_array[row, col]) >> 1

                # bottom value
                out_array[cc1, rr2, 0] = (rgb_array[row1, col1, 0] + rgb_array[row1, col, 0]) >> 1
                out_array[cc1, rr2, 1] = (rgb_array[row1, col1, 1] + rgb_array[row1, col, 1]) >> 1
                out_array[cc1, rr2, 2] = (rgb_array[row1, col1, 2] + rgb_array[row1, col, 2]) >> 1
                out_array[cc1, rr2, 3] = (alpha_array[row1, col1] + alpha_array[row1, col]) >> 1

                # right value
                out_array[cc2, rr1, 0] = (rgb_array[row1, col1, 0] + rgb_array[row, col1, 0]) >> 1
                out_array[cc2, rr1, 1] = (rgb_array[row1, col1, 1] + rgb_array[row, col1, 1]) >> 1
                out_array[cc2, rr1, 2] = (rgb_array[row1, col1, 2] + rgb_array[row, col1, 2]) >> 1
                out_array[cc2, rr1, 3] = (alpha_array[row1, col1] + alpha_array[row, col1]) >> 1

                # mid value
                # mid value
                out_array[cc1, rr1, 0] = (out_array[cc2, rr1, 0] + out_array[cc, rr1, 0]) >> 1
                out_array[cc1, rr1, 1] = (out_array[cc2, rr1, 1] + out_array[cc, rr1, 1]) >> 1
                out_array[cc1, rr1, 2] = (out_array[cc2, rr1, 2] + out_array[cc, rr1, 2]) >> 1
                out_array[cc1, rr1, 3] = (alpha_array[row1, col2] + alpha_array[row1, col]) >> 1


    return pygame.image.frombuffer(out_array, (w2, h2), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef zoomx2_bilineare_c(surface):
    """
    Scale x2 a given pygame.Surface using bilineaire approximation
    Compatible with image 24 - 32bit format with or without per-pixel transparency.
    The final image will be stripped of the alpha channel, returns a 24 bit image.

    :param surface: pygame.Surface 24 -32 bit format image with or without alpha transparency.
    per-pixel alpha transparency otherwise raise a value error 'Incompatible image')
    :return : pygame.Surface, Returns a pygame Surface 24 bit (size x2) without transparency

    """

    assert isinstance(surface, pygame.Surface), \
           'Positional argument surface must be a pygame.Surface got %s ' % type(surface)

    try:
        rgb = pygame.surfarray.pixels3d(surface)
        
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef int w, h
    # determine size of the input array
    try:
        w, h = rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w!=0 and h!=0, 'Incompatible surface got (w:%s, h:%s) ' % (w, h)

    cdef:
        int i = 0, j = 0, row, col, row1, \
            col1, row2, col2, rr, cc, rr2, cc2, rr1, cc1
        int w2 = w * 2, h2 = h * 2
        unsigned char [:, :, ::1] out_array = numpy.zeros((h2, w2, 3), dtype=numpy.uint8)
        unsigned char [:, :, :] rgb_array = rgb

    with nogil:
        for row in prange(w - 1):
            for col in range(h - 1):
                col1 = col + 1
                row1 = row + 1
                row2 = row + 2
                col2 = col + 2
                rr = row * 2
                cc = col * 2
                rr2 = rr + 2
                cc2 = cc + 2
                rr1 = rr + 1
                cc1 = cc + 1
                # Place the 4 known values (topleft, topright, bottom left, bottom right)
                # Top left
                out_array[cc, rr, 0] = rgb_array[row, col, 0]
                out_array[cc, rr, 1] = rgb_array[row, col, 1]
                out_array[cc, rr, 2] = rgb_array[row, col, 2]
                # top right
                out_array[cc2, rr, 0] = rgb_array[row, col1, 0]
                out_array[cc2, rr, 1] = rgb_array[row, col1, 1]
                out_array[cc2, rr, 2] = rgb_array[row, col1, 2]
                # bottom left
                out_array[cc, rr2, 0] = rgb_array[row1, col, 0]
                out_array[cc, rr2, 1] = rgb_array[row1, col, 1]
                out_array[cc, rr2, 2] = rgb_array[row1, col, 2]
                # bottom right
                out_array[cc2, rr2, 0] = rgb_array[row1, col1, 0]
                out_array[cc2, rr2, 1] = rgb_array[row1, col1, 1]
                out_array[cc2, rr2, 2] = rgb_array[row1, col1, 2]
                if row == 0:
                    # top value
                    out_array[cc1, rr, 0] = (rgb_array[row, col1, 0] + rgb_array[row, col, 0]) >> 1
                    out_array[cc1, rr, 1] = (rgb_array[row, col1, 1] + rgb_array[row, col, 1]) >> 1
                    out_array[cc1, rr, 2] = (rgb_array[row, col1, 2] + rgb_array[row, col, 2]) >> 1
                if col == 0:
                    # left value
                    out_array[cc, rr1, 0] = (rgb_array[row1, col, 0] + rgb_array[row, col, 0]) >> 1
                    out_array[cc, rr1, 1] = (rgb_array[row1, col, 1] + rgb_array[row, col, 1]) >> 1
                    out_array[cc, rr1, 2] = (rgb_array[row1, col, 2] + rgb_array[row, col, 2]) >> 1

                # bottom value
                out_array[cc1, rr2, 0] = (rgb_array[row1, col1, 0] + rgb_array[row1, col, 0]) >> 1
                out_array[cc1, rr2, 1] = (rgb_array[row1, col1, 1] + rgb_array[row1, col, 1]) >> 1
                out_array[cc1, rr2, 2] = (rgb_array[row1, col1, 2] + rgb_array[row1, col, 2]) >> 1

                # right value
                out_array[cc2, rr1, 0] = (rgb_array[row1, col1, 0] + rgb_array[row, col1, 0]) >> 1
                out_array[cc2, rr1, 1] = (rgb_array[row1, col1, 1] + rgb_array[row, col1, 1]) >> 1
                out_array[cc2, rr1, 2] = (rgb_array[row1, col1, 2] + rgb_array[row, col1, 2]) >> 1

                # mid value
                out_array[cc1, rr1, 0] = (out_array[cc2, rr1, 0] + out_array[cc, rr1, 0]) >> 1
                out_array[cc1, rr1, 1] = (out_array[cc2, rr1, 1] + out_array[cc, rr1, 1]) >> 1
                out_array[cc1, rr1, 2] = (out_array[cc2, rr1, 2] + out_array[cc, rr1, 2]) >> 1


    return pygame.image.frombuffer(out_array, (w2, h2), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef zoomx2_bilineaire_greyc(np.ndarray[np.uint8_t, ndim=2] array_):
    """
    Scale x2 a given image converted into a 2d greyscale array (using bilineaire approximation)
    The final image will be stripped of the alpha channel, returns a greyscale 24 bit image.

    :param surface: Greyscale pygame.Surface 24-bit format image 
    :return : pygame.Surface, Returns a pygame Surface 24 bit (size x2) with no channel alpha
    """

    assert isinstance(array_, numpy.ndarray), \
           'Positional argument array_ must be a numpy.ndarray got %s ' % type(array_)
    
    cdef int w, h
    try:
        w, h = (<object>array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w!=0 and h!=0, 'Incompatible surface got (w:%s, h:%s) ' % (w, h)

    cdef:
        int i = 0, j = 0, row, col, row1, \
            col1, row2, col2, rr, cc, rr2, cc2, rr1, cc1
        int w2 = w * 2, h2 = h * 2
        unsigned char [:, ::1] out_array = numpy.zeros((h2, w2), dtype=numpy.uint8)
        unsigned char [:, :] grey_array = array_

    with nogil:
        for row in prange(w - 1):
            for col in range(h - 1):
                col1 = col + 1
                row1 = row + 1
                row2 = row + 2
                col2 = col + 2
                rr = row * 2
                cc = col * 2
                rr2 = rr + 2
                cc2 = cc + 2
                rr1 = rr + 1
                cc1 = cc + 1
                # Place the 4 known values (topleft, topright, bottom left, bottom right)
                # Top left
                out_array[cc, rr] = grey_array[row, col]             
                # top right
                out_array[cc2, rr] = grey_array[row, col1]             
                # bottom left
                out_array[cc, rr2] = grey_array[row1, col]                
                # bottom right
                out_array[cc2, rr2] = grey_array[row1, col1]             
                if row == 0:
                    # top value
                    out_array[cc1, rr] = (grey_array[row, col1] + grey_array[row, col]) >> 1
                if col == 0:
                    # left value
                    out_array[cc, rr1] = (grey_array[row1, col] + grey_array[row, col]) >> 1             
                # bottom value
                out_array[cc1, rr2] = (grey_array[row1, col1] + grey_array[row1, col]) >> 1
                # right value
                out_array[cc2, rr1] = (grey_array[row1, col1] + grey_array[row, col1]) >> 1          
                # mid value
                out_array[cc1, rr1] = (out_array[cc2, rr1] + out_array[cc, rr1]) >> 1

    return pygame.image.frombuffer(out_array, (w2, h2), 'RGB')


# horizontal_glitch(surface, 1, 0.3, (50+r)% 20) with r in range [0, 360]
# horizontal_glitch(surface, 1, 0.3, (50-r)% 20) with r in range [0, 360]
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef horizontal_glitch_c(texture_: Surface, double rad1_, double frequency_, double amplitude_):
    """
    Horizontal glitch effect
    Affect the entire texture by adding pixel deformation
    horizontal_glitch_c(texture_, 1, 0.1, 10)

    :param texture_:
    :param rad1_: Angle deformation in degrees (cos(1) * amplitude will represent the deformation magnitude)
    :param frequency_: Angle in degrees to add every iteration for randomizing the effect
    :param amplitude_: Deformation amplitude, 10 is plenty
    :return:
    """

    try:
        source_array = pixels3d(texture_)
    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nIncompatible texture, must be 24-32bit format.')
        raise ValueError('\nMake sure the surface_ contains per-pixel alpha transparency values.')
    cdef int w, h
    w, h = texture_.get_size()

    cdef:
        int i=0, j=0
        double rad = pi/180.0
        double angle = 0.0
        double angle1 = 0.0
        unsigned char [:, :, :] rgb_array = source_array
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=uint8)
        int ii=0

    with nogil:
        for i in range(w):
            for j in range(h):
                ii = (i + <int>(cos(angle) * amplitude_))
                if ii > w - 1:
                    ii = w
                if ii < 0:
                    ii = 0
                new_array[j, i, 0],\
                new_array[j, i, 1],\
                new_array[j, i, 2] = rgb_array[ii, j, 0],\
                    rgb_array[ii, j, 1], rgb_array[ii, j, 2]
            angle1 += frequency_ * rad
            angle += rad1_ * rad + rand() % angle1 - rand() % angle1

    return pygame.image.frombuffer(new_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef wave_xy_c(texture, float rad, int size):
    """
    Create a wave effect on a texture
    
    e.g:
    for angle in range(0, 360):
        surface = wave_xy(image, 8 * r * math.pi/180, 10)
        
    :param texture: pygame.Surface, image compatible format 24, 32-bit without per-pixel information
    :param rad: float,  angle in radian
    :param size: block size to copy
    :return: returns a pygame.Surface 24-bit without per-pixel information
    """
    assert isinstance(texture, Surface), \
        'Argument texture must be a Surface got %s ' % type(texture)
    assert isinstance(rad, float), \
        'Argument rad must be a python float got %s ' % type(rad)
    assert isinstance(size, int), \
        'Argument size must be a python int got %s ' % type(size)

    try:
        rgb_array = pixels3d(texture)

    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nIncompatible pixel format.')

    cdef int w, h, dim
    try:
        w, h, dim = rgb_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % (w, h, dim)
    cdef:
        unsigned char [:, :, ::1] wave_array = zeros((h, w, 3), dtype=uint8)
        unsigned char [:, :, :] rgb = rgb_array
        int x, y, x_pos, y_pos, xx, yy
        int i=0, j=0
        float c1 = 1.0 / float(size * size)
        int w_1 = w - 1
        int h_1 = h - 1

    with nogil:
        for x in prange(0, w_1 - size, size, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            x_pos = x + size + <int>(sin(rad + float(x) * c1) * float(size))
            for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                y_pos = y + size + <int>(sin(rad + float(y) * c1) * float(size))
                for i in range(0, size + 1):
                    for j in range(0, size + 1):
                        xx = x_pos + i
                        yy = y_pos + j

                        if xx > w_1:
                            xx = w_1
                        elif xx < 0:
                            xx = 0
                        if yy > h_1:
                            yy = h_1
                        elif yy < 0:
                            yy = 0
                        wave_array[yy, xx, 0] = rgb[x + i, y + j, 0]
                        wave_array[yy, xx, 1] = rgb[x + i, y + j, 1]
                        wave_array[yy, xx, 2] = rgb[x + i, y + j, 2]

    return pygame.image.frombuffer(wave_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef wave_y_c(texture, float rad, int size):
    """
    Create a wave effect on a texture (only vertical effect)
    e.g:
    for angle in range(0, 360):
        surface = wave_y(image, 8 * r * math.pi/180, 10)
    
    :param texture: pygame.Surface, image compatible format 24, 32-bit without per-pixel information
    :param rad: float,  angle in radian
    :param size: block size to copy
    :return: returns a pygame.Surface 24-bit without per-pixel information
    """
    assert isinstance(texture, Surface), \
        'Argument texture must be a Surface got %s ' % type(texture)
    assert isinstance(rad, float), \
        'Argument rad must be a python float got %s ' % type(rad)
    assert isinstance(size, int), \
        'Argument size must be a python int got %s ' % type(size)

    try:
        rgb_array = pixels3d(texture)

    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nIncompatible pixel format.')

    cdef int w, h, dim
    try:
        w, h, dim = rgb_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % (w, h, dim)
    cdef:
        unsigned char [:, :, ::1] wave_array = zeros((h, w, 3), dtype=uint8)
        unsigned char [:, :, :] rgb = rgb_array
        int x, y, x_pos, y_pos, xx, yy
        int i=0, j=0
        float c1 = 1.0 / float(size * size)
        int w_1 = w - 1
        int h_1 = h - 1

    with nogil:
        for x in prange(0, w_1 - size, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            # x_pos = x + size + <int>(sin(rad + float(x) * c1) * float(size))
            for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                y_pos = y + size + <int>(sin(rad + float(y) * c1) * float(size))
                for i in range(0, size + 1):
                    for j in range(0, size + 1):
                        xx = x + i
                        yy = y_pos + j

                        if xx > w_1:
                            xx = w_1
                        elif xx < 0:
                            xx = 0
                        if yy > h_1:
                            yy = h_1
                        elif yy < 0:
                            yy = 0
                        wave_array[yy, xx, 0] = rgb[x + i, y + j, 0]
                        wave_array[yy, xx, 1] = rgb[x + i, y + j, 1]
                        wave_array[yy, xx, 2] = rgb[x + i, y + j, 2]

    return pygame.image.frombuffer(wave_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef wave_x_c(texture, float rad, int size):
    """
    Create a wave effect on a texture (only horizontal effect)
    e.g:
    for angle in range(0, 360):
        surface = wave_x(image, 8 * r * math.pi/180, 10)
    
    :param texture: pygame.Surface, image compatible format 24, 32-bit without per-pixel information
    :param rad: float,  angle in radian
    :param size: block size to copy
    :return: returns a pygame.Surface 24-bit without per-pixel information
    """
    assert isinstance(texture, Surface), \
        'Argument texture must be a Surface got %s ' % type(texture)
    assert isinstance(rad, float), \
        'Argument rad must be a python float got %s ' % type(rad)
    assert isinstance(size, int), \
        'Argument size must be a python int got %s ' % type(size)

    try:
        rgb_array = pixels3d(texture)

    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nIncompatible pixel format.')

    cdef int w, h, dim
    try:
        w, h, dim = rgb_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % (w, h, dim)
    cdef:
        unsigned char [:, :, ::1] wave_array = zeros((h, w, 3), dtype=uint8)
        unsigned char [:, :, :] rgb = rgb_array
        int x, y, x_pos, y_pos, xx, yy
        int i=0, j=0
        float c1 = 1.0 / float(size * size)
        int w_1 = w - 1
        int h_1 = h - 1

    with nogil:
        for x in prange(0, w_1 - size, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            x_pos = x + size + <int>(sin(rad + float(x) * c1) * float(size))
            for y in prange(0, h_1 - size, size, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                # y_pos = y + size + <int>(sin(rad + float(y) * c1) * float(size))
                for i in range(0, size + 1):
                    for j in range(0, size + 1):
                        xx = x_pos + i
                        yy = y + j

                        if xx > w_1:
                            xx = w_1
                        elif xx < 0:
                            xx = 0
                        if yy > h_1:
                            yy = h_1
                        elif yy < 0:
                            yy = 0
                        wave_array[yy, xx, 0] = rgb[x + i, y + j, 0]
                        wave_array[yy, xx, 1] = rgb[x + i, y + j, 1]
                        wave_array[yy, xx, 2] = rgb[x + i, y + j, 2]

    return pygame.image.frombuffer(wave_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef wave_xy_alphac(texture, float rad, int size):
    """
    Create a wave effect on a texture
    e.g:
    for angle in range(0, 360):
        surface = wave_xy(image, 8 * r * math.pi/180, 10)
    
    :param texture: pygame.Surface, image compatible format 24, 32-bit without per-pixel information
    :param rad: float,  angle in radian
    :param size: block size to copy
    :return: returns a pygame.Surface 32-bit with per-pixel information
    """
    assert isinstance(texture, Surface), \
        'Argument texture must be a Surface got %s ' % type(texture)
    assert isinstance(rad, float), \
        'Argument rad must be a python float got %s ' % type(rad)
    assert isinstance(size, int), \
        'Argument size must be a python int got %s ' % type(size)

    try:
        rgb_array = pixels3d(texture)
        alpha = pygame.surfarray.pixels_alpha(texture)

    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nMake sure the surface_ contains per-pixel alpha transparency values.')

    cdef int w, h, dim
    try:
        w, h, dim = rgb_array.shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % (w, h, dim)

    # force alpha to be contiguous
    if not alpha.flags['C_CONTIGUOUS']:
        alpha = numpy.ascontiguousarray(alpha)

    cdef:
        unsigned char [:, :, ::1] wave_array = zeros((h, w, 4), dtype=uint8)
        unsigned char [:, ::1] alpha_array = alpha
        unsigned char [:, :, :] rgb = rgb_array
        int x, y, x_pos, y_pos, xx, yy
        int i=0, j=0
        float c1 = 1.0 / float(size * size)
        int w_1 = w - 1
        int h_1 = h - 1

    cdef int pack = size
    with nogil:
        for x in prange(0, w_1 - size, pack, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            x_pos = x + size + <int>(sin(rad + float(x) * c1) * float(size))
            for y in prange(0, h_1 - size, pack, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                y_pos = y + size + <int>(sin(rad + float(y) * c1) * float(size))
                for i in range(0, size + 1):
                    for j in range(0, size + 1):
                        xx = x_pos + i
                        yy = y_pos + j

                        if xx > w_1:
                            xx = w_1
                        elif xx < 0:
                            xx = 0
                        if yy > h_1:
                            yy = h_1
                        elif yy < 0:
                            yy = 0
                        wave_array[yy, xx, 0] = rgb[x + i, y + j, 0]
                        wave_array[yy, xx, 1] = rgb[x + i, y + j, 1]
                        wave_array[yy, xx, 2] = rgb[x + i, y + j, 2]
                        wave_array[yy, xx, 3] = alpha_array[x + i, y + j]

    return pygame.image.frombuffer(wave_array, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef heatwave_vertical_c1(texture: pygame.Surface,
                 mask_array,
                 float frequency,
                 float amplitude,
                 float attenuation):
    """
    Vertical displacement 
    
    :param texture: 
    :param mask_array: 
    :param frequency: 
    :param amplitude: 
    :param attenuation: 
    :return: 
    """
    assert isinstance(texture, pygame.Surface),\
           "Arguement texture should be a pygame surface, got %s " % type(texture)
    assert isinstance(mask_array, numpy.ndarray), \
           "Arguement mask_array should be a python numpy.ndarray, got %s " % type(mask_array)
    assert isinstance(frequency, float), \
           "Arguement frequency should be a float, got %s " % type(frequency)
    assert isinstance(amplitude, float), \
           "Arguement amplitude should be a float, got %s " % type(amplitude)

    try:
        array = pixels3d(texture)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible texture, must be 24-32bit format.')
    
    cdef int w, h
    w, h = texture.get_size()
    if w == 0 or h == 0:
        raise ValueError('Image with incorrect dimensions (w>0, h>0) got (w:%s, h:%s)' % (w, h))

    cdef:
        unsigned char [:, :, ::1] new_array = zeros((h, w, 3), dtype=numpy.uint8)
        unsigned char [:, :, :] source_array = array
        unsigned char [:, :, :] mask = mask_array
        int x = 0, y = 0, xx, yy, distortion

    with nogil:
        for x in range(w):
            for y in range(h):
                distortion = int(sin(x * attenuation + frequency) * amplitude * mask[y, x, 0])

                yy = int(y + distortion + rand() * 0.00002 )

                if yy > h - 1:
                    yy = h -1
                if yy < 0:
                    yy = 0
                if not mask[yy, x, 0] == 0:

                    new_array[y, x, 0] = source_array[x, yy, 0]
                    new_array[y, x, 1] = source_array[x, yy, 1]
                    new_array[y, x, 2] = source_array[x, yy, 2]
                else:
                    new_array[y, x, 0] = source_array[x, y, 0]
                    new_array[y, x, 1] = source_array[x, y, 1]
                    new_array[y, x, 2] = source_array[x, y, 2]

    return pygame.image.frombuffer(new_array, (w, h), 'RGB')




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef heatwave_horizontal_c1(texture: pygame.Surface,
                            mask_array,
                            float frequency,
                            float amplitude,
                            float attenuation):
    """
    Horizontal displacement
    
    :param texture: 
    :param mask_array: 
    :param frequency: 
    :param amplitude: 
    :param attenuation: 
    :return: 
    """

    assert isinstance(texture, pygame.Surface),\
           "Arguement texture should be a pygame surface, got %s " % type(texture)
    assert isinstance(mask_array, numpy.ndarray), \
           "Arguement mask_array should be a python numpy.ndarray, got %s " % type(mask_array)
    assert isinstance(frequency, float), \
           "Arguement frequency should be a float, got %s " % type(frequency)
    assert isinstance(amplitude, float), \
           "Arguement amplitude should be a float, got %s " % type(amplitude)

    try:
        array = pixels3d(texture)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible texture, must be 24-32bit format.')

    cdef int w, h
    w, h = texture.get_size()
    if w == 0 or h == 0:
        raise ValueError('Image with incorrect dimensions (w>0, h>0) got (w:%s, h:%s)' % (w, h))

    cdef:
        unsigned char [:, :, ::1] new_array = zeros((h, w, 3), dtype=numpy.uint8)
        unsigned char [:, :, :] source_array = array
        unsigned char [:, :, :] mask = mask_array
        int x = 0, y = 0, xx, yy, distortion

    with nogil:
        for x in range(w):
            for y in range(h):
                distortion = int(sin(x * attenuation + frequency) * amplitude * mask[y, x, 0])

                xx = int(x + distortion)
                if xx > w - 1:
                    xx = w - 1
                if xx < 0:
                    xx = 0

                if not mask[y, xx, 0] == 0:
                    new_array[y, x, 0] = source_array[xx, y, 0]
                    new_array[y, x, 1] = source_array[xx, y, 1]
                    new_array[y, x, 2] = source_array[xx, y, 2]

                else:
                    new_array[y, x, 0] = source_array[x, y, 0]
                    new_array[y, x, 1] = source_array[x, y, 1]
                    new_array[y, x, 2] = source_array[x, y, 2]

    return pygame.image.frombuffer(new_array, (w, h), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned int rgb_to_int(int red, int green, int blue)nogil:
    """
    Convert RGB model into a python integer equivalent to pygame map_rgb()
    :param red: Red color value [0..255] 
    :param green: Green color value [0..255]
    :param blue: Blue color [0.255]
    :return: returns a python integer representing the RGB values(int32)
    """
    return 65536 * red + 256 * green + blue

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned int * int_to_rgb(unsigned int n)nogil:
    """
    Convert a python integer into a RGB colour model (unsigned char values [0..255]).
    Equivalent to pygame unmap_rgb()
    :param n: integer value to convert
    :return: return a pointer containing RGB values (pointer to a list of RGB values)
    """
    cdef:
        unsigned int *rgb = <unsigned int *> malloc(3 * sizeof(unsigned int))
    rgb[0] = n >> 16 & 255  # red int32
    rgb[1] = n >> 8 & 255   # green int32
    rgb[2] = n & 255        # blue int32
    free(rgb)
    return rgb


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int rgba_to_int(unsigned char red,
                            unsigned char green, unsigned char blue, unsigned char alpha)nogil:
    """
    Type      Capacity

    Int16 -- (-32,768 to +32,767)

    Int32 -- (-2,147,483,648 to +2,147,483,647)

    Int64 -- (-9,223,372,036,854,775,808 to +9,223,372,036,854,775,807)
   
    Convert RGBA model into a mapped python integer (int32) equivalent to pygame map_rgb()
    Output integer value between (-2,147,483,648 to +2,147,483,647).
    
    
    :param red:unsigned char; Red color must be in range[0 ...255]
    :param green: unsigned char; Green color value [0 ... 255]
    :param blue: unsigned char; Blue color [0 ... 255]
    :return: returns a python integer (int32) representing the RGBA values
    """
    return (alpha << 24) + (red << 16) + (green << 8) + blue

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char * int_to_rgba(int n)nogil:
    """
    Type      Capacity

    Int16 -- (-32,768 to +32,767)

    Int32 -- (-2,147,483,648 to +2,147,483,647)

    Int64 -- (-9,223,372,036,854,775,808 to +9,223,372,036,854,775,807)
    
    Convert a python integer into a RGBA colour model.
    Equivalent to pygame unmap_rgb() 
    :param n: integer value to convert (c int32)
    :return: return a pointer containing RGBA values (pointer to a list of RGBA values)
    Integer value is unmapped into RGBA values (unsigned char type, [0 ... 255] 
    """
    cdef:
        unsigned char *rgba = <unsigned char *> malloc(4 * sizeof(unsigned char))

    rgba[3] = (n >> 24) & 255  # alpha
    rgba[0] = (n >> 16) & 255  # red
    rgba[1] = (n >> 8) & 255   # green
    rgba[2] = n & 255          # blue
    free(rgba)
    return rgba


def make_palette(size: int, height: int, fh: float=0.25, fs: float=255.0, fl: float=2.0):
    return make_palette_c(size, height, fh, fs, fl)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_palette_c(int width, int height, float fh, float fs, float fl):
    """
    Create a palette of RGB colors (width x height)
    
    e.g: 
        # below: palette of 256 colors & surface (width=256, height=50).
        # hue * 6, saturation = 255.0, lightness * 2.0
        palette, surf = make_palette(256, 50, 6, 255, 2)
        palette, surf = make_palette(256, 50, 4, 255, 2)
        
    :param width: integer, Palette width
    :param height: integer, palette height
    :param fh: float, hue factor, default 1/4 
    :param fs: float, saturation factor, default 255.0 
    :param fl: float, lightness factor, default 2.0
    :return: Return a tuple numpy.ndarray type uint32 and pygame.Surface (width, height) 
    """
    assert width > 0, "Argument width should be > 0, got %s " % width
    assert height > 0, "Argument height should be > 0, got %s " % height

    cdef:
        unsigned int [:] palette = numpy.ndarray(width, numpy.uint32)
        int x
        float h, s, l
        # unsigned int *color
        double *rgb = [0.0, 0.0, 0.0]

    for x in range(width):
        h, s, l = <float>x * fh,  min(fs, 255.0), min(<float>x * fl, 255.0)
        rgb = hls_to_rgb(h * ONE_360, l * ONE_255, s * ONE_255)
        palette[x] = rgb_to_int(<int>(rgb[0] * 255.0),
                                <int>(rgb[1] * 255.0),
                                <int>(rgb[2] * 255.0))
        # max = 16777215 -> RGB=(255, 255, 255)
        # color = int_to_rgb(palette[x])

    pal = numpy.ndarray((width, height, 3), dtype=numpy.uint8)
    i = 0
    for x in range(width):
        color = int_to_rgb(palette[i])
        for y in range(height):
            pal[x, y, 0] = color[0]
            pal[x, y, 1] = color[1]
            pal[x, y, 2] = color[2]
        i += 1
    return numpy.asarray(palette), pygame.surfarray.make_surface(pal)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fire_texture24(int width, int height, int frame, float factor, pal, mask):
    """
    Create an animated flame effect of sizes (width, height).
    The flame effect contains per-pixel transparency
    
    e.g:
        palette, surf = make_palette(256, 50, 4, 255, 2)
        mask = numpy.full((width, height), 255, dtype=numpy.uint8)
        buff = fire_effect24(width, height, 1000, 3.95, palette, mask)
    
    
    :param width: integer; max width of the effect
    :param height: integer; max height of the effect 
    :param frame: integer; number of frames for the animation 
    :param factor: float; change the flame height, default is 3.95 
    :param pal: define a color palette e.g make_palette(256, 50, 6, 255, 2)
    :param mask: Ideally a black and white texture transformed into a 2d array shapes (w, h) 
                 black pixel will cancel the effect. 
                 The mask should have the exact same sizes than passed argument (width, height) 
    :return: Return a python list containing all the per-pixel surfaces.
    """

    assert isinstance(width, int), \
           "Argument width should be a python int, got %s " % type(width)
    assert isinstance(height, int), \
           "Argument height should be a python int, got %s " % type(height)
    assert isinstance(frame, int), \
           "Argument frame should be a python int, got %s " % type(frame)
    assert isinstance(factor, float), \
           "Argument factor should be a python float, got %s " % type(factor)
    assert isinstance(mask, numpy.ndarray), \
           "Argument mask should be a numpy.ndarray, got %s " % type(mask)


    if not frame > 0:
        raise ValueError('Argument frame should be > 0, %s ' % frame)

    if width == 0 or height == 0:
        raise ValueError('Image with incorrect dimensions '
                         '(width>0, height>0) got (width:%s, height:%s)' % (width, height))
    cdef:
        int w, h
    try:
        w, h = mask.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    if width != w or height != h:
        raise ValueError('Incorrect mask dimensions '
                         'mask should be (width=%s, height=%s), '
                         'got (width=%s, height=%s)' %(width, height, w, h))
    cdef:
        float [:, ::1] fire = zeros((height, width), dtype=numpy.float32)
        # flame opacity palette
        unsigned int [::1] alpha = make_palette(256, 1, 1, 0, 2)[0]
        unsigned int [:, :, ::1] out = zeros((height, width, 3), dtype=numpy.uint32)
        unsigned int [::1] palette = pal
        # mask to use, ideally a black and white
        # texture transform into a 2d array shapes (w, h)
        # black pixel will cancel the flame effect.
        unsigned char [:, :] mask_ = mask
        int x = 0, y = 0, i = 0, f
        float d
        unsigned int *color

    list_ = []


    for f in range(frame):
        for x in range(width):
            fire[height-1, x] = random.randint(1, 255)

        with nogil:
            for y in prange(0, height - 1):
                for x in range(0, width - 1):
                    if mask_[x, y] != 0:
                        d = (fire[(y + 1) % height, (x - 1 + width) % width]
                                       + fire[(y + 1) % height, x % width]
                                       + fire[(y + 1) % height, (x + 1) % width]
                                       + fire[(y + 2) % height, x % width]) / factor
                        d -= rand() * 0.0001
                        if d > 255.0:
                            d = 255.0
                        if d < 0:
                            d = 0
                        fire[y, x] = d
                        color = int_to_rgb(palette[<unsigned int>d])
                        out[y, x, 0], out[y, x, 1], out[y, x, 2] = \
                            <unsigned char>color[0], <unsigned char>color[1], <unsigned char>color[2]
                    else:
                        out[y, x, 0], out[y, x, 1], out[y, x, 2] = 0, 0, 0
        surface = pygame.image.frombuffer(numpy.asarray(out, dtype=numpy.uint8), (width, height), 'RGB')
        list_.append(surface)
    return list_

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fire_texture32(int width, int height, int frame, float factor, pal, mask):
    """
    Create an animated flame effect of sizes (width, height).
    The flame effect contains per-pixel transparency
    
    e.g:
        width, height = 200, 200
        image = pygame.image.load("LOAD YOUR IMAGE")
        image = pygame.transform.smoothscale(image, (width, height))
        mask = greyscale_3d_to_2d(pygame.surfarray.pixels3d(image))
        buff = fire_effect32(width, height, 1000, 3.95, palette, mask)
    
    :param width: integer; max width of the effect
    :param height: integer; max height of the effect 
    :param frame: integer; number of frames for the animation 
    :param factor: float; change the flame height, default is 3.95 
    :param pal: define a color palette e.g make_palette(256, 50, 6, 255, 2)
    :param mask: Ideally a black and white texture transformed into a 2d array shapes (w, h) 
                 black pixel will cancel the effect. 
                 The mask should have the exact same sizes than passed argument (width, height) 
    :return: Return a python list containing all the per-pixel surfaces.
    """

    assert isinstance(width, int), \
           "Argument width should be a python int, got %s " % type(width)
    assert isinstance(height, int), \
           "Argument height should be a python int, got %s " % type(height)
    assert isinstance(frame, int), \
           "Argument frame should be a python int, got %s " % type(frame)
    assert isinstance(factor, float), \
           "Argument factor should be a python float, got %s " % type(factor)
    assert isinstance(mask, numpy.ndarray), \
           "Argument mask should be a numpy.ndarray, got %s " % type(mask)


    if not frame > 0:
        raise ValueError('Argument frame should be > 0, %s ' % frame)

    if width == 0 or height == 0:
        raise ValueError('Image with incorrect dimensions '
                         '(width>0, height>0) got (width:%s, height:%s)' % (width, height))
    cdef:
        int w, h
    try:
        w, h = mask.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')
    
    if width != w or height != h:
        raise ValueError('Incorrect mask dimensions '
                         'mask should be (width=%s, height=%s), '
                         'got (width=%s, height=%s)' %(width, height, w, h))
    cdef:
        float [:, ::1] fire = zeros((height, width), dtype=numpy.float32)
        # flame opacity palette
        unsigned int [::1] alpha = make_palette(256, 1, 1, 0, 2)[0]
        unsigned int [:, :, ::1] out = zeros((height, width, 4), dtype=numpy.uint32)
        unsigned int [::1] palette = pal
        # mask to use, ideally a black and white
        # texture transform into a 2d array shapes (w, h)
        # black pixel will cancel the flame effect.
        unsigned char [:, :] mask_ = mask
        int x = 0, y = 0, i = 0, f
        float d
        unsigned int *color

    list_ = []


    for f in range(frame):
        for x in range(width):        
            fire[height-1, x] = random.randint(1, 255)

        with nogil:
            for y in prange(0, height - 1):
                for x in range(0, width - 1):
                    if mask_[x, y] != 0:
                        d = (fire[(y + 1) % height, (x - 1 + width) % width]
                                       + fire[(y + 1) % height, x % width]
                                       + fire[(y + 1) % height, (x + 1) % width]
                                       + fire[(y + 2) % height, x % width]) / factor
                        d -= rand() * 0.0001
                        if d > 255.0:
                            d = 255.0
                        if d < 0:
                            d = 0
                        fire[y, x] = d
                        color = int_to_rgb(palette[<unsigned int>d])
                        out[y, x, 0], out[y, x, 1], \
                        out[y, x, 2], out[y, x, 3]  = <unsigned char>color[0], \
                            <unsigned char>color[1], <unsigned char>color[2],  alpha[<unsigned int>d]
                    else:
                        out[y, x, 0], out[y, x, 1], out[y, x, 2], out[y, x, 3] = 0, 0, 0, 0
        surface = pygame.image.frombuffer(numpy.asarray(out, dtype=numpy.uint8), (width, height), 'RGBA')
        list_.append(surface)
    return list_


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float * hsv2rgb_c(double h, double s, double v)nogil:
    """
    Convert hsv color model to rgb

    :param h: python float hue
    :param s: python float saturation
    :param v: python float value
    :return: Return tuple RGB colors
    """
    cdef:
        int i = 0
        double f, p, q, t
        float *rgb = <float *> malloc(3 * sizeof(float))

    if s == 0.0:
        rgb[0] = v
        rgb[1] = v
        rgb[2] = v
        free(rgb)
        return rgb

    i = <int>(h * 6.0)
    f = (h * 6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s * f)
    t = v*(1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        rgb[0] = v
        rgb[1] = t
        rgb[2] = p
        free(rgb)
        return rgb
    if i == 1:
        rgb[0] = q
        rgb[1] = v
        rgb[2] = p
        free(rgb)
        return rgb
    if i == 2:
        rgb[0] = p
        rgb[1] = v
        rgb[2] = t
        free(rgb)
        return rgb
    if i == 3:
        rgb[0] = p
        rgb[1] = q
        rgb[2] = v
        free(rgb)
        return rgb
    if i == 4:
        rgb[0] = t
        rgb[1] = p
        rgb[2] = v
        free(rgb)
        return rgb
    if i == 5:
        rgb[0] = v
        rgb[1] = p
        rgb[2] = q
        free(rgb)
        return rgb


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float * rgb2hsv_c(double r_, double g_, double b_)nogil:
    """
    Convert rgb color model to hsv
    This method is identical to colorsys.rgb_to_hsv(r, g, b) with rgb in range[0.0 ... 1.0]
    to get the exact same result, value needs to be multiply by 256.
    e.g:
    h, s, v = rgb2hsv(0.1, 0.2, 0.3)
    h1, s1, v1 = colorsys.rgb_to_hsv(0.1, 0.2, 0.3)
    h1 = h, s1 = s, v1 = v * 256 
    
    :param r_: python float red
    :param g_: python float green
    :param b_: python float blue
    :return: Return tuple hsv values corresponding to the RGB input values.
    """
    cdef double r = r_ * ONE_255
    cdef double g = g_ * ONE_255
    cdef double b = b_ * ONE_255
    cdef:
        double mx, mn
        double h, df, s, v, df_
        float *hsv = <float *> malloc(3 * sizeof(float))
        
    mx = fmax_rgb_value(r, g, b)
    mn = fmin_rgb_value(r, g, b)

    df = mx-mn
    df_ = 1.0/df
    if mx == mn:
        h = 0
    
    elif mx == r:
        h = (60 * ((g-b) * df_) + 360) % 360  
    elif mx == g:
        h = (60 * ((b-r) * df_) + 120) % 360  
    elif mx == b:
        h = (60 * ((r-g) * df_) + 240) % 360  
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    hsv[0] = h * ONE_360
    hsv[1] = s
    hsv[2] = v
    free(hsv)
    return hsv



# HLS: Hue, Luminance, Saturation
# H: position in the spectrum
# L: color lightness
# S: color saturation
# All inputs and outputs are triples of floats in the range [0.0...1.0]
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline float * rgb2hls_c(double r, double g, double b)nogil:
    """
    Convert RGB color model into HLS model

    :param r: pygame float, red value
    :param g: pygame float, green value
    :param b: pygame float, blue value
    :return: return HLS values (float)
    """
    cdef:
        double maxc, minc, l, s, rc, gc, bc, h
        double high, low, high_
        float *hls = <float *> malloc(3 * sizeof(float))

    # SLOW DOWN
    maxc = fmax_rgb_value(r, g, b)
    minc = fmin_rgb_value(r, g, b)

    high = maxc-minc
    high_ = 1.0 /high
    low = maxc+minc
    l = (minc+maxc)* 0.5

    if minc == maxc:
        hls[0] = 0.0
        hls[1] = l
        hls[2] = 0.0
        free(hls)
        return hls

    if l <= 0.5:
        s = high / low
    else:
        s = high / (2.0-maxc-minc)

    rc = (maxc-r) * high_
    gc = (maxc-g) * high_
    bc = (maxc-b) * high_

    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc

    h = h * ONE_SIXTH
    if h < 0:
        hls[0] = 1.0 - (h * -1)
    else:
        hls[0] = fmod(h, 1.0)

    hls[1] = l
    hls[2] = s
    free(hls)
    return hls

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef float * hls2rgb_c(double h, double l, double s)nogil:
    """
    Convert HLS color model into RGB model

    :param h: pygame float, hue value
    :param l: pygame float, lightness
    :param s: pygame float, saturation
    :return: return RGB values (float)
    """
    cdef:
        double m1, m2
        float *rgb = <float *> malloc(3 * sizeof(float))

    if s == 0.0:
        rgb[0] = l
        rgb[1] = l
        rgb[2] = l
        free(rgb)
        return rgb

    if l <= 0.5:
        m2 = l * (1.0 + s)
    else:
        m2 = l + s -(l * s)

    m1 = 2.0 * l - m2
    rgb[0] = _v(m1, m2, h+ONE_THIRD)
    rgb[1] = _v(m1, m2, h)
    rgb[2] = _v(m1, m2, h-ONE_THIRD)
    free(rgb)
    return rgb

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef float _v(double m1, double m2, double hue)nogil:

    hue = hue % 1.0  # -1//5 * 5 +(-1 % 5) with a = -1 and b =  5
    if hue < ONE_SIXTH:
        return m1 + (m2 - m1) * hue * 6.0
    if hue < 0.5:
        return m2
    if hue < TWO_THIRD:
        return m1 + (m2 - m1) * (TWO_THIRD-hue) * 6.0
    return m1



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef swap_channels_c(surface_:Surface, model):
    """
    :param surface_: pygame.Surface
    :param model: python string; String representing the channel order e.g
    RGB, RBG, GRB, GBR, BRG, BGR etc. letters can also be replaced by the digit 0
    to null the entire channel. e.g : 'R0B' -> no green channel  
        
    TEST:
        All test performed on a single thread (multi-processing off) 
        IMAGE PNG 320x720, 32 bit depth
         _________________________________________________________________________________       
        |   RESULT   | FORMAT  |   CONVERSION     |      TIMING SINGLE ITERATION          |
        |____________|_________|__________________|_______________________________________|
        1 PASS MODE: | 32-bit  |                  |   
        2 PASS MODE: | 32-bit  | convert()        |   
        3 PASS MODE: | 32-bit  | convert_alpha()  |   
        
        4 PASS MODE: | 24-bit  |                  |   
        5 PASS MODE: | 24-bit  | convert()        |   
        6 PASS MODE: | 24-bit  | convert_alpha()  |   
        
        7 PASS MODE: | 8-bit   |                  |    
        8 PASS MODE: | 8-bit   | convert()        |   
        9 PASS MODE: | 8-bit   | convert_alpha()  |   
    """
    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    if len(model) != 3:
        print("\nArgument model is invalid.")
        raise ValueError("Choose between RGB, RBG, GRB, GBR, BRG, BGR")

    rr, gg, bb = list(model)
    order = {'R' : 0, 'G' : 1, 'B' : 2, '0': -1}

    cdef int width, height
    width, height = surface_.get_size()

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
        try:
            rgb_ = array3d(surface_)
        except(pygame.error, ValueError):
            raise ValueError('\nIncompatible pixel format.')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        short int ri, gi, bi
    ri = order[rr]
    gi = order[gg]
    bi = order[bb]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                if ri == -1:
                    new_array[j, i, 0] = 0
                else:
                    new_array[j, i, 0] = rgb_array[i, j, ri]
                    
                if gi == -1:
                    new_array[j, i, 1] = 0                  
                else:
                    new_array[j, i, 1] = rgb_array[i, j, gi]
                    
                if bi == -1:
                    new_array[j, i, 2] = 0                
                else:
                    new_array[j, i, 2] = rgb_array[i, j, bi]

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


