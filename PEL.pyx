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

from libc.math cimport sin, sqrt, cos, atan2, pi, round, floor, fmax, fmin, pi, tan, exp, ceil
from libc.stdio cimport printf
from libc.stdlib cimport srand, rand, RAND_MAX
from libc.stdlib cimport qsort, malloc

from RippleEffect import droplet_float, droplet_int, droplet_grad

__author__ = "Yoann Berenguer"
__credits__ = ["Yoann Berenguer"]
__version__ = "1.0.0 untested"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"



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
DEF TWO_THIRD = 2.0/3.0


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

# ***********************************************
# **********  METHOD TRANSPARENCY ***************
# ***********************************************

# ADD TRANSPARENCY -----------------------------
def make_transparent(image_: Surface, alpha_: int)->Surface:
    return make_transparent_c(image_, alpha_)

def make_array_transparent(rgb_array_: ndarray, alpha_array_: ndarray, alpha_: int)->Surface:
    return make_array_transparent_c(rgb_array_, alpha_array_, alpha_)


# ADD OPACITY ----------------------------------
def make_opaque(image_:Surface, alpha_: int) -> Surface:
    return make_opaque_c(image_, alpha_)

def make_array_opaque(rgb_array_:ndarray, alpha_array_:ndarray, alpha_: int) -> Surface:
    return make_array_opaque_c(rgb_array_, alpha_array_, alpha_)


# FILTERING RGB VALUES -------------------------
def low_th_alpha(surface_: Surface, new_alpha_: int, threshold_: int) -> Surface:
    return low_threshold_alpha_c(surface_, new_alpha_, threshold_)

def high_th_alpha(surface_: Surface, new_alpha_: int, threshold_: int) -> Surface:
    return high_threshold_alpha_c(surface_, new_alpha_, threshold_)


# ***********************************************
# **********  METHOD GREYSCALE ******************
# ***********************************************

# CONSERVE LIGHTNESS ----------------------------
def greyscale_light_alpha(image: Surface)->Surface:
    return greyscale_lightness_alpha_c(image)

def greyscale_light(image: Surface)->Surface:
    return greyscale_lightness_c(image)


# CONSERVE LUMINOSITY --------------------------
def greyscale_lum_alpha(image: Surface)->Surface:
    return greyscale_luminosity_alpha_c(image)

def greyscale_lum(image: Surface)->Surface:
    return greyscale_luminosity_c(image)


def make_greyscale_alpha(image: Surface)->Surface:
    return make_greyscale_alpha_c(image)

def make_greyscale(image: Surface)->Surface:
    return make_greyscale_c(image)

def make_greyscale_altern(image: Surface)->Surface:
    return make_greyscale_altern_c(image)

def greyscale_arr2surf(array_: ndarray)->Surface:
    return greyscale_arr2surf_c(array_)

def greyscale_array(array_: ndarray)->ndarray:
    return greyscale_array_c(array_)

def greyscale_3d_to_2d(array_: ndarray)->ndarray:
    return greyscale_3d_to_2d_c(array_)

def greyscale_2d_to_3d(array_: ndarray)->ndarray:
    return greyscale_2d_to_3d_c(array_)


# ***********************************************
# **********  METHOD COLORIZE  ******************
# ***********************************************
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


# ***********************************************
# *********  METHOD LOADING PER-PIXEL  **********
# ***********************************************
def load_per_pixel(file: str)->Surface:
    return load_per_pixel_c(file)

def load_image32(path: str)->Surface:
    return load_image32_c(path)


# ***********************************************
# ********* METHOD LOAD SPRITE SHEET ************
# ***********************************************
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

# ***********************************************
# **********  METHOD SHADOW *********************
# ***********************************************
def shadow(image: Surface)->Surface:
    return shadow_c(image)

# ***********************************************
# **********  METHOD MAKE_ARRAY  ****************
# ***********************************************
def make_array(rgb_array_: ndarray, alpha_:ndarray):
    return make_array_c_code(rgb_array_, alpha_)

def make_array_trans(rgb_array_: ndarray, alpha_:ndarray):
    return make_array_c_transpose(rgb_array_, alpha_)

def make_array_from_buffer(buffer_: BufferProxy, size_: tuple)->ndarray:
    return make_array_from_buffer_c(buffer_, size_)

# ***********************************************
# **********  METHOD MAKE_SURFACE ***************
# ***********************************************

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

# *********************************************
# **********  METHOD SPLIT RGB ****************
# *********************************************

def rgb_split_channels(surface_: Surface)->tuple:
    return rgb_split_channels_c(surface_)

def rgb_split_channels_alpha(surface_: Surface):
    return rgb_split_channels_alpha_c(surface_)

def red_channel(surface_: Surface)->Surface:
    return red_channel_c(surface_)

def green_channel(surface_: Surface)->Surface:
    return green_channel_c(surface_)

def blue_channel(surface_: Surface)->Surface:
    return blue_channel_c(surface_)

# *********************************************
# **********  METHOD FISHEYE  *****************
# *********************************************

def fish_eye(image)->Surface:
    return fish_eye_c(image)

def fish_eye_32(image)->Surface:
    raise NotImplementedError

# *********************************************
# **********  METHOD ROTATE  ******************
# *********************************************
def rotate_inplace(image: Surface, angle: int)->Surface:
    return rotate_inplace_c(image, angle)

def rotate_24(image: Surface, angle: int)->Surface:
    return rotate_c24(image, angle)

def rotate_32(image: Surface, angle: int)->Surface:
    return rotate_c32(image, angle)

# *********************************************
# **********  METHOD HUE SHIFT  ***************
# *********************************************
def hue_surface_24(surface_: Surface, float shift_)->Surface:
    return hue_surface_24c(surface_, shift_)

def hue_surface_32(surface_: Surface, float shift_)->Surface:
    return hue_surface_32c(surface_, shift_)

# HUE GIVEN COLOR VALUE FROM A SURFACE 24BIT
def hue_surface_24_color(surface_: Surface, float shift_,
                         color_:Color)->Surface:
    raise NotImplemented

# HUE PIXEL MEAN AVG VALUE OVER OR EQUAL TO A THRESHOLD VALUE
def hsah(surface_: Surface, threshold_: int, shift_: float)->Surface:
    return hsah_c(surface_, threshold_, shift_)

# HUE PIXEL MEAN AVG VALUE BELOW OR EQUAL TO A THRESHOLD VALUE
def hsal(surface_: Surface, threshold_: int, shift_: float)->Surface:
    return hsal_c(surface_, threshold_, shift_)

def hue_array_red(array_: ndarray, shift_: float)->ndarray:
    return hue_array_red_c(array_, shift_)

def hue_array_green(array: ndarray, shift_: float)->ndarray:
    return hue_array_green_c(array, shift_)

def hue_array_blue(array: ndarray, shift_: float)->ndarray:
    return hue_array_blue_c(array, shift_)


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

#*********************************************
#**********  METHOD SATURATION  **************
#*********************************************

def saturation_24(surface_: Surface, shift_: float)->Surface:
    return saturation_24_c(surface_, shift_)

def saturation_32(surface_: Surface, shift_: float)->Surface:
    return saturation_32_c(surface_, shift_)

#*********************************************
#**********  METHOD ROLL/SCROLL  *************
#*********************************************

def scroll_array(array: ndarray, dy: int=0, dx: int=0) -> ndarray:
    return scroll_array_c(array, dy, dx)

# USE NUMPY LIBRARY (NUMPY.ROLL)
def scroll_surface_org(array: ndarray, dx: int=0, dy: int=0)->tuple:
    ...

# Roll the value of an entire array (lateral and vertical)
# Identical algorithm (scroll_array) but returns a tuple (surface, array)
def scroll_surface(array: ndarray, dy: int=0, dx: int=0)->tuple:
    return scroll_surface_c(array, dy, dx)

# ROLL IMAGE TRANSPARENCY INSTEAD
def scroll_array_alpha(array: ndarray, dy: int=0, dx: int=0) -> ndarray:
    raise NotImplemented

# ROLL ARRAY 3D TYPE (W, H, 4) NUMPY.UINT8
def scroll_array_32(array: ndarray, dy: int=0, dx: int=0) -> ndarray:
    raise NotImplemented


#*********************************************
#**********  METHOD GRADIENT  ****************
#*********************************************
def gradient_array(width: int, height: int, start_color: tuple, end_color: tuple)->ndarray:
    return gradient_array_c(width, height, start_color, end_color)

def gradient_color(index: int, gradient: ndarray)-> tuple:
    return gradient_color_c(index, gradient)

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

def guaussian_boxblur3x3(image: Surface)->ndarray:
    return guaussian_boxblur3x3_c(image)

def guaussian_boxblur3x3_approx(image: Surface)->ndarray:
    return guaussian_boxblur3x3_capprox(image)

def guaussian_blur3x3(image: Surface)->ndarray:
    return guaussian_blur3x3_c(image)

def gaussian_blur5x5(rgb_array: ndarray):
    return gaussian_blur5x5_c(rgb_array)

def edge_detection3x3(image: Surface)->ndarray:
    return edge_detection3x3_c(image)

def edge_detection3x3_alter(image: Surface)->ndarray:
    return edge_detection3x3_c1(image)

def edge_detection3x3_fast(image: Surface)->ndarray:
    return edge_detection3x3_c2(image)

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

def sepia32(surface_: Surface)->Surface:
    return sepia32_c(surface_)

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
def array_rgba_to_bgra(rgba_array: ndarray)->ndarray:
    return array_rgba2bgra_c(rgba_array)

def array_rgb_to_bgr(rgb_array: ndarray)->ndarray:
    return array_rgb2bgr_c(rgb_array)

def array_bgra_to_rgba(bgra_array: ndarray)->ndarray:
    return array_bgra2rgba_c(bgra_array)

def array_bgr_to_rgb(bgr_array: ndarray)->ndarray:
    return array_bgr2rgb_c(bgr_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_transparent_c(image_:Surface, int alpha_):
    """
    The pygame surface must be a 32-bit image containing per-pixel.
    
    Add transparency to a pygame surface (alpha channel - alpha_ value).
    <alpha_> must be an integer in range[0 ... 255] otherwise raise a value error.
    <alpha_> = 255, the output surface will be 100% transparent.
    <alpha_> = 0, output image will remains unchanged.
    
    The code will raise a value error if the the surface is not encoded with per-pixel transparency. 
    ValueError: unsupported color masks for alpha reference array. 

    TIP: Do not apply pygame convert_alpha() or convert() to modify pixels format to the surface 
         before calling make_transparent function. The output image will have BGRA colour format 
         instead of RGBA. 
    
    EXAMPLE: 
        image = pygame.image.load('path to your image here')
        output = make_transparent(image, 100) 

    :param image_: pygame.surface, 32-bit format image containing per-pixel  
    :param alpha_: integer value for alpha channel
    :return: Return a 32-bit pygame surface containing per-pixel.
    """

    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)
    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)
    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    if not (image_.get_flags() & pygame.SRCALPHA):
        raise ValueError('Surface without per-pixel information.')

    try:
        # create a buffer_
        # if the image is convert_alpha() or convert() the buffer will
        # have BGRA pixels format instead of RGBA.
        buffer_ = image_.get_view('2')
    except (pygame.error, ValueError):
        raise ValueError('\nSurface incompatible.')

    cdef int w_, h_
    w_, h_ = image_.get_size()

    if w_==0 and h_==0:
        raise ValueError(
            'Image with incorrect shapes must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        # contiguous values
        unsigned char [:, :, ::1] array_ = \
              numpy.frombuffer(buffer_, dtype=uint8).reshape((w_, h_, 4))
        int w = w_
        int h = h_
        int i=0, j=0, a
    with nogil:
        for i in prange(w):
            for j in range(h):
                a = array_[i, j, 3]
                if a > 0:
                    a -= alpha_
                    if a < 0:
                        a = 0
                array_[i, j, 3] = a
    return pygame.image.frombuffer(array_, (w_, h_), 'RGBA')

# TODO BUILD IDENTICAL FUNCTION WITH RGBA ARRAY AS ARGUMENT

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
    """
    assert isinstance(rgb_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument rgb_array_ got %s ' % type(rgb_array_)
    assert isinstance(alpha_array_, numpy.ndarray), \
        'Expecting numpy.ndarray for positional argument alpha_ got %s ' % type(alpha_)

    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)
    if not (0 <= alpha_ <= 255):
        raise ValueError('\n[-] invalid value for argument alpha_, range [0..255] got %s ' % alpha_)

    cdef int shape[2]
    shape = (<object> rgb_array_).shape[:2]

    if shape[0] == 0 or shape[1] == 0:
        raise ValueError('<rgb_array_> with incorrect shapes.')

    cdef:
        int w = shape[0]
        int h = shape[1]
        unsigned char [:, :, :] rgb_array_c = rgb_array_                       # non-contiguous array
        unsigned char [:, ::1] alpha_array_c = alpha_array_                    # contiguous values
        unsigned char [: , :, ::1] new_array_c = empty((h, w, 4), dtype=uint8) # output array with contiguous values
        int i=0, j=0, a=0
    with nogil:
        for i in prange(w):
            for j in range(h):
                a = alpha_array_c[i, j]
                if a > 0:
                    a -= alpha_
                    if a < 0:
                        a = 0
                    alpha_array_c[i, j] = a
                new_array_c[j, i, 0], new_array_c[j, i, 1], new_array_c[j, i, 2], \
                new_array_c[j, i, 3] =  rgb_array_c[i, j, 0], rgb_array_c[i, j, 1], \
                                   rgb_array_c[i, j, 2], a

    return pygame.image.frombuffer(new_array_c, (w, h), 'RGBA')




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_opaque_c(image_: Surface, int alpha_):
    """
    Increase opacity of a texture by adding a value to the entire alpha channel.
    <image_>must be a 32-bit pygame surface containing per-pixel information,
    otherwise raise a value error. 
    ValueError: unsupported color masks for alpha reference array.
    
    <alpha_> must be an integer in range [0...255]
    <alpha_> = 0 output image is identical to the source
    <alpha_> = 255 output image is 100% opaque.
    
    TIP: Do not apply pygame convert_alpha() or convert() to modify pixels format to the surface 
         before calling make_opaque function. The output image will have BGRA colour format 
         instead of RGBA. 
    
    EXAMPLE: 
        image = pygame.image.load('path to your image here')
        output = make_opaque(image, 100) 
    
    :param image_: pygame surface
    :param alpha_: integer in range [0...255]
    :return: Return a 32-bit pygame surface containing per-pixel alpha transparency.
    """
    # below lines will slow down the algorithm (converting c variable into python object)
    assert isinstance(alpha_, int), \
        'Expecting int for positional argument alpha_ got %s ' % type(alpha_)
    assert isinstance(image_, Surface), \
        'Expecting Surface for positional argument image_ got %s ' % type(image_)

    if not (image_.get_flags() & pygame.SRCALPHA):
        raise ValueError('Surface without per-pixel information.')

    cdef int w_, h_
    w_, h_ = image_.get_size()
    if w_==0 or h_==0:
        raise ValueError('image has incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    try:
        # create a buffer_
        buffer_ = numpy.frombuffer(image_.get_view('2'),
                                   dtype=uint8).reshape((w_, h_, 4))
    except (pygame.error, ValueError):
        raise ValueError('\nSurface incompatible.')

    # Create array (w, h, 4) containing contiguous RGBA values
    cdef unsigned char [:, :, ::1] array_ = buffer_
    cdef:
        int w = w_
        int h = h_
        int i=0, j=0, a
    with nogil:
        for i in prange(w):
            for j in range(h):
                a = array_[i, j, 3]
                if a < 255:
                    a += alpha_
                    if a > 255:
                        a = 255
                    array_[i, j, 3] = a

    return pygame.image.frombuffer(array_, (w_, h_), 'RGBA')

# TODO BUILD IDENTICAL FUNCTION WITH RGBA ARRAY AS ARGUMENT INSTEAD

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
        im1 = pygame.image.load('path to your image here')
        rgb = pygame.surfarray.pixels3d(im1)
        alpha = pygame.surfarray.pixels_alpha()
        output = make_array_opaque(rgb, alpha, 100) 
    
    :param rgb_array_: numpy.ndarray (w, h, 3) (uint8) RGB array values 
    :param alpha_array_: numpy.ndarray (w, h) uint8, ALPHA values
    :param alpha_: integer in range [0...255], Value to add to alpha array
    :return: Returns 32-bit pygame surface with per-pixel information
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

    cdef int shape[2]
    shape = (<object> rgb_array_).shape[:2]

    if shape[0] == 0 or shape[1] == 0:
        raise ValueError('<rgb_array_> with incorrect shapes.')

    cdef:
        int w = shape[0]
        int h = shape[1]
        unsigned char [:, :, :] rgb_array_c = rgb_array_                        # non-contiguous values
        unsigned char [:, ::1] alpha_array_c = alpha_array_                     # contiguous values
        unsigned char [: , :, ::1] new_array_c = empty((h, w, 4), dtype=uint8)  # contiguous values
        int i=0, j=0, a=0

    with nogil:
        for i in prange(w):
            for j in range(h):
                a = alpha_array_c[i, j]
                if a < 255:
                    a += alpha_
                    if a > 255:
                        a = 255
                new_array_c[j, i, 0], new_array_c[j, i, 1], new_array_c[j, i, 2], \
                new_array_c[j, i, 3] =  rgb_array_c[i, j, 0], rgb_array_c[i, j, 1], \
                                   rgb_array_c[i, j, 2], a

    return pygame.image.frombuffer(new_array_c, (w, h), 'RGBA')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef low_threshold_alpha_c(surface_: Surface, int new_alpha_, int threshold_):
    """
    Filter all pixels (R, G, B) from a given surface (Re-assign alpha values 
    of all pixels (with sum of RGB values) < <threshold_> to 0)
    
    Compatible with 24 - 32 bit surface with or without per-pixel transparency 
    
    TIP: Do not apply pygame convert_alpha() or convert() to modify pixels format to the surface 
         before calling low_th_alpha function. Output image will have (B, G, R, A) colour format 
         instead of RGBA. 
      
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

    cdef int w_, h_
    w_, h_ = surface_.get_width(), surface_.get_height()

    if w_==0 or h_==0:
        raise ValueError('Surface with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    try:
        # '2' returns a (surface-width, surface-height) array of raw pixels.
        # The pixels are surface-bytesize-d unsigned integers. The pixel
        # format is surface specific. The 3 byte unsigned integers of 24
        # bit surfaces are unlikely accepted by anything other than other pygame functions.
        buffer_ = surface_.get_view('2')
        # Reshape the buffer into a numpy array (w, h, 4) type uint8
        rgb_array = numpy.frombuffer(buffer_, dtype=uint8).reshape((w_, h_, 4))

    except (pygame.error, ValueError):
        raise ValueError('\nSurface incompatible.')

    cdef:
        int w = w_
        int h = h_
        unsigned char [:, :, :] source_array = rgb_array    # non-contiguous array
        int i=0, j=0, v
        float c1 = 1.0/3.0
    with nogil:
        for i in prange(w):
            for j in range(h):
                if source_array[i, j, 3] != new_alpha_:
                    # SUM(R + G + B) / 3
                    v = <int>((source_array[i, j, 0] + source_array[i, j, 1] + source_array[i, j, 2]) * c1)
                    if v < threshold_:
                        source_array[i, j, 3] = new_alpha_

    return pygame.image.frombuffer(source_array, (w_, h_), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef high_threshold_alpha_c(surface_: Surface, int new_alpha_, int threshold_):
    """
    Filter all pixels (R, G, B) from a given surface (Re-assign alpha values 
    of all pixels (with sum of RGB values) > <threshold_> to 0)
    
    Compatible with 24 - 32 bit surface with or without per-pixel transparency 
    
    TIP: Do not apply pygame convert_alpha() or convert() to modify pixels format to the surface 
         before calling low_th_alpha function. Output image will have (B, G, R, A) colour format 
         instead of RGBA. 
      
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

    cdef int w_, h_
    w_, h_ = surface_.get_width(), surface_.get_height()

    if w_==0 or h_==0:
        raise ValueError('Surface with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    try:
        # '2' returns a (surface-width, surface-height) array of raw pixels.
        # The pixels are surface-bytesize-d unsigned integers. The pixel
        # format is surface specific. The 3 byte unsigned integers of 24
        # bit surfaces are unlikely accepted by anything other than other pygame functions.
        buffer_ = surface_.get_view('2')
        # Reshape the buffer into a numpy array (w, h, 4) type uint8
        array_ = numpy.frombuffer(buffer_, dtype=uint8).reshape((w_, h_, 4))

    except (pygame.error, ValueError):
        raise ValueError('\nSurface incompatible.')

    cdef:
        int w = w_
        int h = h_
        unsigned char[:, :, :] source_array = array_    # non-contiguous array
        int i=0, j=0
        int v = 0
        float c1 = 1.0 / 3.0
    with nogil:
        for i in prange(w):
            for j in range(h):
                if source_array[i, j, 3] != new_alpha_:
                    # (R + G + B) / 3
                    v = <int>((source_array[i, j, 0] + source_array[i, j, 1] + source_array[i, j, 2]) * c1)
                    if v > threshold_:
                        source_array[i, j, 3] = new_alpha_

    return pygame.image.frombuffer(source_array, (w_, h_), 'RGBA')


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
    """
    # TODO try cv2.imread
    assert isinstance(image, Surface), \
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
        alpha_ = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef:
        unsigned char [:, :, :] pixels = array_  # non-contiguous values
        unsigned char [:, ::1] alpha = alpha_    # contiguous values
        int w, h

    w, h = image.get_size()
    if w==0 or h==0:
        raise ValueError('Image with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, ::1] grey_c = empty((h, w, 4), dtype=uint8)  # contiguous values
        int i=0, j=0, lightness
        unsigned char r, g, b

    with nogil:
        for i in prange(w):
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
        for i in prange(w):
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
    """
    # TODO CAN GAIN PERFORMANCES WITH cv2.imread
    assert isinstance(image, Surface), \
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
        alpha_ = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef:
        unsigned char [:, :, :] pixels = array_ # non-contiguous values
        unsigned char [:, ::1] alpha = alpha_   # contiguous values
        int w, h

    w, h = image.get_size()
    if w==0 or h==0:
        raise ValueError('Image with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, ::1] grey_c = empty((h, w, 4), dtype=uint8)    # contiguous values
        int i=0, j=0, luminosity
        unsigned char r, g, b

    with nogil:
        for i in prange(w):
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
        for i in prange(w):
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
cdef make_greyscale_alpha_c(image):
    """
    Transform an image into greyscale.
    The image must have per-pixel information encoded otherwise
    a ValueError will be raised (8, 24, 32 format image converted with
    pygame convert_alpha() method will works fine).
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = make_greyscale_alpha(im1)  
        
    :param image: pygame surface with alpha channel 
    :return: Return Greyscale 32-bit Surface with alpha channel  
    """

    assert isinstance(image, Surface), \
        'Argument image is not a valid Surface got %s ' % type(image)

    try:
        array_ = pixels3d(image)
        alpha_ = pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible image.')

    cdef:
        unsigned char [:, :, :] pixels = array_ # non-contiguous values
        unsigned char [:, ::1] alpha = alpha_     # contiguous values
        int w, h

    w, h = image.get_size()

    if w==0 or h==0:
        raise ValueError('Image with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))

    cdef:
        unsigned char [:, :, ::1] grey_c = empty((h, w, 4), dtype=uint8)    # contiguous values
        int i=0, j=0
        unsigned char grey_value
        double c1 = 1.0/3.0
    with nogil:
        for i in prange(w):
            for j in range(h):
                grey_value = <unsigned char>((pixels[i, j, 0] + pixels[i, j, 1] + pixels[i, j, 2]) * c1)
                grey_c[j, i, 0], grey_c[j, i, 1], grey_c[j, i, 2], \
                    grey_c[j, i, 3] = grey_value, grey_value, grey_value, alpha[i, j]

    return pygame.image.frombuffer(grey_c, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_greyscale_c(image):
    """
    Transform a pygame surface into a greyscale.
    Compatible with 8, 24-32 bit format image with or without
    alpha channel.
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        output = make_greyscale(im1)  
        
    :param image: pygame surface 8, 24-32 bit format
    :return: Returns a greyscale 24-bit surface without alpha channel
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
        for i in prange(w):
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
        unsigned char [:, ::1] alpha_array = alpha
        unsigned char [:, :, ::1] grayscale_array  = empty((h, w, 4), dtype=uint8)  # contiguous
        int i=0, j=0
        unsigned char gray
    with nogil:
        for i in prange(w):
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
    Transform array into a greyscale Surface 
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        array = pygame.surfarray.pixels3d(im1)
        output = greyscale_arr2surf(array)  
    
    :param array: numpy.ndarray (w, h, 3) uint8 with RGB values 
    :return: Return a greyscale pygame surface (24-bit) without per-pixel information
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)
    cdef int w_, h_
    w_, h_ = array.shape[:2]

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
        for i in prange(w):
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
    Transform an RGB color array (w, h, 3) into a greyscale array same size.
    
    EXAMPLE:
        im1 = pygame.image.load('path to your image here')
        array = pygame.surfarray.pixels3d(im1)
        greyscale_array = greyscale_array(array)  
    
    :param array: numpy.ndarray (w, h, 3) uint8 containing RGB values 
    :return: Returns a numpy.ndarray (w, h, 3) uint8 with greyscale values 
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)

    cdef int w_, h_
    w_, h_ = array.shape[:2]

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
        for i in prange(w):
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
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)
    cdef int w_, h_
    w_, h_ = array.shape[:2]

    if w_==0 or h_==0:
        raise ValueError('array with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array
        unsigned char[:, ::1] rgb_out = empty((w, h), dtype=uint8)
        int red, green, blue
        int i=0, j=0

    with nogil:
        for i in prange(w):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb_out[i, j] =  <int>((red + green + blue) * 0.33)

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
    """

    assert isinstance(array, numpy.ndarray),\
        'Argument array is not numpy.ndarray got %s ' % type(array)

    cdef int w_, h_
    w_, h_ = array.shape[:2]

    if w_==0 or h_==0:
        raise ValueError('array with incorrect dimensions, must be (w>0, h>0) got (w:%s, h:%s) ' % (w_, h_))

    cdef:
        int w = w_, h = h_
        unsigned char[:, :] rgb_array = array
        unsigned char[:, :, ::1] rgb_out = empty((w, h, 3), dtype=uint8)
        int grey
        int i=0, j=0
    with nogil:
        for i in prange(w):
            for j in range(h):
                grey = rgb_array[i, j]
                rgb_out[i, j, 0], rgb_out[i, j, 1], rgb_out[i, j, 2] =  grey, grey, grey

    return asarray(rgb_out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef redscale_c(surface_:Surface):
    """
    Create a redscale image from the given Surface (compatible with 8, 24-32 bit format image)
    
    :param surface_: pygame surface 8, 24-32 bit format
    :return: Return a redscale pygame surface 24bit without alpha channel.
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        int luminosity

    with nogil:
        for i in prange(width):
            for j in range(height):
                luminosity =\
                <unsigned char>(rgb_array[i, j, 0] * 0.2126 + rgb_array[i, j, 1] * 0.7152 + rgb_array[i, j, 2] * 0.0722)
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2], = luminosity, 0, 0

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
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, ::1] alpha_array = alpha_
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
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        int luminosity

    with nogil:
        for i in prange(width):
            for j in range(height):
                luminosity =\
                <unsigned char>(rgb_array[i, j, 0] * 0.2126 +
                                rgb_array[i, j, 1] * 0.7152 + rgb_array[i, j, 2] * 0.0722)
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2], = 0, luminosity, 0

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
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
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
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
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
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2], = 0, 0, luminosity

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
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int width, height
    width, height = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, ::1] alpha_array = alpha_
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
        # stored in the order of Blue, Green and Red respectively.
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
    width, height, dim = bgra_array.shape[:3]
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
    width, height, dim = bgr_array.shape[:3]

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
    width, height = (<object> rgb_array_c).shape[:2]
    cdef:
        unsigned char[:, :, ::1] new_array =  empty((width, height, 4), dtype=uint8)
        int i=0, j=0
    # Equivalent to a numpy dstack
    with nogil:
        for i in prange(width):
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
    width, height = (<object> rgb_array_c).shape[:2]
    cdef:
        unsigned char[:, :, ::1] new_array =  empty((height, width, 4), dtype=uint8)
        int i=0, j=0

    with nogil:
        for i in prange(width):
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
cdef shadow_c(image):
    """
    Transform an image into a greyscale image with alpha channel
    The image must be converted with the method convert_alpha() or contains
    per-pixels information otherwise and error message will be thrown.
    
    :param image: Surface containing alpha channel 
    :return: Return a greyscale pygame surface (32-bit format) with alpha channel
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
        for i in prange(w):
            for j in range(h):
                gray = <int>((rgb_array[i, j, 0] + rgb_array[i, j, 1] + rgb_array[i, j, 2] ) *  0.01)
                greyscale_array[j, i, 0], greyscale_array[j, i, 1], greyscale_array[j, i, 2] = gray, gray, gray
                greyscale_array[j, i, 3] = alpha_array[i, j]
    return pygame.image.frombuffer(greyscale_array, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef rgb_split_channels_c(surface_: Surface):
    """
    Split RGB channels from a given Surface, image should be colorfull otherwise
    R,G,B channels will be identical.
    Image can be converted to fast blit with convert() or convert_alpha()
    Return a tuple containing 3 surfaces (R, G, B) with no alpha channel
    
    :param surface_: Surface 8, 24-32 bit format 
    :return: Return a tuple containing 3 surfaces (R, G, B) with no alpha channel
    """

    cdef int width, height
    width, height = surface_.get_size()
    assert isinstance(surface_, Surface), \
        '\nPositional argument surface_ must be a Surface, got %s ' % type(surface_)
    if width == 0 or height == 0:
        raise ValueError('\nIncorrect pixel size or wrong format.'
                         '\nsurface_ dimensions (width, height) cannot be null.')
    try:
        r = pygame.surfarray.pixels_red(surface_)
        g = pygame.surfarray.pixels_green(surface_)
        b = pygame.surfarray.pixels_blue(surface_)
    except (pygame.error, ValueError):
        raise ValueError('Incompatible surface.')
        
    cdef:
        unsigned char [:, :, ::1] red_s = zeros((height, width, 3), dtype=uint8)
        unsigned char [:, :, ::1] green_s = zeros((height, width, 3), dtype=uint8)
        unsigned char [:, :, ::1] blue_s = zeros((height, width, 3), dtype=uint8)
        unsigned char [:, :] red = r
        unsigned char [:, :] green = g
        unsigned char [:, :] blue = b
        int i=0, j=0
        
    with nogil:
        for i in prange(width):
            for j in range(height):
                red_s[j, i, 0] = red[i, j]
                green_s[j, i, 1] = green[i, j]
                blue_s[j, i, 2] = blue[i, j]

    return pygame.image.frombuffer(red_s, (width, height), 'RGB'),\
           pygame.image.frombuffer(green_s, (width, height), 'RGB'),\
           pygame.image.frombuffer(blue_s, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef rgb_split_channels_alpha_c(surface_: Surface):
    """
    Split RGB channels from a given Surface, image should be colorfull otherwise
    RGB channels will be identical.
    Image can be converted with method convert_alpha() or contains alpha channel otherwise an
    error message will be thrown.
    Return a tuple containing 3 surfaces (R, G, B) with per-pixel transparency.
    
    :param surface_: Surface 8, 24-32 bit format with alpha channel
    :return: Return a tuple containing 3 surfaces (R, G, B) with alpha channel
    """

    assert isinstance(surface_, Surface), \
        '\nPositional argument surface_ must be a Surface, got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()
    
    if w == 0 or h == 0:
        raise ValueError('\nIncorrect pixel size or wrong format.'
                         '\nsurface_ dimensions (width, height) cannot be null.')
    try:
        alpha_array = pixels_alpha(surface_)
        
    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nMake sure the surface_ contains per-pixel alpha transparency values.')

    try:
        r = pygame.surfarray.pixels_red(surface_)
        g = pygame.surfarray.pixels_green(surface_)
        b = pygame.surfarray.pixels_blue(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible surface.')
    
    zeros = numpy.zeros((w, h, 4), dtype=uint8)
    cdef:
        unsigned char [:, :, ::1] red_s = zeros
        unsigned char [:, :, ::1] green_s = zeros[:]
        unsigned char [:, :, ::1] blue_s = zeros[:]
        unsigned char [:, :] red = r
        unsigned char [:, :] green = g
        unsigned char [:, :] blue = b
        unsigned char [:, ::1] alpha = alpha_array
        unsigned char a
        int i=0, j=0
    # RED CHANNEL
    with nogil:
        for i in prange(w):
            for j in range(h):
              a = alpha[i, j]
              red_s[j, i, 0], red_s[j, i, 3] = red[i, j], a
              green_s[j, i, 1], green_s[j, i, 3] = green[i, j], a
              blue_s[j, i, 2], blue_s[j, i, 3] = blue[i, j], a

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
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    try:
        rgb = pixels3d(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible surface.')

    cdef int w, h
    w, h = surface_.get_size()
    cdef unsigned char [:, :, :] rgba_array = rgb
    cdef unsigned char [:, :, ::1] empty_array = zeros((h, w, 3), dtype=uint8)
    cdef:
        int i = 0, j = 0
    with nogil:
        for i in prange(w):
            for j in range(h):
                empty_array[j, i, 0] = rgba_array[i, j, 0]
    return pygame.image.frombuffer(empty_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef green_channel_c(surface_: Surface):
    """
    Extract green channel from a given surface
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a Surface (green channel) without alpha channel.
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb = pixels3d(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible surface.')

    cdef unsigned char [:, :, :] rgba_array = rgb
    cdef unsigned char [:, :, ::1] empty_array = zeros((h, w, 3), dtype=uint8)
    cdef:
        int i=0, j=0
    with nogil:
        for i in prange(w):
            for j in range(h):
                empty_array[j, i, 1] = rgba_array[i, j, 1]
    return pygame.image.frombuffer(empty_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef blue_channel_c(surface_: Surface):
    """
    Extract blue channel from a given surface
    
    :param surface_: pygane.Surface 8, 24-32 format 
    :return:  Return a Surface (blue channel) without alpha channel.
    """
    assert isinstance(surface_, Surface),\
        'Positional argument surface_ must be a Surface, got %s ' % type(surface_)
    cdef int w, h
    w, h = surface_.get_size()

    try:
        rgb = pixels3d(surface_)
    except (pygame.error, ValueError):
        raise ValueError('\nIncompatible surface.')
    
    cdef unsigned char [:, :, :] rgba_array = rgb
    cdef unsigned char [:, :, ::1] empty_array = zeros((h, w, 3), dtype=uint8)
    cdef:
        int i=0, j=0
        
    with nogil:
        for i in prange(w):
            for j in range(h):
                empty_array[j, i, 2] = rgba_array[i, j, 2]
    return pygame.image.frombuffer(empty_array, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef fish_eye_c(image):
    """
    Transform an image into a fish eye lens model.
    
    :param image: Surface (8, 24-32 bit format) 
    :return: Return a Surface without alpha channel, fish eye lens model.
    """
    assert isinstance(image, Surface), \
        "\nArguement image is not a pygame.Surface type, got %s " % type(image)

    try:
        array = pixels3d(image)
    except (pygame.error, ValueError):
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nMake sure the surface_ contains per-pixel alpha transparency values.')

    cdef double w, h
    w, h = image.get_size()
   
    assert (w!=0 and h!=0),\
        'Incorrect image format (w>0, h>0) got (w:%s h:%s) ' % (w, h)

    cdef:
        cdef unsigned char [:, :, :] rgb_array = array
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
        for y in prange(<int>h, schedule='static', chunksize=8):
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
    Rotate hue for a given pygame surface.
    
    :param surface_: Surface 8, 24-32 bit format 
    :param shift_: pygame float,  hue rotation
    :return: return a Surface with no alpha channel
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
            # unsupported colormasks for alpha reference array
            raise ValueError('\nThis will only work on Surfaces that have 24-bit')
    alpha_ = empty((height, width, 3), dtype=uint8)
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
        for i in prange(width, schedule='static', chunksize=4):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                rr = r / 255.0
                gg = g / 255.0
                bb = b / 255.0
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
                h = (h/360.0) + shift_

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
    Rotate hue for a given pygame surface.
    image must be encoded with per-pixel transparency or converted with method
    convert_alpha() otherwise an error message will be thrown.
    
    :param surface_: Surface 
    :param shift_: pygame float,  hue rotation
    :return: return a Surface with per-pixel information
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
        # unsupported colormasks for alpha reference array
        print('\nUnsupported colormasks for alpha reference array.')
        raise ValueError('\nThis will only work on Surfaces that have 24-bit or 32-bit formats')

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
        # for i in prange(width, schedule='static', chunksize=4):
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                rr = r / 255.0
                gg = g / 255.0
                bb = b / 255.0
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

                h = h/360.0 + shift_

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
cdef float * rotate_hue(float shift_, float r, float g, float b)nogil:

    cdef:
        float h, s, v
        float rr, gg, bb, mx, mn
        float df, df_
        float f, p, q, t, ii
        float *rgb = <float *> malloc(3 * sizeof(float))

    rr = r / 255.0
    gg = g / 255.0
    bb = b / 255.0
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
    h = (h/360.0) + shift_

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
    return rgb

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hsah_c(surface_: Surface, int threshold_, double shift_):
    """
    Hue Surface Average Low (hsal)
    Rotate hue for pixels with average value <= threshold.
    
    :param surface_: Surface 8, 24-32 bit format
    :param threshold_: integer, threshold value 
    :param shift_: pygame float,  hue rotation value
    :return: return a Surface with no alpha channel
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
            # unsupported colormasks for alpha reference array
            raise ValueError('\nThis will only work on Surfaces that have 24-bit')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b, s
        float *rgb = <float *> malloc(3 * sizeof(float))
    with nogil:
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                s = r + g + b
                if s>=threshold_:
                    rgb = rotate_hue(shift_, r, g, b)
                    r = <unsigned char>(min(rgb[0] * 255.0, 255.0))
                    g = <unsigned char>(min(rgb[1] * 255.0, 255.0))
                    b = <unsigned char>(min(rgb[2] * 255.0, 255.0))
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2] = r, g, b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hsal_c(surface_: Surface, int threshold_, double shift_):
    """
    Hue Surface Average Low (hsal)
    Rotate hue for pixels with average value <= threshold.
    
    :param surface_: Surface 8, 24-32 bit format
    :param threshold_: integer, threshold value in range [0 ... 255]
    :param shift_: pygame float,  hue rotation value in range [0.0 ... 1.0]
    :return: return a Surface with no alpha channel
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
            # unsupported colormasks for alpha reference array
            raise ValueError('\nThis will only work on Surfaces that have 24-bit')

    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b, s
        float *rgb = <float *> malloc(3 * sizeof(float))
    with nogil:
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                s = r + g + b
                if s<=threshold_:
                    rgb = rotate_hue(shift_, r, g, b)
                    r = <unsigned char>(min(rgb[0] * 255.0, 255.0))
                    g = <unsigned char>(min(rgb[1] * 255.0, 255.0))
                    b = <unsigned char>(min(rgb[2] * 255.0, 255.0))
                new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2] = r, g, b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_array_red_c(array_, double shift_):
    """
    Rotate red pixels hue from a given colour array.
    
    :param array_: numpy.ndarray (w, h, 3) uint8 colours values. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: numpy.ndarray (w, h, 3) uint8 with red colours shifted 
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
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb = rotate_hue(shift_, r, g, b)
                new_array[i, j, 0], new_array[i, j, 1], \
                new_array[i, j, 2] = min(<unsigned char>(rgb[0]*255.0), 255), g, b

    return asarray(new_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_array_green_c(array_, double shift_):
    """
    Rotate red pixels hue from a given colour array.
    
    :param array_: numpy.ndarray (w, h, 3) uint8 colours values. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: numpy.ndarray (w, h, 3) uint8 with red colours shifted 
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
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                rgb = rotate_hue(shift_, r, g, b)
                gg = rgb[1]*255.0
                new_array[i, j, 0], new_array[i, j, 1], \
                new_array[i, j, 2] = r, min(<unsigned char>gg, 255), b

    return asarray(new_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef hue_array_blue_c(array_, double shift_):
    """
    Rotate red pixels hue from a given colour array.
    
    :param array_: numpy.ndarray (w, h, 3) uint8 colours values. 
    :param shift_: pygame float,  hue rotation in range [0.0 ... 1.0]
    :return: numpy.ndarray (w, h, 3) uint8 with red colours shifted 
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
        float bb

    with nogil:
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                rgb = rotate_hue(shift_, r, g, b)
                bb =  rgb[2]*255.0
                if bb > 255.0:
                    bb =255.0
                new_array[i, j, 0] = r
                new_array[i, j, 1] = g
                new_array[i, j, 2] = <unsigned char>bb

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
        double r, g, b
        double h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]
        float c1 = 1.0 / 255.0
        float maxc, minc, rc, gc, bc
        float high, low, high_

    with nogil:
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                hls = rgb_to_hls(r * c1, g *c1, b * c1)
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
                new_array[j, i, 2] = <int>r, <int>g, <int>b

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
            raise ValueError('\nThis will only work on Surfaces that have 32-bit format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, :] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char [:, ::1] alpha_array = alpha_
        int i=0, j=0
        float r, g, b
        float h, l, s
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]
        float c1 = 1.0 / 255.0

    with nogil:
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]

                hls = rgb_to_hls(r * c1, g *c1, b * c1)
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
    Change the brightness level of a pygame.Surface (compatible with 24-bit format image)
    Change the lightness of an image by decreasing/increasing the sum of RGB values.
    
    :param surface_: pygame.Surface 24 bit format 
    :param shift_: Value must be in range [-1.0 ... 1.0]
    :return: a pygame.Surface 24 bit without per-pixel transparency 
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
            raise ValueError('\nThis will only work on Surfaces that have 24-32 bit format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        float r, g, b

    with nogil:
        for i in prange(width):
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
            raise ValueError('\nThis will only work on Surfaces with 24-32 bit format.')
    cdef:
        unsigned char [:, :, :] rgb_array = rgb_
        unsigned char [:, :, ::1] new_array = zeros((height, width, 4), dtype=uint8)
        unsigned char [:, ::1] alpha_array = alpha_
        int i=0, j=0
        float r, g, b

    with nogil:
        for i in prange(width):
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
        float c1 = 1.0/255.0
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                hls = rgb_to_hls(r * c1, g * c1, b * c1)
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
        float c1 = 1.0/255.0
        double *hls = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width):
            for j in range(height):
                r, g, b = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                hls = rgb_to_hls(r * c1, g * c1, b * c1)
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

    cdef int w, h
    w, h, dimension = (<object> array).shape[:2]

    if w == 0 or h == 0:
        raise ValueError('Incorrect array dimension must be (w>0, h>0) got (w:%s, h:%s) ' % (w, h))
    if dimension != 3:
        raise ValueError('Incompatible array, must be (w, h, 3) got (%s, %s, %s) ' % (w, h, dimension))
    
    zero = zeros((w, h, 3), dtype=uint8)
    cdef:
        int dim = dimension
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
            for i in prange(w):
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
            for i in prange(w):
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
cdef scroll_surface_c(array, int dy, int dx):
    """
    
    :param array: numpy.ndarray (w,h,3) uint8 (array to scroll)
    :param dy: scroll the array vertically (-dy up, +dy down) 
    :param dx: scroll the array horizontally (-dx left, +dx right)
    :return: Return a tuple (surface:Surface, array:numpy.ndarray) type (w, h, 3) numpy uint8
    """
    if not isinstance(dx, int):
        raise TypeError('dx, an integer is required (got type %s)' % type(dx))
    if not isinstance(dy, int):
        raise TypeError('dy, an integer is required (got type %s)' % type(dy))
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array, a numpy.ndarray is required (got type %s)' % type(array))

    cdef int w, h
    w, h, dimension = (<object> array).shape[:2]

    if w == 0 or h == 0:
        raise ValueError('Incorrect array dimension must be (w>0, h>0) got(w:%s, h:%s) ' % (w, h))
    if dimension != 3:
        raise ValueError('Incompatible array, must be (w, h, 3) got (%s, %s %s) ' % (w, h, dimension))

    zero = zeros((w, h, 3), dtype=uint8)
    cdef:
        int dim = dimension
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
            for i in prange(w):
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
            for i in prange(w):
                for j in range(h):
                    jj = (j + dy) % h
                    if jj < 0:
                        jj = jj + h
                    # printf("\n%i %i %i", jj, w, h)
                    tmp_array[i, jj, 0], tmp_array[i, jj, 1], tmp_array[i, jj, 2] = \
                        tmp_array[i, j, 0], tmp_array[i, j, 1], tmp_array[i, j, 2]

    return pygame.image.frombuffer(tmp_array, (w, h), 'RGB'), asarray(tmp_array)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gradient_array_c(int width, int height, start_color, end_color):

    """
    Create an RGB gradient array  (w, h, 3).
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
    :return: returns an RGB gradient array type (w, 1, 3) uint8.
    """
    assert width != 0, 'Positional argument width cannot be equal 0'
    cdef:
        double [:] diff_ =  numpy.array(end_color, dtype=float64) - \
                            numpy.array(start_color, dtype=float64)
        double [::1] row = numpy.arange(width) / (width - 1)
        unsigned char [:, :, ::1] rgb_gradient = empty((width, height, 3), dtype=uint8)
        double [3] start = numpy.array(start_color)
        int i=0, j=0
    with nogil:
        for i in prange(width):
            for j in range(height):
               rgb_gradient[i, j, 0], rgb_gradient[i, j, 1],\
               rgb_gradient[i, j,  2] = <unsigned char>(start[0] + row[i] * diff_[0]), \
               <unsigned char>(start[1] + row[i] * diff_[1]), <unsigned char>(start[2] + row[i] * diff_[2])

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
    width, height = (<object>gradient).shape[:2]
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
        unsigned char [:, ::1] alpha = alpha_channel
        unsigned char [:] f_color = numpy.array(final_color_[:3], dtype=uint8)  # take only rgb values
        int c1, c2, c3
        float c4 = 1.0 / steps # division by zero is checked above with assert statement
        int i=0, j=0

    with nogil:
        for i in prange(w):
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
        for i in prange(w):
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
        for i in prange(w):
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
        for i in prange(w):
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
        raise ValueError('Surface incompatible.')

    cdef int w, h, dim
    w, h, dim = array_.shape[:3]
    
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % \
                (w, h, dim)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] inverted_array  = empty((h, w, 3), dtype=uint8)
        int i=0, j=0
    with nogil:
        for i in prange(w):
            for j in range(h):
                inverted_array[j, i, 0] = 255 -  rgb_array[i, j, 0]
                inverted_array[j, i, 1] = 255 -  rgb_array[i, j, 1]
                inverted_array[j, i, 2] = 255 -  rgb_array[i, j, 2]
    return pygame.image.frombuffer(inverted_array, (w, h), 'RGB')


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
    :return: return a pygame Surface
    """
    try:
        array_ = pixels3d(image)
        alpha_ = pixels_alpha(image)
        
    except (pygame.error, ValueError):
        raise ValueError('Surface incompatible.')

    cdef int w, h, dim
    w, h, dim = array_.shape[:3]
    assert w != 0 or h !=0,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % \
                (w, h, dim)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, ::1 ] alpha_array = alpha_
        unsigned char [:, :, ::1] inverted_array  = empty((h, w, 4), dtype=uint8)
        int i=0, j=0
    with nogil:
        for i in prange(w):
            for j in range(h):
                inverted_array[j, i, 0] = 255 -  rgb_array[i, j, 0]
                inverted_array[j, i, 1] = 255 -  rgb_array[i, j, 1]
                inverted_array[j, i, 2] = 255 -  rgb_array[i, j, 2]
                inverted_array[j, i, 3] = alpha_array[i, j]
    return pygame.image.frombuffer(inverted_array, (w, h), 'RGBA')


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
        for i in prange(w):
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
    w, h, dim = array_.shape[:3]
    
    assert w != 0 or h !=0 or dim!=4,\
            'Array with incorrect shape (w>0, h>0, 3) got (w:%s, h:%s, %s) ' % \
                (w, h, dim)
    cdef:
        unsigned char [:, :, :] rgb_array = array_
        unsigned char [:, :, ::1] inverted_array  = empty((h, w, 4), dtype=uint8)
        int i=0, j=0
    with nogil:
        for i in prange(w):
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
    w, h, dim = rgb_array.shape[:3]
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
    
    rows, cols = rgb_array.shape[:2]
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

    rows, cols = rgb_array.shape[:2]
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
    w, h = (<object> rgb_array_).shape[:2]
    if w == 0 or h == 0:
        raise ValueError('Array rgb_array_ has incorrect shapes.')
    w, h = (<object> alpha_array_).shape[:2]
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
        raise ValueError('Incompatible pygame surface.')

    cdef int w, h
    w, h = (<object> rgb_array_).shape[:2]
    if w == 0 or h == 0:
        raise ValueError('Array rgb_array_ has incorrect shapes.')
    w, h = (<object> alpha_array_).shape[:2]
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
        raise ValueError('Incompatible pygame surface.')

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
        raise ValueError('Incompatible pygame surface.')

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
    w, h = background_rgb.shape[:2]
    lx, ly = (<object>mask_alpha).shape[:2]
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
    w, h = background_rgb.shape[:2]
    lx, ly = (<object>mask_alpha).shape[:2]
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
    Create a volumetric light effect whom shapes and radial intensity is provided by the radial mask texture. 
    This algorithm gives a better light effect illusion than a pure texture (radial mask) blend using pygame
    special flags pygame.BLEND_RGBA_ADD or BLEND_RGBA_MAX.
    Intensity can be constant or variable to create flickering or light attenuation effect
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
    w, h = (<object>alpha).shape[:2]

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
    w, h = (<object>alpha).shape[:2]
    vol_width, vol_height = (<object>volume).shape[:2]

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
    w, h = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
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
cdef sepia32_c(surface_:Surface):
    """
    Create a sepia image from the given Surface (compatible with 8, 24-32 bit with
    per-pixel information (image converted to convert_alpha())
    Surface converted to fast blit with pygame method convert() will raise a ValueError (Incompatible Surface).
     
    :param surface_:  Pygame.Surface converted with pygame method convert_alpha. Surface without per-pixel transparency
    will raise a ValueError (Incompatible Surface).
    :return: Returns a pygame.Surface (surface with per-pixel transparency)
    """

    assert isinstance(surface_, Surface), \
           'Expecting Surface for argument surface_ got %s ' % type(surface_)

    cdef int w, h
    w, h = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
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


#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef unsepia24_c(surface_:Surface):
#     """
#     Cancel the sepia effect of an image from the given Surface (compatible with 8, 24-32 bit format image)
#     Alpha channel will be ignored from image converted with the pygame method convert_alpha.
#
#     :param surface_: Surface, loaded with pygame.image method
#     :return: Return a Surface in sepia ready to be display, the final image will not hold any per-pixel
#     transparency layer
#     """
#
#     assert isinstance(surface_, Surface), \
#            'Expecting Surface for argument surface_ got %s ' % type(surface_)
#
#     cdef int width = surface_.get_size()[0]
#     cdef int height = surface_.get_size()[1]
#     try:
#         rgb_ = pixels3d(surface_)
#     except (pygame.error, ValueError):
#             # unsupported colormasks for alpha reference array
#             raise ValueError('\nIncompatible surface.')
#     cdef:
#         unsigned char [:, :, :] rgb_array = rgb_
#         unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
#         int i=0, j=0
#         int r, g, b
#     with nogil:
#         for i in prange(width):
#             for j in range(height):
#
#                 r = <int>(rgb_array[i, j, 0] * 1.0/0.393 +
#                           rgb_array[i, j, 1] * 1.0/0.769 + rgb_array[i, j, 2] * 1.0/0.189)
#                 g = <int>(rgb_array[i, j, 0] * 1.0/0.349 +
#                           rgb_array[i, j, 1] * 1.0/0.686 + rgb_array[i, j, 2] * 1.0/0.168)
#                 b = <int>(rgb_array[i, j, 0] * 1.0/0.272 +
#                           rgb_array[i, j, 1] * 1.0/0.534 + rgb_array[i, j, 2] * 1.0/0.131)
#
#                 new_array[j, i, 0], new_array[j, i, 1], new_array[j, i, 2], = min(r, 255), min(g, 255), min(b, 255)
#
#     return pygame.image.frombuffer(new_array, (width, height), 'RGB')


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
    w, h = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
    cdef:
        float [:, :, :] rgb_array = rgb_.astype(float32)
        unsigned char [:, :, ::1] reduce = zeros((h, w, 3), uint8, order='C')
        int x=0, y=0

        float new_red, new_green, new_blue
        float oldr, oldg, oldb
        float c1 = 1.0 / 255.0

    with nogil:
        for y in prange(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * c1) * (255.0 / factor)
                new_green = round(factor * oldg * c1) * (255.0 / factor)
                new_blue = round(factor * oldb *c1) * (255.0 / factor)

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
    w, h = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
    cdef:
        float [:, :, :] rgb_array = rgb_.astype(float32)
        unsigned char [:, ::1] alpha = alpha_
        unsigned char [:, :, ::1] reduce = zeros((h, w, 4), uint8, order='C')
        int x=0, y=0

        float new_red, new_green, new_blue
        float oldr, oldg, oldb
        float c1 = 1.0 / 255.0

    with nogil:
        for y in prange(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * c1) * (255.0 / factor)
                new_green = round(factor * oldg * c1) * (255.0 / factor)
                new_blue = round(factor * oldb *c1) * (255.0 / factor)

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
    w, h = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
    cdef:
        float [:, :, :] rgb_array = rgb_.astype(float32)
        int x=0, y=0

        float new_red, new_green, new_blue
        float quant_error_red, quant_error_green, quant_error_blue
        float c1 = 7.0/16.0
        float c2 = 3.0/16.0
        float c3 = 5.0/16.0
        float c4 = 1.0/16.0
        float c5 = 1.0/255.0
        float oldr, oldg, oldb

    with nogil:
        for y in range(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * c5) * (255.0 / factor)
                new_green = round(factor * oldg * c5) * (255.0 / factor)
                new_blue = round(factor * oldb * c5) * (255.0 / factor)

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
    w, h = surface_.get_size()[:2]

    try:
        rgb_ = pixels3d(surface_)
        alpha_ = pixels_alpha(surface_)

    except (pygame.error, ValueError):
            # unsupported colormasks for alpha reference array
            raise ValueError('\nIncompatible surface.')
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
        float c5 = 1.0/255.0
        float oldr, oldg, oldb

    with nogil:
        for y in range(h - 1):
            for x in range(1, w - 1):
                oldr = rgb_array[x, y, 0]
                oldg = rgb_array[x, y, 1]
                oldb = rgb_array[x, y, 2]

                new_red = round(factor * oldr * c5) * (255.0 / factor)
                new_green = round(factor * oldg * c5) * (255.0 / factor)
                new_blue = round(factor * oldb * c5) * (255.0 / factor)

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
    w, h = image.get_size()[:2]

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
    w, h = greyscale_array.get_size()[:2]

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
    w, h = greyscale_array.shape[:2]

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
    w, h = (<object>rgba_array).shape[:2]
    cdef:
        int i, j
        unsigned char [:, :, ::1] bgra_array = empty((w, h, 4), uint8)
    with nogil:
        for i in prange(w):
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
        for i in prange(w):
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
    w, h = (<object>bgra_array).shape[:2]

    cdef:
        int i, j
        unsigned char [:, :, ::1] rgba_array = zeros((w, h, 4), uint8)
    with nogil:
        for i in prange(w):
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
        for i in prange(w):
            for j in range(h):
                rgb_array[i, j, 0] = bgr_array[i, j, 2]   # red
                rgb_array[i, j, 1] = bgr_array[i, j, 1]   # green
                rgb_array[i, j, 2] = bgr_array[i, j, 0]   # blue
    return rgb_array

