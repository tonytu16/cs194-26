import numpy as np
import skimage.transform as sktr
import cv2
import skimage.data as data
import skimage as sk
from skimage.color import rgb2gray
import skimage.io as skio
import scipy.signal as signal
import matplotlib
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import math
from itertools import product
import os

# functions for p1
def convolution(img, kernel, use_color=True):
    return np.dstack([signal.convolve2d(img[:,:,i], kernel, mode='same') for i in range(3)]) \
        if use_color \
        else signal.convolve2d(img, kernel, mode='same')

def generateKernel(size, s):
    tmp = cv2.getGaussianKernel(size, s)
    ret = np.outer(tmp.T, tmp)
    s = np.sum(ret)
    return ret / s

def clip(img):
    return np.clip(img, 0, 1)

def normalize(img):
    if img.ndim > 2:
        img = img - img.min()
        img = img/img.max()
        return img
    # ndim == 2
    img = np.dstack([(img[:,:,i]-img[:,:,i].min()) for i in range(3)])
    img = np.dstack([img[:,:,i]/img[:,:,i].max() for i in range(3)])
    return img

img = skio.imread('data/cameraman.png')
img = data.camera()
print(img.shape)
img = sk.img_as_float(img)

# p1.1
D_x = np.array([[1, -1]])
img_dx = signal.convolve2d(img, D_x, mode='same')
grad_mag_dx = np.abs(img_dx)

D_y = np.array([[1], [-1]])
img_dy = signal.convolve2d(img, D_y, mode='same')
grad_mag_dy = np.abs(img_dy)

gradient_mag = np.sqrt(np.square(grad_mag_dx)+np.square(grad_mag_dy))
binnirized_gradent_mag = np.where(gradient_mag<0.12, 0, 1)

plt.title('gradient manitude')
plt.imshow(gradient_mag, cmap='gray')
plt.savefig('result/1-1-1.jpg')

plt.title('binirized gradient manitude')
plt.imshow(binnirized_gradent_mag, cmap='gray')
plt.savefig('result/1-1-2.jpg')


# p1.2
kernel = generateKernel(11, 1.5)

def genDerivative(D):
    d = signal.convolve2d(kernel, D)
    return signal.convolve2d(img, d, mode="same")

img_blur = signal.convolve2d(img, kernel)
dx2 = genDerivative(D_x)
dy2 = genDerivative(D_y)
d2 = np.sqrt(np.square(dx2)+np.square(dy2))
dx1 = signal.convolve2d(img_blur, D_x, mode="same")
dy1 = signal.convolve2d(img_blur, D_y, mode="same")
d1 = np.sqrt(np.square(dx1)+np.square(dy1))

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 2)
plt.title('gaussian + x-gradient as one filter')
plt.imshow(dx2, cmap='gray')
plt.subplot(1, 2, 1)
plt.title('gaussian then x-gradient')
plt.imshow(dx1, cmap='gray')
plt.savefig('result/1-2-1.jpg')

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 2)
plt.title('gaussian + y-gradient as one filter')
plt.imshow(dy2, cmap='gray')
plt.subplot(1, 2, 1)
plt.title('gaussian then y-gradient')
plt.imshow(dy1, cmap='gray')
plt.savefig('result/1-2-2.jpg')

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 2)
plt.title('gaussian + gradient as one filter')
plt.imshow(d2, cmap='gray')
plt.subplot(1, 2, 1)
plt.title('gaussian then gradient')
plt.imshow(d1, cmap='gray')
plt.savefig('result/1-2-3.jpg')


#p1.3
imgFiles = ['facade', 'mountain', 'tahoe', 'taj-mahal']
for name in imgFiles:
    img = skio.imread(os.path.join('data/', name + '.jpg'))
    img = sk.transform.resize(img, (600, 800))
    
    max_grad_angles = None
    max_r_img_crop = None
    max_deg = None
    max_hist = 0
    cut = 150
    for deg in range(-10, 12, 2):
        r_img = rotate(img, deg)
        r_img_crop = r_img[cut: r_img.shape[0]-cut, cut: r_img.shape[1]-cut]
        img_y = np.dstack( [np.abs(signal.convolve2d(r_img_crop[:, :, i], signal.convolve2d(kernel, D_y), mode='same')) for i in range(3)])
        grad_y = np.sum(img_y, axis = 2)
        img_x = np.dstack( [np.abs(signal.convolve2d(r_img_crop[:, :, i], signal.convolve2d(kernel, D_x), mode='same')) for i in range(3)])
        grad_x = np.sum(img_x, axis = 2)
        n, bins = np.histogram(np.arctan(np.nan_to_num(np.divide(grad_y, grad_x))).flatten(), bins=30)

        grad_angles = np.arctan(np.nan_to_num(np.divide(grad_y, grad_x))) 
        if n[len(n) - 1] > max_hist:
            max_r_img_crop = np.copy(r_img_crop)
            max_grad_angles = np.copy(grad_angles)
            max_hist = np.copy(n[-1])
            max_deg = deg

    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 2)
    plt.imshow(max_r_img_crop)
    plt.title('rotated ' + str(max_deg) + ' degrees')
    plt.subplot(1, 3, 3)
    plt.hist(max_grad_angles.flatten(), bins=30)
    plt.title('histogram')
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(name)
    plt.savefig('result/1-3-'+name+'.jpg')

#p2.1
def sharpen(input_img, alpha, mkernel=None):
    if mkernel is None:
        _kernel = -alpha * generateKernel(11, 1.5)
    else:
        _kernel = -alpha * mkernel
    mid_x, mid_y = _kernel.shape[0]//2, _kernel.shape[1]//2
    _kernel[mid_x, mid_y] += 1 + alpha
    return np.clip(convolution(input_img, _kernel), 0, np.inf)

# sharpen taj.jpg
img = skio.imread('data/taj.jpg')
img = sk.img_as_float(img)

# plt.figure(figsize=(16, 8))
plt.figure()
plt.title('sharpened image')
plt.imshow(sharpen(img, 2))
plt.savefig('result/2-1-1-2.jpg')

plt.title('original image')
plt.imshow(img)
plt.savefig('result/2-1-1-1.jpg')

# test img
img = plt.imread('data/lecture.jpeg')
img = sk.img_as_float(img)
blurred_img = convolution(img, generateKernel(21, 13))
sharpened_img = sharpen(blurred_img, 6, generateKernel(21, 13))

plt.title('original image')
plt.imshow(img)
plt.savefig('result/2-1-2-1.jpg')

plt.title('blurred image')
plt.imshow(blurred_img)
plt.savefig('result/2-1-2-2.jpg')

plt.title('re-sharpened image')
plt.imshow(sharpened_img)
plt.savefig('result/2-1-2-3.jpg')


#p2.2
def get_points(img1, img2):
    print('Please select 2 points in each image for alignment.')
    ret = []
    for img in (img1, img2):
        plt.figure()
        plt.imshow(img)
        p, q = plt.ginput(2)
        ret.append(p)
        ret.append(q)
        plt.close()
    return ret

def center_padding(img, r, c):
    rpad = (int) (np.abs(2*r - img.shape[0] + 1))
    cpad = (int) (np.abs(2*c - img.shape[1] + 1))

    R, C, _ = img.shape
    pad1 = (0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad)
    pad2 = (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)
    return np.pad(img, pad_width=[pad1, pad2, (0, 0)], mode='constant')

def get_centers(p1, p2):
    f = lambda x, y : (x + y) / 2
    return round(f(p1[0], p2[0])), round(f(p1[1], p2[1]))

def match_centers(img1, img2, pts):
    cx1, cy1 = get_centers(pts[0], pts[1])
    img1 = center_padding(img1, cy1, cx1)
    cx2, cy2 = get_centers(pts[2], pts[3])
    img2 = center_padding(img2, cy2, cx2)
    return (img1, img2)

def match_scale(img1, img2, pts):
    p1, p2, p3, p4 = pts
    norm = lambda p, q : np.sqrt((q[1]-p[1])**2 + (q[0]-p[0])**2)
    dscale = norm(p3, p4) / norm(p1, p2)
    if dscale > 1:
        img2 = np.dstack([sk.transform.rescale(img2[:, :, i], 1./dscale) for i in range(3)])
    else:
        img1 = np.dstack([sk.transform.rescale(img1[:, :, i], dscale) for i in range(3)])
    return (img1, img2)

def match_angle(img1, img2, pts):
    arcTan = lambda p, q : math.atan2(p[1] - q[1], q[0] - p[0])
    dtheta = arcTan(pts[2], pts[3]) - arcTan(pts[0], pts[1])
    img1 = sk.transform.rotate(img1, dtheta * (180/math.pi))
    return img1, img2

def match_size(img1, img2, pts):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    if w1 != w2:
        w_low = int(math.floor(abs(w2-w1)/2.)) 
        w_hi = -int(math.ceil(abs(w2-w1)/2.))
        if w1 < w2:
            img2 = img2[:, w_low : w_hi, :]
        else:
            img1 = img1[:, w_low : w_hi, :]
    if h1 != h2:
        h_low = int(math.floor(abs(h2-h1)/2.)) 
        h_hi = -int(math.ceil(abs(h2-h1)/2.))
        if h1 < h2:
            img2 = img2[h_low : h_hi, :, :]
        else:
            img1 = img1[h_low : h_hi, :, :]
    return (img1, img2)

def align_images(img1, img2):
    pts = get_points(img1, img2)
    for f in (match_centers, match_scale, match_angle, match_size):
        img1, img2 = f(img1, img2, pts)
    return img1, img2

def get_hybrid(img1, img2, use_color=True):
    img2_lp = convolution(img2, generateKernel(35, 27), use_color=use_color)
    
    hp_filter = -generateKernel(35, 27)
    midx, midy = hp_filter.shape[0]//2, hp_filter.shape[1]//2
    hp_filter[midx, midy] += 1.0
    img1_hp = convolution(img1, hp_filter, use_color=use_color)
    
    return clip((img1_hp + img2_lp)/2), img1_hp, img2_lp

def merge(name1, name2, out, use_color=False):
    img1 = plt.imread('data/' + name1)/255.  # high sf
    img2 = plt.imread('data/' + name2)/255.  # low sf

    img1_gray = rgb2gray(img1)
    img1_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img1_gray))))
    img2_gray = rgb2gray(img2)
    img2_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img2_gray))))
    img1_aligned, img2_aligned = align_images(img1, img2)
    img1_aligned_gray = rgb2gray(img1_aligned)
    img2_aligned_gray = rgb2gray(img2_aligned)
    if use_color:
        im_hybrid, img1_high, img2_low = get_hybrid(img1_aligned, img2_aligned, use_color=use_color)
    else:
        im_hybrid, img1_high, img2_low = get_hybrid(img1_aligned_gray, img2_aligned_gray, use_color=use_color)
    im_hybrid_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im_hybrid))))
    if use_color:
        plt.imsave('data/hybrid-' + out + '.jpg', im_hybrid)
    else:
        plt.imsave('data/hybrid-' + out + '.jpg', im_hybrid, cmap='gray')

    plt.figure(figsize=(10, 24))
    plt.subplot(3, 2, 1)
    plt.title('Fourier transform of input 1')
    if use_color:
        plt.imshow(img1_fft)
    else:
        plt.imshow(img1_fft, cmap='gray')

    plt.subplot(3, 2, 2)
    plt.title('Fourier transform of input 2')
    if use_color:
        plt.imshow(img2_fft)
    else:
        plt.imshow(img2_fft, cmap='gray')

    plt.subplot(3, 2, 3)
    plt.title('high frequency')
    if use_color:
        plt.imshow(img1_high)
    else:
        plt.imshow(img1_high, cmap='gray')

    plt.subplot(3, 2, 4)
    plt.title('low frequency')
    if use_color:
        plt.imshow(img2_low)
    else:
        plt.imshow(img2_low, cmap='gray')

    plt.subplot(3, 2, 5)
    plt.title('hybrid image')
    if use_color:
        plt.imshow(im_hybrid)
    else:
        plt.imshow(im_hybrid, cmap='gray')

    plt.subplot(3, 2, 6)
    plt.title('Fourier transform of hybrid image')
    if use_color:
        plt.imshow(im_hybrid_fft)
    else:
        plt.imshow(im_hybrid_fft, cmap='gray')

    plt.savefig('result/' + out + '-analysis.jpg')

merge('nutmeg.jpg', 'DerekPicture.jpg', '2-2-1')
merge('paint.jpg', 'water.jpg', '2-2-2')
merge('paint.jpg', 'water.jpg', '2-2-3', use_color=True)
# merge('whitehouse.jpg', 'sky.jpeg', '2-2-4', use_color=True)
merge('nutmeg.jpg', 'DerekPicture.jpg', '2-2-5', use_color=True)

# 2.3
LEVEL = 5

def im_stack(level, img, use_color=True):
    gaussian_stack = [img]
    for i in range(level):
        image = convolution(gaussian_stack[-1], generateKernel(6*2**(i), 2**(i+1)-1), use_color=use_color)
        gaussian_stack.append(image)
    # laplacian_stack = [(fst - sec) for fst, sec in zip(gaussian_stack[:-1], gaussian_stack[1:])]
    laplacian_stack = [(gaussian_stack[i] - gaussian_stack[i+1]) for i in range(len(gaussian_stack)-1)]
    laplacian_stack.append(gaussian_stack[-1])
    return (gaussian_stack, laplacian_stack)

def generateStack(img_name, save_name):
    img = plt.imread(img_name) / 255.
    img = sk.transform.resize(img, (800, 600))
    gaussian_stack, laplacian_stack = im_stack(LEVEL, img)

    plt.figure(figsize=(24, 10))
    length_gau = len(gaussian_stack)
    for i, image in enumerate(gaussian_stack):
        plt.subplot(2, length_gau, i+1)
        plt.imshow(image)

    len_lap = len(laplacian_stack)
    for i, image in enumerate(laplacian_stack):
        plt.subplot(2, len_lap, length_gau+i+1)
        im = normalize(image) if i != len_lap-1 else image
        plt.imshow(im)
    plt.savefig(save_name)

generateStack('data/lisa.jpg', 'result/2-3-1.jpg')
generateStack('data/gala.jpg', 'result/2-3-2.jpg')
generateStack('data/hybrid-2-2-1.jpg', 'result/2-3-3.jpg')
generateStack('data/hybrid-2-2-2.jpg', 'result/2-3-4.jpg')
generateStack('data/hybrid-2-2-3.jpg', 'result/2-3-5.jpg')
generateStack('data/hybrid-2-2-4.jpg', 'result/2-3-6.jpg')
generateStack('data/hybrid-2-2-5.jpg', 'result/2-3-7.jpg')


# 2.4
def blend(img1, img2, level, mask=None):
    if img1.shape != img2.shape:
        return None, None, None, None

    width = img1.shape[0]
    height = img1.shape[1]
    if mask is None:
        mask = np.vstack([np.concatenate([np.zeros(height//2), np.ones(height-height//2)]) for i in range(width)])
        mask = np.dstack([mask for i in range(3)])

    _, L_1 = im_stack(level, img1)
    _, L_2 = im_stack(level, img2)
    GR, _ = im_stack(level, mask)

    ret = np.sum([GR[i]*L_1[i] + (1-GR[i])*L_2[i] for i in range(level+1)], axis=0)

    return ret, L_1, L_2, GR

def generateImg(name1, name2, out, mask=None):
    img1 = plt.imread('data/' + name1) / 255.
    if mask is None:
        img1 = sk.transform.resize(img1, (450, 600))

    img2 = plt.imread('data/' + name2) / 255.
    if mask is None:
        img2 = sk.transform.resize(img2, (450, 600))
    blended_img, L_1, L_2, GR = blend(img1, img2, level=5, mask=mask)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('img1')
    plt.imshow(img1)
    plt.subplot(1, 3, 2)
    plt.title('img2')
    plt.imshow(img2)
    plt.subplot(1, 3, 3)
    plt.imshow(blended_img)
    plt.title('blend')
    plt.savefig('result/' + out + '.jpg')

    plt.figure(figsize=(36, 18))
    length = len(GR)
    for i, (gr, l1, l2) in enumerate(zip(GR, L_1, L_2)):
        plt.subplot(3, length, i+1)
        plt.imshow(normalize(gr*l1 + (1-gr)*l2))
        plt.subplot(3, length, i + length + 1)
        plt.imshow(normalize(gr * l1))
        plt.subplot(3, length, i+ 2*length +1)
        plt.imshow((normalize((1-gr) * l2)))
    plt.savefig('result/' + out + '-1.jpg')


generateImg('orange.jpg', 'apple.jpg', '2-4-1')
generateImg('moon.jpg', 'jupiter.jpg', '2-4-2')

# use mask
mask = plt.imread('data/man_mask.jpg')
mask = mask[:, :, :] // 255
generateImg('man.jpg', 'taj-mahal-normal.jpg', '2-4-3', mask=mask)
