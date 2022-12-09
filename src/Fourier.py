import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def cosineplot(x: list, amplitude: list, name: str) -> None:
    """ График косинуса """
    plt.figure(figsize=(10, 4))
    plt.plot(x, amplitude)
    plt.title(name)
    plt.xlabel('Time')

    plt.ylabel('Amplitude - cosine(time)')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='b')
    plt.show()

time = np.arange(0, 40, 0.1)

amplitude1 = np.cos(time)
amplitude2 = np.cos(time)
amplitude3 = np.cos(time)

# cosineplot(x=time, amplitude=amplitude1, name='Cosine wave, frequency = 1x')
# cosineplot(x=time, amplitude=amplitude3, name='Cosine wave, frequency = 2x')
# cosineplot(x=time, amplitude=amplitude3, name='Cosine wave, frequency = 3x')
#
# amplitude_s = (amplitude1 + 0.5) * (amplitude2 + 0.2) *  (amplitude3)
# cosineplot(x=time, amplitude=amplitude_s, name='Cosine wave, frequency = 3x')

def showDFFT(image):
    img = np.float32(image)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(np.abs(fshift))

    plt.subplot(121)
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, 'Greys', vmin=0, vmax=255)

    s_min = magnitude.min()
    s_max = magnitude.max()
    print(s_min, s_max)
    if s_min == s_max:
        plt.subplot(122)
        plt.imshow(magnitude, 'Greys', vmin=0, vmax=255)
    else:
        plt.subplot(122)
        plt.imshow(magnitude, 'Greys')
    plt.title('Magnitude Spectrum')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def DFFTnp(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def reverseDFFTnp(dfft):
    f_ishift = np.fft.ifftshift(dfft)
    reverse_image = np.fft.ifft2(f_ishift)
    return reverse_image

def sobel(image):
    img = np.float32(image)
    fshift = DFFTnp(img)

    ksize = 3
    kernel = np.zeros(img.shape)
    sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel[0:ksize, 0:ksize] = sobel_h

    fkshift = DFFTnp(kernel)
    mult = np.multiply(fshift, fkshift)
    reverse_image = reverseDFFTnp(mult)
    return reverse_image
    plt.imshow(abs(reverse_image), cmap='gray')
    plt.title('Sobel')
    plt.show()

def gaussian_filter(image):
    img = np.float32(image)
    fshift = DFFTnp(img)

    ksize = 21
    kernel = np.zeros(img.shape)

    blur = cv2.getGaussianKernel(ksize, -1)
    blur = np.matmul(blur, np.transpose(blur))
    kernel[0:ksize, 0:ksize] = blur

    fkshift = DFFTnp(kernel)
    mult = np.multiply(fshift, fkshift)

    reverse_image = reverseDFFTnp(mult)

    plt.imshow(abs(reverse_image), cmap='gray')
    plt.title('Gauss blur')
    plt.show()


if __name__ == '__main__':
    folder = 'data/lab_3'
    for image in os.listdir(folder):
        img = cv2.imread(folder + '/' + image, 0)
        # gaussian_filter(img)
        # sobel(img)
        showDFFT(img)






