import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from PIL import Image
import sys
import pickle


version = sys.version_info
def load_datafile(filename):
  with open(filename, 'rb') as fo:
      if version.major == 3:
          data_dict = pickle.load(fo, encoding='bytes')
      else:
          data_dict = pickle.load(fo)

      assert data_dict[b'data'].dtype == np.uint8
      image_data = data_dict[b'data']
      image_data = image_data.reshape(
          (10000, 3, 32, 32)).transpose(0, 2, 3, 1)
      return image_data, np.array(data_dict[b'labels'])
def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)

        if i < 9:
            raw_r = tmp[:,:,0]
            raw_g = tmp[:,:,1]
            raw_b = tmp[:,:,2]
            ir = Image.fromarray(raw_r.astype(np.uint8))
            ig = Image.fromarray(raw_g.astype(np.uint8))
            ib = Image.fromarray(raw_b.astype(np.uint8))
            demo_images_freq_low = Image.merge("RGB",(ir,ig,ib))
            plt.imshow(demo_images_freq_low)
            demo_images_freq_low.save("demo_images_freq_low_"+str(r)+"_"+str(i)+".png")

        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

        if i < 9:
            raw_r = tmp[:,:,0]
            raw_g = tmp[:,:,1]
            raw_b = tmp[:,:,2]
            ir = Image.fromarray(raw_r.astype(np.uint8))
            ig = Image.fromarray(raw_g.astype(np.uint8))
            ib = Image.fromarray(raw_b.astype(np.uint8))
            demo_images_freq_high = Image.merge("RGB",(ir,ig,ib))
            plt.imshow(demo_images_freq_high)
            demo_images_freq_high.save("demo_images_freq_high_"+str(r)+"_"+str(i)+".png")

    return np.array(Images_freq_low), np.array(Images_freq_high)





if __name__ == '__main__':
    eval_images, eval_labels = load_datafile('../data/CIFAR10/test_batch')

    np.save('../data/CIFAR10/test_data_regular', eval_images)
    #
    eval_image_low_4, eval_image_high_4 = generateDataWithDifferentFrequencies_3Channel(eval_images, 4)
    np.save('../data/CIFAR10/test_data_low_4', eval_image_low_4)
    np.save('../data/CIFAR10/test_data_high_4', eval_image_high_4)

