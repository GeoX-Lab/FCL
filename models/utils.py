import os.path

import torch
import torch.fft as fft
import torch.nn.functional as F


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
def separate_frequency(tensor, radius_threshold):
    tensor_freq = fft.fftn(tensor, dim=(-2, -1))

    tensor_freq_mag = torch.abs(tensor_freq)

    freq_shape = tensor_freq_mag.shape[-2:]
    freq_x = torch.fft.fftfreq(freq_shape[1], d=1.0/freq_shape[1])
    freq_y = torch.fft.fftfreq(freq_shape[0], d=1.0/freq_shape[0])
    freq_x, freq_y = torch.meshgrid(freq_x, freq_y)
    freq_radius = torch.sqrt(freq_x**2 + freq_y**2).unsqueeze(0).unsqueeze(0)
    # freq_radius = torch.fft.fftfreq(freq_shape[0].unsqueeze(1)) + torch.fft.fftfreq(freq_shape[1]).unsqueeze(0)
    # freq_radius = torch.sqrt(freq_radius**2 + freq_radius.t()**2).unsqueeze(0).unsqueeze(0)

    high_freq_mask = (freq_radius > radius_threshold).float()
    low_freq_mask = (freq_radius <= radius_threshold).float()

    high_freq_spectrum = tensor_freq_mag*high_freq_mask
    low_freq_spectrum = tensor_freq_mag*low_freq_mask

    return high_freq_spectrum, low_freq_spectrum

def inverse_transform(tensor, freq_threshold):
    high_freq_tensor, low_freq_tensor = separate_frequency(tensor,freq_threshold)

    low_freq_tensor = torch.real(fft.ifftn(low_freq_tensor, dim=(-2,-1)))
    high_freq_tensor = torch.real(fft.ifftn(high_freq_tensor, dim=(-2,-1)))

    # low_freq_tensor = F.gaussian_filter2d(low_freq_tensor, sigma=1.0)
    # high_freq_tensor = F.gaussian_filter2d(high_freq_tensor, sigma=1.0)

    return high_freq_tensor,low_freq_tensor


def generate_fourier_tensor(tensor, threshold, flags='full'):
    """
    transform an image into the high and low frequency
    :threshold is the threshold of fourier transform
    :tensor is the image tensor from data_loader
    """
    # tensor = tensor.transpose(0,2,3,1)
    h, w = tensor.shape[2], tensor.shape[2]

    lpf = torch.zeros((h,w))
    R = (h + w)//threshold
    for x in range(w):
        for y in range(h):
            if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
                lpf[y,x] = 1
    hpf = 1 - lpf
    hpf, lpf = hpf.cuda(), lpf.cuda()

    tensor = tensor.cuda()
    tensor = F.normalize(tensor,dim= 0)
    f = fft.fftn(tensor, dim=(2,3))
    f = torch.roll(f, (h//2,w//2), dims=(2,3))
    f_l = f*lpf
    f_h = f*hpf
    tensor_l = torch.abs(fft.ifftn(f_l, dim=(2,3)))
    tensor_h = torch.abs(fft.ifftn(f_h, dim=(2,3)))

    if flags=='full':
        return tensor_h, tensor_l
    elif flags=='high':
        return tensor_h
    elif flags=='low':
        return tensor_l



# # Test
# b, c, H, W = 2, 3, 256, 256
# radius_threshold = 10
#
# T = torch.randn(b,c,H,W)
#
# high_freq_tensor1, low_freq_tensor1 = inverse_transform(T, radius_threshold)
# high_freq_tensor2, low_freq_tensor2 = inverse_transform(T, radius_threshold)
#
#
# print("high freq tesnsor\n", high_freq_tensor1)
# print("high freq tesnsor\n", high_freq_tensor2)



