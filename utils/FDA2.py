import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# def gen_gaussian_noise(image, SNR):
#     """
#     Generate Gaussian noise with a given signal-to-noise ratio.
#     :param image: source image (tensor) [batch, C, H, W]
#     :param SNR: signal-noise ratio
#     :return: noise (tensor) [batch, C, H, W]
#     """
#     B, C, H, W = image.shape
#     noise = torch.randn(B, 1, H, W, device=image.device)
#     noise = noise - noise.mean(dim=(1, 2, 3), keepdim=True)
#     image_power = (image**2).mean(dim=(1, 2, 3), keepdim=True)
#     noise_variance = image_power / (10 ** (SNR / 10))
#     noise = (noise * torch.sqrt(noise_variance) / noise.std(dim=(1, 2, 3), keepdim=True))
#     return noise
#
# def normalize_image(img, mean, std):
#     """
#     Normalize or denormalize an image.
#     :param img: image tensor [batch, C, H, W]
#     :param mean: list of means
#     :param std: list of stds
#     :return: processed image tensor [batch, C, H, W]
#     """
#     mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
#     std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
#     return img * std + mean
#
# def my_fft_trans(img):
#     """
#     Apply FFT transformation and add noise to the image.
#     :param img: input image tensor [batch, C, H, W]
#     :param threshold: threshold value (not used in current implementation)
#     :return: processed image tensor [batch, C, H, W]
#     """
#     # mean = [0.485, 0.456, 0.406]
#     # std = [0.229, 0.224, 0.225]
#
#     # img = normalize_image(img, mean, std)
#     img = img.permute(0, 2, 3, 1)  # [batch, C, H, W] -> [batch, H, W, C]
#
#     f_img = torch.fft.fft2(img, dim=(1, 2))
#     abs_f_img = torch.abs(f_img)
#     angle_f_img = torch.angle(f_img)
#
#     noise_phase = (torch.rand_like(angle_f_img) * 0.4 - 0.2) + angle_f_img  # [-0.2, 0.2] + original phase
#     noise_amplitude = (torch.rand_like(abs_f_img) * 0.5 + 0.75) * abs_f_img  # [0.75, 1.25] * original amplitude
#
#     f_noisy_img = noise_amplitude * (torch.cos(noise_phase) + 1j * torch.sin(noise_phase))
#
#     noisy_img = torch.fft.ifft2(f_noisy_img, dim=(1, 2)).abs()
#     noisy_img = torch.clamp(noisy_img, 0, 1) * 255
#     noisy_img = noisy_img.byte()
#
#     noisy_img = torch.flip(noisy_img, [2])  # Flip left-right
#     return noisy_img.permute(0, 3, 1, 2)  # [batch, H, W, C] -> [batch, C, H, W]


def gen_gaussian_noise(image, SNR):
    """
    :param image: source image (tensor) [batch, C, H, W]
    :param SNR: signal-noise ratio
    :return: noise (tensor) [batch, C, H, W]
    """
    assert len(image.shape) == 4
    B, C, H, W = image.shape
    noise = torch.randn(B, 1, H, W, device=image.device)
    noise = noise - noise.mean(dim=(1, 2, 3), keepdim=True)
    image_power = 1 / (H * W * C) * torch.sum(image ** 2, dim=(1, 2, 3), keepdim=True)
    SNR_tensor = torch.tensor(SNR, device=image.device, dtype=image_power.dtype)
    noise_variance = image_power / torch.pow(10, (SNR_tensor / 10))
    noise = (torch.sqrt(noise_variance) / torch.std(noise, dim=(1, 2, 3), keepdim=True)) * noise
    return noise


def decoder_image(img, mean, std):
    """
    Decode image by reversing normalization
    :param img: normalized image tensor [batch, C, H, W]
    :param mean: list of means
    :param std: list of stds
    :return: decoded image tensor [batch, C, H, W]
    """
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1) * 255
    return img


def my_fft_trans(img1):
    # 预处理
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # img1 = decoder_image(img1, mean, std)

    img1 = img1.permute(0, 2, 3, 1)  # 转换为 BCHW -> BHWC
    B, H, W, C = img1.shape
    # crows, ccols = H // 2, W // 2

    f1 = torch.fft.fft2(img1, dim=(1, 2))

    fig_abs_temp = torch.abs(f1)
    fig_pha_temp = torch.angle(f1)

    # 添加噪声
    noise_w_h_p = torch.FloatTensor(B, H, W, 1).uniform_(0, 0.5).to(img1.device)
    noise_b_h_p = torch.FloatTensor(B, H, W, 1).uniform_(-np.pi / 6, np.pi / 6).to(img1.device)
    fig_pha_ag = noise_w_h_p * fig_pha_temp + noise_b_h_p

    noise_w_h_a = torch.FloatTensor(B, H, W, 1).uniform_(0, 0.5).to(img1.device)
    noise_b_h_a = gen_gaussian_noise(fig_abs_temp, 80)
    fig_abs_ag = noise_w_h_a * fig_abs_temp + noise_b_h_a

    f_ag = fig_abs_ag * torch.cos(fig_pha_ag) + fig_abs_ag * torch.sin(fig_pha_ag) * 1j

    # 逆傅里叶变换
    noisy_img = torch.fft.ifft2(f_ag, dim=(1, 2)).abs()
    noisy_img = torch.clamp(noisy_img, 0, 1) * 255
    noisy_img = noisy_img.byte()

    # img_ag = torch.flip(img_ag, [2])  # 左右翻转

    return noisy_img.permute(0, 3, 1, 2).float()  # 转换回 BCHW


# 示例用法
# if __name__ == "__main__":
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#
#     # 生成一个随机图像张量
#     # img = torch.rand(2, 3, 256, 256)  # Batch size 为 2
#     image_path = '../data/AID/test/Airport/airport_16.jpg'
#     img = Image.open(image_path).convert('RGB')
#     img = transform(img)
#     img = img.unsqueeze(0)
#
#     # 处理图像
#     result = my_fft_trans(img)
#
#     # 转换为 PIL 图像以进行可视化
#     result_img = result[0].permute(1, 2, 0).cpu().numpy()  # 提取第一个 batch 并转换为 HWC
#     result_img = Image.fromarray(result_img)
#     result_img.show()
