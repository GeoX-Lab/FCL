import torch
import torch.fft
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from focal_frequency_loss import FocalFrequencyLoss as FFL


def _fre_mse_loss(pred_L, pred_H, target_L, target_H, T=0.5):
    loss_fn_MSE = torch.nn.MSELoss()
    loss_H = loss_fn_MSE(pred_H, target_H)
    loss_L = loss_fn_MSE(pred_L, target_L)

    return T * loss_H + (1 - T) * loss_L


def _fre_focal_loss(pred_L, pred_H, target_L, target_H, T=0.5):
    ffl = FFL(loss_weight=1.0, alpha=1.0)
    loss_H = ffl(pred_H, target_H)
    loss_L = ffl(pred_L, target_L)

    return T * loss_H + (1 - T) * loss_L


def fourier_transform(image):
    return torch.fft.fft2(image, dim=(-2, -1))


def inverse_fourier_transform(image_f):
    return torch.fft.ifft2(image_f, dim=(-2, -1)).real


# def amplitude_mix_single_batch(batch, labels, alpha=0.1, lamda=0.5, num_classes=10, epsilon=0.1, label_smoothing=False):
#     # lamda = torch.rand(1).to(labels.device)
#
#     batch_size, channels, height, width = batch.shape
#     indices = torch.randperm(batch_size)
#     f_batch = fourier_transform(batch)
#     amplitude_batch, phase_batch = torch.abs(f_batch), torch.angle(f_batch)
#     amplitude_perm = amplitude_batch[indices]
#     mixed_amplitude = lamda * amplitude_batch + (1 - lamda) * amplitude_perm
#     mixed_f = torch.polar(mixed_amplitude, phase_batch)
#     mixed_batch = inverse_fourier_transform(mixed_f)
#
#     # Create new labels
#     labels = label_onehot(labels, num_classes)
#     mixed_labels = (1 - lamda * alpha) * labels + lamda * alpha * labels[indices]
#
#     # label smoothing
#     if label_smoothing is True:
#         mixed_labels = mixed_labels * (1 - epsilon) + (epsilon / num_classes)
#     return mixed_batch, mixed_labels

# def amplitude_mix_single_batch(batch, labels, alpha=0.1, lamda=0.5, num_classes=10, epsilon=0.1, label_smoothing=False,
#                                threshold=None):
#     # lamda = torch.rand(1).to(labels.device)
#     batch = batch.cuda()
#
#     batch_size, channels, height, width = batch.shape
#     indices = torch.randperm(batch_size)
#     f_batch = fourier_transform(batch)
#
#     # generate fft images
#     if threshold is None:
#         threshold = torch.randint(0, (height + width) // 4, (1,)).item()
#
#     crows, ccols = height // 2, width // 2
#     mask_l = torch.zeros((channels, height, width), dtype=torch.bool).cuda()
#     mask_l[:, crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = True
#     # Applying mask
#     fshift = torch.fft.fftshift(f_batch)
#
#     fre_l = fshift * mask_l[None, :, :, :]
#     fre_h = fshift * (~mask_l[None, :, :, :])
#     # Inverse FFT
#     fre_l = torch.fft.ifftshift(fre_l, dim=(-2, -1))
#     fre_h = torch.fft.ifftshift(fre_h, dim=(-2, -1))
#
#     img_l = torch.fft.ifft2(fre_l, dim=(-2, -1)).abs()
#     img_h = torch.fft.ifft2(fre_h, dim=(-2, -1)).abs()
#
#     max_values = img_h.view(batch_size, -1).max(dim=1)[0]  # 获取每个图像的最大值
#     img_h = img_h * (255.0 / max_values[:, None, None, None].clamp(min=1e-6))
#     img_l.clamp_(0, 255)
#     img_h.clamp_(0, 255)
#
#     # data frequency augmentation
#     amplitude_batch, phase_batch = torch.abs(f_batch), torch.angle(f_batch)
#     amplitude_perm = amplitude_batch[indices]
#     mixed_amplitude = lamda * amplitude_batch + (1 - lamda) * amplitude_perm
#     mixed_f = torch.polar(mixed_amplitude, phase_batch)
#     mixed_batch = inverse_fourier_transform(mixed_f)
#
#     # label smoothing
#     if label_smoothing is True:
#         # Create new labels
#         labels = label_onehot(labels, num_classes)
#         mixed_labels = (1 - lamda * alpha) * labels + lamda * alpha * labels[indices]
#         mixed_labels = mixed_labels * (1 - epsilon) + (epsilon / num_classes)
#     else:
#         mixed_labels = None
#     return mixed_batch, mixed_labels, img_l, img_h

def amplitude_mix_single_batch(batch, labels, alpha=0.1, lamda=0.5, num_classes=10, epsilon=0.1, label_smoothing=False,
                               threshold=None):
    # lamda = torch.rand(1).to(labels.device)
    batch = batch.cuda()

    batch_size, channels, height, width = batch.shape
    indices = torch.randperm(batch_size)
    f_batch = fourier_transform(batch)

    # data frequency augmentation
    amplitude_batch, phase_batch = torch.abs(f_batch), torch.angle(f_batch)
    amplitude_perm = amplitude_batch[indices]
    mixed_amplitude = lamda * amplitude_batch + (1 - lamda) * amplitude_perm
    mixed_f = torch.polar(mixed_amplitude, phase_batch)
    mixed_batch = inverse_fourier_transform(mixed_f)

    f_batch = fourier_transform(mixed_batch)
    # f_batch = mixed_f
    # generate fft images
    if threshold is None:
        threshold = torch.randint(0, (height + width) // 4, (1,)).item()

    crows, ccols = height // 2, width // 2
    mask_l = torch.zeros((channels, height, width), dtype=torch.bool).cuda()
    mask_l[:, crows - threshold:crows + threshold, ccols - threshold:ccols + threshold] = True
    # Applying mask
    fshift = torch.fft.fftshift(f_batch)

    fre_l = fshift * mask_l[None, :, :, :]
    fre_h = fshift * (~mask_l[None, :, :, :])
    # Inverse FFT
    fre_l = torch.fft.ifftshift(fre_l, dim=(-2, -1))
    fre_h = torch.fft.ifftshift(fre_h, dim=(-2, -1))

    img_l = torch.fft.ifft2(fre_l, dim=(-2, -1)).abs()
    img_h = torch.fft.ifft2(fre_h, dim=(-2, -1)).abs()

    max_values = img_h.view(batch_size, -1).max(dim=1)[0]  # 获取每个图像的最大值
    img_h = img_h * (255.0 / max_values[:, None, None, None].clamp(min=1e-6))
    img_l.clamp_(0, 255)
    img_h.clamp_(0, 255)

    # label smoothing
    if label_smoothing is True:
        # Create new labels
        labels = label_onehot(labels, num_classes)
        mixed_labels = (1 - lamda * alpha) * labels + lamda * alpha * labels[indices]
        mixed_labels = mixed_labels * (1 - epsilon) + (epsilon / num_classes)
    else:
        mixed_labels = None
    return mixed_batch, mixed_labels, img_l, img_h

def label_onehot(labels, num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)
    return one_hot


def show_images(batch, title):
    grid_img = torchvision.utils.make_grid(batch, nrow=4, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def main(alpha=0.5, lamda=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Get a batch of training data
    data_iter = iter(dataloader)
    images, labels = data_iter.next()

    # Convert labels to one-hot encoding
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()

    # Apply amplitude mix augmentation
    augmented_images, augmented_labels = amplitude_mix_single_batch(images, labels_one_hot, alpha=alpha, lamda=lamda)

    # Visualize original and augmented images
    show_images(images, title='Original Images')
    show_images(augmented_images, title='Augmented Images (Amplitude Mix)')

    # Print original and augmented labels
    print("Original Labels (one-hot):\n", labels_one_hot)
    print("Augmented Labels:\n", augmented_labels)


if __name__ == "__main__":
    main()

# import matplotlib.pyplot as plt
# import argparse
# import numpy as np
# import torch
# from torchvision import transforms
# from torchvision.datasets import CIFAR100
# import torchvision.transforms.functional as TF
# from PIL import Image
# import torch.nn as nn
# import torch
# import torch.fft
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
#
# def fourier_transform(image):
#     return torch.fft.fft2(image, dim=(-2, -1))
#
# def inverse_fourier_transform(image_f):
#     return torch.fft.ifft2(image_f, dim=(-2, -1)).real
#
# def amplitude_mix_single_batch(batch, alpha=0.5):
#     batch_size, channels, height, width = batch.shape
#     indices = torch.randperm(batch_size)
#     f_batch = fourier_transform(batch)
#     amplitude_batch, phase_batch = torch.abs(f_batch), torch.angle(f_batch)
#     amplitude_perm = amplitude_batch[indices]
#     mixed_amplitude = alpha * amplitude_batch + (1 - alpha) * amplitude_perm
#     mixed_f = torch.polar(mixed_amplitude, phase_batch)
#     mixed_batch = inverse_fourier_transform(mixed_f)
#     return mixed_batch
#
# def show_images(batch, title):
#     grid_img = torchvision.utils.make_grid(batch, nrow=4, normalize=True)
#     plt.figure(figsize=(10, 10))
#     plt.title(title)
#     plt.imshow(grid_img.permute(1, 2, 0))
#     plt.axis('off')
#     plt.show()
#
# def main():
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#
#     dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
#
#     # Get a batch of training data
#     data_iter = iter(dataloader)
#     images, _ = data_iter.next()
#
#     # Apply amplitude mix augmentation
#     augmented_images = amplitude_mix_single_batch(images, alpha=0.5)
#
#     # Visualize original and augmented images
#     show_images(images, title='Original Images')
#     show_images(augmented_images, title='Augmented Images (Amplitude Mix)')
#
# if __name__ == "__main__":
#     main()
#
# def gen_gaussian_noise(image, SNR):
#     H, W, C = image.shape
#     noise = np.random.randn(H, W, 1)
#     noise = noise - np.mean(noise)
#     image_power = 1 / (H * W * 3) * np.sum(np.power(image, 2))
#     noise_variance = image_power / np.power(10, (SNR / 10))
#     noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
#     return noise
#
# def decoder_image(img, mean, std):
#     img = img.copy().astype(np.float32)
#     for i in range(3):
#         img[:, :, i] = img[:, :, i] * std[i] + mean[i]
#     return img
#
# def my_fft_trans(img, threshold):
#     H, W, C = img.shape
#     f1 = np.fft.fft2(img, axes=(0, 1))
#
#     fig_abs_temp = np.abs(f1)
#     fig_pha_temp = np.angle(f1)
#
#     noise_w_h_p = np.random.uniform(0.5, 1.0, (H, W, 1))
#     noise_b_h_p = np.random.uniform(-np.pi / 6, np.pi / 6, (H, W, 1))
#     fig_pha_ag = noise_w_h_p * fig_pha_temp + noise_b_h_p
#
#     noise_w_h_a = np.random.uniform(0.5, 1.0, (H, W, 1))
#     noise_b_h_a = gen_gaussian_noise(fig_abs_temp, 50)
#     fig_abs_ag = noise_w_h_a * fig_abs_temp + noise_b_h_a
#
#     f_ag = fig_abs_ag * np.cos(fig_pha_ag) + fig_abs_ag * np.sin(fig_pha_ag) * 1j
#
#     img_ag = np.fft.ifft2(f_ag, axes=(0, 1))
#     img_ag = np.abs(img_ag)
#     img_ag = np.uint8(np.clip(img_ag, 0, 255))
#
#     # img_ag = np.array(Image.fromarray(img_ag).transpose(Image.FLIP_LEFT_RIGHT))
#     return img_ag
#
# class MyFFTTransform:
#     def __call__(self, img):
#         img = np.array(img)
#         img = my_fft_trans(img, threshold=0)
#         img = Image.fromarray(img)
#         return img
#
# # 定义自定义的傅里叶变换数据增强方法
# # def gen_gaussian_noise(image, SNR):
# #     H, W, C = image.shape
# #     noise = np.random.randn(H, W, 1)
# #     noise = noise - np.mean(noise)
# #     image_power = 1 / (H * W * 3) * np.sum(np.power(image, 2))
# #     noise_variance = image_power / np.power(10, (SNR / 10))
# #     noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
# #     return noise
# #
# # def decoder_image(img, mean, std):
# #     img = img.copy().astype(np.float32)
# #     for i in range(3):
# #         img[:, :, i] = img[:, :, i] * std[i] + mean[i]
# #     return img
# #
# # def my_fft_trans(img, threshold):
# #     # img = np.transpose(img, (1, 2, 0))
# #
# #     H, W, C = img.shape
# #     crows, ccols = H // 2, W // 2
# #
# #     f1 = np.fft.fft2(img, axes=(0, 1))
# #
# #     fig_abs_temp = np.abs(f1)
# #     fig_pha_temp = np.angle(f1)
# #
# #     noise_w_h_p = np.random.uniform(0.8, 1.2, (H, W, 1))
# #     noise_b_h_p = np.random.uniform(-np.pi / 6, np.pi / 6, (H, W, 1))
# #     fig_pha_ag = noise_w_h_p * fig_pha_temp + noise_b_h_p
# #
# #     noise_w_h_a = np.random.uniform(0.5, 1.5, (H, W, 1))
# #     noise_b_h_a = gen_gaussian_noise(fig_abs_temp, 30)
# #     fig_abs_ag = noise_w_h_a * fig_abs_temp + noise_b_h_a
# #
# #     f_ag = fig_abs_ag * np.cos(fig_pha_ag) + fig_abs_ag * np.sin(fig_pha_ag) * 1j
# #
# #     img_ag = np.fft.ifft2(f_ag, axes=(0, 1))
# #     img_ag = np.abs(img_ag)
# #     img_ag = np.uint8(np.clip(img_ag, 0, 255))
# #
# #     # img_ag = np.array(Image.fromarray(img_ag).transpose(Image.FLIP_LEFT_RIGHT))
# #     return img_ag
# #
# # class MyFFTTransform:
# #     def __call__(self, img):
# #         img = np.array(img)
# #         img = my_fft_trans(img, threshold=0)
# #         img = Image.fromarray(img)
# #         return img
#
#
# # class FrequencyDomainAugmentation(torch.nn.Module):
# #     def __init__(self, alpha_range=(0.9, 1.1), beta_range=(-0.1, 0.1)):
# #         super().__init__()
# #         self.alpha_range = alpha_range
# #         self.beta_range = beta_range
# #
# #     def forward(self, img):
# #         # Convert image to frequency domain using FFT
# #         data_freq = torch.fft.fft2(img, dim=(-2, -1))
# #         amplitude = torch.abs(data_freq)
# #         phase = torch.angle(data_freq)
# #
# #         # Generate random noise for amplitude and phase
# #         alpha = torch.empty_like(amplitude).uniform_(*self.alpha_range)
# #         beta = torch.empty_like(phase).uniform_(*self.beta_range)
# #
# #         # Apply noise to amplitude and phase
# #         disturbed_amplitude = alpha * amplitude + beta
# #         disturbed_freq = disturbed_amplitude * torch.exp(1j * phase)
# #
# #         # Convert back using IFFT
# #         disturbed_image = torch.fft.ifft2(disturbed_freq, dim=(-2, -1)).real
# #         return disturbed_image
# #
# #
# # class FrequencyDomainAugmentation_v1(torch.nn.Module):
# #     def __init__(self, alpha_range=(0.9, 1.1), beta_range=(-0.1, 0.1)):
# #         super().__init__()
# #         self.alpha_range = alpha_range
# #         self.beta_range = beta_range
# #
# #     def forward(self, img):
# #         C, H, W = img.shape
# #         # Convert image to frequency domain using FFT
# #         data_freq = torch.fft.fft2(img, dim=(-2, -1))
# #         fig_abs_temp = torch.abs(data_freq)
# #         fig_pha_temp = torch.angle(data_freq)
# #
# #         # Add noise
# #         noise_w_h_p = torch.from_numpy(np.random.uniform(0.5, 1.5, (C, H, W))).float()
# #         noise_b_h_p = torch.from_numpy(np.random.uniform(-np.pi / 6, np.pi / 6, (C, H, W))).float()
# #         fig_pha_ag = noise_w_h_p * fig_pha_temp + noise_b_h_p
# #
# #         noise_w_h_a = torch.from_numpy(np.random.uniform(0.5, 1.5, (C, H, W))).float()
# #         noise_b_h_a = gen_gaussian_noise(fig_abs_temp, 30)
# #         fig_abs_ag = noise_w_h_a * fig_abs_temp + noise_b_h_a
# #
# #         f_ag = fig_abs_ag * torch.cos(fig_pha_ag) + fig_abs_ag * torch.sin(fig_pha_ag) * 1j
# #
# #         # Inverse FFT
# #         img_ag = torch.fft.ifft2(f_ag, dim=(-2, -1))
# #         img_ag = torch.abs(img_ag)
# #         img_ag = torch.clamp(img_ag, 0, 255)
# #
# #         return img_ag
#
#
# class FrequencyDomainAugmentation(torch.nn.Module):
#     def __init__(self, alpha_range=(0.9, 1.1), beta_range=(-0.1, 0.1)):
#         super().__init__()
#         self.alpha_range = alpha_range
#         self.beta_range = beta_range
#
#     def forward(self, img):
#         # Convert image to frequency domain using FFT
#         data_freq = torch.fft.fft2(img, dim=(-2, -1))
#         amplitude = torch.abs(data_freq)
#         phase = torch.angle(data_freq)
#
#         # Generate random noise for amplitude and phase
#         alpha = torch.empty_like(amplitude).uniform_(*self.alpha_range)
#         beta = torch.empty_like(phase).uniform_(*self.beta_range)
#
#         # Apply noise to amplitude and phase
#         disturbed_amplitude = alpha * amplitude + beta
#         disturbed_freq = disturbed_amplitude * torch.exp(1j * phase)
#
#         # Convert back using IFFT
#         disturbed_image = torch.fft.ifft2(disturbed_freq, dim=(-2, -1)).real
#         return disturbed_image
#
#
# # class FrequencyDomainAugmentation_v1(torch.nn.Module):
# #     def __init__(self, alpha_range=(0.9, 1.1), beta_range=(-0.1, 0.1)):
# #         super().__init__()
# #         self.alpha_range = alpha_range
# #         self.beta_range = beta_range
# #
# #     def forward(self, img):
# #         # Convert image to frequency domain using FFT
# #         data_freq = torch.fft.fft2(img, dim=(-2, -1))
# #         fig_abs_temp = torch.abs(data_freq)
# #         fig_pha_temp = torch.angle(data_freq)
# #
# #         # Add noise
# #         C, H, W = img.size()
# #         noise_w_h_p = torch.from_numpy(np.random.uniform(0.8, 1.2, (C, H, W))).float()
# #         noise_b_h_p = torch.from_numpy(np.random.uniform(-np.pi / 6, np.pi / 6, (C, H, W))).float()
# #         fig_pha_ag = noise_w_h_p * fig_pha_temp + noise_b_h_p
# #
# #         noise_w_h_a = torch.from_numpy(np.random.uniform(0.5, 1.5, (C, H, W))).float()
# #         noise_b_h_a = gen_gaussian_noise(fig_abs_temp, 30)
# #         fig_abs_ag = noise_w_h_a * fig_abs_temp + noise_b_h_a
# #
# #         f_ag = fig_abs_ag * torch.cos(fig_pha_ag) + fig_abs_ag * torch.sin(fig_pha_ag) * 1j
# #
# #         # Inverse FFT
# #         img_ag = torch.fft.ifft2(f_ag, dim=(-2, -1))
# #         img_ag = torch.abs(img_ag)
# #         img_ag = torch.clamp(img_ag, 0, 255).byte()
# #
# #         return img_ag
#
# class FrequencyDomainAugmentation_v1(nn.Module):
#     def __init__(self, alpha_range=(0.9, 1.1), beta_range=(-0.1, 0.1)):
#         super().__init__()
#         self.alpha_range = alpha_range
#         self.beta_range = beta_range
#
#     def forward(self, img):
#         if not isinstance(img, torch.Tensor):
#             raise TypeError(f'Input tensor should be a torch tensor. Got {type(img)}.')
#
#         data_freq = torch.fft.fft2(img, dim=(-2, -1))
#         fig_abs_temp = torch.abs(data_freq)
#         fig_pha_temp = torch.angle(data_freq)
#
#         C, H, W = img.size()
#         noise_w_h_p = torch.from_numpy(np.random.uniform(0.8, 1.2, (C, H, W))).float()
#         noise_b_h_p = torch.from_numpy(np.random.uniform(-np.pi / 6, np.pi / 6, (C, H, W))).float()
#         fig_pha_ag = noise_w_h_p * fig_pha_temp + noise_b_h_p
#
#         noise_w_h_a = torch.from_numpy(np.random.uniform(0.5, 1.5, (C, H, W))).float()
#         noise_b_h_a = gen_gaussian_noise(fig_abs_temp, 30)
#         fig_abs_ag = noise_w_h_a * fig_abs_temp + noise_b_h_a
#
#         f_ag = fig_abs_ag * torch.cos(fig_pha_ag) + fig_abs_ag * torch.sin(fig_pha_ag) * 1j
#
#         img_ag = torch.fft.ifft2(f_ag, dim=(-2, -1))
#         img_ag = torch.abs(img_ag)
#         img_ag = torch.clamp(img_ag, 0, 255)
#
#         return img_ag
#
# # def gen_gaussian_noise(image, SNR):
# #     """
# #     :param image: source image tensor of shape (C, H, W)
# #     :param SNR: signal-to-noise ratio
# #     :return: noise tensor of shape (C, H, W)
# #     """
# #     # assert image.dim() == 3, "Image must have three dimensions (C, H, W)"
# #     C, H, W = image.size()
# #     noise = torch.randn(C, H, W)
# #     noise = noise - torch.mean(noise)
# #     image_power = torch.sum(image ** 2) / (H * W * C)
# #     noise_variance = image_power / (10 ** (SNR / 10))
# #     noise = (torch.sqrt(noise_variance) / torch.std(noise)) * noise
# #     return noise
#
# # def gen_gaussian_noise(image, SNR):
# #     """
# #     :param image: source image tensor of shape (C, H, W)
# #     :param SNR: signal-to-noise ratio
# #     :return: noise tensor of shape (C, H, W)
# #     """
# #     assert image.dim() == 3, "Image must have three dimensions (C, H, W)"
# #     C, H, W = image.shape
# #     noise = torch.randn(C, H, W)
# #     noise = noise - torch.mean(noise)
# #     image_power = torch.sum(image ** 2) / (H * W * C)
# #     noise_variance = image_power / (10 ** (SNR / 10))
# #     noise = (torch.sqrt(noise_variance) / torch.std(noise)) * noise
# #     return noise
#
#
# def visualize_images(original, augmented1, augmented2):
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#     axs[0].imshow(transforms.ToPILImage()(original))
#     axs[0].set_title('Original Image')
#     axs[0].axis('off')
#
#     axs[1].imshow(transforms.ToPILImage()(augmented1))
#     axs[1].set_title('Augmented Image (Method 1)')
#     axs[1].axis('off')
#
#     axs[2].imshow(transforms.ToPILImage()(augmented2))
#     axs[2].set_title('Augmented Image (Method 2)')
#     axs[2].axis('off')
#
#     plt.show()
#
#
# def process_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ])
#
#     img = Image.open(image_path).convert('RGB')
#     img_tensor = transform(img)
#
#     # Apply both augmentations
#     aug1 = FrequencyDomainAugmentation()
#     aug2 = FrequencyDomainAugmentation_v1()
#
#     augmented_image1 = aug1(img_tensor)
#     augmented_image2 = aug2(img_tensor)
#
#     visualize_images(img_tensor, augmented_image1, augmented_image2)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Apply frequency domain data augmentation to a single color image.')
#     parser.add_argument('--image_path', default='./test_data/airplane95.tif', type=str, help='Path to the image file')
#     args = parser.parse_args()
#     process_image(args.image_path)
