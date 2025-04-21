import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import argparse


def generate_fourier_img(sample_path, threshold, flags, new_size=(256, 256)):
    """
    generate fourier images given folders.
    Args:
        sample_path: input images folder
        threshold: threshold of fft, ranges [1, h+w]
        flags: mode of fft, choice from ['low','high']
    """
    img = cv2.imread(sample_path)
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]

    threshold_ = threshold if flags == 'low' else (h + w - threshold)

    y, x = np.ogrid[:h, :w]
    # mask = ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < ((h + w) // threshold) ** 2
    mask = ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < threshold_ ** 2
    lpf = mask.astype(int)
    hpf = 1 - lpf

    filter = lpf if flags == 'low' else hpf

    transformed_channels = []
    for i in range(3):
        freq = np.fft.fft2(img[:, :, i], axes=(0, 1))
        freq_shifted = np.fft.fftshift(freq)
        transformed = freq_shifted * filter
        img_transformed = np.abs(np.fft.ifft2(np.fft.ifftshift(transformed), axes=(0, 1)))
        transformed_channels.append(img_transformed)

    re_img = np.dstack(transformed_channels).astype(np.uint8)

    return re_img


def process_sample(sample_path, output_class_path, threshold, flags):
    output_sample_path = os.path.join(output_class_path, os.path.basename(sample_path))
    reconstructed_image = generate_fourier_img(sample_path, threshold, flags)
    cv2.imwrite(output_sample_path, reconstructed_image)


def batch_fourier_transform(dataset_path, output_path, threshold, flags):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    classes = os.listdir(dataset_path)

    with ThreadPoolExecutor() as executor:
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            output_class_path = os.path.join(output_path, class_name)

            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)

            sample_files = [os.path.join(class_path, f) for f in os.listdir(class_path)]
            futures = [executor.submit(process_sample, sample, output_class_path, threshold, flags) for sample in
                       sample_files]
            for future in futures:
                future.result()


def main():
    parser = argparse.ArgumentParser(description="Batch Fourier Transform on Images.")
    parser.add_argument("--dataset_paths", type=str,
                        default=['../data/imagenet-20/train',
                                 '../data/imagenet-20/val'],
                        nargs='+', help="Paths to the datasets.")
    parser.add_argument("-output_path", type=str, default='../data/imagenet-20/dataset_FR',
                        help="Path to save transformed images.")
    parser.add_argument("--flags", type=str, choices=['low', 'high'], default='high',
                        help="Type of Fourier transform ('low' or 'high').")
    parser.add_argument("--threshold_list", type=int, default=[420, 468, 452],
                        help="Threshold list for Fourier transform.")
    args = parser.parse_args()

    for threshold in args.threshold_list:
        for taskid, dataset_path in enumerate(args.dataset_paths):
            output_dataset_path = os.path.join(
                args.output_path + f'/{args.flags}_{threshold}',
                os.path.basename(dataset_path))
            batch_fourier_transform(dataset_path, output_dataset_path, threshold, args.flags)


if __name__ == "__main__":
    main()
