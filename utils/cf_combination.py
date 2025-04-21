import os
import shutil
import argparse

def create_folder(path="../data/imagenet-20/dataset_FR/increment_var_FR/low2high/FR-combination"):
    """Ensure a folder exists; if it does, remove and recreate it."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def copy_folders(source, destination, folders):
    """Copy specified folders from the source directory to the destination directory."""
    for folder in folders:
        src_path = os.path.join(source, folder)
        dest_path = os.path.join(destination, folder)
        shutil.copytree(src_path, dest_path)

def get_folder_selection(source, start, count):
    """Return a list of folders selected from source directory based on start index and count."""
    folders = sorted(os.listdir(source))
    if count < 0:
        return folders[start:]
    else:
        return folders[start:start + count]

def combine_folders(base_folder1, base_folder2, output_folder, n=10, m=10):
    """Combine specific number of folders from two base folders into a new output folder."""
    sections = ['train', 'val']
    for section in sections:
        source1 = os.path.join(base_folder1, section)
        source2 = os.path.join(base_folder2, section)
        dest = os.path.join(output_folder, section)

        create_folder(dest)
        folders1 = get_folder_selection(source1, 0, n)
        folders2 = get_folder_selection(source2, n, n+m)

        copy_folders(source1, dest, folders1)
        copy_folders(source2, dest, folders2)

def main():
    parser = argparse.ArgumentParser(description='Combine specific folders from two datasets into a new one.')
    parser.add_argument("--folder1", type=str, default="../data/imagenet-20/dataset_FR/low_16", help='Path to the first base folder')
    parser.add_argument("--folder2", type=str, default="../data/imagenet-20/dataset_FR/high_452", help='Path to the second base folder')
    parser.add_argument("--n", type=int, default=10, help='Number of folders to copy from the start of folder1')
    parser.add_argument("--m", type=int, default=10, help='Number of folders to copy from the end of folder2')

    args = parser.parse_args()

    try:
        output_folder = "../data/imagenet-20/dataset_FR/increment_var_FR/low2high/FR-combination"
        create_folder(output_folder)
        combine_folders(args.folder1, args.folder2, output_folder, args.n, args.m)
        print("Folders have been successfully combined.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

