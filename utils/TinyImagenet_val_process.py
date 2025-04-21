import os
import shutil


def copy_images(src_dir, dst_dir, extensions=('jpg', 'jpeg', 'png', 'gif')):
    """
    复制src_dir中所有图片到dst_dir。
    参数:
    - src_dir: 源文件夹路径
    - dst_dir: 目标文件夹路径
    - extensions: 文件类型的元组，默认包括jpg, jpeg, png, gif
    """
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源文件夹
    for item in os.listdir(src_dir):
        # 获取文件完整路径
        file_path = os.path.join(src_dir, item)
        # 检查是否为文件且后缀是否在指定的扩展名内
        if os.path.isfile(file_path) and file_path.lower().endswith(extensions):
            # 构建目标文件的完整路径
            dst_file_path = os.path.join(dst_dir, item)
            # 复制文件
            shutil.copy(file_path, dst_file_path)
            print(f'Copied {file_path} to {dst_file_path}')


# Resolve absolute paths to avoid errors depending on the script's running directory
base_dir = os.path.abspath("../data/tiny-imagenet-200")
train_dir = os.path.join(base_dir, "train")
train_copy_dir = os.path.join(base_dir, "train_copy")

val_dir = os.path.join(base_dir, "val")
val_copy_dir = os.path.join(base_dir, "val_copy")
val_annotations_path = os.path.join(val_copy_dir, "val_annotations.txt")

# Ensure the validation directory exists
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

# Create subfolders in the validation directory based on train labels
for name in os.listdir(train_copy_dir):
    sub_folder_dir = os.path.join(val_dir, name)
    if not os.path.exists(sub_folder_dir):
        os.mkdir(sub_folder_dir)

    sub_folder_dir = os.path.join(train_dir, name)
    if not os.path.exists(sub_folder_dir):
        os.mkdir(sub_folder_dir)

# process training dataset
classes = os.listdir(train_copy_dir)
for class_name in classes:
    class_dir = os.path.join(train_copy_dir, class_name) + '/images'
    class_dir_dst = os.path.join(train_dir, class_name)
    copy_images(class_dir, class_dir_dst)

# Process validation annotations
if os.path.exists(val_annotations_path):
    with open(val_annotations_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.split("\t")
            if len(parts) > 1:
                label_val = parts[1]
                src = os.path.join(val_copy_dir, "images", f"val_{i}.JPEG")
                dst = os.path.join(val_dir, label_val, f"val_{i}.JPEG")
                shutil.copy2(src, dst)
else:
    print("Validation annotations file does not exist.")
