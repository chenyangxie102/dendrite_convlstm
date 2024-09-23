import os
import imageio

# 原始数据集根目录
original_data_root = 'images_55'

# 新数据集根目录
new_data_root = 'dendrit'

# 创建新数据集目录
os.makedirs(new_data_root, exist_ok=True)

# 遍历样本文件夹（sample_1 到 sample_120）
for sample_num in range(1, 121):
    sample_folder = f'sample_{sample_num}'
    sample_path = os.path.join(original_data_root, sample_folder)
    new_sample_path = os.path.join(new_data_root, sample_folder)
    os.makedirs(new_sample_path, exist_ok=True)

    # 遍历样本文件夹中的图片（img_1 到 img_50）
    for img_num in range(1, 51):
        filename = f'img_{img_num}.png'
        image_path = os.path.join(sample_path, filename)
        new_image_path = os.path.join(new_sample_path, filename)

        # 读取原始图片并提取第二个通道的值
        original_image = imageio.imread(image_path)
        second_channel = original_image[:, :, 1]

        # 保存处理后的图片到新的数据集目录
        imageio.imwrite(new_image_path, second_channel)

print("图片处理完成，已保存到新数据集目录 'dendrit' 中。")
