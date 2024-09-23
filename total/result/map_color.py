# import os
# from PIL import Image
# import numpy as np
#
# # 设置路径
# # data_dir = 'test'
# # output_dir = 'test_color'
#
# data_dir = 'dendrite_convlstm_12_star'
# output_dir = 'dendrite_convlstm_12_star_color_1'
# num_samples = 12
# num_imgs_per_sample = 10
#
# # 创建输出目录
# os.makedirs(output_dir, exist_ok=True)
#
# # 水蓝色的RGB值
# aqua_blue = np.array([0, 123, 255], dtype=np.uint8)
# white = np.array([255, 255, 255], dtype=np.uint8)
#
# # 处理每个样本
# for sample_idx in range(1, num_samples + 1):
#     sample_dir = os.path.join(data_dir, f'sample_{sample_idx}')
#     new_sample_dir = os.path.join(output_dir, f'sample_{sample_idx}')
#     os.makedirs(new_sample_dir, exist_ok=True)
#
#     for img_idx in range(41, num_imgs_per_sample + 41 ):
#         img_path = os.path.join(sample_dir, f'img_{img_idx}.png')
#         new_img_path = os.path.join(new_sample_dir, f'img_{img_idx}.png')
#
#         # 打开图像并转换为灰度模式
#         with Image.open(img_path) as img:
#             img = img.convert("L")  # 确保图像是单通道
#             img_array = np.array(img) / 255.0  # 归一化到0-1之间
#             # print(img.shape)
#             # 创建RGB图像
#             rgb_image = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
#
#             # 将值为0的部分设置为水蓝色
#             mask_zero = img_array == 0
#             rgb_image[mask_zero] = aqua_blue
#
#             # 将0到1渐变的值从白色渐变成水蓝色
#             mask_non_zero = img_array > 0
#             rgb_image[mask_non_zero] = (1 - img_array[mask_non_zero][:, None]) * aqua_blue + img_array[mask_non_zero][:, None] * white
#
#             # 保存RGB图像
#             new_img = Image.fromarray(rgb_image, 'RGB')
#             new_img.save(new_img_path)
#
# print("所有图片处理并保存完成。")


import os
from PIL import Image
import numpy as np

# 设置路径
# data_dir = 'test'
# output_dir = 'test_color'

data_dir = 'dendrite_convlstm_12_star'
output_dir = 'dendrite_convlstm_12_star_color_2'
num_samples = 12
num_imgs_per_sample = 10

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 水蓝色的RGB值酒红色
aqua_blue = np.array([152,61,97], dtype=np.uint8)
white = np.array([255, 255, 255], dtype=np.uint8)

# 处理每个样本
for sample_idx in range(1, num_samples + 1):
    sample_dir = os.path.join(data_dir, f'sample_{sample_idx}')
    new_sample_dir = os.path.join(output_dir, f'sample_{sample_idx}')
    os.makedirs(new_sample_dir, exist_ok=True)

    for img_idx in range(41, num_imgs_per_sample + 41):
        img_path = os.path.join(sample_dir, f'img_{img_idx}.png')
        new_img_path = os.path.join(new_sample_dir, f'img_{img_idx}.png')

        # 打开图像并转换为灰度模式
        with Image.open(img_path) as img:
            img = img.convert("L")  # 确保图像是单通道
            img_array = np.array(img) / 255.0  # 归一化到0-1之间

            # 创建RGB图像
            rgb_image = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)

            # 将值为0的部分设置为白色
            mask_zero = img_array == 0
            rgb_image[mask_zero] = white

            # 将值为1的部分设置为水蓝色
            mask_one = img_array == 1
            rgb_image[mask_one] = aqua_blue

            # 将0到1渐变的值从白色渐变成水蓝色
            mask_non_zero_non_one = (img_array > 0) & (img_array < 1)
            rgb_image[mask_non_zero_non_one] = (1 - img_array[mask_non_zero_non_one][:, None]) * white + img_array[mask_non_zero_non_one][:, None] * aqua_blue

            # 保存RGB图像
            new_img = Image.fromarray(rgb_image, 'RGB')
            new_img.save(new_img_path)

print("所有图片处理并保存完成。")

