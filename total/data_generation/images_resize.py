# import os
# from PIL import Image
#
# # 输入和输出目录
# input_dir = 'images_1'
# output_dir = 'images_11'
#
# # 遍历每个 sample_x 文件夹
# for x in range(1, 112):
#     input_sample_dir = os.path.join(input_dir, f'sample_{x}')
#     output_sample_dir = os.path.join(output_dir, f'sample_{x}')
#
#     # 创建输出目录
#     if not os.path.exists(output_sample_dir):
#         os.makedirs(output_sample_dir)
#
#     # 遍历每个 img_y.png 图片
#     for y in range(0, 4001, 5):
#         input_img_path = os.path.join(input_sample_dir, f'img_{y}.png')
#         output_img_path = os.path.join(output_sample_dir, f'img_{y // 5 + 1}.png')
#
#         # 检查输入图片是否存在
#         if os.path.exists(input_img_path):
#             # 打开图片
#             with Image.open(input_img_path) as img:
#                 # 改变图片尺寸为 128x128
#                 resized_img = img.resize((128, 128))
#
#                 # 保存新图片
#                 resized_img.save(output_img_path)


# import os
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor
#
# # 输入和输出目录
# input_dir = 'images_1'
# output_dir = 'images_11'


# 重命名和调整大小的函数
# def process_image(x, y):
#     input_sample_dir = os.path.join(input_dir, f'sample_{x}')
#     output_sample_dir = os.path.join(output_dir, f'sample_{x}')
#
#     # 创建输出目录
#     if not os.path.exists(output_sample_dir):
#         os.makedirs(output_sample_dir)
#
#     input_img_path = os.path.join(input_sample_dir, f'img_{y}.png')
#     output_img_path = os.path.join(output_sample_dir, f'img_{y // 5 + 1}.png')
#
#     # 检查输入图片是否存在
#     if os.path.exists(input_img_path):
#         with Image.open(input_img_path) as img:
#             resized_img = img.resize((128, 128))
#             resized_img.save(output_img_path)
#
#
# # 使用 ThreadPoolExecutor 并行处理
# with ThreadPoolExecutor() as executor:
#     # 遍历每个 sample_x 文件夹
#     for x in range(1, 112):
#         # 使用 executor.map 并行处理每个 img_y.png 图片
#         list(executor.map(process_image, [x] * 1000, range(0, 4001, 5)))

import os
from PIL import Image
from multiprocessing import Pool

# 输入和输出目录
input_dir = 'images_2'
output_dir = 'images_222'

def process_image(x):
    input_sample_dir = os.path.join(input_dir, f'sample_{x}')
    output_sample_dir = os.path.join(output_dir, f'sample_{x}')

    # 创建输出目录
    if not os.path.exists(output_sample_dir):
        os.makedirs(output_sample_dir)

    for y in range(0, 4001, 20):
        input_img_path = os.path.join(input_sample_dir, f'img_{y}.png')
        output_img_path = os.path.join(output_sample_dir, f'img_{y // 20 + 1}.png')

        # 检查输入图片是否存在
        if os.path.exists(input_img_path):
            with Image.open(input_img_path) as img:
                # 改变图片尺寸为 128x128
                resized_img = img.resize((128, 128))

                # 保存新图片
                resized_img.save(output_img_path)

if __name__ == '__main__':
    # 使用多进程池
    with Pool() as p:
        p.map(process_image, range(1, 112))
