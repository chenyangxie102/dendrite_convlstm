import os
import shutil

# 输入和输出目录
input_dir = 'images_22'
output_dir = 'dendrite_data/dendrite_data'

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建 train, validation, test 目录
train_dir = os.path.join(output_dir, 'train')
validation_dir = os.path.join(output_dir, 'validation')
test_dir = os.path.join(output_dir, 'test')

# 创建 train, validation, test 目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 移动和重命名 sample 到 test 目录
for i in range(1, 112, 10):
    sample_name = f'sample_{i}'
    input_sample_dir = os.path.join(input_dir, sample_name)
    output_sample_dir = os.path.join(test_dir, f'sample_{i // 10 + 1}')

    shutil.copytree(input_sample_dir, output_sample_dir)

# 移动和重命名 sample 到 validation 目录
for i in range(5, 106, 10):
    sample_name = f'sample_{i}'
    input_sample_dir = os.path.join(input_dir, sample_name)
    output_sample_dir = os.path.join(validation_dir, f'sample_{(i - 5) // 5 + 1}')

    shutil.copytree(input_sample_dir, output_sample_dir)

# 移动和重命名剩下的 sample 到 train 目录
for i in range(1, 112):
    if i not in range(1, 112, 10) and i not in range(5, 106, 10):
        sample_name = f'sample_{i}'
        input_sample_dir = os.path.join(input_dir, sample_name)
        output_sample_dir = os.path.join(train_dir, f'sample_{i}')

        shutil.copytree(input_sample_dir, output_sample_dir)
