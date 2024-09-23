
import numpy as np
import shutil
import copy
import os


# def nor(frames):
#     new_frames = frames.astype(np.float32)/255.0
#     return new_frames
#
# def de_nor(frames):
#     new_frames = copy.deepcopy(frames)
#     new_frames *= 255.0
#     new_frames = new_frames.astype(np.uint8)
#     return new_frames

def nor(frames):
    new_frames = np.array(frames, dtype=np.float32) /255.0
    return new_frames

def de_nor(frames):
    # new_frames = copy.deepcopy(frames)
    # new_frames *= 255.0
    # new_frames = new_frames.uint8
    new_frames = np.array(frames, dtype=np.float32) * 255.0
    new_frames = np.round(new_frames).astype(np.uint8)
    return new_frames


def normalization(frames,up=80):
    new_frames = frames.astype(np.float32)
    new_frames /= (up/2)
    new_frames -= 1
    return new_frames

def denormalization(frames,up=80):
    new_frames = copy.deepcopy(frames)
    new_frames += 1
    new_frames *= (up/2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def clean_fold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


# 这里定义了一些用于数据处理的函数：
#
# nor(frames): 将输入的图像帧数据进行归一化处理，将数据类型转换为 np.float32 并将像素值除以 255.0，
# 将像素值缩放到 [0, 1] 的范围内。
#
# de_nor(frames): 将归一化处理后的图像帧数据还原为原始像素值。
# 首先复制输入的帧数据，然后将像素值乘以 255.0 并将数据类型转换为 np.uint8，将像素值缩放回 [0, 255] 的范围内。
#
# normalization(frames, up=80): 将输入的图像帧数据进行标准化处理，将数据类型转换为 np.float32，
# 然后将像素值除以 (up/2)，最后减去 1，将像素值缩放到 [-1, 1] 的范围内。
#
# denormalization(frames, up=80): 将标准化处理后的图像帧数据还原为原始像素值。
# 首先复制输入的帧数据，然后将像素值加上 1，然后乘以 (up/2)，最后将数据类型转换为 np.uint8，将像素值缩放回原始范围。
#
# clean_fold(path): 如果指定路径存在，则删除该路径下的所有文件和文件夹，并重新创建一个空的文件夹；
# 如果指定路径不存在，则直接创建一个空的文件夹。这个函数用于清理指定路径下的内容，以便重新使用该路径。