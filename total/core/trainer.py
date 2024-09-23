import os.path
import datetime
import numpy as np
from core.utils import preprocess

# model是要训练的模型，ims是输入图片序列的张量
# real_input_flag是一个张量，指示哪个部分是真实输入，哪些部分是模型的预测的输入
# configs是训练配置参数
# itr是当前的迭代次数。
# 在训练过程中。首先调用模型的‘train’方法来计算损失‘cost’。如果配置中设置了‘reverse_input’为真
# 则会进行反向输入的训练，最后返回计算得到的总损失‘cost’


# cost 是训练过程中的损失值，用来衡量模型预测与真实值之间的差异。
# 在训练过程中，优化算法会尝试最小化损失函数，使得模型能够更好地拟合训练数据，
# 从而提高在测试集或验证集上的性能表现。
# 因此，cost 的作用是作为优化算法的目标函数，指导模型参数的更新，
# 使得模型能够逐渐学习到数据的特征和规律。
def train(model, ims, real_input_flag, configs, itr):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    return cost


