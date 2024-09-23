import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
# import os
import torch
import torch.nn as nn
# import cv2
import numpy as np
#from skimage.measure import compare_ssim
#from skimage.metrics import _structural_similarity
from skimage.metrics import structural_similarity

from core.models.model_factory import Model
from core.utils import preprocess

# 表示从core.utils.util这个模块中导入所有的函数、类和变量
from core.utils.util import *
import core.trainer as trainer
from data_provider.CIKM.data_iterator_1 import *
import math

# tqdm 是一个 Python 库，用于在循环中添加进度条，以便在执行长时间任务时可以实时显示任务的进度。
# 它可以用于各种迭代对象，如列表、元组、集合、字典等，
# 还可以用于文件读写操作的迭代过程。tqdm 提供了简单易用的 API，可以轻松地将进度条添加到你的代码中，并支持自定义样式和功能。
import tqdm
from tqdm import tqdm


# 计算两个输入矩阵的gen_frames和gt_frames之间的平均绝对误差（MAE）
# MAE越小，说明生成帧与真实帧越接近
def batch_mae_frame_float(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    # axis=(1,2)不懂？？？？？
    mae = np.sum(np.absolute(x - y), axis=(1, 2), dtype=np.float32)
    return np.mean(mae)

# 计算两者之间的峰值信噪比，（PSNR）用于评价图片质量的指标
#   [batch_size, height, width, channels]
def batch_psnr(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=(1, 2), dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - ConvLSTM')

# training/test
parser.add_argument('--is_training', type=int, default=0)
# parser.add_argument('--device', type=str, default='gpu:1')
parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--device', type=str, default='cpu')

# data
parser.add_argument('--dataset_name', type=str, default='dendrite')#数据集的名称
parser.add_argument('--is_parallel', type=bool, default=True)#是否使用并行处理，
parser.add_argument('--save_dir', type=str, default='checkpoints_trans/dendrite_convlstm_10')#模型保存路径
parser.add_argument('--gen_frm_dir', type=str, default='result/dendrite_convlstm_10/')#生成帧的目录路径
parser.add_argument('--input_length', type=int, default=30)########输入序列的长度
parser.add_argument('--total_length', type=int, default=50)#总序列长度，默认值是15
parser.add_argument('--img_width', type=int, default=128)
parser.add_argument('--img_channel', type=int, default=1)


# model
parser.add_argument('--model_name', type=str, default='convlstm')
parser.add_argument('--pretrained_model', type=str, default='')#预训练模型的路径或名称，用于指定是否加载预训练模型
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')#有四个隐藏层，每个隐藏层有64个单元
# parser.add_argument('--num_hidden', type=str, default='64,64,64,64')#有四个隐藏层，每个隐藏层有64个单元
parser.add_argument('--filter_size', type=int, default=5)################################
parser.add_argument('--stride', type=int, default=1)#卷积操作的步长，
parser.add_argument('--patch_size', type=int, default=1)#图像块的大小
parser.add_argument('--layer_norm', type=int, default=1)#是否使用层归一化

# scheduled sampling     计划采样
parser.add_argument('--scheduled_sampling', type=int, default=1)#是否使用调度采样，1表示使用
parser.add_argument('--sampling_stop_iter', type=int, default=50)#调度采样停止的迭代次数
parser.add_argument('--sampling_start_value', type=float, default=1.0)#调度采样的初始值
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)#调度采样的变化率

# optimization
parser.add_argument('--lr', type=float, default=0.001)#学习率，用于控制模型参数更新的步长
parser.add_argument('--reverse_input', type=int, default=1)#用于控制是否对输入数据进行反转
parser.add_argument('--batch_size', type=int, default=4)#
parser.add_argument('--max_iterations', type=int, default=1000)#最大迭代次数，即训练过程中允许的最大迭代次数。
parser.add_argument('--display_interval', type=int, default=10)###显示间隔，控制训练过程中输出训练信息的间隔
parser.add_argument('--test_interval', type=int, default=10)####测试间隔，控制进行测试评估的间隔，即在训练过程中每经过多少次迭代后进行一次测试

parser.add_argument('--snapshot_interval', type=int, default=300)#快照间隔，控制保存模型快照的间隔，即在训练过程中每经过多少次迭代后保存一次模型参数。
parser.add_argument('--num_save_samples', type=int, default=10)#保存样本数量，控制在训练过程中保存多少个样本用于后续评估或可视化
parser.add_argument('--n_gpu', type=int, default=4)
# display_interval:

# 这个参数决定了每隔多少次迭代显示一次训练损失（training_loss）。
# 比如，如果 display_interval=10，那么训练过程中每进行10次迭代就会计算并显示一次训练损失。
# test_interval:
#
# 这个参数决定了每隔多少次迭代计算一次验证均方误差（validation_mse）。
# 比如，如果 test_interval=100，那么训练过程中每进行100次迭代就会计算一次验证均方误差，并可能会在验证误差较低时保存模型。
args = parser.parse_args()
batch_size = args.batch_size



#frame_data = imgs = (1, 15, 101, 101, 1)
# （batch_size , sequence_length=15 ， height , width , channels）
def padding_CIKM_data(frame_data):
    shape = frame_data.shape
    batch_size = shape[0]
    seq_length = shape[1]
    padding_frame_dat = np.zeros(( batch_size , seq_length , args.img_width , args.img_width , args.img_channel ))
    padding_frame_dat[:,:,13:-14,13:-14,:] = frame_data
    return padding_frame_dat

def unpadding_CIKM_data(padding_frame_dat):
    return padding_frame_dat[:,:,13:-14,13:-14,:]


# 计划采样技术，eta是计划采样的初始概率，itr是当前的迭代次数或训练步数。
# 计划采样的主要思想是在训练过程中逐渐减少对真实目标序列的依赖，而增加对模型自身预测的依赖。
# 这有助于缓解训练和推理时的不一致问题，因为在推理时，模型只能依赖自身的预测
# 加入有1000个样本训练数据集，设置batch_size为100，则需要10个epoch才能遍历整个数据集。
def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      # //是整除，然后向下取整，比如10除以3等于3.3333，结果向下取整后就等于3.
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))



    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    # real_input_flag = np.reshape(real_input_flag,
    #                        (args.batch_size,
    #                         args.total_length - args.input_length - 1,
    #                         args.img_width // args.patch_size,
    #                         args.img_width // args.patch_size,
    #                         args.patch_size ** 2 * args.img_channel))
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width,
                                  args.img_width ,
                                   args.img_channel))



    return eta, real_input_flag


# wrapper_test,用于测试模型性能，这个函数接受一个模型的输入，然后使用测试数据集来评估模型的性能。
# 初始化损失loss,计数器count,索引index,标志flag,以及存储每帧的均方误差img_mse和结构相似性指数ssim的列表。
def wrapper_test(model):
    test_save_root = args.gen_frm_dir
    clean_fold(test_save_root)
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []
    avg_mse=0
    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    # real_input_flag = np.zeros(
    #     (args.batch_size,
    #      args.total_length - args.input_length - 1,
    #      args.img_width // args.patch_size,
    #      args.img_width // args.patch_size,
    #      args.patch_size ** 2 * args.img_channel))
    real_input_flag = np.zeros(
        (args.batch_size,#4
         args.total_length - args.input_length - 1,#4
         args.img_width,#128
         args.img_width,
         args.img_channel))#1
    output_length = args.total_length - args.input_length
    # MSE = nn.MSELoss(size_average=True)
    # mae = nn.L1Loss(size_average=True)
    MSE = nn.MSELoss(reduction = 'mean')
    mae = nn.L1Loss(reduction = 'mean')
    # print("MSE：", MSE)
    # print("mae：", mae)
    while flag:
        dat, (index, b_cup) = sample(args.batch_size, data_type='test', index=index)

        dat = nor(dat)
        # nor(frames)将输入的图片进行归一化处理
        tars = dat[:, -output_length:]
        # ims = padding_CIKM_data(dat)
        ims=dat
        ims = preprocess.reshape_patch(ims, args.patch_size)
        img_gen, _ = model.test(ims, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen)
        img_out=img_gen[:, -output_length:]

        # img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen[:, -output_length:])

        for i in range(output_length):
            x = dat[:, i + 41, :, :, :]
            gx = img_out[:, i, :, :, :]

            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            for b in range(args.batch_size):
                # score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True, multichannel=False)
                score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True, channel_axis=-1)
                ssim[i] += score

        mse = np.mean(np.square(tars - img_out))
        print('mse:',mse)
        img_out = de_nor(img_out)
        loss = loss + mse
        count = count + 1
        bat_ind = 0

        for ind in range(index - batch_size, index, 1):
            save_fold = test_save_root + 'sample_' + str(ind) + '/'
            clean_fold(save_fold)
            for t in range(41, 51, 1):#5预测10的参数
                imsave(save_fold + 'img_' + str(t) + '.png', img_out[bat_ind, t - 41, :, :, 0])
            bat_ind = bat_ind + 1

        if b_cup == args.batch_size - 1:
            pass
        else:
            flag = False

    avg_mse = avg_mse / (count* args.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(10):
        print(img_mse[i] / (count*args.batch_size))


    ssim = np.asarray(ssim, dtype=np.float32) / (args.batch_size * count)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(10):
        print(ssim[i])

    return loss / count


# 用于验证模型性能，这个函数接受一个模型的输入，然后使用验证集来评估模型的性能。
def wrapper_valid(model):
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):      #total = 15 , input = 10
        img_mse.append(0)   #创建空列表 img_mse 和 ssim，用于存储图像均方误差（MSE）
        ssim.append(0)      #和结构相似性指标（SSIM）

    # real_input_flag = np.zeros(
    #     (args.batch_size,
    #      args.total_length - args.input_length - 1,
    #      args.img_width // args.patch_size,
    #      args.img_width // args.patch_size,
    #      args.patch_size ** 2 * args.img_channel))

    #real_input_flag 是创建一个与模型的输入相关的标志矩阵，用于控制模型输入的方式。
    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width ,
         args.img_width ,
         args.img_channel))

    output_length = args.total_length - args.input_length
    while flag:
        # dat也就是data_iterator中的imgs()为张量
        dat, (index, b_cup) = sample(args.batch_size, data_type='validation', index=index)
        print( "validation:")
        print("dat:",dat.shape)
        print('(index,b_cup):',(index,b_cup))
        dat = nor(dat)
        tars = dat[:, -output_length:]
        # ims = padding_CIKM_data(dat)
        ims=dat
        ims = preprocess.reshape_patch(ims, args.patch_size)
        img_gen, _ = model.test(ims, real_input_flag)
        # print('real_input_flag:',real_input_flag.shape)

        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen)
        img_out=img_gen[:, -output_length:]

        # img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen[:, -output_length:])

        mse = np.mean(np.square(tars-img_out))
        loss = loss+mse
        count = count+1
        if b_cup == args.batch_size-1:
            pass
        else:
            flag = False

    return loss/count


# 接受一个模型的输入，然后使用训练数据集来训练模型。
# 主要用于训练模型，并且在训练过程中监控模型在验证数据集上的性能。当验证性能连续多次没有改善时，就停止训练，
# 并使用最佳的模型在测试数据集上进行评估。这种训练策略被称为早停法。

#tolerate 是容忍次数的计数器。当在验证集上的 MSE（均方误差）不再得到改善时，tolerate 会增加。
# 一旦 tolerate 达到了 limit，表示容忍次数达到了设定的极限，即模型在验证集上的性能已经停止提升。

#limit 是容忍的极限值。当连续 limit 次在验证集上的 MSE 没有得到改善时，就会触发早停策略，
# 停止模型训练，避免过拟合，并且保存在验证集上表现最好的模型参数。
def wrapper_train(model):
    print(args)
    if args.pretrained_model:
        model.load(args.pretrained_model)

    eta = args.sampling_start_value#1
    best_mse = math.inf#表示正无穷大
    tolerate = 0
    limit = 2###########
    best_iter = None
    for itr in tqdm(range(1, args.max_iterations + 1)):

        ims= sample(
            batch_size=batch_size#4
        )
        # ims = padding_CIKM_data(ims)
        # print('ims_type:',type(ims))
        # print('ims:',ims.shape)
        ims = preprocess.reshape_patch(ims, args.patch_size)
        # print('ims_2:',ims.shape)
        ims = nor(ims)
        eta, real_input_flag = schedule_sampling(eta, itr)


        # train函数进行计算
        cost = trainer.train(model, ims, real_input_flag, args, itr)


        if itr % args.display_interval == 0:
            print('itr: ' + str(itr))#str表示一个内置的数据类型，用于表示字符串
            print('training loss: ' + str(cost))



        if itr % args.test_interval == 0:
            print('validation one ')
            valid_mse = wrapper_valid(model)
            print('validation mse is:',str(valid_mse))

            if valid_mse<best_mse:
                best_mse = valid_mse
                best_iter = itr
                tolerate = 0
                model.save()
            else:
                tolerate = tolerate+1

            if tolerate==limit:
                model.load()
                test_mse = wrapper_test(model)
                print('the best valid mse is:',str(best_mse))
                print('the test mse is ',str(test_mse))
                break


# 创建目录，目录的路径是由args.save_dir和args.gen_frm_dir指定的
#save_dir，是用来保存模型的参数，在训练过程中，如果模型在验证数据集上的性能达到了新的最佳水平，就会保存当前的模型参数到这个目录中
#gen_frm_dir是用来保存模型生成的帧的，模型会生成一些帧，这些帧可以直观观察模型的性能。
# if os.path.exists(args.save_dir):
#     shutil.rmtree(args.save_dir)
# os.makedirs(args.save_dir)
#
# if os.path.exists(args.gen_frm_dir):
#     shutil.rmtree(args.gen_frm_dir)
# os.makedirs(args.gen_frm_dir)

# 检查保存目录是否存在，如果不存在则创建
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
# else:
#     shutil.rmtree(args.save_dir)
#     os.makedirs(args.save_dir)

# 检查生成帧目录是否存在，如果不存在则创建
if not os.path.exists(args.gen_frm_dir):
    os.makedirs(args.gen_frm_dir)
# else:
#     shutil.rmtree(args.gen_frm_dir)
#     os.makedirs(args.gen_frm_dir)

gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
args.n_gpu = len(gpu_list)
print('Initializing models')


model = Model(args)
# wrapper_test(model)
# model.load()
# test_mse = wrapper_test(model)
# print('test mse is:',str(test_mse))

if args.is_training:
    wrapper_train(model)
else:
    wrapper_test(model)
#
