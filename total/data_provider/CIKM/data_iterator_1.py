import sys
import os
import imageio
from imageio.v2 import imsave,imread
# os.path.dirname(path)用于获取一个文件路径的目录部分,__file__是内置属性，表示当前文件的路径
# os.path.abspath用于取绝对文件目录路径
curPath = os.path.abspath(os.path.dirname(__file__))
# os.path.split将文件名与路径分割开来，[0]表示选取第n个分片
rootPath = os.path.split(curPath)[0]
# sys.path是一个列表list，它里面包含了已经添加到系统的环境变量路径，添加自己的引用模块搜索目录时，用sys.path.append()
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
# from core.utils.util import *
from torch.utils import data
# from scipy.misc import imsave,imread
from torch.utils.data import DataLoader
import numpy as np
import random
import torch


# 定义一个CIKM_Datasets类，该类继承自Data.Dataset.用于处理和加载数据
# 未使用
class CIKM_Datasets(data.Dataset):
    def __init__(self,root_path):
        self.root_path = root_path

    # index用于指定访问元素的位置。
    def __getitem__(self, index):
        self.folds = self.root_path+'sample_'+str(index+1)+'/'
        files = os.listdir(self.folds)
        files.sort()
        imgs = []
        for file in files:
            imgs.append(imread(self.folds+file)[:,:,np.newaxis])
        imgs = np.stack(imgs,0)
        # torch.from_numpy()表示将numpy数组转化为pytorch张量
        imgs = torch.from_numpy(imgs).cuda()
        # 所以，in_imgs（前五张图片）和out_imgs（剩余的图像）是张量形式
        in_imgs = imgs[:5]
        out_imgs = imgs[5:]
        return in_imgs,out_imgs

    def __len__(self):
        return len(os.listdir(self.root_path))

def data_process(filename,data_type,dim=None,start_point = 0):
    save_root = '/mnt/A/CIKM2017/CIKM_datasets/'+data_type+'/'
    if start_point == 0:
        clean_fold(save_root)

    with open(filename) as fr:
        if data_type == 'train':
            sample_num = 10000
            validation = random.sample(range(1, 10000 + 1), 2000)
            save_validation_root = '/mnt/A/CIKM2017/CIKM_datasets/validation/'
            clean_fold(save_validation_root)
        elif data_type == 'test':
            sample_num = 2000+start_point
        print('the number of '+data_type+' datasets is:',str(sample_num))
        validation_count = 1
        train_count = 1
        for i in range(start_point+1,sample_num+1):
            print(data_type+' data loading complete '+str(100.0*(i+1)/sample_num)+'%')
            if data_type == 'train':
                if i in validation:
                    save_fold = save_validation_root+'sample_'+str(validation_count)+'/'
                    validation_count = validation_count + 1
                else:
                    save_fold = save_root + 'sample_' + str(train_count) + '/'
                    train_count = train_count + 1
            else:
                save_fold = save_root+'sample_'+str(i)+'/'
            clean_fold(save_fold)

            line = fr.readline().strip().split(' ')
            cate = line[0].split(',')
            id_label = [cate[0]]
            record = [int(cate[2])]
            length = len(line)

            for i in range(1, length):
                record.append(int(line[i]))

            mat = np.array(record).reshape(15, 4, 101, 101).astype(np.uint8)

            # deals with -1
            mat[mat == -1] =  0

            if dim == None:
                pass
            else:
                mat = mat[:,dim]

            for t in range(1,16):
                img = mat[t-1]
                # print(img.shape)
                img_name = 'img_'+str(t)+'.png'
                imsave(save_fold+img_name,img)

def sub_sample(batch_size,mode = 'random',data_type='train',index = None,type = 7):
    if type not in [4,5,6,7]:
        raise ('error')
    save_root = '/mnt/A/CIKM2017/CIKM_datasets/' + data_type + '/'
    if data_type == 'train':
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,8000)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []
                for t in range((16-type-8),16):
                    img_path = img_fold + 'img_'+str(t)+'.png'
                    img = imread(img_path)[:,:,np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range((16-type-8), 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>4001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>4001:
                index = 4001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range((16-type-8), 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==4001:
            return imgs,(index,0)
        return imgs,(index,b_cup)

    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>2001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>2001:
                index = 2001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range((16-type-8), 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==2001:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")



# data_type可以等于train（20个）'  、'test（10个）'、'validation（20个）',index是起始索引
#（index,b_cup）表示一个元组，即采样结束后的索引和批次中最后一个有效样本的索引
def sample(batch_size,mode = 'random',data_type='train',index = None):
    # print(data_type)
    # save_root = 'D:/深度学习/convlstm - 001/data/' + data_type + '/'
    # save_root = '/work/home/maohong/chenyang/convlstm/convlstm - 001/data/' + data_type + '/'
    # save_root = '/work/home/maohong/chenyang/convlstm/convlstm - 001/data/' + data_type + '/'
    #绝对路径
    # save_root = 'H:/1Datebase/convlstm - 001(1)/convlstm - 001/data/' + data_type + '/'
    #相对路径
    # save_root = 'H:/1Datebase/dendrite_convlstm/dendrite_data_22/' + data_type + '/'
    save_root =  'dendrite_data_44/'+  data_type + '/'

    # ******train******
    if data_type == 'train':            #20
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,88)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []
                for t in range(1,51):
                    img_path = img_fold + 'img_'+str(t)+'.png'
                    img = imread(img_path)[:,:,np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>89 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1
            for batch_idx in range(batch_size):
                if index>89:
                    index = 89
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 51):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 51:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    # ******test******
    elif data_type == 'test':           #10
        if index == None:
            raise('index need be initialize')
        if index>12 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>12:
                index = 12
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 51):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==12:
            return imgs,(index,0)
        return imgs,(index,b_cup)

    # ******validation******
    elif data_type == 'validation':         #20
        if index == None:
            raise('index need be initialize')
        if index>12 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>12:
                index = 12
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 51):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==12:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")


if __name__ == '__main__':
    pass