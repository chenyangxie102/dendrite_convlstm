import os
import pickle
import torch
import numpy as np
# import netCDF4
from torch.utils.data.dataloader import DataLoader
import torch
from torch.utils.data.dataset import Dataset
# data_path1='D:/unet/data_radar/train/'
# data_path2='D:/unet/data_radar/test/'
class VDTrainDataset(Dataset):
    def __init__(self, data_path1):
    #def __init__(self, data_path, transform=None):
        self.data_path = data_path1
#        self.transform = transform
        self.num = os.listdir(data_path1)


    def __len__(self):
        return len(self.num)

    def __getitem__(self, index):
        filename = 'francedata%d.pickle'%(index+1)
        f = open(self.data_path + filename, 'rb')
        data = pickle.load(f)
        input_data = data['data']
        # input_data=input_data.astype(np.float32)/255.0
        input_data = torch.from_numpy(input_data).float()
        # input_data = torch.from_numpy(input_data).float()
        # label_data = input_data.reshape(20, 128, 128,1).float()
        # # print(label_data.shape)
        # # label_data = self.avepool(label_data)
        # input=label_data

        #rate = (raindata == 0).sum()/(400*400*15)

        return input_data


class VDTestDataset(Dataset):
    def __init__(self, data_path):
    #def __init__(self, data_path, transform=None):
        self.data_path = data_path
#        self.transform = transform
        self.num = os.listdir(data_path)
        self.avepool = torch.nn.AvgPool2d((2, 2))

    def __len__(self):
        return len(self.num)

    def __getitem__(self, index):
        filename = 'francedata%d.pickle'%(index+1)
        f = open(self.data_path + filename, 'rb')
        data = pickle.load(f)
        raindata = data['data'][:, :, :]
        raindata = (raindata)/(700)
        input_data = raindata[::, :, :]
        input_data = torch.from_numpy(input_data).float()
        label_data = input_data.reshape(1, 5, 400, 400).float()
        input = label_data[:, 0:4, :, :].reshape(4, 400, 400)
        label = label_data[:, -1, :, :].reshape(1, 400, 400)
        # rate = (raindata == 0).sum()/(400*400*15)

        return input, label



# if __name__ == '__main__':
#     # train_loader = VDTrainDataset2plus(data_path)
#     # work_loader = iter(train_loader)
#     # print(work_loader)
#     # input, target, mask = next(work_loader)
#     # print(input.size(), target.size(),mask.size(), len(train_loader))
#     data_path="D:/TCTN-pytorch-main/data50/test/"
#     train_Dataset = VDTrainDataset(data_path)
#     # val_Dataset = VDvalDataset(data_path2)
#
#     train_loader = DataLoader(dataset=train_Dataset, batch_size=8, shuffle=True, num_workers=1,
#                               pin_memory=True)
#     print(len(train_loader))
#     for i, (input, target) in enumerate(train_loader ):
#         print(i)
#         print(input.shape)
#         print(target.shape)
