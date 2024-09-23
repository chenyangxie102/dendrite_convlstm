

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import predict


# 首先接收一个参数“configs”该参数包含了模型的配置信息
# configs.num_hidden表示模型中的每一层隐藏单元的数量
# num_layers表示计算隐藏层的数量，通过获取num_hidden的长度
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'convlstm':predict.ConvLSTM,
            'predrnn':predict.PredRNN,
            'predrnn_plus': predict.PredRNN_Plus,
            'interact_convlstm': predict.InteractionConvLSTM,
            'interact_predrnn':predict.InteractionPredRNN,
            'interact_predrnn_plus':predict.InteractionPredRNN_Plus,
            'cst_predrnn':predict.CST_PredRNN,
            'sst_predrnn': predict.SST_PredRNN,
            'dst_predrnn':predict.DST_PredRNN,
            'interact_dst_predrnn': predict.InteractionDST_PredRNN,
        }

        if configs.model_name in networks_map:

            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            # self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)

            # self.network = Network(self.num_layers, self.num_hidden, configs).cuda()
            # self.network = Network(self.num_layers, self.num_hidden, configs).cpu()

        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        if self.configs.is_parallel:
            self.network = nn.DataParallel(self.network)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss(size_average=True)
        self.MAE_criterion = nn.L1Loss(size_average=True)


    def save(self,ite = None):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        if ite == None:
            checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        else:
            checkpoint_path = os.path.join(self.configs.save_dir, 'model_'+str(ite)+'.ckpt')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self):
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])
        print('model has been loaded')



    # 接受两个参数：frames（表示输入的视频帧序列，其形状为（barch_size,total_length, height , width ,channel））
    #               mask （表示掩码序列，用于在预测过程中控制模型是否能够观察到当前的输帧入，如果某一帧在mask中对应值为0，则无法观测到该帧
    #                      其形状与frames相同(batch_size, total_length - input_length - 1, height, width, channel) ）
    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        # frames_tensor = torch.FloatTensor(frames).cuda()
        # mask_tensor = torch.FloatTensor(mask).cuda()
        # frames_tensor = torch.FloatTensor(frames).cpu()
        # mask_tensor = torch.FloatTensor(mask).cpu()

        self.optimizer.zero_grad()
        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])+\
               self.MAE_criterion(next_frames, frames_tensor[:, 1:])
               # 0.02*self.SSIM_criterion(next_frames, frames_tensor[:, 1:])
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()
        # return loss.detach().to(self.configs.device).numpy()
    # 返回一个loss值，表示训练过程中模型预测结果与真实帧之间的损失


    #mask参数是一个张量，用于表示输入帧中哪些部分是需要进行预测的（即需要网络来生成），
    # 哪些部分是已知的（即不需要生成，可以直接使用输入帧）。
    # 在序列预测任务中，通常会将输入帧的一部分作为已知信息，而将另一部分作为待预测的目标
    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        # frames_tensor = torch.FloatTensor(frames).cuda()
        # mask_tensor = torch.FloatTensor(mask).cuda()
        # frames_tensor = torch.FloatTensor(frames).cpu()
        # mask_tensor = torch.FloatTensor(mask).cpu()

        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) +\
               self.MAE_criterion(next_frames,frames_tensor[:,1:])
               # + 0.02 * self.SSIM_criterion(next_frames, frames_tensor[:, 1:])

        return next_frames.detach().cpu().numpy(),loss.detach().cpu().numpy()

        # return next_frames.detach().to(self.configs.device).numpy(), loss.detach().to(self.configs.device).numpy()
    #返回next_frames，表示模型预测的下一帧图像序列(batch_size, total_length - 1, height, width, channel)
    #  和loss 表示模型预测结果与真实帧之间的损失