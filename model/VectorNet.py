import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from subgraph import SubgraphNet
from GNN import GraphAttentionNet


traj_feature_dim:int=4
map_feature_dim:int=8
decoder_size:int=64
prediction_step:int=30


class VectorNet(nn.Module):
    def __init__(self, traj_features, map_features, cfg):
        super().__init__()
        if cfg is None:
            cfg = dict(device=torch.device('cpu'))
        self.cfg = cfg

        #encoder
        self.traj_subgraphNet = SubgraphNet(traj_features)
        self.map_subgraphNet = SubgraphNet(map_features)
        self.graphNet = GraphAttentionNet()

        #decoder

        prediction_step = 2*(49 - self.cfg['last_observe'])  # need change
        self.mlp1 = nn.Linear(decoder_size, decoder_size)
        nn.init.kaiming_normal_(self.mlp1.weight)
        self.layer_norm = nn.LayerNorm(64)
        self.mlp2 = nn.Linear(decoder_size, prediction_step)
        nn.init.kaiming_normal_(self.mlp2.weight)
        # # MSEloss 就是 Gaussian NLL，均方损失函数，reduce为True则loss返回标量（各元素均方差之和），size_average=False不求均值
        self.MSE_loss = nn.MSELoss(size_average=False, reduce=True)

    def _forward_train(self, trajectory_batch, vectornet_map):
        ''' 分别把前2秒traj和vectormap放入traj_subgraphnet和map_subgraphnet，得到的结果放入polyline_list（后3秒traj作为label）
            对polyline_list再做normalize得到polyline_feature放入graphnet
            再经过relu和fc2得到后2秒的traj与label计算loss并返回'''

        # print("trajectory_batch ", trajectory_batch.shape)
        # print("vectornet_map", len(vectornet_map), vectornet_map[0].shape)
        batch_size = trajectory_batch.shape[0]
        label = trajectory_batch[:, self.cfg['last_observe']:, 2:4] # label.shape()->([2, 19, 2]), last 19 trajectory_vector, [x1, y1]
        predict_list = []
        for i in range(batch_size):
            polyline_list = []
            polyline_list.append(self.traj_subgraphNet(trajectory_batch[i, :self.cfg['last_observe']]).unsqueeze(0)) # 将轨迹前last_observe个点数据(30,6)输入sub_graphnet

            # 每个batch有多个vector_map,(轨迹点周围的地图),每个vec_map都是(18,8)
            for vec_map in vectornet_map:
                # vec_map = vec_map.to(device=self.cfg['device'])
                # vec_map = vec_map.to(torch.float)
                vec_map = vec_map.to(self.cfg['device'], torch.float)
                map_feature = self.map_subgraphNet(vec_map.squeeze())
                polyline_list.append(map_feature.unsqueeze(0))

            polyline_feature = F.normalize(torch.cat(polyline_list, dim=0), p=2, dim=1) # L2 Normalize, torch.Size([1+n, 128])
            out = self.graphNet(polyline_feature) # (1+n, 64)
            # print("out: ", out.shape)
            decode_data_perstep = self.mlp2(F.relu(self.layer_norm(self.mlp1(out[0].unsqueeze(0))))).view(1, -1, 2) # (1,19,2)
            decode_data = torch.cumsum(decode_data_perstep, dim=0)
            predict_list.append(decode_data)

        prediction_batch = torch.cat(predict_list, dim=0)
        loss = self.MSE_loss(prediction_batch, label)
        if np.isnan(loss.item()):
            raise Exception("Loss Error")

        return loss

    def _forward_test(self, trajectory_batch):
        batch_size = trajectory_batch.shape[0]

        traj_label = trajectory_batch[:, self.cfg['last_observe']:, 2:4]
        result, label = dict(), dict()
        for i in range(batch_size):
            polyline_feature = self.traj_subgraphNet(trajectory_batch[i, :self.cfg['last_observe']]).unsqueeze(0)
            polyline_feature = F.normalize(polyline_feature, p=2, dim=1)
            out = self.graphNet(polyline_feature)
            decode_data_perstep = self.mlp2(F.relu(self.layer_norm(self.mlp1(out)))).view(-1, 2)
            decode_data = torch.cumsum(decode_data_perstep, dim=0)
            key = str(trajectory_batch[i, 0, -1].int().item())

            predict_step = self.cfg['predict_step']
            result.update({key: decode_data[:predict_step]})
            label.update({key: traj_label[i, :predict_step]})

        return result, label

    def forward(self, trajectory, vectormap):
        if self.training: # extend from nn.Module
            return self._forward_train(trajectory, vectormap)
        else:
            return self._forward_test(trajectory)



