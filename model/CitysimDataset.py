''' CitysimDataset继承了torch.utils.data.Dataset，实现三个函数用于初始化和获取地图数据
     HD map 和 trajectory 数据集，并将地图和轨迹数据进行向量化（vector map）归一化等处理
    由__getitem__函数将处理过的数据转为tensor并返回 '''
import os

import torch
import torch.utils.data
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle
import sys
import pandas as pd
import data_process.process_map as pm

obsevertion_frameNum:int=20
future_frameNum:int=50



class CitysimDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.map_directory = "/home/pinkman/PycharmProjects/VectorNet/data_process/hd_maps/University@Alafaya.net.xml"
        # self.map_directory = "/home/pinkman/PycharmProjects/VectorNet/data_process/hd_maps/University@Alafaya_all.net.xml"
        self.traj_directory = "/home/pinkman/PycharmProjects/VectorNet/data_process/trajectory/train_1.2"
        self.map = pm.SumoMap(self.map_directory)
        self.map_range = self.map.map_range




    def __len__(self):
        file_count = 0
        # read trajectory files and generate vector trajectory
        for filename in sorted(os.listdir(self.traj_directory)):
            if filename.endswith(".csv"):
                # file_path = os.path.join(self.traj_directory, filename)
                file_count += 1

        return file_count

    def __getitem__(self, index):
        self.traj_vector, self.traj_attribute = self.get_trajectory(index)
        center_xy = self.traj_vector[obsevertion_frameNum][0]
        lane_id = self.traj_attribute['lane_id'][obsevertion_frameNum - 1]
        self.traj_feature = []
        self.map_feature = []
        # normalize
        for vector in self.traj_vector:
            self.traj_feature.append(center_xy - vector)

        rotate_matrix = self.map.get_rotate_matrix(self.traj_feature[obsevertion_frameNum])
        self.traj_feature = np.asarray(self.traj_feature).dot(rotate_matrix.T).reshape(-1, 4)
        self.traj_feature = self.map.normalize_coordinate(self.traj_feature)

        # get vectormap of center_xy and normalize
        self.map_vector, self.map_attribute = self.map.generate_vector_map(lane_id, center_xy)
        for vector in self.map_vector:
            self.map_feature.append(center_xy - vector)
        self.map_feature = np.asarray(self.map_feature).dot(rotate_matrix.T).reshape(-1, 4)
        self.map_feature = self.map.normalize_coordinate(self.map_feature)

        # print(f'traj:{self.traj_feature}, length:{len(self.traj_feature)}, map:{self.map_feature}, length:{len(self.map_feature)}')
        self.traj_feature = torch.tensor(self.traj_feature).reshape(-1, 4)
        self.map_feature = torch.tensor(self.map_feature).reshape(-1, 4)

        # self.traj_feature = torch.tensor(self.traj_feature).reshape(-1, 4).unsqueeze(dim=0)
        # self.map_feature = torch.tensor(self.map_feature).reshape(-1, 4).unsqueeze(dim=0)
        # print("dataloader:", self.traj_feature.shape, self.map_feature.shape)
        # print(self.traj_feature, self.map_feature)
        return self.traj_feature, [self.map_feature]

    def get_info(self, index):
        self.traj_vector, self.traj_attribute = self.get_trajectory(index)
        center_xy = self.traj_vector[obsevertion_frameNum][0]
        rotate_matrix = self.map.get_rotate_matrix(self.traj_feature[obsevertion_frameNum])
        return center_xy, rotate_matrix, self.map_range

    def get_trajectory(self, index):
        traj_vector = []
        polyline = []
        attribute = dict(LateralLanePosition=[], heading_course=[], next_laneId=[], timestamp=[], lane_id=[])
        traj_df = self.read_file(index)
        traj_df = traj_df[traj_df['agentType'] == 'ego']
        # observetion_df = traj_df[traj_df['frameNum'] == float(obsevertion_frameNum)]
        # center_x, center_y = observetion_df['head_x'], observetion_df['head_y']
        frameNum = 1.0
        for col, row in traj_df.iterrows():
            # xy = np.array(center_x - [row['head_x'], center_y - row['head_y']])
            xy = np.array([row['head_x'], row['head_y']])
            polyline.append(xy)
            if row['frameNum'] < 50:
                attribute['LateralLanePosition'].append(row['LateralLanePosition'])
                attribute['heading_course'].append(row['heading_course'])
                attribute['next_laneId'].append(row['next_laneId'])
                attribute['timestamp'].append(row['timestamp'])
                attribute['lane_id'].append(row['lane_id'])

        for i in range(len(polyline) - 1):
            start_point = polyline[i]
            end_point = polyline[i + 1]
            traj_vector.append([start_point, end_point])

        return traj_vector, attribute

    def read_file(self, index):
        # ego_list = []
        # file_count = 0
        # df_list = []
        # # read trajectory files and generate vector trajectory
        # for filename in sorted(os.listdir(self.traj_directory)):
        #     if filename.endswith(".csv"):
        #         file_path = os.path.join(self.traj_directory, filename)
        #         traj_df = pd.read_csv(file_path)
        #         # process trajectory data
        #         df_list.append(traj_df)
        #         file_count += 1
        #         ego_list.append(int(df.loc[df['agentType'] == 'ego', 'carId'].iloc[0]))
        filename = str(index) + ".csv"
        file_path = os.path.join(self.traj_directory, filename)
        traj_df = pd.read_csv(file_path)
        # print(filename)
        return traj_df

    def generate_vector_map(self):
        return

if __name__ == "__main__":
    dataloader = CitysimDataset()
    # df = pd.read_csv("/home/pinkman/PycharmProjects/VectorNet/data_process/trajectory/0.csv")
    # print(df.columns)
    dataloader.get_trajectory(0)
    dataloader.__getitem__(0)





