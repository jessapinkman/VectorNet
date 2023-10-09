# -*- encoding: utf-8 -*-
"""
@File    :   test.py   
@Contact :   jessapinkman@163.com
@License :   (C)Copyright 2023
 
@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
2023/8/15 上午10:29   JessaPinkman  1.0         None
"""
# !/usr/bin/python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from CitysimDataset import CitysimDataset
from tqdm import tqdm
import warnings
from model.VectorNet import VectorNet
warnings.filterwarnings('ignore')
import pprint
import time
import sys
import matplotlib.pyplot as plt


def render_traj(traj_batch):
    print(traj_batch)
    traj = traj_batch[0].cpu().numpy()
    rows = np.size(traj, 0)
    print(rows)
    for i in range(rows):
        plt.annotate('', xy=(traj[:, 2][i], traj[:, 3][i]), xytext=(traj[:, 0][i], traj[:, 1][i]),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.show()


def show_result(traj_batch, map_batch, single_result):
    # print(map_batch)
    # print(single_result)
    # sys.exit()
    traj = traj_batch[0].cpu().numpy()
    mv = []
    for vec_map in map_batch:
        vec_map = vec_map[0].cpu().numpy()
        vec_map = np.reshape(vec_map[:, 0:4], (-1, 8))
        mv.append(vec_map)
    for key in single_result:
        pred = single_result[key].cpu().numpy() / 10
    map_vec = np.vstack(mv)
    rows = np.size(map_vec, 0)
    map_count = rows // 9
    print(map_count)
    for i in range(map_count):
        plt.plot(map_vec[i * 9:(i + 1) * 9, 0], map_vec[i * 9:(i + 1) * 9, 1])
        plt.plot(map_vec[i * 9:(i + 1) * 9, 2], map_vec[i * 9:(i + 1) * 9, 3])
        plt.plot(map_vec[i * 9:(i + 1) * 9, 4], map_vec[i * 9:(i + 1) * 9, 5])
        plt.plot(map_vec[i * 9:(i + 1) * 9, 6], map_vec[i * 9:(i + 1) * 9, 7])

    rows = np.size(traj, 0)
    print(rows)
    for i in range(rows):
        plt.annotate('', xy=(traj[:, 2][i], traj[:, 3][i]), xytext=(traj[:, 0][i], traj[:, 1][i]),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    rows = np.size(pred, 0)
    for i in range(rows - 1):
        length = np.sqrt(np.sum(np.square(pred[i + 1] - pred[i])))
        plt.arrow(pred[i][0], pred[i][1], pred[i + 1][0] - pred[i][0], pred[i + 1][1] - pred[i][1],
                  length_includes_head=True,  # 增加的长度包含箭头部分
                  head_width=length * 0.125,
                  head_length=length * 0.25,
                  width=length * 0.03,
                  fc='r',
                  ec='b')
    plt.axis('equal')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


def main():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cfg = dict(device=device, last_observe=20, batch_size=1, predict_step=29,
               data_locate="/home/pinkman/PycharmProjects/VectorNet/data_process/trajectory/test",
               save_path="./model_ckpt/inference/",
               model_path="//model/model_ckpt/model_final.pth")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    print()

    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])

    argo_dst = CitysimDataset()
    val_loader = DataLoader(dataset=argo_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    model = VectorNet(traj_features=4, map_features=4, cfg=cfg)
    model.to(device)

    # load from checkpoint
    # checkpoint = torch.load("./model_ckpt2/model_epoch10.pth")
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    # model.load_state_dict(checkpoint['model_state_dict'])

    # load from model_final
    # model.load_state_dict(torch.load(cfg['model_path']))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cfg['model_path']).items()})
    model.eval()  # Sets training as false.

    start_time = time.strftime('%Y-%m-%d %X', time.localtime(time.time()))
    inference(model, cfg, val_loader)
    end_time = time.strftime('%Y-%m-%d %X', time.localtime(time.time()))
    print('start time -> ' + start_time)
    print('end time -> ' + end_time)


def inference(model, cfg, val_loader):
    device = cfg['device']
    result, label, center_xy, rotate_matrix = [], [], [], []
    file_path = cfg['save_path'] + "inference.txt"
    file_handler = open(file_path, mode='w')
    pbar = tqdm(total=len(os.listdir(cfg['data_locate'])) // 2 * 2)
    pbar.set_description("Calculate Average Displacement Loss on Test Set")
    with torch.no_grad():
        for i, (traj_batch, map_batch) in enumerate(val_loader):
            traj_batch = traj_batch.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            single_result, single_label = model(traj_batch, map_batch)
            single_result, single_label = list(single_result.values()), list(single_label.values())
            single_result = torch.tensor([item.cpu().detach().numpy() for item in single_result]).cuda()
            single_label = torch.tensor([item.cpu().detach().numpy() for item in single_label]).cuda()
            result.append(single_result)
            label.append(single_label)
            a, b, map_range = val_loader.dataset.get_info(i)
            center_xy.append(a)
            rotate_matrix.append(b)
            pbar.update(2)
            # if i==6:
            #     print(single_result, single_label, traj_batch)
            # show_result(traj_batch, map_batch, single_result)
            # print(result)
            # print(label)
            # break
        pbar.close()
        print('length of result : ' + str(len(result)))
        print('length of label : ' + str(len(label)))
        # print(len(center_xy), len(rotate_matrix))
        predictions, loss = evaluate(result, label, center_xy, rotate_matrix, map_range)
        # for (k, v) in predictions.items():
        #     file_handler.write("%06d: " % int(k))
        #     file_handler.writelines("[%.2f, %.2f], " % (i[0], i[1]) for i in v.tolist())
        #     file_handler.write("\n")
    print("-------------------TEST RESULT----------------------")
    print(f'ADE(3s): ', loss)


def evaluate(predictions, labels, a, b, map_range):
    loss_list = []
    pred_coordinate = dict()
    for key in range(len(predictions)):
        max_coordinate = map_range['max']
        min_coordinate = map_range['min']
        rotate_matrix = b[key]
        center_xy = a[key]
        tmp_prediction = predictions[key].cpu().numpy() * (np.asarray(max_coordinate) - np.asarray(min_coordinate)) / 10
        tmp_label = labels[key].cpu().numpy() * (np.asarray(max_coordinate) - np.asarray(min_coordinate)) / 10
        tmp_prediction = tmp_prediction.dot(rotate_matrix)
        tmp_label = tmp_label.dot(rotate_matrix)
        pred_coordinate.update({key: tmp_prediction + center_xy})
        if key==6:
            print(tmp_prediction, tmp_label)
        loss_list.append(np.mean(np.sqrt(np.sum(np.square(tmp_prediction - tmp_label), axis=1))))
    # print("loss list", loss_list)
    loss = np.mean(np.array(loss_list)) / len(loss_list)
    loss1 = np.mean(np.array(loss_list))
    print("loss1    ", loss1)
    return pred_coordinate, loss


if __name__ == "__main__":
    main()