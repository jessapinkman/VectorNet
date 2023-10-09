import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
from model.VectorNet import VectorNet
import logging
import warnings
warnings.filterwarnings('ignore')
import pprint
import time
import numpy as np
from CitysimDataset import CitysimDataset

# writer_loss = SummaryWriter("/home/pinkman/PycharmProjects/VectorNet/model/runs")


def init_logger(cfg):
    """
    init logger class
    :param cfg:
    :return: nstantiating logger

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file = cfg['log_file']
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger



use_gpu:bool=True
run_parallel:bool=True
device_ids:list=[]

# network args
args = {}
args['traj_feature'] = 4
args['map_feature'] = 4
args['learning_rate'] = 0.001
args['learning_rate_decay'] = 0.3


if use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    if torch.cuda.device_count() <= 1:
        run_parallel = False

else:
    device = torch.device('cpu')
    run_parallel = False


cfg = dict(device=device, learning_rate=args['learning_rate'], learning_rate_decay=args['learning_rate_decay'],
           last_observe=20, epochs=12, print_every=100, save_every=5, batch_size=1,
           data_locate="/home/pinkman/PycharmProjects/VectorNet/data_process/trajectory/train",
           save_path="./model_ckpt/",  # /workspace/argoverse-api/train/data
           log_file="runs/log.txt", tensorboard_path="runs/train_visualization")

if not os.path.isdir(cfg['save_path']):
    os.mkdir(cfg['save_path'])
writer = SummaryWriter(cfg['tensorboard_path'])
train_log = init_logger(cfg)

print(cfg)

train_log.info(f'run_parallel{run_parallel}'
            f'cfg{cfg}')

train_dataset = CitysimDataset()
train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

# init model
model = VectorNet(traj_features=args['traj_feature'], map_features=args['map_feature'], cfg=cfg)
model.to(device)
# set the module in training model
model.train()
if run_parallel:
    model = nn.DataParallel(module=model, device_ids=device_ids)

# init optimizer
optimizer = optim.Adadelta(model.parameters(), rho=0.9)
# Gradually reduce LR to achieve global min faster and reduce local oscillations
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=args['learning_rate_decay'])  # MultiStepLR TODO

train_log.info(f'Start Training...')

# start training
start_time = time.strftime('%Y-%m-%d %X', time.localtime(time.time()))

loss_dict = {}
for i in range(0, 4000, 100):
    loss_dict[i] = 0
print(loss_dict)

for e in range(cfg['print_every']):
    for i, (traj_batch, map_batch) in enumerate(train_dataloader):
        # print("batch:", i, traj_batch.shape, len(map_batch), map_batch[0].shape, type(map_batch))
        traj_batch = traj_batch.to(cfg['device'], torch.float)
        # count loss
        loss = model(traj_batch, map_batch).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % cfg['print_every'] == 0:
            train_log.info('Epoch %d/%d: Iteration %d, loss = %f' % (e + 1, cfg['epochs'], i, loss.item()))
            writer.add_scalar('training_loss', loss.item(), e)
            if loss > 1:
                loss_dict[i] += 1
    # print(loss_dict)
    scheduler.step()

    if (e + 1) % cfg['save_every'] == 0:
        file_path = cfg['save_path'] + "model_epoch" + str(e + 1) + ".pth"
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, file_path)
        train_log.info(f'Save model {file_path}')

torch.save(model.state_dict(), cfg['save_path']+"model_final.pth")
train_log.info("Save final model "+cfg['save_path']+"model_final.pth")
train_log.info("Finish Training")
end_time = time.strftime('%Y-%m-%d %X',time.localtime(time.time()))
print('start time -> ' + start_time)
print('end time -> ' + end_time)
