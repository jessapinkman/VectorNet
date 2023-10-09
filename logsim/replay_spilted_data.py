# -*- encoding: utf-8 -*-
"""
@File    :   replay_spilted_data.py   
@Contact :   jessapinkman@163.com
@License :   (C)Copyright 2023
 
@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
2023/8/14 下午2:54   JessaPinkman  1.0         None
"""
import os
import traci
import math
import pandas as pd
import numpy as np


sumoBinary = "/home/pinkman/sumo/bin/sumo-gui"
cfg_dir = os.path.abspath(os.path.join(os.getcwd(),
                                       "University@Alafaya.sumocfg"))
if os.path.exists(cfg_dir):
    sumoCmd = [sumoBinary, "-c", cfg_dir,
               "--delay", "1000", '--step-length', '0.1']
else:
    print(f"{cfg_dir} does not exists")

carId = 3632

df = pd.read_csv("//data_process/trajectory/train/677.csv")
df = df[df['carId'] == carId]

frameNum = 1
carId= str(carId)
traci.start(sumoCmd)
while frameNum < 50:
    if frameNum == 1:
        traci.vehicle.add(vehID=carId, routeID='')
    temp = df[df['frameNum'] == frameNum]
    x = temp['head_x'].values
    y = temp['head_y'].values
    if carId in traci.vehicle.getIDList():
        traci.vehicle.moveToXY(vehID=carId, edgeID='', lane=-1, x=x, y=y, keepRoute=2, matchThreshold=3)
    frameNum += 1
    traci.simulationStep()

traci.close()