# speed, deltaD, direction, lane connection, heading
import math
import os

import pandas as pd
import numpy as np
import loguru

loguru.logger.add(sink="trajectory/datasetLog.log", level="DEBUG", rotation="10 MB",
                  retention="30 days", enqueue=True)
datasetLog = loguru.logger
datasetLog.info(f'Starting spilt data')
pd.set_option('display.max_columns', 6)
# pd.set_option('display.max_rows',None)

trajectory_frame:int=50
history_frame:int=20
future_frame:int=30
Hz:int=30
step:int=3
senario:int=0


def set_agent_type1(df):
    car_ids = df.carId.unique()
    complete_cars = []
    # 查找含有所有帧数据的车辆
    for car_id in car_ids:
        car_df = df[df.carId == car_id]
        #这不对 需要踢掉补0的数据
        if len(car_df) == len(df.frameNum.unique()):
            complete_cars.append(car_id)
    if len(complete_cars) > 0:
        ego_car_id = np.random.choice(complete_cars)
        df['agentType'] = 'envir'
        # 对于ego车辆，将其agentType设置为'ego'
        df.loc[df.carId == ego_car_id, 'agentType'] = 'ego'
        datasetLog.info(f'completed_cars{complete_cars}')
        return df
    # 如果没有车辆有完整的轨迹，则选择轨迹最长的车辆
    if len(complete_cars) == 0:
        longest_traj_id = max(car_ids, key=lambda cid: len(df[df.carId == cid]))
        datasetLog.info(f'longest{longest_traj_id}')
    df['agentType'] = 'envir'
    # 对于ego车辆，将其agentType设置为'ego'
    df.loc[df.carId == longest_traj_id, 'agentType'] = 'ego'
    return df

def set_agent_type2(df, ego_car_id):
    df['agentType'] = 'envir'
    df.loc[df.carId == ego_car_id, 'agentType'] = 'ego'  # For the ego vehicle
    return df


# 对于每一辆车，处理前20帧和后30帧数据
def process_car_padding(df, carId):
    car_df = df[df.carId == carId]
    frame_min = int(df.frameNum.min())
    frame_max = int(df.frameNum.max())

    # 如果后30帧（文件中的后30帧）缺失，则删除此车辆所有数据
    if len(car_df[car_df.frameNum >= frame_max - future_frame+1]) < future_frame:
        return None

    # 如果前20帧（文件中的前20帧）缺失，则填充0
    missing_frames = set(range(frame_min, frame_min + history_frame)) - set(car_df.frameNum)
    for i in missing_frames:
        filler = pd.DataFrame({col: [0 if col != 'carId' else carId] for col in car_df.columns})
        filler['frameNum'] = i
        car_df = car_df.append(filler)

    return car_df.sort_values(by='frameNum')

def process_car_deleteAll(df, carId):
    car_df = df[df.carId == carId]
    valid_frame_count = len(car_df)
    if valid_frame_count < trajectory_frame:  # If the car does not appear in all 50 frames
        return None
    else:
        return car_df.sort_values(by='frameNum')


def set_next_laneId(df):
    next_laneId = df['laneId'].values[-1]
    if next_laneId is None:
        print(senario)
    df['next_laneId'] = next_laneId
    return df




trajectory_dir = "//logsim/processed_trajectory_test"
#trajectory1 = pd.read_csv("/home/pinkman/University @ Alafaya (Signalized Intersection)/Trajectories/University@Alafaya-01.csv")
trajectory2 = pd.read_csv("/home/pinkman/University @ Alafaya (Signalized Intersection)/Trajectories/University@Alafaya-02.csv")

for filename in sorted(os.listdir(trajectory_dir)):
    if filename.endswith(".csv"):
        file_path = os.path.join(trajectory_dir, filename)
        trajectory1 = pd.read_csv(file_path)
        # process trajectory data
    min_carId, max_carID = trajectory1['carId'].min(), trajectory1['carId'].max() #0 3301
    # min, max = trajectory2['carId'].min(), trajectory2['carId'].max() #2935  4485

    # delete stationary vehicle and delte lane is nan
    data1 = trajectory1[trajectory1['speed'] != 0]
    data1 = data1.dropna(subset=['lane_id'])
    #course means vehicle waypoint heading relative to the image coordinate X-axis, x right, y down
    data2 = pd.DataFrame(data=data1, columns=['frameNum', 'carId', 'LateralLanePosition', 'heading_course',
                                              'speed', 'head_x', 'head_y', 'laneId', 'lane_id'])
    # data2.to_csv(r'non-stationary.csv', index=None, encoding='utf-8_sig')


    # down sampled and add timestamp
    # carId  FrameNum  headXft  headYft  speed  course  laneId
    #    1      0        10       20     30     40       3
    #    ...
    # 根据frameNum的范围筛选数据
    data3 = data2[(data2.frameNum >= 2) & (data2.frameNum <= 8999)]
    # 创建一个新列 "car_frame", 将 frameNum 除以 3 保留整数结果
    data3 = data3[data3['frameNum'] % step == 2]
    # 创建新的timestamp列
    data3['timestamp'] = (data3['frameNum']+1) / Hz
    data3['frameNum'] = round(data3['frameNum'] / 3)
    if 'next_laneId' not in data3.columns:
        data3['next_laneId'] = np.nan
    # data3.to_csv(r'down_sampled.csv', index=None, encoding='utf-8_sig')

    # print(data2["frameNum"].value_counts(), data2["frameNum"].value_counts().index, len(data2["frameNum"].value_counts().index))
    min_frameNum, max_frameNum = data3['frameNum'].min(), data3['frameNum'].max()
    batch_size = (max_frameNum+1) / trajectory_frame

    df = data3
    split_indices_time1 = [(i * trajectory_frame + 1, (i + 1) * trajectory_frame) for i in range(int(max_frameNum // trajectory_frame ))]
    split_indices_time2 = [(i * trajectory_frame + 20, i * trajectory_frame + 69) for i in range(int((max_frameNum - 20) // trajectory_frame ))]

    # spilt trajectory as many senarios
    for start, end in split_indices_time1:
        df_slice = df[(df.frameNum >= start) & (df.frameNum <= end)].copy()
        # Process each vehicle's data
        car_ids = df_slice.carId.unique()
        slices = []
        for car_id in car_ids:
            slice = process_car_deleteAll(df_slice, car_id)
            if slice is not None:
                slice = set_next_laneId(slice)
                slices.append(slice)
        # Concatenate processed data

        if slices:
            df_slice = pd.concat(slices).sort_values(by='frameNum')

        # Set frameNum starting from 1 to 50
        df_slice['frameNum'] = df_slice['frameNum'].transform(lambda x: x - start + 1)

        # Find complete cars and generate a CSV file for each one
        complete_cars = [car_id for car_id in car_ids if len(df_slice[df_slice.carId == car_id]) == 50]
        for ego_car_id in complete_cars:
            df_copy = df_slice.copy()  # Copy the DataFrame to avoid affecting the next iteration
            df_ego = set_agent_type2(df_copy, ego_car_id)
            # Output the csv file
            # df_ego.to_csv(f'dataset2/traj_{start}_to_{end}_{ego_car_id}.csv', index=False)
            df_ego.to_csv(f'trajectory/test/{senario}.csv', index=False)
            senario += 1
    datasetLog.info(f'senario_total{senario}')





