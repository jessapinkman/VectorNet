import os
import traci
import math
import pandas as pd
import numpy as np
import coordinate_transformation_info
from log_citysim_csv import LogCitySimCsv

transformation_matrix = coordinate_transformation_info.transformation_matrix
# filepath = 'Examples/example/University@Alafaya_data/trajectories/University@Alafaya-01.csv'
# 6 12
filepath = "/home/pinkman/University @ Alafaya (Signalized Intersection)/Trajectories/University@Alafaya-12.csv"
keys = 'carId'

drop_columns_after = ['headXft', 'headYft', 'angle_1',
                     'headXft_smoothed_1', 'headXft_smoothed_2',
                     'headYft_smoothed_1', 'headYft_smoothed_2',
                     'group_id_r']

drop_columns = [
    'carCenterX', 'carCenterY', 'headX', 'headY', 'tailX', 'tailY',
    'boundingBox1Y', 'boundingBox2Y', 'boundingBox3Y', 'boundingBox4Y',
    'boundingBox1X', 'boundingBox2X', 'boundingBox3X', 'boundingBox4X',
    'carCenterXft', 'carCenterYft', 'heading', 'carCenterLat', 'carCenterLon',
    'tailXft', 'tailYft', 'headLat', 'headLon', 'tailLat', 'tailLon',
    'boundingBox1Lat', 'boundingBox1Lon', 'boundingBox2Lat', 'boundingBox2Lon',
    'boundingBox3Lat', 'boundingBox3Lon', 'boundingBox4Lat', 'boundingBox4Lon']

bounding_box_ft = ['boundingBox1Yft', 'boundingBox2Yft', 'boundingBox3Yft', 'boundingBox4Yft',
                   'boundingBox1Xft', 'boundingBox2Xft', 'boundingBox3Xft', 'boundingBox4Xft']
position_keys = ['headXft_smoothed_2', 'headYft_smoothed_2']
log = LogCitySimCsv(filepath, keys, position_keys, drop_columns, bounding_box_ft, transformation_matrix['University'])
# if 'LateralLanePosition' not in log.trajectory.columns:
#     log.trajectory['LateralLanePosition'] = np.nan

sumoBinary = "/home/pinkman/sumo/bin/sumo-gui"
cfg_dir = os.path.abspath(os.path.join(os.getcwd(),
                                       "University@Alafaya.sumocfg"))
if os.path.exists(cfg_dir):
    sumoCmd = [sumoBinary, "-c", cfg_dir,
               "--delay", "1.0", '--step-length', '0.033']
else:
    print(f"{cfg_dir} does not exists")

print(log.trajectory.columns)
print(log.trajectory.frameNum.max())

# 为什么要求车辆不在路口附近
def calculate_heading_course(veh_id):
    """
    Calculate each vehicle's heading_course relative to the road tangent direction.
    positive return: The vehicle's head is to the left of the Road tangent direction.
    negative return: The vehicle's head is to the right of the Road tangent direction.
    Notion!:This function is only for vehicles which are not close to junctions.
    :param veh_id: vehID
    :return:heading_course
    """
    lane_id = traci.vehicle.getLaneID(veh_id)
    lane_length = traci.lane.getLength(lane_id)
    # change code
    if not is_close_to_junction(veh_id):
        edge_id = lane_id.split('_')[0]
    else:
        temp = lane_id.split('_')
        edge_id = '_'.join(temp[:-1])
    # if veh_id == '2' and is_close_to_junction(veh_id):
    #     print(f'is_close_junction???{is_close_to_junction(veh_id)}, edge_id{edge_id}')
    pos1 = traci.vehicle.getLanePosition(veh_id)
    pos2 = min(pos1 + 5, lane_length)
    laneIndex = traci.vehicle.getLaneIndex(veh_id)
    veh_angle = traci.vehicle.getAngle(veh_id)
    x1, y1 = traci.simulation.convert2D(edgeID=edge_id, pos=pos1, laneIndex=laneIndex)
    x2, y2 = traci.simulation.convert2D(edgeID=edge_id, pos=pos2, laneIndex=laneIndex)
    dx, dy = x2 - x1, y2 - y1
    derivative = abs(dy / (dx + 1e-8))
    lane_angle_degree = math.atan(derivative) * (180 / math.pi)
    if dx > 0 and dy > 0:
        lane_angle_degree = 90 - lane_angle_degree
    elif dx > 0 > dy:
        lane_angle_degree = 90 + lane_angle_degree
    elif dx < 0 and dy < 0:
        lane_angle_degree = 270 - lane_angle_degree
    elif dx < 0 < dy:
        lane_angle_degree = 180 + lane_angle_degree
    if lane_angle_degree == 0 and veh_angle > 270:
        lane_angle_degree = 360
    heading_course = (lane_angle_degree - veh_angle) * math.pi / 180
    return heading_course


def is_close_to_junction(veh_id):
    """
    Check whether the vehicle is close to a junction or not.
    :param veh_id: vehID.
    :return: True or False.
    """
    lane_id = traci.vehicle.getLaneID(veh_id)
    edge_id = lane_id.split('_')[0]
    if edge_id[0] == ':':
        return True
    return False


def simulationStep():
    traci.start(sumoCmd)
    simulation_step_length = traci.simulation.getDeltaT()
    step = 0
    last_step_vehs = set()
    removed_vehs = set()
    # while step < 100:
    while step < log.trajectory.frameNum.max() + 1:

        simulation_step = simulation_step_length * step
        veh_heading_course = {}
        veh_lateral_position = {}

        if simulation_step in log.trajectory.index.levels[0]:
            trajectories = log.trajectory.xs(simulation_step, level=0)
            curr_vehs = set(trajectories.index.tolist())
            new_vehs = curr_vehs - last_step_vehs
            removed_vehs = last_step_vehs - curr_vehs

        # Add vehicles.
        for veh_id in new_vehs:
            traci.vehicle.add(veh_id, '', typeID='veh_passenger')
            last_step_vehs.add(veh_id)

            traci.vehicle.setLength(veh_id, trajectories.loc[veh_id]['v_length'])
            traci.vehicle.setWidth(veh_id, trajectories.loc[veh_id]['v_width'])

        # Move vehicles.
        for veh_id in curr_vehs:
            x, y = trajectories.loc[veh_id][['head_x', 'head_y']].to_list()
            traci.vehicle.moveToXY(veh_id, '', -1, x, y,
                                   angle=trajectories.loc[veh_id]['angle_2'],
                                   keepRoute=2, matchThreshold=3)



        # Delete vehicles.
        for veh_id in removed_vehs:
            # print("current_vehicles: ", curr_vehs)
            # print("last_step_vehicles: ", last_step_vehs)
            # print("removed_vehicles: ", removed_vehs)
            # print("veh_id: ", veh_id)
            # print("current_vehicles_alive: ", traci.vehicle.getIDList())
            # print("Simulation_step: ", simulation_step)
            traci.vehicle.remove(veh_id)
        last_step_vehs = curr_vehs

        traci.simulationStep()

        # Get lane_id and edge_id.
        for veh_id in curr_vehs:
            try:
                lane_id = traci.vehicle.getLaneID(veh_id)
                edge_id = lane_id.split('_')[0]
                trajectories.loc[veh_id, 'lane_id'] = lane_id
                trajectories.loc[veh_id, 'edge_id'] = edge_id
            except:
                trajectories.loc[veh_id, 'lane_id'] = None
                trajectories.loc[veh_id, 'edge_id'] = None

        # Calculate heading course of vehicles.
        for veh_id in curr_vehs:
            # if veh_id == '2' and is_close_to_junction(veh_id):
            #     traci.vehicle.setColor(veh_id, color=(255, 0, 0, 255))
            #     print(f'calculate_heading_course(){calculate_heading_course(veh_id)}')

            try:
                veh_lateral_position[veh_id] = traci.vehicle.getLateralLanePosition(veh_id)
                trajectories.loc[veh_id, 'LateralLanePosition'] = veh_lateral_position[veh_id]
            except:
                trajectories.loc[veh_id, 'LateralLanePosition'] = None
            # change code
            try:
                veh_heading_course[veh_id] = calculate_heading_course(veh_id)
                trajectories.loc[veh_id, 'heading_course'] = veh_heading_course[veh_id]
            except:
                trajectories.loc[veh_id, 'heading_course'] = None
            # try:
            #     if not is_close_to_junction(veh_id):
            #         veh_heading_course[veh_id] = calculate_heading_course(veh_id)
            #         trajectories.loc[veh_id, 'heading_course'] = veh_heading_course[veh_id]
            #
            # except:
            #     trajectories.loc[veh_id, 'heading_course'] = None

        step += 1

    traci.close()
    return


simulationStep()

file_name = filepath.split('/')[-1]
log.trajectory.to_csv(file_name, index=False)

df = pd.read_csv(file_name).drop(columns=drop_columns_after)
df.rename(columns={'angle_2': 'angle'}, inplace=True)
df.to_csv('/home/pinkman/PycharmProjects/VectorNet/logsim/processed_trajectory_train/' + file_name, index=False)