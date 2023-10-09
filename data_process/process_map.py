# -*- encoding: utf-8 -*-
"""
@File    :   process_map.py
@Contact :   jessapinkman@163.com
@License :   (C)Copyright 2023

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
2023/8/2 下午6:33  JessaPinkman  1.0         provide some APIs that process sumo map(.net.xml)
"""

import numpy as np
# import loguru
# import torch
# import tqdm
import xml.etree.ElementTree as ET
import os, sys
import sumolib
import traci

os.environ["SUMO_HOME"] = "/home/pinkman/sumo"
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


hallucinated_lane_bbox_length:int=20
centerline_length:int=10

class SumoMap:
    """
    This class provides the interface to our vector maps and rasterized maps. Exact lane boundaries
    are not provided, but can be hallucinated if one considers an average lane width.And also can
    some lane information like connetion、LaterLanePosition、speed etc.
    """

    def __init__(self, root: str) -> None:
        """Initialize the Argoverse Map."""
        self.root = root
        self.net = sumolib.net.readNet(self.root, withConnections=True, withInternal=True)
        self.edge_dict, self.lane_dict = self.get_edge_and_lane_info()
        self.get_lane_more_info()
        self.intersection_lanes = self.get_intersection_lanes()
        # self.closet_shape_index, self.closet_xy = self.get_closest_shape_point()
        self.map_range = self.get_map_range()


    def get_rotate_matrix(self, trajectory:list) -> list:
        """
        Computes a rotation matrix for the last frame of the history track,
        which when multiplied by a vector, rotates that vector by a specified angle
        :param trajectory: last frame of the history track
        :return:
                [[cos(alpha), -sin(alpha)],
                [sin(alpha),  cos(alpha)]]

        """
        x0, y0, x1, y1 = trajectory.flatten()
        vec1 = np.array([x1 - x0, y1 - y0])
        vec2 = np.array([0, 1])
        cosalpha = vec1.dot(vec2) / (np.sqrt(vec1.dot(vec1)) * 1 + 1e-5)
        sinalpha = np.sqrt(1 - cosalpha * cosalpha)
        if x1 - x0 < 0:
            sinalpha = -sinalpha
        rotate_matrix = np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
        return rotate_matrix


    def normalize_coordinate(self, array):
        """
        normalize coordinate according the map range
        :param array: map coordinate
        :return:
        normalized map coordinate
        """
        max_coordinate = np.asarray(self.map_range['max'])
        min_coordinate = np.asarray(self.map_range['min'])
        array = (100.*(array.reshape(-1, 2)) / (max_coordinate - min_coordinate)).reshape(-1,4)
        return array


    def get_distance_to_junction(self, lane_id: str, xy: list):
        distance = 0
        if lane_id in self.intersection_lanes:
            sys.exit("vehicle already in the real junction")
            return 0
        else:
            is_real_junction = False
        distance = self.distance(xy[0], xy[1], self.lane_dict[lane_id]['shape'][-1][0], self.lane_dict[lane_id]['shape'][-1][1])
        next_lane_id = self.lane_dict[lane_id]['connects'][1]
        print(111, next_lane_id)
        if next_lane_id in self.intersection_lanes:
            is_real_junction = True
        else:
            is_real_junction = False
        while not is_real_junction:
            distance += self.lane_dict[next_lane_id]['length']
            print(next_lane_id, self.lane_dict[next_lane_id])
            next_lane_id = self.lane_dict[next_lane_id]['connects'][1]
            if next_lane_id in self.intersection_lanes:
                is_real_junction = True
            else:
                is_real_junction = False

        return distance


    def get_closest_shape_point(self, lane_id: str, xy: list) -> tuple[int, list]:
        """
        Computes the Euclidean distance from a given coordinate point to each shape point and returns
        the index and coordinates of the point with the shortest distance.
        :param lane_id: lane id
        :param xy: [x, y],  this param usually given by vehicle's current vehicle
        :return:
        index of the current lane shape list and coordinates in the given lane_id
        eg: 28_0, [347, 781] -> 9 [347.32, 781.89]
        """
        # get shape list of lane
        # try:
        #     shape = self.lane_dict[lane_id]['shape']
        # except KeyError:
        #     print(lane_id)
        shape = self.lane_dict[lane_id]['shape']
        #  Calculate the Euclidean distance between the given xy coordinates and each coordinate in the shape
        distances = [(i, ((s[0] - xy[0]) ** 2 + (s[1] - xy[1]) ** 2) ** 0.5) for i, s in enumerate(shape)]
        #  Find the coordinate with the shortest distance, return its index and coordinate value
        min_distance = min(distances, key=lambda x: x[1])
        return min_distance[0], shape[min_distance[0]]


    def get_additional_points(self, target_lane_id:str, is_forward:bool, required_points_count:int) -> tuple[list, list]:
        """
        You can get information about the lane in front of or behind current lane.
        Expand the point set of the central lane line through the connection relationship between roads at a given point
        used by def get_lane_centerline
        :param target_lane_id: current lane
        :param is_forward:  Whether or not you need to get the predecessor of target lane，true means predecessor, false means successor
        :param required_points_count: How many more points do I need to fill the centerline with?
        :return:
        target lane shape and Arguments for the next call to this def
        eg:
        [[382.75, 782.39], [447.65, 785.55]]  ['-13#0_0', False, 4]
        """
        target_lane_shape = self.lane_dict[target_lane_id]['shape']
        if is_forward:  #  Get the point at the end of the shape
            if len(target_lane_shape) >= required_points_count:
                return target_lane_shape[-required_points_count:], None
            else:
                return target_lane_shape, [self.lane_dict[target_lane_id]['connects'][0], is_forward,
                                           required_points_count - len(target_lane_shape)]
        else:  # Get the point at the start of the shape
            if len(target_lane_shape) >= required_points_count:
                return target_lane_shape[:required_points_count], None
            else:
                return target_lane_shape, [self.lane_dict[target_lane_id]['connects'][1], is_forward,
                                           required_points_count - len(target_lane_shape)]


    def get_lane_centerline(self, lane_id:str, closest_point_index:int) -> list:
        """
        Given the vehicle's closest index of the current lane in which it is located,
        take 3 coordinate points forward, then take 6 coordinate points backward,
        and if there are not enough 10 coordinate points in that lane, then continue to
        fill in the lanes like the ones that have a connection between the front and the back
    :param lane_id: the vehicle's current lane id
    :param closest_point_index: closest point index of the current shape
    :return:
    List containing 10 lane centerline coordinates
    eg:
    [[380.26, 782.35], [382.75, 782.39], [447.65, 785.55], [447.65, 785.55], [448.34, 785.57], [449.04, 785.59], [449.84, 785.62], [451.05, 785.68], [452.04, 785.76], [453.18, 785.88]]
    """
        lane_centerline = [[None, None] for _ in range(10)]
        lane_shape = self.lane_dict[lane_id]['shape']
        shape_length = len(lane_shape)


        forward_points_count = closest_point_index - 3 if closest_point_index >= 3 else  0
        # forward_indices = [i for i in range(forward_points_count, closest_point_index)]
        if closest_point_index - 4 <= 0:
            end = -1
        else:
            end = closest_point_index - 4
        forward_indices = [i for i in
                           range(closest_point_index - 1, end, -1)]

        #  Add these points to lane_ Centerline
        lane_centerline_now_index = 0
        if len(forward_indices) == 0:
            lane_centerline_now_index = 3
        else:
            for i, index in enumerate(forward_indices):
                lane_centerline[2 - i] = lane_shape[index]
                if i == len(forward_indices) - 1:
                    lane_centerline_now_index = 2 - i
        # print(lane_centerline, lane_centerline_now_index)

        if closest_point_index < 3:
            additional_points, next_call = self.get_additional_points(self.lane_dict[lane_id]['connects'][0], True,
                                                                 lane_centerline_now_index)

            # first
            while lane_centerline_now_index > 0:
                for i in range(len(additional_points)):
                    lane_centerline[lane_centerline_now_index - 1] = additional_points[len(additional_points)-1-i]
                    lane_centerline_now_index -= 1
                if next_call == None:
                    break
                else:
                    additional_points, next_call = self.get_additional_points(next_call[0], True,
                                                                         lane_centerline_now_index)

        lane_centerline[3] = lane_shape[closest_point_index]

        backward_points_count =  closest_point_index + 6 if shape_length - closest_point_index - 1 >= 6 else shape_length-1
        backward_indices = [i for i in range(closest_point_index + 1,  backward_points_count + 1)]
        # print(backward_indices, backward_points_count)

        lane_centerline_now_index = 0
        if len(backward_indices) == 0:
            lane_centerline_now_index = 3
        else:
            for i, index in enumerate(backward_indices):
                lane_centerline[4 + i] = lane_shape[index]
                if i == len(backward_indices) - 1:
                    lane_centerline_now_index = 4 + i

        # print(lane_centerline, lane_centerline_now_index)

        if shape_length - closest_point_index - 1 < 6:
            additional_points, next_call = self.get_additional_points(self.lane_dict[lane_id]['connects'][1], False,
                                                                 6 - (shape_length - closest_point_index - 1))
            # print("add:", additional_points, next_call)
            # first
            while lane_centerline_now_index < len(lane_centerline):
                for i in range(len(additional_points)):
                    lane_centerline[lane_centerline_now_index + 1] = additional_points[i]
                    lane_centerline_now_index += 1
                if next_call == None:
                    break
                else:
                    additional_points, next_call = self.get_additional_points(next_call[0], False, 9 - lane_centerline_now_index)

        return lane_centerline


    def build_hallucinated_lane_bbox(self, lane_centerline:list, lane_id:str) -> list:
        """
        build hallucinated lane bbox for centerline
    :param lane_centerline: Coordinates of the centerline of a driveway of length 10
    :param lane_id: lane id
    :return:
    centerline of hallucinated lane bbox which along the lane direction, which length is 20
    if direction is along the x-axis, The first coordinate in the hallucinated_lane_bbox list is in the upper left corner,
    and the 11th coordinate in the list is in the lower left corner, hallucinating lane boundaries along the x-axis, respectively;
    if direction is along the y-axis, The first coordinate in the hallucinated_lane_bbox list is in the lower left corner,
    and the 11th coordinate in the list is in the lower right corner, hallucinating lane boundaries along the y-axis, respectively
    """
        lane_width = self.lane_dict[lane_id]['width']
        lane_shape = self.lane_dict[lane_id]['shape']
        shape_length = len(lane_shape)
        half_lane_width = lane_width / 2
        hallucinated_lane_bbox  = [[None, None] for _ in range(hallucinated_lane_bbox_length)]

        # 0 for x and 1 for y, representing lane direction
        direction = None
        differnece_x = abs(lane_shape[0][0] - lane_shape[shape_length-1][0])
        differnece_y = abs(lane_shape[0][1] - lane_shape[shape_length-1][1])
        # print(lane_centerline, lane_id)
        if differnece_x > differnece_y:
            direction = 0
            for i in range(len(lane_centerline)):
                hallucinated_lane_bbox[i] = [lane_centerline[i][0], lane_centerline[i][1] + half_lane_width]
                hallucinated_lane_bbox[i + 10] = [lane_centerline[i][0], lane_centerline[i][1] - half_lane_width]
        else:
            direction = 1
            for i in range(len(lane_centerline)):
                hallucinated_lane_bbox[i] = [lane_centerline[i][0] - half_lane_width, lane_centerline[i][1]]
                hallucinated_lane_bbox[i + 10] = [lane_centerline[i][0] + half_lane_width, lane_centerline[i][1]]

        return hallucinated_lane_bbox


    def get_intersection_lanes(self) -> list:
        """
        Add all lane_ids of intersection edge to intersecton_list
        :return:
        intersecton_list []
        """
        intersecton_list = []

        for edge_id, (lane_ids, function) in self.edge_dict.items():
            #  Check if it is an intersection edge
            if function == 'internal':
                # Merge all lanes of the intersection edge_ Add ID to Intersecton_ List
                intersecton_list.extend(lane_ids)

        return intersecton_list


    def get_lane_more_info(self):
        """
        build on the foundation of lane_dict called by get_edge_and_lane_info, get more information of lane
        :return:
        connects contain the predecessor and successor of the lane,
        is_intersection is the bool to Determine whether an intersection,
        There are three values for the direction： l for left, r for right, s for straight,
        It is worth noting that the edge that is not an intersection in the map has no direction, and we assign it the value s
        {lane ID:  ':442_0_0'
        Info: {'width': 3.25, 'length': 18.01, 'shape': [[320.77, 777.59], [338.76, 778.57]], 'connects': ['29_2', '28_0'],
        'is_intersection': False, 'direction': 's', 'successor': ['29_2'], 'predessor': ['28_0']}
        """
        #  Parsing XML files
        tree = ET.parse(self.root)
        #  Obtain the root element of the XML file
        root = tree.getroot()

        #  Declare a dictionary to store the connection relationship of lane
        lane_connection_dict = {}
        #  Store lane direction relationships
        lane_direction_dict = {}

        #  Traverse all 'connection' elements
        for connection in root.iter('connection'):
            via_lane_id = connection.get('via')
            from_edge = connection.get('from')
            from_lane = connection.get('fromLane')
            to_edge = connection.get('to')
            to_lane = connection.get('toLane')
            lane_direction = connection.get('dir')

            #  Create from_ From Lane and to_ ToLane
            from_fromLane = f'{from_edge}_{from_lane}'
            to_toLane = f'{to_edge}_{to_lane}'

            #  Save this connection relationship to lane_ Connection_ Dict
            lane_connection_dict[via_lane_id] = [from_fromLane, to_toLane]
            lane_direction_dict[via_lane_id] = lane_direction
        # print(len(lane_connection_dict), len(self.lane_dict))

        #  Update lane_ Lane information in dict
        for lane_id, lane_info in self.lane_dict.items():
            if lane_id in lane_connection_dict:
                lane_info['connects'] = lane_connection_dict[lane_id]

            else:
                #  For those who are not in lane_ Connection_ We attempt to infer the connectivity of the lane that
                #  appears in dict through the connectivity relationships of other lanes
                connects = [None, None]
                for connect_lane_id, (from_fromLane, to_toLane) in lane_connection_dict.items():
                    # if from_fromLane.split('_')[0] == lane_id or to_toLane.split('_')[0] == lane_id:
                    #     connects.append(connect_lane_id)
                    if to_toLane == lane_id:
                        connects[0] = connect_lane_id
                    if from_fromLane == lane_id:
                        connects[1] = connect_lane_id
                lane_info['connects'] = connects

        for lane_id, lane_info in self.lane_dict.items():
            if lane_id in lane_direction_dict:
                lane_info['is_intersection'] = True
                lane_info['direction'] = lane_direction_dict[lane_id]
            else:
                lane_info['is_intersection'] = False
                lane_info['direction'] = 's'

        # update competled connection information
        for lane_id, lane_info in self.lane_dict.items():
            # if lane_id[0] == ":":
            #     # lane_id = lane_id[1:]
            #     lane_id = lane_id[:-2]
            temp1 = self.net.getLane(lane_id).getIncoming()
            temp2 = self.net.getLane(lane_id).getOutgoingLanes()
            lane_successor = []
            lane_predessor = []
            for i in temp1:
                lane_successor.append(i.getID())
            for i in temp2:
                lane_predessor.append(i.getID())
            lane_info["successor"] = lane_successor
            lane_info["predessor"] = lane_predessor

        # whether or not true junction
        for edge_id, (lane_ids, function) in self.edge_dict.items():
            if function == 'internal':
                is_real_junction = False
                for id in lane_ids:
                    if self.lane_dict[id]['direction'] != 's':
                        is_real_junction = True
                        break
                if not is_real_junction:
                    self.edge_dict[edge_id][1] = None
                    for id in lane_ids:
                        self.lane_dict[id]['is_intersection'] = False


    def get_edge_and_lane_info(self):
        """
        get some basic edge and lane information, eg: shape、length、width of lane
        :return:  shape、length、width of lane, funtion and shape of edge,
        eg: edge_dict{edge_id:{shape, function}},  lane_dict{lane_id:{length, width, shape}}
        """
        # 解析xml文件
        tree = ET.parse(self.root)
        # get root element
        root = tree.getroot()

        edge_dict = {}
        lane_dict = {}

        for edge in root.iter('edge'):
            edge_id = edge.get('id')
            edge_function = edge.get('function')
            lanes = []
            for lane in edge.iter('lane'):
                lane_id = lane.get('id')
                lane_width = float(lane.get('width'))
                lane_length = float(lane.get('length'))
                #  Convert shape to a 2D list
                shape_xy = [list(map(float, coordinate.split(','))) for coordinate in lane.get('shape').split(' ')]

                lane_dict[lane_id] = {
                    'width': lane_width,
                    'length': lane_length,
                    'shape': shape_xy
                }
                lanes.append(lane_id)

            edge_dict[edge_id] = [lanes, edge_function]

        return edge_dict, lane_dict


    def get_left_right_lane(self, lane_id:str) -> tuple[str, str]:
        """
        get neighbors lane, including left and right lanes
        :param lane_id:
        :return:
        left_lane_id, right_lane_id
        """
        left_lane_id = None
        right_lane_id = None
        last_num = int(lane_id[-1])

        left_lane_id = lane_id[:-1] + str(last_num + 1)
        right_lane_id = lane_id[:-1] + str(last_num - 1)

        if left_lane_id not in self.lane_dict.keys():
            left_lane_id = None
        if right_lane_id not in self.lane_dict.keys():
            right_lane_id = None
        return left_lane_id, right_lane_id

    def generate_vector_map(self, lane_id:str, xy:list) -> tuple[np.array(list), dict]:
        """
        generate vector map for a hallucinated_lane_bbox
        :param lane_id: lane id
        :param xy: the current position of the vehicle
        :return:
        vector format map for a hallucinated_lane_bbox, shape[18,4]
        vector attribute: turn_direction、is_intersection、lane_id
        """
        polyline = []
        attribute = {}

        shape_index, shape_xy = self.get_closest_shape_point(lane_id, xy)
        centerline = self.get_lane_centerline(lane_id, shape_index)
        hallucinated_lane_bbox = self.build_hallucinated_lane_bbox(lane_centerline=centerline, lane_id=lane_id)

        # left boundray
        positive_pts = hallucinated_lane_bbox[:centerline_length]
        negative_pts = hallucinated_lane_bbox[centerline_length: ]

        for i in range(centerline_length - 1):
            v1 = np.array([positive_pts[i], positive_pts[i + 1]])
            v2 = np.array([negative_pts[i], negative_pts[i + 1]])
            polyline.append(v1)
            polyline.append(v2)

        # l for 0, r for 1, s for 2
        if self.lane_dict[lane_id]['direction'] == 'l':
            attribute['turn_direction'] = 0
        if self.lane_dict[lane_id]['direction'] == 'r':
            attribute['turn_direction'] = 1
        if self.lane_dict[lane_id]['direction'] == 's':
            attribute['turn_direction'] = 2

        attribute['lane_id'] = lane_id
        if self.lane_dict[lane_id]['is_intersection']:
            attribute['is_intersection'] = 1
        else:
            attribute['is_intersection'] = 0

        return polyline, attribute

    def get_lane_curve(self, lane_id:str, polynomial:int=5) -> np.poly1d:
        """
        Fit a polynomial curve based on the shape list of lane
        :param lane_id: lane id
        :param polynomial: the number of polynomial
        :return:
        polynomial_function
        7             6             5            4          3
        1.735e-11 x - 1.253e-08 x - 1.186e-06 x + 0.001325 x + 0.5284 x
          2
        - 65.89 x - 1.164e+05 x + 2.436e+07
        """
        centerline = self.lane_dict[lane_id]['shape']
        x = []
        y = []
        for xy in centerline:
            x.append(xy[0])
            y.append(xy[1])
        #  Fit with a 5th degree polynomial, with output coefficients ranging from high to 0
        z = np.polyfit(x, y, polynomial)
        #  Using Degree Composite Polynomials
        polynomial_function = np.poly1d(z)
        return polynomial_function

    def distance(self, x, y, x0, y0):
        """
        Return distance between point
        P[x0,y0] and a curve (x,y)
        """
        d_x = x - x0
        d_y = y - y0
        dis = np.sqrt(d_x ** 2 + d_y ** 2)
        return dis


    def get_lateral_centerline_position(self, xy:list, lane_id:str, percision:int=5) -> int:
        """
        get lateral centerline position of the current vehicle position,
        Compute minimum/a distance/s between a point P[x0,y0] and a curve (x,y) rounded at `precision`.
        :param xy: current vehicle position
        :param lane_id:  lane id
        :param percision:  Precision determines the number of decimal places after the decimal point
        :return:
        Returns min indexes and distances array.
        """

        polynomial_function = self.get_lane_curve(lane_id)
        # need change to map range, and there are some bugs
        x = np.linspace(-1000, 1000, 1000)
        y = polynomial_function(x)
        d = self.distance(x, y, xy[0], xy[1])
        min_distance = np.round(d, percision)
        # find the minima
        glob_min_idxs = np.argwhere(d == np.min(d)).ravel()
        return glob_min_idxs, min_distance

    def get_map_range(self):
        """
        get map range
        :return:
        return xmin,ymin,xmax,ymax network coordinates
        """
        # parse the net
        temp = self.net.getBoundary()
        map_range = {}
        map_range["min"] = [temp[0], temp[1]]
        map_range["max"] = [temp[2], temp[3]]
        return map_range


    def save_data(self):
        return

    def get_lane_ids_in_xy_bbox(self):
        return

    def add_attribute_for_vector(self):
        return



    def build_centerline_index(self):
        return


if __name__ == "__main__":
    # 替换为你的.net.xml文件
    file_path = '//data_process/hd_maps/University@Alafaya.net.xml'
    map = SumoMap(file_path)
    print(map.edge_dict)
    print(map.get_distance_to_junction("29_5", [305,788]))

    # for edge_id, info in list(map.edge_dct.items())[:5]:
    #     print(f'Edge ID: {edge_id}, Info: {info}')
    #
    # for lane_id, info in list(map.lane_dict.items()):
    #     if lane_id == '28_1' or lane_id == ':105_30_0' or lane_id == ':442_0_1': print(f'lane ID: {lane_id}, Info: {info}')
    # 105_12_0  381.51, 807.39
    # [[447.12, 811.45], [382.55, 807.4], [381.51, 807.39], [380.51, 807.37], [379.51, 807.36], [378.52, 807.34], [377.52, 807.33], [None, None], [None, None], [None, None]] :105_12_0

    # xy = [381.51, 807.39]
    # lane_id = ':105_12_0'
    # print(map.lane_dict[lane_id])
    # index, closet_xy = map.get_closest_shape_point(lane_id, xy)
    # print(index, closet_xy) # 9 [347.32, 781.89]
    # lane_centerline = map.get_lane_centerline(lane_id, index)
    # print("centerline", lane_centerline)
    # hallucinated_lane_bbox = map.build_hallucinated_lane_bbox(lane_centerline, lane_id)
    # print("hallucinated_lane_bbox", hallucinated_lane_bbox)
    # print(map.generate_vector_map(lane_id, xy))
    edge = "29"
    lane = ":442_0_3"
    # print(map.net.getEdge(edge).getToNode().getID(), map.net.getEdge(edge).getFromNode().getID())
    # print(map.net.getLane(lane).getOutgoing())
    # print(map.net.getLane(lane).getOutgoingLanes())
    # list1 = map.net.getLane(lane).getOutgoingLanes()
    # list2 = map.net.getLane(lane).getIncoming()
    # for i in list1:
    #     print(i)
    # for i in list2:
    #     print(i.getID())




