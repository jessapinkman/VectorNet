import numpy as np
import math

freewayB_arc = 14
freewayC_arc = 13.5
University_arc = 88
McCulloch_arc = 180.5

scale_matrix = np.diag([0.3048, 0.3048, 1])

rotation_matrix_freewayB = np.mat([[math.cos((freewayB_arc * math.pi / 180)), -math.sin((freewayB_arc * math.pi / 180)), 0],
                                  [math.sin((freewayB_arc * math.pi / 180)), math.cos((freewayB_arc * math.pi / 180)), 0],
                                  [0, 0, 1.0]])

rotation_matrix_freewayC = np.mat([[math.cos((freewayC_arc * math.pi / 180)), -math.sin((freewayC_arc * math.pi / 180)), 0],
                                  [math.sin((freewayC_arc * math.pi / 180)), math.cos((freewayC_arc * math.pi / 180)), 0],
                                  [0, 0, 1.0]])

rotation_matrix_McCulloch = np.mat([[math.cos((McCulloch_arc * math.pi / 180)), -math.sin((McCulloch_arc * math.pi / 180)), 0],
                                  [math.sin((McCulloch_arc * math.pi / 180)), math.cos((McCulloch_arc * math.pi / 180)), 0],
                                  [0, 0, 1.0]])

rotation_matrix_University = np.mat([[math.cos((University_arc * math.pi / 180)), -math.sin((University_arc * math.pi / 180)), 0],
                                     [math.sin((University_arc * math.pi / 180)), math.cos((University_arc * math.pi / 180)), 0],
                                     [0, 0, 1.0]])

parallel_matrix_freewayC = np.mat([[1, 0, 1031.49846],
                                   [0, 1, 260.317925],
                                   [0, 0, 1]])

parallel_matrix_freewayB = np.mat([[1, 0, 659.90646],
                                   [0, 1, 174.815925],
                                   [0, 0, 1]])
parallel_matrix_McCulloch = np.mat([[1, 0, 1134.50646],
                                    [0, 1, 435.415925],
                                    [0, 0, 1]])
parallel_matrix_University = np.mat([[1, 0, 366.557646],
                                    [0, 1, 714.915925],
                                    [0, 0, 1]])


transform_matrix_freewayC = parallel_matrix_freewayC @ scale_matrix @ rotation_matrix_freewayC
transform_matrix_freewayB = parallel_matrix_freewayB @ scale_matrix @ rotation_matrix_freewayB
transform_matrix_McCulloch = parallel_matrix_McCulloch @ scale_matrix @ rotation_matrix_McCulloch
transform_matrix_University = parallel_matrix_University @ scale_matrix @ rotation_matrix_University

transformation_matrix = {
    'FreewayC': transform_matrix_freewayC,
    'FreewayB': transform_matrix_freewayB,
    'McCulloch': transform_matrix_McCulloch,
    'University': transform_matrix_University}

