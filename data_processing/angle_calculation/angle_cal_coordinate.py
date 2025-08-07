# angle_calculation.py

import numpy as np
import pandas as pd


def define_coordinate_thorax(normal_vec_columns, df_o):
    x = []
    y = []
    z = []
    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3) #0.XP，1.T8，2.SN，3.C7 

        origin = points[2]
        midpoint1 = (points[0] + points[1]) / 2  # XP和T8的中点
        midpoint2 = (points[2] + points[3]) / 2  # SN和C7的中点

        # 计算y轴向量
        y_axis = midpoint2 - midpoint1  # 向量a-b，b为起点a为终点 从XP指向SN

        # 计算z轴向量
        plane_points1 = points[2] - midpoint1
        plane_points2 = points[3] - midpoint1
        z_axis = np.cross(plane_points1, plane_points2)  # z轴方向确定方法：叉乘右手螺旋法则

        # 计算x轴向量
        x_axis = np.cross(y_axis, z_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x.append(x_axis)
        y.append(y_axis)
        z.append(z_axis)

    return x, y, z

def define_coordinate_humerus(normal_vec_columns, df_o):
    x = []
    y = []
    z = []
    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3) #0.LE_L，1.ME_L，2.GH1_L，3.GH2_L

        origin = (points[2] + points[3]) / 2
        midpoint1 = (points[0] + points[1]) / 2  # LE和ME的中点
        midpoint2 = (points[2] + points[3]) / 2  # GH1和GH2的中点

        # 计算y轴向量
        y_axis = midpoint2 - midpoint1  # 向量a-b，b为起点a为终点 从LE指向GH

        # 计算z轴向量
        plane_points1 = points[2] - midpoint1
        plane_points2 = points[3] - midpoint1
        z_axis = np.cross(plane_points1, plane_points2)  # z轴方向确定方法：叉乘右手螺旋法则

        # 计算x轴向量
        x_axis = np.cross(y_axis, z_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x.append(x_axis)
        y.append(y_axis)
        z.append(z_axis)

    return x, y, z

#构建髋坐标系
def define_coordinate_hip(normal_vec_columns, df_o): # 0.HF_L, 1.HF_R, 2.HB_L, 3.HB_R
    x = []
    y = []
    z = []

    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3)  # 0.HF_L, 1.HF_R, 2.HB_L, 3.HB_R

        # 定义原点：四个点的几何中心（对角线交点）
        origin = np.mean(points, axis=0)  # (HF_L + HF_R + HB_L + HB_R) / 4

        left_midpoint = (points[0] + points[2]) / 2     # (HF_L + HB_L) / 2
        right_midpoint = (points[1] + points[3]) / 2    # (HF_R + HB_R) / 2

        # 计算Z轴（右向）：从HF_L到HF_R
        z_axis = right_midpoint - left_midpoint

        # 计算髋平面法向量（Y轴，向上）
        # 使用HF_L到HF_R和HF_L到HB_L的向量叉乘，定义髋平面法向量
        vec1 = points[1] - points[0]  # HF_R - HF_L
        vec2 = points[2] - points[0]  # HB_L - HF_L
        y_axis = np.cross(vec2, vec1)  # 髋平面法向量
        # 确保Y轴向上（在解剖位，Y轴应有正的垂直分量）
        # if y_axis[2] < 0:  # 假设z轴为全球上方向，检查法向量方向
        #     y_axis = -y_axis

        # 计算X轴（前向）：通过Y轴和Z轴叉乘
        x_axis = np.cross(y_axis, z_axis)

        # 归一化
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x.append(x_axis)
        y.append(y_axis)
        z.append(z_axis)


    return x, y, z


# #构建髋坐标系
# def define_coordinate_hip(normal_vec_columns, df_o): #0.SN, 1.C7, 2.T8, 3.SC_L, 4.SC_R, 5.AC_L, 6.AC_R 0.HF_L, 1.HF_R, 2.HB_L, 3.HB_R
#     x = []
#     y = []
#     z = []

#     for _, row in df_o.iterrows():
#         points = row[normal_vec_columns].values.astype(float).reshape(-1, 3) 
#         # 定义原点
#         origin = points[0] #SN
#         CLAV_L = (points[3]+ points[5]) / 2 #3.SC_L和5.AC_L的中点为左锁骨定位点
#         CLAV_R = (points[4]+ points[6]) / 2 #4.SC_R和6.AC_R的中点为右锁骨定位点

#         # 计算X轴（前向）
#         # 从C7-T8连线中点指向胸骨方向
#         spine_midpoint = (points[1]+ points[2]) / 2 #1.C7和2.T8
#         temp_y = points[0] - spine_midpoint
    
#         # 计算Z轴（右向）
#         # 从左锁骨指向右锁骨方向
#         temp_x = CLAV_R - CLAV_L
    
#         # 计算Y轴（向上）
#         # 通过Z轴和X轴叉积得到
#         z_axis = np.cross (temp_x, temp_y)
    
#         # 重新计算Y轴，确保正交性
#         y_axis = np.cross (z_axis, temp_x)
#         x_axis = temp_x
    
#         x_axis = x_axis / np.linalg.norm(x_axis)
#         y_axis = y_axis / np.linalg.norm(y_axis)
#         z_axis = z_axis / np.linalg.norm(z_axis)

#         x.append(x_axis)
#         y.append(y_axis)
#         z.append(z_axis)

#     return x, y, z


def define_coordinate_scapula(normal_vec_columns, df_o):
    x = []
    y = []
    z = []
    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3) #0.AA，1.AI，2.TS

        origin = points[0]

        # 计算z轴向量
        z_axis = points[0] - points[2]  # 向量a-b，b为起点a为终点 TS指向AA

        # 计算x轴向量
        x_axis = np.cross(points[1] - points[0], points[2] - points[0]) #（AA指向AI，AA指向TS）叉乘， 左侧肩胛骨X轴指向后侧，右侧肩胛骨X轴指向前侧

        # 计算y轴向量
        y_axis = np.cross(z_axis, x_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x.append(x_axis)
        y.append(y_axis)
        z.append(z_axis)
    return x, y, z

#构建锁骨坐标系

import numpy as np

def define_clavicle_coord(normal_vec_columns, df_o): 
    """
    使用胸廓4点 + 锁骨两点定义锁骨坐标系（右手系）
    """
    x = []
    y = []
    z = []
    for _, row in df_o.iterrows():
       points = row[normal_vec_columns].values.astype(float).reshape(-1, 3) #0.XP, 1.T8, 2.SN, 3.C7, 4.SC_L, 5.AC_L

       # 1. 构建胸廓纵轴 Yt（从下到上）
       midpoint1 = (points[0] + points[1]) / 2  # 胸廓下部中点, XP和T8
       midpoint2 = (points[2] + points[3]) / 2  # 胸廓上部中点, SN和C7
       y_axis = midpoint2 - midpoint1

       # 2. 构建锁骨 Z 轴（SC → AC，锁骨本体方向，向右）
       z_axis = points[5] + points[4] #AC - SC

       # 3. 构建 X 轴为 Yt × Zc（确保右手系）
       x_axis = np.cross(y_axis, z_axis)

       # 4. 重新求 Yc = Z × X
       y_axis = np.cross(z_axis, x_axis)

       x_axis = x_axis / np.linalg.norm(x_axis)
       y_axis = y_axis / np.linalg.norm(y_axis)
       z_axis = z_axis / np.linalg.norm(z_axis)

       x.append(x_axis)
       y.append(y_axis)
       z.append(z_axis)
    return x, y, z


def calculate_angle(vector_list1, vector_list2):
    angles_degrees = []
    for vector1, vector2 in zip(vector_list1, vector_list2):
        # print(vector1)
        # print(vector2)
        dot_product = np.dot(vector1, vector2)  # 计算两个向量的点积
        vector1 = np.linalg.norm(vector1)
        vector2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (vector1 * vector2)  # 使用点积公式计算两个向量之间夹角的余弦值
        angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 计算余弦值对应的弧度角
        angle_degrees = np.degrees(angle_radians)  # 将弧度转换为度

        angles_degrees.append(angle_degrees)  # 将计算出的角度添加到列表

    return angles_degrees


def calculate_angle_xyz(x1, y1, z1, x2, y2, z2):
   x_angles = calculate_angle(x1, x2) # 计算x轴角度   
   y_angles = calculate_angle(y1, y2) # 计算y轴角度
   z_angles = calculate_angle(z1, z2) # 计算z轴角度
   # print(x_angles)
   return x_angles, y_angles, z_angles

# def calculate_angle_yz(y1, z1, y2, z2):
#     y_angles = calculate_angle(y1, y2)
#     z_angles = calculate_angle(z1, z2)
#     # print(y_angles)
#     return y_angles, z_angles

def calculate_angle_elbow(normal_vec_columns, df_o):
    angles_degrees = []
    for _, row in df_o.iterrows():
        points = row[normal_vec_columns].values.astype(float).reshape(-1, 3)  #0.WX，1.WN，2.LE，3，ME，4.GH1，5.GH2

        midpoint1 = (points[0] + points[1]) / 2  # WX and WN midpoint
        midpoint2 = (points[2] + points[3]) / 2  # LE and ME midpoint
        midpoint3 = (points[4] + points[5]) / 2  # GH1 and GH2 midpoint

        vector1 = midpoint1 - midpoint2
        vector2 = midpoint3 - midpoint2

        dot_product = np.dot(vector1, vector2)  # 计算两个向量的点积
        vector1 = np.linalg.norm(vector1)
        vector2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (vector1 * vector2)  # 使用点积公式计算两个向量之间夹角的余弦值
        angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 计算余弦值对应的弧度角。np.clip 确保余弦值在有效范围内，避免计算反余弦时出现数值错误。
        angle_degrees = np.degrees(angle_radians)  # 将弧度转换为度
        angles_degrees.append(angle_degrees)  # 将计算出的角度添加到列表
    return angles_degrees

def calculate_all_angles(df_o):

    # 【0.胸廓】 建立胸廓坐标系
    normal_vec_columns_thorax = ['XP', 'XP.1', 'XP.2', 'T8', 'T8.1', 'T8.2', 'SN', 'SN.1', 'SN.2', 'C7', 'C7.1', 'C7.2']
    x_thorax, y_thorax, z_thorax = define_coordinate_thorax(normal_vec_columns_thorax, df_o)

    # 【1.左肩胛骨】 建立肩胛骨坐标系（左）
    normal_vec_columns_scapula_l = ['AA_L', 'AA_L.1', 'AA_L.2', 'AI_L', 'AI_L.1', 'AI_L.2', 'TS_L', 'TS_L.1', 'TS_L.2']
    x_scapula_l, y_scapula_l, z_scapula_l = define_coordinate_scapula(normal_vec_columns_scapula_l, df_o)

    # 【2.右肩胛骨】建立肩胛骨坐标系（右）
    normal_vec_columns_scapula_r = ['AA_R', 'AA_R.1', 'AA_R.2', 'AI_R', 'AI_R.1', 'AI_R.2', 'TS_R', 'TS_R.1', 'TS_R.2']
    x_scapula_r, y_scapula_r, z_scapula_r = define_coordinate_scapula(normal_vec_columns_scapula_r, df_o)

    # 【3.左肱骨】建立肱骨坐标系（左）
    normal_vec_columns_humerus_l = ['LE_L', 'LE_L.1', 'LE_L.2', 'ME_L', 'ME_L.1', 'ME_L.2', 'GH1_L', 'GH1_L.1', 'GH1_L.2', 'GH2_L', 'GH2_L.1', 'GH2_L.2']
    x_humerus_l, y_humerus_l, z_humerus_l = define_coordinate_humerus(normal_vec_columns_humerus_l, df_o)

    # 【4.右肱骨】建立肱骨坐标系（右）
    normal_vec_columns_humerus_r = ['LE_R', 'LE_R.1', 'LE_R.2', 'ME_R', 'ME_R.1', 'ME_R.2', 'GH1_R', 'GH1_R.1', 'GH1_R.2', 'GH2_R', 'GH2_R.1', 'GH2_R.2']
    x_humerus_r, y_humerus_r, z_humerus_r = define_coordinate_humerus(normal_vec_columns_humerus_r, df_o)

    # 【5.左锁骨】建立锁骨坐标系（左）, 0.XP, 1.T8, 2.SN, 3.C7, 4.SC_L, 5.AC_L
    normal_vec_columns_clavicle_l = ['XP', 'XP.1', 'XP.2', 'T8', 'T8.1', 'T8.2', 'SN', 'SN.1', 'SN.2', 'C7', 'C7.1', 'C7.2', 'SC_L', 'SC_L.1', 'SC_L.2', 'AC_L', 'AC_L.1', 'AC_L.2']
    x_clavicle_l, y_clavicle_l, z_clavicle_l = define_clavicle_coord(normal_vec_columns_clavicle_l, df_o)

    # 【6.右锁骨】建立锁骨坐标系（右）
    normal_vec_columns_clavicle_r = ['XP', 'XP.1', 'XP.2', 'T8', 'T8.1', 'T8.2', 'SN', 'SN.1', 'SN.2', 'C7', 'C7.1', 'C7.2', 'SC_R', 'SC_R.1', 'SC_R.2', 'AC_R', 'AC_R.1', 'AC_R.2']
    x_clavicle_r, y_clavicle_r, z_clavicle_r = define_clavicle_coord(normal_vec_columns_clavicle_r, df_o)

    # 【7.髋关节】 建立髋关节坐标系，0.HF_L, 1.HF_R, 2.HB_L, 3.HB_R
    normal_vec_columns_hip = ['HF_L', 'HF_L.1', 'HF_L.2', 'HF_R', 'HF_R.1', 'HF_R.2', 'HB_L', 'HB_L.1', 'HB_L.2', 'HB_R', 'HB_R.1', 'HB_R.2']
    x_hip, y_hip, z_hip = define_coordinate_hip(normal_vec_columns_hip, df_o)

    

   # 计算角度 【左肱骨坐标系和胸廓坐标系】
    df_o['A_humerus_l_thorax_X'],  df_o['A_humerus_l_thorax_Y'],  df_o['A_humerus_l_thorax_Z'] = calculate_angle_xyz(x_humerus_l, y_humerus_l, z_humerus_l, x_thorax, y_thorax, z_thorax)

    # 计算角度 【右肱骨坐标系和胸廓坐标系】
    df_o['A_humerus_r_thorax_X'],  df_o['A_humerus_r_thorax_Y'],  df_o['A_humerus_r_thorax_Z'] = calculate_angle_xyz(x_humerus_r, y_humerus_r, z_humerus_r, x_thorax, y_thorax, z_thorax)

    # 计算角度 【左肩胛骨坐标系和胸廓坐标系】
    df_o['A_scapula_l_thorax_X'],  df_o['A_scapula_l_thorax_Y'],  df_o['A_scapula_l_thorax_Z'] = calculate_angle_xyz(x_scapula_l, y_scapula_l, z_scapula_l, x_thorax, y_thorax, z_thorax)
    
    # 计算角度 【右肩胛骨坐标系和胸廓坐标系】
    df_o['A_scapula_r_thorax_X'],  df_o['A_scapula_r_thorax_Y'],  df_o['A_scapula_r_thorax_Z'] = calculate_angle_xyz(x_scapula_r, y_scapula_r, z_scapula_r, x_thorax, y_thorax, z_thorax)
    
    # 计算角度 【左锁骨坐标系和胸廓坐标系】
    df_o['A_clavicle_l_thorax_X'],  df_o['A_clavicle_l_thorax_Y'], df_o['A_clavicle_l_thorax_Z'] = calculate_angle_xyz(x_clavicle_l, y_clavicle_l, z_clavicle_l, x_thorax, y_thorax, z_thorax)
    
    # 计算角度 【右锁骨坐标系和胸廓坐标系】
    df_o['A_clavicle_r_thorax_X'],  df_o['A_clavicle_r_thorax_Y'], df_o['A_clavicle_r_thorax_Z'] = calculate_angle_xyz(x_clavicle_r, y_clavicle_r, z_clavicle_r, x_thorax, y_thorax, z_thorax)

    # 计算角度 【左肱骨坐标系和肩胛骨坐标系】
    df_o['A_humerus_l_scapula_X'],  df_o['A_humerus_l_scapula_Y'], df_o['A_humerus_l_scapula_Z'] = calculate_angle_xyz(x_humerus_l, y_humerus_l, z_humerus_l, x_scapula_l, y_scapula_l, z_scapula_l)

    # 计算角度 【右肱骨坐标系和肩胛骨坐标系】
    df_o['A_humerus_r_scapula_X'],  df_o['A_humerus_r_scapula_Y'], df_o['A_humerus_r_scapula_Z'] = calculate_angle_xyz(x_humerus_r, y_humerus_r, z_humerus_r, x_scapula_r, y_scapula_r, z_scapula_r)

    # 计算角度 【胸廓相对于髋关节坐标系】
    df_o['A_thorax_hip_X'],  df_o['A_thorax_hip_Y'], df_o['A_thorax_hip_Z'] = calculate_angle_xyz(x_thorax, y_thorax, z_thorax, x_hip, y_hip, z_hip)




    # # 计算角度1, 2【左肱骨坐标系和胸廓坐标系，3,1】
    # df_o['angle1'],  df_o['angle2'] = calculate_angle_yz(y1, z1, y3, z3)

    # # 计算角度3, 4 【右肱骨坐标系和胸廓坐标系,5,1】
    # df_o['angle3'],  df_o['angle4'] = calculate_angle_yz(y1, z1, y5, z5)

    # # 计算角度5, 6 【左肩胛骨坐标系和胸廓坐标系,2,1】
    # df_o['angle5'],  df_o['angle6'] = calculate_angle_yz(y1, z1, y2, z2)
    
    # # 计算角度7, 8 【右肩胛骨坐标系和胸廓坐标系,4,1】
    # df_o['angle7'],  df_o['angle8'] = calculate_angle_yz(y1, z1, y4, z4)

    # # 计算角度9, 10 【胸廓相对于全局,1,6】
    # df_o['angle9'],  df_o['angle10'] = calculate_angle_yz(y1, z1, y6, z6)

    # # 计算角度11, 12 【左锁骨坐标系和胸廓坐标系,7,1】
    # df_o['angle11'],  df_o['angle12'] = calculate_angle_yz(y7, z7, y1, z1)
    
    # # 计算角度13, 14 【右锁骨坐标系和胸廓坐标系,8,1】
    # df_o['angle13'],  df_o['angle14'] = calculate_angle_yz(y8, z8, y1, z1)

    # # 计算角度15, 16 【左肱骨坐标系和肩胛骨坐标系,3,2】
    # df_o['angle15'],  df_o['angle16'] = calculate_angle_yz(y3, z3, y2, z2)

    # # 计算角度17, 18 【右肱骨坐标系和肩胛骨坐标系,4,5】
    # df_o['angle17'],  df_o['angle18'] = calculate_angle_yz(y4, z4, y5, z5)

    
    return df_o
