import torch
import cv2
import numpy as np

# 定义旋转函数
def rotate_axis(axis, angle):
    """
    输入：
    axis: 旋转轴，一个三维向量，例如 [x, y, z]
    angle: 旋转角度，单位为弧度
    
    输出：
    旋转后的坐标轴
    """
    # 将旋转轴转化为单位向量
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)
    
    # 计算旋转矩阵
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x, y, z = axis
    rotation_matrix = np.array([[cos_theta + (1 - cos_theta) * x**2, (1 - cos_theta) * x * y - sin_theta * z, (1 - cos_theta) * x * z + sin_theta * y],
                                [(1 - cos_theta) * x * y + sin_theta * z, cos_theta + (1 - cos_theta) * y**2, (1 - cos_theta) * y * z - sin_theta * x],
                                [(1 - cos_theta) * x * z - sin_theta * y, (1 - cos_theta) * y * z + sin_theta * x, cos_theta + (1 - cos_theta) * z**2]])
    
    # 应用旋转矩阵到原始坐标轴
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])
    i_prime = np.dot(rotation_matrix, i)
    j_prime = np.dot(rotation_matrix, j)
    k_prime = np.dot(rotation_matrix, k)
    
    return i_prime, j_prime, k_prime

def calc_reward(pos,rotation,vel):
    done = False
    reward = 0
    offset = -np.abs(np.dot(np.array(pos[:-1]),np.array([0.707,-0.707])))
    reward = offset
    reward += np.dot(np.array(vel[:2]),np.array([-0.707,-0.707]))
    axis = rotate_axis(rotation[:3],rotation[-1])
    if offset <-0.5 or axis[-1][-1] <= 0 or pos[2]>0.4:
        reward = -100
        done = True
    return reward , done 
