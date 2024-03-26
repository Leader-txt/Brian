from controller import Robot , Supervisor
import numpy as np
import cv2
from SAC import SAC
from ReplayBuffer import ReplayBuffer
import time
import torch
import Utils
from collections import deque
from matplotlib import pyplot as plt
import kinematic
import cycloid
import math

robot = Supervisor()
base = robot.getSelf()
fll = robot.getDevice("joint_fll")
fll_sensor = robot.getDevice("joint_fll_sensor")
flf = robot.getDevice("joint_flf")
flf_sensor = robot.getDevice("joint_flf_sensor")
frl = robot.getDevice("joint_frl")
frl_sensor = robot.getDevice("joint_frl_sensor")
frf = robot.getDevice("joint_frf")
frf_sensor = robot.getDevice("joint_frf_sensor")
bll = robot.getDevice("joint_bll")
bll_sensor = robot.getDevice("joint_bll_sensor")
blf = robot.getDevice("joint_blf")
blf_sensor = robot.getDevice("joint_blf_sensor")
brl = robot.getDevice("joint_brl")
brl_sensor = robot.getDevice("joint_brl_sensor")
brf = robot.getDevice("joint_brf")
brf_sensor = robot.getDevice("joint_brf_sensor")
motors = [fll,flf,frl,frf,bll,blf,brl,brf]
sensors = [fll_sensor,flf_sensor,frl_sensor,frf_sensor,bll_sensor,blf_sensor,brl_sensor,brf_sensor]
for sensor in sensors:
    sensor.enable(100)
camera = robot.getDevice("camera")
gyro = robot.getDevice("gyro")
accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(100)
gyro.enable(100)
camera.enable(100)
timestep = int(robot.getBasicTimeStep())
device = torch.device('cuda')
# agent = SAC()
# agent.load()
replay_buffer = ReplayBuffer()
episode = 0
total_reward = 0
reward_list = deque(maxlen=1000)
for i in range(100):
    robot.step()
step = 0
kine = kinematic.kinematic(200,300,14.48,180-19.26)
cycl = cycloid.cycloid(100,30,2)
h = 200
a1 , a2 = kine.pos2angle(0,h)
for j in range(100):
    for i in range(4):
        motors[i*2].setPosition(a1*j/100)
        motors[i*2+1].setPosition(a2*j/100)
    robot.step(timestep)

x , y = cycl.generate(0)
a11 , a12 = kine.pos2angle(x,y+h)
x , y = cycl.generate(2)
a21 , a22 = kine.pos2angle(x,y+h)
for i in range(100):
    motors[0].setPosition(a1+(a11-a1)*i/100)
    motors[1].setPosition(a2+(a12-a2)*i/100)
    motors[6].setPosition(a1+(a11-a1)*i/100)
    motors[7].setPosition(a2+(a12-a2)*i/100)
    motors[2].setPosition(a1+(a21-a1)*i/100)
    motors[3].setPosition(a2+(a22-a2)*i/100)
    motors[4].setPosition(a1+(a21-a1)*i/100)
    motors[5].setPosition(a2+(a22-a2)*i/100)
    robot.step(timestep)

while 1:
    for i in range(400):
        x , y = cycl.generate((i/100)%4)
        a1 , a2 = kine.pos2angle(x,y+h)
        motors[0].setPosition(a1)
        motors[1].setPosition(a2)
        motors[6].setPosition(a1)
        motors[7].setPosition(a2)
        x , y = cycl.generate((i/100+2)%4)
        a1 , a2 = kine.pos2angle(x,y+h)
        motors[2].setPosition(a1)
        motors[3].setPosition(a2)
        motors[4].setPosition(a1)
        motors[5].setPosition(a2)
        robot.step(timestep)
# plt.show()
# while robot.step(timestep) != -1:
#     step+=1
#     state = gyro.getValues() + accelerometer.getValues() + [sensors[i].getValue() for i in range(8)]
#     action = agent.take_action(state)
#     for i in range(8):
#         motors[i].setPosition(action[i])
#     # sames = 0
#     # last_error = 0
#     # while robot.step(timestep) != -1:
#     #     errors = 0
#     #     for i in range(8):
#     #         errors +=abs(action[i] - sensors[i].getValue())
#     #     if errors == last_error:
#     #         sames += 1
#     #     else:
#     #         sames = 0
#     #         last_error = errors
#     #     if sames > 3:
#     #         break
        
#     next_state = gyro.getValues() + accelerometer.getValues() + [sensors[i].getValue() for i in range(8)]

#     pos = base.getPosition()
#     vel = base.getVelocity()
#     rotation=base.getField("rotation").getSFVec3f()
#     reward , done = Utils.calc_reward(pos,rotation,vel)
#     total_reward += reward
#     if done :
#         step = 0
#         agent.save()
#         reward_list.append(total_reward)
#         episode += 1
#         print(f"reward:{total_reward} episode:{episode}",end='\r')
#         plt.plot(reward_list,label='reward')
#         plt.legend()
#         plt.title("Rewards")
#         plt.xlabel("episodes")
#         plt.ylabel("Reward")
#         plt.savefig("figure.png")
#         plt.close()
#         total_reward = 0
#         # robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
#         robot.simulationReset()
#         state = gyro.getValues() + accelerometer.getValues() + [sensors[i].getValue() for i in range(8)]
#     if replay_buffer.size()>1024:
#         states,actions,rewards,next_states,dones = replay_buffer.sample(1024)
#         agent.update(states,actions,rewards,next_states,dones)
#     replay_buffer.add(state,action,reward,next_state,done)
# print("end")