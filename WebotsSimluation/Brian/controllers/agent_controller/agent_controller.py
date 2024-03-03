"""agent_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import torch
import os

print(torch.cuda.is_available())
# create the Robot instance.
robot = Robot()
fll = robot.getDevice("joint_fll")
flf = robot.getDevice("joint_flf")
frl = robot.getDevice("joint_frl")
frf = robot.getDevice("joint_frf")
bll = robot.getDevice("joint_bll")
blf = robot.getDevice("joint_blf")
brl = robot.getDevice("joint_brl")
brf = robot.getDevice("joint_brf")
camera = robot.getDevice("camera")
camera.enable(100)
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
pos = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # camera.saveImage("img.png",100)
    fll.setPosition(pos)
    flf.setPosition(pos)
    bll.setPosition(pos)
    blf.setPosition(pos)
    frl.setPosition(pos)
    frf.setPosition(pos)
    brl.setPosition(pos)
    brf.setPosition(pos)
    pos -= 0.1
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
