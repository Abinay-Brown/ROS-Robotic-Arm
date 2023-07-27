#!/usr/bin/env python3

from __future__ import print_function
from robotic_arm.srv import InverseKinematics,InverseKinematicsResponse
import rospy
import numpy as np
from std_msgs.msg import Time
from std_msgs.msg import Header
from std_msgs.msg import Duration
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# Forward Kinematics function
def forward_kinematics(dh_params):
    theta = np.deg2rad(dh_params[0, :])
    alpha = np.deg2rad(dh_params[1, :])
    a = dh_params[2, :]
    d = dh_params[3, :]
    
    num_joints = len(a)
    joint_locations = np.zeros((num_joints + 1, 3))
    T = np.eye(4)

    for i in range(num_joints):
        A = np.array([[np.cos(theta[i]), -np.sin(theta[i]) * np.cos(alpha[i]),
                       np.sin(theta[i]) * np.sin(alpha[i]), a[i] * np.cos(theta[i])],
                      [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]),
                       -np.cos(theta[i]) * np.sin(alpha[i]), a[i] * np.sin(theta[i])],
                      [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                      [0, 0, 0, 1]])

        T = np.dot(T, A)
        joint_locations[i + 1, :] = T[:3, 3].T

    return joint_locations


def jacobian(dh, delta_theta):
    jac = np.zeros((3, len(dh[0,:])))  # Initialize an empty Jacobian matrix
    
    temp = np.copy(dh)
    for i in range(len(dh[0, :])):
        
        
        temp = np.copy(dh)
        temp[0, i] = temp[0, i] - delta_theta
        ee_pos0 = forward_kinematics(temp)  # Assuming func is defined elsewhere
        ee_pos0 = ee_pos0[-1, :].T

        #print(ee_pos0)

        temp = np.copy(dh)
        temp[0, i] = temp[0, i] + delta_theta
        ee_pos1 = forward_kinematics(temp)  # Assuming func is defined elsewhere
        ee_pos1 = ee_pos1[-1, :].T
        
        df_dtheta = (ee_pos1 - ee_pos0) / (2 * delta_theta)
        
        
        jac[0, i] = df_dtheta[0];
        jac[1, i] = df_dtheta[1];
        jac[2, i] = df_dtheta[2]; 
        
    return jac

# Inverse Kinematics function
def inverse_kinematics(dh, ee_pos_target, tol, max_iter):
    
    temp = np.copy(dh)
    
    theta_current = temp[0, :]
    iter = 0
    while True:
        temp[0, :] = theta_current

        ee_pos_current = forward_kinematics(temp) 
        ee_pos_current = ee_pos_current[-1, :]

        jac = jacobian(temp, 0.00001) 
        jac_inv = np.linalg.pinv(jac, rcond=1e-7)
        
        delta = (ee_pos_target - ee_pos_current).T
    
        theta_current = theta_current.T
        
        theta_update = theta_current + np.dot(jac_inv, delta)
                
        theta_current = theta_update.T
        iter = iter + 1 
        val = np.linalg.norm(delta);
        print("Running IK solver, tol: {:.9f}, iter: {:.1f}".format(val, iter))
        if val < tol or iter >=max_iter:
            if val > tol and iter >=max_iter:
                print("IK solver could not converge")
            else:
                print("IK Solver converged!")
            break

    theta_current = np.mod(theta_current, 360)

    for i in range(len(theta_current)):
        if theta_current[i] > 180:
            theta_current[i] -= 360

    theta_out = theta_current
    temp[0, :] = theta_out
    joints = forward_kinematics(temp)

    return theta_out, joints


def handle_IK(req):
    
    theta = rospy.get_param("/DH_params/theta")
    alpha = rospy.get_param("/DH_params/alpha")
    a = rospy.get_param("/DH_params/a")
    d = rospy.get_param("/DH_params/d")
    tol = rospy.get_param("/IK_solver/tol")
    max_iter = rospy.get_param("/IK_solver/max_iter")
    x_target = rospy.get_param("/target/x")
    y_target = rospy.get_param("/target/y")
    z_target = rospy.get_param("/target/z")
      
    ind = req.ind; 
    
    dh_params = np.array([theta, alpha, a, d])
    ee_pos_target = np.array([x_target[ind], y_target[ind], z_target[ind]])
        
    if req.move == True and ind<=len(x_target)-1:
        
        theta_out, joints = inverse_kinematics(dh_params, ee_pos_target, tol, max_iter)

        theta_rad = np.deg2rad(theta_out)

        pub = rospy.Publisher("/arm_controller/command", JointTrajectory, queue_size=10)
        
        joints_str = JointTrajectory()
        joints_str.header = Header()
        joints_str.header.stamp = rospy.Time.now()
        joints_str.joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint']
        
        point = JointTrajectoryPoint()
        point.positions = [theta_rad[0], theta_rad[1], theta_rad[2], theta_rad[3], theta_rad[4], theta_rad[5]]
        point.time_from_start = rospy.Duration(2)
        
        joints_str.points.append(point)
        print("Moving Robotic Arm ...")
        pub.publish(joints_str)
        print("Movement Complete!")

    else:
        if ind>len(x_target)-1:
            print('Target Index Out of Bounds')
        theta_out = [0, 0, 0, 0, 0, 0]

    return InverseKinematicsResponse(theta_out[0], theta_out[1], theta_out[2], theta_out[3], theta_out[4], theta_out[5])

def IK_server():
    rospy.init_node('IK_server')
    s = rospy.Service('Inverse_Kinematics', InverseKinematics, handle_IK)
    print("Ready to perform Inverse Kinematics.")
    rospy.spin()

if __name__ == "__main__":
    IK_server()