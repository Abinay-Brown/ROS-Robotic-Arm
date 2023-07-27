import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy


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
def inverse_kinematics(dh, ee_pos_target, tol):
    
    temp = np.copy(dh)
    
    theta_current = temp[0, :]

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
        
        val = np.linalg.norm(delta);
        print(val)
        if val < tol:
            break

    theta_current = np.mod(theta_current, 360)

    for i in range(len(theta_current)):
        if theta_current[i] > 180:
            theta_current[i] -= 360

    theta_out = theta_current
    temp[0, :] = theta_out
    joints = forward_kinematics(temp)

    return theta_out, joints



# Main code
dh_params = [[0, 0, 0, 0, 0, 0],
             [90, 0, 0, -90, 90, 0],
             [0, 0.269, 0.34516, 0.34516, 0.27, 0.269],
             [0.20271, 0, 0, 0, 0, 0]]
    
dh_params = np.array(dh_params)

joints = forward_kinematics(dh_params)


# Calculate joint locations using Forward Kinematics
#original_joints = joints

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(joints[:, 0], joints[:, 1], joints[:, 2], '-*')
   
# Define end-effector target position
ee_pos_target = np.array([0.7, 0.5, 0.7])

# Calculate inverse kinematics to obtain new joint angles
theta, joints = inverse_kinematics(dh_params, ee_pos_target, 10e-9)
print(theta)
print(joints[-1, :])
ax.plot3D(joints[:, 0], joints[:, 1], joints[:, 2], '-*')

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
# Adjust aspect ratio of the plot
max_range = np.max(joints[:, :3]) - np.min(joints[:, :3])
x_mid = 0.5 * (np.max(joints[:, 0]) + np.min(joints[:, 0]))
y_mid = 0.5 * (np.max(joints[:, 1]) + np.min(joints[:, 1]))
z_mid = 0.5 * (np.max(joints[:, 2]) + np.min(joints[:, 2]))
ax.set_xlim([x_mid - 0.5 * max_range, x_mid + 0.5 * max_range])
ax.set_ylim([y_mid - 0.5 * max_range, y_mid + 0.5 * max_range])
ax.set_zlim([z_mid - 0.5 * max_range, z_mid + 0.5 * max_range])
    
plt.show()
