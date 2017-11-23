#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *


# Define Modified DH Transformation matrix
def TF_Matrix(alpha, a, d, q):
    TF = Matrix([[cos(q), -sin(q), 0, a],
                 [sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                 [sin(q) * sin(alpha), cos(q) * sin(alpha), cos(alpha), cos(alpha) * d],
                 [0, 0, 0, 1]])
    return TF


# functions for Rotation Matrices about x, y, and z given specific angle.
def rot_x(r):
    return Matrix([[1, 0, 0], [0, cos(r), -sin(r)], [0, sin(r), cos(r)]])


def rot_y(p):
    return Matrix([[cos(p), 0, sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]])


def rot_z(y):
    return Matrix([[cos(y), -sin(y), 0], [sin(y), cos(y), 0], [0, 0, 1]])


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        # DH param symbols
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')  # link offset
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')  # link length
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')  # twist angle
        q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')  # Joint angle

        # Create Modified DH parameters (KR210 Forward kinematics section)
        DH_Table = {alpha0: 0, a0: 0, d1: 0.75, q1: q1,
                    alpha1: -pi / 2., a1: 0.35, d2: 0, q2: q2 - pi / 2,
                    alpha2: 0, a2: 1.25, d3: 0, q3: q3,
                    alpha3: -pi / 2., a3: -0.054, d4: 1.5, q4: q4,
                    alpha4: pi / 2., a4: 0, d5: 0, q5: q5,
                    alpha5: -pi / 2., a5: 0, d6: 0, q6: q6,
                    alpha6: 0, a6: 0, d7: 0.303, q7: 0}

        # Create Individual Transformation matrices for each joint transition
        T0_1 = TF_Matrix(alpha0, a0, d1, q1).subs(DH_Table)
        T1_2 = TF_Matrix(alpha1, a1, d2, q2).subs(DH_Table)
        T2_3 = TF_Matrix(alpha2, a2, d3, q3).subs(DH_Table)
        T3_4 = TF_Matrix(alpha3, a3, d4, q4).subs(DH_Table)
        T4_5 = TF_Matrix(alpha4, a4, d5, q5).subs(DH_Table)
        T5_6 = TF_Matrix(alpha5, a5, d6, q6).subs(DH_Table)
        T6_EE = TF_Matrix(alpha6, a6, d7, q7).subs(DH_Table)
        T0_EE = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_EE


        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            # roll, pitch, yaw = end-effector orientation
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            r, p, y = symbols('r p y')  # Gyroscope symbols - Roll, Pitch and Yaw

            ROT_x = rot_x(r)
            ROT_y = rot_y(p)
            ROT_z = rot_z(y)
            ROT_EE = ROT_z * ROT_y * ROT_x

            # KR210 Forward Kinematics section for future details
            Rot_Error = ROT_z.subs(y, radians(180)) * ROT_y.subs(p, radians(-90))
            ROT_EE = ROT_EE * Rot_Error
            ROT_EE = ROT_EE.subs({'r': roll, 'p': pitch, 'y': yaw})

            EE = Matrix([[px],
                         [py],
                         [pz]])

            WC = EE - 0.303 * ROT_EE[:, 2]

            # Calculate joint angles using Geometric IK method.
            # More information can be found in the Inverse Kinematics with Kuka KR210
            WC1_0_dist = sqrt(WC[0] * WC[0] + WC[1] * WC[1])
            theta1 = atan2(WC[1], WC[0])

            side_a = 1.501
            side_b = sqrt(pow((WC1_0_dist - 0.35), 2) + pow((WC[2] - 0.75), 2))
            side_c = 1.25

            angle_a = acos(((side_b * side_b) + side_c * side_c - side_a * side_a) / (2 * side_b * side_c))
            angle_b = acos(((side_a * side_a) + side_c * side_c - side_b * side_b) / (2 * side_a * side_c))
            angle_c = acos(((side_a * side_a) + side_b * side_b - side_c * side_c) / (2 * side_a * side_b))

            theta2 = pi / 2 - angle_a - atan2(WC[2] - 0.75, WC1_0_dist - 0.35)
            theta3 = pi / 2 - (angle_b + 0.036)

            R0_3 = T0_1[0:3, 0:3] * T1_2[0:3, 0:3] * T2_3[0:3, 0:3]
            R0_3 = R0_3.evalf(subs={q1: theta1, q2: theta2, q3: theta3})
            R3_6 = R0_3.inv("LU") * ROT_EE

            # Euler angles from rotation matrix
            # More information can be found in the Euler Angles from a Rotation Matrix section
            theta4 = atan2(R3_6[2, 2], -R3_6[0, 2])
            theta5 = atan2(sqrt(R3_6[0, 2] * R3_6[0, 2] + R3_6[2, 2] * R3_6[2, 2]), R3_6[1, 2])
            theta6 = atan2(-R3_6[1, 1], R3_6[1, 0])

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
