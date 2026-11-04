# 🤖 Human Hand Motion Retargeting for Robotic Grasping and Sorting

## 📖 Overview
This project presents a vision-based robotic manipulation system that captures human hand motions and transfers them to a robotic hand through inverse kinematics. Human finger joint positions are recorded using a hand-tracking pipeline and mapped onto the robotic hand configuration in simulation.

The robotic hand performs grasping and pick-and-place operations on multiple everyday objects including:

Water bottle
Temple bell
Mug with handle

The system further integrates object color detection and classification, enabling autonomous color-based sorting of grasped objects into designated bins.

The project demonstrates how hand motion retargeting combined with vision-based object recognition can provide an intuitive framework for robotic manipulation without requiring extensive robot programming.

video: https://drive.google.com/file/d/1Db1mymY3WHe-386UWCCORtM7ovbT9ORJ/view?usp=sharing

## ✨ Key Features

Human Hand Motion Capture
Real-time recording of human finger movements through a vision-based hand tracking system.
Inverse Kinematics Retargeting
Conversion of tracked human finger positions into robot joint commands using inverse kinematics.
Multi-Object Grasping
Support for grasping differently shaped objects:
Water bottle
Temple bell
Mug with handle
Pick and Place Pipeline

End-to-end object pickup, transport, and placement workflow.
Color-Based Sorting


## 🏗️ System Architecture

1. Hand Tracking
A camera captures the user's hand motion and extracts finger keypoints.
2. Motion Retargeting
The detected hand landmarks are converted into robotic hand joint targets using inverse kinematics.
3. Object Detection and Color Classification
Vision algorithms detect objects and classify them based on color attributes.
4. Manipulation Execution
The robotic hand performs:
Grasping
Lifting
Transportation
Color-based placement


## 📁 Repository Structure

```text
├── all_objects.py
├── all_objects.xml             
├── requirements.txt         
└── README.md                

