import os
import rclpy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import csv
import math
from time import time
from pynput import keyboard as pyk  # Import the keyboard library
import keyboard
from nav_msgs.msg import Odometry
import threading
import numpy as np

class LidarDataProcessor:
    def __init__(self):
        self.node = rclpy.create_node('lidar_data_processor')
        self.lidar_subscription = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10  # QoS profile
        )
        self.cmd_vel_subscription = self.node.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10  # QoS profile
        )
        self.odom_subscription = self.node.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )
        
        self.data_directory = 'data'
        os.makedirs(self.data_directory, exist_ok=True)  # Create 'data' directory if it doesn't exist
        self.csv_file_path = os.path.join(self.data_directory, 'data_map0.csv')
        file_exists = os.path.isfile(self.csv_file_path)
        # Open the CSV file in append mode
        self.csv_file = open(self.csv_file_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write header only if the file is newly created
        if not file_exists:
            self.csv_writer.writerow(['pos_x', 'pos_y', 'orient', 'v_linear_x', 'v_angular_z','ld1','ld2','ld3','ld4','ld5','ld6','ld7','ld8','ld9','ld10', 'action'])
        self.prev_data = None

        # Initialize attributes
        self.posx = 0.0
        self.posy = 0.0
        self.ori = 0.0
        self.v_linear_x = 0.0
        self.v_angular_z = 0.0
        self.key = None

        self.prev_v_linear_x = 0.0
        self.prev_v_angular_z = 0.0
        self.key_lock = threading.Lock()

        self.init_keyboard()
    
    def init_keyboard(self):
            print("Keyboard Listener initialized")
            self.listener = pyk.Listener(
            on_press=self.on_press
            )
            self.listener.start()
        
    def on_press(self, key):
        try:
            key_val = key.char
            print(key_val)
            #if key_val == 'i':
            #key_val = 0
            #elif key_val == 'j':
            #key_val =2
            #elif key_val == 'k':
            #key_val = 1
            #else:
            #key_val =3
            with self.key_lock:
                self.key = key_val
        except AttributeError:
            pass

    def odom_callback(self,msg):
        self.posx = msg.pose.pose.position.x
        self.posy = msg.pose.pose.position.y
        self.ori = msg.pose.pose.orientation.z

    def lidar_callback(self, msg):
        # Extracting relevant information from LiDAR data
        ranges = msg.ranges
        angle_increment = msg.angle_increment
        num_readings = len(ranges)
        # Define the number of readings to capture (adjust as needed)
        num_readings_to_capture = 10
        # Calculate the step size to downsample the readings
        step_size = max(1, num_readings // num_readings_to_capture)
        # Select a subset of readings based on the step size
        selected_readings = ranges[::step_size]
        selected_readings = list(selected_readings)
        min_distance = min(selected_readings)
        angle_of_min_distance = math.degrees(angle_increment * ranges.index(min_distance) - msg.angle_max)
        
        # Writing data to CSV file
        current_data = [self.posx,self.posy,self.ori,self.v_linear_x, self.v_angular_z, selected_readings[0]
					   			,selected_readings[1],selected_readings[2],selected_readings[3],selected_readings[4]
								,selected_readings[5],selected_readings[6],selected_readings[7],selected_readings[8]
								,selected_readings[9] , self.key]
        
        if self.prev_data == None:
            with self.key_lock:
                            self.csv_writer.writerow(current_data)
                            self.csv_file.flush()
                            self.prev_data = current_data
        else:
            if self.key == 'j' or self.key =='k' or self.key == 'l':
                if abs(np.arctan2((current_data[1]-self.prev_data[1]),(current_data[0]-self.prev_data[0]))) > 0.2:   
                    with self.key_lock:
                                self.csv_writer.writerow(current_data)
                                self.csv_file.flush()
                                self.prev_data = current_data

            else:
                if (np.sqrt((current_data[0] - self.prev_data[0])**2 + (current_data[1]- self.prev_data[1])**2)) > 0.2:
                    with self.key_lock:
                            self.csv_writer.writerow(current_data)
                            self.csv_file.flush()
                            self.prev_data = current_data
    
    def cmd_vel_callback(self, msg):
        # Extracting linear and angular velocities from cmd_vel
        self.prev_v_linear_x = self.v_linear_x
        self.prev_v_angular_z = self.v_angular_z
        self.prev_timestamp = time()

        self.v_linear_x = msg.linear.x
        self.v_angular_z = msg.angular.z

    def run(self):
        rclpy.spin(self.node)

if __name__ == '__main__':
    print("Data collection")
    rclpy.init()
    lidar_processor = LidarDataProcessor()
    lidar_processor.run()
    rclpy.shutdown()
