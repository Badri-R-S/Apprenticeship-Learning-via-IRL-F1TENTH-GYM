import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pynput import keyboard
from sensor_msgs.msg import LaserScan
from time import time
import math

msg = """
Control Your Toy!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .
q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key, k : force stop
anything else : stop smoothly
CTRL-C to quit
"""

class TeleopNode(Node):

    def __init__(self):
        self.node = rclpy.create_node('custom_teleop_node')
        self.lidar_subscription = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            2  # QoS profile
        )
        self.cmd_pub = self.node.create_publisher(
            Twist,
            'cmd_vel',
            5
        )
        print("Publisher initilized")
        self.cmd_sub = self.node.create_subscription(
        Twist, 
        'cmd_vel', 
        self.cmd_callback, 
        2
        )
        print("Subscriber initialized")
        

        self.prev_v_linear_x = 0.0
        self.prev_v_angular_z = 0.0
        self.prev_timestamp = time()
        self.cmd = Twist()
        self.current_keys = set()
        self.init_keyboard()
        self.linear_x = 0.0
        self.angular_z = 0.0

    def lidar_callback(self, msg):
        # Extracting relevant information from LiDAR data
        ranges = msg.ranges
        angle_increment = msg.angle_increment
        num_readings = len(ranges)
        min_distance = min(ranges)
        angle_of_min_distance = math.degrees(angle_increment * ranges.index(min_distance) - msg.angle_max)


    def cmd_callback(self, msg):
        # Update class attributes with the received cmd_vel values
        self.prev_v_linear_x = self.linear_x
        self.prev_v_angular_z = self.angular_z
        self.prev_timestamp = time()
        self.linear_x = msg.linear.x
        self.angular_z = msg.angular.z

    def init_keyboard(self):
        print("Keyboard Listener initialized")
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        try:
            key = key.char  # Get the character representation of the key
            self.current_keys.add(key)
            self.update_cmd()
        except AttributeError:
            pass
    
    def on_release(self, key):
        try:
            key = key.char  # Get the character representation of the key
            self.current_keys.remove(key)
            self.update_cmd()
        except AttributeError:
            pass
    
    def update_cmd(self):
        # Reset the command
        #self.cmd.linear.x = 0.0
        #self.cmd.angular.z = 0.0
        #print("here")
        if 'i' in self.current_keys:
            if self.cmd.linear.x < 5:
                self.cmd.angular.z = 0.0 
                self.cmd.linear.x += 1.0  # Increase speed
        elif 'j' in self.current_keys:
            self.cmd.linear.x = 1.0  # Increase speed
            self.cmd.angular.z = 1.0  # Turn right
        elif 'k'in self.current_keys:
            self.cmd.linear.x -= 1.0
            if (self.cmd.linear.x < 0):
                self.cmd.linear.x = 0.0
        elif 'l' in self.current_keys:
            self.cmd.linear.x = 1.0
            self.cmd.angular.z = -1.0
        # Add more key handlers for other movements

        self.send_cmd()

    def send_cmd(self):
        self.cmd_pub.publish(self.cmd)
    
    def run(self):
        rclpy.spin(self.node)
    
def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        print(msg)
        node.run()
    finally:
        rclpy.shutdown()
        node.listener.stop()

if __name__ == '__main__':
    print("Custom Teleoperation")
    main()