#import gym
#from gym import spaces
import gym
from gym import spaces
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped,Twist
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import threading
from ackermann_msgs.msg import AckermannDriveStamped


class f110_gym(gym.Env):
	"""Custom Environment that follows OpenAIgym interface"""
	def __init__(self):
		super(f110_gym, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
	    # Example when using discrete actions:
		self.action_space = spaces.Discrete(4)
		# Lidar observation space:
		self.observation_space = spaces.Box(low=-100, high=100,
											shape=(10,), dtype=np.float32)
		self.render_mode = None
		if not rclpy.ok():
			rclpy.init()
		self.f110_gym_node= rclpy.create_node('f110_gym_node')
		#self.observation_lock=threading.Lock()
		#self.ros_thread = threading.Thread(target=self.run_ros_communication)
		#self.ros_thread.start()
		self.alpha = 0.5013996566712274
		self.beta = 0.10113122177893705
		self.gamma = 0.021383200066075105
		self.prev_vx = 0.0
		self.prev_vz = 0.0
		self.max_vel = 5.0
		self.min_vel = 1.0
		self.pose_x = 0.0
		self.pose_y = 0.0
		self.orient_z = 0.0
		self.selected_readings = None
		self.vel_x = 0.0
		self.vel_z = 0.0
		self.observation = None
		self.reward = 0.0 
		
		#self.velo_publisher = self.f110_gym_node.create_publisher(AckermannDriveStamped,
		#							'/drive',
		#						10)
		self.velo_publisher = self.f110_gym_node.create_publisher(Twist,
									'/cmd_vel',
									10)
		
		self.lidar_subscrirber = self.f110_gym_node.create_subscription(LaserScan,
										 '/scan',
										 self.laser_scan_callback,
										 10)
		
		self.odom_subscriber = self.f110_gym_node.create_subscription(Odometry,
											 '/ego_racecar/odom',
											 self.odom_callback,
											 10)
		
		#self.vel_subscriber = self.f110_gym_node.create_subscription(AckermannDriveStamped,
		#									 '/drive',
		#									 self.vel_callback,
		#									 10)
		self.vel_subscriber = self.f110_gym_node.create_subscription(Twist,
											 '/cmd_vel',
											 self.vel_callback,
											 10)
		#self.run_ros_communication()

	#def run_ros_communication(self):
	#	rclpy.spin_once(self.f110_gym_node)
	
	def step(self,action):
		
			if action == 0:
				"""Forward"""
				if(self.prev_vx>=self.max_vel):
					vx = 5.0
					vz = 0.0
					self.set_velocity(vl_x=vx,vl_z=vz)
				else:
					vx = self.prev_vx+1
					vz = 0.0
					self.set_velocity(vl_x = vx,vl_z=vz)
			elif action == 1:
				"""Reverse"""
				if(self.prev_vx<= self.min_vel):
					vx = -1.0
					vz = 0.0
					self.set_velocity(vl_x = vx, vl_z=vz)
				else:
					vx = self.prev_vx-1
					vz = 0.0
					self.set_velocity(vl_x = vx,vl_z=vz)
			elif action == 2:
				"""Left"""
				vx = 1.0
				vz = 1.0
				self.set_velocity(vl_x = vx, vl_z = vz)
			elif action == 3:
				"""Right"""
				vx = 1.0
				vz = -1.0
				self.set_velocity(vl_x = vx, vl_z = vz)
			else:
				raise ValueError(f"Invalid action: {action}")
			
			# Storing previous values
			self.prev_vx = vx
			self.prev_vz = vz
			#Getting observations
			self.get_observations()
			#self.observation = [self.pose_x,self.pose_y,self.orient_z,self.vel_x,self.vel_z,self.selected_readings[0]
			#		   			,self.selected_readings[1],self.selected_readings[2],self.selected_readings[3],self.selected_readings[4]
			#					,self.selected_readings[5],self.selected_readings[6],self.selected_readings[7],self.selected_readings[8]
			#					,self.selected_readings[9]]
			self.observation = [self.selected_readings[0]
					   			,self.selected_readings[1],self.selected_readings[2],self.selected_readings[3],self.selected_readings[4]
								,self.selected_readings[5],self.selected_readings[6],self.selected_readings[7],self.selected_readings[8]
								,self.selected_readings[9]]
			#print(self.observation)
			observation = [np.atleast_1d(obs) for obs in self.observation]
			flat_obs = np.concatenate(observation).ravel()
			#Getting reward
			self.reward = self.reward_calc(self.observation)
			info ={}
			return flat_obs,self.reward,self.done,info
	
	def reward_calc(self,obs):
		#print(obs)
		speed_reward = self.alpha * self.vel_x
		angular_penalty = -self.gamma * self.vel_z
		#if min(obs[5],obs[6],obs[7],obs[8],obs[9],obs[10],obs[11],obs[12],obs[13],obs[14]) < 0.5:
		if min(obs[0],obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],obs[7],obs[8],obs[9]) < 0.5:
			reward = -1000.0
			self.done = True
		elif min(obs[0],obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],obs[7],obs[8],obs[9]) < 1:
			collision_penalty = -100.0 * (min(obs[0],obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],obs[7],obs[8],obs[9])/np.sum([obs[0],obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],obs[7],obs[8],obs[9]]))
			reward = speed_reward + collision_penalty +angular_penalty
			self.done = False
		else:
			collision_penalty = self.beta * np.sum([obs[0],obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],obs[7],obs[8],obs[9]])
			reward = speed_reward + collision_penalty + angular_penalty
			self.done = False
		return reward
	
	def reset(self):
			self.done = False
			self.set_initial_pose(0.0 , 0.0 , np.pi/2)
			self.set_velocity(vl_x =0.0 ,vl_z= 0.0)
			self.get_observations()
			#self.observation = [self.pose_x,self.pose_y,self.orient_z,self.vel_x,self.vel_z,self.selected_readings[0]
			#		   			,self.selected_readings[1],self.selected_readings[2],self.selected_readings[3],self.selected_readings[4]
			#					,self.selected_readings[5],self.selected_readings[6],self.selected_readings[7],self.selected_readings[8]
			#					,self.selected_readings[9]]
			self.observation = [self.selected_readings[0]
					   			,self.selected_readings[1],self.selected_readings[2],self.selected_readings[3],self.selected_readings[4]
								,self.selected_readings[5],self.selected_readings[6],self.selected_readings[7],self.selected_readings[8]
								,self.selected_readings[9]]
			observation = [np.atleast_1d(obs) for obs in self.observation]
			flat_obs = np.concatenate(observation).ravel()
			return flat_obs
	
	def render(self, mode='human'):
		...
	def close (self):
		...
	
	def set_initial_pose(self,x,y,theta):
		#node = rclpy.create_node('initial_pose_publisher')
		
			publisher = self.f110_gym_node.create_publisher(PoseWithCovarianceStamped, 
										'/initialpose', 
										10)
			pose_msg = PoseWithCovarianceStamped()
			pose_msg.header = Header()
			pose_msg.header.frame_id = 'map'  # Assuming your frame_id is 'map'
			pose_msg.pose.pose.position.x = x
			pose_msg.pose.pose.position.y = y
			pose_msg.pose.pose.orientation.z = theta
			publisher.publish(pose_msg)
			#node.destroy_node()
			#rclpy.shutdown()
	
	#def set_velocity(self,vl_x,vl_z):
	#	
	#		if not rclpy.ok():
	#			rclpy.init()
	#		#node=rclpy.create_node('set_velocity')
	#		vel_msg = AckermannDriveStamped()
	#		vel_msg.drive.speed = vl_x
	#		vel_msg.drive.steering_angle = vl_z
	#		self.velo_publisher.publish(vel_msg)
			#node.destroy_node()
			#rclpy.shutdown()
		
	def set_velocity(self,vl_x,vl_z):
		
			if not rclpy.ok():
				rclpy.init()
			#node=rclpy.create_node('set_velocity')
			vel_msg = Twist()
			vel_msg.linear.x = vl_x
			vel_msg.angular.z = vl_z
			self.velo_publisher.publish(vel_msg)
			#node.destroy_node()
			#rclpy.shutdown()
	def get_observations(self):
		if not rclpy.ok():
			rclpy.init()
		#node = rclpy.create_node('get_data')
		rclpy.spin_once(self.f110_gym_node, timeout_sec=0.1)
		
	
	def laser_scan_callback(self,msg):
		
			ranges = msg.ranges
			angle_increment = msg.angle_increment
			num_readings = len(ranges)
			# Define the number of readings to capture (adjust as needed)
			num_readings_to_capture = 10
			# Calculate the step size to downsample the readings
			step_size = max(1, num_readings // num_readings_to_capture)
			# Select a subset of readings based on the step size
			self.selected_readings = ranges[::step_size]
			#print(self.selected_readings)
		

	def odom_callback(self,msg):
		
			self.pose_x = msg.pose.pose.position.x
			self.pose_y = msg.pose.pose.position.y
			self.orient_z = msg.pose.pose.orientation.z
			#print(self.pose_x,self.pose_y)

	#def vel_callback(self,msg):
	#		self.vel_x = msg.drive.speed
	#		self.vel_z = msg.drive.steering_angle
	
	def vel_callback(self,msg):
			self.vel_x = msg.linear.x
			self.vel_z = msg.angular.z
