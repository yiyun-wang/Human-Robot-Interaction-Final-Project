# Import the necessary libraries
import rclpy                          # Python Client Library for ROS 2
from rclpy.node import Node           # Handles the creation of nodes
from sensor_msgs.msg import Image     # Image is the message type
import cv2                            # OpenCV library
from cv_bridge import CvBridge        # Package to convert between ROS and OpenCV Images
  
class ImagePublisher(Node):
  """
  Create an ImagePublisher class, which is a subclass of the Node class.
  """
  def __init__(self, video_path):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('depth_image_publisher_node')
       
    # Create the publisher. This publisher will publish an Image
    # to the video_frames topic. The queue size is 10 messages.
    self.publisher_ = self.create_publisher(Image, '/stereo/depth', 10)
       
    # We will publish a message every 0.1 seconds
    timer_period = 0.1  # seconds
       
    # Create the timer
    self.timer = self.create_timer(timer_period, self.timer_callback)
          
    # Read in video path
    self.declare_parameter('my_parameter', 'world')

    # Create a VideoCapture object
    # The argument '0' gets the default webcam.
    self.cap = cv2.VideoCapture(video_path)
          
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
    
  def timer_callback(self):
    """
    Callback function.
    This function gets called every 0.1 seconds.
    """
    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame = self.cap.read()
    cv2.imshow(' depth image',frame)
           
    # Publish the image.
    if ret == True:
      # The 'cv2_to_imgmsg' method converts an OpenCV
      # image to a ROS2 image message
      self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
  
    # Display the message on the console
    if(frame is None):
      self.get_logger().info('Could not Publishing video frame')

def main(args=None):
  # Initialize the rclpy library
  rclpy.init(args=args)

  # Set path to video on filesystem 
  video_path = '/home/demo/ros_class_ws/src/proxemic_detector_pkg/proxemic_detector_pkg/people_dataset_depth.mp4'

  # Create the RGB node
  image_publisher = ImagePublisher(video_path)
   
  # Spin the node so the callback function is called.
  rclpy.spin(image_publisher)
   
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_publisher.destroy_node()
   
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()
