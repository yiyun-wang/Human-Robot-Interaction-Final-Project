"""state machine
        -------
        """
self.state1 = 'initialization '
self.state2 = 'run'
self.state3 = 'capture_zone'
self.state4 = 'capture'
        if detect user:
            wait for seconds
            if user in intimate zone:
                proceed to state2
            elif user in capture_zone:
                proceed to state
                wait for 10 seconds
                if detect color:
                    captured
                else: 
                    proceed to state2
        if(self.curr_state == self.state1):
            
            wait for seconds
            if user in intimate zone:
                
                proceed to state2
                self.next_state = self.state2
            elif user in capture zone:
                proceed to state3 
                self.next_state = self.state3
            
            if(self.curr_state == self.state2):
                # Wait for Nav2
                self.navigator.waitUntilNav2Active()
                # Robot alerts user that it is departing
                self.robot_talker(
                    robot_phrase='Sorry, you failed to capture me.')
                # Set goal poses
                self.goal_pose = []
                self.add_goal(self.dest_positions['x2'],
                            self.dest_positions['y2'],
                            self.dest_positions['direction2'])
                # Follow Waypoints
                self.navigator.startFollowWaypoints(self.goal_pose)
                # Finished navigating to docking station
                self.navigator.dock()
                # Advance to next state
                self.next_state = self.state4


        if(self.curr_state == self.state1):
            # Human detetcion 
            
            # Depth measurement 
            
            # Proceed to state 2/3 
            
            
            # Wait for Nav2
            self.navigator.waitUntilNav2Active()
            
            # Set goal poses
            self.goal_pose = []
            # self.add_goal(self.userdatetime A combination of a date and a time. Attributes: 
            self.add_goal(self.user_position['x'], 
                            self.user_position['y'],
                            self.user_position['direction'])
            
            # Robot alerts user that it is apporaching user

            # Follow Waypoints
            self.navigator.startFollowWaypoints(self.goal_pose)
            
            # Robot play sounds of 
            self.sound_play(robot_phrase='')

            # Generate robot expression
            if(self.robot_expression == 'gesture'):
                self.generate_gesture_expression()
            elif(self.robot_expression == 'lightring'):
                self.generate_lightring_expression()
            else: 
                self.generate_gesture_lightring_expression()

            #lightring vs. robotdance
            # Advance to next state
            self.next_state = self.state2
            
        elif(self.curr_state == self.state2):
            # Wait for Nav2
            self.navigator.waitUntilNav2Active()
            
            # Set goal poses
            self.goal_pose = []
            self.add_goal(self.dest_positions['x1'], 
                            self.dest_positions['y1'],
                            self.dest_positions['direction1'])
            
            # Follow Waypoints
            self.navigator.startFollowWaypoints(self.goal_pose)

            # Robot alerts user that they reached destination
            self.robot_talker(robot_phrase='We have reached your destination. Have a good day.')
            # Generate robot expression
            if(self.robot_expression == 'gesture'):
                self.generate_gesture_expression()
            elif(self.robot_expression == 'lightring'):
                self.generate_lightring_expression()
            else: 
                self.generate_gesture_lightring_expression()
                
            # Advance to next state
            self.next_state = self.state3
            
        elif(self.curr_state == self.state3):
            # Wait for Nav2
            self.navigator.waitUntilNav2Active()
            
            # Robot alerts user that it is departing
            self.robot_talker(robot_phrase='I am heading to my docking station.')
            
            # Set goal poses
            self.goal_pose = []
            self.add_goal(self.dest_positions['x2'], 
                            self.dest_positions['y2'],
                            self.dest_positions['direction2'])

            # Follow Waypoints
            self.navigator.startFollowWaypoints(self.goal_pose)

            # Finished navigating to docking station
            self.navigator.dock()
            
            # Advance to next state
            self.next_state = self.state4
            
        elif(self.curr_state == self.state4):
            # Robot alerts user that it is done
            self.robot_talker(robot_phrase='My work is complete. Good day.')
            # End state machine
            self.next_state = None



        
    def update_state_machine(self):
        # Detect the distance to objects
        selected_bbox, distance_to_object = self.detection_object_distance()
        #Detecting User
        if(self.curr_state == self.state1): # Condition to next state
            # Do something
                if selected_bbox == None:                    
                    self.next_state = self.state2
                else:
                    self.next_state = self.state3
            # Condition to next stat
        #START
        if(self.curr_state == self.state1): # Condition to next state
            #If no user is detected, move the robot
                if selected_bbox == None:                    
                    self.next_state = self.state2
            #else, start to detect the distance with the user
                else:
                    self.next_state = self.state3
            # Condition to next stat

        #ROTATE until detecting an user
        elif(self.curr_state == self.state2): # Condition to next state
            if selected_bbox == None:
                 self.move_robot(0,z)
            else:
                self.counter = 10
                self.next_state = self.state3
                self.stop_robot()
        #Detected an user and check if in a capture zone
        elif(self.curr_state == self.state3): 
            in_proxemic = False

            if (distance_to_object >= self.proxemic_ranges["public_depth_threshold_min"] and distance_to_object <= self.proxemic_ranges["public_depth_threshold_max"]):
                in_proxemic = True
                proxemic = "Public"
                #go to capture state
                self.next_state= self.state4
            # if it's too close to the user, end the game and the robot will go back to it's docking station
            elif(distance_to_object >= self.proxemic_ranges["intimate_depth_threshold_min"] and distance_to_object <= self.proxemic_ranges["intimate_depth_threshold_max"]):
                self.navigator.waitUntilNav2Active()
                # Robot alerts user that it is departing
                self.robot_talker(
                    robot_phrase='Sorry, you failed to capture me.')
                # Set goal poses
                self.goal_pose = []
                self.add_goal(self.dest_positions['x2'],
                            self.dest_positions['y2'],
                            self.dest_positions['direction2'])
                # Follow Waypoints
                self.navigator.startFollowWaypoints(self.goal_pose)
                # Finished navigating to docking station
                self.navigator.dock()
                # Advance to end state
                self.next_state= self.state5

        #ALERT
        elif(self.curr_state == self.state4): # Condition to next state
            # alert user
            self.robot_talker("Inside proxemix zone " + proxemic)
            self.robot_talker("How comfortable are you inside the " + proxemic + " zone?")
            self.next_state= self.state5
            time.sleep(10)
            # Detect the distance to objects
            selected_bbox, distance_to_object = self.detection_object_distance()
            
            if selected_bbox is None:
                self.next_state= self.state3
            else: 
                
                
            
        # End State
        elif(self.curr_state == self.state5):
            self.robot_talker("I have reached my final state")
            self.next_state = None

        # Advance to next state
        self.curr_state = self.next_state
        time.sleep(1)
        
    
    def rgb_callback(self, msg):
        """Convert ROS RGB sensor_msgs to opencv image
        ----------
        msg : Depth sensor_msgs image data
            A Depth image of format sensor_msgs Image with one channel measuring the distance to objects.
        Returns
        -------
        None
        """
        try:
            # TASK 1: Convert ROS Image message to OpenCV image
            self.rgb_image = self.rgb_bridge.imgmsg_to_cv2(msg)

            # Run color detection on image to generate bounding box color blobs
            self.rgb_image_labeled, self.bboxes = self.read_process_image_color(self.rgb_image, color=self.color)
            
            # Display image
            if(self.DISPLAY and self.rgb_image is not None):
                cv2.imshow("RGB Image", self.rgb_image_labeled)
                cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

        if(self.rgb_image is None):
            self.get_logger().info('Failed to load RGB image')

    def depth_callback(self, msg): 
        """Convert ROS depth sensor_msgs to opencv image
        ----------
        msg : Depth sensor_msgs image data
            A Depth image of format sensor_msgs Image with one channel measuring the distance to objects.
        Returns
        -------
        None
        """
        # Convert ROS Image message to OpenCV image
        try:
            self.depth_image = self.depth_bridge.imgmsg_to_cv2(msg)
                            
            # Display image
            if(self.DISPLAY and self.depth_image is not None):
                cv2.imshow("Depth Image", self.depth_image)
                cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

        if(self.depth_image is None):
            self.get_logger().info('Failed to load Depth image')

        # Update state machine
        self.update_state_machine()

    def color_detection(self, img, hsvFrame, kernal, min_width, bounding_boxes, lower_bound, upper_bound, filter, color=None):
        """Apply color filter to RGB image
        ----------
        img : RGB image data
            A RGB image of format numpy array with RGB channels.
        hsvFrame : filename of input image
            The image_filename is format a string with the path image.
        kernal : opencv image kernal 
        min_width : RGB image data
            A minmum width of bounding boxes of format int.
        bounding_boxes : dict
            Dictionary of bounding boxes for colors
        lower_bound : numpy array
            The upper bound for color is format array of numpy array with filter values.
        upper_bound : numpy array
            The upper bound for color is format array of numpy array with filter values.
        color : color to idenity in images. 
            The color is format string. Options include 'red', 'green', 'blue', or None for all colors
        Returns
        -------
        img
            Annotated image with color bounding boxes
        hsvFrame, bounding_boxes, kernal (ass as input)
        """
        # Set upper and lower bound for color detection
        mask = cv2.inRange(hsvFrame, lower_bound, upper_bound)
        mask = cv2.dilate(mask, kernal)

        # Creating contour to track  color
        contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        # Generate color blob contours and draw boxes around them on image
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                if(w < 50 or w > min_width): continue
                bounding_boxes[color].append([x, y, w, h])
                img = cv2.rectangle(img, (x, y), 
                                        (x + w, y + h), 
                                        filter, 2)
                cv2.putText(img, color+" colour", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            filter)  

        return img, hsvFrame, bounding_boxes, kernal

    def read_process_image_color(self, image, image_filename=None, color=None):
        """Read in image or take an image as input, run color filtering for red, green, and blue.
        Parameters
        ----------
        image : RGB image data
            A RGB image of format numpy array with RGB channels.
        image_filename : filename of input image
        The image_filename is format a string with the path image.
        color : color to idenity in images. 
            The color is format string. Options include 'red', 'green', 'blue', or None for all colors
        Returns
        -------
        image
            Image with bounding boxes that show color(s) annotated on the input image
        bounding_boxes
            List of dicts of bounding box coordinates for color blobs with fields 'red', 'green', blue
        """
        # Reading the image
        if(image_filename is None):
            img = image
        else:
            img = cv2.imread(image_filename)
        
        width, height, channels = img.shape
        min_width = width/4
        kernal = np.ones((5, 5), "uint8")
        bounding_boxes = {'red':[], 'green':[], 'blue':[]}
        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set range for red color and define mask
        if(color == 'red' or color is None):
            red_lower = np.array([136, 87, 111], np.uint8)
            red_upper = np.array([180, 255, 255], np.uint8)
            filter = (0, 0, 255)
            img, hsvFrame, bounding_boxes, kernal = self.color_detection(img, 
                                                                    hsvFrame, 
                                                                    kernal, 
                                                                    min_width, 
                                                                    bounding_boxes,  
                                                                    red_lower, 
                                                                    red_upper,
                                                                    filter, 
                                                                    color='red')

        # Set range for green color and 
        # define mask
        if(color == 'green' or color is None):
            green_lower = np.array([25, 52, 72], np.uint8)
            green_upper = np.array([102, 255, 255], np.uint8)
            filter = (0, 255, 0)
            img, hsvFrame, bounding_boxes, kernal = self.color_detection(img, 
                                                                    hsvFrame, 
                                                                    kernal, 
                                                                    min_width, 
                                                                    bounding_boxes, 
                                                                    green_lower, 
                                                                    green_upper, 
                                                                    filter,
                                                                    color='green')

        # Set range for blue color and
        # define mask
        if(color == 'blue' or color is None):
            blue_lower = np.array([94, 80, 2], np.uint8)
            blue_upper = np.array([120, 255, 255], np.uint8)
            filter = (255, 0, 0)
            img, hsvFrame, bounding_boxes, kernal = self.color_detection(img, 
                                                                    hsvFrame, 
                                                                    kernal, 
                                                                    min_width, 
                                                                    bounding_boxes, 
                                                                    blue_lower, 
                                                                    blue_upper, 
                                                                    filter,
                                                                    color='blue')

        return img, bounding_boxes
    
    def detection_object_distance(self):
        """Detects distance to objects in color blobs and alerts user of proxemic zone
        Returns
        x : float
            linear x velocity.
        selected_bbox : array-like
            Boundingbox with [top-left x, top-left y, width, height]
        distance_to_object : float
            mean depth distance to object
        """
        # Initialize variabes
        # Process image data to detect nearby objects; set distance_to_object
        # Compute to average depth pixel distance to nearby objects
        # Use min distance to detect proximitis zones
        selected_bbox = None
        min_dist = float('inf')
        for color in ["red", "green", "blue"]:
            for bbox in self.bboxes[color]:
                img_patch = self.extract_image_patch(self.depth_image, bbox) 
                if(img_patch is not None): 
                    img_patch_mean = np.mean(img_patch)
                    if img_patch_mean < min_dist:
                        min_dist = img_patch_mean
                        selected_bbox = bbox
        return selected_bbox, min_dist
                    

    def update_robot_position(self, x, z, bbox, buffer=10):
        """Update the robot's position based on location of bounding box.
        Parameters
        ----------
        x : float
            linear x velocity.
        z : float
            angualr z velocity.
        bbox : array_like
            The bounding box in format (top-left x, top-left y, width, height).
        buffer : int
            center pixel buffer size 
        Returns
        -------
        None
        """
        # Compute center of bbox and image
        width, height, channels = self.rgb_image.shape
        img_center_line = width/2
        box_center_line = bbox[0]+bbox[2]/2
        print(f"x: {x}, z: {z} , img center line: {img_center_line}, box center line: {box_center_line}")
        # if box on right of center
        if box_center_line > img_center_line and (box_center_line - img_center_line) > buffer:
            self.move_robot(0.0, z, clockwise=True)
            print("Rotate clockwise")
        # elif box on left of center
        elif box_center_line < img_center_line and (img_center_line - box_center_line) > buffer:
            self.move_robot(0.0, z, clockwise=False)
            print("Rotate counter-clockwise")
        # else forward
        else:
            self.move_robot(x, 0, clockwise=True)
            print("Going forward")

    def extract_image_patch(self, image, bbox, patch_shape=(20,20)):
        """Extract image patch from bounding box.
        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.
        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.
        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

    def robot_talker(self, robot_phrase='Welcome to human robot interaction', output_filename='robot_talker.mp3'):
        """Uses text to speech software to enable to robot to 
            alert users when they are in the intimateand public zons                                                    
        ----------
        robot_phrase : robot phrase
            String of text phrase 
        output_filename : name of file to store audio file
            String of outputfile name
        Returns
        -------
        None
        """
        # Language in which you want to convert
        language = 'en'
        
        # Passing the text and language to the engine, 
        # here we have marked slow=False. Which tells 
        # the module that the converted audio should 
        # have a high speed
        myobj = gTTS(text=robot_phrase, lang=language, slow=False)
        
        # Saving the converted audio in a mp3 file named
        # welcome 
        myobj.save(output_filename)

        # Play audio file with playsound library
        playsound.playsound(output_filename, True)



def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # User set parameters
    display=True
    color=None # Options include 'red', 'green', 'blue', or None for all colors
    proxemic_ranges = {'intimate_depth_threshold_min':10,
                        'intimate_depth_threshold_max':20,
                        'public_depth_threshold_min':50,
                        'public_depth_threshold_max':60}

    # Create the node. 
    proxemic_detector = ProxemicDetection(selected_zone, proxemic_ranges, display=display, color=None, robot_speed=robot_speed)

    if(proxemic_detector.curr_state == proxemic_detector.state5):
        proxemic_detector.destroy_node()

    # Spin the node so the callback function is called.
    rclpy.spin(proxemic_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    proxemic_detector.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()
   
if __name__ == '__main__':
  main()