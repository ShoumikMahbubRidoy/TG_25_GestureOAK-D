import numpy as np
import depthai as dai
import cv2
from ..utils import mediapipe_utils as mpu
from pathlib import Path
from ..utils.FPS import FPS
import marshal
from string import Template


class HandDetector:
    """
    Simplified MediaPipe-based hand detector for OAK-D using IR stereo camera
    """
    
    def __init__(self, 
                 fps=30, 
                 resolution=(640, 480),
                 pd_score_thresh=0.15,  # Lower for back-of-hand detection
                 pd_nms_thresh=0.3,
                 use_gesture=True,
                 use_rgb=True):  # Add RGB camera option
        
        self.fps_target = fps
        self.resolution = resolution
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_gesture = use_gesture
        
        # Model paths
        project_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.pd_model = str(project_dir / "models/palm_detection_sh4.blob")
        self.lm_model = str(project_dir / "models/hand_landmark_lite_sh4.blob")
        self.pp_model = str(project_dir / "models/PDPostProcessing_top2_sh1.blob")
        self.template_script = str(project_dir / "src/gesture_oak/utils/template_manager_script_solo.py")
        
        # Pipeline components
        self.device = None
        self.pipeline = None
        self.q_video = None
        self.q_manager_out = None
        self.q_depth = None
        
        # Frame processing parameters
        self.pd_input_length = 128
        self.lm_input_length = 224
        self.frame_size = max(resolution)
        self.img_w, self.img_h = resolution
        self.pad_h = (self.frame_size - self.img_h) // 2 if self.frame_size > self.img_h else 0
        self.pad_w = (self.frame_size - self.img_w) // 2 if self.frame_size > self.img_w else 0
        
        self.fps_counter = FPS()
        
        # Hand tracking continuity
        self.last_hand_positions = []
        self.frames_without_detection = 0
        self.max_frames_without_detection = 5
        
    def setup_pipeline(self):
        """Create DepthAI pipeline for hand detection"""
        print("Creating hand detection pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
        
        # IR Stereo Cameras (Left and Right) - Optimized for dark environments
        print("Setting up IR stereo cameras for dark environment detection...")
        cam_left = pipeline.createMonoCamera()
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setFps(min(self.fps_target, 30))  # Reduced FPS for better IR sensitivity
        
        cam_right = pipeline.createMonoCamera()
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setFps(min(self.fps_target, 30))
        
        # Create depth perception
        depth = pipeline.createStereoDepth()
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(False)
        cam_left.out.link(depth.left)
        cam_right.out.link(depth.right)
        
        # Convert mono to RGB format with enhancement for dark environments
        mono_to_rgb = pipeline.createImageManip()
        mono_to_rgb.initialConfig.setResize(self.img_w, self.img_h)
        mono_to_rgb.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        # Apply gamma correction and contrast enhancement for better IR detection
        mono_to_rgb.initialConfig.setColormap(dai.Colormap.NONE, 128)  # Neutral mapping
        cam_left.out.link(mono_to_rgb.inputImage)
        
        # Depth output
        depth_out = pipeline.createXLinkOut()
        depth_out.setStreamName("depth_out")
        depth.depth.link(depth_out.input)
        
        # Camera output
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        mono_to_rgb.out.link(cam_out.input)
        
        # Manager script node
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())
        
        # Palm detection preprocessing
        print("Setting up palm detection preprocessing...")
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length * self.pd_input_length * 3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        mono_to_rgb.out.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)
        
        # Palm detection neural network
        print("Setting up palm detection neural network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        pd_nn.setNumInferenceThreads(2)
        pre_pd_manip.out.link(pd_nn.input)
        
        # Palm detection post processing
        print("Setting up palm detection post processing...")
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])
        
        # Hand landmark preprocessing
        print("Setting up hand landmark preprocessing...")
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length * self.lm_input_length * 3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        mono_to_rgb.out.link(pre_lm_manip.inputImage)
        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)
        
        # Hand landmark neural network
        print("Setting up hand landmark neural network...")
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(2)
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])
        
        # Manager output
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs['host'].link(manager_out.input)
        
        print("Pipeline created successfully.")
        return pipeline
    
    def build_manager_script(self):
        """Build the manager script from template"""
        with open(self.template_script, 'r') as file:
            template = Template(file.read())
        
        code = template.substitute(
            _TRACE1="#",
            _TRACE2="#",
            _pd_score_thresh=0.1,  # Very low for IR detection
            _lm_score_thresh=0.1,  # Very low for IR back-of-hand detection
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w,
            _frame_size=self.frame_size,
            _crop_w=0,
            _IF_XYZ='"""',
            _IF_USE_HANDEDNESS_AVERAGE='"""',
            _single_hand_tolerance_thresh=15,  # More tolerant for IR
            _IF_USE_SAME_IMAGE='"""',
            _IF_USE_WORLD_LANDMARKS='"""',
        )
        
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\n\s*\n', '\n', code)
        
        return code
    
    def connect(self):
        """Connect to OAK-D device and start pipeline"""
        try:
            self.pipeline = self.setup_pipeline()
            self.device = dai.Device(self.pipeline)
            
            print(f"Connected to device: {self.device.getDeviceName()}")
            print(f"USB Speed: {self.device.getUsbSpeed()}")
            
            # Setup output queues
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=4, blocking=False)
            self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=4, blocking=False)
            self.q_depth = self.device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to OAK-D: {e}")
            return False
    
    def extract_hand_data(self, res, hand_idx):
        """Extract hand data from inference results"""
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx]
        hand.rect_points = mpu.rotated_rect_to_points(
            hand.rect_x_center_a, hand.rect_y_center_a, 
            hand.rect_w_a, hand.rect_h_a, hand.rotation
        )
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1, 3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1, 2).astype(np.int32)
        
        # Remove padding if applied
        if self.pad_h > 0:
            hand.landmarks[:, 1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:, 0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w
        
        if self.use_gesture:
            mpu.recognize_gesture(hand)
        
        return hand
    
    def filter_hands_by_depth(self, hands, depth_frame):
        """Filter and improve hand detection using depth information"""
        if depth_frame is None or len(hands) == 0:
            return hands
        
        filtered_hands = []
        depth_h, depth_w = depth_frame.shape
        
        for hand in hands:
            if not hasattr(hand, 'landmarks') or hand.landmarks is None:
                continue
                
            # Get hand center position
            center_x = int(hand.rect_x_center_a * depth_w / self.img_w)
            center_y = int(hand.rect_y_center_a * depth_h / self.img_h)
            
            # Check if coordinates are within bounds
            if 0 <= center_x < depth_w and 0 <= center_y < depth_h:
                # Get depth at hand center
                hand_depth = depth_frame[center_y, center_x]
                
                # Filter by reasonable hand distance (30cm to 150cm)
                if 300 < hand_depth < 1500:  # depth in mm
                    # Calculate average depth around hand region
                    region_size = 20
                    y1 = max(0, center_y - region_size)
                    y2 = min(depth_h, center_y + region_size)
                    x1 = max(0, center_x - region_size)
                    x2 = min(depth_w, center_x + region_size)
                    
                    region_depths = depth_frame[y1:y2, x1:x2]
                    valid_depths = region_depths[region_depths > 0]
                    
                    if len(valid_depths) > 0:
                        avg_depth = np.mean(valid_depths)
                        depth_std = np.std(valid_depths)
                        
                        # More lenient depth consistency for back-of-hand
                        if depth_std < 150:  # mm - more tolerant for back-of-hand
                            hand.depth = avg_depth
                            hand.depth_confidence = 1.0 - (depth_std / 150.0)
                            filtered_hands.append(hand)
        
        return filtered_hands

    def enhance_ir_frame(self, frame):
        """Minimal IR frame enhancement to preserve hand features"""
        # Convert to grayscale if RGB (IR camera should already be grayscale-like)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Simple contrast enhancement - preserve hand features
        # Use adaptive threshold to enhance contrast without losing details
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Light noise reduction - preserve edges and features
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # Convert to RGB format for MediaPipe (simple grayscale replication)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb

    def get_frame_and_hands(self):
        """Get frame, depth, and detected hands with depth filtering"""
        try:
            self.fps_counter.update()
            
            # Get video frame
            in_video = self.q_video.get()
            raw_frame = in_video.getCvFrame()
            
            # Enhance IR frame for better detection
            frame = self.enhance_ir_frame(raw_frame)
            
            # Get depth frame
            depth_frame = None
            if self.q_depth.has():
                in_depth = self.q_depth.get()
                depth_frame = in_depth.getFrame()
            
            # Get hand detection results
            res = marshal.loads(self.q_manager_out.get().getData())
            hands = []
            
            for i in range(len(res.get("lm_score", []))):
                hand = self.extract_hand_data(res, i)
                hands.append(hand)
            
            # Filter hands using depth information to reduce false positives
            if depth_frame is not None:
                hands = self.filter_hands_by_depth(hands, depth_frame)
            
            # Update hand tracking continuity
            if hands:
                self.frames_without_detection = 0
                self.last_hand_positions = [(hand.rect_x_center_a, hand.rect_y_center_a) for hand in hands]
            else:
                self.frames_without_detection += 1
                
            # If no hands detected but we had recent detections, provide hints
            if not hands and self.frames_without_detection <= self.max_frames_without_detection and self.last_hand_positions:
                # Keep tracking info available for continuity
                pass
            
            return frame, hands, depth_frame
            
        except Exception as e:
            print(f"Error getting frame and hands: {e}")
            return None, [], None
    
    def close(self):
        """Close device connection"""
        if self.device:
            self.device.close()
            self.device = None
        print("Hand detector closed.")