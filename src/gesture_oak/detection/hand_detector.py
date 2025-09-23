# src/gesture_oak/detection/hand_detector.py
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
    MediaPipe-based hand detector for OAK-D using IR mono cameras (fast path).
    Changes for performance:
      - 720p -> 400p mono (fewer pixels, lower latency)
      - Small, non-blocking queues
      - NN threads bumped a bit (2)
      - Very light host-side enhancement
      - Non-blocking tryGet() to avoid stalls (lost fast swipes)
    """

    def __init__(self,
                 fps=30,
                 resolution=(640, 480),
                 pd_score_thresh=0.15,  # Lower for back-of-hand detection
                 pd_nms_thresh=0.3,
                 use_gesture=True,
                 use_rgb=True):  # Keep switch, though we use IR path
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

        # IR Mono Cameras (Left and Right) - optimized for low light
        print("Setting up IR mono cameras for dark environment detection...")
        cam_left = pipeline.createMonoCamera()
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # lighter than 720p
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setFps(min(self.fps_target, 30))

        cam_right = pipeline.createMonoCamera()
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setFps(min(self.fps_target, 30))

        # Depth (optional; used for filtering if present)
        depth = pipeline.createStereoDepth()
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(False)
        cam_left.out.link(depth.left)
        cam_right.out.link(depth.right)

        # Convert mono to RGB888p (for NN that expects 3 channels)
        mono_to_rgb = pipeline.createImageManip()
        mono_to_rgb.initialConfig.setResize(self.img_w, self.img_h)
        mono_to_rgb.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        cam_left.out.link(mono_to_rgb.inputImage)

        # Depth out
        depth_out = pipeline.createXLinkOut()
        depth_out.setStreamName("depth_out")
        depth.depth.link(depth_out.input)

        # Camera (RGB) out
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        mono_to_rgb.out.link(cam_out.input)

        # Manager script node
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        # Palm detection preproc
        print("Setting up palm detection preprocessing...")
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length * self.pd_input_length * 3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        mono_to_rgb.out.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

        # Palm detection NN
        print("Setting up palm detection neural network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        pd_nn.setNumInferenceThreads(2)
        pre_pd_manip.out.link(pd_nn.input)

        # Palm postprocess
        print("Setting up palm detection post processing...")
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])

        # Landmark preproc
        print("Setting up hand landmark preprocessing...")
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length * self.lm_input_length * 3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        mono_to_rgb.out.link(pre_lm_manip.inputImage)
        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)

        # Landmark NN
        print("Setting up hand landmark neural network...")
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(2)
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])

        # Manager -> host
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs['host'].link(manager_out.input)

        print("Pipeline created successfully.")
        return pipeline

    def build_manager_script(self):
        """Build the manager script from template"""
        with open(self.template_script, 'r', encoding='utf-8') as file:
            template = Template(file.read())

        code = template.substitute(
            _TRACE1="#",
            _TRACE2="#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=0.1,          # lenient for IR/back-of-hand
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w,
            _frame_size=self.frame_size,
            _crop_w=0,
            _IF_XYZ='"""',
            _IF_USE_HANDEDNESS_AVERAGE='"""',
            _single_hand_tolerance_thresh=15,
            _IF_USE_SAME_IMAGE='"""',
            _IF_USE_WORLD_LANDMARKS='"""',
        )

        # Light cleanup: strip triple-quoted blocks and comments
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

            # Output queues (small & non-blocking = lower latency)
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=3, blocking=False)
            self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=3, blocking=False)
            self.q_depth = self.device.getOutputQueue(name="depth_out", maxSize=3, blocking=False)

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

            # Hand center in depth map coords
            center_x = int(hand.rect_x_center_a * depth_w / self.img_w)
            center_y = int(hand.rect_y_center_a * depth_h / self.img_h)

            if 0 <= center_x < depth_w and 0 <= center_y < depth_h:
                hand_depth = depth_frame[center_y, center_x]

                # Reasonable hand distance (in mm)
                if 300 < hand_depth < 1500:
                    # Average depth in a small region
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

                        # More lenient consistency for back-of-hand
                        if depth_std < 150:
                            hand.depth = avg_depth
                            hand.depth_confidence = 1.0 - (depth_std / 150.0)
                            filtered_hands.append(hand)

        return filtered_hands

    def enhance_ir_frame(self, frame):
        """Minimal IR frame enhancement to preserve hand features (fast)."""
        # If RGB, to gray; if already single channel, use as-is
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Fast contrast stretch (avoid expensive CLAHE+bilateral in hot path)
        # Normalize to [0,255] with a gentle clip
        low, high = np.percentile(gray, (2.0, 98.0))
        if high - low > 1e-3:
            norm = np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        else:
            norm = gray

        # Sharpen a bit to help keypoints
        norm = cv2.GaussianBlur(norm, (0, 0), 1.0)
        norm = cv2.addWeighted(gray, 1.5, norm, -0.5, 0)

        # Back to 3-channel RGB for downstream blocks
        enhanced_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        return enhanced_rgb

    def get_frame_and_hands(self):
        """
        Unified accessor that returns (frame, hands, depth_frame).
        Non-blocking queue reads + robust against transient bad detections.
        """
        try:
            try:
                self.fps_counter.update()
            except Exception:
                pass

            # Camera frame (non-blocking)
            in_video = self.q_video.tryGet() if self.q_video is not None else None
            if in_video is None:
                return None, [], None
            raw_frame = in_video.getCvFrame()

            # Light IR enhancement
            frame = self.enhance_ir_frame(raw_frame)

            # Depth (non-blocking)
            depth_frame = None
            if self.q_depth is not None:
                in_depth = self.q_depth.tryGet()
                if in_depth is not None:
                    depth_frame = in_depth.getFrame()

            # Manager/script output (non-blocking)
            in_mgr = self.q_manager_out.tryGet() if self.q_manager_out is not None else None
            hands = []
            if in_mgr is not None:
                res = marshal.loads(in_mgr.getData())
                lm_scores = res.get("lm_score", [])
                for i in range(len(lm_scores)):
                    try:
                        hand = self.extract_hand_data(res, i)
                        if hand is not None:
                            hands.append(hand)
                    except Exception:
                        # Skip malformed hand; keep others
                        continue

            # Optional depth-based filtering
            if depth_frame is not None and hands:
                hands = self.filter_hands_by_depth(hands, depth_frame)

            # Tracking continuity (don’t let these throw)
            if hands:
                self.frames_without_detection = 0
                self.last_hand_positions = [(h.rect_x_center_a, h.rect_y_center_a) for h in hands]
            else:
                self.frames_without_detection += 1

            return frame, hands, depth_frame

        except Exception as e:
            # Compact log to avoid flooding
            print(f"Error getting frame and hands: {e}")
            return None, [], None

    def close(self):
        """Close device connection"""
        if self.device:
            self.device.close()
            self.device = None
        print("Hand detector closed.")
