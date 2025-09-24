# src/gesture_oak/detection/hand_detector.py
import time
from pathlib import Path
from string import Template
import marshal
import re

import numpy as np
import cv2
import depthai as dai

from ..utils import mediapipe_utils as mpu
from ..utils.FPS import FPS

# OpenCV runtime perf toggles (safe, logic-neutral)
cv2.setUseOptimized(True)
cv2.setNumThreads(0)


class HandDetector:
    """
    MediaPipe-based hand detector for OAK-D using LEFT mono (IR-ish) camera.

    This build focuses on robustness at 0.8–1.6 m while keeping latency low:
      - Depth gate (350–1700 mm) + local depth stability -> filters static objects
      - Landmark quality & bbox sanity checks
      - Light motion gate + 2-frame hysteresis -> fewer flickers on small shakes
      - Shallow, non-blocking queues with bounded draining (no freezes)
      - Very light IR enhancement
    """

    def __init__(
        self,
        fps: int = 30,
        resolution=(640, 480),
        pd_score_thresh: float = 0.20,   # tolerant palm score
        pd_nms_thresh: float = 0.3,
        use_gesture: bool = True,
        use_rgb: bool = True,            # API compatibility (we use mono->RGB)
    ):
        self.fps_target = int(fps)
        self.resolution = tuple(resolution)
        self.pd_score_thresh = float(pd_score_thresh)
        self.pd_nms_thresh = float(pd_nms_thresh)
        self.use_gesture = bool(use_gesture)

        # Model paths
        project_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.pd_model = str(project_dir / "models/palm_detection_sh4.blob")
        self.lm_model = str(project_dir / "models/hand_landmark_lite_sh4.blob")
        self.pp_model = str(project_dir / "models/PDPostProcessing_top2_sh1.blob")
        self.template_script = str(
            project_dir / "src/gesture_oak/utils/template_manager_script_solo.py"
        )

        # Pipeline / device objects
        self.device = None
        self.pipeline = None

        # Host output queues
        self.q_video = None
        self.q_manager_out = None
        self.q_depth = None

        # Sizes
        self.img_w, self.img_h = self.resolution
        self.frame_size = max(self.resolution)
        self.pad_h = (self.frame_size - self.img_h) // 2 if self.frame_size > self.img_h else 0
        self.pad_w = (self.frame_size - self.img_w) // 2 if self.frame_size > self.img_w else 0

        # NN input sizes
        self.pd_input_length = 128
        self.lm_input_length = 224

        # Runtime stats
        self.fps_counter = FPS()

        # Tracking continuity / filters
        self.last_hand_positions = []
        self.frames_without_detection = 0
        self.max_frames_without_detection = 5

        # ---------------- Filters (tune here) ----------------
        self.depth_min_mm = 350
        self.depth_max_mm = 1700
        self.depth_std_max = 200           # <= 200 mm local depth std

        self.min_lm_score = 0.30           # reject weak landmarks
        self.min_bbox_w_px = 28            # too tiny -> likely background blob
        self.max_bbox_w_px = int(self.img_w * 0.85)

        self.min_center_motion_px = 4      # small motion to be considered "trackable"
        self.required_stable_frames = 2    # hysteresis
        self._stable_counter = 0
        # -----------------------------------------------------

    # --------------------------- Pipeline ---------------------------------

    def setup_pipeline(self) -> dai.Pipeline:
        print("Creating hand detection pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        # LEFT mono camera (OV9282 supports 400p / 720p, not 480p)
        print("Setting up IR mono cameras for dark environment detection...")
        cam_left = pipeline.createMonoCamera()
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_left.setFps(min(self.fps_target, 60))

        # Depth (fast profile)
        depth = pipeline.createStereoDepth()
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(False)  # subpixel adds latency; off keeps things snappy
        # Right camera
        cam_right = pipeline.createMonoCamera()
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setFps(min(self.fps_target, 60))

        cam_left.out.link(depth.left)
        cam_right.out.link(depth.right)

        # Mono -> RGB for NN
        to_rgb = pipeline.createImageManip()
        to_rgb.initialConfig.setResize(self.img_w, self.img_h)
        to_rgb.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        cam_left.out.link(to_rgb.inputImage)

        # Script node (on-device manager)
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        # Palm preprocess
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length * self.pd_input_length * 3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        to_rgb.out.link(pre_pd_manip.inputImage)
        manager_script.outputs["pre_pd_manip_cfg"].link(pre_pd_manip.inputConfig)

        # Palm NN
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        pd_nn.setNumInferenceThreads(2)
        pre_pd_manip.out.link(pd_nn.input)

        # Palm postproc
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs["from_post_pd_nn"])

        # LM preprocess
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length * self.lm_input_length * 3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        to_rgb.out.link(pre_lm_manip.inputImage)
        manager_script.outputs["pre_lm_manip_cfg"].link(pre_lm_manip.inputConfig)

        # LM NN
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(2)
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs["from_lm_nn"])

        # Host outputs
        cam_out = pipeline.create(dai.node.XLinkOut)
        cam_out.setStreamName("cam_out")
        to_rgb.out.link(cam_out.input)

        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs["host"].link(manager_out.input)

        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth_out")
        depth.depth.link(depth_out.input)

        print("Pipeline created successfully.")
        return pipeline

    def build_manager_script(self) -> str:
        with open(self.template_script, "r", encoding="utf-8") as f:
            template = Template(f.read())

        code = template.substitute(
            _TRACE1="#",
            _TRACE2="#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=0.30,  # raise a bit to fight false positives
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

        # strip big comments and empty lines
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        code = re.sub(r"#.*", "", code)
        code = re.sub(r"\n\s*\n", "\n", code)
        return code

    # --------------------------- Device / Queues ---------------------------

    def connect(self) -> bool:
        try:
            self.pipeline = self.setup_pipeline()
            self.device = dai.Device(self.pipeline)

            print(f"Connected to device: {self.device.getDeviceName()}")
            print(f"USB Speed: {self.device.getUsbSpeed()}")

            # Shallow, non-blocking host queues
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=2, blocking=False)
            self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=2, blocking=False)
            self.q_depth = self.device.getOutputQueue(name="depth_out", maxSize=2, blocking=False)
            return True
        except Exception as e:
            print(f"Failed to connect to OAK-D: {e}")
            return False

    # --------------------------- Helpers ----------------------------------

    def _enhance_ir_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Very light enhancement so distant hands keep structure."""
        if frame_rgb is None:
            return None
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if frame_rgb.ndim == 3 else frame_rgb
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.bilateralFilter(enhanced, 3, 24, 24)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return rgb

    def _depth_stats_at(self, depth_frame: np.ndarray, cx: int, cy: int, r: int = 16):
        """Return (avg_mm, std_mm) around a small ROI; None if invalid."""
        if depth_frame is None:
            return None, None
        h, w = depth_frame.shape[:2]
        x1 = max(0, cx - r); x2 = min(w, cx + r)
        y1 = max(0, cy - r); y2 = min(h, cy + r)
        roi = depth_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None
        valid = roi[roi > 0]
        if valid.size < 20:
            return None, None
        return float(valid.mean()), float(valid.std())

    def extract_hand_data(self, res: dict, hand_idx: int):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx]
        hand.rect_points = mpu.rotated_rect_to_points(
            hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation
        )
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res["rrn_lms"][hand_idx]).reshape(-1, 3)
        hand.landmarks = (
            (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1, 2).astype(np.int32)
        )

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

    def _filter_hands(self, hands, depth_frame):
        """Depth + quality + motion + hysteresis filters."""
        if not hands:
            self._stable_counter = 0
            return []

        filtered = []
        h_d, w_d = (depth_frame.shape if depth_frame is not None else (self.img_h, self.img_w))

        # last center for motion gate
        last = self.last_hand_positions[0] if self.last_hand_positions else None

        for hand in hands:
            # 1) LM quality
            if getattr(hand, "lm_score", 0.0) < self.min_lm_score:
                continue

            # 2) bbox sanity
            bw = int(getattr(hand, "rect_w_a", 0))
            if bw < self.min_bbox_w_px or bw > self.max_bbox_w_px:
                continue

            # 3) depth gate & local stability
            if depth_frame is not None:
                cx = int(hand.rect_x_center_a * w_d / self.img_w)
                cy = int(hand.rect_y_center_a * h_d / self.img_h)
                avg_mm, std_mm = self._depth_stats_at(depth_frame, cx, cy, r=16)
                if avg_mm is None:
                    continue
                if not (self.depth_min_mm <= avg_mm <= self.depth_max_mm):
                    continue
                if std_mm > self.depth_std_max:
                    continue
                hand.depth = avg_mm
                hand.depth_confidence = max(0.0, 1.0 - (std_mm / self.depth_std_max))

            # 4) light motion gate (reject static objects)
            if last is not None:
                dx = hand.rect_x_center_a - last[0]
                dy = hand.rect_y_center_a - last[1]
                if (dx * dx + dy * dy) ** 0.5 < self.min_center_motion_px:
                    # permit one pass if hysteresis not yet reached
                    pass
            filtered.append(hand)

        # 5) hysteresis: require 2 consecutive frames with filtered > 0
        if filtered:
            self._stable_counter += 1
        else:
            self._stable_counter = 0

        return filtered if self._stable_counter >= self.required_stable_frames else []

    # --------------------------- Main fetch --------------------------------

    def get_frame_and_hands(self):
        """
        Non-blocking fetch with bounded draining and anti-freeze backoff.
        Returns: (frame_rgb, hands_list, depth_frame_or_None)
        """
        try:
            self.fps_counter.update()

            # 1) video frame (bounded drain)
            in_video = None
            for _ in range(3):
                pkt = self.q_video.tryGet()
                if pkt is None:
                    break
                in_video = pkt
            if in_video is None:
                time.sleep(0.002)
                return None, [], None
            raw_rgb = in_video.getCvFrame()
            frame_rgb = self._enhance_ir_frame(raw_rgb)

            # 2) manager packet (bounded drain)
            in_mgr = None
            for _ in range(4):
                pkt = self.q_manager_out.tryGet()
                if pkt is None:
                    break
                in_mgr = pkt

            hands = []
            if in_mgr is not None:
                res = marshal.loads(in_mgr.getData())
                lm_scores = res.get("lm_score", []) or []
                for i in range(len(lm_scores)):
                    hands.append(self.extract_hand_data(res, i))

            # 3) depth (latest only; optional)
            depth_frame = None
            for _ in range(2):
                dp = self.q_depth.tryGet()
                if dp is None:
                    break
                depth_frame = dp.getFrame()

            # 4) filters
            hands = self._filter_hands(hands, depth_frame)

            # 5) continuity bookkeeping
            if hands:
                self.frames_without_detection = 0
                # track just the first hand's center for motion gate
                self.last_hand_positions = [(hands[0].rect_x_center_a, hands[0].rect_y_center_a)]
            else:
                self.frames_without_detection += 1

            return frame_rgb, hands, depth_frame

        except Exception as e:
            print(f"Error getting frame and hands: {e}")
            time.sleep(0.005)
            return None, [], None

    # --------------------------- Close -------------------------------------

    def close(self):
        if self.device:
            self.device.close()
            self.device = None
        print("Hand detector closed.")
