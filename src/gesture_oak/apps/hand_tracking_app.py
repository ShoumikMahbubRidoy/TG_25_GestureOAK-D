#!/usr/bin/env python3

import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # let OpenCV choose best for this process
import numpy as np
from ..detection.hand_detector import HandDetector
from ..detection.swipe_detector import SwipeDetector
from ..logic.gesture_classifier import classify_hand  # NEW

def draw_hand_landmarks(frame, hand, gesture_info=None):
    """Draw hand landmarks, bbox, labels, and (NEW) gesture info"""
    # Landmarks
    if hasattr(hand, 'landmarks') and hand.landmarks is not None:
        for idx, landmark in enumerate(hand.landmarks):
            x, y = int(landmark[0]), int(landmark[1])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            if idx in [0, 4, 8, 12, 16, 20]:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    # BBox
    if hasattr(hand, 'rect_points') and hand.rect_points is not None:
        points = np.array(hand.rect_points, dtype=np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 255), 2)

    # Label/conf/depth
    if hasattr(hand, 'label') and hasattr(hand, 'lm_score'):
        depth_info = ""
        if hasattr(hand, 'depth'):
            depth_info = f" D:{hand.depth:.0f}mm"
        if hasattr(hand, 'depth_confidence'):
            depth_info += f" C:{hand.depth_confidence:.2f}"
        label_text = f"{hand.label}: {hand.lm_score:.2f}{depth_info}"

        if hasattr(hand, 'rect_x_center_a'):
            x = int(hand.rect_x_center_a - hand.rect_w_a // 2)
            y = int(hand.rect_y_center_a - hand.rect_h_a // 2 - 10)
            cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # (NEW) Gesture text block
    if gesture_info and hasattr(hand, 'rect_x_center_a'):
        gx = int(hand.rect_x_center_a - hand.rect_w_a // 2)
        gy = int(hand.rect_y_center_a + hand.rect_h_a // 2 + 20)
        gesture_line = f"Gesture: {gesture_info['gesture']}"
        fingers_line = "Up: " + ",".join(gesture_info["fingers_up_list"]) if gesture_info["fingers_up_list"] else "Up: none"
        cv2.putText(frame, gesture_line, (gx, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, fingers_line, (gx, gy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

def main():
    print("OAK-D Hand Detection Demo with Swipe Detection")
    print("=" * 45)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to reset swipe statistics")

    # Detector (same defaults you used for IR; leave use_rgb=False if you want IR path)
    detector = HandDetector(
        fps=30,
        resolution=(640, 480),
        pd_score_thresh=0.1,
        use_gesture=True,
        use_rgb=False
    )

    # Swipe detector (your relaxed params; UDP is still fired inside SwipeDetector on confirm)
    swipe_detector = SwipeDetector(
        buffer_size=12,
        min_distance=80,
        min_duration=0.2,
        max_duration=3.0,
        min_velocity=30,
        max_velocity=1000,
        max_y_deviation=0.5
    )

    if not detector.connect():
        print("Failed to connect to OAK-D device")
        return

    print("Hand detection started. Showing live preview...")

    frame_count = 0
    last_swipe_alert = 0

    try:
        while True:
            frame, hands, depth_frame = detector.get_frame_and_hands()
            if frame is None:
                continue

            frame_count += 1

            # Choose a hand center for swipe detection (first hand)
            hand_center = None
            if hands:
                h0 = hands[0]
                if hasattr(h0, 'rect_x_center_a') and hasattr(h0, 'rect_y_center_a'):
                    hand_center = (h0.rect_x_center_a, h0.rect_y_center_a)

            swipe_detected = swipe_detector.update(hand_center)

            # FPS
            fps = detector.fps_counter.get()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Counts, depth status
            cv2.putText(frame, f"Hands: {len(hands)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Depth: ON" if depth_frame is not None else "Depth: OFF",
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if depth_frame is not None else (0, 0, 255), 2)

            # Tips when empty
            if len(hands) == 0:
                cv2.putText(frame, "IR Mode: Move hand slowly in front of camera",
                            (frame.shape[1] // 2 - 220, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Works best in dark environments (80-160 cm)",
                            (frame.shape[1] // 2 - 220, frame.shape[0] // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Swipe stats
            stats = swipe_detector.get_statistics()
            cv2.putText(frame, f"Swipes: {stats['total_swipes_detected']}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Swipe progress
            progress = swipe_detector.get_current_swipe_progress()
            if progress:
                state_color = (0, 255, 255) if progress['state'] != 'idle' else (128, 128, 128)
                cv2.putText(frame, f"State: {progress['state']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
                if progress['distance'] > 0:
                    cv2.putText(frame, f"Distance: {progress['distance']:.0f}px (need: 80px)", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                    cv2.putText(frame, f"Velocity: {progress['velocity']:.0f}px/s (need: 30-1000)", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                    cv2.putText(frame, f"Progress: {progress['progress']:.1%}", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            # Swipe visual feedback
            if swipe_detected:
                last_swipe_alert = frame_count
                print(f"🚀 LEFT-TO-RIGHT SWIPE DETECTED! (Total: {stats['total_swipes_detected']})")

            if frame_count - last_swipe_alert < 90:  # ~3s at 30fps
                cv2.putText(frame, "SWIPE DETECTED!", (frame.shape[1] // 2 - 110, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.arrowedLine(frame, (frame.shape[1] // 2 - 60, 80),
                                (frame.shape[1] // 2 + 60, 80), (0, 255, 0), 5)

            # Draw each hand + (NEW) classify & overlay gesture
            for i, hand in enumerate(hands):
                gesture_info = None
                if hasattr(hand, "landmarks") and hand.landmarks is not None and len(hand.landmarks) == 21:
                    try:
                        # hand.handedness: >0.5 => 'right' per your extractor
                        handedness = getattr(hand, "handedness", 0.5)
                        gesture_info = classify_hand(np.asarray(hand.norm_landmarks if hasattr(hand, "norm_landmarks") else hand.landmarks),
                                                     handedness=handedness)
                        # Attach for other consumers if needed
                        hand.gesture = gesture_info["gesture"]
                        hand.fingers_up = gesture_info["fingers_up"]
                    except Exception:
                        gesture_info = None

                draw_hand_landmarks(frame, hand, gesture_info)
                # Optional debug log every few frames
                # if i == 0 and (frame_count % 15 == 0) and gesture_info:
                #     print(f"[Hand {i+1}] {gesture_info}")

            cv2.imshow("OAK-D Hand Detection with Swipe + Gestures", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"hand_detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                swipe_detector.reset_statistics()
                print("Swipe statistics reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        detector.close()
        cv2.destroyAllWindows()

        total_fps = detector.fps_counter.get_global()
        final_stats = swipe_detector.get_statistics()
        print(f"\nSession Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {total_fps:.2f}")
        print(f"Total swipes detected: {final_stats['total_swipes_detected']}")
        print(f"False positives filtered: {final_stats['false_positives_filtered']}")
        print("Hand detection demo completed.")

if __name__ == "__main__":
    main()
