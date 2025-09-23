#!/usr/bin/env python3

import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # let OpenCV choose best for this process
import numpy as np
from ..detection.hand_detector import HandDetector
from ..detection.swipe_detector import SwipeDetector
from ..utils import mediapipe_utils as mpu


def draw_hand_landmarks(frame, hand):
    """Draw hand landmarks and bounding box on frame"""
    # Draw landmarks
    if hasattr(hand, 'landmarks') and hand.landmarks is not None:
        for idx, landmark in enumerate(hand.landmarks):
            x, y = int(landmark[0]), int(landmark[1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if idx in [0, 4, 8, 12, 16, 20]:  # Fingertips and wrist
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
    
    # Draw bounding box
    if hasattr(hand, 'rect_points') and hand.rect_points is not None:
        points = np.array(hand.rect_points, dtype=np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 255), 2)
    
    # Draw label and confidence with depth info
    if hasattr(hand, 'label') and hasattr(hand, 'lm_score'):
        depth_info = ""
        if hasattr(hand, 'depth'):
            depth_info = f" D:{hand.depth:.0f}mm"
        if hasattr(hand, 'depth_confidence'):
            depth_info += f" C:{hand.depth_confidence:.2f}"
            
        label_text = f"{hand.label}: {hand.lm_score:.2f}{depth_info}"
        if hasattr(hand, 'rect_x_center_a'):
            x = int(hand.rect_x_center_a - hand.rect_w_a//2)
            y = int(hand.rect_y_center_a - hand.rect_h_a//2 - 10)
            cv2.putText(frame, label_text, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw gesture if available
    if hasattr(hand, 'gesture') and hand.gesture is not None:
        if hasattr(hand, 'rect_x_center_a'):
            x = int(hand.rect_x_center_a - hand.rect_w_a//2)
            y = int(hand.rect_y_center_a + hand.rect_h_a//2 + 20)
            cv2.putText(frame, f"Gesture: {hand.gesture}", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def main():
    print("OAK-D Hand Detection Demo with Swipe Detection")
    print("=" * 45)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to reset swipe statistics")
    
    # Initialize hand detector optimized for IR detection in dark environments
    detector = HandDetector(
        fps=30,
        resolution=(640, 480),
        pd_score_thresh=0.1,  # Very low for IR detection
        use_gesture=True,
        use_rgb=False  # Force IR camera for dark environments
    )
    
    # Initialize swipe detector with relaxed parameters for easier detection
    swipe_detector = SwipeDetector(
        buffer_size=12,        # Smaller buffer for faster response
        min_distance=80,       # Reduced minimum distance
        min_duration=0.2,      # Shorter minimum duration
        max_duration=3.0,      # Longer maximum duration
        min_velocity=30,       # Lower minimum velocity
        max_velocity=1000,     # Higher maximum velocity
        max_y_deviation=0.5    # Allow more Y deviation
    )
    
    # Connect to device
    if not detector.connect():
        print("Failed to connect to OAK-D device")
        return
    
    print("Hand detection started. Showing live preview...")
    
    frame_count = 0
    last_swipe_alert = 0
    
    try:
        while True:
            # Get frame, depth, and hand detections
            frame, hands, depth_frame = detector.get_frame_and_hands()
            
            if frame is None:
                continue
                
            frame_count += 1
            
            # スワイプ検出用の手の中心位置を取得
            hand_center = None
            if hands:
                # 最初の手の中心位置を使用
                hand = hands[0]
                if hasattr(hand, 'rect_x_center_a') and hasattr(hand, 'rect_y_center_a'):
                    hand_center = (hand.rect_x_center_a, hand.rect_y_center_a)
            
            # スワイプ検出を更新
            swipe_detected = swipe_detector.update(hand_center)
            
            # Draw FPS
            fps = detector.fps_counter.get()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw number of detected hands and depth status
            cv2.putText(frame, f"Hands: {len(hands)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            depth_status = "Depth: ON" if depth_frame is not None else "Depth: OFF"
            cv2.putText(frame, depth_status, (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if depth_frame is not None else (0, 0, 255), 2)
            
            # Show detection tips when no hands are detected
            if len(hands) == 0:
                cv2.putText(frame, "IR Mode: Move hand slowly in front of camera", 
                           (frame.shape[1]//2 - 220, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Works best in dark environments (30-150cm)", 
                           (frame.shape[1]//2 - 200, frame.shape[0]//2 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # スワイプ統計表示
            stats = swipe_detector.get_statistics()
            cv2.putText(frame, f"Swipes: {stats['total_swipes_detected']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # スワイプ進行状況表示（詳細）
            progress = swipe_detector.get_current_swipe_progress()
            if progress:
                state_color = (0, 255, 255) if progress['state'] != 'idle' else (128, 128, 128)
                cv2.putText(frame, f"State: {progress['state']}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
                if progress['distance'] > 0:
                    cv2.putText(frame, f"Distance: {progress['distance']:.0f}px (need: 80px)", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                    cv2.putText(frame, f"Velocity: {progress['velocity']:.0f}px/s (need: 30-1000)", (10, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                    cv2.putText(frame, f"Progress: {progress['progress']:.1%}", (10, 210), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
            
            # スワイプ検出時の視覚的フィードバック
            if swipe_detected:
                last_swipe_alert = frame_count
                print(f"🚀 LEFT-TO-RIGHT SWIPE DETECTED! (Total: {stats['total_swipes_detected']})")
            
            # スワイプアラート表示（3秒間）
            if frame_count - last_swipe_alert < 90:  # 30fps * 3sec
                cv2.putText(frame, "SWIPE DETECTED!", (frame.shape[1]//2 - 100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                # スワイプ方向の矢印を描画
                cv2.arrowedLine(frame, (frame.shape[1]//2 - 50, 80), 
                               (frame.shape[1]//2 + 50, 80), (0, 255, 0), 5)
            
            # Draw each detected hand
            for i, hand in enumerate(hands):
                draw_hand_landmarks(frame, hand)
                
                # Print hand info every 2 frames
                if frame_count % 2 == 0:
                    print(f"Hand {i+1}: {hand.label} (confidence: {hand.lm_score:.3f})")
                    if hasattr(hand, 'gesture') and hand.gesture:
                        print(f"  Gesture: {hand.gesture}")
            
            # Display frame
            cv2.imshow("OAK-D Hand Detection with Swipe", frame)
            
            # Handle key presses
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
        # Cleanup
        detector.close()
        cv2.destroyAllWindows()
        
        # Print final statistics
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