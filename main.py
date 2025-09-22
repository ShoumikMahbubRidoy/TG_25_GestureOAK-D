from src.gesture_oak.core.oak_camera import OAKCamera, test_camera_connection
from src.gesture_oak.apps.hand_tracking_app import main as hand_app_main
from src.gesture_oak.apps.swipe_detection_app import main as swipe_app_main
from src.gesture_oak.apps.motion_swipe_app import main as motion_swipe_app_main
import sys


def main():
    print("OAK-D Gesture Recognition Application")
    print("=" * 40)
    print("1. Test camera connection")
    print("2. Run hand tracking app")
    print("3. Run swipe detection app")
    print("4. Run motion-based swipe detection (Approach 2)")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                print("\n--- Camera Connection Test ---")
                if test_camera_connection():
                    print("\n✅ OAK-D camera setup completed successfully!")
                    print("Ready to proceed with gesture recognition implementation.")
                else:
                    print("\n❌ OAK-D camera setup failed!")
                    print("Please check your camera connection and try again.")
            
            elif choice == '2':
                print("\n--- Hand Tracking Application ---")
                hand_app_main()
            
            elif choice == '3':
                print("\n--- Swipe Detection Application ---")
                swipe_app_main()
            
            elif choice == '4':
                print("\n--- Motion-Based Swipe Detection (Approach 2) ---")
                motion_swipe_app_main()
            
            elif choice == '5':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1, 2, 3, 4, or 5.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()
