import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

class HandMeasurement:
    """
    Hand Measurement Tool
    
    A computer vision application that detects hand landmarks and measures distances
    between different points on the hand in both pixels and inches (when calibrated).
    Features include multiple measurement modes, custom measurements, calibration,
    and data recording capabilities.
    """
    #-------------------------------------------------------------------
    # Initialization and Setup
    #-------------------------------------------------------------------
    def __init__(self):
        # Initialize MediaPipe hands detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,      # For video streams, not static images
            max_num_hands=2,              # Detect up to two hands
            min_detection_confidence=0.7, # Higher value = more precise but might miss some hands
            min_tracking_confidence=0.5   # Balance between accuracy and smoothness
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Set up webcam capture
        self.cap = cv2.VideoCapture(0)  # Use default camera (usually the built-in webcam)

        # Define text appearance settings for on-screen information
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.text_color = (255, 255, 255)  # White text

        # Define available measurement modes
        self.measurement_modes = [
            # Single hand measurements
            {"name": "Index-Middle Tip", "landmarks": [8, 12], "color": (0, 255, 0), "multi_hand": False},   # Distance between index and middle fingertips
            {"name": "Thumb-Pinky Span", "landmarks": [4, 20], "color": (255, 0, 0), "multi_hand": False},   # Maximum hand span (thumb to pinky)
            {"name": "Hand Width (Knuckles)", "landmarks": [5, 17], "color": (0, 0, 255), "multi_hand": False},  # Width across knuckles
            
            # Two-hand measurement
            {"name": "Index-to-Index (Two Hands)", "landmarks": [8, 8], "color": (255, 0, 255), "multi_hand": True}  # Measure between hands
        ]
        self.current_mode = 0  # Start with the first measurement mode
        self.last_mode_change = time.time()  # Prevent too-rapid mode changes

        # Get webcam resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calibration system (for converting pixels to inches)
        self.is_calibrated = False             # Tracks if we've done calibration
        self.pixels_per_inch = None            # Conversion ratio from calibration
        self.calibration_mode = False          # Whether we're in active calibration mode
        self.calibration_points = []           # Points selected during calibration
        self.calibration_known_distance = 1.0  # Default reference distance (1 inch)

        # Measurement recording system
        self.recording = False                 # Whether we're currently recording measurements
        self.record_start_time = None          # When the recording started
        self.recorded_data = []                # Storage for recorded measurements

        # Custom measurement creation
        self.custom_mode_creation = False      # Whether we're creating a custom measurement
        self.custom_mode_points = []           # Points selected for custom measurement

        # Performance optimization
        self.performance_mode = "balanced"     # Current performance mode: "high_quality", "balanced", "performance"

    #-------------------------------------------------------------------
    # Core Measurement Functions
    #-------------------------------------------------------------------
    def calculate_distance(self, point1, point2):
        """Calculate straight-line (Euclidean) distance between two points in pixels"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def process_frame(self, frame):
        """
        Main function to process each camera frame:
        - Detects hands using MediaPipe
        - Draws landmarks on the hands
        - Calculates and displays measurements
        """
        # Convert BGR image to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run hand detection on the frame
        results = self.hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            # Get current measurement mode
            mode = self.measurement_modes[self.current_mode]

            # Draw all hand landmarks for all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # For single-hand modes, analyze and show the hand dimensions
                if not mode["multi_hand"]:
                    # Get hand size metrics (width, height, center position)
                    gesture_info = self.analyze_gesture_size(hand_landmarks)
                    
                    # Draw a yellow bounding box around the hand
                    topleft = (int(gesture_info['center'][0] - gesture_info['width']/2),
                              int(gesture_info['center'][1] - gesture_info['height']/2))
                    bottomright = (int(gesture_info['center'][0] + gesture_info['width']/2),
                                  int(gesture_info['center'][1] + gesture_info['height']/2))
                    cv2.rectangle(frame, topleft, bottomright, (0, 255, 255), 1)
                    
                    # Show the hand dimensions above the hand
                    hand_size_text = f"Hand size: {int(gesture_info['width'])}x{int(gesture_info['height'])} px"
                    cv2.putText(frame, hand_size_text, 
                                (int(gesture_info['center'][0] - 80), int(gesture_info['center'][1] - gesture_info['height']/2 - 10)),
                                self.font, 0.5, (0, 255, 255), 1)

            h, w, c = frame.shape

            # Handle different modes based on single or multi-hand requirements
            if mode["multi_hand"] and len(results.multi_hand_landmarks) >= 2:
                # Multi-hand mode - get landmarks from two different hands
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1]
                landmark1 = mode["landmarks"][0]
                landmark2 = mode["landmarks"][1]

                point1 = (
                    int(hand1.landmark[landmark1].x * w),
                    int(hand1.landmark[landmark1].y * h)
                )
                point2 = (
                    int(hand2.landmark[landmark2].x * w),
                    int(hand2.landmark[landmark2].y * h)
                )

                # Calculate distance in pixels
                distance = self.calculate_distance(point1, point2)

                # Draw the measurement line
                cv2.line(frame, point1, point2, mode["color"], 3)

                # Draw circles at the landmark points
                cv2.circle(frame, point1, 8, mode["color"], -1)
                cv2.circle(frame, point2, 8, mode["color"], -1)

                # Display distance
                self.draw_measurement_info(frame, distance, mode["name"])
                
                # Record measurement if recording is active
                if self.recording:
                    self.record_measurement(distance)

                return True

            elif not mode["multi_hand"]:
                # Single hand mode - get landmarks from first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark1 = mode["landmarks"][0]
                landmark2 = mode["landmarks"][1]

                point1 = (
                    int(hand_landmarks.landmark[landmark1].x * w),
                    int(hand_landmarks.landmark[landmark1].y * h)
                )
                point2 = (
                    int(hand_landmarks.landmark[landmark2].x * w),
                    int(hand_landmarks.landmark[landmark2].y * h)
                )

                # Calculate distance in pixels
                distance = self.calculate_distance(point1, point2)

                # Draw the measurement line
                cv2.line(frame, point1, point2, mode["color"], 3)

                # Draw circles at the landmark points
                cv2.circle(frame, point1, 8, mode["color"], -1)
                cv2.circle(frame, point2, 8, mode["color"], -1)

                # Display distance
                self.draw_measurement_info(frame, distance, mode["name"])
                
                # Record measurement if recording is active
                if self.recording:
                    self.record_measurement(distance)

                return True

            # If we're in multi-hand mode but only one hand is detected
            elif mode["multi_hand"] and len(results.multi_hand_landmarks) < 2:
                self.draw_need_two_hands_message(frame)
                return True

        # If no hands were detected
        return False

    #-------------------------------------------------------------------
    # Display and UI Functions
    #-------------------------------------------------------------------
    def draw_measurement_info(self, frame, distance, mode_name):
        """Draw measurement information on the frame"""
        # Create a semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, self.frame_height - 160), (450, self.frame_height - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw measurement mode and distance
        cv2.putText(
            frame,
            f"Mode: {mode_name}",
            (20, self.frame_height - 130),
            self.font,
            0.8,
            self.text_color,
            self.font_thickness
        )

        # Display distance in pixels and inches if calibrated
        if self.is_calibrated:
            inches = distance / self.pixels_per_inch
            cv2.putText(
                frame,
                f"Distance: {int(distance)} pixels ({inches:.2f} inches)",
                (20, self.frame_height - 100),
                self.font,
                0.8,
                self.text_color,
                self.font_thickness
            )
        else:
            cv2.putText(
                frame,
                f"Distance: {int(distance)} pixels",
                (20, self.frame_height - 100),
                self.font,
                0.8,
                self.text_color,
                self.font_thickness
            )

        # Draw calibration info or note
        if self.is_calibrated:
            cv2.putText(
                frame,
                f"Calibration: {self.pixels_per_inch:.1f} pixels per inch",
                (20, self.frame_height - 70),
                self.font,
                0.6,
                (0, 255, 255),
                1
            )
        else:
            cv2.putText(
                frame,
                "Press 'C' to calibrate for inch measurements",
                (20, self.frame_height - 70),
                self.font,
                0.6,
                (0, 255, 255),
                1
            )

        # Show recording status
        if self.recording:
            duration = time.time() - self.record_start_time
            cv2.putText(
                frame,
                f"RECORDING: {duration:.1f}s ({len(self.recorded_data)} measurements)",
                (20, self.frame_height - 40),
                self.font,
                0.6,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "Press 'R' to start recording measurements",
                (20, self.frame_height - 40),
                self.font,
                0.6,
                (255, 255, 255),
                1
            )

    def draw_instructions(self, frame):
        """Draw user instructions on the frame"""
        # Create a semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw instructions
        cv2.putText(
            frame,
            "Press 'M' to Change Measurement Mode",
            (20, 40),
            self.font,
            0.7,
            self.text_color,
            self.font_thickness
        )
        cv2.putText(
            frame,
            "Press 'C' for Calibration Mode",
            (20, 70),
            self.font,
            0.7,
            self.text_color,
            self.font_thickness
        )
        cv2.putText(
            frame,
            "Press 'X' to Create Custom Measurement",
            (20, 100),
            self.font,
            0.7,
            self.text_color,
            self.font_thickness
        )
        cv2.putText(
            frame,
            "Press 'R' to Record Measurements, 'S' to Save",
            (20, 130),
            self.font,
            0.7,
            self.text_color,
            self.font_thickness
        )
        cv2.putText(
            frame,
            "Press 'Q' to Quit",
            (20, 160),
            self.font,
            0.7,
            self.text_color,
            self.font_thickness
        )

    def draw_no_hand_message(self, frame):
        """Draw message when no hand is detected"""
        # Create a semi-transparent overlay in the center of the frame
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self.frame_width//2 - 200, self.frame_height//2 - 30),
            (self.frame_width//2 + 200, self.frame_height//2 + 30),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw message
        cv2.putText(
            frame,
            "No hand detected, please show your hand",
            (self.frame_width//2 - 190, self.frame_height//2 + 10),
            self.font,
            0.7,
            (0, 0, 255),
            self.font_thickness
        )

    def draw_need_two_hands_message(self, frame):
        """Draw message when two hands are needed but not detected"""
        # Create a semi-transparent overlay in the center of the frame
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self.frame_width//2 - 220, self.frame_height//2 - 30),
            (self.frame_width//2 + 220, self.frame_height//2 + 30),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw message
        cv2.putText(
            frame,
            "This mode needs two hands visible",
            (self.frame_width//2 - 210, self.frame_height//2 + 10),
            self.font,
            0.7,
            (0, 255, 255),
            self.font_thickness
        )

    #-------------------------------------------------------------------
    # Main Application Loop
    #-------------------------------------------------------------------
    def run(self):
        """
        Main application loop that handles:
        - Webcam frame capture and processing
        - Different operating modes (measurement, calibration, custom creation)
        - User input handling
        - Display updates
        """
        while self.cap.isOpened():
            # PART 1: CAPTURE AND PREPARE FRAME
            # ----------------------------------------
            # Get a new frame from the webcam
            ret, frame = self.cap.read()
            if not ret:
                print("⚠ Failed to grab frame from camera")
                break

            # Mirror the frame for a more natural experience (like looking in a mirror)
            frame = cv2.flip(frame, 1)

            # Update frame dimensions in case they changed
            self.frame_height, self.frame_width = frame.shape[:2]

            # PART 2: PROCESS FRAME BASED ON CURRENT MODE
            # ----------------------------------------
            if self.calibration_mode:
                # In calibration mode - user is setting up inch conversion
                self.handle_calibration_mode(frame)
            elif self.custom_mode_creation:
                # In custom mode creation - user is creating a new measurement
                self.handle_custom_mode_creation(frame)
            else:
                # In normal measurement mode - detect hands and show measurements
                hand_detected = self.process_frame(frame)
                
                # Show guidance if no hand is detected
                if not hand_detected:
                    self.draw_no_hand_message(frame)

            # PART 3: UI ELEMENTS
            # ----------------------------------------
            # Add instructions and controls to the frame
            self.draw_instructions(frame)

            # PART 4: DISPLAY THE FRAME
            # ----------------------------------------
            # Create a resizable window to fit different screens
            cv2.namedWindow('Hand Measurement Tool', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Hand Measurement Tool', 1280, 720)
            cv2.imshow('Hand Measurement Tool', frame)

            # PART 5: HANDLE USER INPUT
            # ----------------------------------------
            key = cv2.waitKey(1) & 0xFF
            self.handle_key_press(key)

        # CLEANUP: Release all resources when done
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
    def handle_key_press(self, key):
        """
        Process keyboard input from the user
        Returns False if the application should exit, True otherwise
        """
        """Handle keyboard input from the user"""
        # Exit the application
        if key == ord('q'):
            return False
            
        # Change measurement mode (with time debounce to prevent accidental double-presses)
        elif key == ord('m') and time.time() - self.last_mode_change > 0.3 and not self.calibration_mode and not self.custom_mode_creation:
            self.current_mode = (self.current_mode + 1) % len(self.measurement_modes)
            self.last_mode_change = time.time()
            print(f"Switched to measurement mode: {self.measurement_modes[self.current_mode]['name']}")
            
        # Toggle calibration mode
        elif key == ord('c'):
            self.calibration_mode = not self.calibration_mode
            if self.calibration_mode:
                print("Calibration mode enabled. Click to select two points of known distance.")
                self.calibration_points = []
            else:
                print("Calibration mode disabled.")
                
        # Toggle custom measurement creation mode
        elif key == ord('x'):
            self.custom_mode_creation = not self.custom_mode_creation
            if self.custom_mode_creation:
                print("Custom measurement mode creation enabled. Click to select two landmarks.")
                self.custom_mode_points = []
            else:
                print("Custom measurement mode creation disabled.")
                
        # Toggle measurement recording
        elif key == ord('r'):
            self.recording = not self.recording
            if self.recording:
                self.record_start_time = time.time()
                self.recorded_data = []
                print("Recording started. Perform measurements to record them.")
            else:
                print(f"Recording stopped. {len(self.recorded_data)} measurements recorded.")
                
        # Save recorded measurements
        elif key == ord('s') and not self.recording and len(self.recorded_data) > 0:
            self.export_data()
            
        # Cycle through performance modes
        elif key == ord('p'):
            if self.performance_mode == "high_quality":
                self.performance_mode = "balanced"
            elif self.performance_mode == "balanced":
                self.performance_mode = "performance"
            else:
                self.performance_mode = "high_quality"
                
            self.update_performance_settings()
            print(f"Performance mode changed to: {self.performance_mode}")
            
        return True

    #-------------------------------------------------------------------
    # Calibration System
    #-------------------------------------------------------------------
    def handle_calibration_mode(self, frame):
        """
        Handles the calibration workflow:
        - Lets user select two points of known distance
        - Calculates pixels-per-inch conversion ratio
        - Enables real-world measurements in inches
        """
        # Show clear instructions for the calibration process
        cv2.putText(
            frame,
            "CALIBRATION MODE: Click on two points with known distance",
            (20, self.frame_height - 200),
            self.font,
            0.8,
            (0, 255, 255),
            2
        )

        # Draw calibration points
        for i, point in enumerate(self.calibration_points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Point {i+1}",
                (point[0] + 10, point[1]),
                self.font,
                0.7,
                (0, 0, 255),
                2
            )

        # Draw line between points if we have two
        if len(self.calibration_points) == 2:
            cv2.line(
                frame,
                self.calibration_points[0],
                self.calibration_points[1],
                (0, 255, 0),
                2
            )

            # Calculate distance in pixels
            pixel_distance = self.calculate_distance(self.calibration_points[0], self.calibration_points[1])

            # Display pixel distance and known distance
            cv2.putText(
                frame,
                f"Pixel distance: {pixel_distance:.1f}",
                (20, self.frame_height - 170),
                self.font,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Known distance: {self.calibration_known_distance:.1f} inches (press 1-9 to change)",
                (20, self.frame_height - 140),
                self.font,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press ENTER to apply calibration",
                (20, self.frame_height - 110),
                self.font,
                0.7,
                (0, 255, 255),
                2
            )

        # Set mouse callback for calibration points
        cv2.setMouseCallback('Hand Measurement Tool', self.calibration_mouse_callback)

    def calibration_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for calibration"""
        if event == cv2.EVENT_LBUTTONDOWN and self.calibration_mode and len(self.calibration_points) < 2:
            self.calibration_points.append((x, y))
            print(f"Calibration point {len(self.calibration_points)} set at ({x}, {y})")

            # Apply calibration if we have two points and Enter is pressed
            if len(self.calibration_points) == 2:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    self.apply_calibration()
                # Check for number keys to adjust known distance
                elif key >= ord('1') and key <= ord('9'):
                    self.calibration_known_distance = float(chr(key))
                    print(f"Known distance set to {self.calibration_known_distance} inches")

    def apply_calibration(self):
        """
        Finalize the calibration process:
        - Calculate the pixels-per-inch ratio
        - Enable calibrated measurements
        - Exit calibration mode
        """
        if len(self.calibration_points) == 2:
            # Calculate pixel distance between the two selected points
            pixel_distance = self.calculate_distance(self.calibration_points[0], self.calibration_points[1])
            
            # Convert to pixels per inch using the known physical distance
            self.pixels_per_inch = pixel_distance / self.calibration_known_distance
            self.is_calibrated = True
            
            print(f"✓ Calibration successful: {self.pixels_per_inch:.1f} pixels per inch")
            self.calibration_mode = False  # Exit calibration mode

    #-------------------------------------------------------------------
    # Custom Measurement Creation
    #-------------------------------------------------------------------
    def handle_custom_mode_creation(self, frame):
        """
        Guides the user through creating custom measurements:
        - Detects and displays hand landmarks
        - Lets the user select any two landmarks to measure between
        - Creates a new measurement mode with those landmarks
        """
        # Show instructions for creating a custom measurement
        cv2.putText(
            frame,
            "CUSTOM MEASUREMENT MODE: Show your hand and click on two landmarks",
            (20, self.frame_height - 200),
            self.font,
            0.7,
            (255, 0, 255),
            2
        )

        # Process hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            # Draw custom mode points if available
            for i, point_info in enumerate(self.custom_mode_points):
                point, landmark_idx = point_info
                cv2.circle(frame, point, 8, (255, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Point {i+1}: Landmark {landmark_idx}",
                    (point[0] + 10, point[1]),
                    self.font,
                    0.7,
                    (255, 0, 255),
                    2
                )

            # Draw line between points if we have two
            if len(self.custom_mode_points) == 2:
                cv2.line(
                    frame,
                    self.custom_mode_points[0][0],
                    self.custom_mode_points[1][0],
                    (255, 0, 255),
                    2
                )

                # Display information
                cv2.putText(
                    frame,
                    "Press ENTER to save this custom measurement mode",
                    (20, self.frame_height - 170),
                    self.font,
                    0.7,
                    (255, 0, 255),
                    2
                )

            # Set mouse callback for custom mode points
            cv2.setMouseCallback('Hand Measurement Tool', self.custom_mode_mouse_callback, results.multi_hand_landmarks[0])
        else:
            cv2.putText(
                frame,
                "No hand detected. Please show your hand clearly.",
                (20, self.frame_height - 170),
                self.font,
                0.7,
                (0, 0, 255),
                2
            )

    def custom_mode_mouse_callback(self, event, x, y, flags, hand_landmarks):
        """Handle mouse events for custom measurement mode creation"""
        if event == cv2.EVENT_LBUTTONDOWN and self.custom_mode_creation and len(self.custom_mode_points) < 2:
            # Find the closest landmark to the click position
            h, w = self.frame_height, self.frame_width
            closest_landmark = 0
            min_distance = float('inf')

            for i in range(21):  # MediaPipe has 21 hand landmarks
                landmark_x = int(hand_landmarks.landmark[i].x * w)
                landmark_y = int(hand_landmarks.landmark[i].y * h)
                distance = np.sqrt((x - landmark_x)**2 + (y - landmark_y)**2)

                if distance < min_distance:
                    min_distance = distance
                    closest_landmark = i

            # Store the closest landmark and its position
            self.custom_mode_points.append(((x, y), closest_landmark))
            print(f"Custom mode point {len(self.custom_mode_points)} set at landmark {closest_landmark}")

            # Create custom mode if we have two points and Enter is pressed
            if len(self.custom_mode_points) == 2:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    self.save_custom_mode()

    def save_custom_mode(self):
        """Save custom measurement mode"""
        if len(self.custom_mode_points) == 2:
            landmark1 = self.custom_mode_points[0][1]
            landmark2 = self.custom_mode_points[1][1]

            # Add new measurement mode
            new_mode = {
                "name": f"Custom ({landmark1}-{landmark2})",
                "landmarks": [landmark1, landmark2],
                "color": (255, 0, 255),
                "multi_hand": False
            }

            self.measurement_modes.append(new_mode)
            print(f"Custom measurement mode added: {new_mode['name']}")

            # Switch to the new mode
            self.current_mode = len(self.measurement_modes) - 1
            self.custom_mode_creation = False

    #-------------------------------------------------------------------
    # Recording and Data Export
    #-------------------------------------------------------------------
    def record_measurement(self, distance):
        """Record measurement to history"""
        if self.recording:
            current_time = time.time() - self.record_start_time
            mode = self.measurement_modes[self.current_mode]['name']

            measurement_data = {
                'timestamp': current_time,
                'mode': mode,
                'pixels': distance,
                'inches': distance / self.pixels_per_inch if self.is_calibrated else None
            }

            self.recorded_data.append(measurement_data)

    def export_data(self, format='csv'):
        """
        Save all recorded measurements to a file:
        - Creates a timestamped file in the measurements folder
        - Organizes data with timestamps, measurement modes, and distances
        - Shows both pixel and inch values when calibrated
        """
        try:
            # Make sure we have a folder to store measurements
            import os
            if not os.path.exists('measurements'):
                os.makedirs('measurements')

            # Create a filename with the current date and time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"measurements/hand_measurements_{timestamp}.{format}"

            # Handle CSV export format
            if format == 'csv':
                import csv
                with open(filename, 'w', newline='') as csvfile:
                    # Determine which columns to include
                    fieldnames = ['timestamp', 'mode', 'pixels']
                    if any(item['inches'] is not None for item in self.recorded_data):
                        fieldnames.append('inches')  # Only include if we have inch values

                    # Create the CSV file
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write each measurement as a row
                    for record in self.recorded_data:
                        row_data = {
                            'timestamp': f"{record['timestamp']:.2f}",  # Time since recording started
                            'mode': record['mode'],                     # Measurement mode name
                            'pixels': record['pixels']                  # Distance in pixels
                        }
                        if 'inches' in fieldnames:
                            row_data['inches'] = record['inches']       # Distance in inches (if calibrated)
                        writer.writerow(row_data)

            print(f"✓ Measurements successfully exported to {filename}")
            return filename
        except Exception as e:
            print(f"⚠ Error exporting data: {e}")
            return None

    #-------------------------------------------------------------------
    # Hand Analysis and Metrics
    #-------------------------------------------------------------------
    def analyze_gesture_size(self, hand_landmarks):
        """
        Analyze the size and position of the hand in the frame:
        - Finds the minimum bounding rectangle that contains all landmarks
        - Calculates dimensions, center position, and area
        - Used for visualization and potential size-based features
        """
        # Convert normalized landmark coordinates to pixel coordinates
        h, w = self.frame_height, self.frame_width
        points_x = [landmark.x * w for landmark in hand_landmarks.landmark]
        points_y = [landmark.y * h for landmark in hand_landmarks.landmark]

        # Find the extremes to create a bounding box
        min_x, max_x = min(points_x), max(points_x)
        min_y, max_y = min(points_y), max(points_y)

        # Calculate dimensions and center
        width = max_x - min_x
        height = max_y - min_y
        center = ((min_x + max_x)/2, (min_y + max_y)/2)

        return {
            'width': width,           # Width of hand in pixels
            'height': height,         # Height of hand in pixels
            'center': center,         # Center point of the hand
            'area': width * height    # Area of the bounding box
        }

    #-------------------------------------------------------------------
    # Performance Settings
    #-------------------------------------------------------------------
    def update_performance_settings(self):
        """
        Adjust MediaPipe hand detection settings based on the selected performance mode:
        - high_quality: Better accuracy but slower processing
        - balanced: Good balance between accuracy and speed
        - performance: Faster processing for lower-end systems
        """
        # Properly close the existing hands detection object
        self.hands.close()

        # Configure new detection settings based on performance mode
        if self.performance_mode == "high_quality":
            # High quality mode - prioritize accuracy over speed
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.8,  # Higher confidence threshold
                min_tracking_confidence=0.6    # Better tracking stability
            )
        elif self.performance_mode == "balanced":
            # Balanced mode - default settings
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        else:  # performance mode
            # Performance mode - optimize for speed
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,                # Only detect one hand to save processing
                min_detection_confidence=0.6,   # Lower confidence threshold
                min_tracking_confidence=0.4     # Less strict tracking
            )

#-------------------------------------------------------------------
# Application Entry Point
#-------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("Hand Measurement Tool")
        print("---------------------")
        print("Controls:")
        print("  M - Change measurement mode")
        print("  C - Enter/exit calibration mode")
        print("  X - Create custom measurement")
        print("  R - Start/stop recording measurements")
        print("  S - Save recorded measurements")
        print("  P - Cycle performance modes")
        print("  Q - Quit")
        print("---------------------")
        
        measurement_tool = HandMeasurement()
        measurement_tool.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import sys
        sys.exit(1)
