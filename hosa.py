import kivy
import time
import cv2
import mediapipe as mp
import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from plyer import notification

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Set the window size
Window.size = (405, 720)

class HosaUI(FloatLayout):
    def __init__(self, **kwargs):
        super(HosaUI, self).__init__(**kwargs)
        # Background image filling the whole screen
        home_image = Image(source  = "home_screen.png", allow_stretch=True, keep_ratio=False, size_hint=(1, 1))
        self.add_widget(home_image)

        # Start Button
        self.start_button = Button(size_hint=(None, None), size=(250, 100), 
                       pos_hint={'center_x': 0.5, 'center_y': 0.24}, background_normal='',  
                       background_color=(0, 0, 0, 0))
        self.add_widget(self.start_button)
        self.start_button.bind(on_press=self.start_calibration)

        # Label to show calibration results
        self.result_label = Label(text="", size_hint=(1, None), height=50,
                                  pos_hint={'center_x': 0.5, 'center_y': 0.2}, font_size=20)
        self.add_widget(self.result_label)

        self.capture = None
        self.running = False
        self.start_time = None  # Track calibration start time
        self.is_tracking_active = False  
        self.ear_values = []
        self.mar_values = []
        self.eed_values = []

        # Initialize counter frames
        self.blink_frame_counter = 0
        self.yawn_frame_counter = 0
        self.yawn_count = 0
        self.blink_count = 0

    def start_calibration(self, event):
        self.clear_widgets()

        # Camera Display
        self.camera_display = Image(size_hint=(1, 1))
        self.add_widget(self.camera_display)

        self.stats_label = Label(text="Calibration in progress...", size_hint=(1, None), height=50,
                                 pos_hint={'center_x': 0.5, 'center_y': 0.8}, font_size=20)
        self.add_widget(self.stats_label)

        """Start capturing frames from the webcam."""
        self.running = True
        self.start_button.disabled = True
        self.capture = cv2.VideoCapture(0)
        
        # Start updating frames
        Clock.schedule_interval(self.update_frame, 1.0 / 30)  # 30 FPS
        self.start_time = time.time()  # Start calibration timer

    def update_frame(self, dt):
        """Continuously update the camera feed and perform calibration."""
        success, frame = self.capture.read()
        if not success:
            return

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Detection with MediaPipe
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green dots

            # Extract facial metrics
            self.collect_calibration_data(results.multi_face_landmarks[0], frame.shape)

          # Display countdown
        cv2.putText(frame, f"Time left: {10 - int(time.time() - self.start_time)}s", (10, 30), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)

        # Flip frame to match Kivy coordinate system
        frame = cv2.flip(frame, 0)

        # Convert frame to Kivy texture
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

        # Update Kivy image
        self.camera_display.texture = texture

        # Update real-time stats display
        if self.ear_values:
            self.stats_label.text = (f"EAR: {self.ear_values[-1]:.2f}\n"
                                     f"MAR: {self.mar_values[-1]:.2f}\n"
                                     f"EED: {self.eed_values[-1]:.2f}")

        # Check if calibration is complete
        if time.time() - self.start_time >= 10:
            self.complete_calibration()

    def collect_calibration_data(self, face_landmarks, frame_shape):
        """Extract and store EAR, MAR, and EED values."""
        left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame_shape)
        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2
        if ear > 0.2:  # Exclude closed eyes
            self.ear_values.append(ear)

        mouth = self.extract_mouth_landmarks(face_landmarks, frame_shape)
        mar = self.mouth_aspect_ratio(mouth)
        self.mar_values.append(mar)

        eyebrow = self.extract_eyebrow_landmarks(face_landmarks, frame_shape)
        eed = self.eyebrow_eye_distance(eyebrow, left_eye)
        self.eed_values.append(eed)

    def complete_calibration(self):
        """Calculate baseline values and update UI."""
        if not self.ear_values or not self.mar_values or not self.eed_values:
            self.result_label.text = "Calibration failed. Try again!"
        else:
            self.baseline_ear = np.mean(self.ear_values)
            self.baseline_mar = np.mean(self.mar_values)
            self.baseline_eed = np.mean(self.eed_values)
            self.result_label.text = (f"Calibration Complete!\n"
                                      f"EAR: {self.baseline_ear:.2f}\n"
                                      f"MAR: {self.baseline_mar:.2f}\n"
                                      f"EED: {self.baseline_eed:.2f}")
        self.add_widget(self.result_label)

        self.running = False
        Clock.unschedule(self.update_frame)
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        # Wait 1 second before switching to the tracker
        Clock.schedule_once(self.record_process, 1)

    def calculate_fatigue_score(self, avg_ear, avg_mar, avg_eed, baseline_ear, baseline_mar, baseline_eed, yawn_count, blink_count, baseline_bpm):
        # Compute deviations
        ear_dev = max(0, (baseline_ear - avg_ear) / baseline_ear)
        bpm_dev = max(0, abs(blink_count - baseline_bpm) / baseline_bpm)
        mar_dev = max(0, (baseline_mar - avg_mar) / baseline_mar)
        eed_dev = max(0, (baseline_eed-avg_eed) / baseline_eed)
        

        # Weights for each component
        w_ear = 3
        w_mar = 2
        w_eed = 1
        w_yawn = 0.4
        w_bpm = 0.3

        fatigue_score = (w_ear * ear_dev +
                     w_mar * mar_dev +
                     w_eed * eed_dev +
                     w_yawn * yawn_count +
                     w_bpm * bpm_dev)
        
        return fatigue_score

    def record_process(self, dt = None):
        print("Enter recording process")

        if self.is_tracking_active:
            print("record_process skipped because tracking is active")
            return  # Prevent re-entering tracking process

        self.is_tracking_active = True  # Mark tracking as active

        self.clear_widgets()
        # Recalibration Button
        self.recalibrate_button = Button(text = 'Recalibrate', size_hint=(None, None), size=(250, 100), 
                       pos_hint={'center_x': 0.5, 'center_y': 0.24}, background_normal='',  
                       background_color=(0, 0, 0, 0))

        on_image = Image(source  = "HOSA_ON.png", allow_stretch=True, keep_ratio=False, size_hint=(1, 1))
        self.add_widget(on_image)
        print("Added on_image")

        self.add_widget(self.recalibrate_button)
        self.recalibrate_button.bind(on_press=self.start_calibration)
        print("Added recalibrate_button")

        # Initialize values
        self.ear_track = []
        self.mar_track = []
        self.eed_track = []

        # Camera Display
        self.camera_display = Image(size_hint=(1, 1))
        #self.add_widget(self.camera_display)
        print("Added camera_display")
        
        self.stats_tracker = Label(size_hint=(1, None), height=50,
                                 pos_hint={'center_x': 0.5, 'center_y': 0.8}, font_size=20)
        self.add_widget(self.stats_tracker)
        print("Added stats_tracker")

        """Start capturing frames from the webcam."""
        self.running = True
        self.start_button.disabled = True

        if self.capture is None or not self.capture.isOpened():
            self.capture = cv2.VideoCapture(0)
            print("Camera initialized")
        
        # Start updating frames
        Clock.schedule_interval(self.track_frame, 1.0 / 30)  # 30 FPS

    def track_frame(self, dt):
        """Continuously update the camera feed and perform calibration."""

        success, frame = self.capture.read()
        if not success:
            return  

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Detection with MediaPipe
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green dots

            # Extract facial metrics
            self.collect_tracking_data(results.multi_face_landmarks[0], frame.shape)

        # Flip frame to match Kivy coordinate system
        frame = cv2.flip(frame, 0)

        # Convert frame to Kivy texture
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

        # Update Kivy image
        self.camera_display.texture = texture

        # Update real-time stats display
        if self.ear_track:
            self.stats_tracker.text = (f"EAR: {self.ear_track[-1]:.2f}\n"
                                     f"MAR: {self.mar_track[-1]:.2f}\n"
                                     f"EED: {self.eed_track[-1]:.2f}")
        
        # Stops after 30 seconds
        Clock.schedule_once(lambda dt: self.stop_tracking(), 30)
            
    def collect_tracking_data(self, face_landmarks, frame_shape):
        mar_yawn_threshold = 0.4
        yawn_consec_frames = 7
        ear_blink_threshold = 0.21
        blink_consec_frames=2
        
        """Extract and store EAR, MAR, and EED values."""
        left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame_shape)
        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2
        self.ear_track.append(ear)

        # Compute dynamic blink threshold using median of last 10 EAR values
        if len(self.ear_track) >= 10:
            recent_baseline = np.median(self.ear_track[-10:])
            dynamic_threshold = 0.9 * recent_baseline
        else:
            dynamic_threshold = ear_blink_threshold
        if ear < dynamic_threshold:
            self.blink_frame_counter += 1
        else:
            if self.blink_frame_counter >= blink_consec_frames:
                self.blink_count += 1
                self.blink_frame_counter = 0

        mouth = self.extract_mouth_landmarks(face_landmarks, frame_shape)
        mar = self.mouth_aspect_ratio(mouth)
        self.mar_track.append(mar)

        if mar > mar_yawn_threshold:
                self.yawn_frame_counter += 1
        else:
            if self.yawn_frame_counter >= yawn_consec_frames:
                self.yawn_count += 1
                self.yawn_frame_counter = 0

        eyebrow = self.extract_eyebrow_landmarks(face_landmarks, frame_shape)
        eed = self.eyebrow_eye_distance(eyebrow, left_eye)
        self.eed_track.append(eed)

    def stop_tracking(self):
        """Calculate median values after the 30-second recording session and schedule the next one."""
        
        Clock.unschedule(self.track_frame)  # Stop data collection
        Clock.unschedule(self.record_process)
        Clock.unschedule(self.off_interval)

        if self.ear_track and self.mar_track and self.eed_track:
            median_ear = np.mean(self.ear_track)
            median_mar = np.mean(self.mar_track)
            median_eed = np.mean(self.eed_track)

            fatigue = self.calculate_fatigue_score(median_ear, median_mar, median_eed, self.baseline_ear, self.baseline_mar, self.baseline_eed, self.yawn_count, self.blink_count, 17)
            fatigue_threshold = 0.7

            # Label to show tracking results
            self.track_results = Label(text="", size_hint=(1, None), height=50,
                                    pos_hint={'center_x': 0.5, 'center_y': 0.15}, font_size=20)

            # Display latest median values
            self.track_results.text = (f"Latest Median Stats:\n"
                                    f"EAR: {median_ear:.2f}\n"
                                    f"MAR: {median_mar:.2f}\n"
                                    f"EED: {median_eed:.2f}\n"
                                    f"Fatigue Score: {fatigue:.2f}\n"
                                    f"Blink Count: {self.blink_count}\n"
                                    f"Yawn Count: {self.yawn_count}\n") 

        if self.capture is not None:
            self.capture.release()
            self.capture = None
           
        # Properly release the camera before waiting**
        self.running = False
        self.is_tracking_active = False
        if self.track_results not in self.children:
            self.add_widget(self.track_results)
        
        if hasattr(self, 'fatigue') and fatigue > fatigue_threshold:
            self.fatigue_notification()

        # Ensure camera is fully released before switching
        Clock.schedule_once(self.off_interval, 0.1)
        
    def off_interval(self, dt):
        Clock.unschedule(self.stop_tracking)
        Clock.unschedule(self.record_process)
        
        self.clear_widgets()
        off_image = Image(source  = "HOSA_OFF.png", allow_stretch=True, keep_ratio=False, size_hint=(1, 1))
        self.add_widget(off_image)

        # Wait 10 seconds before switching to the tracker
        if not self.is_tracking_active and not self.running:
          
            Clock.schedule_once(self.record_process, 600)
            return

    def extract_eye_landmarks(self, face_landmarks, frame_shape):
        """Extract eye landmarks."""
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        return self.extract_landmarks(face_landmarks, frame_shape, left_eye_indices), \
               self.extract_landmarks(face_landmarks, frame_shape, right_eye_indices)

    def extract_mouth_landmarks(self, face_landmarks, frame_shape):
        """Extract mouth landmarks."""
        mouth_indices = [61, 291, 13, 14]
        return self.extract_landmarks(face_landmarks, frame_shape, mouth_indices)

    def extract_eyebrow_landmarks(self, face_landmarks, frame_shape):
        """Extract eyebrow landmarks."""
        eyebrow_indices = [70, 63, 105, 66, 107]
        return self.extract_landmarks(face_landmarks, frame_shape, eyebrow_indices)

    def extract_landmarks(self, face_landmarks, frame_shape, indices):
        """Extract specific facial landmarks as NumPy array."""
        return np.array([(face_landmarks.landmark[i].x * frame_shape[1],
                          face_landmarks.landmark[i].y * frame_shape[0])
                         for i in indices], dtype=np.int32)

    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)."""
        vertical = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5]) + np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return vertical / (2.0 * horizontal)

    def mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio (MAR)."""
        vertical = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])
        horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
        return vertical / horizontal

    def eyebrow_eye_distance(self, eyebrow_landmarks, eye_landmarks):
        """Calculate Eyebrow-Eye Distance (EED)."""
        eyebrow_y = np.mean([pt[1] for pt in eyebrow_landmarks])
        eye_y = np.mean([pt[1] for pt in eye_landmarks])
        return eyebrow_y - eye_y
    
    def fatigue_notification(self):
        if not hasattr(self, 'notification_sent') or not self.notification_sent:
            notification.notify(
                title="Fatigue Detected",
                message="Take a break.",
                app_name="FaceQ",
                timeout=10
            )
        self.notification_sent = True  # Set flag to prevent re-notification

class HosaApp(App):
    def build(self):
        return HosaUI()

if __name__ == "__main__":
    HosaApp().run() 
