from ursina import *
from ursina.texture_importer import load_texture
import cv2
import PIL
import mediapipe as mp
import numpy as np
import random
import time
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class Phase1Scene(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.player_data = []
        self.debug_entities = []
        self.face_detection_active = True
        
        # Set camera resolution higher to better detect faces
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)  # Will set to max supported width
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)  # Will set to max supported height
        
        # Capture initial frame
        # Take a throwaway frame to let camera adjust exposure
        ret, _ = self.capture.read()
        time.sleep(0.2)  # Wait 200ms
        ret, self.frame = self.capture.read()  # Take the real frame
        if ret:
            self.process_frame()
            self.show_debug_screen()

    def process_frame(self):
        # Clear previous debug entities
        for entity in self.debug_entities:
            destroy(entity)
        self.debug_entities.clear()  # Properly clear the list

        if self.face_detection_active:
            # Clear previous player data only during face detection
            self.player_data = []
            
            with mp_face_detection.FaceDetection(
                min_detection_confidence=0.5,
                model_selection=1  # Use full-range model instead of short-range
            ) as face_detection:
                # Flip the frame horizontally first
                self.frame = cv2.flip(self.frame, 1)
                rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for i, detection in enumerate(results.detections):
                        h, w = self.frame.shape[:2]
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Add padding around face detection
                        padding = 0.1  # 10% padding
                        x = int((bbox.xmin - padding) * w)
                        y = int((bbox.ymin - padding) * h)
                        width = int((bbox.width + 2*padding) * w)
                        height = int((bbox.height + 2*padding) * h)
                        
                        # Ensure coordinates are within frame bounds
                        x = max(0, min(x, w-1))
                        y = max(0, min(y, h-1))
                        width = min(width, w-x)
                        height = min(height, h-y)
                        
                        # Get dominant color from face region
                        face_roi = self.frame[y:y+height, x:x+width]
                        avg_color = cv2.mean(face_roi)[:3]
                        
                        # Store player data
                        data = {
                            'player_num': i+1,
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height,
                            'bbox': bbox,
                            'key_points': detection.location_data.relative_keypoints,
                            'texture': None,
                            'dominant_color': avg_color,
                            'temp_file': f'temp_face_{i}.png'
                        }
                        self.player_data.append(data)
                        
                        # Save face image to temp file
                        if width > 0 and height > 0:  # Ensure valid dimensions
                            face_img = self.frame[y:y+height, x:x+width]
                            # Save to random file name
                            temp_file = f'temp_face_{i}_{random.randint(1, 1000000)}.png'
                            cv2.imwrite(temp_file, face_img)
                            self.player_data[-1]['texture'] = load_texture(temp_file)
                            os.remove(temp_file)
        else:
            # Just track colors in existing bounding boxes
            self.frame = cv2.flip(self.frame, 1)
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            for data in self.player_data:
                bbox = data['bbox']
                h, w = self.frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                if x >= 0 and y >= 0 and width > 0 and height > 0:
                    roi = rgb_frame[y:y+height, x:x+width]
                    if roi.size > 0:
                        # Calculate dominant color
                        # Use k-means clustering to find dominant color
                        pixels = roi.reshape(-1, 3)
                        pixels = np.float32(pixels)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                        _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                        avg_color = centers[0]
                        self.player_data[self.player_data.index(data)]['dominant_color'] = avg_color
                        
        # Update debug screen with new data
        self.show_debug_screen()

    def show_debug_screen(self):
        # Create a background panel that fills the screen
        panel = Entity(
            parent=self,
            model='quad',
            color=color.black66,
            scale=(window.aspect_ratio * 2, 2),  # Double the scale to fill screen
            z=1
        )
        self.debug_entities.append(panel)

        # Debug display setup
        grid_size = max(1, int(len(self.player_data) ** 0.5))
        spacing = 1.2  # Increased spacing between faces
        
        for i, data in enumerate(self.player_data):
            # Load texture from temp file
            texture = data['texture']
            
            if texture:  # Only create entity if texture exists
                col = i % grid_size
                row = i // grid_size
                
                # Calculate centered position
                x = (col - (grid_size-1)/2) * spacing
                y = ((grid_size-1)/2 - row) * spacing
                
                # Create colored border first
                if data['dominant_color'] is not None:
                    r, g, b = data['dominant_color']
                    border = Entity(
                        parent=self,
                        model='quad',
                        color=color.rgb(r/255, g/255, b/255),
                        position=Vec2(x, y),
                        scale=(1.1, 1.1),  # Slightly larger than face
                        z=0.1
                    )
                    self.debug_entities.append(border)
                
                # Display face crop with larger scale
                face_entity = Entity(
                    parent=self,
                    model='quad',  # Explicitly set model
                    texture=texture,
                    position=Vec2(x, y),
                    scale=(1.0, 1.0),  # Increased scale
                    z=0
                )
                self.debug_entities.append(face_entity)
                
                # Display player number and color info with larger text
                color_info = ""
                if data['dominant_color'] is not None:
                    r, g, b = data['dominant_color']
                    color_info = f"\nColor: R:{int(r)} G:{int(g)} B:{int(b)}"
                    
                text_entity = Text(
                    parent=self,
                    text=f'Player {data["player_num"]}{color_info}',
                    position=face_entity.position + Vec2(0, -0.6),
                    origin=(0, 0),
                    color=color.rgb(r, g, b),
                    scale=4,  # Increased scale
                    font='VeraMono.ttf',
                    z=0
                )
                self.debug_entities.append(text_entity)
                
                # Add color swatch
                if data['dominant_color'] is not None:
                    r, g, b = data['dominant_color']
                    color_swatch = Entity(
                        parent=self,
                        model='quad',
                        color=color.rgb(r, g, b),
                        position=face_entity.position + Vec2(0.6, 0),
                        scale=(0.2, 0.2),
                        z=0
                    )
                    self.debug_entities.append(color_swatch)

        # Add debug instruction with larger text
        instruction = Text(
            parent=self,
            text="DEBUG SCREEN - Press 'space' to continue",
            position=(0, -0.8),
            origin=(0, 0),
            color=color.yellow,
            scale=4,  # Increased scale
            font='VeraMono.ttf',
            z=0
        )
        self.debug_entities.append(instruction)

    def update(self):
        # Continuously capture and process frames
        ret, self.frame = self.capture.read()
        if ret:
            self.process_frame()

        # Check for spacebar input here instead of global input handler
        if held_keys['space']:
            if self.face_detection_active:
                print("Stopping face detection")
                self.face_detection_active = False
            else:
                phase2 = Phase2Scene(player_data=self.player_data)
                destroy(self)
                scene.entities.append(phase2)

    def on_destroy(self):
        if self.capture.isOpened():
            self.capture.release()
        # Clean up temporary files
        for data in self.player_data:
            if os.path.exists(data['temp_file']):
                os.remove(data['temp_file'])

class Phase2Scene(Entity):
    def __init__(self, player_data, **kwargs):
        super().__init__(**kwargs)
        self.player_data = player_data
        self.capture = cv2.VideoCapture(0)
        self.fishes = []
        self.color_detector = ColorDetector()
        
        # Add example hooks
        self.color_detector.add_red_hook(lambda: print("Red detected!"))
        self.color_detector.add_blue_hook(lambda: print("Blue detected!"))
        
        # Create aquarium background
        self.background = Entity(model='quad', texture='assets/water', scale=(16, 9))
        
        # Spawn fish
        for data in self.player_data:
            texture = load_texture(data['temp_file'])
            if texture:  # Only create fish if texture exists
                fish = Fish(player_num=data['player_num'], texture=texture)
                self.fishes.append(fish)

    def update(self):
        # Color detection loop
        ret, frame = self.capture.read()
        if ret:
            for data in self.player_data:
                x, y, w, h = data['x'], data['y'], data['width'], data['height']
                if x >= 0 and y >= 0 and w > 0 and h > 0:  # Ensure valid coordinates
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size != 0:
                            detected_color = self.color_detector.detect_color(face_roi)
                            if detected_color:
                                print(f'Player {data["player_num"]} activated {detected_color}!')
                    except:
                        pass  # Handle any array access errors

class ColorDetector:
    def __init__(self):
        # Convert RGB targets to BGR for OpenCV
        self.target_red = [50, 70, 140]  # RGB -> BGR
        self.target_blue = [150, 50, 100]  # RGB -> BGR
        self.color_threshold = 20  # Allowed color difference
        self.red_hooks = []
        self.blue_hooks = []
        
    def add_red_hook(self, callback):
        self.red_hooks.append(callback)
        
    def add_blue_hook(self, callback):
        self.blue_hooks.append(callback)
        
    def detect_color(self, roi):
        # Calculate percentage of pixels that are red/blue
        pixels = roi.reshape(-1, 3)  # Reshape to list of pixels
        total_pixels = len(pixels)
        
        # Count pixels where red channel > 110
        red_pixels = np.sum(pixels[:, 2] > 110)  # Red is channel 2 in BGR
        red_percentage = red_pixels / total_pixels
        
        # Count pixels where blue channel > 110 
        blue_pixels = np.sum(pixels[:, 0] > 110)  # Blue is channel 0 in BGR
        blue_percentage = blue_pixels / total_pixels

        # If more than 50% of pixels are red/blue
        if red_percentage > 0.5:
            for hook in self.red_hooks:
                hook()
            print("Red detected!")
            return 'red'
        elif blue_percentage > 0.5:
            for hook in self.blue_hooks:
                hook()
            print("Blue detected!")
            return 'blue'
        return None

    def on_destroy(self):
        if self.capture.isOpened():
            self.capture.release()

class Fish(Entity):
    def __init__(self, player_num, texture, **kwargs):
        super().__init__(model='quad', texture='assets/fish_body', scale=(0.5, 0.3))
        self.player_num = player_num
        self.speed = Vec2(random.uniform(-1, 1), random.uniform(-1, 1)).normalized() * 0.02
        
        # Replace head with player face
        self.head = Entity(
            parent=self,
            texture=texture,
            scale=(0.3, 0.3),
            position=(0.15, 0)
        )

    def update(self):
        self.position += self.speed
        # Simple boundary check
        if abs(self.x) > 7 or abs(self.y) > 4:
            self.speed *= -1

if __name__ == '__main__':
    app = Ursina()
    window.fullscreen = False
    phase1 = Phase1Scene()
    scene.entities.append(phase1)
    app.run()
