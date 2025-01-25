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
        # Clear previous player data
        self.player_data = []
        # Clear previous debug entities
        for entity in self.debug_entities:
            destroy(entity)
        self.debug_entities = []

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
                    
                    # Store player data
                    data = {
                        'player_num': i+1,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'bbox': bbox,
                        'texture': None,
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
                
                # Display player number with larger text
                text_entity = Text(
                    parent=self,
                    text=f'Player {data["player_num"]}',
                    position=face_entity.position + Vec2(0, -0.6),
                    origin=(0, 0),
                    color=color.white,
                    scale=4,  # Increased scale
                    font='VeraMono.ttf',
                    z=0
                )
                self.debug_entities.append(text_entity)

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
                            avg_color = cv2.mean(face_roi)[:3]
                            if self.is_blue(avg_color):
                                print(f'Player {data["player_num"]} activated blue!')
                    except:
                        pass  # Handle any array access errors

    def is_blue(self, color):
        # Simple blue detection (BGR format)
        return color[0] > 100 and color[1] < 50 and color[2] < 50

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

def input(key):
    global phase1
    if key == 'space' and isinstance(scene.entities[0], Phase1Scene):
        phase2 = Phase2Scene(player_data=phase1.player_data)
        destroy(phase1)
        scene.add(phase2)

if __name__ == '__main__':
    app = Ursina()
    window.fullscreen = True
    phase1 = Phase1Scene()
    scene.entities.append(phase1)
    app.run()
