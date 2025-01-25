Here's a fleshed-out implementation using Ursina Engine and MediaPipe. Create a new Python file and install required packages (pip install ursina mediapipe opencv-python):

python
Copy
from ursina import *
import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class Phase1Scene(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.player_data = []
        self.debug_textures = []
        
        # Capture initial frame
        ret, self.frame = self.capture.read()
        if ret:
            self.process_frame()
            self.show_debug_screen()

    def process_frame(self):
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for i, detection in enumerate(results.detections):
                    h, w = self.frame.shape[:2]
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Store player data
                    self.player_data.append({
                        'player_num': i+1,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'texture': None
                    })
                    
                    # Crop and store face texture
                    face_img = self.frame[y:y+height, x:x+width]
                    self.player_data[-1]['texture'] = Texture(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

    def show_debug_screen(self):
        # Debug display setup
        grid_size = max(1, int(len(self.player_data) ** 0.5))
        for i, data in enumerate(self.player_data):
            col = i % grid_size
            row = i // grid_size
            
            # Display face crop
            face_entity = Entity(
                parent=self,
                texture=data['texture'],
                position=(window.left + 0.2 + col*0.3, window.top - 0.2 - row*0.4),
                scale=(0.25, 0.25)
            )
            
            # Display player number
            Text(
                text=f'Player {data["player_num"]}',
                position=face_entity.position + Vec2(0, -0.15),
                origin=(0, 0),
                color=color.white
            )

        # Add debug instruction
        Text(
            text="DEBUG SCREEN - Press 'space' to continue",
            position=(0, -0.4),
            origin=(0, 0),
            color=color.yellow
        )

class Phase2Scene(Entity):
    def __init__(self, player_data, **kwargs):
        super().__init__(**kwargs)
        self.player_data = player_data
        self.capture = cv2.VideoCapture(0)
        self.fishes = []
        
        # Create aquarium background
        self.background = Entity(model='quad', texture='assets/water', scale=(16, 9))
        
        # Spawn fish
        for i, data in enumerate(self.player_data):
            fish = Fish(player_num=data['player_num'], texture=data['texture'])
            self.fishes.append(fish)

    def update(self):
        # Color detection loop
        ret, frame = self.capture.read()
        if ret:
            for data in self.player_data:
                x, y, w, h = data['x'], data['y'], data['width'], data['height']
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size != 0:
                    avg_color = cv2.mean(face_roi)[:3]
                    if self.is_blue(avg_color):
                        print(f'Player {data["player_num"]} activated blue!')

    def is_blue(self, color):
        # Simple blue detection (BGR format)
        return color[0] > 100 and color[1] < 50 and color[2] < 50

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
    if key == 'space' and isinstance(scene.active[0], Phase1Scene):
        destroy(scene.active[0])
        phase2 = Phase2Scene(player_data=phase1.player_data)
        scene.add(phase2)

if __name__ == '__main__':
    app = Ursina()
    window.fullscreen = True
    phase1 = Phase1Scene()
    scene.add(phase1)
    app.run()
Key Features:

Phase 1 - Face Detection & Debug Screen:

Uses MediaPipe for face detection

Captures and processes webcam image

Shows debug screen with detected faces and player numbers

Stores face positions and textures for Phase 2

Phase 2 - Aquarium Game:

Spawns fish with player faces as heads

Continuous color detection loop

Blue detection in original face regions

Simple fish movement with boundary checking