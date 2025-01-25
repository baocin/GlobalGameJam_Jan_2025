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

def init_camera():
    """Initialize and configure camera capture"""
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)  # Will set to max supported width
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)  # Will set to max supported height
    return capture

def capture_initial_frame(capture):
    """Capture initial frame with exposure adjustment"""
    ret, _ = capture.read()  # Throwaway frame
    time.sleep(0.2)  # Wait 200ms
    ret, frame = capture.read()  # Take the real frame
    return ret, frame

def process_face_detection(frame, face_detection):
    """Process frame for face detection and return results"""
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return face_detection.process(rgb_frame), frame, rgb_frame

def get_face_coordinates(detection, frame_shape, padding=0.1):
    """Calculate face coordinates with padding and bounds checking"""
    h, w = frame_shape[:2]
    bbox = detection.location_data.relative_bounding_box
    
    x = int((bbox.xmin - padding) * w)
    y = int((bbox.ymin - padding) * h)
    width = int((bbox.width + 2*padding) * w)
    height = int((bbox.height + 2*padding) * h)
    
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    width = min(width, w-x)
    height = min(height, h-y)
    
    return x, y, width, height, bbox

def get_dominant_color(frame, x, y, width, height):
    """Extract dominant color from face region"""
    face_roi = frame[y:y+height, x:x+width]
    return cv2.mean(face_roi)[:3]

def save_face_texture(frame, x, y, width, height, player_num):
    """Save face image to temporary file and return texture"""
    if width <= 0 or height <= 0:
        return None, None
        
    face_img = frame[y:y+height, x:x+width]
    temp_file = f'temp_face_{player_num}_{random.randint(1, 1000000)}.png'
    cv2.imwrite(temp_file, face_img)
    texture = load_texture(temp_file)
    os.remove(temp_file)
    return texture, temp_file

def calculate_kmeans_color(roi):
    """Calculate dominant color using k-means clustering"""
    pixels = roi.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers[0]

def create_debug_panel(parent):
    """Create background panel for debug display"""
    panel = Entity(
        parent=parent,
        model='quad',
        color=color.black66,
        scale=(window.aspect_ratio * 2, 2),
        z=1
    )
    return panel

def create_face_display(parent, texture, position, dominant_color, scale=(1.0, 1.0)):
    """Create face display entity with border"""
    entities = []
    
    if dominant_color is not None:
        r, g, b = dominant_color
        border = Entity(
            parent=parent,
            model='quad',
            color=color.rgb(r/255, g/255, b/255),
            position=position,
            scale=(1.1, 1.1),
            z=0.1
        )
        entities.append(border)
    
    face_entity = Entity(
        parent=parent,
        model='quad',
        texture=texture,
        position=position,
        scale=scale,
        z=0
    )
    entities.append(face_entity)
    
    return face_entity, entities

def create_player_info(parent, player_data, face_position):
    """Create player number and color information display"""
    entities = []
    
    if player_data['dominant_color'] is not None:
        r, g, b = player_data['dominant_color']
        color_info = f"\nColor: R:{int(r)} G:{int(g)} B:{int(b)}"
        
        text_entity = Text(
            parent=parent,
            text=f'Player {player_data["player_num"]}{color_info}',
            position=face_position + Vec2(0, -0.6),
            origin=(0, 0),
            color=color.rgb(r, g, b),
            scale=4,
            font='VeraMono.ttf',
            z=0
        )
        entities.append(text_entity)
        
        color_swatch = Entity(
            parent=parent,
            model='quad',
            color=color.rgb(r, g, b),
            position=face_position + Vec2(0.6, 0),
            scale=(0.2, 0.2),
            z=0
        )
        entities.append(color_swatch)
    
    return entities

def on_player_color(player_num, color):
    """Called when a player shows a strong color"""
    print(f'Player {player_num} activated {color}!')

class Phase1Scene(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = init_camera()
        self.player_data = []
        self.debug_entities = []
        self.face_detection_active = True
        
        ret, self.frame = capture_initial_frame(self.capture)
        if ret:
            self.process_frame()
            self.show_debug_screen()

    def process_frame(self):
        # Clear previous debug entities
        for entity in self.debug_entities:
            destroy(entity)
        self.debug_entities.clear()

        if self.face_detection_active:
            self.player_data = []
            
            with mp_face_detection.FaceDetection(
                min_detection_confidence=0.5,
                model_selection=1
            ) as face_detection:
                results, self.frame, rgb_frame = process_face_detection(self.frame, face_detection)

                if results.detections:
                    for i, detection in enumerate(results.detections):
                        x, y, width, height, bbox = get_face_coordinates(detection, self.frame.shape)
                        avg_color = get_dominant_color(self.frame, x, y, width, height)
                        
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
                        
                        texture, temp_file = save_face_texture(self.frame, x, y, width, height, i)
                        if texture:
                            self.player_data[-1]['texture'] = texture
                            self.player_data[-1]['temp_file'] = temp_file
        else:
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
                        avg_color = calculate_kmeans_color(roi)
                        self.player_data[self.player_data.index(data)]['dominant_color'] = avg_color
                        
        self.show_debug_screen()

    def show_debug_screen(self):
        panel = create_debug_panel(self)
        self.debug_entities.append(panel)

        grid_size = max(1, int(len(self.player_data) ** 0.5))
        spacing = 1.2
        
        for i, data in enumerate(self.player_data):
            texture = data['texture']
            
            if texture:
                col = i % grid_size
                row = i // grid_size
                
                x = (col - (grid_size-1)/2) * spacing
                y = ((grid_size-1)/2 - row) * spacing
                position = Vec2(x, y)
                
                face_entity, face_entities = create_face_display(
                    self, texture, position, data['dominant_color']
                )
                self.debug_entities.extend(face_entities)
                
                info_entities = create_player_info(self, data, face_entity.position)
                self.debug_entities.extend(info_entities)

        instruction = Text(
            parent=self,
            text="DEBUG SCREEN - Press 'space' to continue",
            position=(0, -0.8),
            origin=(0, 0),
            color=color.yellow,
            scale=4,
            font='VeraMono.ttf',
            z=0
        )
        self.debug_entities.append(instruction)

    def update(self):
        ret, self.frame = self.capture.read()
        if ret:
            self.process_frame()

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
        for data in self.player_data:
            if os.path.exists(data['temp_file']):
                os.remove(data['temp_file'])

class Phase2Scene(Entity):
    def __init__(self, player_data, **kwargs):
        super().__init__(**kwargs)
        self.player_data = player_data
        self.capture = cv2.VideoCapture(0)
        self.fishes = []
        
        self.background = Entity(model='quad', texture='assets/water', scale=(16, 9))
        
        for data in self.player_data:
            texture = load_texture(data['temp_file'])
            if texture:
                fish = Fish(player_num=data['player_num'], texture=texture)
                self.fishes.append(fish)

    def update(self):
        ret, frame = self.capture.read()
        if ret:
            for data in self.player_data:
                x, y, w, h = data['x'], data['y'], data['width'], data['height']
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size != 0:
                            avg_color = cv2.mean(face_roi)[:3]
                            b, g, r = avg_color
                            if r > 200 and g < 100 and b < 100:
                                on_player_color(data["player_num"], "red")
                            elif b > 200 and r < 100 and g < 100:
                                on_player_color(data["player_num"], "blue")
                    except:
                        pass

        # Debug key controls
        if held_keys['left arrow']:
            on_player_color(1, "blue")
        if held_keys['right arrow']:
            on_player_color(1, "red")
        if held_keys['up arrow']:
            on_player_color(2, "blue")
        if held_keys['down arrow']:
            on_player_color(2, "red")

    def on_destroy(self):
        if self.capture.isOpened():
            self.capture.release()

class Fish(Entity):
    def __init__(self, player_num, texture, **kwargs):
        super().__init__(model='quad', texture='assets/fish_body', scale=(0.5, 0.3))
        self.player_num = player_num
        self.speed = Vec2(random.uniform(-1, 1), random.uniform(-1, 1)).normalized() * 0.02
        
        self.head = Entity(
            parent=self,
            texture=texture,
            scale=(0.3, 0.3),
            position=(0.15, 0)
        )

    def update(self):
        self.position += self.speed
        if abs(self.x) > 7 or abs(self.y) > 4:
            self.speed *= -1

if __name__ == '__main__':
    app = Ursina()
    window.fullscreen = False
    phase1 = Phase1Scene()
    scene.entities.append(phase1)
    app.run()
