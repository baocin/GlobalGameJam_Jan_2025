from ursina import *
from ursina.texture_importer import load_texture
import cv2
from PIL import Image
import io
import mediapipe as mp
import numpy as np
import random
import time
import os
from collections import defaultdict

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class Phase1Scene(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.player_data = []
        self.debug_entities = []
        self.face_detection_active = True
        self.player_colors = {}
        self.last_process_time = 0
        self.process_interval = 1/15  # Target 15 FPS processing
        
        # Initialize debug screen elements
        self.debug_panel = None
        self.debug_texts = []
        self.debug_swatches = []
        
        # Initialize face detection
        self.face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1
        ).process
        
        # Find actual max supported resolution
        self.actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.actual_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.actual_height)
        self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, _ = self.capture.read()
        time.sleep(0.2)
        ret, self.frame = self.capture.read()
        if ret:
            self.process_frame()
            self.show_debug_screen()

    def process_frame(self):
        # Clear previous debug entities
        for entity in self.debug_entities:
            destroy(entity)
        self.debug_entities.clear()

        if self.face_detection_active:
            # Clear previous player data only during face detection
            self.player_data = []
            
            rgb_frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
            results = self.face_detection(rgb_frame)

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
                        'dominant_color': avg_color
                    }
                    self.player_data.append(data)
                    
                    # Process face texture in memory
                    if width > 0 and height > 0:
                        face_img = self.frame[y:y+height, x:x+width]
                        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        with io.BytesIO() as output:
                            pil_img.save(output, format="PNG")
                            contents = output.getvalue()
                            
                        # Convert bytes to PIL Image before creating texture
                        img = Image.open(io.BytesIO(contents))
                        self.player_data[-1]['texture'] = Texture(img)
        else:
            # Just track colors in existing bounding boxes
            rgb_frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
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
                        # Calculate dominant color using mean
                        avg_color = np.mean(roi, axis=(0,1))
                        self.player_data[self.player_data.index(data)]['dominant_color'] = avg_color
                        
        # Update debug screen with new data
        self.show_debug_screen()

    def show_debug_screen(self):
        # Create or reuse debug panel
        if not self.debug_panel:
            self.debug_panel = Entity(
                parent=self,
                model='quad',
                color=color.black66,
                scale=(window.aspect_ratio * 2, 2),
                z=1
            )
            
            # Create pool of text entities
            for _ in range(8):
                self.debug_texts.append(Text(
                    parent=self,
                    text="",
                    origin=(0, 0),
                    color=color.white,
                    scale=4,
                    font='VeraMono.ttf',
                    z=0
                ))
                self.debug_swatches.append(Entity(
                    parent=self,
                    model='quad',
                    scale=(0.2, 0.2),
                    z=0
                ))

        # Update color percentage display
        color_text = ""
        for player_num, colors in self.player_colors.items():
            color_text += f"Player {player_num}: Green {colors['green']}% / Red {colors['red']}%   "
        
        if len(self.debug_texts) > 0:
            self.debug_texts[0].text = color_text
            self.debug_texts[0].position = (0, 0.9)
            self.debug_texts[0].enabled = True

        # Update debug display
        grid_size = max(1, int(len(self.player_data) ** 0.5))
        spacing = 1.2
        
        for i, data in enumerate(self.player_data):
            if i < len(self.debug_texts)-1:  # Reserve first text for color display
                texture = data['texture']
                if texture:
                    col = i % grid_size
                    row = i // grid_size
                    x = (col - (grid_size-1)/2) * spacing
                    y = ((grid_size-1)/2 - row) * spacing
                    
                    # Update text and swatch
                    text_entity = self.debug_texts[i+1]
                    swatch = self.debug_swatches[i]
                    
                    if data['dominant_color'] is not None:
                        r, g, b = data['dominant_color']
                        color_info = f"\nColor: R:{int(r)} G:{int(g)} B:{int(b)}"
                        text_entity.text = f'Player {data["player_num"]}{color_info}'
                        text_entity.position = Vec2(x, y-0.6)
                        text_entity.color = color.rgb(r/255, g/255, b/255)
                        text_entity.enabled = True
                        
                        swatch.color = color.rgb(r/255, g/255, b/255)
                        swatch.position = Vec2(x+0.6, y)
                        swatch.enabled = True
                    
        # Hide unused entities
        for j in range(len(self.player_data)+1, len(self.debug_texts)):
            self.debug_texts[j].enabled = False
            if j-1 < len(self.debug_swatches):
                self.debug_swatches[j-1].enabled = False

        # Update instruction text
        if len(self.debug_texts) > 0:
            instruction = self.debug_texts[-1]
            instruction.text = "DEBUG SCREEN - Press 'space' to continue"
            instruction.position = (0, -0.8)
            instruction.color = color.yellow
            instruction.enabled = True

    def update(self):
        ret, self.frame = self.capture.read()
        if ret and time.time() - self.last_process_time > self.process_interval:
            self.process_frame()
            self.last_process_time = time.time()

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

class Phase2Scene(Entity):
    def __init__(self, player_data, **kwargs):
        super().__init__(**kwargs)
        self.player_data = player_data
        self.capture = cv2.VideoCapture(0)
        
        # Set to 720p for better performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.fishes = []
        self.color_detector = ColorDetector()
        self.red_cooldown = 0
        self.green_cooldown = 0
        self.collision_cooldown = {}
        self.start_time = time.time()
        self.player_colors = {}
        self.last_process_time = 0
        self.process_interval = 1/30

        self.background = Entity(model='quad', texture='assets/water', scale=(16, 9), z=1)
        for data in self.player_data:
            if data['texture']:
                fish = Fish(player_num=data['player_num'], generation=0)
                self.fishes.append(fish)
                
        EditorCamera()
        self.color_display = Text(
            text="",
            position=(0, 0.45),
            origin=(0, 0),
            color=color.white,
            scale=2
        )

    def update(self):
        if held_keys['r']:
            new_fish = Fish(player_num=1, generation=0)
            self.fishes.append(new_fish)
        
        if time.time() - self.start_time < 3:
            return
            
        # Spatial partitioning for collision detection
        spatial_grid = defaultdict(list)
        grid_size = 2  # 2x2 meter cells
        
        # Bin fish into grid cells
        for fish in self.fishes:
            cell_x = int(fish.x // grid_size)
            cell_y = int(fish.y // grid_size)
            spatial_grid[(cell_x, cell_y)].append(fish)
        
        # Check collisions within same cell
        for cell_fishes in spatial_grid.values():
            for i, fish1 in enumerate(cell_fishes):
                for fish2 in cell_fishes[i+1:]:
                    if fish1.player_num == fish2.player_num:
                        continue
                        
                    pair_key = tuple(sorted([id(fish1), id(fish2)]))
                    
                    if pair_key in self.collision_cooldown:
                        if time.time() - self.collision_cooldown[pair_key] < 2:
                            continue
                            
                    distance = (fish1.position - fish2.position).length()
                    if distance < 1:
                        new_gen = max(fish1.generation, fish2.generation) + 1
                        new_fish = Fish(player_num=fish1.player_num, generation=new_gen)
                        new_fish.position = fish1.position
                        self.fishes.append(new_fish)
                        
                        self.collision_cooldown[pair_key] = time.time()
                        for fish in self.fishes:
                            if fish != new_fish:
                                pair_key = tuple(sorted([id(new_fish), id(fish)]))
                                self.collision_cooldown[pair_key] = time.time()
                        
                        print(f"Collision! Spawned new fish with generation {new_gen}")
        
        # Color detection with throttling
        if time.time() - self.last_process_time > self.process_interval:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                for data in self.player_data:
                    player_num = data['player_num']
                    bbox = data['bbox']
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    if x >= 0 and y >= 0 and width > 0 and height > 0:
                        try:
                            face_roi = frame[y:y+height, x:x+width]
                            if face_roi.size > 0:
                                detected_color, red_pct, green_pct = self.color_detector.detect_color(face_roi, player_num)
                                self.player_colors[player_num] = {'red': red_pct, 'green': green_pct}
                                if detected_color == 'red':
                                    self.handle_red_action(player_num)
                                elif detected_color == 'green':
                                    self.handle_green_action(player_num)
                        except:
                            pass
                            
                color_text = ""
                for player_num in sorted(self.player_colors.keys()):
                    colors = self.player_colors[player_num]
                    color_text += f"{player_num}: {colors['green']}% / {colors['red']}%   "
                self.color_display.text = color_text
                
            self.last_process_time = time.time()

    def handle_red_action(self, player_num):
        print(f"Player {player_num} triggered red action")
        for fish in self.fishes:
            if fish.player_num == player_num:
                fish.dash()
                
    def handle_green_action(self, player_num):
        print(f"Player {player_num} triggered green action")
        for fish in self.fishes:
            if fish.player_num == player_num:
                fish.rotate(time.dt * 360)

    def on_destroy(self):
        if self.capture.isOpened():
            self.capture.release()
        # Explicit texture cleanup
        for data in self.player_data:
            if data.get('texture'):
                data['texture']._texture.release(data['texture']._texture)
                data['texture'] = None

class ColorDetector:
    def __init__(self):
        pass
    
    def detect_color(self, roi, player_num):
        # Use simplified HSV conversion and vectorized operations
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # Use regular HSV instead of HSV_FULL
        
        # Adjust HSV ranges for better green detection
        red_mask = ((hsv[...,0] < 10) | (hsv[...,0] > 170)) & (hsv[...,1] > 100) & (hsv[...,2] > 50)
        green_mask = (hsv[...,0] > 35) & (hsv[...,0] < 85) & (hsv[...,1] > 50) & (hsv[...,2] > 50)
        
        red_pct = np.mean(red_mask) * 100
        green_pct = np.mean(green_mask) * 100
        
        print(f"Player {player_num} - Red: {int(red_pct)}%, Green: {int(green_pct)}%")
        
        # Lower threshold for green detection
        if red_pct > 30 and red_pct > green_pct:
            return 'red', int(red_pct), int(green_pct)
        elif green_pct > 20 and green_pct > red_pct:  # Lower threshold for green
            return 'green', int(red_pct), int(green_pct)
        return None, int(red_pct), int(green_pct)
    
class Fish(Entity):
    def __init__(self, player_num, generation=0, **kwargs):
        model_files = {
            0: 'assets/red_fish.glb',
            1: 'assets/orange_fish.glb',
            2: 'assets/yellow_fish.glb',
            3: 'assets/green_fish.glb'
        }
        model_file = model_files.get(min(generation, 3))
        
        super().__init__(model=model_file, scale=1)
        self.collider = None  # Disable complex collision mesh

        self.player_num = player_num
        self.generation = generation
        self.forward = Vec3(1, 0, 0)
        self.speed = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0).normalized() * 0.02
        self.acceleration = 1.0
        self.dash_duration = 0
        self.rotation_x = 90
        self.rotation_z = random.uniform(0, 360)
        
        self.player_text = Text(
            text=str(player_num),
            parent=self,
            y=-0.20,
            color=color.white,
            scale=6,
            origin=(0,0)
        )
        self.player_text.billboard = True

    def update(self):
        if self.dash_duration > 0:
            self.acceleration = min(self.acceleration + 0.1, 3.0)
            self.dash_duration -= time.dt
            if self.dash_duration <= 0:
                self.reset_speed()
                
        self.position += self.forward * self.speed * time.dt
        
        if self.x > 7:
            self.x = -7
        elif self.x < -7:
            self.x = 7
        if self.y > 4:
            self.y = -4
        elif self.y < -4:
            self.y = 4
            
    def dash(self):
        world_forward = self.forward.rotate(self.rotation_z)
        self.speed = world_forward.normalized() * 0.02
        self.dash_duration = 0.5
        
    def reset_speed(self):
        self.speed = self.speed.normalized() * 0.02
        self.acceleration = 1.0
        
    def rotate(self, direction):
        self.rotation_z += 15 * direction * time.dt

    def look_at_closest_fish(self, fish_list):
        if not fish_list:
            return
            
        closest_fish = None
        min_distance = float('inf')
        for fish in fish_list:
            if fish == self:
                continue
            distance = (fish.position - self.position).length()
            if distance < min_distance:
                min_distance = distance
                closest_fish = fish
                
        if closest_fish:
            self.look_at(closest_fish.position, axis='z')
            
if __name__ == '__main__':
    window.vsync = False
    app = Ursina(
        title='Bang Bang Bubbles',
        development_mode=False,
        editor_ui=False
    )
    phase1 = Phase1Scene()
    scene.entities.append(phase1)
    app.run()
