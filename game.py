from ursina import *
from ursina.texture_importer import load_texture
from ursina.prefabs.editor_camera import EditorCamera
import cv2
from PIL import Image, ImageDraw
import io
import mediapipe as mp
import numpy as np
import random
import time
import os
import math
from collections import defaultdict

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global player data
player_data = []

class Bubble(Entity):
    def __init__(self, position, direction, speed=1, texture=None, **kwargs):
        # If texture is a 3D model, use it as the model instead of texture
        if texture and isinstance(texture, str) and texture.endswith('.glb'):
            super().__init__(
                model=texture,
                position=position,
                scale=0.2,
                **kwargs
            )
            self.model = texture  # Explicitly set model
        else:
            super().__init__(
                model='quad',
                texture=texture,
                position=position,
                scale=0.2,
                **kwargs
            )
        self.billboard = True
        self.direction = direction.normalized()
        self.speed = speed
        # Randomize lifespan between 1-2 seconds with normal distribution
        self.lifespan = random.gauss(1.5, 0.2)  # Mean 1.5s, std dev 0.2s
        self.lifespan = max(1.0, min(2.0, self.lifespan))  # Clamp between 1-2s
        self.start_time = time.time()
        self.direction_z = random.uniform(0.05, 0.15)
        self.color = color.rgba(1, 1, 1, 0.7)  # White with 70% opacity

    def update(self):
        # Add bobbing motion
        self.position += self.direction * self.speed * time.dt
        self.z += self.direction_z * time.dt
        self.y += math.sin(time.time() * 3) * 0.01  # Gentle bobbing up and down
        
        # Add slight random wobble
        self.x += random.uniform(-0.02, 0.02) * time.dt
        
        # Slow down over time like real bubbles
        self.speed = lerp(self.speed, self.speed * 0.5, time.dt)
        
        # Rotate gently
        self.rotation_z += 30 * time.dt
        
        # Fade out and grow slightly before popping
        elapsed = time.time() - self.start_time
        if elapsed >= self.lifespan:
            destroy(self)
        else:
            # Bubble gets slightly bigger as it rises
            growth = min(1.2, 1 + (elapsed / self.lifespan) * 0.3)
            base_scale = 0.2 * growth
            
            # Add pulsing effect
            pulse = math.sin(time.time() * 4) * 0.05
            self.scale = base_scale * (1 + pulse)
            
            # Fade out
            alpha = 1 - (elapsed / self.lifespan)
            self.color = color.rgba(1, 1, 1, alpha * 0.7)

class Phase1Scene(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        global player_data
        player_data = []
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
            global player_data
            player_data = []
            
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
                    player_data.append(data)
                    
                    # Process face texture in memory
                    if width > 0 and height > 0:
                        # Convert face image to PIL format
                        face_img = self.frame[y:y+height, x:x+width]
                        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        
                        # Create circular mask
                        mask = Image.new('L', pil_img.size, 0)
                        draw = ImageDraw.Draw(mask)
                        draw.ellipse((0, 0, pil_img.size[0], pil_img.size[1]), fill=255)
                        
                        # Create output image with transparency
                        output = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                        output.paste(pil_img, (0, 0))
                        output.putalpha(mask)
                        
                        # Generate random filename
                        temp_filename = f'temp_face_{random.randint(1000,9999)}.png'
                        
                        # Save to temp file with transparency
                        output.save(temp_filename, 'PNG')
                        
                        # Load back from file and create texture
                        player_data[-1]['texture'] = load_texture(temp_filename)
                        
                        # Clean up temp file
                        os.remove(temp_filename)
        else:
            # Just track colors in existing bounding boxes
            rgb_frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
            for data in player_data:
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
                        player_data[player_data.index(data)]['dominant_color'] = avg_color
                        
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
        grid_size = max(1, int(len(player_data) ** 0.5))
        spacing = 1.2
        
        for i, data in enumerate(player_data):
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
        for j in range(len(player_data)+1, len(self.debug_texts)):
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
                phase2 = Phase2Scene(player_data=player_data)
                destroy(self)
                scene.entities.append(phase2)

    def on_destroy(self):
        if self.capture.isOpened():
            self.capture.release()

class Phase2Scene(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global player_data
        self.player_data = player_data
        self.capture = cv2.VideoCapture(0)
        
        # Set to 720p for better performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Play background music
        self.bg_audio = Audio('assets/65077_bg_audio.mp3', loop=True, autoplay=True)
        
        self.fishes = []
        self.color_detector = ColorDetector()
        self.red_cooldown = 0
        self.green_cooldown = 0
        self.collision_cooldown = {}
        self.start_time = time.time()
        self.player_colors = {}
        self.last_process_time = 0
        self.process_interval = 1/30
        self.bubble_texture = load_texture('assets/bubble.png')

        # Flat background
        self.background = Entity(
            model='quad',
            texture='assets/water',
            scale=(18, 10),
            z=0.1,
            
        )

        for data in self.player_data:
            if data['texture']:
                fish = Fish(player_num=data['player_num'], generation=0, bubble_texture=self.bubble_texture)
                self.fishes.append(fish)

        # Configure camera
        camera.position = (0, 0, -15)
        camera.fov = 54
        
        # Add editor camera
        self.editor_camera = EditorCamera()

        self.color_display = Text(
            text="",
            position=(0, 0.45),
            origin=(0, 0),
            color=color.white,
            scale=2
        )

    def update(self):
        if held_keys['r']:
            if len(self.fishes) < 50:  # Only spawn if under 50 fish
                new_fish = Fish(player_num=1, generation=0)
                self.fishes.append(new_fish)
        
        if time.time() - self.start_time < 3:
            return
            
        # Spatial partitioning for collision detection
        spatial_grid = defaultdict(list)
        grid_size = 2  # 2x2 meter cells
        
        # Only check collisions if less than 50 fish
        if len(self.fishes) < 50:
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
                                
                        # Calculate 2.5D distance with depth consideration
                        depth_factor = 1 - abs(fish1.z - fish2.z)
                        distance = (fish1.position - fish2.position).length()
                        if distance * depth_factor < 1:
                            new_gen = max(fish1.generation, fish2.generation) + 1
                            if new_gen <= 5:  # Only spawn if generation would be 5 or less
                                new_fish = Fish(player_num=fish1.player_num, generation=new_gen, 
                                             parent1=fish1, parent2=fish2)
                                new_fish.position = fish1.position
                                self.fishes.append(new_fish)
                                
                                self.collision_cooldown[pair_key] = time.time()
                                for fish in self.fishes:
                                    if fish != new_fish:
                                        pair_key = tuple(sorted([id(new_fish), id(fish)]))
                                        self.collision_cooldown[pair_key] = time.time()
                                
                                print(f"Collision! Spawned new fish with generation {new_gen}")
        else:
            # Transition to Phase3Scene when max population reached
            phase3 = Phase3Scene(fishes=self.fishes)
            destroy(self)
            scene.entities.append(phase3)
            return
        
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
                            detected_color, red_pct, green_pct = self.color_detector.detect_color(face_roi, player_num)
                            self.player_colors[player_num] = {'red': red_pct, 'green': green_pct}
                            
                            # Update control states
                            for fish in self.fishes:
                                if fish.player_num == player_num:
                                    if detected_color == 'red':
                                        fish.dash()
                                    elif detected_color == 'green':
                                        fish.rotation_direction = 1 if green_pct > 50 else -1
                                    else:
                                        fish.rotation_direction = 0
                        except:
                            pass
                
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
                # Rotate based on face position in frame
                if self.player_colors[player_num]['green'] > 50:
                    fish.rotate(1)  # Clockwise
                else:
                    fish.rotate(-1)  # Counter-clockwise

    def on_destroy(self):
        if self.capture.isOpened():
            self.capture.release()
        # Explicit texture cleanup
        # for data in self.player_data:
        #     if data.get('texture'):
        #         data['texture']._texture.release(data['texture']._texture)
        #         data['texture'] = None

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
    def __init__(self, player_num, generation=0, parent1=None, parent2=None, bubble_texture=None, **kwargs):
        model_files = {0: 'assets/red_fish.glb', 1: 'assets/orange_fish.glb', 
                      2: 'assets/yellow_fish.glb', 3: 'assets/green_fish.glb'}
        super().__init__(model=model_files.get(min(generation, 3), 'assets/red_fish.glb'), scale=1)
        
        self.player_num = player_num
        self.generation = generation
        self.parent1 = parent1
        self.parent2 = parent2
        self.children = []
        self.bubble_texture = bubble_texture
        
        # Add self to parents' children lists
        if parent1:
            parent1.children.append(self)
        if parent2 and parent2 != parent1:
            parent2.children.append(self)
        
        # Movement parameters
        self.base_speed = 0.1
        self.max_speed = 1.5
        self.current_speed = self.base_speed
        self.rotation_speed = 240
        self.dash_duration = 0.3
        
        # Movement state
        self.rotation_direction = 0
        self.is_dashing = False
        self.dash_end_time = 0
        self.normal_scale = (1, 1, 1)
        self.dash_scale = (1.3, 0.7, 1.3)
        self.z -= 1
        
        # Add red direction arrow
        self.direction_arrow = Entity(
            parent=self,
            model='arrow',
            color=color.rgba(1, 0, 0, 0.1),
            scale=(0.1, 0.1, 0.1),
            rotation_x=-90,  # Changed from rotation_z to rotation_x
            position=(0.5, 0, 0),  # Position in front of the fish
            z=-0.1  # Ensure the arrow renders above the fish model
        )
        
        # Random initial position and rotation
        self.position = (random.uniform(-7, 7), random.uniform(-4, 4), random.uniform(-1, 0))
        self.rotation_z = random.uniform(0, 360)  # Changed from rotation_y to rotation_z
        
        # Player number display
        self.player_text = Text(text=str(player_num), parent=self, y=-0.2, 
                              color=color.white, scale=6, origin=(0,0), billboard=True)

    def update(self):
        # Smooth rotation around Z-axis for direction
        self.rotation_z += self.rotation_direction * self.rotation_speed * time.dt
        
        # Tilt effect during rotation (using X-axis)
        if self.rotation_direction != 0:
            target_roll = -self.rotation_direction * 25
            self.rotation_x = lerp(self.rotation_x, target_roll, time.dt * 8)
        else:
            self.rotation_x = lerp(self.rotation_x, 0, time.dt * 8)
        
        # Get movement direction from direction arrow
        arrow_direction = Vec3(
            self.direction_arrow.world_x - self.world_x,
            self.direction_arrow.world_y - self.world_y,
            self.direction_arrow.world_z - self.world_z
        ).normalized()
        
        # Handle dash acceleration
        if self.is_dashing:
            self.position += arrow_direction * self.current_speed * time.dt
            self.current_speed = lerp(self.current_speed, self.max_speed, time.dt * 15)
            self.scale = lerp(self.scale, self.dash_scale, time.dt * 10)
        else:
            # Regular movement
            self.position += arrow_direction * self.current_speed * time.dt
            self.current_speed = lerp(self.current_speed, self.base_speed, time.dt * 5)
            self.scale = lerp(self.scale, self.normal_scale, time.dt * 10)
        # Depth-based effects
        self.z += random.uniform(-0.01, 0.01)
        self.z = clamp(self.z, -1, 0)
        self.color = color.white * (0.7 + (-self.z * 0.3))
        
        # Screen wrapping with proper edge detection
        screen_width = window.aspect_ratio * 10  # Adjust based on camera FOV
        screen_height = 10  # Adjust based on camera FOV
        
        if self.x > screen_width/2:
            self.x = -screen_width/2
        elif self.x < -screen_width/2:
            self.x = screen_width/2
            
        if self.y > screen_height/2:
            self.y = -screen_height/2
        elif self.y < -screen_height/2:
            self.y = screen_height/2

    def dash(self):
        """Trigger dash with visual effects"""
        self.is_dashing = True
        self.dash_end_time = time.time() + self.dash_duration
        if not hasattr(self, 'dash_task') or self.dash_task.finished:
            self.dash_task = invoke(setattr, self, 'is_dashing', False, delay=self.dash_duration)
        self.spawn_bubbles()

    def spawn_bubbles(self):
        num_bubbles = 8
        # Get direction from direction arrow (opposite of dash direction)
        arrow_direction = Vec3(
            self.direction_arrow.world_x - self.world_x,
            self.direction_arrow.world_y - self.world_y,
            self.direction_arrow.world_z - self.world_z
        ).normalized()
        # Reverse direction for bubbles
        head_dir = -arrow_direction
        spread_angle = 45
        
        for _ in range(num_bubbles):
            angle = math.radians(random.uniform(-spread_angle, spread_angle))
            cos_ang = math.cos(angle)
            sin_ang = math.sin(angle)
            dir_x = head_dir.x * cos_ang - head_dir.y * sin_ang
            dir_y = head_dir.x * sin_ang + head_dir.y * cos_ang
            direction = Vec3(dir_x, dir_y, 0).normalized()
            direction += Vec3(random.uniform(-0.3,0.3), random.uniform(-0.3,0.3), 0)
            direction = direction.normalized()
            
            bubble = Bubble(
                position=self.position + head_dir*0.5,
                direction=direction,
                speed=random.uniform(2,4),
                texture=self.bubble_texture,
                parent=scene
            )
            bubble.z = self.z - random.uniform(0.1,0.3)
class Phase3Scene(Entity):
    def __init__(self, fishes, **kwargs):
        super().__init__(**kwargs)
        global player_data
        self.fishes = fishes
        
        # Find fish with most descendants
        self.winner = max(fishes, key=lambda f: len(f.children), default=None)
        
        if self.winner:
            # Get winner's player data texture
            winner_texture = None
            for data in player_data:
                if data['player_num'] == self.winner.player_num:
                    winner_texture = data['texture']
                    break
                
            # Display winner's face texture
            if winner_texture:
                # Create spiky glowing background
                num_spikes = 12
                vertices = []
                uvs = []
                triangles = []
                
                # Create star/spiky shape vertices
                for i in range(num_spikes * 2):
                    angle = math.pi * 2 * i / (num_spikes * 2)
                    radius = 1.5 if i % 2 == 0 else 0.8  # Alternate between outer and inner points
                    x = math.cos(angle) * radius
                    y = math.sin(angle) * radius
                    vertices.append((x, y, 0))
                    uvs.append((x/2+0.5, y/2+0.5))
                
                # Create triangles connecting to center
                center_idx = len(vertices)
                vertices.append((0, 0, 0))
                uvs.append((0.5, 0.5))
                
                for i in range(num_spikes * 2):
                    triangles.append((i, (i+1) % (num_spikes*2), center_idx))
                
                # Create spiky glow mesh with collider
                self.glow = Entity(
                    parent=self,
                    model=Mesh(vertices=vertices, triangles=triangles, uvs=uvs),
                    color=color.yellow,
                    scale=(2, 2),
                    position=(0, 0, -0.5),
                    collider='box'  # Add collider for physics
                )
                
                # Create circular mask for winner photo with collider
                self.winner_photo = Entity(
                    parent=self,
                    model='quad', 
                    texture=winner_texture,
                    scale=(2, 2),
                    position=(0, 0, -1),
                    collider='box'  # Add collider for physics
                )
                
                # Add physics properties
                self.spin_speed = 60  # Degrees per second
                self.velocity = Vec3(random.uniform(-2,2), random.uniform(-2,2), 0)
                
                # Animate the spiky glow and handle physics
                def update_glow():
                    t = time.time()
                    pulse = math.sin(t * 2) * 0.1 + 0.9  # Pulse between 0.8 and 1.0 scale
                    
                    # Update position based on velocity
                    self.glow.position += self.velocity * time.dt
                    self.winner_photo.position = self.glow.position + Vec3(0,0,-0.5)
                    
                    # Spin both entities
                    self.glow.rotation_z += self.spin_speed * time.dt
                    self.winner_photo.rotation_z = self.glow.rotation_z
                    
                    # Bounce off screen edges
                    if abs(self.glow.x) > 5:
                        self.velocity.x *= -1
                    if abs(self.glow.y) > 3:
                        self.velocity.y *= -1
                        
                    # Check collisions with fish
                    for fish in self.fishes:
                        if fish.enabled and self.glow.intersects(fish):
                            # Bounce off fish
                            self.velocity = (self.glow.position - fish.position).normalized() * 2
                            fish.animate_scale(1.2, duration=0.1, curve=curve.out_elastic)
                            fish.animate_scale(1, duration=0.1, delay=0.1)
                    
                    # Update visual effects
                    self.glow.scale = (2.4 * pulse, 2.4 * pulse)
                    self.glow.alpha = 0.3 + math.sin(t * 2) * 0.1
                    self.winner_photo.scale = self.glow.scale
                
                self.glow.update = update_glow
        # Add title and instructions
        self.title = Text(
            text="Winner!",
            position=(0, 0.8),
            origin=(0, 0),
            color=color.yellow,
            scale=4
        )
        
        if self.winner:
            self.winner_text = Text(
                text=f"Player {self.winner.player_num} had the most offspring!",
                position=(0, -0.8),
                origin=(0, 0),
                color=color.white,
                scale=2
            )
        
        self.instructions = Text(
            text="Press 'space' to restart",
            position=(0, -0.9),
            origin=(0, 0),
            color=color.white,
            scale=2
        )
        
if __name__ == '__main__':
    window.vsync = False
    app = Ursina(
        title='Bang Bang Bubbles',
        development_mode=False,
        editor_ui=False,
        fullscreen=False
    )
    phase1 = Phase1Scene()
    scene.entities.append(phase1)
    app.run()
