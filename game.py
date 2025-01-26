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
        
        # Disable auto color correction, white balance and flicker detection
        self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Disable auto exposure
        self.capture.set(cv2.CAP_PROP_MONOCHROME, 0)  # Disable monochrome mode
        self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Disable color conversion
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        
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
        self.red_cooldown = 0
        self.green_cooldown = 0
        self.collision_cooldown = {}  # Track collision cooldowns per fish pair
        self.start_time = time.time()  # Track when Phase2 starts
        
        # Create aquarium background
        self.background = Entity(model='quad', texture='assets/water', scale=(16, 9), z=1)
        # self.background = Entity(model='quad', texture='assets/aquarium_background_2.mp4', scale=(16, 9), z=1)
        
        # Spawn fish
        for data in self.player_data:
            if data['texture']:  # Only create fish if texture exists
                fish = Fish(player_num=data['player_num'], generation=0)
                self.fishes.append(fish)
                
        # Add editor camera
        EditorCamera()

    def update(self):
        # Check for 'r' key to spawn new fish for player 1
        if held_keys['r']:
            new_fish = Fish(player_num=1, generation=0)
            self.fishes.append(new_fish)
        
        # Skip collision checks for first 3 seconds
        if time.time() - self.start_time < 3:
            return
            
        # Check for collisions between fish
        for i, fish1 in enumerate(self.fishes):
            for j, fish2 in enumerate(self.fishes[i+1:], i+1):
                # Skip if same player's fish
                if fish1.player_num == fish2.player_num:
                    continue
                    
                # Create unique key for this fish pair
                pair_key = tuple(sorted([id(fish1), id(fish2)]))
                
                # Check if cooldown has expired
                if pair_key in self.collision_cooldown:
                    if time.time() - self.collision_cooldown[pair_key] < 2:  # 2 second cooldown
                        continue
                        
                # Check distance between fish
                distance = (fish1.position - fish2.position).length()
                if distance < 1:  # Collision threshold
                    # Spawn new fish with higher generation
                    new_gen = max(fish1.generation, fish2.generation) + 1
                    new_fish = Fish(player_num=fish1.player_num, generation=new_gen)
                    new_fish.position = fish1.position
                    self.fishes.append(new_fish)
                    
                    # Set cooldown for this pair
                    self.collision_cooldown[pair_key] = time.time()
                    # Also add created fish to collision cooldown
                    # Add cooldown for new fish with each existing fish
                    for fish in self.fishes:
                        if fish != new_fish:
                            pair_key = tuple(sorted([id(new_fish), id(fish)]))
                            self.collision_cooldown[pair_key] = time.time()
                    
                    print(f"Collision! Spawned new fish with generation {new_gen}")
                    
        
        # Color detection loop
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip horizontally
            h, w = frame.shape[:2]
            
            # Process each player
            for data in self.player_data:
                player_num = data['player_num']
                bbox = data['bbox']
                
                # Calculate ROI coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h) 
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Check valid ROI dimensions
                if x >= 0 and y >= 0 and width > 0 and height > 0:
                    try:
                        # Extract and process face region
                        face_roi = frame[y:y+height, x:x+width]
                        if face_roi.size > 0:
                            detected_color = self.color_detector.detect_color(face_roi, player_num)
                            if detected_color == 'red':
                                self.handle_red_action(player_num)
                            elif detected_color == 'green':
                                self.handle_green_action(player_num)
                    except:
                        pass

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

class ColorDetector:
    def __init__(self):
        pass 
    
    def detect_color(self, roi, player_num):
        # Convert BGR to RGB
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pixels = rgb_roi.reshape(-1, 3)  # Reshape to list of pixels
        
        # Calculate standard deviation for each channel
        r_std = np.std(pixels[:, 0])
        g_std = np.std(pixels[:, 1]) 
        b_std = np.std(pixels[:, 2])
        
        # Calculate average standard deviation across channels
        avg_std = (r_std + g_std + b_std) / 3
        
        # Only proceed if pixels are similar in color (low standard deviation)
        if avg_std < 20:  # Threshold for color similarity
            total_pixels = len(pixels)
            
            # Count pixels where red channel > 150 and other channels < 100
            red_mask = (pixels[:, 0] > 120) & (pixels[:, 1] < 100) & (pixels[:, 2] < 100)
            red_pixels = np.sum(red_mask)
            red_percentage = red_pixels / total_pixels
            
            # Count pixels where green channel > 120 and other channels < 100
            green_mask = (pixels[:, 1] > 120) & (pixels[:, 0] < 100) & (pixels[:, 2] < 100)
            green_pixels = np.sum(green_mask)
            green_percentage = green_pixels / total_pixels

            print(f"Player {player_num} - Red percentage: {red_percentage}, Green percentage: {green_percentage}")
            
            if red_percentage > 0.3:
                return 'red'
            elif green_percentage > 0.3:
                return 'green'
        
        return None
    
class Fish(Entity):
    def __init__(self, player_num, generation=0, **kwargs):
        # Map generation to model file
        model_files = {
            0: 'assets/red_fish.glb',
            1: 'assets/orange_fish.glb', 
            2: 'assets/yellow_fish.glb',
            3: 'assets/green_fish.glb'
        }
        model_file = model_files.get(min(generation, 3))  # Default to green for generation > 3
        
        super().__init__(model=model_file, scale=1, collider='mesh')

        self.player_num = player_num
        self.generation = generation
        # Calculate initial direction vector based on model's left side
        self.forward = Vec3(1, 0, 0)  # Model faces right initially
        self.speed = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0).normalized() * 0.02
        self.acceleration = 1.0
        self.dash_duration = 0
        self.rotation_x = 90
        
        # Add text entity to show player number
        self.player_text = Text(
            text=str(player_num),
            parent=self,
            y=-0.20,
            color=color.white,
            scale=6,
            origin=(0,0),
            # background=True,
            # background_color=color.bla
        )
        self.player_text.billboard = True  # Make text always face camera

    def update(self):
        # Handle movement with acceleration
        if self.dash_duration > 0:
            self.acceleration = min(self.acceleration + 0.1, 3.0)
            self.dash_duration -= time.dt
            if self.dash_duration <= 0:
                self.reset_speed()
                
        self.position += self.speed * self.acceleration
        
        # Wrap around screen edges
        if self.x > 7:
            self.x = -7
        elif self.x < -7:
            self.x = 7
        if self.y > 4:
            self.y = -4
        elif self.y < -4:
            self.y = 4
            
    def dash(self):
        # Convert local forward vector to world space based on current rotation
        # world_forward = self.forward.rotate(self.rotation_z)
        world_forward = self.forward.rotate(self.rotation_z)
        self.speed = world_forward.normalized() * 0.02
        # self.speed = self.forward * 0.02
        
        self.dash_duration = 0.5
        
    def reset_speed(self):
        self.speed = self.speed.normalized() * 0.02
        self.acceleration = 1.0
        
    def rotate(self, direction):
        self.rotation_z += 15 * direction * time.dt
        # Update speed direction when rotating
        # world_forward = self.forward.rotate(self.rotation_z)
        # self.speed = world_forward.normalized() * 0.02

    def look_at_closest_fish(self, fish_list):
        if not fish_list:
            return
            
        # Find closest fish
        closest_fish = None
        min_distance = float('inf')
        for fish in fish_list:
            if fish == self:  # Skip self
                continue
            distance = (fish.position - self.position).length()
            if distance < min_distance:
                min_distance = distance
                closest_fish = fish
                
        if closest_fish:
            # Look at closest fish along z-axis
            self.look_at(closest_fish.position, axis='z')
            
if __name__ == '__main__':
    app = Ursina()
    window.fullscreen = False
    phase1 = Phase1Scene()
    scene.entities.append(phase1)
    app.run()
