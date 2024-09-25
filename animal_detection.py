import torch
import cv2
import os
from playsound import playsound
from tkinter import *
from tkinter import messagebox
from tkinter import scrolledtext
from PIL import Image, ImageTk

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # small version of YOLOv5

# Animal-sound mapping
ANIMAL_SOUNDS = {
    'cat': 'sounds/cat_scary_sound.mp3',
    'dog': 'sounds/dog_scary_sound.mp3',
    'bird': 'sounds/bird_scary_sound.mp3'
}

# Global variable for camera
cap = None
is_running = False

# Function to play sound based on detected animal
def play_sound(animal):
    sound_file = ANIMAL_SOUNDS.get(animal)
    if sound_file and os.path.exists(sound_file):
        log(f'Playing sound for {animal}...')
        playsound(sound_file)
    else:
        log(f'No sound found for {animal}.')

# Function to start detection
def start_detection():
    global is_running, cap
    if is_running:
        messagebox.showinfo("Info", "Detection is already running.")
        return
    
    cap = cv2.VideoCapture(0)  # Initialize webcam or video file
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source.")
        return

    is_running = True
    detect_animals()

# Function to stop detection
def stop_detection():
    global is_running, cap
    is_running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()

# Function to detect animals and update the GUI
def detect_animals():
    global is_running, cap
    
    if not is_running:
        return
    
    ret, frame = cap.read()
    
    if not ret:
        log("Error: Could not read frame.")
        stop_detection()
        return
    
    # YOLOv5 model inference
    results = model(frame)
    
    # Render detections on the frame
    annotated_frame = results.render()[0]

    # Convert frame (BGR to RGB) for Tkinter display
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_frame)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    # Update the video panel
    video_panel.config(image=img_tk)
    video_panel.image = img_tk
    
    # Process detected objects
    for result in results.xyxy[0]:
        # Extract label for the detected object
        label = model.names[int(result[5])]
        
        # Check if the detected object is in our list of animals
        if label in ANIMAL_SOUNDS:
            log(f"Detected: {label}")
            play_sound(label)
    
    # Schedule the next frame
    root.after(10, detect_animals)

# Function to log messages in the GUI
def log(message):
    log_area.config(state=NORMAL)
    log_area.insert(END, message + "\n")
    log_area.config(state=DISABLED)
    log_area.yview(END)

# Create the main GUI window
root = Tk()
root.title("YOLO Animal Detection System")
root.geometry("800x600")

# Create a video panel for displaying the video feed
video_panel = Label(root)
video_panel.pack(pady=10)

# Start and Stop buttons
start_btn = Button(root, text="Start Detection", command=start_detection, width=15)
start_btn.pack(pady=5)

stop_btn = Button(root, text="Stop Detection", command=stop_detection, width=15)
stop_btn.pack(pady=5)

# Scrollable log area for displaying detected animals and status
log_area = scrolledtext.ScrolledText(root, width=80, height=10, state=DISABLED)
log_area.pack(pady=10)

# Start the GUI event loop
root.mainloop()

