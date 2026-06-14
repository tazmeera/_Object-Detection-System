import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
import threading
from datetime import datetime

# -----------------------------
# Globals
# -----------------------------
model = None
selected_image = None
is_detecting = False

OUTPUT_DIR = "detections"


# -----------------------------
# Model loading (runs in background so the GUI doesn't freeze on startup)
# -----------------------------
def load_model():
    global model
    try:
        loaded = YOLO("yolov8n.pt")
        model = loaded
        root.after(0, on_model_loaded)
    except Exception as e:
        root.after(0, lambda: on_model_load_error(str(e)))


def on_model_loaded():
    status_label.config(text="Model loaded - No image selected")
    detect_btn.config(state="normal")


def on_model_load_error(err):
    status_label.config(text="Failed to load YOLO model")
    messagebox.showerror("Model Error", f"Could not load YOLO model:\n{err}")


# -----------------------------
# Image selection
# -----------------------------
def select_image():
    global selected_image
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp")
        ]
    )
    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img.thumbnail((500, 400))
        photo = ImageTk.PhotoImage(img)
    except Exception as e:
        messagebox.showerror("Error", f"Could not open image:\n{e}")
        return

    selected_image = file_path
    image_label.config(image=photo)
    image_label.image = photo
    status_label.config(
        text=f"Selected: {os.path.basename(file_path)}"
    )


# -----------------------------
# Detection (runs in a background thread)
# -----------------------------
def detect_objects():
    global is_detecting

    if model is None:
        messagebox.showwarning("Warning", "The model is still loading, please wait a moment.")
        return

    if not selected_image:
        messagebox.showwarning(
            "Warning",
            "Please select an image first."
        )
        return

    if is_detecting:
        return

    is_detecting = True
    detect_btn.config(state="disabled", text="Detecting...")
    select_btn.config(state="disabled")
    status_label.config(text="Running detection...")

    threading.Thread(target=run_detection, daemon=True).start()


def run_detection():
    global is_detecting
    try:
        conf = confidence_slider.get() / 100.0
        results = model(selected_image, conf=conf)
        annotated = results[0].plot()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(selected_image))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_detected_{timestamp}.jpg")

        cv2.imwrite(output_path, annotated)
        num_detections = len(results[0].boxes)

        root.after(0, lambda: on_detection_done(output_path, num_detections))
    except Exception as e:
        root.after(0, lambda: on_detection_error(str(e)))


def on_detection_done(output_path, num_detections):
    global is_detecting

    img = Image.open(output_path)
    img.thumbnail((500, 400))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo

    status_label.config(
        text=f"Detection complete - {num_detections} object(s) found - Saved as {output_path}"
    )

    detect_btn.config(state="normal", text="Detect Objects")
    select_btn.config(state="normal")
    is_detecting = False

    messagebox.showinfo(
        "Success",
        f"Found {num_detections} object(s).\nOutput saved as:\n{output_path}"
    )


def on_detection_error(err):
    global is_detecting

    detect_btn.config(state="normal", text="Detect Objects")
    select_btn.config(state="normal")
    is_detecting = False

    status_label.config(text="Detection failed")
    messagebox.showerror("Error", f"Detection failed:\n{err}")


# -----------------------------
# GUI
# -----------------------------
root = tk.Tk()
root.title("Object Detection System")
root.geometry("700x720")
root.configure(bg="#1e1e2f")
root.resizable(False, False)

title = tk.Label(
    root,
    text="🎯 Object Detection System",
    font=("Segoe UI", 20, "bold"),
    bg="#1e1e2f",
    fg="#7c6ef5"
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="CodeAlpha Artificial Intelligence Internship",
    bg="#1e1e2f",
    fg="gray"
)
subtitle.pack()

# --- Buttons ---
btn_frame = tk.Frame(root, bg="#1e1e2f")
btn_frame.pack(pady=15)

select_btn = tk.Button(
    btn_frame,
    text="Select Image",
    command=select_image,
    bg="#7c6ef5",
    fg="white",
    font=("Segoe UI", 10, "bold")
)
select_btn.pack(side="left", padx=10)

detect_btn = tk.Button(
    btn_frame,
    text="Detect Objects",
    command=detect_objects,
    bg="#40a02b",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    state="disabled"  # enabled once the model finishes loading
)
detect_btn.pack(side="left", padx=10)

# --- Confidence slider ---
slider_frame = tk.Frame(root, bg="#1e1e2f")
slider_frame.pack(pady=(0, 10))

slider_label = tk.Label(
    slider_frame,
    text="Confidence Threshold:",
    bg="#1e1e2f",
    fg="#cccccc",
    font=("Segoe UI", 10)
)
slider_label.pack(side="left", padx=(0, 10))

confidence_slider = tk.Scale(
    slider_frame,
    from_=5,
    to=95,
    orient="horizontal",
    length=250,
    bg="#1e1e2f",
    fg="#cccccc",
    troughcolor="#2b2b3c",
    highlightthickness=0,
    activebackground="#7c6ef5",
    font=("Segoe UI", 9)
)
confidence_slider.set(25)  # default confidence = 0.25
confidence_slider.pack(side="left")

# --- Image display ---
image_label = tk.Label(
    root,
    bg="#2b2b3c",
    width=500,
    height=400
)
image_label.pack(pady=20)

# --- Status ---
status_label = tk.Label(
    root,
    text="Loading model, please wait...",
    bg="#1e1e2f",
    fg="#cccccc",
    font=("Segoe UI", 10)
)
status_label.pack()

# Start loading the YOLO model in the background so the window opens instantly
threading.Thread(target=load_model, daemon=True).start()

root.mainloop()