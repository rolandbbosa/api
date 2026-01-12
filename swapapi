import cv2
import mediapipe as mp
import numpy as np
import os
import random
import string
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES
from moviepy.editor import VideoFileClip
import subprocess

# --- CORE LOGIC ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def apply_visibility_boost(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.6, gaussian_3, -0.6, 0)
    gamma = 1.05
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    return img

def color_transfer(source, target):
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    s_mean, s_std = cv2.meanStdDev(src_lab)
    t_mean, t_std = cv2.meanStdDev(tgt_lab)
    s_mean, s_std = s_mean.reshape((1, 1, 3)), s_std.reshape((1, 1, 3))
    t_mean, t_std = t_mean.reshape((1, 1, 3)), t_std.reshape((1, 1, 3))
    res_lab = (src_lab - s_mean) * (t_std / (s_std + 1e-5)) + t_mean
    res_lab = np.clip(res_lab, 0, 255).astype("uint8")
    return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

def get_landmarks(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    if not results.multi_face_landmarks:
        return None
    h, w, _ = image.shape
    return np.array([(int(l.x * w), int(l.y * h)) for l in results.multi_face_landmarks[0].landmark], np.int32)

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# --- GUI APPLICATION CLASS ---
class FacesFuckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FacesFuck💋 Studio")
        self.root.geometry("1100x850")
        self.root.configure(bg="#0f0f14")

        self.source_path = None
        self.video_path = None
        self.is_processing = False
        self.output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "FacesFuck")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.setup_styles()
        self.setup_ui()

    # ... [Include all other methods like setup_ui, process_video, etc. from your original code here] ...
    # (Omitted here for brevity, but keep them in your uploaded file)
