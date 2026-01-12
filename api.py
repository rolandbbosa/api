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
from tkinterdnd2 import DND_FILES, TkinterDnD
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

# --- GUI APPLICATION ---
class FacesFuckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FacesFuck💋 Studio")
        self.root.geometry("1100x850")
        self.root.configure(bg="#0f0f14") # Deep Dark Background

        self.source_path = None
        self.video_path = None
        self.is_processing = False
        self.output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "FacesFuck")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", thickness=8, troughcolor="#1e1e2e", background="#cba6f7", borderwidth=0)

    def setup_ui(self):
        # 1. Header
        self.nav_bar = tk.Frame(self.root, bg="#181825", height=70)
        self.nav_bar.pack(side="top", fill="x")
        
        tk.Label(self.nav_bar, text="FacesFuck STUDIO", fg="#cba6f7", bg="#181825", 
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=30)

        tk.Button(self.nav_bar, text="📂 VIEW CREATIONS", command=self.open_output_folder,
                  bg="#313244", fg="#cdd6f4", font=("Segoe UI", 9, "bold"),
                  relief="flat", activebackground="#45475a", cursor="hand2", padx=20, pady=8).pack(side="right", padx=30)

        # 2. Workspace
        self.work_area = tk.Frame(self.root, bg="#0f0f14")
        self.work_area.pack(expand=True, fill="both", padx=50, pady=20)

        # Drop Zone Grid
        self.grid_f = tk.Frame(self.work_area, bg="#0f0f14")
        self.grid_f.pack(pady=10, fill="x")
        self.grid_f.columnconfigure(0, weight=1); self.grid_f.columnconfigure(1, weight=1)

        self.img_panel, self.img_btn, self.img_label_frame = self.create_drop_zone("SOURCE IMAGE", 0)
        self.vid_panel, self.vid_btn, self.vid_label_frame = self.create_drop_zone("TARGET VIDEO", 1)

        # 3. Status & Progress
        self.status_container = tk.Frame(self.work_area, bg="#0f0f14")
        self.status_container.pack(pady=20, fill="x")

        self.status_var = tk.StringVar(value="Ready to Swap")
        tk.Label(self.status_container, textvariable=self.status_var, bg="#0f0f14", 
                 fg="#9399b2", font=("Segoe UI", 11)).pack()

        self.progress = ttk.Progressbar(self.status_container, orient="horizontal", length=700, mode="determinate")
        
        # 4. Action Buttons
        self.btn_container = tk.Frame(self.work_area, bg="#0f0f14")
        self.btn_container.pack(pady=20)

        self.gen_btn = tk.Button(self.btn_container, text="START GENERATION", command=self.start_processing, 
                                 bg="#a6e3a1", fg="#11111b", font=("Segoe UI", 12, "bold"),
                                 relief="flat", padx=40, pady=15, cursor="hand2", 
                                 activebackground="#94e2d5")

        self.cancel_btn = tk.Button(self.btn_container, text="CANCEL PROCESS", command=self.confirm_cancel, 
                                    bg="#f38ba8", fg="#11111b", font=("Segoe UI", 12, "bold"),
                                    relief="flat", padx=40, pady=15, cursor="hand2")

    def create_drop_zone(self, title, col):
        # Outer Frame for padding
        f = tk.Frame(self.grid_f, bg="#1e1e2e", bd=2, highlightthickness=1, highlightbackground="#313244")
        f.grid(row=0, column=col, padx=20, sticky="nsew")
        
        file_type = "img" if col == 0 else "vid"
        
        tk.Label(f, text=title, bg="#1e1e2e", fg="#bac2de", font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))

        btn = tk.Button(f, text="Choose File", bg="#45475a", fg="#cdd6f4", relief="flat",
                  font=("Segoe UI", 9), command=lambda t=file_type: self.browse_file(t),
                  activebackground="#585b70", cursor="hand2")
        btn.pack(pady=5)

        thumb = tk.Label(f, text="\n\nDrag & Drop\nFile Here", bg="#181825", fg="#585b70", 
                         width=35, height=12, font=("Segoe UI", 10))
        thumb.pack(pady=15, padx=20, expand=True, fill="both")
        
        f.drop_target_register(DND_FILES)
        f.dnd_bind('<<Drop>>', lambda e, t=file_type: self.load_file(e.data, t))
        setattr(self, f"thumb_{file_type}", thumb)
        return f, btn, f

    def toggle_inputs(self, state):
        tk_state = "normal" if state else "disabled"
        self.img_btn.config(state=tk_state)
        self.vid_btn.config(state=tk_state)
        alpha = "#1e1e2e" if state else "#11111b"
        self.img_panel.config(bg=alpha)
        self.vid_panel.config(bg=alpha)
        
        if state:
            self.img_panel.drop_target_register(DND_FILES)
            self.vid_panel.drop_target_register(DND_FILES)
        else:
            self.img_panel.drop_target_unregister()
            self.vid_panel.drop_target_unregister()

    def confirm_cancel(self):
        self.toast = tk.Frame(self.root, bg="#181825", highlightbackground="#f38ba8", highlightthickness=2)
        self.toast.place(relx=0.5, rely=0.5, anchor="center", width=400, height=200)
        
        tk.Label(self.toast, text="Abort Processing?", fg="#f38ba8", bg="#181825",
                 font=("Segoe UI", 14, "bold")).pack(pady=(30, 10))
        tk.Label(self.toast, text="All progress will be lost.", fg="#a6adc8", bg="#181825",
                 font=("Segoe UI", 10)).pack(pady=(0, 20))
        
        bf = tk.Frame(self.toast, bg="#181825")
        bf.pack()
        tk.Button(bf, text="CONTINUE", width=12, bg="#313244", fg="#cdd6f4", relief="flat", command=self.toast.destroy).pack(side="left", padx=10)
        tk.Button(bf, text="YES, ABORT", width=12, bg="#f38ba8", fg="#11111b", relief="flat", command=self.reset_app).pack(side="left", padx=10)

    def reset_app(self):
        self.is_processing = False
        if hasattr(self, 'toast'): self.toast.destroy()
        self.source_path = None
        self.video_path = None
        self.thumb_img.config(image='', text="\n\nDrag & Drop\nFile Here", bg="#181825")
        self.thumb_vid.config(image='', text="\n\nDrag & Drop\nFile Here", bg="#181825")
        self.thumb_img.image = None
        self.thumb_vid.image = None
        self.img_panel.config(highlightbackground="#313244")
        self.vid_panel.config(highlightbackground="#313244")
        self.gen_btn.pack_forget()
        self.cancel_btn.pack_forget()
        self.progress.pack_forget()
        self.status_var.set("Ready to Swap")
        self.toggle_inputs(True)

    def open_output_folder(self):
        os.startfile(self.output_dir) if os.name == 'nt' else subprocess.call(['open', self.output_dir])

    def browse_file(self, type):
        if type == "img":
            f = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.webp")])
        else:
            f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if f: self.load_file(f, type)

    def load_file(self, path, type):
        path = path.strip('{}')
        ext = os.path.splitext(path)[1].lower()
        
        if type == "img":
            if ext not in ['.jpg', '.jpeg', '.png', '.webp']: return
            self.source_path = path
            self.update_thumbnail(path, self.thumb_img)
            self.img_panel.config(highlightbackground="#cba6f7")
        else:
            if ext not in ['.mp4', '.avi', '.mov', '.mkv']: return
            self.video_path = path
            self.update_thumbnail(path, self.thumb_vid, is_video=True)
            self.vid_panel.config(highlightbackground="#cba6f7")
        
        if self.source_path and self.video_path:
            self.gen_btn.pack(pady=10)

    def update_thumbnail(self, path, label, is_video=False):
        try:
            if is_video:
                cap = cv2.VideoCapture(path); ret, frame = cap.read(); cap.release()
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(path)
            img.thumbnail((300, 250))
            photo = ImageTk.PhotoImage(img); label.config(image=photo, text="", bg="#11111b"); label.image = photo
        except: label.config(text="File Loaded")

    def start_processing(self):
        self.is_processing = True
        self.toggle_inputs(False) 
        self.gen_btn.pack_forget() 
        self.cancel_btn.pack()
        self.progress.pack(pady=10)
        self.progress['value'] = 0
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        try:
            random_name = f"Swap_{''.join(random.choices(string.digits, k=6))}.mp4"
            final_path = os.path.join(self.output_dir, random_name)
            temp_silent = "temp_processing.mp4"

            self.status_var.set("Initialising Face Mesh Engine...")
            img_src_raw = apply_visibility_boost(cv2.imread(self.source_path))
            pts_src = get_landmarks(img_src_raw)
            if pts_src is None: raise Exception("No face found in source!")

            hull_src = cv2.convexHull(pts_src)
            rect = cv2.boundingRect(hull_src)
            subdiv = cv2.Subdiv2D(rect)
            for p in pts_src: subdiv.insert((float(p[0]), float(p[1])))
            triangles = subdiv.getTriangleList()

            idx_tri = []
            for t in triangles:
                pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
                i1 = np.where((pts_src == pt1).all(axis=1))[0]
                i2 = np.where((pts_src == pt2).all(axis=1))[0]
                i3 = np.where((pts_src == pt3).all(axis=1))[0]
                if i1.size > 0 and i2.size > 0 and i3.size > 0:
                    idx_tri.append([i1[0], i2[0], i3[0]])

            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(3)), int(cap.get(4))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            count = 0
            prev_pts_dst = None

            while cap.isOpened() and self.is_processing:
                ret, frame = cap.read()
                if not ret: break
                
                curr_pts_dst = get_landmarks(frame)
                if curr_pts_dst is not None:
                    pts_dst = (curr_pts_dst * 0.75 + prev_pts_dst * 0.25).astype(np.int32) if prev_pts_dst is not None else curr_pts_dst
                    prev_pts_dst = pts_dst
                    img_source = color_transfer(img_src_raw, frame)
                    canvas_warped = np.zeros_like(frame)
                    
                    for tri_idx in idx_tri:
                        s_tri, d_tri = pts_src[tri_idx], pts_dst[tri_idx]
                        r_s, r_d = cv2.boundingRect(s_tri), cv2.boundingRect(d_tri)
                        stc, dtc = s_tri - (r_s[0], r_s[1]), d_tri - (r_d[0], r_d[1])
                        img_s_c = img_source[r_s[1]:r_s[1]+r_s[3], r_s[0]:r_s[0]+r_s[2]]
                        if img_s_c.shape[0] > 0 and img_s_c.shape[1] > 0:
                            warp = apply_affine_transform(img_s_c, stc, dtc, (r_d[2], r_d[3]))
                            mask = np.zeros((r_d[3], r_d[2], 3), dtype=np.uint8)
                            cv2.fillConvexPoly(mask, dtc, (255, 255, 255))
                            roi = canvas_warped[r_d[1]:r_d[1]+r_d[3], r_d[0]:r_d[0]+r_d[2]]
                            if roi.shape[0] == warp.shape[0] and roi.shape[1] == warp.shape[1]:
                                canvas_warped[r_d[1]:r_d[1]+r_d[3], r_d[0]:r_d[0]+r_d[2]] = \
                                    cv2.bitwise_and(roi, cv2.bitwise_not(mask)) + cv2.bitwise_and(warp, mask)

                    hull_dst = cv2.convexHull(pts_dst)
                    mask_dst = np.zeros_like(frame)
                    cv2.fillConvexPoly(mask_dst, hull_dst, (255, 255, 255))
                    mask_dst = cv2.GaussianBlur(cv2.erode(mask_dst, np.ones((5,5), np.uint8)), (11, 11), 0)
                    try:
                        frame = cv2.seamlessClone(canvas_warped, frame, mask_dst, (cv2.boundingRect(hull_dst)[0] + cv2.boundingRect(hull_dst)[2] // 2, cv2.boundingRect(hull_dst)[1] + cv2.boundingRect(hull_dst)[3] // 2), cv2.NORMAL_CLONE)
                    except: pass

                out.write(frame)
                count += 1
                self.progress['value'] = (count / total_frames) * 100
                self.status_var.set(f"Swapping Frames: {count} / {total_frames}")
                self.root.update_idletasks()

            cap.release(); out.release()
            if not self.is_processing:
                if os.path.exists(temp_silent): os.remove(temp_silent)
                return

            self.status_var.set("Syncing Audio & Finalizing...")
            proc = VideoFileClip(temp_silent); orig = VideoFileClip(self.video_path)
            proc.set_audio(orig.audio).write_videofile(final_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            proc.close(); orig.close()
            if os.path.exists(temp_silent): os.remove(temp_silent)
            
            messagebox.showinfo("Complete", f"Successfully exported to:\n{random_name}")
            self.reset_app()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.reset_app()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = FacesFuckApp(root)
    root.mainloop()
