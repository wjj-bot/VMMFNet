import os
import sys
import numpy as np
import rasterio
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.path import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
import json
import glob
import re
import warnings
import threading
from collections import defaultdict

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', module='rasterio')
warnings.filterwarnings('ignore', category=UserWarning)

INPUT_DIR = r"D:\gisdata\testdata\test"
OUTPUT_DIR = r"D:\gisdata\testdata\testresult"
REF_VALUES = np.array([0.75, 0.50, 0.25])
BANDS_TO_PROCESS = ['Green', 'NIR', 'Red', 'RedEdge']
BAND_SUFFIXES = {
    'Green': '_MS_G.tif',
    'NIR': '_MS_NIR.tif',
    'Red': '_MS_R.tif',
    'RedEdge': '_MS_RE.tif'
}

class ThreadSafeFigure:
    def __init__(self, figure, block=True):
        self.figure = figure
        self.block = block
        self.completed = threading.Event()

    def show(self):
        if threading.current_thread() != threading.main_thread():
            def run_figure():
                plt.figure(self.figure.number)
                plt.show(block=True)
                self.completed.set()
            root = tk.Tk()
            root.withdraw()
            root.after(0, run_figure)
            self.completed.wait()
            root.destroy()
        else:
            plt.figure(self.figure.number)
            plt.show(block=self.block)
            self.completed.set()


class PolygonRegionSelector:
    def __init__(self, image, band_name, num_points=4):
        self.fig, self.ax = plt.subplots(figsize=(12, 10), num=f"PolygonRegionSelector_{band_name}")
        self.image = image
        self.band_name = band_name
        self.region_means = []
        self.region_coords = []
        self.selected_regions = []
        self.current_points = []
        self.num_points = num_points
        self.patches = []
        vmin = np.percentile(image, 1)
        vmax = np.percentile(image, 99)
        self.ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        self.ax.set_title(f"{band_name} Calibration Panel - Select Region 1/3")
        self.ax.axis('off')
        self.status_text = self.ax.text(
            0.98, 0.98, "Select four points (0/4)",
            transform=self.ax.transAxes, fontsize=12,
            ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7)
        )
        btn_ax = plt.axes([0.65, 0.05, 0.15, 0.05])
        self.undo_btn = Button(btn_ax, 'Undo')
        self.undo_btn.on_clicked(self.undo_point)
        btn_ax2 = plt.axes([0.82, 0.05, 0.15, 0.05])
        self.clear_btn = Button(btn_ax2, 'Reset')
        self.clear_btn.on_clicked(self.clear_current)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        guide = (
            "Instructions:\n"
            "1. Click four points to define the polygon\n"
            "2. Press Enter to confirm\n"
            "3. Select all 3 regions (75% → 50% → 25%)"
        )
        self.ax.text(
            0.02, 0.02, guide,
            transform=self.ax.transAxes, fontsize=11,
            ha='left', va='bottom', bbox=dict(facecolor='yellow', alpha=0.5)
        )
        plt.tight_layout()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if len(self.current_points) >= self.num_points:
            self.status_text.set_text("Four points selected. Press Enter to confirm.")
            return
        x, y = event.xdata, event.ydata
        self.current_points.append((x, y))
        pt = self.ax.scatter(x, y, s=50, c='red', edgecolor='white')
        self.patches.append(pt)
        idx = len(self.current_points)
        txt = self.ax.text(x + 15, y - 15, str(idx), color='white', fontsize=10,
                           bbox=dict(facecolor='red', alpha=0.8))
        self.patches.append(txt)
        if len(self.current_points) >= 2:
            pts = np.array(self.current_points)
            line = self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.5)[0]
            self.patches.append(line)
            if len(self.current_points) == self.num_points:
                pts = np.array(self.current_points)
                poly = matplotlib.patches.Polygon(
                    pts, closed=True, facecolor='none',
                    edgecolor='orange', linewidth=2.5, alpha=0.8
                )
                self.ax.add_patch(poly)
                self.patches.append(poly)
        self.status_text.set_text(f"Selected points: {len(self.current_points)}/4")
        self.fig.canvas.draw_idle()

    def undo_point(self, event=None):
        if not self.current_points:
            return
        self.current_points.pop()
        while self.patches:
            obj = self.patches.pop()
            try:
                obj.remove()
            except:
                pass
        for i, (x, y) in enumerate(self.current_points):
            pt = self.ax.scatter(x, y, s=50, c='red', edgecolor='white')
            self.patches.append(pt)
            txt = self.ax.text(
                x + 15, y - 15, str(i + 1),
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.8)
            )
            self.patches.append(txt)
        if len(self.current_points) >= 2:
            pts = np.array(self.current_points)
            line = self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.5)[0]
            self.patches.append(line)
        self.status_text.set_text(f"Selected points: {len(self.current_points)}/4")
        self.fig.canvas.draw_idle()

    def clear_current(self, event=None):
        while self.patches:
            obj = self.patches.pop()
            try:
                obj.remove()
            except:
                pass
        self.current_points = []
        self.status_text.set_text("Reset. Please select four points again.")
        self.fig.canvas.draw_idle()

    def confirm_current_region(self):
        if len(self.current_points) < 4:
            self.status_text.set_text("Four points required.")
            return False
        poly_path = Path(self.current_points)
        h, w = self.image.shape
        yy, xx = np.mgrid[:h, :w]
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        mask = poly_path.contains_points(pts).reshape(h, w)
        region = self.image[mask]
        if region.size == 0:
            self.status_text.set_text("Invalid region.")
            return False
        mean_val = np.mean(region)
        idx = len(self.selected_regions)
        sel = {
            'points': np.copy(self.current_points),
            'mean': mean_val,
            'ref_idx': idx,
            'ref_value': REF_VALUES[idx]
        }
        self.selected_regions.append(sel)
        cx = np.mean([p[0] for p in self.current_points])
        cy = np.mean([p[1] for p in self.current_points])
        lbl = f"{REF_VALUES[idx]*100:.0f}%\n{mean_val:.4f}"
        self.ax.text(cx, cy, lbl, ha='center', va='center',
                     color='white', fontsize=11,
                     bbox=dict(facecolor='blue', alpha=0.8))
        self.current_points = []
        self.patches = []
        if len(self.selected_regions) < 3:
            nxt = REF_VALUES[len(self.selected_regions)] * 100
            self.status_text.set_text(
                f"Region saved. Select next region ({nxt:.0f}%)"
            )
            self.ax.set_title(
                f"{self.band_name} Calibration Panel - Select Region {len(self.selected_regions)+1}/3"
            )
        else:
            self.status_text.set_text("All 3 regions completed. Press Enter to finish.")
        self.fig.canvas.draw_idle()
        return True

    def on_key(self, event):
        if event.key == 'enter':
            if len(self.current_points) == 4:
                ok = self.confirm_current_region()
                if ok and len(self.selected_regions) == 3:
                    plt.close(self.fig)
        elif event.key == 'backspace':
            self.undo_point()
        elif event.key == 'delete':
            self.clear_current()

    def get_results(self):
        if len(self.selected_regions) < 3:
            raise ValueError("Three regions required.")
        means = [r['mean'] for r in self.selected_regions]
        coords = [r['points'] for r in self.selected_regions]
        return np.array(means), coords

    def show(self):
        ts = ThreadSafeFigure(self.fig, block=True)
        ts.show()
        return ts.completed.is_set()


class RectangleRegionSelector:
    def __init__(self, image, band_name):
        self.image = image
        self.band_name = band_name
        self.means = []
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.imshow(image, cmap='gray')
        self.ax.set_title(f"{band_name} Select Three Rectangular Regions (75% → 50% → 25%)")
        self.ax.axis('off')
        self.selector = RectangleSelector(self.ax, self.on_select, interactive=True)

    def on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        region = self.image[ymin:ymax+1, xmin:xmax+1]
        mean_dn = np.mean(region)
        self.means.append(mean_dn)
        print(f"{self.band_name} Region {len(self.means)} Mean {mean_dn:.4f}")
        if len(self.means) == 3:
            plt.close(self.fig)

    def run(self):
        plt.show()
        if len(self.means) != 3:
            raise ValueError("Three regions required")
        return self.means


class CalibrationApp(tk.Tk):
    def __init__(self, bands):
        super().__init__()
        self.title("Multispectral Calibration Panel Selection")
        self.geometry("550x320")
        self.bands = bands
        self.cal_paths = {}
        if sys.platform == "win32":
            self.option_add("*Font", "Arial 10")
        elif sys.platform == "darwin":
            self.option_add("*Font", "Arial Unicode MS 12")
        else:
            self.option_add("*Font", "Noto Sans CJK SC 10")
        self.build_ui()

    def build_ui(self):
        main = tk.Frame(self, padx=15, pady=15)
        main.pack(fill=tk.BOTH, expand=True)
        tk.Label(main, text="Select panel images for each band", font=("", 11, "bold")).pack(pady=(0, 15))
        box = tk.Frame(main)
        box.pack(fill=tk.X, padx=5, pady=5)
        self.path_entries = {}
        for b in self.bands:
            row = tk.Frame(box)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=f"{b}:", width=8, anchor='e').pack(side=tk.LEFT)
            ent = tk.Entry(row, width=40)
            ent.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.path_entries[b] = ent
            tk.Button(row, text="Browse",
                      command=lambda x=b: self.select_file(x),
                      width=8).pack(side=tk.LEFT, padx=(5, 0))
        btn = tk.Frame(main, pady=15)
        btn.pack(fill=tk.X)
        tk.Button(btn, text="Start", command=self.confirm,
                  bg="#4CAF50", fg="white", width=10).pack(side=tk.RIGHT, padx=(10, 5))
        tk.Button(btn, text="Cancel", command=self.destroy,
                  bg="#f44336", fg="white", width=8).pack(side=tk.RIGHT)

    def select_file(self, band):
        d = os.path.dirname(self.path_entries[band].get() or INPUT_DIR)
        path = filedialog.askopenfilename(parent=self, title=f"Select {band} panel",
                                          filetypes=[("TIFF", "*.tif")],
                                          initialdir=d)
        if path:
            self.path_entries[band].delete(0, tk.END)
            self.path_entries[band].insert(0, path)
            self.cal_paths[band] = path

    def confirm(self):
        missing = []
        for b in self.bands:
            p = self.path_entries[b].get()
            if not p:
                missing.append(b)
            else:
                self.cal_paths[b] = p
        if missing:
            messagebox.showerror("Error", f"Missing: {', '.join(missing)}")
            return
        wrong = [f"{b}: {os.path.basename(p)}"
                 for b, p in self.cal_paths.items()
                 if not os.path.exists(p)]
        if wrong:
            messagebox.showerror("Error", "\n".join(wrong))
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cfg = os.path.join(OUTPUT_DIR, "calibration_config.json")
        with open(cfg, 'w') as f:
            json.dump(self.cal_paths, f, indent=4)
        self.destroy()

    def run(self):
        self.mainloop()


def load_calibration_config():
    cfg = os.path.join(OUTPUT_DIR, "calibration_config.json")
    if os.path.exists(cfg):
        with open(cfg, 'r') as f:
            data = json.load(f)
            if all(b in data for b in BANDS_TO_PROCESS):
                return data
    return None
