import tkinter as tk
import os
import json
from tkinter import filedialog, messagebox, ttk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nilearn.glm.first_level.hemodynamic_models import glover_hrf  

# THEME_COLOR = "#dfe9d1"

COLOR_THEMES = {
    "Light": {
        "bg": "#dfe9d1",
        "fg": "#000000",
        "entry_bg": "#ffffff",
        "button_bg": "#f2f2f2"
    },
    "Dark": {
        "bg": "#2e2e2e",
        "fg": "#ffffff",
        "entry_bg": "#444444",
        "button_bg": "#555555"
    }
}


def place_near_master(child_window, master_window, x_offset=20, y_offset=30):
    master_window.update_idletasks()
    x = master_window.winfo_rootx()
    y = master_window.winfo_rooty()
    w = master_window.winfo_width()
    child_window.geometry(f"+{x + w + x_offset}+{y + y_offset}")
    child_window.transient(master_window)
    child_window.lift()
    child_window.attributes("-topmost", True)


# def build_brain_mask_from_template(template_3d: np.ndarray,
#                                    affine=None,
#                                    percentile: float = 10.0) -> np.ndarray:


#     try:
#         from nilearn.masking import compute_brain_mask
#         import nibabel as nib
#         if isinstance(template_3d, np.ndarray):
#             img = nib.Nifti1Image(template_3d.astype(np.float32), affine if affine is not None else np.eye(4))
#         else:
#             img = template_3d
#         mask_img = compute_brain_mask(img)
#         return mask_img.get_fdata().astype(bool)
#     except Exception:
#         pass

#     data = np.asarray(template_3d)
#     pos = data[data > 0]
#     thr = np.percentile(pos, percentile) if pos.size > 0 else 0.0
#     mask = data > thr

#     try:
#         from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label
#         mask = binary_opening(mask, iterations=1)
#         mask = binary_closing(mask, iterations=1)
#         mask = binary_fill_holes(mask)
#         lab, n = label(mask)
#         if n > 0:
#             sizes = np.bincount(lab.ravel())
#             sizes[0] = 0
#             mask = (lab == sizes.argmax())  
#     except Exception:
      
#         pass

#     return mask

def add_masked_gaussian_noise(fmri_data: np.ndarray, mask: np.ndarray, std: float, seed=None) -> np.ndarray:
    """
    Добавляет Гауссов шум только внутри mask (вне маски сигнал не меняется).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, std, fmri_data.shape)
    else:
        noise = np.random.normal(0.0, std, fmri_data.shape)

    mask3d = mask.astype(bool)
    if mask3d.ndim == 4:
        mask3d = mask3d.any(axis=3)
    mask4d = mask3d[..., None]

    return fmri_data + noise * mask4d
def add_linear_trend(
    fmri_data: np.ndarray,
    z_per_sec: float,
    tr: float,
    mask: np.ndarray | None = None,
    auto_mask_percentile: float = 5.0,
    order: int = 1,
) -> np.ndarray:
    """
    Добавляет полиномиальный тренд внутри mask:
    increment(t) = (z_per_sec * tr) * (t ** order)
    где order — положительное целое 
    """
    if fmri_data.ndim != 4:
        raise ValueError("fmri_data has to be 4D (X, Y, Z, T).")
    if tr <= 0:
        raise ValueError("TR has to be > 0.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer (>=1).")

    X, Y, Z, T = fmri_data.shape
    data = fmri_data.copy().astype(np.float32, copy=False)

    # Маска
    if mask is None:
        mean_img = np.nanmean(data, axis=3)
        pos = mean_img[mean_img > 0]
        if pos.size == 0:
            mask3d = np.ones((X, Y, Z), dtype=bool)
        else:
            thr = np.percentile(pos, auto_mask_percentile)
            mask3d = mean_img > thr
    else:
        mask3d = mask.astype(bool)
        if mask3d.ndim == 4:
            mask3d = mask3d.any(axis=3)

    # Тренд с порядком
    delta = float(z_per_sec) * float(tr)
    t = np.arange(T, dtype=np.float32)
    trend = delta * (t ** order)                
    trend = trend.reshape((1, 1, 1, T))

    mask4d = mask3d[..., None]
    data += trend * mask4d
    return data


def add_gaussian_noise(fmri_data, mean=0.0, std=0.01, seed=None, trend_value=0.0, tr=2.0):
 
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(mean, std, fmri_data.shape)
    else:
        noise = np.random.normal(mean, std, fmri_data.shape)

    if trend_value != 0.0:
        fmri_data = add_linear_trend(fmri_data, trend_value, tr)

    return fmri_data + noise
    



class Template_shower:
    def __init__(self):
        self.mask_bounds = None
        self.mask = None

    def open_input_window(self, master, img, text):
        input_window = tk.Toplevel(master)
        input_window.title(text)
        

        # Расположить рядом с главным окном
        master.update_idletasks()
        x = master.winfo_rootx()
        y = master.winfo_rooty()
        w = master.winfo_width()
        input_window.geometry(f"+{x + w + 20}+{y}")

        # Приоритет и независимость
        input_window.transient(master)
        input_window.lift()
        input_window.attributes("-topmost", True)

        left_frame = tk.Frame(input_window)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(input_window)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        axial_slider = tk.Scale(right_frame, from_=0, to=img.shape[2] - 1, orient=tk.HORIZONTAL, length=600)
        coronal_slider = tk.Scale(right_frame, from_=0, to=img.shape[1] - 1, orient=tk.HORIZONTAL, length=600)
        sagittal_slider = tk.Scale(right_frame, from_=0, to=img.shape[0] - 1, orient=tk.HORIZONTAL, length=600)

        axial_slider.pack()
        coronal_slider.pack()
        sagittal_slider.pack()

        def update_slices():
            axial_slice = img.get_fdata()[:, :, axial_slider.get()]
            coronal_slice = img.get_fdata()[:, coronal_slider.get(), :]
            sagittal_slice = img.get_fdata()[sagittal_slider.get(), :, :]

            ax[0].imshow(axial_slice.T, cmap='gray', origin='lower')
            ax[0].set_title('Axial Slice')

            ax[1].imshow(coronal_slice.T, cmap='gray', origin='lower')
            ax[1].set_title('Coronal Slice')

            ax[2].imshow(sagittal_slice.T, cmap='gray', origin='lower')
            ax[2].set_title('Sagittal Slice')

            fig.canvas.draw()

        axial_slider.config(command=lambda x: update_slices())
        coronal_slider.config(command=lambda x: update_slices())
        sagittal_slider.config(command=lambda x: update_slices())

        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        update_slices()

    def import_mask(self, mask_path):
        self.mask = nib.load(mask_path)


class RoI_data:
    
    def __init__(self, roi_values=None):
        if roi_values is None:
            roi_values = [[] for _ in range(10)]
        self.n_roi = 10
        self.roi_status = ["Not Set"] * self.n_roi
        self.list_roi = [0] * 10
        self.roi_maps = [None] * 10
        self.roi_values = roi_values

    def open_main_window(self, master):
        input_window = tk.Toplevel(master)
        input_window.title("Enter ROI Information")

        master.update_idletasks()
        main_height = master.winfo_height()
        input_window.geometry(f"400x{main_height}")
        input_window.resizable(False, False)
        app.apply_theme(input_window)
        place_near_master(input_window, master)

        self.omw_place = input_window

        top_frame = tk.Frame(input_window, bg=COLOR_THEMES[app.current_theme]["bg"])
        top_frame.pack(pady=20)

        # Label: делаемт одинаковый размер по высоте
        self.roi_num_label = tk.Label(top_frame, text="ROIs # (up to 10):", font=("Segoe UI", 11, "bold"), height=2)
        self.roi_num_label.grid(row=0, column=0, padx=5, sticky="e")

        # Entry: увеличиваемт высоту за счёт шрифта и вставки в Frame с выравниванием
        self.roi_num = tk.Entry(top_frame, width=10, font=("Segoe UI", 12))
        self.roi_num.grid(row=0, column=1, padx=5, pady=5)

        # Submit: делает такой же высоты и ширины
        self.submit_button = tk.Button(top_frame, text="Submit", command=self.add_roi_entries, width=12, height=2,
                                    font=("Segoe UI", 11, "bold"))
        self.submit_button.grid(row=0, column=2, padx=5)

        bottom_frame = tk.Frame(input_window, bg=COLOR_THEMES[app.current_theme]["bg"])
        bottom_frame.pack(side=tk.BOTTOM, pady=20)

        self.close_button = tk.Button(bottom_frame, text="Done!", command=input_window.destroy, width=25, height=2)
        self.close_button.pack()

        input_window.transient(master)
        input_window.lift()
        input_window.attributes("-topmost", True)


    def add_roi_entries(self):
        try:
            self.n_roi = int(self.roi_num.get())
            if self.n_roi <= 0 or self.n_roi > 10:
                raise ValueError

            y_start = 100
            x_left = 40
            x_right = 220
            row_height = 60

            for i in range(self.n_roi):
                x = x_left if i % 2 == 0 else x_right
                y = y_start + (i // 2) * row_height

                roi_button = tk.Button(self.omw_place,
                                    text=f"Load ROI {i + 1}",
                                    command=lambda i=i: self.load_roi(i),
                                    width=14, height=2,
                                    font=("Segoe UI", 10, "bold"))
                roi_button.place(x=x, y=y)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of RoIs (1–10).", parent=self.omw_place)


    def load_roi(self, idx):
        roi_path = filedialog.askopenfilename(
            parent=self.omw_place,
            filetypes=[("NIfTI files", "*.nii")]
        )
        if roi_path:
            print(f"ROI {idx + 1} loaded:", roi_path)
            roi_img = nib.load(roi_path)
            roi_data = np.array(roi_img.get_fdata())

            # Получение размеров ROI и шаблона
            roi_shape = roi_data.shape
            template_data = np.array(app.img.get_fdata())
            template_shape = template_data.shape

            # Проверка на совпадение размеров
            if roi_shape != template_shape:
                messagebox.showerror(
                    "Size Mismatch", 
                    f"The loaded ROI size {roi_shape} does not match the template size {template_shape}.", 
                    parent=self.omw_place
                )
                #  Полностью игнорируем этот ROI
                self.list_roi[idx] = 0
                self.roi_maps[idx] = None
                return

            # Проверка на максимальный размер
            max_size = (182, 218, 182)
            if any(dim > max_dim for dim, max_dim in zip(roi_shape, max_size)):
                messagebox.showerror(
                    "Size Exceeded", 
                    f"The loaded ROI size {roi_shape} exceeds the maximum allowed size of {max_size}.", 
                    parent=self.omw_place
                )
                #  Полностью игнорируем этот ROI
                self.list_roi[idx] = 0
                self.roi_maps[idx] = None
                return

            # Если дошли сюда — ROI корректный
            roi_norm = roi_data
            self.list_roi[idx] = 1
            self.roi_maps[idx] = roi_norm

            # Отобразить ROI если включен режим Show
            if app.rsc_dropdown.get() in ["Show ROI masks", "Show & save ROI masks"]:
                combined_img = nib.Nifti1Image(
                    np.add(template_data / np.max(template_data), roi_norm),
                    app.img.affine
                )
                app.temshow.open_input_window(app.master, combined_img, f"ROI {idx + 1} Loaded")

            self.print_roi_status()
            if hasattr(app, "update_button_states"):
                app.update_button_states()


    def print_roi_status(self):
        print("ROI status:")
        for i, roi in enumerate(self.list_roi):
            status = "Set" if roi == 1 else "Not Set"
            print(f"ROI {i + 1}: {int(roi)} - {status}")

    def get_roi_data(self, idx):
        """Retrieve stored RoI data array for a specific index."""
        if 0 <= idx < len(self.roi_maps):
            return self.roi_maps[idx]
        else:
            print("Invalid ROI index.")
            return None

    def save_roi_masks(self, save_dir="roi_masks"):
        """Сохраняет все загруженные маски RoI как NIfTI файлы с кадрами, основанными на заданных значениях."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Создание папки, если она не существует

        for idx, roi_data in enumerate(self.roi_maps):
            if roi_data is not None and np.any(roi_data):  # Проверка на валидные данные
                num_frames = len(self.roi_values[idx])

                # Создание 4D массива для хранения кадров для данного RoI
                roi_4d = np.zeros((*roi_data.shape, num_frames))

                for frame in range(num_frames):
                    # Вычисление каждого кадра как произведение маски и значения для конкретного кадра
                    roi_4d[..., frame] = roi_data * self.roi_values[idx][frame]

                # Сохранение 4D данных как NIfTI файл
                affine = np.eye(4)
                roi_img_4d = nib.Nifti1Image(roi_4d, affine)
                file_path = os.path.join(save_dir, f"roi_{idx + 1}_mask.nii")

                # Попытка сохранения файла
                try:
                    nib.save(roi_img_4d, file_path)
                    print(f"Файл сохранен для ROI {idx + 1} по пути {file_path}")
                except Exception as e:
                    print(f"Ошибка при сохранении ROI {idx + 1}: {e}")
            else:
                print(f"Предупреждение: ROI {idx + 1} не содержит данных маски и был пропущен.")

class Design_data:
    def __init__(self):
        self.n_design = 0
        self.list_design = [0] * 11  # Number of designs changed
        self.collapsed_designs = [None] * 11  # To store collapsed designs
        self.tr = 0  # TR value, will be updated later
        self.volumes_quant = 0
        self.design_buttons = []
        self.input_widgets = []
        self.current_input_window = None
        self.current_design_window = None
        self.current_design_number = None

    def open_main_window(self, master):
        input_window = tk.Toplevel(master)
        input_window.title("Enter Design Information")

        master.update_idletasks()
        main_height = master.winfo_height()
        input_window.geometry(f"400x{main_height}")
        input_window.resizable(False, False)
        app.apply_theme(input_window)
        place_near_master(input_window, master)

        self.omw_place = input_window
        self.current_input_window = input_window

        # Верхняя часть 
        top_frame = tk.Frame(input_window, bg=COLOR_THEMES[app.current_theme]["bg"])
        top_frame.pack(pady=15)

        self.design_tr_label = tk.Label(top_frame, text="TR (s):")
        self.design_tr_label.grid(row=0, column=0, padx=5)
        self.design_tr_num = tk.Entry(top_frame, width=7, font=("Segoe UI", 12))
        self.design_tr_num.grid(row=0, column=1, padx=5)

        self.design_vol_label = tk.Label(top_frame, text="Volumes #:")
        self.design_vol_label.grid(row=0, column=2, padx=5)
        self.design_vol_num = tk.Entry(top_frame, width=7, font=("Segoe UI", 12))
        self.design_vol_num.grid(row=0, column=3, padx=5)

        # Отдельно Designs # ниже 
        mid_frame = tk.Frame(input_window, bg=COLOR_THEMES[app.current_theme]["bg"])
        mid_frame.pack(pady=10)

        self.design_num_label = tk.Label(mid_frame, text="Designs # (up to 10):")
        self.design_num_label.pack(side=tk.LEFT, padx=5)
        self.design_num = tk.Entry(mid_frame, width=7, font=("Segoe UI", 12))
        self.design_num.pack(side=tk.LEFT, padx=5)

        #  Кнопки Submit и Close 
        bottom_frame = tk.Frame(input_window, bg=COLOR_THEMES[app.current_theme]["bg"])
        bottom_frame.pack(side=tk.BOTTOM, pady=20)

        self.submit_button = tk.Button(mid_frame, text="Submit", command=self.add_design_entries, width=10, height=1)
        self.submit_button.pack(pady=5)

        self.close_button = tk.Button(bottom_frame, text="Close", command=input_window.destroy, width=18, height=2)
        self.close_button.pack(pady=5)
        
    def add_design_entries(self):
        for widget in self.design_buttons:
            widget.destroy()
        self.design_buttons.clear()
        for widget in self.input_widgets:
            widget.destroy()
        self.input_widgets.clear()

        # Get TR and Volumes Quant
        self.tr = float(self.design_tr_num.get())
        self.volumes_quant = int(self.design_vol_num.get())

        design_text = self.design_num.get()
        if not design_text.strip().isdigit():
            messagebox.showerror("Input Error", "Please enter a valid number of designs (1–10).", parent=self.omw_place)
            return
        n_design = int(design_text)

        # Validate TR and Volumes
        if (self.tr <= 0 or self.tr > 5) or (self.volumes_quant <= 0 or self.volumes_quant > 1000):
            messagebox.showerror("Error", "TR must be between 0 and 5. Volumes must be between 1 and 1000.", parent=self.omw_place)
            return

        if not (1 <= n_design <= 10):
            messagebox.showerror("Error", "Number of designs must be between 1 and 10.", parent=self.omw_place)
            return

        # Create buttons for design
        for i in range(n_design):
            col_offset = 160
            row_offset = 50

            x = 30 if i % 2 == 0 else 30 + col_offset + 40
            y = 100 + (i // 2) * row_offset

            btn = tk.Button(self.omw_place, text=f"Define Design {i + 1}",
                            command=lambda i=i: self.define_design(i + 1))
            btn.place(x=x, y=y, width=120, height=40)

            self.design_buttons.append(btn)
        self.n_design = n_design

    def define_design(self, number):
        if self.current_design_window:
            self.current_design_window.destroy()

        self.current_design_number = number
        input_window = tk.Toplevel()
        input_window.title(f"Enter Design {number} Information")

        self.omw_place.update_idletasks()
        main_height = self.omw_place.winfo_toplevel().winfo_height()
        input_window.geometry(f"420x{main_height}")
        input_window.resizable(False, False)

        place_near_master(input_window, self.omw_place)
        app.apply_theme(input_window)

        self.current_design_window = input_window
        self.osw_place = input_window

        self.design_enter_choice = ["Enter framewise", "Enter Ons-Dur-Int"]
        self.dec_var = tk.StringVar(value=self.design_enter_choice[0])
        self.dec_dropdown = ttk.Combobox(self.osw_place, textvariable=self.dec_var, values=self.design_enter_choice,
                                         state="readonly")
        self.dec_dropdown.place(x=15, y=15, width=160, height=35)

        self.des_type_sub_button = tk.Button(self.osw_place, text="Submit",
                                             command=lambda: self.add_design_collection(number), width=20, height=2)
        self.des_type_sub_button.place(x=245, y=15, width=120, height=35)

        self.design_conv_choice = ["No convolution", "Standard HRF"]
        self.dcc_var = tk.StringVar(value=self.design_conv_choice[0])
        self.dcc_dropdown = ttk.Combobox(self.osw_place, textvariable=self.dcc_var, values=self.design_conv_choice,
                                         state="readonly")
        self.dcc_dropdown.place(x=15, y=main_height - 80, width=160, height=35)

        self.close_button = tk.Button(input_window, text="Done!", command=self.close_design_window, width=20, height=2)
        self.close_button.place(x=245, y=main_height - 80, width=120, height=35)

    def add_design_collection(self, number):
        for widget in self.input_widgets:
            widget.destroy()
        self.input_widgets.clear()

        if self.dec_dropdown.get() == "Enter framewise":
            self.des_01_opt_01_label = tk.Label(
                self.osw_place,
                text="Enter intensity volume-wise (separate with spaces):",
                anchor="center", justify="center",
                font=("Segoe UI", 11)
            )
            self.des_01_opt_01_label.place(x=15, y=60, width=390, height=25)

            self.des_01_opt_01_dataentry = tk.Entry(self.osw_place, font=("Segoe UI", 11))
            self.des_01_opt_01_dataentry.place(x=15, y=90, width=390, height=40)

            self.input_widgets.append(self.des_01_opt_01_label)
            self.input_widgets.append(self.des_01_opt_01_dataentry)

        elif self.dec_dropdown.get() == "Enter Ons-Dur-Int":
            label_width = 390
            entry_width = 390
            label_font = ("Segoe UI", 11)
            entry_font = ("Segoe UI", 11)

            self.des_01_opt_02_label_1 = tk.Label(
                self.osw_place,
                text="Enter onset timings, in s (separate with spaces):",
                anchor="center", justify="center",
                font=label_font
            )
            self.des_01_opt_02_label_1.place(x=15, y=60, width=label_width, height=25)

            self.des_01_opt_02_dataentry_o = tk.Entry(self.osw_place, font=entry_font)
            self.des_01_opt_02_dataentry_o.place(x=15, y=90, width=entry_width, height=35)

            self.des_01_opt_02_label_2 = tk.Label(
                self.osw_place,
                text="Enter durations, in s (separate with spaces):",
                anchor="center", justify="center",
                font=label_font
            )
            self.des_01_opt_02_label_2.place(x=15, y=135, width=label_width, height=25)

            self.des_01_opt_02_dataentry_d = tk.Entry(self.osw_place, font=entry_font)
            self.des_01_opt_02_dataentry_d.place(x=15, y=165, width=entry_width, height=35)

            self.des_01_opt_02_label_3 = tk.Label(
                self.osw_place,
                text="Enter signal intensities (separate with spaces):",
                anchor="center", justify="center",
                font=label_font
            )
            self.des_01_opt_02_label_3.place(x=15, y=210, width=label_width, height=25)

            self.des_01_opt_02_dataentry_i = tk.Entry(self.osw_place, font=entry_font)
            self.des_01_opt_02_dataentry_i.place(x=15, y=240, width=entry_width, height=35)

            self.input_widgets.extend([
                self.des_01_opt_02_label_1, self.des_01_opt_02_dataentry_o,
                self.des_01_opt_02_label_2, self.des_01_opt_02_dataentry_d,
                self.des_01_opt_02_label_3, self.des_01_opt_02_dataentry_i
            ])

    def close_design_window(self):
        if self.current_design_window:
            if self.dec_dropdown.get() == "Enter framewise":
                design_values = list(map(float, self.des_01_opt_01_dataentry.get().split()))
                if len(design_values) > self.volumes_quant:
                    messagebox.showerror("Error", "Too many values! The number of values cannot exceed the number of volumes.",  parent=self.omw_place)
                    return
                elif len(design_values) < self.volumes_quant:
                    design_values += [0.0] * (self.volumes_quant - len(design_values))

                self.list_design[self.current_design_number - 1] = design_values

                if self.dcc_dropdown.get() == "No convolution":
                    self.collapsed_designs[self.current_design_number - 1] = design_values
                elif self.dcc_dropdown.get() == "Standard HRF":
                    hrf = glover_hrf(t_r=self.tr, oversampling=1)
                    if len(hrf) > self.volumes_quant:
                        hrf = hrf[:self.volumes_quant]
                    convolved = np.convolve(design_values, hrf)[:self.volumes_quant]
                    self.collapsed_designs[self.current_design_number - 1] = convolved.tolist()

            elif self.dec_dropdown.get() == "Enter Ons-Dur-Int":
                try:
                    onset_values = list(map(float, self.des_01_opt_02_dataentry_o.get().split()))
                    duration_values = list(map(float, self.des_01_opt_02_dataentry_d.get().split()))
                    intensity_values = list(map(float, self.des_01_opt_02_dataentry_i.get().split()))
                except ValueError:
                    messagebox.showerror("Input Error", "Ensure all onset, duration, and intensity values are numeric.", parent=self.omw_place)
                    return

                if not (len(onset_values) == len(duration_values) == len(intensity_values)):
                    messagebox.showerror("Input Error", "Onset, duration, and intensity lists must be the same length.", parent=self.omw_place)
                    return

                design_events = [[onset, duration, intensity] for onset, duration, intensity in
                                zip(onset_values, duration_values, intensity_values)]
                self.list_design[self.current_design_number - 1] = design_events

                stim = np.zeros(self.volumes_quant)
                frame_times = np.arange(self.volumes_quant) * self.tr  # реальные времена каждого кадра

                for onset, duration, intensity in design_events:
                    # логическая маска временных точек, попадающих в окно события
                    mask = (frame_times >= onset) & (frame_times < onset + duration)
                    stim[mask] += intensity
                if self.dcc_dropdown.get() == "No convolution":
                    self.collapsed_designs[self.current_design_number - 1] = stim.tolist()

                elif self.dcc_dropdown.get() == "Standard HRF":
                    hrf = glover_hrf(t_r=self.tr, oversampling=1)
                    if len(hrf) > self.volumes_quant:
                        hrf = hrf[:self.volumes_quant]
                    convolved = np.convolve(stim, hrf)[:self.volumes_quant]
                    self.collapsed_designs[self.current_design_number - 1] = convolved.tolist()

            self.current_design_window.destroy()
            self.current_design_window = None

        self.print_design_status()
        if hasattr(app, "update_button_states"):
            app.update_button_states()
            
    def print_design_status(self):
       
        print(f"Design status: {self.n_design} designs defined with TR={self.tr}, {self.volumes_quant} volumes")
        for i in range(self.n_design):
            original = self.list_design[i]
            collapsed = self.collapsed_designs[i]
            print(f"Design {i + 1}:")
            print(f"  Raw:       {original}")
            print(f"  Converted: {collapsed}")



class Define_Matrix:
    def __init__(self, master):
        self.master = master
        self.entries = []
        self.three_d_matrix = None
        self.matrix = []

        # Получаем данные от пользователя
        self.n_roi = app.roidata.n_roi
        self.n_design = app.designdata.n_design
        self.n_volumes = app.designdata.volumes_quant

        # Проверка на валидность
        if not (self.n_roi and self.n_design and self.n_volumes):
            messagebox.showerror("Missing Data", "Please make sure ROI, Design, and Volume data are defined first.", parent=master)
            return

        self.open_main_window()

    def open_main_window(self):
        input_window = tk.Toplevel(self.master)
        input_window.title("Fill ROI x Design Matrix")

        app.apply_theme(input_window)

        self.master.update_idletasks()
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()
        w = self.master.winfo_width()
        input_window.geometry(f"+{x + w + 20}+{y + 90}")
        input_window.transient(self.master)
        input_window.lift()
        input_window.attributes("-topmost", True)

        self.omw_place = input_window

        # Верхняя часть — Canvas с прокруткой
        top_frame = tk.Frame(input_window)
        top_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(top_frame)
        scroll_y = tk.Scrollbar(top_frame, orient="vertical", command=canvas.yview)
        scroll_x = tk.Scrollbar(top_frame, orient="horizontal", command=canvas.xview)

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        app.apply_theme(scrollable_frame)

        self.entries = []

        tk.Label(scrollable_frame, text="ROI/Design", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, padx=10, pady=10)

        for j in range(self.n_design):
            tk.Label(scrollable_frame, text=f"Design {j + 1}", font=("Segoe UI", 10, "bold")).grid(row=0, column=j + 1, padx=10, pady=10)

        for i in range(self.n_roi):
            tk.Label(scrollable_frame, text=f"RoI {i + 1}", font=("Segoe UI", 10, "bold")).grid(row=i + 1, column=0, padx=10, pady=10)
            row_entries = []
            for j in range(self.n_design):
                entry = tk.Entry(scrollable_frame, font=("Segoe UI", 10), width=8)
                entry.insert(0, "0.0")
                entry.grid(row=i + 1, column=j + 1, padx=5, pady=5)
                row_entries.append(entry)
            self.entries.append(row_entries)

        # Нижняя рамка с кнопкой
        bottom_frame = tk.Frame(input_window)
        bottom_frame.pack(fill="x", pady=10)
        app.apply_theme(bottom_frame)

        self.save_and_close_button = tk.Button(bottom_frame, text="Save and Done!", command=self.save_and_close)
        self.save_and_close_button.pack(pady=5)

    def get_matrix_data(self):
        if self.three_d_matrix is None:
            messagebox.showerror("Data Error", "Matrix data is not available. Please confirm matrix input first.", parent=self.omw_place)
            return None
        return self.three_d_matrix.tolist()

    def save_and_close(self):
        matrix = []
        for row in self.entries:
            matrix_row = []
            for entry in row:
                try:
                    value = float(entry.get())
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter valid float values in all cells.", parent=self.omw_place)
                    return
                matrix_row.append(value)
            matrix.append(matrix_row)

        self.three_d_matrix = np.zeros((self.n_roi, self.n_design, self.n_volumes))

        for i in range(self.n_roi):
            for j in range(self.n_design):
                design_values = app.designdata.collapsed_designs[j]

                if design_values is None:
                    messagebox.showerror("Design Error", f"Design {j + 1} is missing or not properly initialized.", parent=self.omw_place)
                    return
                if len(design_values) != self.n_volumes:
                    messagebox.showerror("Design Error",
                                         f"Design {j + 1} has {len(design_values)} values, but the number of volumes is {self.n_volumes}.",
                                         parent=self.omw_place)
                    return

                for k in range(self.n_volumes):
                    self.three_d_matrix[i, j, k] = matrix[i][j] * design_values[k]

        print("3D Matrix saved:")
        print(self.three_d_matrix)

        app.bids_data['matrix'] = self.three_d_matrix.tolist()

        matrix_str = '\n'.join([str(row) for row in self.three_d_matrix])
        messagebox.showinfo("Matrix Output", f"Matrix: \n{matrix_str}", parent=self.omw_place)
        self.omw_place.destroy()


class RoI_to_RoI_MatrixInput:
    def __init__(self, master, n_roi, n_design, design_labels, title, save_callback):
        self.master = master
        self.n_roi = n_roi
        self.n_design = n_design
        self.save_callback = save_callback

        # One matrix per design
        self.matrices = [np.zeros((n_roi, n_roi)) for _ in range(n_design)]
        # entries_per_design[d][i][j] -> Entry widget
        self.entries_per_design = []

        self.window = tk.Toplevel(master)
        self.window.title(title)

        master.update_idletasks()
        x = master.winfo_rootx()
        y = master.winfo_rooty()
        w = master.winfo_width()
        self.window.geometry(f"+{x + w + 20}+{y}")
        self.window.transient(master)
        self.window.lift()
        self.window.attributes("-topmost", True)

        app.apply_theme(self.window)

        tk.Label(
            self.window, text="Enter values from -1 to 1 (one matrix per Design)",
            font=("Segoe UI", 10, "italic"), fg="red"
        ).pack(pady=(8, 4))

        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        for d in range(n_design):
            tab = tk.Frame(self.notebook)
            app.apply_theme(tab)
            label = design_labels[d] if design_labels and d < len(design_labels) else f"Design {d + 1}"
            self.notebook.add(tab, text=label)

            entries = self._build_matrix_grid(tab, d)
            self.entries_per_design.append(entries)

        save_button = tk.Button(self.window, text="Save Matrices", command=self.save_and_close)
        save_button.pack(pady=10)

    def _build_matrix_grid(self, parent, design_idx):
        tk.Label(parent, text="RoI → RoI").grid(row=0, column=0, padx=5, pady=5)
        for j in range(self.n_roi):
            tk.Label(parent, text=f"RoI {j + 1}").grid(row=0, column=j + 1, padx=5, pady=5)

        entries = []
        for i in range(self.n_roi):
            tk.Label(parent, text=f"RoI {i + 1}").grid(row=i + 1, column=0, padx=5, pady=5)
            row_entries = []
            for j in range(self.n_roi):
                entry = tk.Entry(parent, width=5)
                if i == j:
                    entry.insert(0, "1.0")
                    entry.config(state='readonly')
                else:
                    entry.insert(0, "0.0")
                    entry.bind(
                        "<FocusOut>",
                        lambda e, x=i, y=j, d=design_idx: self.sync_entries(d, x, y)
                    )
                entry.grid(row=i + 1, column=j + 1, padx=3, pady=3)
                row_entries.append(entry)
            entries.append(row_entries)
        return entries

    def sync_entries(self, d, i, j):
        """Keep the FC matrix symmetric within a given design's tab."""
        try:
            val = self.entries_per_design[d][i][j].get()
            self.entries_per_design[d][j][i].delete(0, tk.END)
            self.entries_per_design[d][j][i].insert(0, val)
        except Exception as e:
            print(f"Sync error (design {d + 1}): {e}")

    def save_and_close(self):
        try:
            for d in range(self.n_design):
                for i in range(self.n_roi):
                    for j in range(self.n_roi):
                        val = float(self.entries_per_design[d][i][j].get())
                        if val < -1.0 or val > 1.0:
                            messagebox.showerror(
                                "Input Error",
                                f"Design {d + 1}: all values must be between -1 and 1.",
                                parent=self.window
                            )
                            return
                        self.matrices[d][i, j] = val
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid float values in all cells.", parent=self.window)
            return

        self.save_callback(self.matrices)
        self.window.destroy()


class EffectiveConnectivityMatrixInput:
    def __init__(self, master, n_roi, n_design, design_labels, save_callback):
        self.master = master
        self.n_roi = n_roi
        self.n_design = n_design
        self.save_callback = save_callback

        self.ec_matrices = [np.zeros((n_roi, n_roi)) for _ in range(n_design)]
        self.delay_matrices = [np.zeros((n_roi, n_roi)) for _ in range(n_design)]

        self.entries_ec_per_design = []
        self.entries_delay_per_design = []

        self.window = tk.Toplevel(master)
        self.window.title("Define Effective Connectivity (EC) and Delays")

        master.update_idletasks()
        x = master.winfo_rootx()
        y = master.winfo_rooty()
        w = master.winfo_width()
        self.window.geometry(f"+{x + w + 20}+{y + 100}")
        self.window.transient(master)
        self.window.lift()
        self.window.attributes("-topmost", True)

        app.apply_theme(self.window)

        tk.Label(
            self.window, text="EC: -1 to 1 | Delay: any positive float (one pair per Design)",
            fg="red", font=("Segoe UI", 10, "italic")
        ).pack(pady=(8, 4))

        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        for d in range(n_design):
            tab = tk.Frame(self.notebook)
            app.apply_theme(tab)
            label = design_labels[d] if design_labels and d < len(design_labels) else f"Design {d + 1}"
            self.notebook.add(tab, text=label)

            ec_entries, delay_entries = self._build_ec_delay_grid(tab, d)
            self.entries_ec_per_design.append(ec_entries)
            self.entries_delay_per_design.append(delay_entries)

        save_button = tk.Button(self.window, text="Save Both Matrices", command=self.save_and_close)
        save_button.pack(pady=10)

    def _build_ec_delay_grid(self, parent, design_idx):
        n_roi = self.n_roi

        tk.Label(parent, text="EC Matrix (RoI → RoI)", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=n_roi + 1, pady=5)
        tk.Label(parent, text="Delay Matrix (RoI → RoI)", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=n_roi + 2, columnspan=n_roi + 1, pady=5)

        for j in range(n_roi):
            tk.Label(parent, text=f"RoI {j + 1}").grid(row=1, column=j + 1)
            tk.Label(parent, text=f"RoI {j + 1}").grid(row=1, column=n_roi + j + 3)

        entries_ec = []
        entries_delay = []

        for i in range(n_roi):
            tk.Label(parent, text=f"RoI {i + 1}").grid(row=i + 2, column=0)
            tk.Label(parent, text=f"RoI {i + 1}").grid(row=i + 2, column=n_roi + 2)

            row_ec = []
            row_delay = []

            for j in range(n_roi):
                ec_entry = tk.Entry(parent, width=6)
                delay_entry = tk.Entry(parent, width=6)

                ec_entry.grid(row=i + 2, column=j + 1)
                delay_entry.grid(row=i + 2, column=n_roi + j + 3)

                if i == j:
                    ec_entry.insert(0, "1.0")
                    ec_entry.config(state='readonly')
                else:
                    ec_entry.insert(0, "0.0")
                    ec_entry.bind(
                        "<FocusOut>",
                        lambda e, x=i, y=j, d=design_idx: self.zero_reverse_entry(d, x, y)
                    )

                delay_entry.insert(0, "0.0")

                row_ec.append(ec_entry)
                row_delay.append(delay_entry)

            entries_ec.append(row_ec)
            entries_delay.append(row_delay)

        return entries_ec, entries_delay

    def zero_reverse_entry(self, d, i, j):
        """Prevent bidirectional EC within a given design's tab."""
        if i == j:
            return
        try:
            val = float(self.entries_ec_per_design[d][i][j].get())
            if val != 0:
                self.entries_ec_per_design[d][j][i].delete(0, tk.END)
                self.entries_ec_per_design[d][j][i].insert(0, "0.0")
        except ValueError:
            pass

    def save_and_close(self):
        try:
            for d in range(self.n_design):
                for i in range(self.n_roi):
                    for j in range(self.n_roi):
                        ec_val = float(self.entries_ec_per_design[d][i][j].get())
                        if ec_val < -1.0 or ec_val > 1.0:
                            messagebox.showerror(
                                "Input Error",
                                f"Design {d + 1}: EC values must be between -1 and 1.",
                                parent=self.window
                            )
                            return

                        delay_val = float(self.entries_delay_per_design[d][i][j].get())
                        if delay_val < 0:
                            messagebox.showerror(
                                "Input Error",
                                f"Design {d + 1}: Delay values must be positive.",
                                parent=self.window
                            )
                            return

                        self.ec_matrices[d][i, j] = ec_val
                        self.delay_matrices[d][i, j] = delay_val
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid float values in all cells.", parent=self.window)
            return

        self.save_callback(self.ec_matrices, self.delay_matrices)
        self.window.destroy()

class SpontaneousFCInput:
    def __init__(self, master, n_roi, app_reference):
        self.master = master
        self.n_roi = n_roi
        self.app = app_reference

        self.fc_matrix = np.zeros((n_roi, n_roi))
        self.spont_fc_maps = []
        self.fc_mean_value = 0.0

        self.entries = []
        self.window = tk.Toplevel(master)
        self.window.title("Spontaneous FC")
        self.app.apply_theme(self.window)

        # Window positioning
        master.update_idletasks()
        x = master.winfo_rootx()
        y = master.winfo_rooty()
        w = master.winfo_width()
        self.window.geometry(f"+{x + w + 20}+{y}")
        self.window.transient(master)
        self.window.lift()
        self.window.attributes("-topmost", True)

        # Step 1: Show the existing FC Matrix
        self.show_fc_matrix()
    def show_fc_matrix(self):
        # 1) отдельный контейнер для матрицы в grid окна
        matrix_frame = tk.Frame(self.window)
        matrix_frame.grid(row=0, column=0, columnspan=self.n_roi + 1, sticky="w", padx=8, pady=6)
        self.app.apply_theme(matrix_frame)

        # 2) компактная разметка ВНУТРИ matrix_frame (свой grid)
        small_font = ("Segoe UI", 9)

        tk.Label(matrix_frame, text="FC Matrix (RoI ↔ RoI):",
                font=("Segoe UI", 10, "bold"))\
            .grid(row=0, column=0, columnspan=self.n_roi + 1, sticky="w", pady=(0, 4))

        # узкие колонки только внутри matrix_frame
        matrix_frame.grid_columnconfigure(0, minsize=28, weight=0)      # метки строк
        for c in range(1, self.n_roi + 1):
            matrix_frame.grid_columnconfigure(c, minsize=30, weight=0)  # значения

        # заголовки столбцов
        for j in range(self.n_roi):
            tk.Label(matrix_frame, text=f"R{j+1}", font=small_font, anchor="center")\
                .grid(row=1, column=j + 1, padx=1, pady=(0, 2), sticky="w")

        # ячейки матрицы
        self.entries = []
        for i in range(self.n_roi):
            tk.Label(matrix_frame, text=f"R{i+1}", font=small_font, anchor="e")\
                .grid(row=i + 2, column=0, padx=(0, 2), pady=1, sticky="e")

            row_entries = []
            for j in range(self.n_roi):
                e = tk.Entry(matrix_frame, width=3, justify="center")
                if i == j:
                    e.insert(0, "1.0")
                    e.config(state="readonly")
                else:
                    e.insert(0, "0.0")
                e.grid(row=i + 2, column=j + 1, padx=1, pady=1, sticky="w")
                row_entries.append(e)
            self.entries.append(row_entries)


        self.ask_for_number_of_masks()

    def ask_for_number_of_masks(self):
        # Prompt the user to enter the number of masks to load (up to 10)
        self.mask_count_label = tk.Label(self.window, text="How many masks do you want to load (1 to 10)?", font=("Segoe UI", 12))
        self.mask_count_label.grid(row=self.n_roi + 3, column=0, columnspan=2, pady=20)

        self.mask_count_entry = tk.Entry(self.window, font=("Segoe UI", 12))
        self.mask_count_entry.grid(row=self.n_roi + 4, column=0, columnspan=2, pady=5)

        self.submit_mask_count_button = tk.Button(self.window, text="Submit", command=self.on_mask_count_submit)
        self.submit_mask_count_button.grid(row=self.n_roi + 5, column=0, columnspan=2, pady=10)

    def on_mask_count_submit(self):
        try:
            # Get the number of masks the user wants to load
            self.num_masks = int(self.mask_count_entry.get().strip())

            # Check if the number is valid (between 1 and 10)
            if self.num_masks < 1 or self.num_masks > 10:
                raise ValueError("Please enter a number between 1 and 10.")

            # Hide the previous widgets (mask count input)
            self.mask_count_label.grid_forget()
            self.mask_count_entry.grid_forget()
            self.submit_mask_count_button.grid_forget()

            # Step 2: Ask for the mean FC value for each mask
            self.ask_for_fc_values()

        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.window)

    def ask_for_fc_values(self):
        self.fc_mean_entries = []
        tk.Label(self.window, text="Enter mean FC values for each mask:", font=("Segoe UI", 12)).grid(row=self.n_roi + 3, column=0, columnspan=2, pady=20)

        for i in range(self.num_masks):
            tk.Label(self.window, text=f"Mask {i+1}:").grid(row=self.n_roi + 4 + i, column=0, padx=10, pady=5)
            fc_entry = tk.Entry(self.window, font=("Segoe UI", 12))
            fc_entry.grid(row=self.n_roi + 4 + i, column=1, padx=10, pady=5)
            fc_entry.insert(0, "0.0")  # Default value
            self.fc_mean_entries.append(fc_entry)

            # Add Load button for each mask
            load_button = tk.Button(self.window, text=f"Load Mask {i+1}", command=lambda i=i: self.load_nifti_file(i))
            load_button.grid(row=self.n_roi + 4 + i, column=2, padx=10, pady=5)

        self.save_button = tk.Button(self.window, text="Save FC Values and Masks", command=self.save_data)
        self.save_button.grid(row=self.n_roi + 4 + self.num_masks + 1, column=0, columnspan=3, pady=10)

    def load_nifti_file(self, mask_index):
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")], parent=self.window)
        if file_path:
            try:
                nii_img = nib.load(file_path)
                self.spont_fc_maps.append(nii_img)
                messagebox.showinfo("File Loaded", f"Mask {mask_index+1} loaded successfully.", parent=self.window)
            except Exception as e:
                messagebox.showerror("Error loading file", f"Failed to load {file_path}: {e}", parent=self.window)

    def save_data(self):
        try:
            # Step 3: Save the mean FC values for each mask
            self.fc_matrix = np.zeros((self.num_masks, self.num_masks))

            for i in range(self.num_masks):
                fc_val = float(self.fc_mean_entries[i].get().strip())
                if fc_val < -1.0 or fc_val > 1.0:
                    raise ValueError(f"Invalid FC value for mask {i+1}: {fc_val}")
                self.fc_matrix[i, i] = fc_val  # Set diagonal to the mean FC values entered by the user

            # Save the FC matrix and other parameters
            self.app.fc_matrix = self.fc_matrix
            self.app.spont_fc_maps = self.spont_fc_maps

            print("FC Matrix saved:")
            print(self.fc_matrix)

            messagebox.showinfo("Success", "Spontaneous FC data saved.", parent=self.window)
            self.window.destroy()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e), parent=self.window)


class NoiseSettingsWindow:
    MAX_MASKS_PER_TYPE = 5

    # label -> list of (key, display_label, caster, validator)
    NOISE_TYPE_DEFS = {
        "Gaussian": [("Z", "z", float, lambda v: v > 0)],
        "Trend":    [("Z", "z(mask)", float, lambda v: v > 0),
                     ("Order", "order", int, lambda v: v >= 1)],
        "Autocor":  [("MeanR", "mean r", float, lambda v: v > 0)],
        "Heart":    [("MinFreq", "min f", float, lambda v: v > 0),
                     ("MaxFreq", "max f", float, lambda v: v > 0),
                     ("Amplitude", "z", float, lambda v: v > 0)],
        "Resp":     [("MinFreq", "min f", float, lambda v: v > 0),
                     ("MaxFreq", "max f", float, lambda v: v > 0),
                     ("Amplitude", "z", float, lambda v: v > 0)],
    }

    def __init__(self, master, app_reference):
        self.master = master
        self.app = app_reference
        self.window = tk.Toplevel(master)
        self.window.title("Noise Settings")
        self.app.apply_theme(self.window)

        self.DEFAULT_GAUSSIAN_SEED = None

        self.vars = {}            # label -> BooleanVar
        self.mask_rows = {}       # label -> [row_dict, ...]
        self.mask_containers = {} # label -> Frame holding the rows
        self.add_buttons = {}     # label -> "Add mask" Button
        self.row = 0

        self.window.geometry("700x600")
        self.window.transient(master)
        self.window.lift()
        self.window.attributes("-topmost", True)

        master.update_idletasks()
        x = master.winfo_rootx()
        y = master.winfo_rooty()
        w = master.winfo_width()
        self.window.geometry(f"+{x + w + 20}+{y}")

        self._build_scrollable_container()
        self.create_widgets()

    # ---------- layout scaffolding ----------

    def _build_scrollable_container(self):
        outer = tk.Frame(self.window, bg=self.window["bg"])
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=self.window["bg"], highlightthickness=0)
        scroll_y = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll_y.set)

        scroll_y.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame = tk.Frame(canvas, bg=self.window["bg"])
        self.scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

    def create_widgets(self):
        for label, spec in self.NOISE_TYPE_DEFS.items():
            self._build_mask_block(label, spec)

        save_button = tk.Button(self.scroll_frame, text="Apply", command=self.save_noise_config)
        save_button.grid(row=self.row, column=0, sticky="w", padx=10, pady=12)
        self.row += 1

    def _build_mask_block(self, label, spec):
        self.vars[label] = tk.BooleanVar()
        self.mask_rows[label] = []

        frame = tk.LabelFrame(self.scroll_frame, text=label, bg=self.window["bg"], padx=8, pady=6)
        frame.grid(row=self.row, column=0, sticky="we", padx=10, pady=5)
        self.row += 1

        header = tk.Frame(frame, bg=self.window["bg"])
        header.grid(row=0, column=0, sticky="w", pady=(0, 4))

        tk.Checkbutton(
            header, text="Enable", variable=self.vars[label], bg=self.window["bg"],
            command=lambda lb=label: self._on_toggle_block(lb)
        ).pack(side="left")

        add_btn = tk.Button(
            header, text="Add mask", state="disabled",
            command=lambda lb=label: self.add_mask_row(lb)
        )
        add_btn.pack(side="left", padx=8)
        self.add_buttons[label] = add_btn

        hdr = tk.Frame(frame, bg=self.window["bg"])
        hdr.grid(row=1, column=0, sticky="w", pady=(2, 2))
        tk.Label(hdr, text="Mask file", width=26, anchor="w", bg=self.window["bg"]).grid(
            row=0, column=0, padx=(4, 10)
        )
        col = 1
        for key, display, caster, validator in spec:
            tk.Label(hdr, text=display, anchor="center", bg=self.window["bg"]).grid(
                row=0, column=col, padx=10
            )
            col += 1

        container = tk.Frame(frame, bg=self.window["bg"])
        container.grid(row=2, column=0, sticky="w")
        self.mask_containers[label] = container

    # ---------- generic enable/add/remove ----------

    def _on_toggle_block(self, label):
        enabled = self.vars[label].get()
        rows = self.mask_rows[label]
        self.add_buttons[label].config(
            state=("normal" if enabled and len(rows) < self.MAX_MASKS_PER_TYPE else "disabled")
        )
        if not enabled:
            for r in rows:
                self._grid_remove_row_widgets(r)
        else:
            if not rows:
                self.add_mask_row(label)
            else:
                self._refresh_mask_rows(label)

    def add_mask_row(self, label):
        rows = self.mask_rows[label]
        container = self.mask_containers[label]
        spec = self.NOISE_TYPE_DEFS[label]
        row_index = len(rows)

        load_btn = tk.Button(container, text="Load")
        load_btn.grid(row=row_index, column=0, padx=4, sticky="w")

        name_label = tk.Label(container, text="", bg=self.window["bg"], anchor="w", width=26)
        name_label.grid(row=row_index, column=1, padx=4, sticky="w")

        param_entries = {}
        col = 2
        for key, display, caster, validator in spec:
            entry = tk.Entry(container, width=8)
            entry.grid(row=row_index, column=col, padx=4, sticky="w")
            param_entries[key] = entry
            col += 1

        remove_btn = tk.Button(container, text="✕", width=2)
        remove_btn.grid(row=row_index, column=col, padx=4, sticky="w")

        row_dict = {
            "path": None,
            "name_label": name_label,
            "param_entries": param_entries,
            "load_btn": load_btn,
            "remove_btn": remove_btn,
        }
        load_btn.config(command=lambda r=row_dict, lb=label: self.load_mask_for_row(lb, r))
        remove_btn.config(command=lambda r=row_dict, lb=label: self.remove_mask_row(lb, r))
        rows.append(row_dict)

        if len(rows) >= self.MAX_MASKS_PER_TYPE:
            self.add_buttons[label].config(state="disabled")
            messagebox.showwarning(
                "Limit", f"Maximum {self.MAX_MASKS_PER_TYPE} masks for {label}.", parent=self.window
            )

    def remove_mask_row(self, label, row_dict):
        self._grid_remove_row_widgets(row_dict)
        self.mask_rows[label].remove(row_dict)
        self._refresh_mask_rows(label)
        if self.vars[label].get() and len(self.mask_rows[label]) < self.MAX_MASKS_PER_TYPE:
            self.add_buttons[label].config(state="normal")

    def _grid_remove_row_widgets(self, row_dict):
        widgets = [row_dict["load_btn"], row_dict["name_label"], row_dict["remove_btn"]]
        widgets += list(row_dict["param_entries"].values())
        for w in widgets:
            try:
                w.grid_forget()
            except Exception:
                pass

    def _refresh_mask_rows(self, label):
        for i, r in enumerate(self.mask_rows[label]):
            r["load_btn"].grid(row=i, column=0, padx=4, sticky="w")
            r["name_label"].grid(row=i, column=1, padx=4, sticky="w")
            col = 2
            for entry in r["param_entries"].values():
                entry.grid(row=i, column=col, padx=4, sticky="w")
                col += 1
            r["remove_btn"].grid(row=i, column=col, padx=4, sticky="w")

    def load_mask_for_row(self, label, row_dict):
        file_path = filedialog.askopenfilename(
            parent=self.window,
            title=f"Select {label} Mask (NIfTI)",
            filetypes=[("NIfTI files", "*.nii *.nii.gz")]
        )
        if not file_path:
            return
        try:
            nii = nib.load(file_path)
            mask_data = nii.get_fdata()
            template_img = getattr(self.app, "img", None)
            if template_img is None:
                messagebox.showerror(
                    "Template not loaded", "Please load a template before adding a mask.", parent=self.window
                )
                return
            tmpl_shape3d = template_img.get_fdata().shape[:3]
            if mask_data.ndim == 4:
                mask_data = mask_data[..., 0]
            if mask_data.shape != tmpl_shape3d:
                messagebox.showerror(
                    "Size Mismatch",
                    f"Mask shape {mask_data.shape} does not match template shape {tmpl_shape3d}.",
                    parent=self.window
                )
                return
            row_dict["path"] = file_path
            row_dict["name_label"].config(text=os.path.basename(file_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask: {e}", parent=self.window)

    # ---------- save ----------

    def save_noise_config(self):
        config = {}

        for label, spec in self.NOISE_TYPE_DEFS.items():
            if not self.vars[label].get():
                continue

            rows = self.mask_rows.get(label, [])
            if not rows:
                messagebox.showerror(
                    "Input Error", f"Please add at least one mask for {label}.", parent=self.window
                )
                return

            masks_cfg = []
            for r in rows:
                path = r["path"]
                if not path:
                    messagebox.showerror(
                        "Input Error", f"One of the {label} masks has no file selected.", parent=self.window
                    )
                    return

                mask_entry = {"path": path}
                for key, display, caster, validator in spec:
                    raw = r["param_entries"][key].get().strip()
                    try:
                        val = caster(raw)
                        if not validator(val):
                            raise ValueError
                    except Exception:
                        messagebox.showerror(
                            "Input Error",
                            f"Invalid value in {label} - {display} for mask {os.path.basename(path)}.",
                            parent=self.window
                        )
                        return
                    mask_entry[key] = val
                masks_cfg.append(mask_entry)

            config[label] = {"masks": masks_cfg}

        if "Gaussian" in config:
            config["Gaussian"]["Seed"] = self.DEFAULT_GAUSSIAN_SEED

        if not config:
            messagebox.showwarning("No Data", "No noise types selected.", parent=self.window)
            return

        self.app.noise_config = config
        try:
            with open("noise_config.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print("Failed to save to file:", e)

        print("Saved noise parameters:", config)
        messagebox.showinfo("Success", "Noise parameters saved.", parent=self.window)
        self.window.destroy()

class EstimateWindow:
    def __init__(self, master, app_reference):
        self.master = master
        self.app = app_reference
        self.window = tk.Toplevel(master)
        self.window.title("Estimate Settings")
        self.app.apply_theme(self.window)
        self.window.geometry("420x500")
        self.window.transient(master)
        self.window.lift()
        self.window.attributes("-topmost", True)
 

        self.create_widgets()

        master.update_idletasks()
        x = master.winfo_rootx()
        y = master.winfo_rooty()
        w = master.winfo_width()
        self.window.geometry(f"+{x + w + 20}+{y}")

    def create_widgets(self):
        row = 0

        #  Ratio fields 
        tk.Label(self.window, text="Spont FC:").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.spont_entry = tk.Entry(self.window, width=8)
        self.spont_entry.insert(0, "0.0")
        self.spont_entry.grid(row=row, column=1, padx=5, pady=5)

        tk.Label(self.window, text="Task-related FC/EC:").grid(row=row, column=2, padx=5, pady=5, sticky="e")
        self.task_entry = tk.Entry(self.window, width=8)
        self.task_entry.insert(0, "0.0")
        self.task_entry.grid(row=row, column=3, padx=5, pady=5)
        row += 1

        tk.Label(self.window, text="Ratio Activity:").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.ratio_entry = tk.Entry(self.window, width=8)
        self.ratio_entry.insert(0, "0.0")
        self.ratio_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        #  Start method 
        tk.Label(self.window, text="Start method:", font=("Segoe UI", 11, "bold")).grid(
            row=row, column=0, columnspan=4, pady=10
        )
        row += 1
        self.start_method = tk.StringVar(value="Random")
        methods = ["Random", "Template", "Activity"]
        for col, method in enumerate(methods):
            tk.Radiobutton(self.window, text=method, variable=self.start_method, value=method,
                           bg=self.window["bg"]).grid(row=row, column=col, padx=10, sticky="w")
        row += 1

        # subjects & Sessions
        tk.Label(self.window, text="Subjects #:").grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.subject_entry = tk.Entry(self.window, width=8)
        self.subject_entry.insert(0, "1")
        self.subject_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

        tk.Label(self.window, text="Sessions #:").grid(row=row, column=2, padx=5, pady=5, sticky="e")
        self.session_entry = tk.Entry(self.window, width=8)
        self.session_entry.insert(0, "1")
        self.session_entry.grid(row=row, column=3, padx=5, pady=5, sticky="w")
        row += 1

        # Headers for parameter blocks 
        tk.Label(self.window, text="Subject Parameters:", font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=20, pady=(10, 2)
        )
        tk.Label(self.window, text="Session Parameters:", font=("Segoe UI", 10, "bold")).grid(
            row=row, column=2, columnspan=2, sticky="w", padx=20, pady=(10, 2)
        )
        row += 1

        #  Parameters in both blocks 
        param_list = [
            "RoI position", "RoI shape", "Event timing",
            "Event textures", "FC value", "ER FC/EC value", "ER EC delay"
        ]
        self.param_vars_subject = {}
        self.param_vars_session = {}

        for p in param_list:
            subj_var = tk.BooleanVar(value=True)
            sess_var = tk.BooleanVar(value=True)

            self.param_vars_subject[p] = subj_var
            self.param_vars_session[p] = sess_var

            tk.Checkbutton(self.window, text=p, variable=subj_var, bg=self.window["bg"]).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=40)
            tk.Checkbutton(self.window, text=p, variable=sess_var, bg=self.window["bg"]).grid(
                row=row, column=2, columnspan=2, sticky="w", padx=40)
            row += 1

        # Z values for Subject and Session 
        tk.Label(self.window, text="Z — Subject:", font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, columnspan=2, padx=20, pady=10, sticky="w"
        )
        tk.Label(self.window, text="Z — Session:", font=("Segoe UI", 10, "bold")).grid(
            row=row, column=2, columnspan=2, padx=20, pady=10, sticky="w"
        )
        row += 1

        self.z_subject_entry = tk.Entry(self.window, width=8)
        self.z_subject_entry.insert(0, "0.1")
        self.z_subject_entry.grid(row=row, column=0, columnspan=2, padx=20, sticky="w")

        self.z_session_entry = tk.Entry(self.window, width=8)
        self.z_session_entry.insert(0, "0.1")
        self.z_session_entry.grid(row=row, column=2, columnspan=2, padx=20, sticky="w")
        row += 1

        #Start button
        self.start_button = tk.Button(self.window, text="Start", command=self.validate_and_start)
        self.start_button.grid(row=row, column=0, columnspan=4, pady=15)

    def validate_and_start(self):
        try:
            spont = float(self.spont_entry.get())
            task = float(self.task_entry.get())
            ratio = float(self.ratio_entry.get())
            z_subject = float(self.z_subject_entry.get())
            z_session = float(self.z_session_entry.get())
            subjects = int(self.subject_entry.get())
            sessions = int(self.session_entry.get())

            if any(val < 0 for val in [spont, task, ratio]):
                raise ValueError("Values must be ≥ 0.")
            if spont + task + ratio > 1:
                raise ValueError("Sum of Spont FC, Task FC/EC and Ratio Activity must be ≤ 1.")
            if z_subject <= 0 or z_session <= 0:
                raise ValueError("Both Z values must be > 0.")
            if subjects < 1 or sessions < 1:
                raise ValueError("Subjects and Sessions must be ≥ 1.")

            # Передаём значения в app
            self.app.subjects = subjects
            self.app.sessions = sessions

            # Показываем окно поверх Estimate
            messagebox.showinfo(
                "Success",
                f"Starting with:\nSpont={spont}, Task={task}, Ratio={ratio}, "
                f"Z_subject={z_subject}, Z_session={z_session}, "
                f"Subjects={subjects}, Sessions={sessions}",
                parent=self.window
            )

            self.window.destroy()

        except ValueError as e:
            messagebox.showerror("Input Error", str(e), parent=self.window)

class TemplateLoaderApp:
    def __init__(self, master):
        self.master = master
        self.current_theme = "Light"  # или "Dark"
        self.themes = COLOR_THEMES
        self.temshow = Template_shower()
        self.roidata = RoI_data()
        self.designdata = Design_data()
        self.matrixdata = None
        self.bids_data = {}
        self.output_path = None
        self.volumes_quant = None
        self.selected_connectivity_type = None
        self.ec_matrix = None
        self.ec_delay_matrix = None
        self.subjects = 1
        self.sessions = 1


        master.title("Main window")
        master.geometry("500x460")
        master.configure(bg=self.themes[self.current_theme]["bg"])

        
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("TButton",
            font=("Segoe UI", 12, "bold"),
            padding=10,
            relief="raised",
            background="#f2f2f2",
            foreground="#000000"
        )

        style.map("TButton",
            background=[("active", "#d0e0c0")],
            foreground=[("active", "#000000")]
        )

        style.configure("TCombobox",
            font=("Segoe UI", 11),
            padding=6,
            background="white",
            fieldbackground="white"
        )

        
        # Enable auto-resizing
        for col in range(2):
            master.columnconfigure(col, weight=1)

        self.template_path = None

        # Subject/session inputs
        # self.subject_label = tk.Label(master, text="Subjects #:")
        # self.subject_label.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        # self.subject_entry = tk.Entry(master)
        # self.subject_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # self.session_label = tk.Label(master, text="Sessions #:")
        # self.session_label.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        # self.session_entry = tk.Entry(master)
        # self.session_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Template loading
        self.template_show_choice = ["Show nothing", "Show template"]
        self.tsc_var = tk.StringVar(value=self.template_show_choice[0])
        self.tsc_dropdown = ttk.Combobox(master, textvariable=self.tsc_var, values=self.template_show_choice, state="readonly")
        self.tsc_dropdown.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.template_button = ttk.Button(master, text="Load Template", command=self.load_template)
        self.template_button.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # RoI section
        self.roi_show_choice = ["Show & save nothing", "Show ROI masks", "Save ROI masks", "Show & save ROI masks"]
        self.rsc_var = tk.StringVar(value=self.roi_show_choice[0])
        self.rsc_dropdown = ttk.Combobox(master, textvariable=self.rsc_var, values=self.roi_show_choice, state="readonly")
        self.rsc_dropdown.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.roi_button = ttk.Button(master, text="Define ROIs", command=self.define_rois, state = "disabled")
        self.roi_button.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        # Design & Matrix
        self.define_designs_button = ttk.Button(master, text="Define Designs", command=self.define_designs)
        self.define_designs_button.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        self.define_matrix_button = ttk.Button(master, text="Define Matrix", command=self.define_matrix)
        self.define_matrix_button.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
            
        # FC buttons
        self.fc_button = ttk.Button(master, text="Spontaneous FC", command=self.open_spontaneous_fc_window)
        self.fc_button.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        self.task_fc_button = ttk.Button(master, text="Task-related FC & EC",  command=self.task_fc_ec_selection_window)
        self.task_fc_button.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

        # Noise & Estimate
        self.noise_button = ttk.Button(master, text="Add Noise", command=self.open_noise_settings_window)
        self.noise_button.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        self.estimate_button = ttk.Button(master, text="Estimate", command=self.open_estimate_window)
        self.estimate_button.grid(row=6, column=1, padx=10, pady=5, sticky="ew")

        # Output section
        self.save_nii_type_choice = ["Save series of 3D .nii files", "Save one 4D .nii file"]
        self.sntc_var = tk.StringVar(value=self.save_nii_type_choice[0])
        self.sntc_dropdown = ttk.Combobox(master, textvariable=self.sntc_var, values=self.save_nii_type_choice, state="readonly")
        self.sntc_dropdown.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        self.output_button = ttk.Button(master, text="Select Output Path", command=self.select_output_path)
        self.output_button.grid(row=7, column=1, padx=10, pady=5, sticky="ew")
        self.theme_button = ttk.Button(master, text="Switch Theme", command=self.toggle_theme)
        self.theme_button.grid(row=8, column=1, padx=10, pady=5, sticky="ew")

    def toggle_theme(self):
        self.current_theme = "Dark" if self.current_theme == "Light" else "Light"
        self.apply_theme(self.master)
        
    def set_volumes(self):
        try:
            self.volumes_quant = int(self.volume_entry.get())
            if self.volumes_quant <= 0:
                raise ValueError("Volume count must be a positive integer.", parent=self.window)
            messagebox.showinfo("Success", f"Volumes set to {self.volumes_quant}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid positive integer for the number of volumes.",  parent=self.window)
            self.volumes_quant = None


    def update_button_states(self):
        """Обновляет доступность кнопок в зависимости от введённых данных."""
        #  есть хотя бы один загруженный RoI и хотя бы один дизайн
        roi_ready = any(self.roidata.list_roi)
        design_ready = self.designdata.n_design > 0 and self.designdata.volumes_quant > 0

        # Если и RoI, и Design присутствуют, делаем кнопку активной, vversa - неактивной
        if roi_ready and design_ready:
            self.define_matrix_button.config(state="normal")
        else:
            self.define_matrix_button.config(state="disabled")

    def load_template(self):
        self.template_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
        if self.template_path:
            print("Template loaded:", self.template_path)
            self.img = nib.load(self.template_path)

            #  Делаем кнопку Define ROIs активной
            self.roi_button.config(state="normal")

            # Check if we have a 3D image and if volumes_quant is set, then create 4D array
            if len(self.img.shape) == 3 and self.volumes_quant is not None:
                img_data_3d = self.img.get_fdata()
                img_data_4d = np.repeat(img_data_3d[..., np.newaxis], self.volumes_quant, axis=-1)
                self.img = nib.Nifti1Image(img_data_4d, affine=self.img.affine)
                print(f"Created 4D image with {self.volumes_quant} volumes.")
            else:
                print("Loaded 3D image as is.")

            if self.tsc_dropdown.get() == "Show template":
                self.temshow.open_input_window(self.master, self.img, "Template Image")

    def define_rois(self):
        self.roidata.open_main_window(self.master)
        self.master.after(100, self.print_roi_status)

    def define_designs(self):
        self.designdata.open_main_window(self.master)
        self.master.after(100, self.print_design_status)

    def print_roi_status(self):
        self.roidata.print_roi_status()

    def print_design_status(self):
        self.designdata.print_design_status()

    def define_matrix(self):
        roi_ready = any(self.roidata.list_roi)
        design_ready = self.designdata.n_design > 0 and self.designdata.volumes_quant > 0

        if not roi_ready or not design_ready:
            return

        self.matrixdata = Define_Matrix(master=self.master)
        
    def open_spontaneous_fc_window(self):
        SpontaneousFCInput(self.master, self.roidata.n_roi, self)

    def open_noise_settings_window(self):
        NoiseSettingsWindow(self.master, self)
    
    def open_estimate_window(self):
        EstimateWindow(self.master, self)
    def task_fc_ec_selection_window(self):
        # Require at least one design before letting the user define task-related FC/EC
        if self.designdata.n_design <= 0:
            messagebox.showerror("Error", "Please define at least one Design before setting Task-related FC/EC.")
            return

        selection_window = tk.Toplevel(self.master)
        selection_window.title("Select Analysis Type")
        selection_window.geometry("320x180")
        self.apply_theme(selection_window)

        # расположение справа от главного окна
        self.master.update_idletasks()
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()
        w = self.master.winfo_width()
        selection_window.geometry(f"+{x + w + 20}+{y + 100}")

        selection_window.transient(self.master)
        selection_window.lift()
        selection_window.attributes("-topmost", True)

        tk.Label(selection_window, text="Choose connectivity type:", font=("Segoe UI", 11, "bold"),
                bg=COLOR_THEMES[self.current_theme]["bg"]).pack(pady=10)

        self.connectivity_choice = tk.StringVar(value="FC")
        fc_radio = tk.Radiobutton(selection_window, text="Functional Connectivity (FC)",
                                variable=self.connectivity_choice, value="FC",
                                bg=COLOR_THEMES[self.current_theme]["bg"])
        ec_radio = tk.Radiobutton(selection_window, text="Effective Connectivity (EC)",
                                variable=self.connectivity_choice, value="EC",
                                bg=COLOR_THEMES[self.current_theme]["bg"])

        fc_radio.pack(anchor="w", padx=30)
        ec_radio.pack(anchor="w", padx=30)

        def proceed():
            self.selected_connectivity_type = self.connectivity_choice.get()
            print(f"Selected: {self.selected_connectivity_type}")
            selection_window.destroy()
            if self.selected_connectivity_type == "FC":
                self.open_fc_matrix_input()
            elif self.selected_connectivity_type == "EC":
                self.open_ec_matrix_input()

        proceed_button = tk.Button(selection_window, text="Proceed", command=proceed)
        proceed_button.pack(pady=20)


    def open_fc_matrix_input(self):
        n_design = self.designdata.n_design
        design_labels = [f"Design {i + 1}" for i in range(n_design)]

        def save_fc_matrices(matrices):
            # list of n_design matrices, each n_roi x n_roi
            self.fc_matrix = matrices
            print("FC matrices saved (per design):")
            for d, m in enumerate(matrices):
                print(f"Design {d + 1}:\n{m}")

        RoI_to_RoI_MatrixInput(
            self.master, self.roidata.n_roi, n_design, design_labels,
            "Define Functional Connectivity (FC)", save_fc_matrices
        )


    def open_ec_matrix_input(self):
        n_design = self.designdata.n_design
        design_labels = [f"Design {i + 1}" for i in range(n_design)]

        def save_ec_matrices(ec_matrices, delay_matrices):
            self.ec_matrix = ec_matrices
            self.ec_delay_matrix = delay_matrices
            print("EC matrices saved (per design):")
            for d, m in enumerate(ec_matrices):
                print(f"Design {d + 1}:\n{m}")
            print("Delay matrices saved (per design):")
            for d, m in enumerate(delay_matrices):
                print(f"Design {d + 1}:\n{m}")

        EffectiveConnectivityMatrixInput(
            self.master, self.roidata.n_roi, n_design, design_labels, save_ec_matrices
    )
    # def open_fc_matrix_input(self):
    #     def save_fc_matrix(matrix):
    #         self.fc_matrix = matrix
    #         print("FC matrix saved:")
    #         print(matrix)

    #     fc_window = RoI_to_RoI_MatrixInput(self.master, self.roidata.n_roi, "Define Functional Connectivity (FC)", save_fc_matrix)

    #     # Автозаполнение: 0 везде, 1 на диагонали
    #     for i in range(self.roidata.n_roi):
    #         for j in range(self.roidata.n_roi):
    #             if i == j:
    #                 fc_window.entries[i][j].delete(0, tk.END)
    #                 fc_window.entries[i][j].insert(0, "1.0")
    #                 fc_window.entries[i][j].config(state='readonly')
    #             else:
    #                 fc_window.entries[i][j].delete(0, tk.END)
    #                 fc_window.entries[i][j].insert(0, "0.0")

    # def open_ec_matrix_input(self):
    #     def save_ec_matrices(ec_matrix, delay_matrix):
    #         self.ec_matrix = ec_matrix
    #         self.ec_delay_matrix = delay_matrix
    #         print("EC matrix saved:")
    #         print(ec_matrix)
    #         print("Delay matrix saved:")
    #         print(delay_matrix)

    #     ec_window = EffectiveConnectivityMatrixInput(self.master, self.roidata.n_roi, save_ec_matrices)

    #     #  Автозаполнение значениями:
    #     for i in range(self.roidata.n_roi):
    #         for j in range(self.roidata.n_roi):
    #             # EC matrix: диагональ = 1.0, остальные = 0.0
    #             ec_window.entries_ec[i][j].delete(0, tk.END)
    #             if i == j:
    #                 ec_window.entries_ec[i][j].insert(0, "1.0")
    #                 ec_window.entries_ec[i][j].config(state='readonly')
    #             else:
    #                 ec_window.entries_ec[i][j].insert(0, "0.0")

    #             # Delay matrix: всё = 0.0
    #             ec_window.entries_delay[i][j].delete(0, tk.END)
    #             ec_window.entries_delay[i][j].insert(0, "0.0")

    # def task_fc_ec_selection_window(self):
    #     selection_window = tk.Toplevel(self.master)
    #     selection_window.title("Select Analysis Type")
    #     selection_window.geometry("320x180")
    #     self.apply_theme(selection_window)

    #     # расположение справа от главного окна
    #     self.master.update_idletasks()
    #     x = self.master.winfo_rootx()
    #     y = self.master.winfo_rooty()
    #     w = self.master.winfo_width()
    #     selection_window.geometry(f"+{x + w + 20}+{y + 100}")

    #     selection_window.transient(self.master)
    #     selection_window.lift()
    #     selection_window.attributes("-topmost", True)

    #     tk.Label(selection_window, text="Choose connectivity type:", font=("Segoe UI", 11, "bold"),
    #             bg=COLOR_THEMES[self.current_theme]["bg"]).pack(pady=10)

    #     self.connectivity_choice = tk.StringVar(value="FC")
    #     fc_radio = tk.Radiobutton(selection_window, text="Functional Connectivity (FC)",
    #                             variable=self.connectivity_choice, value="FC",
    #                             bg=COLOR_THEMES[self.current_theme]["bg"])
    #     ec_radio = tk.Radiobutton(selection_window, text="Effective Connectivity (EC)",
    #                             variable=self.connectivity_choice, value="EC",
    #                             bg=COLOR_THEMES[self.current_theme]["bg"])

    #     fc_radio.pack(anchor="w", padx=30)
    #     ec_radio.pack(anchor="w", padx=30)

    #     def proceed():
    #         self.selected_connectivity_type = self.connectivity_choice.get()
    #         print(f"Selected: {self.selected_connectivity_type}")
    #         selection_window.destroy()
    #         # здесь вызвать анализ:
    #         if self.selected_connectivity_type == "FC":
    #             self.open_fc_matrix_input()
    #         elif self.selected_connectivity_type == "EC":
    #             self.open_ec_matrix_input()
    #     proceed_button = tk.Button(selection_window, text="Proceed", command=proceed)
    #     proceed_button.pack(pady=20)

        
    def select_output_path(self):
        self.output_path = filedialog.askdirectory()
        if self.output_path:
            messagebox.showinfo("Output Path Selected", f"Output path set to: {self.output_path}")
            self.submit_all_data()
        else:
            messagebox.showwarning("No Path Selected", "Please select a valid output path for BIDS export.")

    def validate_tr_and_volumes(self):
        try:
            self.tr = float(self.design_tr_num.get())
            self.volumes_quant = int(self.design_vol_num.get())

            if self.tr <= 0 or self.tr > 5:
                raise ValueError("TR must be between 0 and 5.")
            if self.volumes_quant <= 0 or self.volumes_quant > 1000:
                raise ValueError("Number of volumes must be between 1 and 1000.", parent=self.omw_place)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False
            return True

    def submit_all_data(self):
        # использует значения из EstimateWindow
        subjects = int(self.subjects)
        sessions = int(self.sessions)

        if self.template_path is None:
            messagebox.showerror("Error", "Please load a template before submitting.")
            return
        if not any(self.roidata.list_roi):
            messagebox.showerror("Error", "Please load at least one RoI before submitting.")
            return
        if not self.matrixdata:
            messagebox.showerror("Error", "Please define the matrix before submitting.")
            return
        if not self.output_path:
            messagebox.showerror("Error", "Please select an output path.")
            return

        self.bids_data['subjects'] = subjects
        self.bids_data['sessions'] = sessions
        self.bids_data['template_path'] = self.template_path
        self.bids_data['roi_list'] = self.roidata.list_roi
        self.bids_data['matrix'] = self.matrixdata.get_matrix_data()

        self.create_bids_structure(self.output_path)

    def create_bids_structure(self, output_path):
        subjects = int(self.bids_data['subjects'])
        sessions = int(self.bids_data['sessions'])
        matrix_data = self.matrixdata.three_d_matrix if self.matrixdata and self.matrixdata.three_d_matrix is not None else None

        if matrix_data is None:
            messagebox.showerror("Error", "Matrix data is missing. Please ensure it is defined before exporting.")
            return

        # volumes_quant из Design
        if not self.volumes_quant or self.volumes_quant <= 0:
            self.volumes_quant = int(self.designdata.volumes_quant or 0)
        if not self.volumes_quant:
            messagebox.showerror("Error", "Number of volumes is not set.")
            return

        func_metadata = {
            "RepetitionTime": float(self.designdata.tr or 2.0),
            "TaskName": "ExampleTask"
        }
        anat_metadata = {
            "Manufacturer": "ExampleManufacturer",
            "MagneticFieldStrength": 3
        }

        try:
            # Базовый шаблон  3D 
            img_data = self.img.get_fdata()
            template_data_3d = img_data[..., 0] if img_data.ndim == 4 else img_data
            template_data_3d = template_data_3d.astype(np.float32)

            # Контейнер 4D
            func_data_4d = np.zeros((*template_data_3d.shape, self.volumes_quant), dtype=np.float32)

 
            for t in range(self.volumes_quant):
                combined_roi = np.zeros_like(template_data_3d, dtype=np.float32)

                for roi_index, roi_mask_data in enumerate(self.roidata.roi_maps):
                    if roi_mask_data is None:
                        continue

                    if roi_index < matrix_data.shape[0] and t < matrix_data.shape[2]:
                        roi_effect_percent_t = float(np.sum(matrix_data[roi_index, :, t]))  # это %!
                    else:
                        roi_effect_percent_t = 0.0

                    if roi_effect_percent_t != 0.0:
                        combined_roi += (roi_mask_data.astype(np.float32) * roi_effect_percent_t)

                
                frame_data = template_data_3d * (1.0 + combined_roi / 100.0)
                func_data_4d[..., t] = frame_data

            
            # Гауссов шум, если задан
            cfg = getattr(self, "noise_config", {}) or {}
            if "Gaussian" in cfg:
                z = float(cfg["Gaussian"]["Z"])
                seed = cfg["Gaussian"].get("Seed", None)
                func_data_4d = add_gaussian_noise(func_data_4d, mean=0.0, std=z, seed=seed)

           
            trend_cfg = cfg.get("Trend", {})
            if "masks" in trend_cfg:
                tr_val = float(self.designdata.tr or 0.0)
                if tr_val <= 0:
                    raise ValueError("TR not given.")

               
                for m in trend_cfg["masks"]:
                    try:
                        mpath = m.get("path", "")
                        mz = float(m.get("Z", 0.0))
                        # читает и валидирует порядок; по умолчанию 1
                        morder = int(m.get("order", 1))
                        if morder < 1:
                            morder = 1

                        if not mpath or mz <= 0:
                            continue

                        mnii = nib.load(mpath)
                        mdata = mnii.get_fdata()
                        if mdata.ndim == 4:
                            mdata = mdata[..., 0]
                        if mdata.shape != template_data_3d.shape:
                            raise ValueError(
                                f"Trend mask shape {mdata.shape} does not match template shape {template_data_3d.shape}."
                            )
                        mbool = (mdata > 0)

                        # Добавляет полиномиальный тренд с указанным порядком
                        func_data_4d = add_linear_trend(
                            func_data_4d,
                            z_per_sec=mz,
                            tr=tr_val,
                            mask=mbool,
                            auto_mask_percentile=5.0,
                            order=morder,   #ключевая строка
                        )
                    except Exception as me:
                        print(f"Trend mask skipped due to error: {me}")



            mode = self.sntc_dropdown.get()

            for sub in range(1, subjects + 1):
                for ses in range(1, sessions + 1):
                    sub_folder = os.path.join(output_path, f"sub-{sub:02d}")
                    ses_folder = os.path.join(sub_folder, f"ses-{ses:02d}")
                    anat_folder = os.path.join(ses_folder, "anat")
                    func_folder = os.path.join(ses_folder, "func")
                    os.makedirs(anat_folder, exist_ok=True)
                    os.makedirs(func_folder, exist_ok=True)

                    # sidecars
                    anat_json_path = os.path.join(anat_folder, f"sub-{sub:02d}_ses-{ses:02d}_T1w.json")
                    func_json_path = os.path.join(func_folder, f"sub-{sub:02d}_ses-{ses:02d}_task-example_bold.json")
                    with open(anat_json_path, 'w') as f: json.dump(anat_metadata, f, indent=4)
                    with open(func_json_path, 'w') as f: json.dump(func_metadata, f, indent=4)

                    # anat 
                    anat_nii_path = os.path.join(anat_folder, f"sub-{sub:02d}_ses-{ses:02d}_T1w.nii")
                    anat_img_to_save = nib.Nifti1Image(template_data_3d, self.img.affine)
                    nib.save(anat_img_to_save, anat_nii_path)

                    # func 
                    if mode == "Save one 4D .nii file":
                        func_nii_path = os.path.join(func_folder, f"sub-{sub:02d}_ses-{ses:02d}_task-example_bold.nii")
                        func_img = nib.Nifti1Image(func_data_4d, self.img.affine)
                        nib.save(func_img, func_nii_path)
                    else:
                        for t in range(self.volumes_quant):
                            vol3d = func_data_4d[..., t]
                            vol_img = nib.Nifti1Image(vol3d, self.img.affine)
                            vol_path = os.path.join(
                                func_folder,
                                f"sub-{sub:02d}_ses-{ses:02d}_task-example_bold_frame-{t+1:04d}.nii"
                            )
                            nib.save(vol_img, vol_path)

            messagebox.showinfo("BIDS Export", "Data exported successfully.")
        except Exception as e:
            print(f"Export error: {e}")
            messagebox.showerror("Error", f"Failed to export BIDS: {e}")

    def save_roi_masks_as_nifti(self):
        if not self.roidata.list_roi or not self.output_path:
            messagebox.showerror("Error",
                                 "RoI masks or output path not set. Please define RoIs and select output path.")
            return

        roi_data = self.roidata.list_roi 
        designs = self.designdata.collapsed_designs  
        n_volumes = self.volumes_quant

       
        try:
            for roi_idx, roi_mask in enumerate(roi_data):
                if roi_idx >= len(designs):
                    messagebox.showerror("Design Error", f"Design for RoI {roi_idx + 1} not found.")
                    return

                
                roi_4d_data = np.repeat(roi_mask[..., np.newaxis], n_volumes, axis=-1)

                
                affine = np.eye(4)
                roi_img = nib.Nifti1Image(roi_4d_data, affine)

                
                roi_output_path = os.path.join(self.output_path, f"roi_{roi_idx + 1}_mask.nii")
                nib.save(roi_img, roi_output_path)

                print(f"Saved RoI mask for RoI {roi_idx + 1} with {n_volumes} volumes at {roi_output_path}")

            messagebox.showinfo("RoI Masks Saved", "RoI masks saved as 4D NIfTI files successfully.")

        except Exception as e:
            print(f"Error saving RoI masks as NIfTI files: {e}")
            messagebox.showerror("Error", f"Failed to save RoI masks. Error: {e}")



    def save_roi_masks(self):
        if not self.output_path:
            messagebox.showerror("Output Path Error", "Please select an output path for BIDS and ROI masks.")
            return

        if self.volumes_quant is None:
            messagebox.showerror("Volume Error", "Please set the number of volumes before saving RoI masks.")
            return

        subjects = int(self.subject_entry.get())
        sessions = int(self.session_entry.get())

        template_affine = self.img.affine if self.img else np.eye(4)
        template_data = self.img.get_fdata()

        for sub in range(1, subjects + 1):
            for ses in range(1, sessions + 1):
                func_folder = os.path.join(self.output_path, f"sub-{sub:02d}", f"ses-{ses:02d}", "func")
                os.makedirs(func_folder, exist_ok=True)

                for roi_index, roi_mask_data in enumerate(self.roidata.roi_maps):
                    if isinstance(roi_mask_data, np.ndarray) and np.any(roi_mask_data):
                        updated_template_4d = np.zeros((*template_data.shape[:3], self.volumes_quant))

                        matrix_values = self.matrixdata.three_d_matrix[roi_index]

                    for frame in range(self.volumes_quant):
                        frame_multiplier = matrix_values[0, frame]  
                       
                        frame_adjustment = roi_mask_data * template_data * frame_multiplier / 100

                    
                        updated_template_4d[..., frame] = template_data + frame_adjustment

                        updated_template_img = nib.Nifti1Image(updated_template_4d, template_affine)
                        template_file_path = os.path.join(func_folder, f"sub-{sub:02d}_ses-{ses:02d}_roi_{roi_index + 1}_mask.nii")

                        try:
                            nib.save(updated_template_img, template_file_path)
                            print(f"Updated template with RoI {roi_index + 1} saved: {template_file_path}")
                        except IOError as e:
                            print(f"Failed to save updated template for RoI {roi_index + 1}. Error: {e}")
                            messagebox.showerror("Error", f"Failed to save updated template for RoI {roi_index + 1}. Error: {e}")
                    else:
                        print(f"Warning: RoI {roi_index + 1} does not contain valid mask data and was skipped.")

        messagebox.showinfo("Updated Templates Saved", "Updated templates with RoI effects have been saved successfully.")

      
    def select_output_path(self):
        self.output_path = filedialog.askdirectory()
        if not self.output_path:
            messagebox.showwarning("No Path Selected", "Please select a valid output path for BIDS export.")
            return

        # volumes из Design_data
        if self.volumes_quant is None:
            self.volumes_quant = self.designdata.volumes_quant
        if not self.volumes_quant or self.volumes_quant <= 0:
            messagebox.showerror("Volume Error", "Please set a valid number of volumes in the Design section.")
            return

        messagebox.showinfo("Output Path Selected", f"Output path set to: {self.output_path}")
        self.submit_all_data()

    def apply_theme(self, widget):
        colors = self.themes[self.current_theme]
        widget.configure(bg=colors["bg"])

        for child in widget.winfo_children():
            try:
                if isinstance(child, (tk.Label, tk.Button)):
                    child.configure(bg=colors["bg"], fg=colors["fg"])
                elif isinstance(child, tk.Entry):
                    child.configure(bg=colors["entry_bg"], fg=colors["fg"])
                elif isinstance(child, ttk.Combobox):
                    style = ttk.Style()
                    style.theme_use('clam')
                    style.configure("TCombobox",
                                    fieldbackground=colors["entry_bg"],
                                    background=colors["entry_bg"],
                                    foreground=colors["fg"])
                elif isinstance(child, tk.Frame):
                    self.apply_theme(child)  # Рекурсивно
            except:
                pass


root = tk.Tk()
app = TemplateLoaderApp(root)
root.mainloop()
