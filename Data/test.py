import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageTk
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import svgwrite
from networkx.algorithms.bipartite.projection import overlap_weighted_projected_graph


class VectorImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Scale")
        self.root.geometry("900x700")

        # Переменные для хранения параметров
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.scale_factor = tk.DoubleVar(value=2.0)
        self.contrast_factor = tk.DoubleVar(value=1.5)
        self.blur_strength = tk.DoubleVar(value=0.5)
        self.blur_type = tk.StringVar(value="gaussian")
        self.file_types = tk.StringVar(value="*.jpg;*.jpeg;*.png;*.bmp")

        # Параметры векторной трассировки
        self.vectorize_enabled = tk.BooleanVar(value=False)
        self.vector_method = tk.StringVar(value="vectorizers")
        self.corners_threshold = tk.DoubleVar(value=0.01)
        self.simplify_tolerance = tk.DoubleVar(value=1.0)
        self.min_path_length = tk.IntVar(value=5)
        self.vector_format = tk.StringVar(value="svg")
        self.color_reduction = tk.IntVar(value=8)
        self.trace_speed = tk.StringVar(value="normal")

        # Списки файлов
        self.image_files = []
        self.processed_count = 0

        self.setup_ui()

    def setup_ui(self):
        """Создание графического интерфейса"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Notebook для вкладок
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Вкладка растровой обработки
        raster_frame = ttk.Frame(notebook, padding="10")
        notebook.add(raster_frame, text="Растровая обработка")

        # Настройка растровой вкладки
        self.setup_raster_tab(raster_frame)

        # Общие элементы под notebook
        self.setup_common_controls(main_frame, 1)

    def setup_raster_tab(self, parent):
        """Настройка вкладки растровой обработки"""
        parent.columnconfigure(1, weight=1)

        # Input folder selection
        ttk.Label(parent, text="Папка с изображениями:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.input_folder, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent, text="Обзор", command=self.browse_input_folder).grid(row=0, column=2, padx=5)

        # Output folder selection
        ttk.Label(parent, text="Папка для результатов:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.output_folder, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent, text="Обзор", command=self.browse_output_folder).grid(row=1, column=2, padx=5)

        # File types
        ttk.Label(parent, text="Типы файлов:").grid(row=2, column=0, sticky=tk.W, pady=5)
        file_types_combo = ttk.Combobox(parent, textvariable=self.file_types, width=50)
        file_types_combo['values'] = (
            "*.jpg;*.jpeg;*.png;*.bmp",
            "*.jpg;*.jpeg",
            "*.png",
            "*.bmp",
            "*.*"
        )
        file_types_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)

        # Parameters frame
        params_frame = ttk.LabelFrame(parent, text="Параметры обработки", padding="10")
        params_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        params_frame.columnconfigure(1, weight=1)

        # Scale factor
        ttk.Label(params_frame, text="Увеличение:").grid(row=0, column=0, sticky=tk.W, pady=5)
        scale_scale = ttk.Scale(params_frame, from_=1.0, to=4.0, variable=self.scale_factor,
                                orient=tk.HORIZONTAL)
        scale_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        scale_label = ttk.Label(params_frame, textvariable=self.scale_factor)
        scale_label.grid(row=0, column=2, padx=5)

        # Contrast factor
        ttk.Label(params_frame, text="Контрастность:").grid(row=1, column=0, sticky=tk.W, pady=5)
        contrast_scale = ttk.Scale(params_frame, from_=1.0, to=3.0, variable=self.contrast_factor,
                                   orient=tk.HORIZONTAL)
        contrast_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        contrast_label = ttk.Label(params_frame, textvariable=self.contrast_factor)
        contrast_label.grid(row=1, column=2, padx=5)

        # Blur strength
        ttk.Label(params_frame, text="Размытие:").grid(row=2, column=0, sticky=tk.W, pady=5)
        blur_scale = ttk.Scale(params_frame, from_=0.0, to=2.0, variable=self.blur_strength,
                               orient=tk.HORIZONTAL)
        blur_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        blur_label = ttk.Label(params_frame, textvariable=self.blur_strength)
        blur_label.grid(row=2, column=2, padx=5)

        # Blur type
        ttk.Label(params_frame, text="Тип размытия:").grid(row=3, column=0, sticky=tk.W, pady=5)
        blur_combo = ttk.Combobox(params_frame, textvariable=self.blur_type, state="readonly")
        blur_combo['values'] = ("gaussian", "bilateral", "median")
        blur_combo.grid(row=3, column=1, sticky=tk.W, padx=5)

    def setup_vector_tab(self, parent):
        """Настройка вкладки векторной трассировки"""
        parent.columnconfigure(1, weight=1)

        # Enable vectorization
        ttk.Checkbutton(parent, text="Включить векторную трассировку",
                        variable=self.vectorize_enabled).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=10)

        # Vector method
        ttk.Label(parent, text="Метод трассировки:").grid(row=1, column=0, sticky=tk.W, pady=5)
        method_combo = ttk.Combobox(parent, textvariable=self.vector_method, state="readonly")
        method_combo['values'] = ("vectorizers", "contours", "color_zones", "edges")
        method_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)

        # Output format
        ttk.Label(parent, text="Формат вывода:").grid(row=2, column=0, sticky=tk.W, pady=5)
        format_combo = ttk.Combobox(parent, textvariable=self.vector_format, state="readonly")
        format_combo['values'] = ("svg", "eps")
        format_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)

        # Color reduction
        ttk.Label(parent, text="Уменьшение цветов:").grid(row=3, column=0, sticky=tk.W, pady=5)
        color_spin = ttk.Spinbox(parent, from_=2, to=64, textvariable=self.color_reduction, width=10)
        color_spin.grid(row=3, column=1, sticky=tk.W, padx=5)

        # Trace speed
        ttk.Label(parent, text="Скорость трассировки:").grid(row=4, column=0, sticky=tk.W, pady=5)
        speed_combo = ttk.Combobox(parent, textvariable=self.trace_speed, state="readonly")
        speed_combo['values'] = ("fast", "normal", "detailed")
        speed_combo.grid(row=4, column=1, sticky=tk.W, padx=5)

        # Advanced parameters frame
        adv_frame = ttk.LabelFrame(parent, text="Дополнительные параметры", padding="10")
        adv_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        adv_frame.columnconfigure(1, weight=1)

        # Corners threshold
        ttk.Label(adv_frame, text="Порог углов:").grid(row=0, column=0, sticky=tk.W, pady=5)
        corners_scale = ttk.Scale(adv_frame, from_=0.001, to=0.1, variable=self.corners_threshold,
                                  orient=tk.HORIZONTAL)
        corners_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        corners_label = ttk.Label(adv_frame, textvariable=self.corners_threshold)
        corners_label.grid(row=0, column=2, padx=5)

        # Simplify tolerance
        ttk.Label(adv_frame, text="Допуск упрощения:").grid(row=1, column=0, sticky=tk.W, pady=5)
        simplify_scale = ttk.Scale(adv_frame, from_=0.1, to=5.0, variable=self.simplify_tolerance,
                                   orient=tk.HORIZONTAL)
        simplify_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        simplify_label = ttk.Label(adv_frame, textvariable=self.simplify_tolerance)
        simplify_label.grid(row=1, column=2, padx=5)

        # Min path length
        ttk.Label(adv_frame, text="Мин. длина пути:").grid(row=2, column=0, sticky=tk.W, pady=5)
        min_path_spin = ttk.Spinbox(adv_frame, from_=1, to=100, textvariable=self.min_path_length, width=10)
        min_path_spin.grid(row=2, column=1, sticky=tk.W, padx=5)

        # Info text
        info_text = ("Векторная трассировка преобразует растровые изображения в векторные форматы.\n"
                     "Рекомендуется для логотипов, текста и простых графических элементов.\n"
                     "Метод 'vectorizers' использует специализированную библиотеку для лучшего качества.")
        ttk.Label(parent, text=info_text, foreground="blue", justify=tk.LEFT).grid(
            row=6, column=0, columnspan=3, sticky=tk.W, pady=10)

    def setup_common_controls(self, parent, row):
        """Общие элементы управления"""
        # Progress section
        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(progress_frame, text="Готов к работе")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=5)

        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row + 1, column=0, columnspan=3, pady=10)

        self.start_button = ttk.Button(button_frame, text="Начать обработку", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Предпросмотр", command=self.show_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Очистить", command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # File list
        list_frame = ttk.LabelFrame(parent, text="Найденные файлы", padding="5")
        list_frame.grid(row=row + 2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Treeview for file list
        columns = ('filename', 'status')
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        self.file_tree.heading('filename', text='Имя файла')
        self.file_tree.heading('status', text='Статус')
        self.file_tree.column('filename', width=400)
        self.file_tree.column('status', width=150)
        self.file_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure main frame row weights
        parent.rowconfigure(row + 2, weight=1)

    def browse_input_folder(self):
        """Выбор папки с исходными изображениями"""
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder:
            self.input_folder.set(folder)
            self.scan_image_files()

    def browse_output_folder(self):
        """Выбор папки для сохранения результатов"""
        folder = filedialog.askdirectory(title="Выберите папку для результатов")
        if folder:
            self.output_folder.set(folder)

    def scan_image_files(self):
        """Сканирование папки на наличие изображений"""
        if not self.input_folder.get():
            return

        input_path = Path(self.input_folder.get())
        file_patterns = self.file_types.get().split(';')

        self.image_files = []
        for pattern in file_patterns:
            self.image_files.extend(input_path.glob(pattern.strip()))

        # Обновляем список файлов в интерфейсе
        self.update_file_list()

        self.status_label.config(text=f"Найдено файлов: {len(self.image_files)}")

    def update_file_list(self):
        """Обновление списка файлов в Treeview"""
        # Очищаем текущий список
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)

        # Добавляем файлы
        for file_path in self.image_files:
            self.file_tree.insert('', tk.END, values=(file_path.name, "В ожидании"))

    def start_processing(self):
        """Запуск обработки в отдельном потоке"""
        if not self.input_folder.get():
            messagebox.showerror("Ошибка", "Выберите папку с изображениями")
            return

        if not self.output_folder.get():
            messagebox.showerror("Ошибка", "Выберите папку для результатов")
            return

        if not self.image_files:
            messagebox.showerror("Ошибка", "В выбранной папке нет изображений")
            return

        # Создаем папку для результатов если не существует
        Path(self.output_folder.get()).mkdir(parents=True, exist_ok=True)

        # Запускаем обработку в отдельном потоке
        self.start_button.config(state='disabled')
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()

    def process_images(self):
        """Обработка всех изображений"""
        total_files = len(self.image_files)
        self.processed_count = 0

        # Настраиваем прогресс-бар
        self.progress['maximum'] = total_files
        self.progress['value'] = 0

        for i, file_path in enumerate(self.image_files):
            try:
                self.status_label.config(text=f"Обрабатывается: {file_path.name}")

                # Обрабатываем изображение
                output_path = Path(self.output_folder.get()) / f"processed_{file_path.stem}"

                if self.vectorize_enabled.get():
                    # Векторная трассировка
                    output_path = output_path.with_suffix(f'.{self.vector_format.get()}')
                    self.vector_trace_image(str(file_path), str(output_path))
                else:
                    # Растровая обработка
                    output_path = output_path.with_suffix('.png')
                    self.process_single_image(str(file_path), str(output_path))

                # Обновляем статус
                self.processed_count += 1
                self.progress['value'] = self.processed_count

                # Обновляем статус в списке
                self.root.after(0, self.update_file_status, i, "✓ Готово")

            except Exception as e:
                print(f"Ошибка при обработке {file_path}: {e}")
                self.root.after(0, self.update_file_status, i, f"✗ Ошибка: {str(e)}")

        # Завершение обработки
        self.root.after(0, self.processing_finished)

    def update_file_status(self, index, status):
        """Обновление статуса файла в интерфейсе"""
        items = self.file_tree.get_children()
        if index < len(items):
            self.file_tree.set(items[index], 'status', status)

    def processing_finished(self):
        """Действия после завершения обработки"""
        self.status_label.config(
            text=f"Обработка завершена! Обработано: {self.processed_count}/{len(self.image_files)}")
        self.start_button.config(state='normal')
        messagebox.showinfo("Готово",
                            f"Обработка завершена!\nОбработано файлов: {self.processed_count}/{len(self.image_files)}")

    def process_single_image(self, input_path, output_path):
        """
        Обработка одного растрового изображения
        """
        # Загружаем изображение
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")

        # 1. Увеличиваем разрешение
        upscaled = self.upscale_image_simple(image, self.scale_factor.get())

        # 2. Повышаем контрастность
        contrasted = self.enhance_contrast_simple(upscaled, self.contrast_factor.get())

        # 3. Применяем размытие
        if self.blur_strength.get() > 0:
            final_image = self.apply_blur_simple(contrasted, self.blur_strength.get(), self.blur_type.get())
        else:
            final_image = contrasted

        # Сохраняем результат
        cv2.imwrite(output_path, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def vector_trace_image(self, input_path, output_path):
        """
        Векторная трассировка изображения
        """
        # Загружаем и предобрабатываем изображение
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")

        # Предварительная обработка для векторной трассировки
        processed = self.prepare_for_vector_tracing(image)

        # Выбираем метод трассировки
        if self.vector_method.get() == "vectorizers":
            self.trace_with_vectorizers(processed, output_path)
        elif self.vector_method.get() == "contours":
            self.trace_with_contours(processed, output_path)
        elif self.vector_method.get() == "color_zones":
            self.trace_color_zones(image, output_path)
        else:
            self.trace_with_edges(processed, output_path)

    def prepare_for_vector_tracing(self, image):
        """
        Подготовка изображения для векторной трассировки
        """
        # Увеличиваем разрешение
        if self.scale_factor.get() > 1.0:
            h, w = image.shape[:2]
            new_w, new_h = int(w * self.scale_factor.get()), int(h * self.scale_factor.get())
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Повышаем контрастность
        if self.contrast_factor.get() > 1.0:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(self.contrast_factor.get())
            image = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

        # Применяем размытие если нужно
        if self.blur_strength.get() > 0:
            image = self.apply_blur_simple(image, self.blur_strength.get() * 0.5, self.blur_type.get())

        return image
    def hi(self):
        print("Hi!")
    def trace_with_vectorizers(self, image, output_path):
        """
        Трассировка с использованием библиотеки vectorizers
        """
        try:
            import vectorizers

            # Конвертируем BGR в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Определяем параметры трассировки в зависимости от выбранной скорости
            speed_to_params = {
                "fast": {"corner_threshold": 100, "length_threshold": 3.0, "max_iterations": 10},
                "normal": {"corner_threshold": 80, "length_threshold": 2.0, "max_iterations": 20},
                "detailed": {"corner_threshold": 60, "length_threshold": 1.0, "max_iterations": 40}
            }

            params = speed_to_params.get(self.trace_speed.get(), speed_to_params["normal"])

            # Выполняем векторную трассировку
            if self.color_reduction.get() <= 8:
                # Для небольшого количества цветов используем VectorizedImage
                vectorized = vectorizers.VectorizedImage.from_image(
                    image_rgb,
                    colors=self.color_reduction.get(),
                    corner_threshold=params["corner_threshold"],
                    length_threshold=params["length_threshold"],
                    max_iterations=params["max_iterations"]
                )
            else:
                # Для большего количества цветов используем сегментацию
                vectorized = vectorizers.VectorizedImage.from_image(
                    image_rgb,
                    corner_threshold=params["corner_threshold"],
                    length_threshold=params["length_threshold"],
                    max_iterations=params["max_iterations"]
                )

            # Сохраняем в SVG
            vectorized.save_svg(output_path)

        except ImportError:
            messagebox.showerror("Ошибка",
                                 "Библиотека vectorizers не установлена!\n"
                                 "Установите её с помощью: pip install vectorizers\n"
                                 "Используется резервный метод трассировки.")
            self.trace_with_contours(image, output_path)
        except Exception as e:
            print(f"Ошибка vectorizers: {e}")
            messagebox.showerror("Ошибка", f"Ошибка vectorizers: {e}\nИспользуется резервный метод.")
            self.trace_with_contours(image, output_path)

    def trace_with_contours(self, image, output_path):
        """
        Трассировка с использованием контуров OpenCV
        """
        # Конвертируем в grayscale и бинаризуем
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Создаем SVG
        height, width = image.shape[:2]
        dwg = svgwrite.Drawing(output_path, size=(width, height), profile='full')

        # Добавляем белый фон
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))

        # Добавляем контуры в SVG
        for contour in contours:
            if len(contour) < self.min_path_length.get():
                continue

            # Упрощаем контур
            epsilon = self.corners_threshold.get() * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Создаем путь
            if len(approx) > 1:
                path_data = f"M {approx[0][0][0]},{approx[0][0][1]}"
                for point in approx[1:]:
                    path_data += f" L {point[0][0]},{point[0][1]}"
                path_data += " Z"

                dwg.add(dwg.path(d=path_data, fill="black", stroke="black", stroke_width=1))

        dwg.save()

    def trace_color_zones(self, image, output_path):
        """
        Трассировка цветовых зон с уменьшением количества цветов
        """
        # Уменьшаем количество цветов
        data = np.float32(image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, self.color_reduction.get(), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        reduced = centers[labels.flatten()].reshape(image.shape)

        height, width = reduced.shape[:2]
        dwg = svgwrite.Drawing(output_path, size=(width, height), profile='full')

        # Добавляем белый фон
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))

        # Для каждого цвета создаем маску и находим контуры
        for i, color in enumerate(centers):
            # Создаем маску для текущего цвета
            mask = cv2.inRange(reduced, color, color)

            # Находим контуры
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Добавляем контуры в SVG
            for contour in contours:
                if len(contour) < self.min_path_length.get():
                    continue

                # Упрощаем контур
                epsilon = self.corners_threshold.get() * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Создаем путь
                if len(approx) > 1:
                    path_data = f"M {approx[0][0][0]},{approx[0][0][1]}"
                    for point in approx[1:]:
                        path_data += f" L {point[0][0]},{point[0][1]}"
                    path_data += " Z"

                    # Преобразуем BGR в RGB для SVG
                    fill_color = f"rgb({color[2]},{color[1]},{color[0]})"
                    dwg.add(dwg.path(d=path_data, fill=fill_color, stroke="none"))

        dwg.save()

    def trace_with_edges(self, image, output_path):
        """
        Трассировка на основе детекции границ
        """
        # Детекция границ
        edges = cv2.Canny(image, 50, 150)

        # Утолщаем границы
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        height, width = edges.shape
        dwg = svgwrite.Drawing(output_path, size=(width, height), profile='full')

        # Добавляем белый фон
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))

        # Находим контуры границ
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Добавляем контуры в SVG
        for contour in contours:
            if len(contour) < self.min_path_length.get():
                continue

            # Упрощаем контур
            epsilon = self.corners_threshold.get() * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Создаем путь
            if len(approx) > 1:
                path_data = f"M {approx[0][0][0]},{approx[0][0][1]}"
                for point in approx[1:]:
                    path_data += f" L {point[0][0]},{point[0][1]}"

                dwg.add(dwg.path(d=path_data, fill="none", stroke="black", stroke_width=2))

        dwg.save()

    def upscale_image_simple(self, image, scale_factor):
        """Простое увеличение разрешения"""
        if scale_factor == 1.0:
            return image

        h, w = image.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def enhance_contrast_simple(self, image, contrast_factor):
        """Увеличение контрастности"""
        if contrast_factor == 1.0:
            return image

        # Используем PIL для контраста
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(contrast_factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

    def apply_blur_simple(self, image, blur_strength, blur_type):
        """Применение размытия"""
        if blur_type == "gaussian":
            kernel_size = max(3, int(blur_strength * 4) * 2 + 1)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_strength)
        elif blur_type == "bilateral":
            return cv2.bilateralFilter(image, 9, blur_strength * 50, blur_strength * 50)
        elif blur_type == "median":
            kernel_size = max(3, int(blur_strength * 4) * 2 + 1)
            return cv2.medianBlur(image, kernel_size)
        else:
            return image

    def show_preview(self):
        """Показать предпросмотр"""
        if not self.image_files:
            messagebox.showwarning("Предупреждение", "Нет изображений для предпросмотра")
            return

        first_image = self.image_files[0]
        self.create_preview_window(str(first_image))

    def create_preview_window(self, image_path):
        """Создание окна предпросмотра"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Предпросмотр обработки")
        preview_window.geometry("800x600")

        # Загружаем и обрабатываем изображение для предпросмотра
        original_image = cv2.imread(image_path)

        if self.vectorize_enabled.get():
            # Для векторной трассировки показываем бинаризованную версию
            processed = self.prepare_for_vector_tracing(original_image)

            if self.vector_method.get() == "vectorizers":
                # Для vectorizers создаем временный SVG и конвертируем в изображение
                temp_svg = "temp_preview.svg"
                self.trace_with_vectorizers(processed, temp_svg)

                # Конвертируем SVG в изображение для предпросмотра
                try:
                    import cairosvg
                    cairosvg.svg2png(url=temp_svg, write_to="temp_preview.png")
                    processed = cv2.imread("temp_preview.png")
                    # Удаляем временные файлы
                    if os.path.exists(temp_svg):
                        os.remove(temp_svg)
                    if os.path.exists("temp_preview.png"):
                        os.remove("temp_preview.png")
                except ImportError:
                    # Если cairosvg не установлен, используем обычную обработку
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            else:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        else:
            # Для растровой обработки
            upscaled = self.upscale_image_simple(original_image, self.scale_factor.get())
            contrasted = self.enhance_contrast_simple(upscaled, self.contrast_factor.get())

            if self.blur_strength.get() > 0:
                processed = self.apply_blur_simple(contrasted, self.blur_strength.get(), self.blur_type.get())
            else:
                processed = contrasted

            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # Конвертируем для отображения в Tkinter
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        original_pil = Image.fromarray(original_rgb)
        processed_pil = Image.fromarray(processed)

        # Масштабируем для отображения
        display_size = (400, 300)
        original_display = original_pil.resize(display_size, Image.LANCZOS)
        processed_display = processed_pil.resize(display_size, Image.LANCZOS)

        original_photo = ImageTk.PhotoImage(original_display)
        processed_photo = ImageTk.PhotoImage(processed_display)

        # Создаем интерфейс предпросмотра
        ttk.Label(preview_window, text="Оригинал", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=5)
        ttk.Label(preview_window, text="После обработки", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=10,
                                                                                           pady=5)

        original_label = ttk.Label(preview_window, image=original_photo)
        original_label.image = original_photo  # Keep a reference
        original_label.grid(row=1, column=0, padx=10, pady=5)

        processed_label = ttk.Label(preview_window, image=processed_photo)
        processed_label.image = processed_photo  # Keep a reference
        processed_label.grid(row=1, column=1, padx=10, pady=5)

        # Информация о параметрах
        mode = "Векторная трассировка" if self.vectorize_enabled.get() else "Растровая обработка"
        method = self.vector_method.get() if self.vectorize_enabled.get() else "N/A"
        info_text = f"Режим: {mode}\nМетод: {method}\nМасштаб: {self.scale_factor.get()}x\nКонтраст: {self.contrast_factor.get()}\nРазмытие: {self.blur_strength.get()}"
        ttk.Label(preview_window, text=info_text, justify=tk.LEFT).grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(preview_window, text="Закрыть", command=preview_window.destroy).grid(row=3, column=0, columnspan=2,
                                                                                        pady=10)

    def clear_all(self):
        """Очистка всех полей"""
        self.input_folder.set("")
        self.output_folder.set("")
        self.image_files = []
        self.update_file_list()
        self.progress['value'] = 0
        self.status_label.config(text="Готов к работе")


def main():
    """Запуск приложения"""
    root = tk.Tk()
    app = VectorImageProcessor(root)
    root.mainloop()


if __name__ == "__main__":
    main()