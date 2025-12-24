import cv2
import numpy as np
from PIL import Image, ImageEnhance
import argparse
import os


def upscale_image(image, scale_factor, model_path='FSRCNN_x2.pb'):
    """
    Увеличивает разрешение изображения
    """
    # Загружаем модель
    super_res = cv2.dnn_superres.DnnSuperResImpl_create()
    super_res.readModel(model_path)
    super_res.setModel('fsrcnn', 2)  # FSRCNN всегда работает с масштабом 2

    # Применяем апскейл несколько раз если нужно больше чем 2x
    current_image = image.copy()
    current_scale = 1

    while current_scale < scale_factor:
        # Определяем масштаб для текущей итерации
        next_scale = min(2, scale_factor // current_scale)
        if next_scale == 1:
            next_scale = scale_factor // current_scale

        # Применяем апскейл
        if next_scale == 2:
            current_image = super_res.upsample(current_image)
        else:
            # Для нечетных масштабов используем интерполяцию
            h, w = current_image.shape[:2]
            new_w, new_h = int(w * next_scale), int(h * next_scale)
            current_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        current_scale *= next_scale

    return current_image


def enhance_contrast(image, contrast_factor):
    """
    Повышает контрастность изображения
    """
    # Преобразуем в PIL для удобства работы с контрастом
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Увеличиваем контраст
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(contrast_factor)

    # Конвертируем обратно в OpenCV формат
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)


def apply_blur(image, blur_strength, blur_type='gaussian'):
    """
    Применяет размытие к изображению
    """
    if blur_type == 'gaussian':
        # Гауссово размытие
        kernel_size = max(3, int(blur_strength * 2) * 2 + 1)  # Нечетное число
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_strength)

    elif blur_type == 'bilateral':
        # Билатеральное размытие (сохраняет границы)
        return cv2.bilateralFilter(image, 9, blur_strength * 20, blur_strength * 20)

    elif blur_type == 'median':
        # Медианное размытие
        kernel_size = max(3, int(blur_strength * 2) * 2 + 1)
        return cv2.medianBlur(image, kernel_size)

    else:
        return image

def main():
    """
    Основная функция с парсингом аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='Увеличение разрешения, контрастности и размытие изображения')
    parser.add_argument('input', help='Путь к входному изображению')
    parser.add_argument('output', help='Путь для сохранения результата')
    parser.add_argument('--scale', type=float, default=2.0, help='Коэффициент увеличения (по умолчанию: 2.0)')
    parser.add_argument('--contrast', type=float, default=1.5, help='Коэффициент контрастности (по умолчанию: 1.5)')
    parser.add_argument('--blur', type=float, default=0.5, help='Сила размытия (по умолчанию: 0.5)')
    parser.add_argument('--blur-type', choices=['gaussian', 'bilateral', 'median'],
    default='gaussian', help='Тип размытия (по умолчанию: gaussian)')
    args = parser.parse_args()
    # Проверяем существование входного файла
    if not os.path.exists(args.input):
        print(f"Ошибка: Файл {args.input} не найден!")
        return

    print("Загрузка изображения...")
    image = cv2.imread(args.input)

    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {args.input}")
        return

    original_size = image.shape[:2]
    print(f"Исходный размер: {original_size[1]}x{original_size[0]}")
    print(f"Параметры: масштаб={args.scale}x, контраст={args.contrast}, размытие={args.blur}")

    # 1. Увеличиваем разрешение
    print("Увеличение разрешения...")
    upscaled = upscale_image(image, args.scale)

    # 2. Повышаем контрастность
    print("Повышение контрастности...")
    contrasted = enhance_contrast(upscaled, args.contrast)

    # 3. Применяем размытие (если сила размытия > 0)
    if args.blur > 0:
        print("Применение размытия...")
        final_image = apply_blur(contrasted, args.blur, args.blur_type)
    # Сохраняем результат
    cv2.imwrite(args.output, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    final_size = final_image.shape[:2]
    print(f"Обработка завершена!")
    print(f"Итоговый размер: {final_size[1]}x{final_size[0]}")
    print(f"Результат сохранен в: {args.output}")

if __name__ == "__main__":
    # Проверяем аргументы командной строки
    import sys

    if len(sys.argv) > 1:
        # Используем аргументы командной строки
        main()
