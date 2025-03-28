import os
import io
import base64
import uuid
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use('Agg') # Важно: использовать бэкенд без GUI
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_from_directory

# --- Конфигурация ---
RESULTS_FOLDER = 'results'
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Лимит 16MB для загрузки

# --- Вспомогательная функция для конвертации фигуры Matplotlib в base64 ---
def fig_to_base64_uri(fig):
    """Конвертирует фигуру Matplotlib в base64 Data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Закрываем фигуру, чтобы освободить память
    return f'data:image/png;base64,{img_str}'

# --- Маршруты ---
@app.route('/')
def index():
    """Отображает главную страницу с редактором."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Анализирует часть изображения: строит 2D спектр и сечение."""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"status": "error", "message": "Нет данных изображения"}), 400

        image_data_url = data['image_data']

        # 1. Декодирование Base64 Data URL
        try:
            header, encoded = image_data_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            image_format = header.split(';')[0].split('/')[1] # Определяем формат (png, jpeg, etc.)
        except Exception as e:
            app.logger.error(f"Ошибка декодирования base64: {e}")
            return jsonify({"status": "error", "message": "Неверный формат Data URL"}), 400

        # 2. Открытие изображения и конвертация в Ч/Б NumPy массив
        try:
            img = Image.open(io.BytesIO(image_data))
            img_gray = img.convert('L')
            img_array = np.array(img_gray)
        except Exception as e:
            app.logger.error(f"Ошибка открытия изображения: {e}")
            return jsonify({"status": "error", "message": "Не удалось обработать изображение"}), 400

        if img_array.size == 0:
             return jsonify({"status": "error", "message": "Изображение пустое"}), 400

        h, w = img_array.shape

        # 3. Расчет 2D Фурье-спектра (амплитудный)
        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f) # Сдвиг нулевой частоты в центр
        # Амплитудный спектр в логарифмическом масштабе для лучшей визуализации
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9) # +1e-9 чтобы избежать log(0)

        # 4. Расчет сечения спектра под 45 градусов через центр
        center_y, center_x = h // 2, w // 2
        # Определяем длину диагонали для профиля
        diag_len = int(np.sqrt(h**2 + w**2))
        # Генерируем точки вдоль линии y - cy = x - cx (или row - cr = col - cc)
        # Используем параметрическое представление: x = cx + t*cos(45), y = cy + t*sin(45)
        # В координатах массива: col = cx + t/sqrt(2), row = cy + t/sqrt(2)
        num_points = max(h, w) # Количество точек для интерполяции
        t_values = np.linspace(-diag_len / 2, diag_len / 2, num_points)
        row_coords = center_y + t_values / np.sqrt(2)
        col_coords = center_x + t_values / np.sqrt(2)

        # Интерполяция значений спектра вдоль линии с помощью map_coordinates
        profile_45 = ndimage.map_coordinates(magnitude_spectrum,
                                             np.vstack((row_coords, col_coords)),
                                             order=1, # Линейная интерполяция
                                             mode='nearest') # Обработка выхода за границы

        # 5. Генерация изображений для спектра и профиля
        # Спектр
        fig_spectrum, ax_spectrum = plt.subplots(figsize=(6, 6))
        im = ax_spectrum.imshow(magnitude_spectrum, cmap='gray')
        ax_spectrum.set_title('2D Амплитудный Спектр (log scale)')
        ax_spectrum.axis('off')
        fig_spectrum.colorbar(im, ax=ax_spectrum, fraction=0.046, pad=0.04)
        spectrum_url = fig_to_base64_uri(fig_spectrum)

        # Профиль
        fig_profile, ax_profile = plt.subplots(figsize=(8, 4))
        # Используем t_values или просто индексы для оси X
        profile_x_axis = np.linspace(-diag_len / (2*np.sqrt(2)), diag_len / (2*np.sqrt(2)), num_points)
        # Используем profile_x_axis как более осмысленный масштаб, связанный с расстоянием от центра
        ax_profile.plot(profile_x_axis, profile_45)
        ax_profile.set_title('Сечение спектра под 45°')
        ax_profile.set_xlabel('Расстояние от центра вдоль диагонали')
        ax_profile.set_ylabel('Log амплитуды')
        ax_profile.grid(True)
        profile_url = fig_to_base64_uri(fig_profile)

        # 6. Сохранение исходного (обрезанного) изображения, спектра и профиля
        unique_id = uuid.uuid4()
        crop_filename = f"{unique_id}_crop.{image_format}"
        spectrum_filename = f"{unique_id}_spectrum.png"
        profile_filename = f"{unique_id}_profile.png"

        crop_path = os.path.join(app.config['RESULTS_FOLDER'], crop_filename)
        spectrum_path = os.path.join(app.config['RESULTS_FOLDER'], spectrum_filename)
        profile_path = os.path.join(app.config['RESULTS_FOLDER'], profile_filename)

        # Сохраняем обрезанное изображение
        img.save(crop_path)

        # Сохраняем графики (Matplotlib уже закрыл фигуры в fig_to_base64_uri, создаем их заново для сохранения)
        # Спектр
        fig_spec_save, ax_spec_save = plt.subplots(figsize=(6, 6))
        im_save = ax_spec_save.imshow(magnitude_spectrum, cmap='gray')
        ax_spec_save.set_title('2D Амплитудный Спектр (log scale)')
        ax_spec_save.axis('off')
        fig_spec_save.colorbar(im_save, ax=ax_spec_save, fraction=0.046, pad=0.04)
        fig_spec_save.savefig(spectrum_path, bbox_inches='tight')
        plt.close(fig_spec_save)

        # Профиль
        fig_prof_save, ax_prof_save = plt.subplots(figsize=(8, 4))
        ax_prof_save.plot(profile_x_axis, profile_45)
        ax_prof_save.set_title('Сечение спектра под 45°')
        ax_prof_save.set_xlabel('Расстояние от центра вдоль диагонали')
        ax_prof_save.set_ylabel('Log амплитуды')
        ax_prof_save.grid(True)
        fig_prof_save.savefig(profile_path, bbox_inches='tight')
        plt.close(fig_prof_save)


        # 7. Отправка результатов клиенту
        return jsonify({
            "status": "success",
            "spectrum_url": spectrum_url,
            "profile_url": profile_url,
            "saved_files": {
                "crop": f"/results/{crop_filename}",
                "spectrum": f"/results/{spectrum_filename}",
                "profile": f"/results/{profile_filename}"
            }
        })

    except Exception as e:
        app.logger.error(f"Неожиданная ошибка в /analyze: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Внутренняя ошибка сервера: {e}"}), 500

@app.route('/results/<filename>')
def serve_result(filename):
    """Отдает сохраненные файлы из папки results."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


# --- Запуск приложения ---
if __name__ == '__main__':
    # Убедитесь, что папка results существует
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    app.run(debug=True) # debug=True для разработки