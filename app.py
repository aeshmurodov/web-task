import os
import io
import base64
import uuid
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2 # Оставим для Гаусса

# Импортируем session из Flask
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from flask_session import Session # Импортируем

# --- Конфигурация ---
RESULTS_FOLDER = 'results'
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

app = Flask(__name__)

# !!! ВАЖНО: Установите секретный ключ для работы сессий !!!
# В реальном приложении используйте более надежный ключ и храните его безопасно
app.config['SECRET_KEY'] = os.urandom(24) # Генерирует случайный ключ при каждом запуске

# --- Настройка Flask-Session ---
# Выбираем тип хранения. 'filesystem' - самый простой для начала.
# Для продакшена лучше Redis или база данных.
app.config['SESSION_TYPE'] = 'filesystem'
# Указываем папку для файлов сессий (создастся автоматически)
app.config['SESSION_FILE_DIR'] = './.flask_session/'
# Не хранить сессии вечно (опционально)
app.config['SESSION_PERMANENT'] = False
# Применить настройки сессии к приложению
Session(app)
# --------------------------------

app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Вспомогательные функции ---
def fig_to_base64_uri(fig):
    """Конвертирует фигуру Matplotlib в base64 Data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img_str}'

# --- Маршруты ---
@app.route('/')
def index():
    """Отображает главную страницу с редактором."""
    # Очищаем данные графиков при первой загрузке страницы (или принудительно)
    # session.pop('profile_plot_data', None) # Раскомментируйте, если нужно чистить при каждом заходе
    return render_template('index.html')

@app.route('/reset_plots', methods=['POST'])
def reset_plots():
    """Сбрасывает накопленные данные графиков профилей в сессии."""
    session.pop('profile_plot_data', None) # Удаляем ключ из сессии
    app.logger.info("Данные графиков профилей сброшены для текущей сессии.")
    return jsonify({"status": "success", "message": "Графики сброшены"})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Анализирует часть изображения: размытие, 2D спектр, накопление сечений."""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"status": "error", "message": "Нет данных изображения"}), 400

        image_data_url = data['image_data']
        gaussian_sigma = 1.5
        gaussian_ksize = (5, 5)

        # Инициализируем/получаем список данных для графиков из сессии
        if 'profile_plot_data' not in session:
            session['profile_plot_data'] = [] # Список для хранения пар (x_coords, y_coords)
            app.logger.info("Инициализирован пустой список для данных профилей в сессии.")

        # 1. Декодирование Base64
        header, encoded = image_data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        try:
            image_format = header.split(';')[0].split('/')[-1]
            if image_format not in ['png', 'jpeg', 'jpg', 'gif', 'bmp']: image_format = 'png'
            if image_format == 'jpeg': image_format = 'jpg'
        except: image_format = 'png'

        # 2. Открытие изображения PIL и сохранение кадра
        unique_id = uuid.uuid4()
        crop_filename = f"{unique_id}_crop.{image_format}"
        crop_path = os.path.join(app.config['RESULTS_FOLDER'], crop_filename)
        try:
            img_pil_color = Image.open(io.BytesIO(image_data))
            if img_pil_color.mode in ['RGBA', 'P']: img_pil_color = img_pil_color.convert('RGB')
            img_pil_color.save(crop_path)
            app.logger.info(f"Сохранено исходное обрезанное: {crop_path}")
            img_pil_gray = img_pil_color.convert('L')
            img_array_gray = np.array(img_pil_gray)
        except Exception as e:
            app.logger.error(f"Ошибка открытия/сохранения: {e}", exc_info=True)
            return jsonify({"status": "error", "message": "Не удалось обработать изображение"}), 400

        if img_array_gray.size == 0: return jsonify({"status": "error", "message": "Изображение пустое"}), 400
        h, w = img_array_gray.shape

        # 3. Фильтр Гаусса (OpenCV)
        try:
            img_blurred = cv2.GaussianBlur(img_array_gray, gaussian_ksize, gaussian_sigma)
        except Exception as e:
            app.logger.warning(f"Ошибка фильтра Гаусса: {e}. Используется неразмытое изображение.", exc_info=True)
            img_blurred = img_array_gray

        # 4. Расчет 2D Фурье-спектра (по размытому)
        f = np.fft.fft2(img_blurred)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)

        # 5. Генерация и СОХРАНЕНИЕ изображения спектра (текущего)
        spectrum_filename = f"{unique_id}_spectrum.png"
        spectrum_path = os.path.join(app.config['RESULTS_FOLDER'], spectrum_filename)
        spectrum_url = None
        try:
            fig_spectrum, ax_spectrum = plt.subplots(figsize=(6, 6))
            im_spec = ax_spectrum.imshow(magnitude_spectrum, cmap='gray')
            ax_spectrum.set_title('2D Спектр (текущий)')
            ax_spectrum.axis('off')
            fig_spectrum.savefig(spectrum_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig_spectrum) # Закрываем фигуру для сохранения
            app.logger.info(f"Сохранен текущий спектр: {spectrum_path}")
            # Генерируем base64 для немедленного показа
            fig_spec_uri, ax_spec_uri = plt.subplots(figsize=(6, 6))
            ax_spec_uri.imshow(magnitude_spectrum, cmap='gray')
            ax_spec_uri.set_title('2D Спектр (текущий)')
            ax_spec_uri.axis('off')
            spectrum_url = fig_to_base64_uri(fig_spec_uri) # Эта функция закроет фигуру fig_spec_uri
        except Exception as e:
            app.logger.error(f"Ошибка генерации/сохранения спектра: {e}", exc_info=True)
            # Не критично, можем продолжить без показа спектра

        # 6. Расчет сечения спектра под 45 градусов (текущего)
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        diag_len = int(np.sqrt(magnitude_spectrum.shape[0]**2 + magnitude_spectrum.shape[1]**2))
        num_points = max(magnitude_spectrum.shape)
        t_values = np.linspace(-diag_len / 2, diag_len / 2, num_points)
        row_coords = center_y + t_values / np.sqrt(2)
        col_coords = center_x + t_values / np.sqrt(2)
        current_profile_45 = ndimage.map_coordinates(magnitude_spectrum,
                                                     np.vstack((row_coords, col_coords)),
                                                     order=1, mode='nearest')
        # Используем общую ось X для всех графиков для простоты
        # (предполагаем, что размер кадра сильно не меняется)
        # Можно сделать сложнее и хранить X для каждого графика отдельно
        current_profile_x_axis = np.linspace(-diag_len / (2*np.sqrt(2)), diag_len / (2*np.sqrt(2)), num_points)

        # 7. Добавляем ТЕКУЩИЕ данные профиля в сессию
        # Преобразуем numpy массивы в списки для совместимости с JSON (сессии)
        session['profile_plot_data'].append({
            'x': current_profile_x_axis.tolist(),
            'y': current_profile_45.tolist()
        })
        # Flask автоматически сохранит изменения в сессии в конце запроса
        session.modified = True # Явно указываем, что сессия изменена (на всякий случай для вложенных структур)
        num_plots = len(session['profile_plot_data'])
        app.logger.info(f"Добавлены данные профиля №{num_plots} в сессию.")

        # 8. Генерация и СОХРАНЕНИЕ КОМБИНИРОВАННОГО графика профилей
        profile_filename = f"{unique_id}_profile_combined.png" # Имя файла для комбинированного графика
        profile_path = os.path.join(app.config['RESULTS_FOLDER'], profile_filename)
        profile_url = None
        try:
            fig_profile_comb, ax_profile_comb = plt.subplots(figsize=(8, 5)) # Увеличим высоту для легенды

            # Выбираем цветовую карту
            colors = plt.get_cmap('viridis', num_plots + 2) # +2 чтобы цвета были более разнесены

            # Рисуем все графики из сессии
            for i, plot_data in enumerate(session['profile_plot_data']):
                x_data = np.array(plot_data['x']) # Обратно в numpy для plot
                y_data = np.array(plot_data['y'])
                ax_profile_comb.plot(x_data, y_data, color=colors(i / max(1, num_plots -1)), label=f'Анализ {i+1}') # Деление на N-1 для полного спектра

            ax_profile_comb.set_title(f'Накопленные сечения спектра под 45° ({num_plots} шт.)')
            ax_profile_comb.set_xlabel('Расстояние от центра')
            ax_profile_comb.set_ylabel('Log амплитуды')
            ax_profile_comb.grid(True)
            ax_profile_comb.legend(fontsize='small') # Добавляем легенду

            fig_profile_comb.tight_layout() # Улучшаем размещение элементов
            fig_profile_comb.savefig(profile_path, bbox_inches='tight', dpi=100)
            plt.close(fig_profile_comb) # Закрываем фигуру для сохранения
            app.logger.info(f"Сохранен комбинированный профиль: {profile_path}")

            # Генерируем base64 для немедленного показа
            # Пересоздаем фигуру для base64 (так как plt.close() закрывает объект)
            fig_prof_uri, ax_prof_uri = plt.subplots(figsize=(8, 5))
            for i, plot_data in enumerate(session['profile_plot_data']):
                 x_data_uri = np.array(plot_data['x'])
                 y_data_uri = np.array(plot_data['y'])
                 ax_prof_uri.plot(x_data_uri, y_data_uri, color=colors(i / max(1, num_plots -1)), label=f'Анализ {i+1}')
            ax_prof_uri.set_title(f'Накопленные сечения спектра под 45° ({num_plots} шт.)')
            ax_prof_uri.set_xlabel('Расстояние от центра')
            ax_prof_uri.set_ylabel('Log амплитуды')
            ax_prof_uri.grid(True)
            ax_prof_uri.legend(fontsize='small')
            fig_prof_uri.tight_layout()
            profile_url = fig_to_base64_uri(fig_prof_uri) # Эта функция закроет фигуру fig_prof_uri

        except Exception as e:
            app.logger.error(f"Ошибка при генерации/сохранении комбинированного профиля: {e}", exc_info=True)
            # Можем продолжить без показа профиля

        # 9. Отправка результатов клиенту (без наложения)
        return jsonify({
            "status": "success",
            "spectrum_url": spectrum_url, # Спектр текущего анализа
            "profile_url": profile_url,   # Комбинированный профиль
            "num_profiles": num_plots,      # Передаем кол-во графиков для информации
            "saved_files": {
                "crop": f"/results/{crop_filename}",
                "spectrum": f"/results/{spectrum_filename}",
                "profile": f"/results/{profile_filename}", # Ссылка на файл комбинированного профиля
            }
        })

    except Exception as e:
        app.logger.error(f"Неожиданная ошибка в /analyze: {e}", exc_info=True)
        # Важно: Если произошла ошибка до сохранения сессии, изменения могут не сохраниться
        return jsonify({"status": "error", "message": f"Внутренняя ошибка сервера: {e}"}), 500

# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: открыть Base64 и конвертировать в CV ---
def open_base64_to_cv(image_data_url):
    """Декодирует Data URL, открывает PIL, конвертирует в CV (BGR или L)."""
    try:
        header, encoded = image_data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        img_pil_color = Image.open(io.BytesIO(image_data))

        # Конвертация для OpenCV
        if img_pil_color.mode == 'RGBA':
            img_pil_proc = img_pil_color.convert('RGB')
        elif img_pil_color.mode == 'P':
            img_pil_proc = img_pil_color.convert('RGB')
        elif img_pil_color.mode == 'L':
            img_pil_proc = img_pil_color
        else:
            img_pil_proc = img_pil_color

        img_array = np.array(img_pil_proc)
        if img_pil_proc.mode == 'RGB':
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img_pil_proc.mode == 'L':
            img_cv = img_array
        else: # Неожиданный формат
            raise ValueError(f"Неподдерживаемый режим PIL: {img_pil_proc.mode}")

        # Определяем формат для ответа (предпочитаем PNG)
        try:
            mime_type = header.split(';')[0].split(':')[-1]
            original_format = mime_type.split('/')[-1]
            if original_format == 'jpeg': original_format = 'jpg'
        except:
            original_format = 'png' # По умолчанию

        output_format = 'png' # Всегда отвечаем PNG после обработки

        return img_cv, output_format
    except Exception as e:
        app.logger.error(f"Ошибка в open_base64_to_cv: {e}", exc_info=True)
        raise ValueError(f"Ошибка обработки входного изображения: {e}")


# --- Маршрут для ПРИМЕНЕНИЯ РАЗМЫТИЯ (Гаусс) ---
@app.route('/apply_blur', methods=['POST'])
def apply_blur_route():
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"status": "error", "message": "Нет данных изображения"}), 400

        image_data_url = data['image_data']
        try:
            sigma = float(data.get('sigma', 0)) # Теперь это просто сигма
            if sigma <= 0: # Размытие только для положительных
                 return jsonify({"status": "error", "message": "Сигма для размытия должна быть > 0"}), 400
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Неверное значение сигмы"}), 400

        img_cv, output_format = open_base64_to_cv(image_data_url)
        app.logger.info(f"Применяем размытие: sigma={sigma}")

        # Применяем Гаусс
        img_blurred_cv = cv2.GaussianBlur(img_cv, (0, 0), sigma)

        # Конвертируем результат в Base64
        fmt = f".{output_format}"
        is_success, buffer = cv2.imencode(fmt, img_blurred_cv)
        if not is_success: raise ValueError("Ошибка кодирования размытого изображения")
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        result_data_url = f'data:image/{output_format};base64,{result_base64}'

        return jsonify({ "status": "success", "processed_image_data": result_data_url })

    except ValueError as ve: # Ошибки обработки изображения
         return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Ошибка в /apply_blur: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Внутренняя ошибка сервера при размытии"}), 500

# --- НОВЫЙ Маршрут для ПОВЫШЕНИЯ РЕЗКОСТИ (Unsharp Mask) ---
@app.route('/apply_sharpen', methods=['POST'])
def apply_sharpen_route():
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"status": "error", "message": "Нет данных изображения"}), 400

        image_data_url = data['image_data']
        try:
            # Получаем "силу" резкости (ожидаем отрицательное значение от слайдера)
            strength = float(data.get('strength', 0))
            if strength >= 0: # Резкость только для отрицательных
                return jsonify({"status": "error", "message": "Сила резкости должна быть < 0"}), 400
            # Преобразуем отрицательную силу в положительные параметры для алгоритма
            # Можно подобрать коэффициенты экспериментально
            sigma = abs(strength) * 0.8 # Сигма для размытия в unsharp mask
            amount = abs(strength) * 0.3 # Коэффициент усиления маски (beta в addWeighted)
            if amount > 1.5: amount = 1.5 # Ограничим усиление
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Неверное значение силы резкости"}), 400

        img_cv, output_format = open_base64_to_cv(image_data_url)
        app.logger.info(f"Применяем резкость: strength={strength} (sigma={sigma:.2f}, amount={amount:.2f})")

        # 1. Создаем размытую версию
        img_blurred_cv = cv2.GaussianBlur(img_cv, (0, 0), sigma)

        # 2. Создаем маску резкости (оригинал + (оригинал - размытое) * amount)
        # Используем cv2.addWeighted: sharpened = original * (1 + amount) + blurred * (-amount)
        # alpha = 1 + amount
        # beta = -amount
        # gamma = 0
        alpha = 1.0 + amount
        beta = -amount

        # Убедимся, что типы данных подходят для addWeighted и избегаем переполнения
        # Конвертируем во float32 для расчета
        img_cv_float = img_cv.astype(np.float32)
        img_blurred_float = img_blurred_cv.astype(np.float32)

        img_sharpened_float = cv2.addWeighted(img_cv_float, alpha, img_blurred_float, beta, 0)

        # 3. Обрезаем значения, чтобы остались в диапазоне 0-255
        img_sharpened_float = np.clip(img_sharpened_float, 0, 255)

        # 4. Конвертируем обратно в исходный тип данных (обычно uint8)
        img_sharpened_cv = img_sharpened_float.astype(img_cv.dtype)


        # Конвертируем результат в Base64
        fmt = f".{output_format}"
        is_success, buffer = cv2.imencode(fmt, img_sharpened_cv)
        if not is_success: raise ValueError("Ошибка кодирования резкого изображения")
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        result_data_url = f'data:image/{output_format};base64,{result_base64}'

        return jsonify({ "status": "success", "processed_image_data": result_data_url })

    except ValueError as ve: # Ошибки обработки изображения
         return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Ошибка в /apply_sharpen: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Внутренняя ошибка сервера при повышении резкости"}), 500

@app.route('/results/<filename>')
def serve_result(filename):
    """Отдает сохраненные файлы из папки results."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    # Убедимся, что папка для сессий существует, если используем filesystem
    session_dir = app.config.get('SESSION_FILE_DIR', './.flask_session/')
    if not os.path.exists(session_dir):
         os.makedirs(session_dir)
    app.run(debug=True, host='0.0.0.0') 