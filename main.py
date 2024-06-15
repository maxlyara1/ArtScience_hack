import streamlit as st
import os
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image
import uuid
from datetime import datetime
import plotly.graph_objs as go
import time
from huggingface_hub import from_pretrained_keras


trend_model_path = '/Users/maksimlyara/.cache/huggingface/hub/models--jkang--drawing-artistic-trend-classifier/snapshots/4f9e21b7e27eac30cb59e9690aed62b9c163c270'
artist_model_path = '/Users/maksimlyara/.cache/huggingface/hub/models--jkang--drawing-artist-classifier/snapshots/6718d117e35ec850f7a4c0181c7c75f4a3e71af2'

trend_class_mapping = {
    0: 'кубизм',
    1: 'экспрессионизм',
    2: 'фовизм',
    3: 'граффити',
    4: 'импрессионизм',
    5: 'поп-арт',
    6: 'постимпрессионизм',
    7: 'сюрреализм'
}

artist_class_mapping = {
    0: 'Клод Моне',
    1: 'Анри Матисс',
    2: 'Жан-Мишель Баския',
    3: 'Кейт Харинг',
    4: 'Пабло Пикассо',
    5: 'Пьер Огюст Ренуар',
    6: 'Рене Магритт',
    7: 'Рой Лихтенштейн',
    8: 'Винсент Ван Гог',
    9: 'Василий Кандинский'
}

def load_model(model_path):
    return TFSMLayer(model_path, call_endpoint='serving_default')

def get_class_mapping(model_type):
    if model_type == 'Стили':
        return trend_class_mapping
    else:
        return artist_class_mapping

st.title("Интерфейс с возможностью классификации различных произведений искусства")
st.write("У вас есть возможность классифицировать произведения искусства по различным категориям: стили и художники, используя предварительно обученную нейронную сеть.")

model_type = st.selectbox('Что вы хотите классифицировать?', ('Стили', 'Художники'))
model_path = trend_model_path if model_type == 'Стили' else artist_model_path
class_mapping = get_class_mapping(model_type)
model = load_model(model_path)

def save_image_to_folder(image, category, original_filename, base_folder):
    folder_path = os.path.join(base_folder, category)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = original_filename
    while os.path.exists(os.path.join(folder_path, unique_filename)):
        unique_filename = f"{uuid.uuid4()}{file_extension}"

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(os.path.join(folder_path, unique_filename))

def plot_histogram(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    fig = go.Figure(data=[go.Bar(x=classes, y=counts, marker_color='gray')])
    fig.update_layout(title='Количество файлов в каждом классе', xaxis_title='Классы', yaxis_title='Количество файлов')
    st.plotly_chart(fig)

uploaded_files = st.file_uploader("Загрузите изображения (только JPEG и PNG)", type=["jpeg", "jpg", "png"], accept_multiple_files=True)

folder_uploaded = st.text_input("Или введите путь к папке с изображениями:")

submit_button = st.button('Готово')

if submit_button:
    all_files = []
    error_messages = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            all_files.append(uploaded_file)

    if folder_uploaded:
        folder_files = [os.path.join(folder_uploaded, file) for file in os.listdir(folder_uploaded) if file.lower().endswith(('jpeg', 'jpg', 'png'))]
        all_files.extend(folder_files)

    if all_files:
        base_folder = datetime.now().strftime("Результаты_Классификации_%Y_%m_%d_%H:%M:%S")
        os.makedirs(base_folder, exist_ok=True)
        
        class_counts = {label: 0 for label in class_mapping.values()}
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_files = len(all_files)

        start_time = time.time()
        
        for i, file in enumerate(all_files):
            try:
                if isinstance(file, str):
                    with open(file, "rb") as f:
                        image = Image.open(f).convert('RGB')
                    original_filename = os.path.basename(file)
                else:
                    image = Image.open(file).convert('RGB')
                    original_filename = file.name
                
                image_resized = image.resize((299, 299))
                img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
                img_array = tf.expand_dims(img_array, axis=0)
                img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)

                predictions = model(img_array)
                probabilities = predictions['output'].numpy()

                top_2_indices = tf.argsort(probabilities, axis=1, direction='DESCENDING').numpy()[0][:2]
                predicted_labels = [class_mapping[i] for i in top_2_indices]
                predicted_probabilities = [probabilities[0, i] for i in top_2_indices]

                category = predicted_labels[0]

                save_image_to_folder(image, category, original_filename, base_folder)
                class_counts[category] += 1

                with st.expander(f'Результаты для изображения: {original_filename}', expanded=False):
                    st.image(image, caption=f'Загруженное изображение: {original_filename}', use_column_width=True)
                    
                    fig = go.Figure(data=[go.Bar(x=predicted_labels, y=predicted_probabilities, marker_color='gray')])
                    fig.update_layout(title=f'Наиболее вероятный класс: {predicted_labels[0]}', xaxis_title='Классы', yaxis_title='Вероятности')
                    st.plotly_chart(fig)

                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / (i + 1)
                remaining_time = avg_time_per_image * (total_files - (i + 1))
                
                if remaining_time >= 60:
                    minutes = int(remaining_time // 60)
                    seconds = int(remaining_time % 60)
                    time_str = f"{minutes} минут {seconds} секунд"
                else:
                    seconds = int(remaining_time)
                    time_str = f"{seconds} секунд"
                
                processing_speed = 1 / avg_time_per_image
                
                progress_bar.progress((i + 1) / total_files)
                progress_text.text(
f"Обработка изображения {i + 1} из {total_files}. Осталось: {total_files - (i + 1)} изображений. "
f"Примерное оставшееся время: {time_str}. "
f"Скорость обработки: {processing_speed:.2f} изображений/сек.")

            except Exception as e:
                error_message = f"Ошибка при обработке файла: {file}. Ошибка: {str(e)}"
                error_messages.append(error_message)

        st.success(f"Изображения успешно классифицированы и сохранены! \n \n Папка: {base_folder}")
        
        plot_histogram(class_counts)

        if error_messages:
            with st.expander("Список файлов с ошибками"):
                for error in error_messages:
                    st.error(error)
    else:
        st.error('Пожалуйста, загрузите хотя бы одно изображение или укажите путь к папке.')
