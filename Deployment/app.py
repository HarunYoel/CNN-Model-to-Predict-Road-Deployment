import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Header aplikasi
st.title('Graded Challenge 7: Model to Predict Road Deployment')
st.write("Created by Harun")
st.write("This application predicts whether a road is **Plain** or **Pothole**.")

# Load model dan kelas
model = load_model('model.h5')
with open('class.txt', 'r') as file:
    class_names = [line.strip() for line in file]

# Upload gambar
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

# Mengecek apakah file diunggah
if uploaded_file is not None:
    # Tampilkan gambar
    img = image.load_img(uploaded_file, target_size=(150, 150))  # Sesuaikan ukuran dengan model Anda
    
    # Gunakan kolom untuk memusatkan gambar
    col1, col2, col3 = st.columns([1, 3, 1])  # Membagi kolom menjadi 3, dengan tengah lebih lebar
    with col1:
        st.write("")  # Kolom kosong untuk membuat ruang
    with col2:
        st.image(img, caption='Gambar yang diunggah', use_column_width=True)  # Gunakan kolom tengah
    with col3:
        st.write("")  # Kolom kosong untuk membuat ruang

    # Mengolah gambar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi jika diperlukan

    # Buat prediksi
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    predicted_class_name = class_names[predicted_class_index]

    # Tampilkan hasil prediksi dalam bentuk kartu
    st.subheader("Prediction Result")
    st.write(f"Predicted Class: **{predicted_class_name}**")

    # Tampilkan probabilitas
    st.write("Probabilities:")
    probability_df = {class_name: predictions[0][i] for i, class_name in enumerate(class_names)}
    st.bar_chart(probability_df)

else:
    st.warning("Silakan unggah gambar untuk melakukan prediksi.")
