import streamlit as st
from streamlit_option_menu import option_menu
import warnings
import webbrowser
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import os
import datetime as dt
# Function to count image files in a folder
def count_image_files(folder):
    file_count = 0
    for root, _, files in os.walk(f"./DATASET_PLAT_389/{folder}"):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_count += 1
    return file_count

# Fungsi untuk mendapatkan path folder berdasarkan pilihan pengguna
def get_folder_path(selected_folder):
    base_folder = "./DATASET_PLAT_389"  # Ganti dengan path folder utama
    return os.path.join(base_folder, selected_folder)

# Get the absolute path to the current directory
current_directory = Path(__file__).resolve().parent
# Append the path to your modules
sys.path.append(str(current_directory))

warnings.filterwarnings("ignore")
st.config.set_option("deprecation.showPyplotGlobalUse", False)
st.set_page_config(
    page_title="ANPR Yolo5 & EasyOCR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "ANPR Yolo5 & EasyOCR"
    }
)

try:
    # from ANPR_Database import *  # Import from Database.py 
    from ANPR_File import *  # Import from Detect.py
    from ANPR_Setup import *  # Import from Train.py
    from ANPR_Folder import *  # Import from ANPR_Folder.py
    from ANPR_Db import *  # Import from ANPR_Db.py
    
except ModuleNotFoundError:
    st.warning("Error: Module not found. Check file names and paths.")

        

selected = option_menu(menu_title=None,options=["Upload Image","Database","Training & Testing","Predict","ANPR"],orientation='horizontal',
                        icons=['cloud-upload', 'database-fill-check','gear','images', 'file-earmark-arrow-up'], menu_icon="cast",
                        default_index=0)

if selected == "Upload Image":
    #title center
    st.markdown("<h1 style='text-align: center; color: White;'>ANPR Yolo5 & EasyOCR</h1>", unsafe_allow_html=True)
    st.title('ANPR Database Operations')
    st.write('Upload images to perform ANPR on them.')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #resize image
        # image = image.resize((720, 720))
        path_image= "./DATASET_PLAT_389/"
        col1, col2,col3 = st.columns(3)
        with col1:
            st.empty()
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            selected_folder = st.selectbox("Pilih Folder", ['K1', 'K2', 'K3', 'K4'])
            name_img = st.text_input("Filename (TANPA EKSTENSI)", value=uploaded_file.name)
            class_img = st.text_input("Kelas", value=selected_folder)
            if st.button("Upload"):
                with st.spinner("Uploading..."):
                    db = connect_db()
                    db.create_record(path_image,selected_folder,name_img,class_img)
                    image.save(os.path.join(path_image, selected_folder, name_img))
                    st.success(f"Image Uploaded successfully to {path_image}/{selected_folder}/{name_img}!")
        with col3:
            st.empty()
    
if selected == "Database":
    #title center
    st.markdown("<h1 style='text-align: center; color: White;'>ANPR Yolo5 & EasyOCR</h1>", unsafe_allow_html=True)
    st.title('ANPR Database Operations')
    st.write("Perform CRUD operations on the database.")

    # Folders to search for .JPG files
    folders = ['K1', 'K2', 'K3', 'K4']

    # Count image files in each folder
    folder_counts = {}
    total_count = 0
    for folder in folders:
        count = count_image_files(folder)
        folder_counts[folder] = count
        total_count += count

    # Create a DataFrame with folder counts
    df = pd.DataFrame(list(folder_counts.items()), columns=['Folder', 'File Count'])

    # Add total count
    df.loc[len(df.index)] = ['Total', total_count]
    
    st.table(df)
    st.subheader("Update and Delete Dataset Images")

    # Show a selectbox to choose the folder
    update_delete_folder = st.selectbox("Choose folder to Update or Delete", ['K1', 'K2', 'K3', 'K4'])

    folder_path_for_update_delete = get_folder_path(update_delete_folder)
    
    # Menyertakan valid_image_extensions untuk mendapatkan daftar file gambar
    valid_image_extensions = ['.jpg', '.jpeg', '.png']
    all_image_files = [file for file in os.listdir(folder_path_for_update_delete) if os.path.isfile(os.path.join(folder_path_for_update_delete, file)) and any(file.lower().endswith(ext) for ext in valid_image_extensions)]

    # Pilihan gambar untuk update atau delete
    page_number = st.number_input("Section", min_value=1, max_value=(len(all_image_files) // 20) + 1, value=1, step=1)
    start_index = (page_number - 1) * 20
    end_index = min(page_number * 20, len(all_image_files))

    # Organize images into rows of five
    rows_of_images = [all_image_files[i:i+5] for i in range(start_index, end_index, 5)]

    # Display each row of images
    for row_images in rows_of_images:
        st.image([os.path.join(folder_path_for_update_delete, image) for image in row_images], width=250, caption=row_images)

    table_data = {"Filename": [file.split(".")[0] for file in all_image_files[start_index:end_index]],
                  "Format": [file.split(".")[-1] for file in all_image_files[start_index:end_index]]}

    # Tampilkan data Filename dan format dalam tabel dengan indeks dimulai dari 1
    st.table(pd.DataFrame(table_data).reset_index(drop=True))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Update Image")
        # Pilih gambar untuk diupdate
        selected_image_for_update = st.selectbox("Pick image to be updated", table_data["Filename"])
        if selected_image_for_update is not None:
            db = connect_db()
            #read record if exist and get ID
            get_id = db.read_record(selected_image_for_update)
                
        # Form untuk mengupdate Filename
        new_file_name = st.text_input("New Filename (Without Extention)", selected_image_for_update)
        # Show a selectbox to choose the folder
        new_class = st.selectbox("Choose folder to Update", ['K1', 'K2', 'K3', 'K4'], index=folders.index(update_delete_folder))
        # Tampilkan tombol update
        update_button = st.button("Update")

        if update_button:
            if get_id is not None:
                db.update_record(get_id[0],new_file_name,new_class)
            # Lakukan pembaruan Filename
            old_file_path = os.path.join(folder_path_for_update_delete, f"{selected_image_for_update}.jpg")
            path_image= "./DATASET_PLAT_389/"
            new_file_path = os.path.join(path_image, new_class, f"{new_file_name}.jpg")

            os.rename(old_file_path, new_file_path)
            st.success(f"Image Name {selected_image_for_update} updated to {new_file_name}.jpg")

    with col2:
        st.subheader("Delete Image")
        # Pilihan gambar untuk update atau delete
        selected_image_for_update_delete = st.selectbox("Pick image to be deleted", table_data["Filename"])

        # Tampilkan tombol delete
        delete_button = st.button("Delete")

        if delete_button:
            # Hapus gambar jika tombol di tekan
            db = connect_db()
            #read record if exist and get ID
            get_id = db.read_record(selected_image_for_update_delete)
            if get_id is not None:
                db.delete_record(get_id[0])
            os.remove(os.path.join(folder_path_for_update_delete, selected_image_for_update_delete + ".jpg"))
            st.success(f"Image {selected_image_for_update_delete} successfully deleted.")
    

if selected == "Training & Testing":
    #title center
    st.markdown("<h1 style='text-align: center; color: White;'>ANPR Yolo5 & EasyOCR</h1>", unsafe_allow_html=True)
    st.title('Training & Testing Yolo5')   
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Click toSetup Environment Locally")
        st.write("To check Cuda If Available")
        if st.button("Setup Environment Local"):
            if st.session_state.trainer == None:
                st.session_state.trainer = Train()
                st.write(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")            
            else:
                st.warning("Please Setup Environment Locally.")
    with col2:
        st.subheader("Click to Start Training On Google Collab")
        if st.button("Open Collab"):
            # Link to your Google Colab notebook
            colab_link = "https://colab.research.google.com/drive/1AQkYvTr0JdS_fsCEO120VUbYHon5bBYR?usp=sharing"
            
            # Open the link in a new tab when the button is clicked
            webbrowser.open_new_tab(colab_link)
            st.success("Training started. Please check the Colab notebook for progress.")


if selected =='Predict':
    #title center
    st.markdown("<h1 style='text-align: center; color: White;'>ANPR Yolo5 & EasyOCR</h1>", unsafe_allow_html=True)   
    st.title('Predict with YOLOv5 & EasyOCR')
    anpr = ANPR()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_in = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            resized_image = None
            annotated_image = None
            st.subheader("Click to Predict with YOLOv5")
            button_pred = st.button("Predict")
        with col2:
            if button_pred:
                with st.spinner("Predicting..."):
                    model_path = "best_V5.pt"  # Replace this with your model's path
                    resized_image, annotated_image,result_anpr = anpr.predict_image(image_in, model_path=model_path)
                st.text_input("Hasil ANPR", value=result_anpr)
                st.success("Prediction complete!")

if selected == 'ANPR':
    #title center
    st.markdown("<h1 style='text-align: center; color: White;'>ANPR Yolo5 & EasyOCR</h1>", unsafe_allow_html=True)   
    st.title('ANPR for Multiple Images')
    st.write('Upload images to perform ANPR on them.')

    model_path = "best_V5.pt"  # Replace this with your model's path
    anpr_folder = ANPR_Folder()

    uploaded_files = st.file_uploader("Choose image(s) to process...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files is not None:
        if st.button("Predict All"):
            all_results = []
            with st.spinner("Predicting..."):
                images_count = len(uploaded_files)
                images_per_row = 4
                rows_count = -(-images_count // images_per_row)  # Ceiling division to get the total number of rows

                for i in range(rows_count):
                    cols = st.columns(images_per_row)
                    for j in range(images_per_row):
                        idx = i * images_per_row + j
                        if idx < images_count:
                            uploaded_file = uploaded_files[idx]
                            image = Image.open(uploaded_file)
                            image = np.array(image)

                            resized_image, result_text, class_label = anpr_folder.predict_image(image, model_path)  # Provide model_path
                            all_results.append({'Image': uploaded_file.name, 'Prediction': result_text})

                            # Display uploaded image
                            with cols[j]:
                                st.image(resized_image, caption=f"Uploaded Image {idx + 1}", use_column_width=True)
            st.success("All images processed successfully!")
            # Save results to a text file
            results_text = ""
            for result in all_results:
                results_text += f"Filename {result['Image']} => Prediction {result['Prediction']}\n"
            dt_now = dt.datetime.now().strftime("%d-%m-%Y_%H-%M")
            st.download_button(label="Download Results", data=results_text, file_name=f"results_{images_count}_{dt_now}.txt", mime="text/plain")
            st.dataframe(all_results)
