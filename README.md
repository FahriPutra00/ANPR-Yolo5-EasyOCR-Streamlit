# ANPR-Yolo5-EasyOCR-Streamlit

This project combines the power of YOLOv5 for license plate detection and EasyOCR for optical character recognition, providing an Automatic Number Plate Recognition (ANPR) system. The integration is showcased through a Streamlit interface for easy usage.

## System Testing Steps

Follow these steps to test the network on your system:

1. **Clone or Download the Program:**
   - Clone or download the program files from the GitHub repository: [https://github.com/FahriPutra00/ANPR-Yolo5-EasyOCR-Streamlit](https://github.com/FahriPutra00/ANPR-Yolo5-EasyOCR-Streamlit)

2. **Install Conda and Run Command Prompt with PyTorch_Cuda.Txt:**
   - Perform the installation of Conda on your system.
   - Open Command Prompt and execute the commands provided in the PyTorch_Cuda.Txt file to set up the PyTorch environment with CUDA support.
     ```console
      conda create -n ANPR-Py python=3.10 
      conda activate ANPR-Py
      conda install -c conda-forge cudatoolkit=11.8 cudnn=8.8.0
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      pip3 install streamlit
      pip3 install streamlit_option_menu
      pip3 install easyocr
     ```

3. **Import image_dataset.sql to DBMS:**
   - Import the provided `image_dataset.sql` file into your preferred Database Management System (DBMS) such as MySQL or PostgreSQL. This file contains the necessary dataset for testing.

4. **Run the Program using Virtual Environment:**
   - Execute the program within the virtual environment created earlier.
     ```console
      streamlit run main.py
     ```


## About YOLOv5 and EasyOCR

### YOLOv5
YOLOv5 is a state-of-the-art real-time object detection system that stands for "You Only Look Once." It is widely used for its speed and accuracy in detecting objects within images or video frames. In this project, YOLOv5 is employed for license plate detection.

Learn more about YOLOv5: [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)

### EasyOCR
EasyOCR is a comprehensive Optical Character Recognition (OCR) tool that simplifies text extraction from images. It supports multiple languages and provides an easy-to-use interface. In this project, EasyOCR is utilized for recognizing the characters on the detected license plates.

Learn more about EasyOCR: [EasyOCR GitHub Repository](https://github.com/JaidedAI/EasyOCR)

---

Feel free to reach out for any issues or inquiries related to the project. Happy testing!
