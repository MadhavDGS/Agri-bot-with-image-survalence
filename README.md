# Agri Bot with Image Surveillance

An advanced agricultural monitoring system that uses computer vision and deep learning to detect diseases in cotton plants. Developed by Team Electronaunts.

## Features

- **Single Image Analysis**: Upload and analyze individual cotton plant images
- **Bulk Image Analysis**: Process multiple images simultaneously
- **Video Analysis**: Analyze video footage of cotton plants
- **Detailed Reports**: Generate comprehensive PDF reports with analysis results
- **Real-time Detection**: Uses YOLO model for accurate disease detection
- **GPU Acceleration**: Supports GPU acceleration for faster processing

## Technical Specifications

- **Maximum Video File Size**: 1024 MB
- **Supported Image Formats**: JPG, JPEG, PNG
- **Supported Video Formats**: MP4, AVI, MOV, MPEG4
- **Video Analysis Interval**: Every 4 seconds
- **Batch Processing**: Processes 4 frames simultaneously

## Requirements

- Python 3.x
- PyTorch
- Ultralytics YOLO
- OpenCV
- Streamlit
- ReportLab
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MadhavDGS/Agri-bot-with-image-survalence.git
cd Agri-bot-with-image-survalence
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. **Single Image Analysis**
   - Upload a single image
   - Click "Analyze Image"
   - View results and download report

2. **Bulk Analysis**
   - Upload multiple images
   - Click "Analyze All Images"
   - View consolidated results and download report

3. **Video Analysis**
   - Upload a video file (up to 1GB)
   - Click "Analyze Video"
   - View frame-by-frame analysis and download report

## Team Electronaunts

Developed with ❤️ by Team Electronaunts 