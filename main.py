from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import warnings
import logging
from datetime import datetime
from collections import Counter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import torch

# Configure logging and warnings
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')

# Increase maximum file size limit to 1GB
MAX_VIDEO_SIZE = 1024  # Size in MB

# Set page config
st.set_page_config(
    page_title="Agri Bot with Image Surveillance",
    page_icon="🌿",
    layout="centered",
    # Increase server size limit to 1GB
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Agri Bot with Image Surveillance by Team Electronaunts'
    }
)

# Add custom CSS
st.markdown("""
<style>
/* Container width control */
.main > div {
    max-width: 1000px;
    padding: 1rem;
    margin: 0 auto;
}

/* Image container */
.image-container {
    margin: 1rem 0;
    padding: 1rem;
    border: 1px solid rgba(0, 210, 255, 0.3);
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.1);
}

/* Summary card */
.summary-card {
    background: rgba(0, 210, 255, 0.1);
    border: 1px solid rgba(0, 210, 255, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Modern button styling */
.stButton > button {
    background: linear-gradient(135deg, #00d2ff 0%, #3a47d5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(0, 210, 255, 0.25) !important;
    transition: all 0.3s ease !important;
    width: auto !important;
    margin: 0 auto !important;
    display: block !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0, 210, 255, 0.4) !important;
    background: linear-gradient(135deg, #00c4f0 0%, #3040c0 100%) !important;
}

/* Progress bar styling */
.stProgress > div > div {
    background-color: #00d2ff !important;
}

/* Center align headers */
h1, h2, h3, h4 {
    text-align: center;
}

/* Result text styling */
.result-text {
    font-size: 1em;
    margin: 0.5rem 0;
    color: white;
    text-align: center;
}

.confidence-high {
    color: #00ff00;
}

.confidence-medium {
    color: #ffff00;
}

.confidence-low {
    color: #ff0000;
}

/* Divider styling */
.divider {
    border-bottom: 1px solid rgba(0, 210, 255, 0.3);
    margin: 1.5rem 0;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    max-width: 600px;
    margin: 0 auto;
}

[data-testid="stFileUploader"] > section {
    border: 2px dashed rgba(0, 210, 255, 0.4) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background: rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"] > section:hover {
    border-color: #00d2ff !important;
    background: rgba(0, 210, 255, 0.05) !important;
}

/* Add styles for file size warning */
.file-size-warning {
    color: #ff9800;
    font-size: 0.8em;
    margin-top: 5px;
    text-align: center;
}

/* Add styles for file size error */
.file-size-error {
    color: #f44336;
    font-size: 0.8em;
    margin-top: 5px;
    text-align: center;
}

/* Video info card styling */
.video-info-card {
    background: rgba(0, 210, 255, 0.1);
    border: 1px solid rgba(0, 210, 255, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
}

.video-info-header {
    color: #00d2ff;
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 15px;
    text-align: center;
}

.video-info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

.video-info-item {
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    text-align: center;
}

.info-label {
    color: #00d2ff;
    font-size: 0.9em;
    margin-bottom: 5px;
}

.info-value {
    color: white;
    font-size: 1.1em;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

def analyze_single_image(image_data):
    """Analyze a single image with YOLO model"""
    try:
        # Convert image data to numpy array
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, Image.Image):
            image = np.array(image_data)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif isinstance(image_data, np.ndarray):
            image = image_data
        else:
            raise ValueError("Unsupported image format")

        # Load the model
        try:
            model = YOLO("cotton-v1.pt")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, []

        # Run inference
        try:
            results = model(image, verbose=False)
        except Exception as e:
            st.error(f"Error running inference: {str(e)}")
            return None, []

        # Process results
        for result in results:
            img = result.plot(labels=True)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with st.container():
                # Center the image using columns
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(img_rgb, caption="Detection Result", use_column_width=True)

                if len(result.boxes) > 0:
                    detections = []
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        detections.append((class_name, conf))
                    
                    detections.sort(key=lambda x: x[1], reverse=True)
                    best_detection = detections[0]
                    unique_classes = len(set(d[0] for d in detections))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Detected Diseases (Total: {unique_classes})",
                            value=best_detection[0],
                            delta=f"{len(detections)} Detection(s)"
                        )
                    with col2:
                        st.metric(
                            label="Highest Confidence Score",
                            value=f"{best_detection[1]:.2%}",
                            delta="Confidence Level"
                        )
                        st.progress(best_detection[1])
                    
                    if len(detections) > 1:
                        with st.expander("View All Detections"):
                            for class_name, conf in detections:
                                st.text(f"{class_name}: {conf:.2%}")
                    
                    # Create single image report
                    single_result = [{
                        'image_name': "Analyzed Image",
                        'image': img_rgb,
                        'detections': detections
                    }]
                    
                    # Generate and offer PDF download
                    pdf_buffer = create_pdf_report(single_result, detections, 1)
                    st.download_button(
                        label="Download Analysis Report",
                        data=pdf_buffer,
                        file_name=f"agri_bot_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
                    st.markdown(
                        f"<p style='text-align: center; color: gray; font-size: 0.8em;'>"
                        f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        f"</p>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No diseases detected in the image")
                    # Create report even for no detections
                    single_result = [{
                        'image_name': "Analyzed Image",
                        'image': img_rgb,
                        'detections': []
                    }]
                    pdf_buffer = create_pdf_report(single_result, [], 1)
                    st.download_button(
                        label="Download Analysis Report",
                        data=pdf_buffer,
                        file_name=f"agri_bot_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

            return img_rgb, detections if len(result.boxes) > 0 else []

    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None, []

def create_pdf_report(results, all_detections, uploaded_files_count):
    """Create a PDF report with analysis results"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#00008B'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#000080'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#000080'),
        spaceBefore=20,
        spaceAfter=10
    )

    # Add title and team info
    story.append(Paragraph("Agri Bot with Image Surveillance", title_style))
    story.append(Paragraph("Analysis Report", subtitle_style))
    story.append(Paragraph(f"Generated by Team Electronaunts", subtitle_style))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Add summary statistics
    story.append(Paragraph("Summary Statistics", header_style))
    
    # Create summary table
    disease_counter = Counter(d[0] for d in all_detections)
    total_detections = len(all_detections)
    
    summary_data = [
        ["Total Images Analyzed", str(uploaded_files_count)],
        ["Total Detections", str(total_detections)],
        ["Unique Diseases Found", str(len(disease_counter))]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # Add disease distribution
    if disease_counter:
        story.append(Paragraph("Disease Distribution", header_style))
        dist_data = [["Disease", "Occurrences", "Percentage"]]
        for disease, count in disease_counter.most_common():
            percentage = (count / total_detections) * 100
            dist_data.append([disease, str(count), f"{percentage:.1f}%"])
        
        dist_table = Table(dist_data, colWidths=[200, 100, 100])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER')
        ]))
        story.append(dist_table)
        story.append(Spacer(1, 20))

    # Add individual image results
    story.append(Paragraph("Detailed Analysis Results", header_style))
    
    for idx, result in enumerate(results, 1):
        # Image title
        story.append(Paragraph(f"Image {idx}: {result['image_name']}", styles['Heading3']))
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(result['image'])
        
        # Resize image to fit in PDF
        img_width = 400
        aspect = img.height / img.width
        img_height = int(img_width * aspect)
        img = img.resize((img_width, img_height))
        
        # Save image to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Add image to PDF
        img = RLImage(img_buffer, width=6*inch, height=6*inch*aspect)
        story.append(img)
        
        # Add detection results
        if result['detections']:
            det_data = [["Disease", "Confidence"]]
            for disease, conf in result['detections']:
                det_data.append([disease, f"{conf:.2%}"])
            
            det_table = Table(det_data, colWidths=[200, 200])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(det_table)
        else:
            story.append(Paragraph("No diseases detected in this image", styles['Normal']))
        
        story.append(Spacer(1, 20))

    # Add footer
    story.append(Spacer(1, 20))
    footer_text = """
    <para alignment="center">
    <font size="8">Generated by Agri Bot with Image Surveillance<br/>
    Team Electronaunts</font>
    </para>
    """
    story.append(Paragraph(footer_text, styles['Normal']))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def analyze_bulk_images(uploaded_files):
    """Analyze multiple images and display results in a clean, centered format"""
    if not uploaded_files:
        return

    # Load model once for all images
    try:
        model = YOLO("cotton-v1.pt")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Initialize containers for consolidated results
    all_detections = []
    all_results = []

    # Process each image
    with st.spinner("Processing images..."):
        for uploaded_file in uploaded_files:
            try:
                # Read and process image
                file_bytes = uploaded_file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Run inference
                results = model(image, verbose=False)
                
                # Process results
                for result in results:
                    img = result.plot(labels=True)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get detections
                    detections = []
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = model.names[cls]
                            detections.append((class_name, conf))
                        detections.sort(key=lambda x: x[1], reverse=True)
                        all_detections.extend(detections)

                    # Store results
                    all_results.append({
                        'image_name': uploaded_file.name,
                        'image': img_rgb,
                        'detections': detections
                    })

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    # Display consolidated results only if there are detections
    if all_detections:
        st.markdown("### Analysis Summary")
        
        # Count disease occurrences
        disease_counter = Counter(d[0] for d in all_detections)
        total_detections = len(all_detections)
        
        # Display summary metrics in a clean table format
        summary_data = [
            ["Total Images Analyzed", len(uploaded_files)],
            ["Total Detections", total_detections],
            ["Unique Diseases Found", len(disease_counter)]
        ]
        
        # Create a clean table for summary
        st.table(summary_data)

        # Display disease distribution in a clean table format
        st.markdown("### Disease Distribution")
        
        distribution_data = []
        for disease, count in disease_counter.most_common():
            percentage = (count / total_detections) * 100
            distribution_data.append([disease, count, f"{percentage:.1f}%"])
            
        st.table(distribution_data)

        # Generate and offer PDF download
        pdf_buffer = create_pdf_report(all_results, all_detections, len(uploaded_files))
        st.download_button(
            label="Download Analysis Report",
            data=pdf_buffer,
            file_name=f"agri_bot_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

    # Display individual results
    for idx, result in enumerate(all_results, 1):
        st.markdown(f"### Image {idx}: {result['image_name']}")
        
        # Display image
        st.image(result['image'], caption="Analysis Result", width=600)
        
        # Display detections in a clean table format
        if result['detections']:
            detection_data = []
            for disease, conf in result['detections']:
                detection_data.append([disease, f"{conf:.2%}"])
            st.table(detection_data)
        else:
            st.warning("No diseases detected in this image")
        
        # Add divider between images, except for the last one
        if idx < len(all_results):
            st.markdown("---")

def check_video_size(video_file):
    """Check if video file size is within limit"""
    if video_file is None:
        return True
    
    file_size_mb = video_file.size / (1024 * 1024)  # Convert to MB
    if file_size_mb > MAX_VIDEO_SIZE:
        st.error(f"File size ({file_size_mb:.1f} MB) exceeds the limit of {MAX_VIDEO_SIZE} MB")
        return False
    elif file_size_mb > MAX_VIDEO_SIZE * 0.8:  # Warning at 80% of limit
        st.warning(f"File size ({file_size_mb:.1f} MB) is approaching the limit of {MAX_VIDEO_SIZE} MB")
    return True

def format_video_info(video_info):
    """Format video information in a visually appealing way"""
    st.markdown("""
        <div class="video-info-card">
            <div class="video-info-header">Video Information</div>
            <div class="video-info-grid">
                <div class="video-info-item">
                    <div class="info-label">Duration</div>
                    <div class="info-value">{duration}</div>
                </div>
                <div class="video-info-item">
                    <div class="info-label">Frame Rate</div>
                    <div class="info-value">{fps}</div>
                </div>
                <div class="video-info-item">
                    <div class="info-label">Resolution</div>
                    <div class="info-value">{resolution}</div>
                </div>
                <div class="video-info-item">
                    <div class="info-label">Total Frames</div>
                    <div class="info-value">{frames:,}</div>
                </div>
                <div class="video-info-item">
                    <div class="info-label">Analysis Interval</div>
                    <div class="info-value">{interval}</div>
                </div>
                <div class="video-info-item">
                    <div class="info-label">Key Frames</div>
                    <div class="info-value">{keyframes:,}</div>
                </div>
            </div>
        </div>
    """.format(
        duration=f"{float(video_info['Duration'].split()[0]):,.1f} seconds",
        fps=video_info['Frame Rate'],
        resolution=video_info['Resolution'],
        frames=int(video_info['Total Frames']),
        interval=video_info['Analysis Interval'],
        keyframes=int(video_info['Total Key Frames'])
    ), unsafe_allow_html=True)

def analyze_video(video_file):
    """Analyze video frames for cotton diseases at 4-second intervals with optimized processing"""
    try:
        # Check file size
        if not check_video_size(video_file):
            return

        # Read video file
        video_bytes = video_file.read()
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(video_bytes)

        # Open video
        cap = cv2.VideoCapture(temp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate frame interval (4 seconds)
        frame_interval = 4 * fps  # Number of frames to skip (4 seconds worth of frames)
        total_keyframes = total_frames // frame_interval + 1

        # Display video information
        video_info = {
            "Duration": f"{duration:.1f} seconds",
            "Frame Rate": f"{fps} fps",
            "Total Frames": total_frames,
            "Analysis Interval": "4 seconds",
            "Total Key Frames": total_keyframes,
            "Resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        }
        format_video_info(video_info)

        # Load model
        try:
            model = YOLO("cotton-v1.pt")
            # Enable GPU acceleration if available
            model.to('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

        # Initialize results storage
        all_detections = []
        all_results = []
        frame_count = 0
        keyframe_count = 0

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Batch processing setup
        batch_size = 4  # Process 4 frames at once
        batch_frames = []
        batch_timestamps = []

        # Process video frames at 4-second intervals
        while cap.isOpened():
            # Skip to next key frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Add frame to batch
                batch_frames.append(frame)
                batch_timestamps.append(frame_count / fps)
                keyframe_count += 1

                # Process batch when it's full or at the end
                if len(batch_frames) == batch_size or frame_count + frame_interval >= total_frames:
                    # Update progress
                    progress = keyframe_count / total_keyframes
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frames {keyframe_count-len(batch_frames)+1} to {keyframe_count} of {total_keyframes}")

                    # Run inference on batch
                    try:
                        results = model(batch_frames, verbose=False)
                        
                        # Process results
                        for idx, result in enumerate(results):
                            img = result.plot(labels=True)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # Get detections
                            detections = []
                            if len(result.boxes) > 0:
                                for box in result.boxes:
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    class_name = model.names[cls]
                                    detections.append((class_name, conf))
                                detections.sort(key=lambda x: x[1], reverse=True)
                                all_detections.extend(detections)

                            # Store results with timestamp
                            all_results.append({
                                'image_name': f"Frame at {batch_timestamps[idx]:.1f}s",
                                'image': img_rgb,
                                'detections': detections,
                                'timestamp': batch_timestamps[idx]
                            })

                    except Exception as e:
                        st.error(f"Error processing batch: {str(e)}")

                    # Clear batch
                    batch_frames = []
                    batch_timestamps = []

            # Skip to next interval
            frame_count += frame_interval

        # Clean up
        cap.release()
        import os
        os.remove(temp_file_path)

        # Remove progress indicators
        progress_bar.empty()
        status_text.empty()

        # Display results
        if all_detections:
            st.markdown("### Video Analysis Summary")
            
            # Count disease occurrences
            disease_counter = Counter(d[0] for d in all_detections)
            total_detections = len(all_detections)
            
            # Display summary metrics
            summary_data = [
                ["Total Key Frames Analyzed", keyframe_count],
                ["Analysis Interval", "4 seconds"],
                ["Total Detections", total_detections],
                ["Unique Diseases Found", len(disease_counter)]
            ]
            st.table(summary_data)

            # Display disease distribution
            st.markdown("### Disease Distribution")
            distribution_data = []
            for disease, count in disease_counter.most_common():
                percentage = (count / total_detections) * 100
                distribution_data.append([disease, count, f"{percentage:.1f}%"])
            st.table(distribution_data)

            # Generate and offer PDF download
            pdf_buffer = create_pdf_report(all_results, all_detections, keyframe_count)
            st.download_button(
                label="Download Video Analysis Report",
                data=pdf_buffer,
                file_name=f"agri_bot_video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

            # Display key frames with detections
            st.markdown("### Key Frames with Detections")
            for idx, result in enumerate(all_results, 1):
                if result['detections']:  # Only show frames with detections
                    st.markdown(f"### Frame at {result['timestamp']:.1f} seconds")
                    st.image(result['image'], caption=f"Detection Result", width=600)
                    
                    # Display detections in table format
                    detection_data = []
                    for disease, conf in result['detections']:
                        detection_data.append([disease, f"{conf:.2%}"])
                    st.table(detection_data)
                    
                    if idx < len(all_results):
                        st.markdown("---")

        else:
            st.warning("No diseases detected in the video")

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

def main():
    st.title("Agri Bot with Image Surveillance")
    st.markdown("### Team Electronaunts")
    st.markdown("### Select Analysis Mode")

    # Create tabs for single image, bulk analysis, and video analysis
    tab1, tab2, tab3 = st.tabs(["Single Image Analysis", "Bulk Analysis", "Video Analysis"])

    with tab1:
        # Center the upload section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                help="Upload a single cotton plant image for disease detection",
                key="single_upload"
            )

            if uploaded_file:
                try:
                    file_bytes = uploaded_file.read()
                    image = Image.open(io.BytesIO(file_bytes))
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("Analyze Image", key="single_analyze"):
                        with st.spinner("Analyzing image..."):
                            analyze_single_image(file_bytes)

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

    with tab2:
        # Center the upload section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_files = st.file_uploader(
                "Choose multiple images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Upload multiple cotton plant images for bulk analysis",
                key="bulk_upload"
            )

            if uploaded_files:
                if st.button("Analyze All Images", key="bulk_analyze"):
                    analyze_bulk_images(uploaded_files)

    with tab3:
        # Center the upload section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"Maximum video file size: {MAX_VIDEO_SIZE} MB")
            uploaded_video = st.file_uploader(
                "Choose a video",
                type=["mp4", "avi", "mov"],
                help=f"Upload a video of cotton plants (max {MAX_VIDEO_SIZE} MB)",
                key="video_upload"
            )

            if uploaded_video:
                # Display video file details
                file_size_mb = uploaded_video.size / (1024 * 1024)
                st.markdown("""
                    <div style="background: rgba(0, 210, 255, 0.1); border: 1px solid rgba(0, 210, 255, 0.3); 
                              border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <div style="display: flex; align-items: center; margin: 5px 0; padding: 8px; 
                                  background: rgba(0, 0, 0, 0.2); border-radius: 6px;">
                            <span style="color: #00d2ff; font-size: 0.9em; min-width: 100px; margin-right: 10px;">
                                Filename:
                            </span>
                            <span style="color: white; font-size: 1em;">
                                {filename}
                            </span>
                        </div>
                        <div style="display: flex; align-items: center; margin: 5px 0; padding: 8px; 
                                  background: rgba(0, 0, 0, 0.2); border-radius: 6px;">
                            <span style="color: #00d2ff; font-size: 0.9em; min-width: 100px; margin-right: 10px;">
                                File size:
                            </span>
                            <span style="color: white; font-size: 1em;">
                                {filesize:.2f} MB
                            </span>
                        </div>
                    </div>
                """.format(
                    filename=uploaded_video.name,
                    filesize=file_size_mb
                ), unsafe_allow_html=True)

                if check_video_size(uploaded_video):
                    if st.button("Analyze Video", key="video_analyze"):
                        with st.spinner("Processing video..."):
                            analyze_video(uploaded_video)

if __name__ == "__main__":
    main()