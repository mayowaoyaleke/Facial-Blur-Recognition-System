# Face Blurriness Detection System

A Python tool for automatically detecting and analyzing face quality in images. Quickly identify clear vs. blurry faces and organize photo collections by quality.

## 🌟 Features

- **Automatic Face Detection**: Advanced face recognition with quality analysis
- **Blur Quality Scoring**: Uses Laplacian variance and Sobel gradients
- **Visual Analysis**: Images with bounding boxes and quality overlays
- **Batch Processing**: Process entire folders automatically
- **File Organization**: Sort images into quality-based folders
- **Detailed Reports**: Comprehensive analysis summaries
- **Multiple Formats**: JPG, PNG, BMP, TIFF support

## 🎯 Use Cases

- Photography quality control and portfolio curation
- Social media content optimization
- Dataset cleaning for machine learning
- Family photo organization
- Professional photo service automation

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

**Note**: On some systems, you may need to install additional dependencies:
```bash
# Windows/macOS
pip install cmake dlib

# Ubuntu/Linux
sudo apt-get install cmake python3-dev
pip install dlib
```

### 2. Basic Usage

1. Update folder path in the script:
```python
folder_path = r"C:\path\to\your\images"
```

2. Run the detection:
```bash
python face_blur_detector.py
```

### 3. Advanced Usage

```python
from face_blur_detector import FaceBlurrinessDetector

detector = FaceBlurrinessDetector(blur_threshold=100)

# Process folder with options
summary = detector.process_folder(
    folder_path="path/to/images",
    display_results=True,      # Show visual analysis
    save_summary=True,         # Generate report
    organize_by_quality=False  # Create quality folders
)
```

## ⚙️ Configuration

### Blur Threshold Settings

```python
detector = FaceBlurrinessDetector(blur_threshold=100)
```

**Quality Guidelines:**
- `< 30`: Very Blurry (red)
- `30-100`: Blurry (orange)
- `100-200`: Fair (yellow) 
- `> 200`: Sharp (green)

**Recommended for different uses:**
- Professional Photography: 150-200
- Social Media: 80-120
- General Use: 50-100
- Dataset Cleaning: 30-80

## 📊 Output

### Console Display
- Real-time processing progress
- Face count and quality per image
- Summary table with recommendations
- Overall statistics

### File Outputs
- **Analysis Report**: `face_quality_analysis.txt` with detailed results
- **Quality Folders** (optional): `clear_faces/`, `blurry_faces/`, `no_faces/`

### Visual Display
Images shown with colored bounding boxes and quality scores overlaid.

## 🔧 Processing Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `display_results` | Show matplotlib visualizations | `True` |
| `save_summary` | Create detailed text report | `True` |
| `organize_by_quality` | Copy files into quality folders | `False` |

## ⚠️ Troubleshooting

**Common Issues:**

1. **dlib installation fails**: Install Visual Studio Build Tools (Windows) or Xcode (macOS)
2. **Slow processing**: Set `display_results=False`, use smaller images
3. **Memory issues**: Process in smaller batches, close matplotlib windows
4. **False blur detections**: Adjust `blur_threshold` parameter

**Performance Tips:**
- ~30-60 seconds for 10 images (1920x1080)
- Use `display_results=False` for faster batch processing
- Resize very large images (>2000px) for better performance

## 🔬 Technical Details

**Blur Detection Algorithms:**
1. **Laplacian Variance**: Measures edge sharpness (primary method)
2. **Sobel Gradients**: Alternative edge detection for validation

**Face Detection**: Uses face_recognition library with HOG algorithm

## 🎛️ Customization

Modify quality thresholds or add custom metrics by editing the `FaceBlurrinessDetector` class methods.

## 📝 Requirements

- Python 3.7+
- 4GB+ RAM recommended
- See `requirements.txt` for full dependency list

---

**Quick Setup**: `pip install -r requirements.txt` → Update folder path → `python face_blur_detector.py`
