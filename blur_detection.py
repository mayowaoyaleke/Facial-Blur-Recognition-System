import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import logging
import glob
import pandas as pd
from tabulate import tabulate
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceBlurrinessDetector:
    def __init__(self, blur_threshold=100):
        """
        Initialize the face blurriness detection system
        
        Args:
            blur_threshold: Threshold for determining if face is blurry (lower = more blurry)
                          Typical values: <50=very blurry, 50-100=blurry, >100=sharp
        """
        logger.info("Face Blurriness Detection System initialized")
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.blur_threshold = blur_threshold
    
    def calculate_face_sharpness(self, face_image):
        """
        Calculate sharpness/blurriness of a face using Laplacian variance
        
        Args:
            face_image: PIL Image or numpy array of the face
            
        Returns:
            float: Sharpness score (higher = sharper, lower = more blurry)
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(face_image, Image.Image):
                face_array = np.array(face_image)
            else:
                face_array = face_image
            
            # Convert to grayscale if needed
            if len(face_array.shape) == 3:
                gray_face = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_array
            
            # Calculate Laplacian variance (standard method for blur detection)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            return round(laplacian_var, 2)
            
        except Exception as e:
            logger.error(f"Error calculating face sharpness: {e}")
            return 0.0
    
    def calculate_alternative_sharpness(self, face_image):
        """
        Alternative sharpness calculation using Sobel gradients
        
        Args:
            face_image: PIL Image or numpy array of the face
            
        Returns:
            float: Alternative sharpness score
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(face_image, Image.Image):
                face_array = np.array(face_image)
            else:
                face_array = face_image
            
            # Convert to grayscale if needed
            if len(face_array.shape) == 3:
                gray_face = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_array
            
            # Calculate Sobel gradients
            sobel_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate magnitude
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Return mean of magnitude
            return round(np.mean(sobel_magnitude), 2)
            
        except Exception as e:
            logger.error(f"Error calculating alternative sharpness: {e}")
            return 0.0
    
    def convert_image_format(self, image_path):
        """Convert image to RGB format"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Error converting image format: {e}")
            return None
    
    def analyze_face_quality(self, image_path):
        """
        Analyze faces in image for blurriness/quality
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with face quality analysis
        """
        try:
            # Convert image to proper format
            pil_image = self.convert_image_format(image_path)
            if pil_image is None:
                return {
                    'image_path': image_path,
                    'success': False,
                    'error': 'Failed to convert image format',
                    'faces': [],
                    'face_count': 0,
                    'has_clear_faces': False,
                    'has_blurry_faces': False
                }
            
            # Convert to numpy for face_recognition
            image_array = np.array(pil_image)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image_array, model="hog")
            
            # Analyze each face for blurriness
            face_analyses = []
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Extract face region
                face_crop = image_array[top:bottom, left:right]
                face_pil = Image.fromarray(face_crop)
                
                # Calculate sharpness scores
                laplacian_score = self.calculate_face_sharpness(face_crop)
                sobel_score = self.calculate_alternative_sharpness(face_crop)
                
                # Determine if face is blurry
                is_blurry = laplacian_score < self.blur_threshold
                
                # Calculate face size (helps with quality assessment)
                face_width = right - left
                face_height = bottom - top
                face_area = face_width * face_height
                
                # Quality assessment
                if laplacian_score < 30:
                    quality = "Very Blurry"
                    quality_color = "red"
                elif laplacian_score < self.blur_threshold:
                    quality = "Blurry"
                    quality_color = "orange"
                elif laplacian_score < 200:
                    quality = "Fair"
                    quality_color = "yellow"
                else:
                    quality = "Sharp"
                    quality_color = "green"
                
                face_analysis = {
                    'face_number': i + 1,
                    'location': (top, right, bottom, left),
                    'face_crop': face_pil,
                    'laplacian_score': laplacian_score,
                    'sobel_score': sobel_score,
                    'is_blurry': is_blurry,
                    'quality': quality,
                    'quality_color': quality_color,
                    'face_size': (face_width, face_height),
                    'face_area': face_area
                }
                
                face_analyses.append(face_analysis)
            
            # Overall assessment
            has_clear_faces = any(not face['is_blurry'] for face in face_analyses)
            has_blurry_faces = any(face['is_blurry'] for face in face_analyses)
            
            return {
                'image_path': image_path,
                'success': True,
                'pil_image': pil_image,
                'faces': face_analyses,
                'face_count': len(face_analyses),
                'has_clear_faces': has_clear_faces,
                'has_blurry_faces': has_blurry_faces,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return {
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'faces': [],
                'face_count': 0,
                'has_clear_faces': False,
                'has_blurry_faces': False
            }
    
    def get_image_files(self, folder_path):
        """Get all supported image files from a folder"""
        image_files = []
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return image_files
        
        for ext in self.supported_extensions:
            pattern = os.path.join(folder_path, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(folder_path, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        return sorted(list(set(image_files)))
    
    def process_folder(self, folder_path, display_results=True, save_summary=True, organize_by_quality=False):
        """
        Process all images in folder for face blur analysis
        
        Args:
            folder_path: Path to folder containing images
            display_results: Whether to display images with analysis
            save_summary: Whether to save summary file
            organize_by_quality: Whether to organize files into quality folders
            
        Returns:
            Dictionary with processing results
        """
        print(f"🔍 Analyzing face quality in folder: {folder_path}")
        print(f"📊 Blur threshold: {self.blur_threshold} (lower = more blurry)")
        
        image_files = self.get_image_files(folder_path)
        if not image_files:
            print("❌ No supported image files found!")
            return {}
        
        print(f"📸 Found {len(image_files)} image files to analyze")
        
        # Process each image
        results = []
        clear_faces_count = 0
        blurry_faces_count = 0
        no_faces_count = 0
        error_count = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\nAnalyzing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self.analyze_face_quality(image_path)
            results.append(result)
            
            if result['success']:
                if result['face_count'] > 0:
                    clear_count = sum(1 for face in result['faces'] if not face['is_blurry'])
                    blurry_count = sum(1 for face in result['faces'] if face['is_blurry'])
                    
                    print(f"  👥 Found {result['face_count']} face(s): {clear_count} clear, {blurry_count} blurry")
                    
                    for face in result['faces']:
                        print(f"    Face {face['face_number']}: {face['quality']} (Score: {face['laplacian_score']})")
                    
                    if result['has_clear_faces']:
                        clear_faces_count += 1
                    if result['has_blurry_faces']:
                        blurry_faces_count += 1
                        
                    if display_results:
                        self.display_analysis_result(result)
                else:
                    print("  ⚠️  No faces detected")
                    no_faces_count += 1
            else:
                print(f"  ❌ Error: {result['error']}")
                error_count += 1
        
        # Create summary
        summary = {
            'total_images': len(results),
            'images_with_clear_faces': clear_faces_count,
            'images_with_blurry_faces': blurry_faces_count,
            'images_no_faces': no_faces_count,
            'images_with_errors': error_count,
            'results': results
        }
        
        # Display results
        self.display_results_table(results)
        self.print_summary(summary)
        
        # Save summary if requested
        if save_summary:
            self.save_summary_file(folder_path, summary)
        
        # Organize files if requested
        if organize_by_quality:
            self.organize_files_by_quality(folder_path, results)
        
        return summary
    
    def display_analysis_result(self, result):
        """Display image with face quality analysis"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Display image
            ax.imshow(result['pil_image'])
            
            filename = os.path.basename(result['image_path'])
            clear_count = sum(1 for face in result['faces'] if not face['is_blurry'])
            blurry_count = sum(1 for face in result['faces'] if face['is_blurry'])
            
            title = f'{filename}\n👥 {result["face_count"]} face(s): {clear_count} Clear, {blurry_count} Blurry'
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Draw bounding boxes with quality info
            for face in result['faces']:
                top, right, bottom, left = face['location']
                
                # Color based on quality
                color_map = {
                    'red': 'red',
                    'orange': 'orange', 
                    'yellow': 'yellow',
                    'green': 'green'
                }
                color = color_map.get(face['quality_color'], 'blue')
                
                # Draw bounding box
                bbox = patches.Rectangle(
                    (left, top), right - left, bottom - top,
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                ax.add_patch(bbox)
                
                # Add quality label
                label_text = f"Face {face['face_number']}: {face['quality']}\nScore: {face['laplacian_score']}"
                ax.text(left, top - 20, label_text,
                       bbox=dict(facecolor=color, alpha=0.8),
                       fontsize=9, color='white' if color in ['red', 'green'] else 'black',
                       fontweight='bold')
            
            ax.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error displaying result: {e}")
    
    def display_results_table(self, results):
        """Display results in formatted table"""
        print(f"\n{'='*100}")
        print("FACE QUALITY ANALYSIS RESULTS")
        print(f"{'='*100}")
        
        table_data = []
        for result in results:
            filename = os.path.basename(result['image_path'])
            
            if result['success']:
                if result['face_count'] > 0:
                    clear_faces = sum(1 for face in result['faces'] if not face['is_blurry'])
                    blurry_faces = sum(1 for face in result['faces'] if face['is_blurry'])
                    
                    # Get quality breakdown
                    qualities = [face['quality'] for face in result['faces']]
                    quality_summary = ', '.join(qualities) if len(qualities) <= 3 else f"{', '.join(qualities[:2])}, +{len(qualities)-2} more"
                    
                    # Get score range
                    scores = [face['laplacian_score'] for face in result['faces']]
                    min_score = min(scores)
                    max_score = max(scores)
                    score_range = f"{min_score}-{max_score}" if min_score != max_score else str(min_score)
                    
                    status = f"✅ {result['face_count']} face(s)"
                    recommendation = "Keep" if clear_faces > 0 else "⚠️ Review/Discard"
                    
                    table_data.append([
                        filename[:30] + "..." if len(filename) > 30 else filename,
                        status,
                        clear_faces,
                        blurry_faces,
                        quality_summary[:35] + "..." if len(quality_summary) > 35 else quality_summary,
                        score_range,
                        recommendation
                    ])
                else:
                    table_data.append([
                        filename[:30] + "..." if len(filename) > 30 else filename,
                        "⚠️ No faces",
                        0,
                        0,
                        "No faces detected",
                        "N/A",
                        "No faces"
                    ])
            else:
                table_data.append([
                    filename[:30] + "..." if len(filename) > 30 else filename,
                    "❌ Error",
                    0,
                    0,
                    result['error'][:35] + "..." if len(result['error']) > 35 else result['error'],
                    "N/A",
                    "Fix error"
                ])
        
        headers = ["Filename", "Status", "Clear Faces", "Blurry Faces", "Quality Details", "Score Range", "Recommendation"]
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
    
    def print_summary(self, summary):
        """Print processing summary"""
        print(f"\n{'='*60}")
        print("FACE QUALITY SUMMARY")
        print(f"{'='*60}")
        
        total = summary['total_images']
        clear = summary['images_with_clear_faces']
        blurry = summary['images_with_blurry_faces']
        no_faces = summary['images_no_faces']
        errors = summary['images_with_errors']
        
        print(f"Total images processed: {total}")
        print(f"✅ Images with clear faces: {clear}")
        print(f"⚠️  Images with blurry faces: {blurry}")
        print(f"👤 Images with no faces: {no_faces}")
        print(f"❌ Images with errors: {errors}")
        
        if total > 0:
            print(f"\n📊 Quality Distribution:")
            print(f"  Clear face rate: {(clear/total)*100:.1f}%")
            print(f"  Blurry face rate: {(blurry/total)*100:.1f}%")
            
        print(f"\n💡 Recommendations:")
        print(f"  - Keep images with clear faces: {clear} files")
        print(f"  - Review/discard blurry images: {blurry} files")
        print(f"  - Check images with no faces: {no_faces} files")
    
    def organize_files_by_quality(self, folder_path, results):
        """Organize files into quality-based subfolders"""
        try:
            # Create quality folders
            clear_folder = os.path.join(folder_path, "clear_faces")
            blurry_folder = os.path.join(folder_path, "blurry_faces")
            no_faces_folder = os.path.join(folder_path, "no_faces")
            
            os.makedirs(clear_folder, exist_ok=True)
            os.makedirs(blurry_folder, exist_ok=True)
            os.makedirs(no_faces_folder, exist_ok=True)
            
            # Move files based on quality
            moved_count = 0
            for result in results:
                if not result['success']:
                    continue
                    
                source_path = result['image_path']
                filename = os.path.basename(source_path)
                
                if result['face_count'] == 0:
                    dest_path = os.path.join(no_faces_folder, filename)
                elif result['has_clear_faces'] and not result['has_blurry_faces']:
                    dest_path = os.path.join(clear_folder, filename)
                elif result['has_blurry_faces'] and not result['has_clear_faces']:
                    dest_path = os.path.join(blurry_folder, filename)
                else:
                    # Mixed quality - put in clear folder but add prefix
                    dest_path = os.path.join(clear_folder, f"mixed_{filename}")
                
                # Copy file (don't move to avoid data loss)
                import shutil
                shutil.copy2(source_path, dest_path)
                moved_count += 1
            
            print(f"\n📁 Files organized into quality folders:")
            print(f"  Clear faces: {clear_folder}")
            print(f"  Blurry faces: {blurry_folder}")
            print(f"  No faces: {no_faces_folder}")
            print(f"  Total files organized: {moved_count}")
            
        except Exception as e:
            logger.error(f"Error organizing files: {e}")
    
    def save_summary_file(self, folder_path, summary):
        """Save detailed summary to file"""
        try:
            summary_file = os.path.join(folder_path, "face_quality_analysis.txt")
            
            with open(summary_file, 'w') as f:
                f.write("FACE QUALITY ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Folder: {folder_path}\n")
                f.write(f"Blur threshold: {self.blur_threshold}\n")
                f.write(f"Total images: {summary['total_images']}\n")
                f.write(f"Clear faces: {summary['images_with_clear_faces']}\n")
                f.write(f"Blurry faces: {summary['images_with_blurry_faces']}\n")
                f.write(f"No faces: {summary['images_no_faces']}\n")
                f.write(f"Errors: {summary['images_with_errors']}\n\n")
                
                f.write("DETAILED ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                for result in summary['results']:
                    filename = os.path.basename(result['image_path'])
                    f.write(f"\n{filename}:\n")
                    
                    if result['success']:
                        if result['face_count'] > 0:
                            for face in result['faces']:
                                f.write(f"  Face {face['face_number']}: {face['quality']} "
                                       f"(Laplacian: {face['laplacian_score']}, "
                                       f"Sobel: {face['sobel_score']})\n")
                        else:
                            f.write("  No faces detected\n")
                    else:
                        f.write(f"  Error: {result['error']}\n")
            
            print(f"📄 Detailed analysis saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")

def main():
    """Main function for face blur detection"""
    # Initialize detector with blur threshold
    # Adjust threshold: <50=very blurry, 50-100=blurry, >100=sharp
    detector = FaceBlurrinessDetector(blur_threshold=100)
    
    # UPDATE THIS PATH TO YOUR FOLDER
    folder_path = r"C:\Users\Olumayowa.Oyaleke\Downloads\Detection"
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    print("🔍 Face Blurriness Detection System")
    print("=" * 50)
    print("✅ Detects and analyzes face quality/sharpness")
    print("✅ Separates clear faces from blurry faces")
    print("✅ Provides quality scores and recommendations")
    print("✅ Can organize files by quality")
    print()
    
    # Process folder
    summary = detector.process_folder(
        folder_path=folder_path,
        display_results=True,      # Set False to skip individual displays
        save_summary=True,         # Save detailed report
        organize_by_quality=False  # Set True to organize files into folders
    )
    
    print("\n🎉 Analysis complete!")
    print("\n💡 Tips:")
    print("  - Lower Laplacian scores = more blurry")
    print("  - Adjust blur_threshold in code if needed")
    print("  - Set organize_by_quality=True to auto-organize files")

if __name__ == "__main__":
    print("Required packages: pip install face-recognition opencv-python matplotlib pillow pandas tabulate")
    print()
    main()