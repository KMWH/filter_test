import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.feature import match_template
from scipy.fft import fft2

# Helper functions
def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if not exif_data:
        return {}
    exif = {TAGS.get(tag): value for tag, value in exif_data.items()}
    return exif

def is_edited_by_software(exif_data):
    software = exif_data.get('Software', '')
    known_editing_software = ['Adobe Photoshop', 'Instagram', 'Snow']
    for software_name in known_editing_software:
        if software_name in software:
            return True
    return False

def ela_image(image_path, scale=10):
    original = Image.open(image_path)
    original.save("temp.jpg", quality=90)
    resaved = Image.open("temp.jpg")

    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def noise_analysis(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noise_img = random_noise(image, mode='s&p', amount=0.05)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img

def detect_clones(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = gray[0:50, 0:50]
    result = match_template(gray, template)
    return result

def jpeg_quant_analysis(image_path):
    image = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    dct_y = cv2.dct(np.float32(y) / 255.0)
    dct_cr = cv2.dct(np.float32(cr) / 255.0)
    dct_cb = cv2.dct(np.float32(cb) / 255.0)
    return dct_y, dct_cr, dct_cb

def lighting_analysis(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return gradient_x, gradient_y

def spectral_analysis(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f_transform = fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
    return magnitude_spectrum

def compression_artifacts_analysis(image_path):
    image = cv2.imread(image_path)
    _, encoded_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    decoded_img = cv2.imdecode(encoded_img, 1)
    diff = cv2.absdiff(image, decoded_img)
    return diff

def mark_changes(image, changes):
    marked_image = image.copy()
    for change in changes:
        x, y, w, h = change
        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return marked_image

def process_image(image_path, result_directory):
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    
    exif_data = get_exif_data(image_path)
    edited = is_edited_by_software(exif_data)
    
    # Save metadata analysis result
    with open(os.path.join(result_directory, f"{os.path.basename(image_path)}_metadata.txt"), 'w') as f:
        f.write(f"Is the image edited? {'Yes' if edited else 'No'}\n")
        f.write(f"EXIF Data: {exif_data}\n")
    
    changes = []
    
    # ELA
    ela_result = ela_image(image_path)
    ela_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_ela.png")
    ela_result.save(ela_image_path)
    
    # Noise Analysis
    noise_result = noise_analysis(image_path)
    noise_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_noise.png")
    plt.imsave(noise_image_path, noise_result, cmap='gray')
    
    # Clone Detection
    clone_result = detect_clones(image_path)
    clone_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_clone.png")
    plt.imsave(clone_image_path, clone_result, cmap='gray')
    
    # JPEG Quantization Analysis
    dct_y, dct_cr, dct_cb = jpeg_quant_analysis(image_path)
    dct_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_dct.png")
    plt.imsave(dct_image_path, dct_y, cmap='gray')
    
    # Lighting Analysis
    gradient_x, gradient_y = lighting_analysis(image_path)
    gradient_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_gradient.png")
    plt.imsave(gradient_image_path, gradient_x, cmap='gray')
    
    # Spectral Analysis
    magnitude_spectrum = spectral_analysis(image_path)
    spectral_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_spectrum.png")
    plt.imsave(spectral_image_path, magnitude_spectrum, cmap='gray')
    
    # Compression Artifacts Analysis
    diff = compression_artifacts_analysis(image_path)
    diff_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_diff.png")
    cv2.imwrite(diff_image_path, diff)
    
    # Load original image for marking
    original_image = cv2.imread(image_path)
    
    # Mark changes
    if edited:
        changes.append((0, 0, original_image.shape[1], original_image.shape[0]))
    if np.max(noise_result) > 0:
        changes.append((0, 0, noise_result.shape[1], noise_result.shape[0]))
    if np.max(clone_result) > 0.5:
        changes.append((0, 0, clone_result.shape[1], clone_result.shape[0]))
    
    marked_image = mark_changes(original_image, changes)
    marked_image_path = os.path.join(result_directory, f"{os.path.basename(image_path)}_marked.png")
    cv2.imwrite(marked_image_path, marked_image)

def analyze_images(input_directory, result_directory):
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_directory, filename)
            process_image(image_path, result_directory)

# Paths
input_directory = './images'
result_directory = './result'

# Analyze images
analyze_images(input_directory, result_directory)
