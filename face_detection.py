import cv2
import numpy as np
import argparse
import os
import urllib.request

FACE_SCALE = 1.1
FACE_NEIGHBORS = 5
EYE_SCALE = 1.1
EYE_NEIGHBORS = 5
THICKNESS_DIVISOR = 500
TEXT = "this is me"

def download_cascades():
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    files = ['haarcascade_frontalface_default.xml', 'haarcascade_eye.xml']
    for filename in files:
        if not os.path.exists(filename):
            urllib.request.urlretrieve(base_url + filename, filename)

def load_cascades():
    download_cascades()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    return face_cascade, eye_cascade

def fix_orientation_if_available(img):
    return img

def detect_first_face(gray, face_cascade):
    if face_cascade is None:
        return None

    H, W = gray.shape
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE,
        minNeighbors=FACE_NEIGHBORS,
        minSize=(int(W*0.1), int(H*0.1))
    )

    if len(faces) > 0:
        return faces[0]
    return None

def detect_or_synthesize_eyes(gray, face_bbox, eye_cascade):
    x, y, w, h = face_bbox
    roi = gray[y:y+h, x:x+w]

    detected_eyes = []

    if eye_cascade is not None:
        eyes = eye_cascade.detectMultiScale(
            roi,
            scaleFactor=EYE_SCALE,
            minNeighbors=EYE_NEIGHBORS,
            minSize=(int(w*0.1), int(h*0.1)),
            maxSize=(int(w*0.5), int(h*0.5))
        )

        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            if ey + eh/2 < h*0.6 and 0.15*w < ew < 0.4*w:
                valid_eyes.append((ex + x, ey + y, ew, eh))

        valid_eyes.sort(key=lambda e: e[0])
        detected_eyes = valid_eyes[:2]

    if len(detected_eyes) < 2:
        ew = int(0.22*w)
        eh = int(0.18*h)
        left_eye = (x + int(0.18*w), y + int(0.28*h), ew, eh)
        right_eye = (x + int(0.60*w), y + int(0.28*h), ew, eh)

        if len(detected_eyes) == 0:
            detected_eyes = [left_eye, right_eye]
        elif len(detected_eyes) == 1:
            existing_x = detected_eyes[0][0]
            if existing_x < x + w//2:
                detected_eyes.append(right_eye)
            else:
                detected_eyes.insert(0, left_eye)

    return detected_eyes[:2]

def compute_face_circle(face_bbox, image_shape):
    x, y, w, h = face_bbox
    H, W = image_shape[:2]

    cx = x + w//2
    cy = y + h//2
    r = int(0.55 * max(w, h))

    r = min(r, cx, cy, W-cx, H-cy)

    return (cx, cy), r

def draw_annotations(img, circle, eyes, text):
    H, W = img.shape[:2]
    thickness = max(2, int(max(W, H) / THICKNESS_DIVISOR))

    center, radius = circle
    cv2.circle(img, center, radius, (0, 255, 0), thickness, lineType=cv2.LINE_AA)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), thickness, lineType=cv2.LINE_AA)

    font_scale = max(0.6, max(W, H) / 1000)
    text_x, text_y = 20, H - 20

    if center[1] + radius > H - 60:
        text_y = center[1] - radius - 3 * thickness

    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.jpg', help='Input image path')
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        input_path = '/Users/eli/Downloads/Module 3 Image.jpeg'

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_annotated.jpg")

    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not load image: {input_path}")
        return

    img = fix_orientation_if_available(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade, eye_cascade = load_cascades()

    face_bbox = detect_first_face(gray, face_cascade)

    if face_bbox is None:
        H, W = img.shape[:2]
        thickness = max(2, int(max(W, H) / THICKNESS_DIVISOR))
        font_scale = max(0.6, max(W, H) / 1000)
        cv2.putText(img, TEXT, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, TEXT, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    else:
        eyes = detect_or_synthesize_eyes(gray, face_bbox, eye_cascade)
        circle = compute_face_circle(face_bbox, img.shape)
        draw_annotations(img, circle, eyes, TEXT)

    cv2.imwrite(output_path, img)
    print(f"Processed image saved to: {output_path}")

    try:
        os.system(f'open "{output_path}"')
    except:
        pass

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('noisy_image.jpg', cv2.IMREAD_GRAYSCALE)

noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
noisy_img = cv2.add(img, noise)

gaussian_filtered = cv2.GaussianBlur(noisy_img, (5, 5), sigmaX=1.5)

mean_filtered = cv2.blur(noisy_img, (5, 5))
median_filtered = cv2.medianBlur(noisy_img, 5)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(noisy_img, cmap='gray')
axes[0, 0].set_title('Noisy Image')
axes[0, 1].imshow(gaussian_filtered, cmap='gray')
axes[0, 1].set_title('Gaussian Filtered (σ=1.5)')
axes[1, 0].imshow(mean_filtered, cmap='gray')
axes[1, 0].set_title('Mean Filtered')
axes[1, 1].imshow(median_filtered, cmap='gray')
axes[1, 1].set_title('Median Filtered')
plt.tight_layout()
plt.savefig('filtering_comparison.png')
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
sigmas = [0.5, 1.5, 3.0, 5.0]
for i, sigma in enumerate(sigmas):
    filtered = cv2.GaussianBlur(noisy_img, (9, 9), sigmaX=sigma)
    axes[i].imshow(filtered, cmap='gray')
    axes[i].set_title(f'σ = {sigma}')
plt.tight_layout()
plt.savefig('gaussian_sigma_comparison.png')
plt.show()

