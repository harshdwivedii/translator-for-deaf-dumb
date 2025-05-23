import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import time
from collections import deque # For captions queue

# ─────────────────────────────── MONKEY PATCH ───────────────────────────────────
_orig_load = tf.keras.models.load_model
def _load_model_with_patch(path, *args, **kwargs):
    print("INFO: Applying Keras DepthwiseConv2D monkey patch.")
    co = kwargs.setdefault("custom_objects", {})
    def _patched_depthwise(*layers_args, **layers_kwargs):
        layers_kwargs.pop("groups", None)
        return DepthwiseConv2D(*layers_args, **layers_kwargs)
    co["DepthwiseConv2D"] = _patched_depthwise
    return _orig_load(path, *args, **kwargs)
tf.keras.models.load_model = _load_model_with_patch
# ─────────────────────────── END OF MONKEY PATCH ────────────────────────────────

# ─────────────────────────────── CONFIGURATION ──────────────────────────────────
# --- Model and Label Paths ---
BASE_DIR = r"C:\Users\VIGHNESHWAR MISHRA\Desktop\sign language detect\converted_keras"
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "labels.txt")

# --- Detection & Image Parameters ---
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
IMG_SIZE = 300
OFFSET = 20
BORDER_THICKNESS = 4

# --- UI Parameters ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_INFO = 0.6
FONT_SCALE_PREDICTION = 0.9
FONT_THICKNESS = 2
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_BLACK = (0, 0, 0)
BG_COLOR_GREEN = (0, 255, 0)
BG_COLOR_RED = (0, 0, 255)
BG_COLOR_BLUE = (255, 100, 50)
BOX_COLOR_DETECTED = (0, 255, 0)

# --- Caption Parameters ---
MAX_CAPTIONS_IN_BOX = 5  # Number of recent captions to display
CAPTION_CONFIDENCE_THRESHOLD = 0.75 # Minimum confidence to add to caption
CAPTION_BOX_HEIGHT_PERCENT = 0.25 # Percentage of screen height for caption box
CAPTION_BOX_COLOR = (30, 30, 30, 200) # Semi-transparent dark grey (B,G,R,Alpha for overlay)
CAPTION_FONT_SCALE = 0.6
CAPTION_LINE_SPACING = 5 # Pixels between caption lines
CAPTION_PADDING = 10

# --- Debug ---
SHOW_DEBUG_WINDOWS = False # Press 'd' to toggle
# ─────────────────────────── END OF CONFIGURATION ───────────────────────────────

def load_labels(label_path):
    try:
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
            if not labels:
                print(f"WARNING: No labels found in {label_path}. Using default.")
                return ["Unknown"]
            print(f"INFO: Loaded {len(labels)} labels: {labels}")
            return labels
    except FileNotFoundError:
        print(f"ERROR: Label file not found at {label_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"ERROR: Could not read labels from {label_path}: {e}. Exiting.")
        exit()

def draw_ui_elements(img, status_text, prediction_text, fps_text, show_debug, captions_queue):
    h_img, w_img, _ = img.shape
    overlay = img.copy() # For transparent elements

    # --- Status Bar ---
    status_bar_height = 40
    cv2.rectangle(img, (0, 0), (w_img, status_bar_height), BG_COLOR_BLUE, cv2.FILLED)
    cv2.putText(img, status_text, (10, status_bar_height - 10), FONT, FONT_SCALE_INFO, TEXT_COLOR_WHITE, FONT_THICKNESS)
    cv2.putText(img, fps_text, (w_img - 100, status_bar_height - 10), FONT, FONT_SCALE_INFO, TEXT_COLOR_WHITE, FONT_THICKNESS)

    # --- Prediction Text (near hand) ---
    # This is now drawn in the main loop where hand coordinates are known

    # --- Instructions ---
    instruction_y_start = h_img - 10
    instructions = [
        f"Press 'd' to toggle debug ({'ON' if show_debug else 'OFF'})",
        "Press 'q' to quit.",
        "Show hand to detect sign."
    ]
    for i, instruction in enumerate(instructions):
        (text_w, text_h), _ = cv2.getTextSize(instruction, FONT, FONT_SCALE_INFO * 0.8, FONT_THICKNESS -1)
        y_pos = instruction_y_start - i * (text_h + 5)
        cv2.putText(img, instruction, (10, y_pos), FONT, FONT_SCALE_INFO*0.8, BG_COLOR_BLUE, FONT_THICKNESS+1)
        cv2.putText(img, instruction, (10, y_pos), FONT, FONT_SCALE_INFO*0.8, TEXT_COLOR_WHITE, FONT_THICKNESS-1)

    # --- Caption Box ---
    caption_box_actual_height = int(h_img * CAPTION_BOX_HEIGHT_PERCENT)
    caption_box_y1 = status_bar_height + CAPTION_PADDING
    caption_box_y2 = caption_box_y1 + caption_box_actual_height
    caption_box_x1 = w_img - 300 - CAPTION_PADDING # Assuming a width for the caption box
    caption_box_x2 = w_img - CAPTION_PADDING

    if caption_box_x1 < 0: caption_box_x1 = CAPTION_PADDING # ensure it's not off screen if window too small

    # Create a semi-transparent rectangle for the caption box
    # Ensure overlay has 4 channels if using alpha
    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    cv2.rectangle(overlay, (caption_box_x1, caption_box_y1), (caption_box_x2, caption_box_y2),
                  CAPTION_BOX_COLOR, cv2.FILLED)
    # Blend the overlay with the original image
    alpha = CAPTION_BOX_COLOR[3] / 255.0 # Extract alpha from CAPTION_BOX_COLOR
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_bgra = cv2.addWeighted(overlay, alpha, img_bgra, 1 - alpha, 0)
    img[:] = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR) # Convert back to BGR


    cv2.putText(img, "Captions:", (caption_box_x1 + CAPTION_PADDING, caption_box_y1 + CAPTION_PADDING + 10),
                FONT, CAPTION_FONT_SCALE * 0.9, TEXT_COLOR_WHITE, 1)

    # Display captions (newest at the top of the box)
    current_y = caption_box_y1 + CAPTION_PADDING + 10 + CAPTION_LINE_SPACING + 15 # Start below "Captions:" title
    for i, caption_text in enumerate(reversed(list(captions_queue))): # Newest first
        if current_y + (CAPTION_LINE_SPACING + 10) > caption_box_y2: # Check if text will overflow
            break
        cv2.putText(img, f"- {caption_text}",
                    (caption_box_x1 + CAPTION_PADDING, current_y),
                    FONT, CAPTION_FONT_SCALE, TEXT_COLOR_WHITE, 1)
        current_y += (cv2.getTextSize(caption_text, FONT, CAPTION_FONT_SCALE, 1)[0][1] + CAPTION_LINE_SPACING)

def main():
    global SHOW_DEBUG_WINDOWS

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
    print("INFO: Model exists at", MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector = HandDetector(staticMode=False, maxHands=MAX_HANDS, modelComplexity=1, detectionCon=DETECTION_CONFIDENCE, minTrackCon=TRACKING_CONFIDENCE)
    classifier = Classifier(MODEL_PATH, LABEL_PATH)
    labels = load_labels(LABEL_PATH)

    prev_time = 0
    status_text = "Initializing..."
    prediction_text_on_hand = "" # Text to draw near the hand

    captions_queue = deque(maxlen=MAX_CAPTIONS_IN_BOX)
    last_added_caption_label = None

    while True:
        success, img = cap.read()
        if not success:
            status_text = "ERROR: Failed to grab frame."
            time.sleep(0.5)
            continue

        img = cv2.flip(img, 1)
        imgOutput = img.copy()
        hands, _ = detector.findHands(img, draw=False) # We'll draw our own box

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        fps_text = f"FPS: {int(fps)}"
        status_text = "Detecting..."
        prediction_text_on_hand = "" # Reset per frame

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            lmList = hand['lmList'] # Get landmarks for more precise drawing if needed

            imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
            x1, y1 = max(0, x - OFFSET), max(0, y - OFFSET)
            x2, y2 = min(img.shape[1], x + w + OFFSET), min(img.shape[0], y + h + OFFSET)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                status_text = "Processing... (Crop failed)"
                # Draw UI and continue
                draw_ui_elements(imgOutput, status_text, prediction_text_on_hand, fps_text, SHOW_DEBUG_WINDOWS, captions_queue)
                cv2.imshow("Sign Language Detection", imgOutput)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('d'): SHOW_DEBUG_WINDOWS = not SHOW_DEBUG_WINDOWS
                continue

            status_text = "Processing..."
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = IMG_SIZE / h
                    wCal = math.ceil(k * w)
                    if wCal <= 0 or IMG_SIZE <=0: raise ValueError("Calculated width non-positive")
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = IMG_SIZE / w
                    hCal = math.ceil(k * h)
                    if IMG_SIZE <= 0 or hCal <=0: raise ValueError("Calculated height non-positive")
                    imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                    hGap = math.ceil((IMG_SIZE - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                prediction_scores, index = classifier.getPrediction(imgWhite, draw=False)

                if 0 <= index < len(labels):
                    current_label = labels[index]
                    confidence = prediction_scores[index]
                    prediction_text_on_hand = f"{current_label} ({confidence*100:.1f}%)"

                    # Add to caption queue logic
                    if current_label != "Unknown" and \
                       current_label != last_added_caption_label and \
                       confidence >= CAPTION_CONFIDENCE_THRESHOLD:
                        captions_queue.append(current_label)
                        last_added_caption_label = current_label

                else:
                    current_label = "Unknown"
                    prediction_text_on_hand = "Unknown Prediction"
                    print(f"Warning: Index {index} out of bounds for labels (len: {len(labels)})")

                # Draw hand bounding box and prediction text near hand
                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), BOX_COLOR_DETECTED, BORDER_THICKNESS)
                (text_w, text_h), _ = cv2.getTextSize(prediction_text_on_hand, FONT, FONT_SCALE_PREDICTION, FONT_THICKNESS)
                cv2.rectangle(imgOutput, (x1, max(0,y1 - text_h - 10)), (x1 + text_w + 10, y1), BG_COLOR_GREEN, cv2.FILLED)
                cv2.putText(imgOutput, prediction_text_on_hand, (x1 + 5, y1 - 5), FONT, FONT_SCALE_PREDICTION, TEXT_COLOR_BLACK, FONT_THICKNESS)

                if SHOW_DEBUG_WINDOWS:
                    cv2.imshow("Hand Crop", imgCrop)
                    cv2.imshow("Processed Hand", imgWhite)

            except ValueError as ve:
                print(f"ERROR during image processing: {ve}")
                status_text = f"Error: {ve}"
            except Exception as e:
                print(f"ERROR during classification or drawing: {e}")
                status_text = "Error in classification!"
        else:
            status_text = "No hand detected."
            # No hand, so the next different detected sign should be added
            # We don't reset last_added_caption_label here, so if the same sign appears after
            # no hand was detected, it won't be added again unless it was different
            # from the one *before* the hand disappeared. This feels reasonable.
            if SHOW_DEBUG_WINDOWS:
                if cv2.getWindowProperty("Hand Crop", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Hand Crop")
                if cv2.getWindowProperty("Processed Hand", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Processed Hand")

        draw_ui_elements(imgOutput, status_text, prediction_text_on_hand, fps_text, SHOW_DEBUG_WINDOWS, captions_queue)
        cv2.imshow("Sign Language Detection", imgOutput)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            SHOW_DEBUG_WINDOWS = not SHOW_DEBUG_WINDOWS
            if not SHOW_DEBUG_WINDOWS:
                 if cv2.getWindowProperty("Hand Crop", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Hand Crop")
                 if cv2.getWindowProperty("Processed Hand", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Processed Hand")

    print("INFO: Exiting program...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    main()