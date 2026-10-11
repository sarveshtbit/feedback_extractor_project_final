import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter

MODEL_PATH = "outputs/digit_cnn.keras"

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer - CNN",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognizer - CNN")
st.write("Upload a handwritten digit image and the CNN will predict the digit.")


# -----------------------------
# Cached Model Loader
# -----------------------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None


model = load_trained_model()

if model is None:
    st.error(
        f"⚠️ **Model file not found at `{MODEL_PATH}`!**\n\n"
        "Please train the CNN model first by running:\n"
        "```bash\npython train_cnn.py\n```"
    )
    st.stop()


# -----------------------------
# Image Preprocessing Function
# -----------------------------
def preprocess_digit_image(uploaded_image: Image.Image):
    """
    Preprocess user uploaded image to match MNIST-style 28x28 grayscale format:
    1. Convert to Grayscale
    2. Apply light Gaussian Blur to reduce noise
    3. Estimate background from borders and invert contrast if necessary (dark digit -> bright digit)
    4. Isolate foreground digit using thresholding
    5. Crop tightly around the digit with padding
    6. Resize preserving aspect ratio into a 20x20 bounding box
    7. Center digit on a 28x28 black canvas using intensity-weighted center of mass
    8. Normalize pixel values [0, 1] and reshape to (1, 28, 28, 1)
    """
    # 1. Grayscale
    image = uploaded_image.convert("L")
    # 2. Gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    arr = np.array(image).astype(np.float32)

    # 3. Estimate background from border pixels
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    background = np.median(border)

    # Foreground extraction & Contrast inversion
    if background > 127:
        # Light background + dark digit (e.g. black ink on white paper)
        foreground = np.clip(background - arr, 0, 255)
        mask = foreground > max(20, background * 0.18)
    else:
        # Dark background + bright digit (MNIST style)
        foreground = np.clip(arr - background, 0, 255)
        mask = foreground > max(20, (255 - background) * 0.18)

    # 4. Locate bounding box of the digit
    ys, xs = np.where(mask)

    if len(xs) > 20:
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        margin = max(2, int(0.08 * max(x2 - x1, y2 - y1)))
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(foreground.shape[1], x2 + margin), min(foreground.shape[0], y2 + margin)

        digit_crop = foreground[y1:y2, x1:x2]
    else:
        digit_crop = foreground

    digit_img = Image.fromarray(np.clip(digit_crop, 0, 255).astype(np.uint8), "L")

    # 5. Resize to max side 20 while preserving aspect ratio
    max_side = max(digit_img.size)
    if max_side > 0:
        scale = 20.0 / max_side
        new_w = max(1, int(digit_img.width * scale))
        new_h = max(1, int(digit_img.height * scale))
        digit_img = digit_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6. Paste onto 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    paste_x = (28 - digit_img.width) // 2
    paste_y = (28 - digit_img.height) // 2
    canvas.paste(digit_img, (paste_x, paste_y))

    # 7. Center by intensity-weighted center of mass
    a = np.array(canvas).astype(np.float32)
    total_intensity = a.sum()

    if total_intensity > 0:
        yy, xx = np.indices(a.shape)
        cx = (xx * a).sum() / total_intensity
        cy = (yy * a).sum() / total_intensity

        dx = int(round(13.5 - cx))
        dy = int(round(13.5 - cy))

        centered_canvas = Image.new("L", (28, 28), 0)
        centered_canvas.paste(canvas, (dx, dy))
        canvas = centered_canvas

    # 8. Normalize & Reshape for model input
    final_arr = np.array(canvas).astype("float32") / 255.0
    input_tensor = final_arr.reshape(1, 28, 28, 1)

    return canvas, input_tensor


# -----------------------------
# User Interface - Main Flow
# -----------------------------
st.markdown("---")
st.header("📸 Upload Image")

uploaded_file = st.file_uploader(
    "Choose a handwritten digit image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    original_img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(original_img, use_container_width=True)

    processed_canvas, input_array = preprocess_digit_image(original_img)

    with col2:
        st.subheader("🖼️ Preprocessed Image")
        st.caption("Reshaped & Centered to 28x28 for CNN input")
        st.image(processed_canvas.resize((280, 280), Image.Resampling.NEAREST), use_container_width=True)

    # Inference
    probabilities = model.predict(input_array, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence_pct = float(probabilities[predicted_digit]) * 100.0

    st.markdown("---")
    st.header("🎯 Prediction Result")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.success(f"### Predicted Digit: **{predicted_digit}**")
    with res_col2:
        st.metric(label="Confidence", value=f"{confidence_pct:.2f}%")

    st.markdown("---")
    st.header("📊 Class Probabilities")
    prob_dict = {f"Digit {i}": float(probabilities[i]) for i in range(10)}
    st.bar_chart(prob_dict)

    st.info(
        "💡 **Note**: The confidence score represents the Softmax output probability of the CNN for each digit (0-9)."
    )
else:
    st.info("👆 Please upload a handwritten digit image above to generate predictions.")
