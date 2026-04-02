from flask import Flask, render_template, request, send_file
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import gdown

app = Flask(__name__)

# ==============================
# MODEL SETUP
# ==============================
MODEL_PATH = "malaria_model.h5"

MODEL_URL = "https://drive.google.com/uc?id=1s_nk0OXVgsukkjRvFxb4TCUKou72wzaB"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ==============================
# ROUTES
# ==============================

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    image = load_img(filepath, target_size=(150, 150))
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0) / 255.0

    result = model.predict(img)
    confidence = float(result[0][0])

    img_path = "/" + filepath

    if confidence < 0.5:
        status = "Malaria Detected"
        conf = round((1 - confidence) * 100, 2)
        return render_template("diseased.html",
                               user_image=img_path,
                               status=status,
                               value=conf)
    else:
        status = "No Malaria"
        conf = round(confidence * 100, 2)
        return render_template("healthy.html",
                               user_image=img_path,
                               status=status,
                               value=conf)


@app.route('/download_report/<status>/<confidence>')
def download_report(status, confidence):
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    file_path = "report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("<b>Malaria Detection Report</b>", styles['Title']))
    content.append(Paragraph(f"Result: {status}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))

    doc.build(content)

    return send_file(file_path, as_attachment=True)


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)