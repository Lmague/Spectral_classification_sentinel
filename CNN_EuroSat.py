from PIL import Image 
import base64 
import io 
from collections import Counter
import numpy as np
from tensorflow import keras

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='Templates', static_folder='static')
CORS(app)

my_model = keras.models.load_model("CNN_Weights.keras")

IMG_SIZE = 64 


# We define a list of class names predicted by the model. Each class corresponds to an output index.
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

URBAN_CLASSES = {"Residential", "Industrial", "Highway"}
URBAN_IDX = [i for i, n in enumerate(class_names) if n in URBAN_CLASSES]

STRIDE = 32 
TTA = True  
URBAN_THRESH = 0.35




def preprocess_tile(tile_img_pil):
    # We check is the PIL tile is  RBG beofre resizing it
    tile_img_pil_rgb = tile_img_pil.convert('RGB')
    
    img_resized = tile_img_pil_rgb.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0 # Normalization
    
    # Make sure the numpy array has 3 channels (RGB)
    # This is a double safety check after .convert(‘RGB’) and np.array()
    if img_array.ndim == 2:  # Grayscale image (should not arrive after convert(‘RGB’))
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 1: # Another form of grayscale
         img_array = np.concatenate([img_array]*3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA image (should not arrive after convert(‘RGB’))
        img_array = img_array[:, :, :3]
    
    # Check that the image has 3 channels after processing
    if img_array.shape[-1] != 3:
        raise ValueError(f"The pre-processed image must have 3 channels (RGB), but has {img_array.shape[-1]}.")

    # Add batch size (Keras wants a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    

@app.route('/predict', methods=['POST'])
def predict_route():

    data = request.get_json()
    
    image_base64_string = data['image_base64']

    # Decode base64 image
    # Remove prefix “data:image/png;base64,” or similar if present
    if "," in image_base64_string:
        header, image_base64_string = image_base64_string.split(',', 1)
    
    img_bytes = base64.b64decode(image_base64_string)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB') # Make sure it's RGB

    # Divide the image into 64 x 64 tiles and predict
    img_width, img_height = pil_image.size
    predictions_classes = [] # Stores predicted class names

    # Calculate the number of complete tiles
    num_tiles_x = img_width // IMG_SIZE
    num_tiles_y = img_height // IMG_SIZE
    
    if num_tiles_x == 0 or num_tiles_y == 0:
        return jsonify({"error": f"The image supplied ({img_width}x{img_height}) is too small to create 64x64 tiles."}), 400

    # Tuilage avec recouvrement
    predictions_classes = []
    urban_probs = []
    img_width, img_height = pil_image.size

    if img_width < IMG_SIZE or img_height < IMG_SIZE:
        return jsonify({"error": f"The image supplied ({img_width}x{img_height}) is too small to create 64x64 tiles."}), 400

    # itérations avec stride
    for top in range(0, img_height - IMG_SIZE + 1, STRIDE):
        for left in range(0, img_width - IMG_SIZE + 1, STRIDE):
            right, bottom = left + IMG_SIZE, top + IMG_SIZE
            tile_pil = pil_image.crop((left, top, right, bottom))

            x = preprocess_tile(tile_pil)  # shape (1,64,64,3)

            # Test-time augmentation (moyenne original + flip horizontal)
            if TTA:
                x_flip = np.flip(x, axis=2)  # flip largeur
                p1 = my_model.predict(x, verbose=0)[0]
                p2 = my_model.predict(x_flip, verbose=0)[0]
                probs = (p1 + p2) / 2.0
            else:
                probs = my_model.predict(x, verbose=0)[0]

            # Classe argmax et probabilité "urbain" (somme des 3 classes)
            predictions_classes.append(class_names[int(np.argmax(probs))])
            urban_probs.append(float(np.sum(probs[URBAN_IDX])))

    if not predictions_classes:
        return jsonify({"error": "No complete tile could be processed."}), 400

    class_counts = Counter(predictions_classes)
    total_tiles_processed = len(predictions_classes)

    urban_score = float(np.mean(urban_probs))  # moyenne des proba (0..1)
    urban_detected = urban_score >= URBAN_THRESH

    aggregated_results = {
        "total_tiles_processed": total_tiles_processed,
        "class_distribution": {
            name: f"{(class_counts[name] / total_tiles_processed * 100):.2f}%"
            for name in class_names if class_counts[name] > 0
        },
        "dominant_class_info": {
            "class": class_counts.most_common(1)[0][0] if class_counts else "N/A",
            "count": class_counts.most_common(1)[0][1] if class_counts else 0
        },
        # Nouveau: détection urbain robuste
        "urban": {
            "probability_mean": round(urban_score, 4),  # 0..1
            "threshold": URBAN_THRESH,
            "detected": urban_detected,
            "classes_used": list(URBAN_CLASSES)
        }
    }
    return jsonify(aggregated_results)


@app.route('/')
def index():
    return render_template("html_carte.html")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)