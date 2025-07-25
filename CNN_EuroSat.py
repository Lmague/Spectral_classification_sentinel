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

mon_model = keras.models.load_model("ModelCNN_WEBIA.keras")

IMG_SIZE = 64 


# On défini une liste avec les noms des classes que le modèle prédit, elles correspondent à l'ordre de la sortie du modèle
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]



def preprocess_tile(tile_img_pil):
    # S'assurer que la tuile PIL est bien en mode RGB avant le resize
    tile_img_pil_rgb = tile_img_pil.convert('RGB')
    
    img_resized = tile_img_pil_rgb.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0 # Normalisation
    
    # S'assurer que l'array numpy a 3 canaux (RGB)
    # Cette vérification est une double sécurité après .convert('RGB') et np.array()
    if img_array.ndim == 2:  # Image en niveaux de gris (ne devrait pas arriver après convert('RGB'))
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 1: # Autre forme de niveaux de gris
         img_array = np.concatenate([img_array]*3, axis=-1)
    elif img_array.shape[-1] == 4:  # Image RGBA (ne devrait pas arriver après convert('RGB'))
        img_array = img_array[:, :, :3]
    
    # Vérifier que l'image a bien 3 canaux après tout traitement
    if img_array.shape[-1] != 3:
        raise ValueError(f"L'image prétraitée doit avoir 3 canaux (RGB), mais en a {img_array.shape[-1]}.")

    # Ajouter la dimension du batch (Keras veut un batch d'images)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    

@app.route('/predict', methods=['POST'])
def predict_route():

    data = request.get_json()
    
    image_base64_string = data['image_base64']

    # Décoder l'image base64
    # Enlever le préfixe "data:image/png;base64," ou similaire si présent
    if "," in image_base64_string:
        header, image_base64_string = image_base64_string.split(',', 1)
    
    img_bytes = base64.b64decode(image_base64_string)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB') # S'assurer qu'elle est en RGB

    # Diviser l'image en tuiles de 64 x 64 et prédire
    img_width, img_height = pil_image.size
    predictions_classes = [] # Stocke les noms des classes prédites

    # Calculer le nombre de tuiles complètes
    num_tiles_x = img_width // IMG_SIZE
    num_tiles_y = img_height // IMG_SIZE
    
    if num_tiles_x == 0 or num_tiles_y == 0:
        return jsonify({"error": f"L'image fournie ({img_width}x{img_height}) est trop petite pour créer des tuiles de 64x64."}), 400

    for i in range(num_tiles_y): 
        for j in range(num_tiles_x):
            # Coordonnées pour extraire la tuile
            left = j * IMG_SIZE
            top = i * IMG_SIZE
            right = left + IMG_SIZE
            bottom = top + IMG_SIZE
            
            tile_pil = pil_image.crop((left, top, right, bottom))
            
            processed_tile_array = preprocess_tile(tile_pil)

            # Prédiction
            probs = mon_model.predict(processed_tile_array)[0]
            predicted_class_index = np.argmax(probs)
            predicted_class_name = class_names[predicted_class_index]
            predictions_classes.append(predicted_class_name)

    if not predictions_classes:
        return jsonify({"error": "Aucune tuile complète n'a pu être traitée."}), 400

    # Agréger les résultats
    class_counts = Counter(predictions_classes)
    total_tiles_processed = len(predictions_classes)
    
    aggregated_results = {
        "total_tiles_processed": total_tiles_processed,
        "class_distribution": {
            name: f"{(class_counts[name] / total_tiles_processed * 100):.2f}%" for name in class_names if class_counts[name] > 0
        },
        "dominant_class_info": {
            "class": class_counts.most_common(1)[0][0] if class_counts else "N/A",
            "count": class_counts.most_common(1)[0][1] if class_counts else 0
        }
    }
    return jsonify(aggregated_results)

@app.route('/')
def index():
    return render_template("html_carte.html")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)