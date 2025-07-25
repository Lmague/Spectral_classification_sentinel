// Initialise la carte Leaflet
var map = L.map('map').setView([45.721867937410565, 4.916832597175374], 13);

// Ajoute une couche de tuiles OpenStreetMap sur la carte
L.tileLayer('https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpeg', {
    attribution: 'Sentinel-2 cloudless by EOX',
    maxZoom: 19
}).addTo(map);

var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({
    draw: { polyline: false,
            polygon: false,
            marker: false,
            circle: false,
            circlemarker: false,
            rectangle: true }, // Seul dessiner un rectangle est autorisé
    edit: { featureGroup: drawnItems }
});
map.addControl(drawControl);

const Res_Sentinel_2 = 10.0; // résolution de Snetinel-2 en m/pixel (chaque pixel fait 10m2)

map.on(L.Draw.Event.CREATED, function (event) {
    var layer = event.layer;
    var bounds = layer.getBounds();

    drawnItems.clearLayers();
    drawnItems.addLayer(layer);

    const resultsDiv = document.getElementById('ResPred');

    // Fonction centralisée pour afficher et logger les erreurs
    function displayError(message, err = null) {
        console.error(message, err);                   // Log dans la console pour debug
        if (resultsDiv) {
            resultsDiv.innerHTML = `<p style='color:red;'>${message}</p>`; // Affiche en HTML si il y a une erreur
        }
    }

    resultsDiv.innerHTML = "<p>Capture, redimensionnement et analyse en cours...</p>"; // texte lors de l'operération de prédiction

    var sw = bounds.getSouthWest();
    var ne = bounds.getNorthEast();
    var nw = bounds.getNorthWest();
    var se = bounds.getSouthEast();

    // Erreur qui apparait si il y a une erreur dans l'importation des coins de la zone sélectionnée.
    if (!sw || !ne || !nw || !se) {
        return displayError('Impossible d\'obtenir les coins du rectangle.');
    }

    var rectWidthMeters = nw.distanceTo(L.latLng(nw.lat, ne.lng)); 
    var rectHeightMeters = nw.distanceTo(L.latLng(sw.lat, nw.lng)); // Pour obtenir la taille en metres à la place des coordonées des points

    // Les dimensions de l'image sont nulles ou <= 0 (pas possible d'être <0 mais au cas où si un bug apparait)
    if (isNaN(rectWidthMeters) || isNaN(rectHeightMeters) || rectWidthMeters <= 0 || rectHeightMeters <= 0) {
        return displayError('Dimensions géographiques du rectangle invalides ou nulles.');
    }

    var finalImageWidthPx = Math.round(rectWidthMeters / Res_Sentinel_2);
    var finalImageHeightPx = Math.round(rectHeightMeters / Res_Sentinel_2); // Transforme la hauteur et la largeur du rectangle en pixels plutôt qu'en metres

    // Erreur si la zone séléctionnée est trop petite pour passer dans notre alogrithme
    if (finalImageWidthPx < 64 || finalImageHeightPx < 64) {
        let message = `Zone sélectionnée trop petite (${finalImageWidthPx}px x ${finalImageHeightPx}px). Veuillez sélectionner une zone plus grande.`;
        return displayError(message);
    }

    leafletImage(map, function(err, originalMapCanvas) {""

        var nwPointOnCanvas = map.latLngToContainerPoint(nw);
        var sePointOnCanvas = map.latLngToContainerPoint(se); // Capture les coordonées Sud-Est et Nord-Ouest du rectangle pour avoir le rectangle en entier
    

        var onScreenCropX = nwPointOnCanvas.x;
        var onScreenCropY = nwPointOnCanvas.y;
        var onScreenCropWidthPx = sePointOnCanvas.x - onScreenCropX;
        var onScreenCropHeightPx = sePointOnCanvas.y - onScreenCropY;

        // Erreur qui vérifie que la zone à rogner sur le canvas HTML est valide (par exemple si la zone sélectionnée se trouve partiellement ou totalement hors écran).
        if (onScreenCropWidthPx <= 0 || onScreenCropHeightPx <= 0 || isNaN(onScreenCropWidthPx) || isNaN(onScreenCropHeightPx)) {
            return displayError('Zone sélectionnée à l\'écran trop petite ou invalide pour le rognage.');
        }

        // Création d'un canvas temporaire pour le rognage
        var tempCroppedCanvas = document.createElement('canvas');
        tempCroppedCanvas.width = onScreenCropWidthPx;
        tempCroppedCanvas.height = onScreenCropHeightPx;
        var tempCtx = tempCroppedCanvas.getContext('2d');
        try {
            tempCtx.drawImage(
                originalMapCanvas,
                onScreenCropX, onScreenCropY,
                onScreenCropWidthPx, onScreenCropHeightPx,
                0, 0,
                onScreenCropWidthPx, onScreenCropHeightPx
            );
        } catch (e) {
            // Erreur qui apparait si il y a eu une erreur lors du rognage
            return displayError(`Erreur technique lors du rognage de l'image: ${e.message}`);
        }

        // Redimensionnement vers la résolution cible
        var finalResizedCanvas = document.createElement('canvas');
        finalResizedCanvas.width = finalImageWidthPx;
        finalResizedCanvas.height = finalImageHeightPx;
        var finalCtx = finalResizedCanvas.getContext('2d');
        finalCtx.drawImage(tempCroppedCanvas, 0, 0, finalResizedCanvas.width, finalResizedCanvas.height);

        // Encode en base64
        var imgData;
        try {
            imgData = finalResizedCanvas.toDataURL('image/png');
        } catch (e) {
            // Si problème lors de la conversion de l'image en un URL
            return displayError(`Erreur technique lors de la conversion de l'image: ${e.message}`);
        }

        // Envoi au backend (API Flask sur Python) 
        fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: imgData })
        })
        .then(response => response.json())
        .then(data => {
            let resultText = `<h2>Résultats :</h2>`;
            resultText += `<p>Nombre total de tuiles traitées : ${data.total_tiles_processed}</p>`;
            resultText += `<p><i>La zone sélectionnée fait ${finalImageWidthPx}px x ${finalImageHeightPx}px soit ${finalImageWidthPx*10}m par ${finalImageHeightPx*10}m</i></p>`;

            // Classe dominante
            if (data.dominant_class_info && data.dominant_class_info.class !== 'N/A') {
                resultText += `<p><strong>Classe dominante :</strong> ${data.dominant_class_info.class} (${data.dominant_class_info.count} tuiles)</p>`;
            } else {
                resultText += `<p>Aucune classe dominante identifiée.</p>`;
            }

            // Répartition complète
            if (data.class_distribution) {
                resultText += `<h3>Répartition des classes :</h3><ul>`;
                for (const [className, percentage] of Object.entries(data.class_distribution)) {
                    resultText += `<li>${className} : ${percentage}</li>`;
                }
                resultText += `</ul>`;
            }

            document.getElementById("ResPred").innerHTML = resultText;
        })
            
    })
});
