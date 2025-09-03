// Initialize Leaflet map
var map = L.map('map').setView([45.721867937410565, 4.916832597175374], 13);
    // GPS coordinates are those of Université Lumière Lyon 2

// Adds an OpenStreetMap tile layer to the map
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
            rectangle: true }, // Only rectangle are a allowed to be drawn
    edit: { featureGroup: drawnItems }
});
map.addControl(drawControl);

const Res_Sentinel_2 = 10.0; // Snetinel-2 resolution in m/pixel (each pixel is 10m2)

map.on(L.Draw.Event.CREATED, function (event) {
    var layer = event.layer;
    var bounds = layer.getBounds();

    drawnItems.clearLayers();
    drawnItems.addLayer(layer);

    const resultsDiv = document.getElementById('ResPred');

    // Centralized function for displaying and logging errors
    function displayError(message, err = null) {
        console.error(message, err);                   // Log to console for debugging
        if (resultsDiv) {
            resultsDiv.innerHTML = `<p style='color:red;'>${message}</p>`; // Displays in HTML if there is an error
        }
    }

    resultsDiv.innerHTML = "<p>Capture, resize and analyze in progress...</p>"; // Text during prediction operation

    var sw = bounds.getSouthWest();
    var ne = bounds.getNorthEast();
    var nw = bounds.getNorthWest();
    var se = bounds.getSouthEast();

    // Error that appears if there is an error in importing the corners of the selected zone.
    if (!sw || !ne || !nw || !se) {
        return displayError('Impossible to get the corners of the rectangle.');
    }

    var rectWidthMeters = nw.distanceTo(L.latLng(nw.lat, ne.lng)); 
    var rectHeightMeters = nw.distanceTo(L.latLng(sw.lat, nw.lng)); // To obtain the size in meters instead of point coordinates

    // Image dimensions are zero or <= 0 (not possible to be <0 but just in case if a bug appears)
    if (isNaN(rectWidthMeters) || isNaN(rectHeightMeters) || rectWidthMeters <= 0 || rectHeightMeters <= 0) {
        return displayError('Geographical dimensions of rectangle invalid or null.');
    }

    var finalImageWidthPx = Math.round(rectWidthMeters / Res_Sentinel_2);
    var finalImageHeightPx = Math.round(rectHeightMeters / Res_Sentinel_2); // Transform rectangle height and width into pixels rather than meters
    // Error if the selected zone is too small to pass through our alogrithm
    if (finalImageWidthPx < 64 || finalImageHeightPx < 64) {
        let message = `Zone selected too small (${finalImageWidthPx}px x ${finalImageHeightPx}px). Please select a bigger one.`;
        return displayError(message);
    }

    leafletImage(map, function(err, originalMapCanvas) {""

        var nwPointOnCanvas = map.latLngToContainerPoint(nw);
        var sePointOnCanvas = map.latLngToContainerPoint(se); // Capture the south-east and north-west coordinates of the rectangle to obtain the entire rectangle
    

        var onScreenCropX = nwPointOnCanvas.x;
        var onScreenCropY = nwPointOnCanvas.y;
        var onScreenCropWidthPx = sePointOnCanvas.x - onScreenCropX;
        var onScreenCropHeightPx = sePointOnCanvas.y - onScreenCropY;

        // Error verifying that the area to be cropped on the HTML canvas is valid (e.g. if the selected area is partially or totally off-screen).
        if (onScreenCropWidthPx <= 0 || onScreenCropHeightPx <= 0 || isNaN(onScreenCropWidthPx) || isNaN(onScreenCropHeightPx)) {
            return displayError('Area selected on screen too small or invalid for cropping.');
        }

        // Create a temporary canvas for trimming
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
            // Error that appears if there was an error during trimming
            return displayError(`Technical error when cropping image: ${e.message}`);
        }

        // Resize to target resolution
        var finalResizedCanvas = document.createElement('canvas');
        finalResizedCanvas.width = finalImageWidthPx;
        finalResizedCanvas.height = finalImageHeightPx;
        var finalCtx = finalResizedCanvas.getContext('2d');
        finalCtx.drawImage(tempCroppedCanvas, 0, 0, finalResizedCanvas.width, finalResizedCanvas.height);

        // Base64 encoding
        var imgData;
        try {
            imgData = finalResizedCanvas.toDataURL('image/png');
        } catch (e) {
            // If there's a problem converting the image to a URL
            return displayError(`Technical error during image conversion: ${e.message}`);
        }

        // Send to backend (Flask API on Python)
        fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: imgData })
        })
        .then(response => response.json())
        .then(data => {
            let resultText = `<h2>Results:</h2>`;
            resultText += `<p>Number of process tiles: ${data.total_tiles_processed}</p>`;
            resultText += `<p><i>The selected area is ${finalImageWidthPx}px x ${finalImageHeightPx}px which is ${finalImageWidthPx*10}m by ${finalImageHeightPx*10}m</i></p>`;

            // Dominant class
            if (data.dominant_class_info && data.dominant_class_info.class !== 'N/A') {
                resultText += `<p><strong>Dominant class :</strong> ${data.dominant_class_info.class} (${data.dominant_class_info.count} tiles)</p>`;
            } else {
                resultText += `<p>No dominant class determined.</p>`;
            }

            // Full class breakdown
            if (data.class_distribution) {
                resultText += `<h3>Breakdown of classes :</h3><ul>`;
                for (const [className, percentage] of Object.entries(data.class_distribution)) {
                    resultText += `<li>${className} : ${percentage}</li>`;
                }
                resultText += `</ul>`;
            }

            document.getElementById("ResPred").innerHTML = resultText;
        })
            
    })
});
