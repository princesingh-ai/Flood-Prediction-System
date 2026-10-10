// =====================================
// AI FLOOD PREDICTION SYSTEM
// =====================================

// Automatically detect environment:
// - Local dev (Live Server): points to local FastAPI on http://127.0.0.1:8000
// - Vercel / Production: uses relative path ("") on the same domain
const isLocal = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";
const API_URL = isLocal && window.location.port !== "8000" && window.location.port !== "" ? "http://127.0.0.1:8000" : "";

// All features used by the trained model
const features = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors"
];

// =====================================
// MAP SETUP
// =====================================

const map = L.map("map").setView([20, 0], 2);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19
}).addTo(map);

let locationMarker = null;

// =====================================
// LOCATION SEARCH
// =====================================

const locationInput = document.getElementById("locationInput");
const searchButton = document.getElementById("searchButton");
const locationStatus = document.getElementById("locationStatus");

document.getElementById("environmentStatus").textContent =
    "Awaiting a real environmental data source";

searchButton.addEventListener("click", searchLocation);

locationInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        searchLocation();
    }
});

async function searchLocation() {
    const place = locationInput.value.trim();

    if (!place) {
        locationStatus.textContent = "Please enter a location.";
        return;
    }

    locationStatus.textContent = "Searching for location...";

    try {
        const url =
            "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" +
            encodeURIComponent(place);

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error("Location search service is unavailable.");
        }

        const results = await response.json();

        if (results.length === 0) {
            locationStatus.textContent = "Location not found. Try another name.";
            return;
        }

        const result = results[0];
        const lat = Number(result.lat);
        const lon = Number(result.lon);

        map.setView([lat, lon], 10);

        if (locationMarker) {
            map.removeLayer(locationMarker);
        }

        locationMarker = L.marker([lat, lon])
            .addTo(map)
            .bindPopup(result.display_name)
            .openPopup();

        document.getElementById("selectedLocation").textContent =
            result.display_name;

        document.getElementById("latitude").textContent = lat.toFixed(5);
        document.getElementById("longitude").textContent = lon.toFixed(5);

        locationStatus.textContent = "Location found successfully.";

    } catch (error) {
        locationStatus.textContent =
            "Could not search for this location. Check your internet connection.";
        console.error(error);
    }
}

// =====================================
// CREATE INPUT FIELDS
// =====================================

const featureInputs = document.getElementById("featureInputs");

features.forEach(function (feature) {
    const label = document.createElement("label");

    label.className = "feature-field";
    label.textContent = feature;

    const input = document.createElement("input");

    input.type = "number";
    input.name = feature;
    input.id = feature;
    input.required = true;
    input.step = "any";
    input.min = "0";
    input.placeholder = "Enter value";

    label.appendChild(input);
    featureInputs.appendChild(label);
});

// =====================================
// FLOOD PREDICTION
// =====================================

const predictionForm = document.getElementById("predictionForm");
const predictionResult = document.getElementById("predictionResult");

predictionForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const inputValues = {};

    for (const feature of features) {
        const input = document.getElementById(feature);
        const value = input.value.trim();

        if (value === "" || !Number.isFinite(Number(value))) {
            predictionResult.textContent =
                "Please enter a valid number for " + feature + ".";
            return;
        }

        inputValues[feature] = Number(value);
    }

    predictionResult.textContent = "Generating model estimate...";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                features: inputValues
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof result.detail === "string"
                    ? result.detail
                    : JSON.stringify(result.detail)
            );
        }

        const estimate = Number(result.estimated_flood_probability);

predictionResult.className = "result";

predictionResult.innerHTML = `
    <h3>📊 Model Prediction Result</h3>

    <p class="prediction-value">
        Model output: ${estimate.toFixed(4)}
    </p>

    <div class="prediction-bar">
        <div class="prediction-fill"></div>
    </div>

    <p class="prediction-note">
        ${result.note || "This is a model estimate."}
    </p>

    <p class="prediction-warning">
        ⚠️ This output is not a validated real-world flood probability
        and should not be used for emergency or safety decisions.
    </p>
`;

const fill = predictionResult.querySelector(".prediction-fill");

// Display the output visually; this is NOT a calibrated risk score.
fill.style.width = `${Math.max(0, Math.min(100, estimate * 100))}%`;

    } catch (error) {
        predictionResult.textContent =
            "Prediction failed: " + error.message;

        console.error(error);
    }
});

// =====================================
// CHECK BACKEND CONNECTION
// =====================================

const healthCheckUrl = API_URL ? `${API_URL}/` : `/health`;

fetch(healthCheckUrl)
    .then(function (response) {
        if (!response.ok) {
            throw new Error("Backend is unavailable.");
        }
        return response.json();
    })
    .then(function (data) {
        console.log("Backend connected:", data);
    })
    .catch(function (error) {
        console.error("Backend connection failed:", error);
        locationStatus.textContent =
            "Backend is not connected. Start the FastAPI server.";
    });