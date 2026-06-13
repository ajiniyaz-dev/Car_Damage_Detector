const API_BASE =
    document.querySelector('meta[name="api-base"]')?.content?.trim()
    || "https://car-damage-detector-5z60.onrender.com";

console.log("API_BASE:", API_BASE);

const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const detectButton = document.getElementById("detectButton");
const loadingMessage = document.getElementById("loadingMessage");
const originalSection = document.getElementById("originalSection");
const originalPreview = document.getElementById("originalPreview");
const detectionSection = document.getElementById("detectionSection");
const detectionPreview = document.getElementById("detectionPreview");
const analysisSection = document.getElementById("analysisSection");
const analysisContent = document.getElementById("analysisContent");

let previewObjectUrl = null;

function showSection(section) {
    section.classList.remove("hidden");
}

function hideSection(section) {
    section.classList.add("hidden");
}

function setLoading(isLoading) {
    detectButton.disabled = isLoading;

    if (isLoading) {
        loadingMessage.classList.remove("hidden");
        return;
    }

    loadingMessage.classList.add("hidden");
}

function formatConfidence(confidence) {
    return `${Math.round(confidence * 100)}%`;
}

function getSeverityClass(severity) {
    return `severity severity-${severity.toLowerCase()}`;
}

function clearResults() {
    hideSection(detectionSection);
    hideSection(analysisSection);
    detectionPreview.removeAttribute("src");
    analysisContent.innerHTML = "";
}

function renderAnalysis(data) {
    const damageBadgeClass = data.damage_found
        ? "badge badge-danger"
        : "badge badge-success";

    const damageBadgeText = data.damage_found
        ? "Damage Found"
        : "No Damage";

    let damagesHtml = "";

    if (data.detections.length > 0) {
        damagesHtml = `
            <h3 class="section-heading">Detected Damages</h3>
            <div class="damage-list">
                ${data.detections.map((detection, index) => `
                    <article class="damage-item">
                        <h3>Damage #${index + 1}</h3>
                        <div class="damage-meta">
                            <span>
                                Confidence:
                                ${formatConfidence(detection.confidence)}
                            </span>

                            <span class="${getSeverityClass(detection.severity)}">
                                Severity:
                                ${detection.severity}
                            </span>
                        </div>
                    </article>
                `).join("")}
            </div>
        `;
    } else {
        damagesHtml = `
            <p class="no-damages">
                No damage regions were detected in this image.
            </p>
        `;
    }

    analysisContent.innerHTML = `
        <div class="stats-row">

            <div class="stat-item">
                <span>Damage Found:</span>
                <span class="${damageBadgeClass}">
                    ${damageBadgeText}
                </span>
            </div>

            <div class="stat-item">
                <span>Damage Count:</span>
                <span class="badge badge-neutral">
                    ${data.damage_count}
                </span>
            </div>

        </div>

        ${damagesHtml}
    `;
}

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {

        fileName.textContent = "No file selected";

        hideSection(originalSection);

        clearResults();

        return;
    }

    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
    }

    previewObjectUrl = URL.createObjectURL(file);

    originalPreview.src = previewObjectUrl;

    fileName.textContent = file.name;

    showSection(originalSection);

    clearResults();
});

detectButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {

        alert("Please select an image.");

        return;
    }

    const formData = new FormData();

    formData.append(
        "file",
        file
    );

    setLoading(true);

    try {

        const response = await fetch(
            `${API_BASE}/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {

            throw new Error(
                `Prediction failed with status ${response.status}`
            );
        }

        const data = await response.json();

        detectionPreview.src =
            `${API_BASE}${data.prediction_image_url}?t=${Date.now()}`;

        showSection(detectionSection);

        renderAnalysis(data);

        showSection(analysisSection);

    } catch (error) {

        console.error(error);

        alert(
            "Failed to analyze the image. Please try again."
        );

    } finally {

        setLoading(false);
    }
});