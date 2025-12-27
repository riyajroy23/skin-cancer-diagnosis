const imageInput = document.getElementById("imageInput");
const previewImg = document.getElementById("preview");
const resultBox = document.getElementById("result");
const resources = document.getElementById("resources");
const analyzeBtn = document.getElementById("analyzeBtn");

imageInput.addEventListener("change", async () => {
    const file = imageInput.files[0];
    if (!file) return;

    // Show preview
    previewImg.src = URL.createObjectURL(file);
    previewImg.classList.remove("hidden");
})

analyzeBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];
  if (!file) return;

  // Show preview
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.remove("hidden");

  const formData = new FormData();
  formData.append("image", file);

  resultBox.textContent = "Analyzing image…";
  resultBox.classList.remove("hidden");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const data = await response.json();

    resultBox.innerHTML = `
        <strong>Prediction:</strong> ${data.name}<br />
        <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
        <p style="margin-top: 0.75rem; font-weight: 400;">
            ${data.description}
        </p>
    `;

    resources.classList.remove("hidden");
  } catch (err) {
    resultBox.textContent = "Error processing image. Please try again.";
    console.error(err);
  }
});