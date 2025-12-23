const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview");
const resultBox = document.getElementById("result");

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Show preview
  const reader = new FileReader();
  reader.onload = () => {
    previewImg.src = reader.result;
    previewImg.style.display = "block";
  };
  reader.readAsDataURL(file);

  // Send image to backend
  const formData = new FormData();
  formData.append("image", file);

  resultBox.textContent = "Analyzing image…";

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
      <strong>Prediction:</strong> ${data.label}<br />
      <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
    `;
  } catch (err) {
    resultBox.textContent = "Error processing image. Please try again.";
    console.error(err);
  }
});
