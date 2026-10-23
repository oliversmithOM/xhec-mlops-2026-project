"use strict";

const form = document.querySelector("#prediction-form");
const panel = document.querySelector("#result-panel");
const errorBox = document.querySelector("#form-error");
const fingerprint = document.querySelector("#model-fingerprint");
const exportButton = document.querySelector("#export-button");
const exampleButton = document.querySelector("#example-button");
const resetButton = document.querySelector("#reset-button");
const predictButton = document.querySelector("#predict-button");
const example = {
  sex: "M",
  length: 0.455,
  diameter: 0.365,
  height: 0.095,
  whole_weight: 0.514,
  shucked_weight: 0.2245,
  viscera_weight: 0.101,
  shell_weight: 0.15,
};
let savedResult = null;
let activeHash = "";
let busy = false;

function connection(online) {
  document.querySelector("#connection").dataset.state = online
    ? "online"
    : "offline";
  document.querySelector("#connection-label").textContent = online
    ? "Model connected"
    : "Model unavailable";
}

function showHash(hash) {
  fingerprint.textContent = hash ? `${hash.slice(0, 12)}…` : "Unavailable";
  fingerprint.title = hash;
}

function clearResult(label = "Ready when you are") {
  savedResult = null;
  exportButton.disabled = true;
  panel.dataset.state = "empty";
  document.querySelector("#age-display").hidden = true;
  document.querySelector("#result-heading").textContent =
    "Every shell has a story.";
  document.querySelector("#result-description").textContent =
    "Start with the example or enter your own measurements to estimate its age.";
  document.querySelector("#result-badge").textContent = label;
  errorBox.hidden = true;
  form
    .querySelectorAll("[aria-invalid]")
    .forEach((input) => input.removeAttribute("aria-invalid"));
  showHash(activeHash);
}

function setBusy(value) {
  busy = value;
  document.querySelector("#measurement-fields").disabled = value;
  predictButton.disabled = value;
  exampleButton.disabled = value;
  resetButton.disabled = value;
  panel.setAttribute("aria-busy", String(value));
  document.querySelector("#predict-label").textContent = value
    ? "Estimating…"
    : "Estimate age";
}

async function checkHealth() {
  try {
    const response = await fetch("/health", {
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw new Error("Unavailable");
    const data = await response.json();
    if (data.status !== "ok" || typeof data.model_sha256 !== "string")
      throw new Error("Invalid health response");
    activeHash = data.model_sha256;
    connection(true);
    if (!savedResult) showHash(activeHash);
  } catch {
    connection(false);
    if (!savedResult) showHash("");
  }
}

form.addEventListener("input", () => {
  if (!busy) clearResult("Measurements updated");
});

exampleButton.addEventListener("click", () => {
  for (const [key, value] of Object.entries(example))
    form.elements.namedItem(key).value = value;
  clearResult("Example loaded");
});

resetButton.addEventListener("click", () => {
  form.querySelectorAll('input[type="number"]').forEach((input) => {
    input.value = "";
  });
  form.elements.namedItem("sex").value = "M";
  clearResult();
  document.querySelector("#length").focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !form.reportValidity()) return;
  const values = new FormData(form);
  const payload = Object.fromEntries(
    [...values].map(([key, value]) => [
      key,
      key === "sex" ? value : Number(value),
    ]),
  );
  clearResult("Estimating…");
  setBusy(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) {
      if (response.status === 422) {
        const body = await response.json();
        const errors = Array.isArray(body.detail) ? body.detail : [];
        errors.forEach((error) => {
          const input = form.elements.namedItem(error.loc?.at(-1));
          if (input instanceof HTMLElement)
            input.setAttribute("aria-invalid", "true");
        });
        throw new Error(
          "Please check your measurements. Use finite, nonnegative numbers in the original dataset scale.",
        );
      }
      throw new Error(
        "The model could not complete this estimate. Please try again.",
      );
    }
    const result = await response.json();
    if (
      !Number.isFinite(result.age_years) ||
      typeof result.model_sha256 !== "string"
    )
      throw new Error(
        "The model returned an unexpected response. Please try again.",
      );
    savedResult = {
      measurements: payload,
      ...result,
      estimated_at: new Date().toISOString(),
    };
    activeHash = result.model_sha256;
    connection(true);
    showHash(activeHash);
    panel.dataset.state = "complete";
    document.querySelector("#result-badge").textContent = "Estimate complete";
    document.querySelector("#result-heading").textContent = "Estimated age";
    document.querySelector("#age-value").textContent =
      result.age_years.toFixed(2);
    document.querySelector("#age-display").hidden = false;
    document.querySelector("#result-description").textContent =
      result.age_years < 0
        ? "This negative estimate is not biologically plausible. Check your inputs and the dataset scale."
        : "Based on your measurements and the fitted linear-regression pipeline. Read the model limitations below.";
    exportButton.disabled = false;
    if (window.matchMedia("(max-width: 760px)").matches) {
      panel.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "start",
      });
    }
  } catch (error) {
    const networkFailure =
      error instanceof TypeError ||
      error.name === "TimeoutError" ||
      error.name === "AbortError";
    errorBox.textContent = networkFailure
      ? "We couldn’t reach the model in time. Check that the service is running, then try again."
      : error.message;
    errorBox.hidden = false;
    document.querySelector("#result-badge").textContent =
      "Estimate unavailable";
    if (networkFailure) connection(false);
  } finally {
    setBusy(false);
  }
});

exportButton.addEventListener("click", () => {
  if (!savedResult) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(savedResult, null, 2)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "abalone-estimate.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

checkHealth();
