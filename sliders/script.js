// ---------- element refs ----------
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const iterationIndicator = document.getElementById("iterationIndicator");
const controls = document.getElementById("controls");
const referenceSection = document.getElementById("referenceSection");
const referenceImg = document.getElementById("referenceImg");
const previewSection = document.getElementById("previewSection");
const previewImg = document.getElementById("previewImg");
const sliderPanel = document.getElementById("sliderPanel");
const sliderList = document.getElementById("sliderList");
const renderBtn = document.getElementById("renderBtn");
const historySection = document.getElementById("historySection");
const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

let currentIteration = null;
let sliderRange = [0, 1];
let sliderLabels = [];
let sliderState = [];
let sliderThumbnails = [];
let historyEntries = [];

if (iterationIndicator) iterationIndicator.classList.add("hidden");
if (referenceSection) referenceSection.classList.add("hidden");
if (sliderPanel) sliderPanel.classList.add("hidden");
if (historySection) historySection.classList.add("hidden");

function setIterationDisplay(value, { commit = true } = {}) {
  if (!iterationIndicator) return;
  if (value === null || value === undefined || value === "") {
    if (commit) currentIteration = null;
    iterationIndicator.textContent = "";
    iterationIndicator.classList.add("hidden");
    return;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return;
  if (commit) currentIteration = numeric;
  iterationIndicator.textContent = `Iteration ${numeric + 1}`;
  iterationIndicator.classList.remove("hidden");
}

function updateReferenceImage(src) {
  if (!referenceSection || !referenceImg) return;
  if (!src) {
    referenceImg.removeAttribute("src");
    referenceSection.classList.add("hidden");
    return;
  }
  const canonical = src.split("?")[0];
  referenceImg.dataset.src = canonical;
  referenceImg.src = `${canonical}?v=${Date.now()}`;
  referenceSection.classList.remove("hidden");
}

function updatePreviewImage(src) {
  if (!previewSection || !previewImg) return;
  if (!src) {
    previewImg.removeAttribute("src");
    previewSection.classList.add("hidden");
    return;
  }
  const canonical = src.split("?")[0];
  previewImg.dataset.src = canonical;
  previewImg.src = `${canonical}?t=${Date.now()}`;
  previewSection.classList.remove("hidden");
}

function formatSliderValue(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function updateSliderInputsFromState() {
  if (!sliderList) return;
  sliderList.querySelectorAll("input[type=range]").forEach((input) => {
    const idx = Number(input.dataset.index);
    const nextVal = sliderState[idx] ?? sliderRange[0];
    input.value = nextVal;
    const valueBadge = sliderList.querySelector(`.slider-value[data-index="${idx}"]`);
    if (valueBadge) valueBadge.textContent = formatSliderValue(nextVal);
  });
}

function handleSliderChange(idx, rawValue) {
  sliderState[idx] = Number(rawValue);
}

function createSliderRow(label, value, index, thumbnailUrl) {
  const row = document.createElement("div");
  row.className = "slider-row";

  const body = document.createElement("div");
  body.className = "slider-row-body";
  const fallbackLabel = label || `Slider ${index + 1}`;

  if (thumbnailUrl) {
    const thumbStack = document.createElement("div");
    thumbStack.className = "slider-thumb-stack";

    const thumbWrap = document.createElement("div");
    thumbWrap.className = "slider-thumb";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    const canonical = thumbnailUrl.split("?")[0];
    const cacheKey = Date.now();
    img.src = `${canonical}?thumb=${cacheKey}`;
    img.alt = `${fallbackLabel} reference`;
    thumbWrap.appendChild(img);
    thumbStack.appendChild(thumbWrap);

    const preview = document.createElement("div");
    preview.className = "slider-thumb-preview";
    const previewImg = document.createElement("img");
    previewImg.loading = "lazy";
    previewImg.decoding = "async";
    previewImg.src = `${canonical}?preview=${cacheKey}`;
    previewImg.alt = `${fallbackLabel} large preview`;
    preview.appendChild(previewImg);
    thumbStack.appendChild(preview);

    body.appendChild(thumbStack);
  }

  const detail = document.createElement("div");
  detail.className = "slider-row-detail";

  const head = document.createElement("div");
  head.className = "slider-row-head";

  const name = document.createElement("span");
  name.className = "slider-name";
  name.textContent = fallbackLabel;

  const valueBadge = document.createElement("span");
  valueBadge.className = "slider-value";
  valueBadge.dataset.index = index;
  valueBadge.textContent = formatSliderValue(value);

  head.appendChild(name);
  head.appendChild(valueBadge);

  const input = document.createElement("input");
  input.type = "range";
  input.min = sliderRange[0];
  input.max = sliderRange[1];
  input.step = Math.max(0.001, (sliderRange[1] - sliderRange[0]) / 200);
  input.value = value;
  input.dataset.index = index;

  input.addEventListener("input", (event) => {
    const val = Number(event.target.value);
    valueBadge.textContent = formatSliderValue(val);
    handleSliderChange(index, val);
    renderBtn.disabled = sliderState.length === 0;
  });

  detail.appendChild(head);
  detail.appendChild(input);
  body.appendChild(detail);
  row.appendChild(body);
  return row;
}

function buildSliderInterface(meta) {
  const labels = meta && Array.isArray(meta.labels) ? meta.labels : [];
  sliderRange = meta && Array.isArray(meta.range) && meta.range.length >= 2
    ? meta.range.slice(0, 2)
    : [0, 1];
  const defaults = meta && Array.isArray(meta.default) ? meta.default : labels.map(() => 0);
  const thumbnails = meta && Array.isArray(meta.thumbnails) ? meta.thumbnails : [];

  sliderLabels = labels;
  sliderState = defaults.slice(0, labels.length);
  sliderThumbnails = thumbnails.slice(0, labels.length);

  if (!sliderList) return;
  sliderList.innerHTML = "";
  labels.forEach((label, idx) => {
    const row = createSliderRow(
      label,
      sliderState[idx] ?? sliderRange[0],
      idx,
      sliderThumbnails[idx] || null,
    );
    sliderList.appendChild(row);
  });

  if (sliderPanel) sliderPanel.classList.toggle("hidden", labels.length === 0);
  if (renderBtn) renderBtn.disabled = labels.length === 0;
}

function renderHistory() {
  if (!historyList || !historySection) return;
  historyList.innerHTML = "";
  if (!historyEntries.length) {
    historySection.classList.add("hidden");
    return;
  }

  historyEntries.slice(0, 10).forEach((entry, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-entry";

    const title = document.createElement("span");
    title.className = "history-entry-title";
    title.textContent = entry.label || `Attempt ${historyEntries.length - index}`;

    const vector = document.createElement("span");
    vector.className = "history-entry-vector";
    vector.textContent = entry.x.map((v) => v.toFixed(2)).join(" · ");

    button.appendChild(title);
    button.appendChild(vector);

    button.addEventListener("click", () => {
      sliderState = entry.x.slice();
      updateSliderInputsFromState();
      updatePreviewImage(entry.image || null);
      statusEl.textContent = "Loaded sliders from history.";
    });

    historyList.appendChild(button);
  });

  historySection.classList.remove("hidden");
}

function addHistoryEntry(payload) {
  const canonicalImage = payload && typeof payload.image === "string"
    ? payload.image.split("?")[0]
    : null;
  const entry = {
    x: payload && Array.isArray(payload.x) ? payload.x.slice() : sliderState.slice(),
    similarity: payload && typeof payload.similarity === "number" ? payload.similarity : undefined,
    label: payload && payload.timestamp ? new Date(payload.timestamp * 1000).toLocaleTimeString() : undefined,
    image: canonicalImage,
  };
  historyEntries = [entry, ...historyEntries].slice(0, 10);
  renderHistory();
}

async function startProcess() {
  if (!startBtn) return;
  try {
    startBtn.disabled = true;
    statusEl.textContent = "Starting engine…";
    const resp = await fetch("/api/start", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error((data && data.error) || "Failed to start");
    }

    const iter = Number(data.iteration ?? data.step);
    setIterationDisplay(Number.isFinite(iter) ? iter : null);
    updateReferenceImage(data.gt_image);
    buildSliderInterface(data.slider || {});
    updatePreviewImage(data.latest_image || null);

    if (controls) controls.classList.add("hidden");
    if (sliderPanel && sliderLabels.length) sliderPanel.classList.remove("hidden");
    statusEl.textContent = sliderLabels.length ? "Adjust each slider, then render." : "No sliders available.";
  } catch (err) {
    const msg = err && err.message ? err.message : err;
    statusEl.textContent = `Error: ${msg}`;
    updateReferenceImage(null);
    if (sliderPanel) sliderPanel.classList.add("hidden");
  } finally {
    startBtn.disabled = false;
  }
}

if (startBtn) {
  startBtn.addEventListener("click", startProcess);
}

async function renderFromSliders() {
  if (!Array.isArray(sliderState) || !sliderState.length) return;
  try {
    renderBtn.disabled = true;
    statusEl.textContent = "Rendering…";

    const resp = await fetch("/api/slider/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vector: sliderState, normalize: true }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error((data && data.error) || "Failed to render");
    }

    if (Array.isArray(data.x)) {
      sliderState = data.x.slice();
      updateSliderInputsFromState();
    }

    updatePreviewImage(data.image || null);
    addHistoryEntry(data);
    statusEl.textContent = "Render complete.";
  } catch (err) {
    const msg = err && err.message ? err.message : err;
    statusEl.textContent = `Error: ${msg}`;
  } finally {
    renderBtn.disabled = sliderState.length === 0;
  }
}

if (renderBtn) {
  renderBtn.addEventListener("click", renderFromSliders);
}

if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener("click", () => {
    historyEntries = [];
    renderHistory();
    statusEl.textContent = "History cleared.";
  });
}
