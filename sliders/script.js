// ---------- element refs ----------
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const iterationIndicator = document.getElementById("iterationIndicator");
const controls = document.getElementById("controls");
const referenceSection = document.getElementById("referenceSection");
const referenceImg = document.getElementById("referenceImg");
const previewSection = document.getElementById("previewSection");
const previewImg = document.getElementById("previewImg");
const previewBrick = document.querySelector(".preview-brick");
const sliderPanel = document.getElementById("sliderPanel");
const sliderList = document.getElementById("sliderList");
const renderBtn = document.getElementById("renderBtn");
const historySection = document.getElementById("historySection");
const historyList = document.getElementById("historyList");
const descriptionToggle = document.getElementById("descriptionToggle");
const descriptionOverlay = document.getElementById("descriptionOverlay");
const descriptionClose = document.getElementById("descriptionClose");
const tutorialToggle = document.getElementById("tutorialToggle");
const tutorialOverlay = document.getElementById("tutorialOverlay");
const tutorialClose = document.getElementById("tutorialClose");

let currentIteration = null;
let sliderRange = [0, 1];
let sliderLabels = [];
let sliderState = [];
let sliderThumbnails = [];
let historyEntries = [];
const safetyIndex = new Map();
let safetyRefreshPromise = null;
let safetyRefreshQueued = false;

const HISTORY_EPSILON = 1e-4;

function setBodyScrollLock(locked) {
  document.body.classList.toggle("no-scroll", Boolean(locked));
}

function openDescriptionOverlay() {
  if (!descriptionOverlay) return;
  descriptionOverlay.classList.remove("hidden");
  setBodyScrollLock(true);
}

function closeDescriptionOverlay() {
  if (!descriptionOverlay) return;
  descriptionOverlay.classList.add("hidden");
  setBodyScrollLock(false);
}

function openTutorialOverlay() {
  if (!tutorialOverlay) return;
  tutorialOverlay.classList.remove("hidden");
  setBodyScrollLock(true);
}

function closeTutorialOverlay() {
  if (!tutorialOverlay) return;
  tutorialOverlay.classList.add("hidden");
  setBodyScrollLock(false);
}

if (descriptionToggle) {
  descriptionToggle.addEventListener("click", openDescriptionOverlay);
}
if (descriptionClose) {
  descriptionClose.addEventListener("click", closeDescriptionOverlay);
}
if (descriptionOverlay) {
  descriptionOverlay.addEventListener("click", (event) => {
    if (event.target === descriptionOverlay) {
      closeDescriptionOverlay();
    }
  });
}

if (tutorialToggle) {
  tutorialToggle.addEventListener("click", openTutorialOverlay);
}
if (tutorialClose) {
  tutorialClose.addEventListener("click", closeTutorialOverlay);
}
if (tutorialOverlay) {
  tutorialOverlay.addEventListener("click", (event) => {
    if (event.target === tutorialOverlay) {
      closeTutorialOverlay();
    }
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (descriptionOverlay && !descriptionOverlay.classList.contains("hidden")) {
      closeDescriptionOverlay();
    }
    if (tutorialOverlay && !tutorialOverlay.classList.contains("hidden")) {
      closeTutorialOverlay();
    }
  }
});

function clampSliderValue(value) {
  const min = Number(sliderRange[0] ?? 0);
  const max = Number(sliderRange[1] ?? 1);
  if (!Number.isFinite(value)) return min;
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function canonicalizeImage(src) {
  if (typeof src !== "string" || !src) return null;
  return src.split("?")[0];
}

function canonicalizeImageSrc(src) {
  return canonicalizeImage(src);
}

function normalizeImageEntry(entry) {
  if (!entry) {
    return { url: null, canonical: null, isSafe: null };
  }
  if (typeof entry === "string") {
    const canonical = canonicalizeImageSrc(entry);
    return { url: entry, canonical, isSafe: null };
  }
  if (typeof entry === "object") {
    const rawUrl = typeof entry.url === "string"
      ? entry.url
      : typeof entry.src === "string"
        ? entry.src
        : typeof entry.path === "string"
          ? entry.path
          : null;
    const fallbackPath = typeof entry.basename === "string"
      ? (entry.url && String(entry.url).startsWith("/outputs") ? `/outputs/${entry.basename}` : `/slots/${entry.basename}`)
      : null;
    const canonicalSource = rawUrl || fallbackPath;
    const canonical = canonicalizeImageSrc(canonicalSource);
    let isSafe;
    if ("is_safe" in entry) {
      if (entry.is_safe === null) {
        isSafe = null;
      } else if (typeof entry.is_safe === "boolean") {
        isSafe = entry.is_safe;
      }
    }
    if (isSafe === undefined && "isSafe" in entry) {
      if (entry.isSafe === null) {
        isSafe = null;
      } else if (typeof entry.isSafe === "boolean") {
        isSafe = entry.isSafe;
      }
    }
    if (isSafe === undefined && "safe" in entry) {
      if (entry.safe === null) {
        isSafe = null;
      } else if (typeof entry.safe === "boolean") {
        isSafe = entry.safe;
      }
    }
    if (isSafe === undefined && typeof entry.nsfw === "boolean") {
      isSafe = !entry.nsfw;
    }
    if (isSafe === undefined) {
      isSafe = null;
    }
    return {
      url: rawUrl || canonical,
      canonical,
      isSafe,
    };
  }
  return { url: null, canonical: null, isSafe: null };
}

function rememberImageSafety(canonical, isSafe) {
  if (!canonical) return;
  if (isSafe === null || isSafe === undefined) {
    safetyIndex.delete(canonical);
    return;
  }
  safetyIndex.set(canonical, Boolean(isSafe));
}

function markImageSafetyUnknown(canonical, brick) {
  if (!canonical) return;
  safetyIndex.delete(canonical);
  if (brick) {
    applySafetyOverlay(brick, null);
  }
}

function applySafetyOverlay(brick, isSafe = true) {
  if (!brick) return;
  const overlay = brick.querySelector(".nsfw-overlay");
  if (isSafe === true) {
    brick.classList.remove("nsfw-flagged", "nsfw-pending");
    if (overlay) overlay.remove();
    return;
  }
  if (isSafe === false) {
    brick.classList.remove("nsfw-pending");
    brick.classList.add("nsfw-flagged");
    if (!overlay) {
      const mask = document.createElement("div");
      mask.className = "nsfw-overlay";
      mask.innerHTML = "<span>Blurred for Safety</span>";
      brick.appendChild(mask);
    }
    return;
  }
  brick.classList.add("nsfw-pending");
  brick.classList.remove("nsfw-flagged");
  if (overlay) overlay.remove();
}

function syncSafetyOverlays() {
  const previewCanonical = previewImg?.dataset?.src || null;
  const previewDatasetSafe = previewImg?.dataset?.safe;
  let previewState = null;
  if (previewDatasetSafe === "true") previewState = true;
  else if (previewDatasetSafe === "false") previewState = false;
  if (previewCanonical && safetyIndex.has(previewCanonical)) {
    previewState = safetyIndex.get(previewCanonical);
  }
  if (previewCanonical) {
    applySafetyOverlay(previewBrick, previewState);
  }
  if (historyList) {
    historyList.querySelectorAll(".history-thumb").forEach((thumb) => {
      const img = thumb.querySelector("img");
      const canonical = img?.dataset?.src || null;
      const safeAttr = img?.dataset?.safe;

      let state = null;
      if (safeAttr === "true") state = true;
      else if (safeAttr === "false") state = false;
      if (canonical) {
        const indexed = safetyIndex.has(canonical) ? safetyIndex.get(canonical) : null;
        if (indexed !== null && indexed !== undefined) {
          state = indexed;
        }
      }
      applySafetyOverlay(thumb, state);
    });
  }
}

async function refreshSafetyFromServer() {
  if (safetyRefreshPromise) {
    safetyRefreshQueued = true;
    return safetyRefreshPromise;
  }

  safetyRefreshPromise = (async () => {
    try {
      const resp = await fetch("/api/images");
      if (!resp.ok) return;
      const data = await resp.json();
      const entries = Array.isArray(data.images) ? data.images : [];
      for (const entry of entries) {
        const normalized = normalizeImageEntry(entry);
        if (!normalized.canonical) continue;
        rememberImageSafety(normalized.canonical, normalized.isSafe);
      }
      syncSafetyOverlays();
    } catch (err) {
      console.warn("Safety refresh failed", err);
    }
  })();

  const currentPromise = safetyRefreshPromise;
  try {
    await currentPromise;
  } finally {
    safetyRefreshPromise = null;
    if (safetyRefreshQueued) {
      safetyRefreshQueued = false;
      return refreshSafetyFromServer();
    }
  }

  return currentPromise;
}

function formatHistoryTimestamp(ts) {
  if (!Number.isFinite(ts)) return undefined;
  try {
    return new Date(ts * 1000).toLocaleTimeString();
  } catch (err) {
    return undefined;
  }
}

function resolveHistoryLabel(entry, fallback) {
  if (!entry) return fallback;
  if (entry.label) return entry.label;
  if (typeof entry.timestamp === "number") {
    const formatted = formatHistoryTimestamp(entry.timestamp);
    if (formatted) return formatted;
  }
  return fallback;
}

function buildHistoryEntry(raw) {
  if (!raw || !Array.isArray(raw.x)) return null;
  const vector = raw.x.map((value) => Number(value));
  if (!vector.length || vector.some((v) => Number.isNaN(v))) return null;
  const timestamp = typeof raw.timestamp === "number" ? raw.timestamp : Date.now() / 1000;
  const normalizedImage = normalizeImageEntry({
    url: raw.image,
    is_safe: raw.is_safe,
    basename: raw.image ? raw.image.split("/").pop() : undefined,
  });
  return {
    x: vector,
    similarity: typeof raw.similarity === "number" ? raw.similarity : undefined,
    image: normalizedImage.canonical,
    isSafe: normalizedImage.isSafe,
    timestamp,
    label: formatHistoryTimestamp(timestamp),
  };
}

function formatScalarValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "0";
  const fixed = numeric.toFixed(3);
  return Number(fixed).toString();
}

function describeNonZeroWeights(entry) {
  if (!entry || !Array.isArray(entry.x)) return "";
  const parts = [];
  entry.x.forEach((value, idx) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return;
    if (Math.abs(numeric) <= HISTORY_EPSILON) return;
    const label = sliderLabels[idx] || `Slider ${idx + 1}`;
    parts.push(`${label}: ${formatScalarValue(numeric)}`);
  });
  if (!parts.length) {
    return "All sliders at 0";
  }
  return parts.join(" · ");
}

function hydrateHistoryFromPayload(payload) {
  const entries = Array.isArray(payload)
    ? payload.map((item) => buildHistoryEntry(item)).filter(Boolean)
    : [];
  historyEntries = entries.slice(0, 10);
  renderHistory();
}

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
  const normalized = normalizeImageEntry(src);
  if (!normalized.canonical) {
    previewImg.removeAttribute("src");
    previewSection.classList.add("hidden");
    return;
  }
  const canonical = normalized.canonical;
  previewImg.dataset.src = canonical;
  if (normalized.isSafe === null || normalized.isSafe === undefined) {
    delete previewImg.dataset.safe;
  } else {
    previewImg.dataset.safe = String(normalized.isSafe);
  }
  previewImg.src = `${canonical}?t=${Date.now()}`;
  previewSection.classList.remove("hidden");

  if (normalized.isSafe === null || normalized.isSafe === undefined) {
    markImageSafetyUnknown(canonical, previewBrick);
  } else {
    rememberImageSafety(canonical, normalized.isSafe);
    applySafetyOverlay(previewBrick, normalized.isSafe);
  }
  syncSafetyOverlays();
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
  });
  sliderList.querySelectorAll("input.slider-number-input").forEach((input) => {
    const idx = Number(input.dataset.index);
    const nextVal = sliderState[idx] ?? sliderRange[0];
    input.value = Number(nextVal).toFixed(3);
  });
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

  const controlsWrap = document.createElement("div");
  controlsWrap.className = "slider-head-controls";

  const rangeInput = document.createElement("input");
  rangeInput.type = "range";
  rangeInput.min = sliderRange[0];
  rangeInput.max = sliderRange[1];
  rangeInput.step = Math.max(0.001, (sliderRange[1] - sliderRange[0]) / 200);
  rangeInput.value = value;
  rangeInput.dataset.index = index;

  const numberInput = document.createElement("input");
  numberInput.type = "number";
  numberInput.className = "slider-number-input";
  numberInput.min = sliderRange[0];
  numberInput.max = sliderRange[1];
  numberInput.step = Math.max(0.001, (sliderRange[1] - sliderRange[0]) / 200);
  numberInput.value = Number(value).toFixed(3);
  numberInput.dataset.index = index;

  controlsWrap.appendChild(numberInput);

  head.appendChild(name);
  head.appendChild(controlsWrap);

  numberInput.addEventListener("input", (event) => {
    const raw = String(event.target.value || "").trim();
    if (raw === "" || raw === "-" || raw === "." || raw === "-." ) {
      return;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    const clamped = clampSliderValue(parsed);
    sliderState[index] = clamped;
    rangeInput.value = clamped;
    if (renderBtn) renderBtn.disabled = sliderState.length === 0;
  });

  numberInput.addEventListener("blur", () => {
    const committed = sliderState[index] ?? sliderRange[0];
    numberInput.value = Number(committed).toFixed(3);
  });

  numberInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      numberInput.blur();
    }
  });

  rangeInput.addEventListener("input", (event) => {
    const val = Number(event.target.value);
    const clamped = clampSliderValue(val);
    sliderState[index] = clamped;
    numberInput.value = clamped.toFixed(3);
    if (renderBtn) renderBtn.disabled = sliderState.length === 0;
  });

  detail.appendChild(head);
  detail.appendChild(rangeInput);
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
    title.textContent = resolveHistoryLabel(entry, `Attempt ${historyEntries.length - index}`);

    const vector = document.createElement("span");
    vector.className = "history-entry-vector";
    vector.textContent = describeNonZeroWeights(entry);

    button.appendChild(title);
    button.appendChild(vector);

    button.addEventListener("click", () => {
      sliderState = entry.x.slice();
      updateSliderInputsFromState();
      renderFromSliders({ source: "history" });
    });

    historyList.appendChild(button);
  });

  historySection.classList.remove("hidden");
  syncSafetyOverlays();
}

function addHistoryEntry(payload) {
  const entryFromPayload = buildHistoryEntry(payload);
  const timestamp = Date.now() / 1000;
  const entry = entryFromPayload || {
    x: sliderState.slice(),
    similarity: payload && typeof payload.similarity === "number" ? payload.similarity : undefined,
    image: canonicalizeImage(payload && payload.image),
    timestamp,
    label: formatHistoryTimestamp(timestamp),
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
    hydrateHistoryFromPayload(data.history);
    if (historyEntries.length) {
      sliderState = historyEntries[0].x.slice();
      updateSliderInputsFromState();
      if (!data.latest_image) {
        updatePreviewImage(historyEntries[0].image || null);
      }
    }

    refreshSafetyFromServer();

    if (controls) controls.classList.add("hidden");
    if (sliderPanel && sliderLabels.length) sliderPanel.classList.remove("hidden");
    if (historyEntries.length) {
      statusEl.textContent = "Restored previous sliders. Adjust and render when ready.";
    } else {
      statusEl.textContent = sliderLabels.length ? "Adjust each slider, then render." : "No sliders available.";
    }
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

async function renderFromSliders(options = {}) {
  if (!Array.isArray(sliderState) || !sliderState.length) return;
  const source = options && options.source ? String(options.source) : null;
  const isHistoryRender = source === "history";
  try {
    if (renderBtn) renderBtn.disabled = true;
    statusEl.textContent = isHistoryRender ? "Re-rendering saved sliders…" : "Rendering…";

    const resp = await fetch("/api/slider/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vector: sliderState,
        normalize: true,
        record_history: !isHistoryRender,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error((data && data.error) || "Failed to render");
    }

    if (Array.isArray(data.x)) {
      sliderState = data.x.slice();
      updateSliderInputsFromState();
    }

    if (data.iteration !== undefined) {
      const iterValue = Number(data.iteration);
      if (Number.isFinite(iterValue)) {
        setIterationDisplay(iterValue);
      }
    }

    const normalizedPreview = normalizeImageEntry({
      url: data.image,
      is_safe: data.is_safe,
      basename: data.image ? data.image.split("/").pop() : undefined,
    });
    if (normalizedPreview.canonical) {
      if (normalizedPreview.isSafe === null || normalizedPreview.isSafe === undefined) {
        markImageSafetyUnknown(normalizedPreview.canonical, previewBrick);
      } else {
        rememberImageSafety(normalizedPreview.canonical, normalizedPreview.isSafe);
      }
    }
    updatePreviewImage(normalizedPreview);
    if (normalizedPreview.canonical) {
      refreshSafetyFromServer();
    }
    if (!isHistoryRender) {
      addHistoryEntry(data);
    }
    statusEl.textContent = isHistoryRender ? "History render complete." : "Render complete.";
  } catch (err) {
    const msg = err && err.message ? err.message : err;
    statusEl.textContent = `Error: ${msg}`;
  } finally {
    if (renderBtn) renderBtn.disabled = sliderState.length === 0;
  }
}

if (renderBtn) {
  renderBtn.addEventListener("click", renderFromSliders);
}
