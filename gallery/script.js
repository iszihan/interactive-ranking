// ---------- element refs ----------
const container = document.getElementById("container");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const iterationIndicator = document.getElementById("iterationIndicator");
const gallery = document.getElementById("gallery");
const nextBtn = document.getElementById("nextBtn");
const nextWrap = document.getElementById("nextWrap");
const stageBtn = document.getElementById("stageBtn");
const rankSection = document.getElementById("rankSection");
const controls = document.getElementById("controls");
const referenceSection = document.getElementById("referenceSection");
const referenceImg = document.getElementById("referenceImg");
const zoomSection = document.getElementById("zoomSection");
const zoomImg = document.getElementById("zoomImg");
let currentIteration = null;

const defaultStageState = {
  hasStages: false,
  nextStageReady: false,
  nextStageNumber: null,
  currentStage: 1,
  totalStages: 1,
  iterations: [],
  step: null,
};
let stageState = { ...defaultStageState };
let isGenerationInFlight = false;

function normalizeStagePayload(stage) {
  if (!stage || typeof stage !== "object") {
    return { ...defaultStageState, iterations: [] };
  }
  const normalized = {
    ...defaultStageState,
    ...stage,
  };
  normalized.iterations = Array.isArray(stage.iterations) ? [...stage.iterations] : [];
  normalized.hasStages = Boolean(stage.hasStages || normalized.iterations.length);
  normalized.nextStageReady = Boolean(normalized.hasStages && stage.nextStageReady);
  if (!Number.isFinite(normalized.currentStage)) {
    normalized.currentStage = 1;
  }
  if (!Number.isFinite(normalized.totalStages)) {
    normalized.totalStages = normalized.hasStages ? normalized.iterations.length + 1 : 1;
  }
  if ((normalized.nextStageNumber === null || normalized.nextStageNumber === undefined) && normalized.hasStages) {
    normalized.nextStageNumber = normalized.currentStage + 1;
  }
  const stepValue = Number(stage.step ?? normalized.step);
  normalized.step = Number.isFinite(stepValue) ? stepValue : null;
  return normalized;
}

function applyStagePayload(stage) {
  stageState = normalizeStagePayload(stage);
  if (currentIteration !== null && Number.isFinite(stageState.step)) {
    setIterationDisplay(stageState.step);
  }
  updateActionButtons();
}

function updateActionButtons() {
  const stageReady = stageState.hasStages && stageState.nextStageReady;
  if (nextBtn) {
    nextBtn.disabled = Boolean(isGenerationInFlight || stageReady);
  }
  if (stageBtn) {
    const shouldShow = stageReady;
    stageBtn.classList.toggle("hidden", !shouldShow);
    stageBtn.disabled = Boolean(isGenerationInFlight);
    if (shouldShow) {
      const labelNumber = Number.isFinite(stageState.nextStageNumber)
        ? stageState.nextStageNumber
        : stageState.currentStage + 1;
      stageBtn.textContent = Number.isFinite(labelNumber)
        ? `Start Stage ${labelNumber}`
        : "Start Next Stage";
    }
  }
}

updateActionButtons();

// hide Next/ranking initially
if (nextWrap) nextWrap.classList.add("hidden");
if (rankSection) rankSection.classList.add("hidden");
if (referenceSection) referenceSection.classList.add("hidden");
if (iterationIndicator) iterationIndicator.classList.add("hidden");

// ---------- selection (no drag ranking) ----------
let selectedBrick = null;
const SLOTS_BASE = "/slots"; // make sure this matches server

function setIterationDisplay(value, { commit = true } = {}) {
  if (!iterationIndicator) return;
  if (value === null || value === undefined || value === "") {
    if (commit) currentIteration = null;
    iterationIndicator.textContent = "";
    iterationIndicator.classList.add("hidden");
    return;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return;
  }
  if (commit) currentIteration = numeric;
  iterationIndicator.textContent = `Iteration ${numeric+1}`;
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

function indexOfElement(el) { return [...container.children].indexOf(el); }
function getClientX(e) { return e.touches ? e.touches[0].clientX : e.clientX; }

function onBrickClick(e) {
  const brick = e.target.closest(".brick");
  if (!brick) return;
  showZoomForBrick(brick);
}
if (container) {
  container.addEventListener("click", onBrickClick);
}

function clearZoomSelection() {
  if (selectedBrick) selectedBrick.classList.remove("selected");
  selectedBrick = null;
  if (zoomImg) {
    zoomImg.removeAttribute("src");
    zoomImg.removeAttribute("data-src");
  }
  if (zoomSection) zoomSection.classList.add("hidden");
}

function showZoomForBrick(brick) {
  if (!brick || !zoomImg || !zoomSection) return;
  const img = brick.querySelector("img");
  if (!img || !img.src) return;

  if (selectedBrick && selectedBrick !== brick) {
    selectedBrick.classList.remove("selected");
  }
  selectedBrick = brick;
  selectedBrick.classList.add("selected");

  const canonical = (img.dataset && img.dataset.src) ? img.dataset.src : img.src;
  const zoomSrc = img.currentSrc || img.src;
  zoomImg.src = zoomSrc;
  zoomImg.dataset.src = canonical;
  zoomSection.classList.remove("hidden");
}

// selection happens on click

// ---------- layout helpers ----------
function setTilesPerRow(N) {
  // choose a reasonable vw per tile so N tiles are visible-ish; clamp 8–18vw
  const basis = Math.max(8, Math.min(18, Math.floor(80 / Math.max(1, N))));
  document.documentElement.style.setProperty("--tile-ideal", `${basis}vw`);
}

// ---------- start (initial images) ----------
async function startProcess() {
  try {
    startBtn.disabled = true;
    statusEl.textContent = "Running...";
    const resp = await fetch("/api/start", { method: "POST" });
    if (!resp.ok) throw new Error("start failed");
    const data = await resp.json();
    if (data.stage) {
      applyStagePayload(data.stage);
    } else {
      updateActionButtons();
    }
    const iterFromStart = Number(data.iteration ?? data.step);
    setIterationDisplay(Number.isFinite(iterFromStart) ? iterFromStart : null);
    const images = data.images || [];
    updateReferenceImage(data.gt_image);

    if (controls) controls.classList.add("hidden");

    const renderedCount = renderImageList(images);

    statusEl.textContent = renderedCount ? `Loaded ${renderedCount} images` : "No images returned";
  } catch (err) {
    statusEl.textContent = "Error: " + (err && err.message ? err.message : "unknown");
    updateReferenceImage(null);
    setIterationDisplay(null);
  } finally {
    startBtn.disabled = false;
  }
}

if (startBtn) startBtn.addEventListener("click", startProcess);

// ---------- placeholders + per-slot reveal ----------
function renderPlaceholders(n) {
  container.innerHTML = "";
  clearZoomSelection();
  container.style.display = "grid";
  container.style.gap = "16px";
  container.style.overflow = "visible";

  for (let i = 0; i < n; i++) {
    const brick = document.createElement("div");
    brick.className = "brick";
    brick.dataset.slot = String(i);

    const img = new Image();
    img.alt = "";
    img.dataset.basename = `slot-${i}.png`;
    img.dataset.src = `${SLOTS_BASE}/slot-${i}.png`;
    brick.appendChild(img);

    const overlay = document.createElement("div");
    overlay.className = "loading";
    overlay.innerHTML = `<div class="loading-wrap"><div class="spinner"></div><div class="label">Generating…</div></div>`;
    brick.appendChild(overlay);

    container.appendChild(brick);
  }
}

function renderImageList(images) {
  if (!container) return 0;
  const list = Array.isArray(images) ? images : [];

  container.innerHTML = "";
  clearZoomSelection();
  container.style.display = "grid";
  container.style.gap = "16px";
  container.style.overflow = "visible";

  for (const src of list) {
    const div = document.createElement("div");
    div.className = "brick";
    const img = document.createElement("img");

    const canonical = src.split("?")[0];
    img.src = `${canonical}?v=${Date.now()}`;
    img.alt = "";
    img.style.maxWidth = "100%";
    img.style.display = "block";
    img.dataset.basename = canonical.split("/").pop();
    img.dataset.src = canonical;

    div.appendChild(img);
    container.appendChild(div);
  }

  if (rankSection) rankSection.classList.remove("hidden");
  if (nextWrap) nextWrap.classList.remove("hidden");

  setTilesPerRow(list.length || 6);
  return list.length;
}

async function setSlotImage(slot, round) {
  const brick = container.querySelector(`.brick[data-slot="${slot}"]`);
  if (!brick) return;
  const img = brick.querySelector("img");
  const overlay = brick.querySelector(".loading");

  const canonical = `${SLOTS_BASE}/slot-${slot}.png`;
  img.dataset.basename = `slot-${slot}.png`;
  img.dataset.src = canonical;
  img.src = `${canonical}?v=${round}&t=${Date.now()}`;

  if (img.decode) { try { await img.decode(); } catch { } }
  overlay.style.display = "none";

  if (selectedBrick === brick) {
    showZoomForBrick(brick);
  }
}

// ---------- SSE (push) ----------
const es = new EventSource("/api/events");
let currentRound = null;
let expected = 0;
let received = 0;

es.addEventListener("begin", (ev) => {
  const payload = JSON.parse(ev.data);
  const { round, n, iteration, stage } = payload;
  currentRound = round;
  expected = n;
  received = 0;

  if (iteration !== undefined) {
    setIterationDisplay(iteration);
  }

  isGenerationInFlight = true;
  if (stage) {
    applyStagePayload(stage);
  } else {
    updateActionButtons();
  }
  // if (rankSection) rankSection.classList.add("hidden");

  renderPlaceholders(n);
  setTilesPerRow(n);

  statusEl.textContent = `Generating… (0/${expected})`;
});
window.addEventListener("resize", () => setTilesPerRow(expected || 6));

es.addEventListener("slot", (ev) => {
  const { round, slot, iteration } = JSON.parse(ev.data);
  if (round !== currentRound) return;
  if (iteration !== undefined) {
    setIterationDisplay(iteration);
  }
  setSlotImage(slot, round);
  received += 1;
  statusEl.textContent = `Loaded ${received}/${expected}`;
  // if (rankSection) rankSection.classList.remove("hidden");
});

es.addEventListener("done", (ev) => {
  const payload = JSON.parse(ev.data);
  const { round, iteration, stage } = payload;
  if (round !== currentRound) return;

  if (iteration !== undefined) {
    setIterationDisplay(iteration);
  }

  statusEl.textContent = `Loaded ${received}/${expected}`;
  isGenerationInFlight = false;
  if (stage) {
    applyStagePayload(stage);
  } else {
    updateActionButtons();
  }
});

es.onerror = () => {
  statusEl.textContent = "Stream error.";
  isGenerationInFlight = false;
  updateActionButtons();
};

es.addEventListener("stage", (ev) => {
  try {
    const payload = JSON.parse(ev.data);
    if (payload && payload.stage) {
      applyStagePayload(payload.stage);
    }
    if (payload && Array.isArray(payload.images) && payload.images.length) {
      renderImageList(payload.images);
    }
  } catch (err) {
    console.error("stage event parse failed", err);
  }
});

async function refreshStageStatus() {
  try {
    const resp = await fetch("/api/stage/status");
    if (!resp.ok) return;
    const data = await resp.json();
    if (data && data.stage) {
      applyStagePayload(data.stage);
    }
    if (data && Array.isArray(data.images) && data.images.length) {
      renderImageList(data.images);
    }
  } catch (err) {
    console.warn("stage status refresh failed", err);
  }
}

refreshStageStatus();

// ---------- ranking ----------
function getSelection() {
  const img = selectedBrick?.querySelector("img");
  if (!img) return null;
  return img.dataset.src || img.src || null;
}

// ---------- Next button ----------

if (nextBtn) {
  nextBtn.addEventListener("click", async () => {
    if (stageState.hasStages && stageState.nextStageReady) {
      statusEl.textContent = "Start the next stage before continuing.";
      updateActionButtons();
      return;
    }
    const selection = getSelection();
    if (!selection) {
      statusEl.textContent = "Please select an image.";
      return;
    }
    isGenerationInFlight = true;
    updateActionButtons();
    statusEl.textContent = "Starting…";

    const previewIteration = (currentIteration ?? 0) + 1;
    setIterationDisplay(previewIteration, { commit: false });

    // show skeleton now (guess N from current tiles or default)
    const nGuess = 9;
    renderPlaceholders(nGuess);
    setTilesPerRow(nGuess);

    try {
      const resp = await fetch("/api/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selection, n: nGuess })
      });
      if (!resp.ok) {
        throw new Error("Next failed.");
      }
      // SSE will drive begin/slot/done
    } catch (err) {
      statusEl.textContent = err && err.message ? err.message : "Next failed.";
      isGenerationInFlight = false;
      updateActionButtons();
      setIterationDisplay(currentIteration, { commit: false });
    }
  });
}

if (stageBtn) {
  stageBtn.addEventListener("click", async () => {
    if (isGenerationInFlight) return;
    stageBtn.disabled = true;
    if (statusEl) statusEl.textContent = "Advancing to next stage…";
    try {
      const resp = await fetch("/api/stage/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      let data = {};
      try {
        data = await resp.json();
      } catch {
        data = {};
      }
      if (data && data.stage) {
        applyStagePayload(data.stage);
      } else {
        updateActionButtons();
      }
      if (!resp.ok) {
        const reasonCode = data && data.reason ? String(data.reason) : null;
        let friendly = "Next stage unavailable.";
        if (reasonCode === "not-ready") friendly = "Finish the current stage before advancing.";
        else if (reasonCode === "no-stages") friendly = "No additional stages are configured.";
        else if (reasonCode === "completed") friendly = "All configured stages are complete.";
        throw new Error(friendly);
      }

      const rendered = renderImageList(data && Array.isArray(data.images) ? data.images : []);
      if (statusEl) {
        statusEl.textContent = rendered
          ? `Next stage initialized with ${rendered} candidates.`
          : "Next stage initialized. Waiting for new images…";
      }
    } catch (err) {
      if (statusEl) {
        statusEl.textContent = `Next stage unavailable: ${err && err.message ? err.message : "unknown error"}`;
      }
    } finally {
      updateActionButtons();
    }
  });
}
