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
const descriptionToggle = document.getElementById("descriptionToggle");
const descriptionOverlay = document.getElementById("descriptionOverlay");
const descriptionClose = document.getElementById("descriptionClose");
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
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && descriptionOverlay && !descriptionOverlay.classList.contains("hidden")) {
    closeDescriptionOverlay();
  }
});

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

// ---------- drag-to-reorder ----------
let draggingElem = null;
let startX = 0;
let startIndex = 0;
let currentIndex = 0;
let lastInteractionWasDrag = false;
let selectedBrick = null;
const DRAG_THRESHOLD_PX = 5;

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

function onMouseDown(e) {
  const brick = e.target.closest(".brick");
  if (!brick) return;
  e.preventDefault();
  lastInteractionWasDrag = false;

  draggingElem = brick;
  startX = getClientX(e);
  startIndex = indexOfElement(draggingElem);
  currentIndex = startIndex;

  draggingElem.classList.add("dragging");
  document.addEventListener("mousemove", onMouseMove);
  document.addEventListener("mouseup", onMouseUp);
  document.addEventListener("touchmove", onMouseMove, { passive: false });
  document.addEventListener("touchend", onMouseUp);
}
container.addEventListener("mousedown", onMouseDown);
container.addEventListener("touchstart", onMouseDown, { passive: false });

function onMouseMove(e) {
  e.preventDefault();
  if (!draggingElem) return;

  const deltaX = getClientX(e) - startX;
  if (!lastInteractionWasDrag && Math.abs(deltaX) > DRAG_THRESHOLD_PX) {
    lastInteractionWasDrag = true;
  }
  draggingElem.style.transform = `translateX(${deltaX}px)`;

  const all = [...container.children];
  const others = all.filter(el => el !== draggingElem);

  function middle(el) { const r = el.getBoundingClientRect(); return r.left + r.width / 2; }
  const midX = middle(draggingElem);

  let newIndex = others.findIndex(other => midX < middle(other));
  if (newIndex === -1) newIndex = others.length;

  if (newIndex !== currentIndex) {
    currentIndex = newIndex;
    reorderGhosts();
  }
}

function reorderGhosts() {
  const bricks = [...container.children].filter(el => el !== draggingElem);
  const spacing = draggingElem.getBoundingClientRect().width;
  const cs = getComputedStyle(container);
  const gap = parseFloat(cs.columnGap || cs.getPropertyValue?.("column-gap") || cs.gap || "0") || 0;
  const offset = spacing + gap;

  bricks.forEach((b, i) => {
    b.style.transform = "";
    if (startIndex < currentIndex && i >= startIndex && i < currentIndex) {
      b.style.transform = `translateX(-${offset}px)`;
    } else if (startIndex > currentIndex && i >= currentIndex && i < startIndex) {
      b.style.transform = `translateX(${offset}px)`;
    }
  });
}

function onMouseUp() {
  if (!draggingElem) return;

  const droppedBrick = draggingElem;
  const wasDrag = lastInteractionWasDrag;
  const bricks = [...container.children].filter(el => el !== draggingElem);
  bricks.forEach(b => { b.style.transform = ""; b.classList.add("resetting"); });

  draggingElem.style.transform = "";
  draggingElem.classList.remove("dragging");

  const referenceNode = bricks[currentIndex] ?? null;
  container.insertBefore(draggingElem, referenceNode);

  draggingElem = null;
  document.removeEventListener("mousemove", onMouseMove);
  document.removeEventListener("mouseup", onMouseUp);
  document.removeEventListener("touchmove", onMouseMove);
  document.removeEventListener("touchend", onMouseUp);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bricks.forEach(b => b.classList.remove("resetting"));
    });
  });

  if (!wasDrag) {
    showZoomForBrick(droppedBrick);
  }

  lastInteractionWasDrag = false;
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

// selection happens on mouse/touch release (see onMouseUp)

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
  container.style.display = "flex";
  container.style.gap = "12px";
  container.style.overflowX = "auto";

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
  container.style.display = "flex";
  container.style.flexDirection = "row";
  container.style.flexWrap = "nowrap";
  container.style.gap = "12px";
  container.style.overflowX = "auto";

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
function getRanking() {
  const order = [...container.querySelectorAll(".brick img")].map(img =>
    img.dataset.basename ||
    new URL(img.src, location.href).pathname.split("/").pop()
  ).filter(Boolean);

  // sanity check
  console.log("ranking to send:", order);
  return order;
}

// ---------- Next button ----------

if (nextBtn) {
  nextBtn.addEventListener("click", async () => {
    if (stageState.hasStages && stageState.nextStageReady) {
      statusEl.textContent = "Start the next stage before continuing.";
      updateActionButtons();
      return;
    }
    const order = getRanking();
    isGenerationInFlight = true;
    updateActionButtons();
    statusEl.textContent = "Starting…";

    const previewIteration = (currentIteration ?? 0) + 1;
    setIterationDisplay(previewIteration, { commit: false });

    // show skeleton now (guess N from current tiles or default)
    const nGuess = container.querySelectorAll(".brick").length || 6;
    renderPlaceholders(nGuess);
    setTilesPerRow(nGuess);

    try {
      const resp = await fetch("/api/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ranking: order })
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
