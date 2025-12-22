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
const rankMidLabel = document.getElementById("rankMidLabel");
const controls = document.getElementById("controls");
const referenceSection = document.getElementById("referenceSection");
const referenceImg = document.getElementById("referenceImg");
const zoomSection = document.getElementById("zoomSection");
const zoomImg = document.getElementById("zoomImg");
const zoomBrick = document.querySelector(".zoom-brick");
const zoomSafetyOverlay = document.getElementById("zoomSafetyOverlay");
const gridLabel = document.getElementById("gridLabel");
const gridSection = document.getElementById("gridSection");
const gridContainer = document.getElementById("gridContainer");
const descriptionToggle = document.getElementById("descriptionToggle");
const descriptionOverlay = document.getElementById("descriptionOverlay");
const descriptionClose = document.getElementById("descriptionClose");
const tutorialToggle = document.getElementById("tutorialToggle");
const tutorialOverlay = document.getElementById("tutorialOverlay");
const tutorialClose = document.getElementById("tutorialClose");
const demoOverlay = document.getElementById("demoOverlay");
const demoClose = document.getElementById("demoClose");
const demoForm = document.getElementById("demoForm");
const demoError = document.getElementById("demoError");
const demoSubmit = document.getElementById("demoSubmit");
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
let demographicsSubmitted = false;
let demographicsEnabled = false;
let demographicsParticipantId = null;
let demographicsConfigLoaded = false;
let topK = null;
let candidates = [];
let selectedOrder = [];
let zoomedBrick = null;

// Keep the grid label in sync with the current topK value (or fall back).
function updateGridLabel() {
  if (!gridLabel) return;
  const label = topK && topK > 0
    ? `All images (right-click to select the top ${topK})`
    : "All images (right-click to select)";
  gridLabel.textContent = label;
}

// Allow a query param ?top_k= to set the initial label immediately (useful when
// the API response arrives later or is cached).
const initialTopKParam = Number(new URLSearchParams(window.location.search).get("top_k"));
if (Number.isFinite(initialTopKParam) && initialTopKParam > 0) {
  topK = Math.floor(initialTopKParam);
}
updateGridLabel();

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
  if (( !tutorialOverlay || tutorialOverlay.classList.contains("hidden")) && (!demoOverlay || demoOverlay.classList.contains("hidden"))) {
    setBodyScrollLock(false);
  }
}

function openTutorialOverlay() {
  if (!tutorialOverlay) return;
  tutorialOverlay.classList.remove("hidden");
  setBodyScrollLock(true);
}

function closeTutorialOverlay() {
  if (!tutorialOverlay) return;
  tutorialOverlay.classList.add("hidden");
  if (( !descriptionOverlay || descriptionOverlay.classList.contains("hidden")) && (!demoOverlay || demoOverlay.classList.contains("hidden"))) {
    setBodyScrollLock(false);
  }
}

function openDemoOverlay() {
  if (!demoOverlay) {
    startProcess();
    return;
  }
  demoOverlay.classList.remove("hidden");
  setBodyScrollLock(true);
}

function closeDemoOverlay() {
  if (!demoOverlay) return;
  demoOverlay.classList.add("hidden");
  const descHidden = !descriptionOverlay || descriptionOverlay.classList.contains("hidden");
  const tutHidden = !tutorialOverlay || tutorialOverlay.classList.contains("hidden");
  if (descHidden && tutHidden) {
    setBodyScrollLock(false);
  }
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
if (demoClose) {
  demoClose.addEventListener("click", closeDemoOverlay);
}
if (demoOverlay) {
  demoOverlay.addEventListener("click", (event) => {
    if (event.target === demoOverlay) {
      closeDemoOverlay();
    }
  });
}
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (descriptionOverlay && !descriptionOverlay.classList.contains("hidden")) {
    closeDescriptionOverlay();
  }
  if (tutorialOverlay && !tutorialOverlay.classList.contains("hidden")) {
    closeTutorialOverlay();
  }
  if (demoOverlay && !demoOverlay.classList.contains("hidden")) {
    closeDemoOverlay();
  }
});

if (gridContainer) {
  gridContainer.addEventListener("contextmenu", (event) => {
    const brick = event.target.closest(".brick");
    if (!brick) return;
    event.preventDefault();
    toggleSelection(brick.dataset?.canonical || brick.dataset?.src);
  });

  gridContainer.addEventListener("click", (event) => {
    const brick = event.target.closest(".brick");
    if (!brick) return;
    setZoomByCanonical(brick.dataset?.canonical || brick.dataset?.src);
  });
}

async function loadDemographicsConfig(force = false) {
  if (demographicsConfigLoaded && !force) return { enabled: demographicsEnabled, participant_id: demographicsParticipantId };
  try {
    const resp = await fetch("/api/demographics/config");
    if (!resp.ok) throw new Error("config fetch failed");
    const data = await resp.json();
    demographicsEnabled = Boolean(data.enabled);
    demographicsParticipantId = data.participant_id || null;
  } catch (err) {
    demographicsEnabled = false;
    demographicsParticipantId = null;
  } finally {
    demographicsConfigLoaded = true;
  }
  return { enabled: demographicsEnabled, participant_id: demographicsParticipantId };
}

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
  const selectionRequired = topK && topK > 0;
  const selectionOk = !selectionRequired || selectedOrder.length >= topK;
  if (nextBtn) {
    nextBtn.disabled = Boolean(isGenerationInFlight || stageReady || !selectionOk);
    nextBtn.classList.toggle("hidden", stageReady);
  }
  if (stageBtn) {
    const shouldShow = stageReady;
    stageBtn.classList.toggle("hidden", !shouldShow);
    stageBtn.disabled = Boolean(isGenerationInFlight);
    if (shouldShow) {
      // Keep the label user-facing as "Next" so stage transitions feel seamless.
      stageBtn.textContent = "Next";
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
const DRAG_THRESHOLD_PX = 5;

const SLOTS_BASE = "/slots"; // make sure this matches server
const safetyIndex = new Map();
let safetyRefreshPromise = null;
let safetyRefreshQueued = false;

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

function canonicalizeImageSrc(src) {
  if (typeof src !== "string" || !src) return null;
  return src.split("?")[0];
}

function normalizeImageEntry(entry) {
  if (!entry) {
    return { url: null, canonical: null, isSafe: true };
  }
  if (typeof entry === "string") {
    const canonical = canonicalizeImageSrc(entry);
    return { url: entry, canonical, isSafe: true };
  }
  if (typeof entry === "object") {
    const rawUrl = typeof entry.url === "string"
      ? entry.url
      : typeof entry.src === "string"
        ? entry.src
        : typeof entry.path === "string"
          ? entry.path
          : null;
    const fallbackPath = typeof entry.basename === "string" ? `${SLOTS_BASE}/${entry.basename}` : null;
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
      isSafe = true;
    }
    return {
      url: rawUrl || canonical,
      canonical,
      isSafe,
    };
  }
  return { url: null, canonical: null, isSafe: true };
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
  if (zoomImg && zoomImg.dataset?.src === canonical) {
    updateZoomSafety(canonical);
  }
}

function applySafetyOverlay(brick, isSafe = true) {
  if (!brick) return;
  const overlay = brick.querySelector(".nsfw-overlay");
  if (isSafe === true) {
    brick.classList.remove("nsfw-flagged", "nsfw-pending");
    if (overlay) overlay.remove();
    return;
  } else if (isSafe === false) {
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
  const bricks = [];
  if (container) bricks.push(...container.querySelectorAll(".brick"));
  if (gridContainer) bricks.push(...gridContainer.querySelectorAll(".brick"));
  bricks.forEach((brick) => {
    const img = brick.querySelector("img");
    if (!img) return;
    const canonical = img.dataset?.src;
    let state = true;
    if (canonical) {
      state = safetyIndex.has(canonical) ? safetyIndex.get(canonical) : null;
    }
    applySafetyOverlay(brick, state);
  });

  const zoomCanonical = zoomImg?.dataset?.src || null;
  updateZoomSafety(zoomCanonical);
}

function updateZoomSafety(canonical) {
  if (!zoomBrick) return;
  let state = true;
  if (canonical) {
    state = safetyIndex.has(canonical) ? safetyIndex.get(canonical) : null;
  }
  zoomBrick.classList.toggle("nsfw-flagged", state === false);
  zoomBrick.classList.toggle("nsfw-pending", state === null);
  if (zoomSafetyOverlay) {
    zoomSafetyOverlay.classList.toggle("hidden", state !== false);
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

function getSelectionLimit() {
  return topK && topK > 0 ? Math.min(topK, candidates.length) : candidates.length;
}

function setTopKValue(value) {
  const numeric = Number(value);
  topK = Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : null;
  clampSelectionToLimit();
  updateRankMidLabel();
  updateGridLabel();
  updateActionButtons();
}

function updateRankMidLabel() {
  if (!rankMidLabel) return;
  rankMidLabel.textContent = topK && topK > 0 ? `Rank top ${topK}` : "Rank";
}

function syncSelectedOrderToCandidates() {
  const validCanonicals = new Set((candidates || []).map((c) => c.canonical));
  selectedOrder = selectedOrder.filter((c) => validCanonicals.has(c));
}

function clampSelectionToLimit() {
  syncSelectedOrderToCandidates();
  const limit = getSelectionLimit();
  if (limit < 0) return;
  selectedOrder = selectedOrder.slice(0, limit);
}

function resetSelectionToCandidateOrder() {
  // Do not auto-select; start each round with an empty selection.
  selectedOrder = [];
}

function getCandidateByCanonical(canonical) {
  return (candidates || []).find((c) => c.canonical === canonical);
}

function getCandidateBySlot(slot) {
  return (candidates || []).find((c) => Number(c.slot) === Number(slot));
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

  selectedOrder = [...container.querySelectorAll(".brick img")]
    .map((img) => img.dataset?.src || img.src)
    .filter(Boolean);
  syncSelectionStyles();

  if (!wasDrag) {
    showZoomForBrick(droppedBrick);
  }

  lastInteractionWasDrag = false;
}

function clearZoomSelection() {
  if (zoomedBrick) zoomedBrick.classList.remove("zoomed");
  zoomedBrick = null;
  if (zoomImg) {
    zoomImg.removeAttribute("src");
    zoomImg.removeAttribute("data-src");
  }
  if (zoomSection) zoomSection.classList.add("hidden");
  updateZoomSafety(null);
}

function showZoomForBrick(brick) {
  if (!brick || !zoomImg || !zoomSection) return;
  const img = brick.querySelector("img");
  if (!img || !img.src) return;

  if (zoomedBrick && zoomedBrick !== brick) {
    zoomedBrick.classList.remove("zoomed");
  }
  zoomedBrick = brick;
  zoomedBrick.classList.add("zoomed");

  const canonical = (img.dataset && img.dataset.src) ? img.dataset.src : img.src;
  const zoomSrc = img.currentSrc || img.src;
  zoomImg.src = zoomSrc;
  zoomImg.dataset.src = canonical;
  zoomSection.classList.remove("hidden");
  updateZoomSafety(canonical);
}

// selection happens on mouse/touch release (see onMouseUp)

// ---------- layout helpers ----------
function setTilesPerRow(N) {
  // choose a reasonable vw per tile so N tiles are visible-ish; clamp 8–18vw
  const basis = Math.max(8, Math.min(18, Math.floor(80 / Math.max(1, N))));
  document.documentElement.style.setProperty("--tile-ideal", `${basis}vw`);
}

function renderGrid({ showLoading = false } = {}) {
  if (!gridContainer) return;
  gridContainer.innerHTML = "";
  const list = Array.isArray(candidates) ? candidates : [];

  for (let idx = 0; idx < list.length; idx++) {
    const entry = list[idx];
    const brick = document.createElement("div");
    brick.className = "brick";
    brick.dataset.slot = entry.slot ?? idx;
    brick.dataset.canonical = entry.canonical;
    brick.title = "Right-click to select/deselect";

    const img = document.createElement("img");
    img.alt = "";
    img.dataset.basename = entry.basename || entry.canonical.split("/").pop();
    img.dataset.src = entry.canonical;
    img.src = `${entry.canonical}?v=${Date.now()}`;
    brick.appendChild(img);

    if (showLoading || !entry.loaded) {
      const overlay = document.createElement("div");
      overlay.className = "loading";
      overlay.innerHTML = `<div class="loading-wrap"><div class="spinner"></div><div class="label">Generating…</div></div>`;
      brick.appendChild(overlay);
    }

    let safetyState = entry.isSafe;
    if (safetyState === undefined || safetyState === null) {
      safetyState = safetyIndex.has(entry.canonical) ? safetyIndex.get(entry.canonical) : null;
    }
    rememberImageSafety(entry.canonical, safetyState);
    applySafetyOverlay(brick, safetyState);
    if (selectedOrder.includes(entry.canonical)) {
      brick.classList.add("selected");
    }

    gridContainer.appendChild(brick);
  }

  if (gridSection) {
    gridSection.classList.toggle("hidden", !list.length);
  }
}

function renderRankingFromSelection({ showLoading = false } = {}) {
  if (!container) return 0;
  container.innerHTML = "";
  const canonicalSet = new Set((candidates || []).map((c) => c.canonical));
  selectedOrder = selectedOrder.filter((c) => canonicalSet.has(c));
  clampSelectionToLimit();

  let rendered = 0;
  for (const canonical of selectedOrder) {
    const entry = getCandidateByCanonical(canonical);
    if (!entry) continue;

    const brick = document.createElement("div");
    brick.className = "brick";
    brick.dataset.canonical = entry.canonical;
    brick.dataset.slot = entry.slot ?? rendered;

    const img = document.createElement("img");
    img.alt = "";
    img.dataset.basename = entry.basename || entry.canonical.split("/").pop();
    img.dataset.src = entry.canonical;
    img.src = `${entry.canonical}?v=${Date.now()}`;
    brick.appendChild(img);

    if (showLoading || !entry.loaded) {
      const overlay = document.createElement("div");
      overlay.className = "loading";
      overlay.innerHTML = `<div class="loading-wrap"><div class="spinner"></div><div class="label">Generating…</div></div>`;
      brick.appendChild(overlay);
    }

    let safetyState = entry.isSafe;
    if (safetyState === undefined || safetyState === null) {
      safetyState = safetyIndex.has(entry.canonical) ? safetyIndex.get(entry.canonical) : null;
    }
    rememberImageSafety(entry.canonical, safetyState);
    applySafetyOverlay(brick, safetyState);

    container.appendChild(brick);
    rendered += 1;
  }

  if (rendered === 0) {
    const placeholder = document.createElement("div");
    placeholder.className = "rank-placeholder";
    const limit = topK && topK > 0 ? topK : null;
    placeholder.textContent = limit ? `Select top ${limit} images` : "Select images to rank";
    container.appendChild(placeholder);
  }

  if (rankSection) rankSection.classList.remove("hidden");
  if (nextWrap) nextWrap.classList.remove("hidden");
  setTilesPerRow(rendered || 6);
  syncSafetyOverlays();
  syncSelectionStyles();
  updateActionButtons();

  const currentZoom = zoomImg?.dataset?.src || null;
  const zoomTarget = currentZoom && selectedOrder.includes(currentZoom)
    ? currentZoom
    : (selectedOrder.length ? selectedOrder[0] : null);
  if (zoomTarget) {
    setZoomByCanonical(zoomTarget);
  } else {
    clearZoomSelection();
  }

  return rendered;
}

function syncSelectionStyles() {
  const selectedSet = new Set(selectedOrder);
  if (gridContainer) {
    gridContainer.querySelectorAll(".brick").forEach((brick) => {
      const canonical = brick.dataset?.canonical;
      brick.classList.toggle("selected", selectedSet.has(canonical));
    });
  }
}

function toggleSelection(canonical) {
  if (!canonical) return;
  const limit = topK && topK > 0 ? topK : candidates.length;
  const isSelected = selectedOrder.includes(canonical);

  if (isSelected) {
    selectedOrder = selectedOrder.filter((c) => c !== canonical);
  } else {
    if (selectedOrder.length >= limit) {
      if (statusEl) statusEl.textContent = `Select up to ${limit} images to rank.`;
      return;
    }
    selectedOrder.push(canonical);
  }

  renderRankingFromSelection();
  syncSelectionStyles();
  updateActionButtons();
}

function setZoomByCanonical(canonical) {
  if (!canonical) {
    clearZoomSelection();
    return;
  }
  const bricks = [];
  if (container) bricks.push(...container.querySelectorAll(".brick"));
  if (gridContainer) bricks.push(...gridContainer.querySelectorAll(".brick"));
  const target = bricks.find((b) => (b.dataset?.canonical || b.dataset?.src) === canonical);
  if (target) {
    showZoomForBrick(target);
  }
}

// ---------- demographics gating ----------
function getCheckedValue(name) {
  if (!demoForm) return null;
  const el = demoForm.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

function getCheckedValues(name) {
  if (!demoForm) return [];
  return Array.from(demoForm.querySelectorAll(`input[name="${name}"]:checked`)).map((el) => el.value);
}

function collectDemographicsPayload() {
  if (!demoForm) return {};
  const payload = {
    age_group: getCheckedValue("age_group"),
    gender_identity: getCheckedValue("gender_identity"),
    gender_self_describe: demoForm.querySelector("#genderSelfDescribe")?.value || null,
    familiarity: getCheckedValue("familiarity"),
    usage_frequency: getCheckedValue("usage_frequency"),
    experience_depth: getCheckedValues("experience_depth"),
    domain_background: getCheckedValues("domain_background"),
    participant_id: demographicsParticipantId,
  };
  return payload;
}

async function submitDemographics(event) {
  if (event) event.preventDefault();
  if (demographicsSubmitted || !demographicsEnabled) {
    closeDemoOverlay();
    startProcess();
    return;
  }
  if (!demoForm) {
    startProcess();
    return;
  }
  if (demoError) demoError.classList.add("hidden");
  try {
    if (demoSubmit) demoSubmit.disabled = true;
    const payload = collectDemographicsPayload();
    const resp = await fetch("/api/demographics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error("Failed to save responses");
    demographicsSubmitted = true;
    closeDemoOverlay();
    await startProcess();
  } catch (err) {
    if (demoError) {
      demoError.textContent = err && err.message ? err.message : "Could not save responses. Please try again.";
      demoError.classList.remove("hidden");
    }
  } finally {
    if (demoSubmit) demoSubmit.disabled = false;
  }
}

if (demoForm) {
  demoForm.addEventListener("submit", submitDemographics);
}

async function handleStartClick(event) {
  if (event) event.preventDefault();
  await loadDemographicsConfig();
  if (!demographicsEnabled || demographicsSubmitted || !demoForm) {
    await startProcess();
    return;
  }
  openDemoOverlay();
}

// ---------- start (initial images) ----------
async function startProcess() {
  try {
    startBtn.disabled = true;
    statusEl.textContent = "Running...";
    const resp = await fetch("/api/start", { method: "POST" });
    if (!resp.ok) throw new Error("start failed");
    const data = await resp.json();
    if (data.top_k !== undefined) {
      setTopKValue(data.top_k);
    }
    updateRankMidLabel();
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

if (startBtn) startBtn.addEventListener("click", handleStartClick);

// ---------- placeholders + per-slot reveal ----------
function renderPlaceholders(n) {
  candidates = [];
  for (let i = 0; i < n; i++) {
    const canonical = `${SLOTS_BASE}/slot-${i}.png`;
    candidates.push({
      slot: i,
      canonical,
      basename: `slot-${i}.png`,
      isSafe: null,
      loaded: false,
    });
    safetyIndex.delete(canonical);
  }
  resetSelectionToCandidateOrder();
  renderGrid({ showLoading: true });
  renderRankingFromSelection({ showLoading: true });
}

function renderImageList(images) {
  if (!container) return 0;
  const list = Array.isArray(images) ? images : [];

  candidates = [];
  list.forEach((entry, idx) => {
    const normalized = normalizeImageEntry(entry);
    const canonical = normalized?.canonical || canonicalizeImageSrc(normalized?.url);
    if (!canonical) return;
    candidates.push({
      slot: idx,
      canonical,
      basename: canonical.split("/").pop(),
      isSafe: normalized.isSafe,
      loaded: true,
    });
  });

  resetSelectionToCandidateOrder();
  renderGrid({ showLoading: false });
  const rendered = renderRankingFromSelection({ showLoading: false });
  syncSelectionStyles();
  return rendered;
}

async function setSlotImage(slot, round) {
  const canonical = `${SLOTS_BASE}/slot-${slot}.png`;
  const entry = getCandidateBySlot(slot);
  if (entry) {
    entry.canonical = canonical;
    entry.basename = `slot-${slot}.png`;
    entry.loaded = true;
  }

  const bricks = [];
  if (gridContainer) {
    const gridBrick = gridContainer.querySelector(`.brick[data-slot="${slot}"]`);
    if (gridBrick) bricks.push(gridBrick);
  }
  if (container) {
    const rankBrick = container.querySelector(`.brick[data-slot="${slot}"]`);
    if (rankBrick) bricks.push(rankBrick);
  }

  for (const brick of bricks) {
    const img = brick.querySelector("img");
    const overlay = brick.querySelector(".loading");
    if (!img) continue;

    img.dataset.basename = `slot-${slot}.png`;
    img.dataset.src = canonical;
    markImageSafetyUnknown(canonical, brick);
    img.src = `${canonical}?v=${round}&t=${Date.now()}`;

    if (img.decode) { try { await img.decode(); } catch { } }
    if (overlay) overlay.style.display = "none";

    let safetyState = true;
    if (canonical) {
      safetyState = safetyIndex.has(canonical) ? safetyIndex.get(canonical) : null;
    }
    if (entry && safetyState !== undefined) {
      entry.isSafe = safetyState;
    }
    applySafetyOverlay(brick, safetyState);
  }

  refreshSafetyFromServer();
  setZoomByCanonical(zoomImg?.dataset?.src || null);
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

  refreshSafetyFromServer();
});

es.onerror = () => {
  statusEl.textContent = "Stream error.";
  isGenerationInFlight = false;
  updateActionButtons();
};

es.addEventListener("stage", (ev) => {
  try {
    const payload = JSON.parse(ev.data);
    if (payload && payload.top_k !== undefined) {
      setTopKValue(payload.top_k);
    }
    updateRankMidLabel();
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
    if (data.top_k !== undefined) {
      setTopKValue(data.top_k);
    }
    updateRankMidLabel();
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

function getAllBasenames() {
  return (candidates || []).map((c) => c.basename || (c.canonical ? c.canonical.split("/").pop() : null)).filter(Boolean);
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
    const allBasenamesSnapshot = getAllBasenames();
    const required = topK && topK > 0 ? topK : order.length;
    if (required && order.length < required) {
      statusEl.textContent = `Select ${required} images to rank.`;
      return;
    }
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
        body: JSON.stringify({ ranking: order, all_basenames: allBasenamesSnapshot })
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
    if (statusEl) statusEl.textContent = "Loading next images…";
    const order = getRanking();
    const required = topK && topK > 0 ? topK : order.length;
    if (required && order.length < required) {
      if (statusEl) statusEl.textContent = `Select ${required} images to rank.`;
      stageBtn.disabled = false;
      return;
    }
    try {
      const resp = await fetch("/api/stage/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ranking: order }),
      });
      let data = {};
      try {
        data = await resp.json();
      } catch {
        data = {};
      }
      if (data && data.top_k !== undefined) {
        setTopKValue(data.top_k);
      }
      if (data && data.stage) {
        applyStagePayload(data.stage);
      } else {
        updateActionButtons();
      }
      if (!resp.ok) {
        const reasonCode = data && data.reason ? String(data.reason) : null;
        let friendly = "Unable to continue right now.";
        if (reasonCode === "not-ready") friendly = "Finish the current set before continuing.";
        else if (reasonCode === "no-stages") friendly = "No additional images are available.";
        else if (reasonCode === "completed") friendly = "All image sets are complete.";
        throw new Error(friendly);
      }

      const rendered = renderImageList(data && Array.isArray(data.images) ? data.images : []);
      if (statusEl) {
        statusEl.textContent = rendered
          ? `Loaded ${rendered} new candidates.`
          : "Loading next set…";
      }
    } catch (err) {
      if (statusEl) {
        statusEl.textContent = `Unable to continue: ${err && err.message ? err.message : "unknown error"}`;
      }
    } finally {
      updateActionButtons();
    }
  });
}
