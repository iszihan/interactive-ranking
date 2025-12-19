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
const zoomBrick = document.querySelector(".zoom-brick");
const zoomSafetyOverlay = document.getElementById("zoomSafetyOverlay");
const descriptionToggle = document.getElementById("descriptionToggle");
const descriptionOverlay = document.getElementById("descriptionOverlay");
const descriptionClose = document.getElementById("descriptionClose");
const tutorialToggle = document.getElementById("tutorialToggle");
const tutorialOverlay = document.getElementById("tutorialOverlay");
const tutorialClose = document.getElementById("tutorialClose");
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
  if (!tutorialOverlay || tutorialOverlay.classList.contains("hidden")) {
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
  if (!descriptionOverlay || descriptionOverlay.classList.contains("hidden")) {
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
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (descriptionOverlay && !descriptionOverlay.classList.contains("hidden")) {
    closeDescriptionOverlay();
  }
  if (tutorialOverlay && !tutorialOverlay.classList.contains("hidden")) {
    closeTutorialOverlay();
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
  if (!container) return;
  container.querySelectorAll(".brick").forEach((brick) => {
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
  updateZoomSafety(null);
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
  updateZoomSafety(canonical);
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
    safetyIndex.delete(img.dataset.src);
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

  let rendered = 0;
  for (const entry of list) {
    const normalized = normalizeImageEntry(entry);
    const canonical = normalized?.canonical || canonicalizeImageSrc(normalized?.url);
    if (!canonical) continue;

    const div = document.createElement("div");
    div.className = "brick";
    const img = document.createElement("img");

    img.src = `${canonical}?v=${Date.now()}`;
    img.alt = "";
    img.style.maxWidth = "100%";
    img.style.display = "block";
    img.dataset.basename = canonical.split("/").pop();
    img.dataset.src = canonical;

    rememberImageSafety(canonical, normalized.isSafe);
    applySafetyOverlay(div, normalized.isSafe);

    div.appendChild(img);
    container.appendChild(div);
    rendered += 1;
  }

  if (rankSection) rankSection.classList.remove("hidden");
  if (nextWrap) nextWrap.classList.remove("hidden");

  setTilesPerRow(rendered || 6);
  syncSafetyOverlays();
  return rendered;
}

async function setSlotImage(slot, round) {
  const brick = container.querySelector(`.brick[data-slot="${slot}"]`);
  if (!brick) return;
  const img = brick.querySelector("img");
  const overlay = brick.querySelector(".loading");

  const canonical = `${SLOTS_BASE}/slot-${slot}.png`;
  img.dataset.basename = `slot-${slot}.png`;
  img.dataset.src = canonical;
  markImageSafetyUnknown(canonical, brick);
  img.src = `${canonical}?v=${round}&t=${Date.now()}`;

  if (img.decode) { try { await img.decode(); } catch { } }
  overlay.style.display = "none";

  let safetyState = true;
  if (canonical) {
    safetyState = safetyIndex.has(canonical) ? safetyIndex.get(canonical) : null;
  }
  applySafetyOverlay(brick, safetyState);
  refreshSafetyFromServer();

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
