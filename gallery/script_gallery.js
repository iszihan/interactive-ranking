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
const refreshUiBtn = document.getElementById("refreshUiBtn");
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
let selectedBrick = null;
const safetyIndex = new Map();
let safetyRefreshPromise = null;
let safetyRefreshQueued = false;
const SLOTS_BASE = "/slots"; // make sure this matches server
let demographicsEnabled = false;
let demographicsParticipantId = null;
let demographicsSubmitted = false;
let demographicsConfigLoaded = false;

function createCenterHoverZoom({ allowShow } = {}) {
  const overlay = document.createElement("div");
  overlay.className = "hover-zoom-overlay hidden";
  const overlayImg = document.createElement("img");
  overlay.append(overlayImg);
  document.body.append(overlay);

  const show = (imgEl) => {
    if (!imgEl) return;
    if (typeof allowShow === "function" && !allowShow(imgEl)) {
      hide();
      return;
    }
    const src = imgEl.currentSrc || imgEl.src;
    if (!src) return;
    overlayImg.src = src;
    overlayImg.alt = imgEl.alt || "";
    overlay.classList.remove("hidden");
  };

  const hide = () => overlay.classList.add("hidden");

  const attach = (imgEl) => {
    if (!imgEl) return;
    imgEl.addEventListener("mouseenter", () => show(imgEl));
    imgEl.addEventListener("mouseleave", hide);
    imgEl.addEventListener("focus", () => show(imgEl));
    imgEl.addEventListener("blur", hide);
  };

  return { attach, hide };
}

function isHoverZoomAllowed(imgEl) {
  if (!imgEl) return false;
  const unsafeAncestor = imgEl.closest(".nsfw-flagged, .nsfw-pending");
  if (unsafeAncestor) return false;
  const canonical = canonicalizeImageSrc(imgEl.dataset?.src || imgEl.currentSrc || imgEl.src);
  if (!canonical) return true;
  const state = safetyIndex.has(canonical) ? safetyIndex.get(canonical) : null;
  return state !== false;
}

const hoverZoom = createCenterHoverZoom({ allowShow: isHoverZoomAllowed });
hoverZoom.attach(referenceImg);
hoverZoom.attach(zoomImg);

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
  if ((!tutorialOverlay || tutorialOverlay.classList.contains("hidden")) && (!demoOverlay || demoOverlay.classList.contains("hidden"))) {
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
  if ((!descriptionOverlay || descriptionOverlay.classList.contains("hidden")) && (!demoOverlay || demoOverlay.classList.contains("hidden"))) {
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
  if (event.key === "Escape") {
    if (descriptionOverlay && !descriptionOverlay.classList.contains("hidden")) {
      closeDescriptionOverlay();
    }
    if (tutorialOverlay && !tutorialOverlay.classList.contains("hidden")) {
      closeTutorialOverlay();
    }
    if (demoOverlay && !demoOverlay.classList.contains("hidden")) {
      closeDemoOverlay();
    }
  }
});

async function loadDemographicsConfig(force = false) {
  if (demographicsConfigLoaded && !force) return { enabled: demographicsEnabled, participant_id: demographicsParticipantId };
  try {
    const resp = await fetch("/api/demographics/config");
    if (!resp.ok) throw new Error("config fetch failed");
    const data = await resp.json();
    demographicsEnabled = Boolean(data.enabled);
    demographicsParticipantId = data.participant_id || null;
    demographicsConfigLoaded = true;
  } catch (err) {
    demographicsEnabled = false;
    demographicsParticipantId = null;
    demographicsConfigLoaded = false; // allow retry on failure
  }
  return { enabled: demographicsEnabled, participant_id: demographicsParticipantId };
}

// Preload demographics config so the first Start click has data
loadDemographicsConfig().catch(() => {
  demographicsConfigLoaded = false;
});

async function autoOpenDemographicsIfEnabled() {
  try {
    const cfg = await loadDemographicsConfig();
    if (cfg && cfg.enabled && !demographicsSubmitted && demoOverlay && demoForm) {
      openDemoOverlay();
    }
  } catch (err) {
    // best-effort; ignore errors to avoid blocking the app load
  }
}

autoOpenDemographicsIfEnabled();

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
    const hasSelection = Boolean(selectedBrick);
    nextBtn.disabled = Boolean(isGenerationInFlight || stageReady || !hasSelection);
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
  updateZoomSafety(null);
  updateActionButtons();
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
  updateActionButtons();
}

// selection happens on click

// ---------- layout helpers ----------
function setTilesPerRow(N) {
  // choose a reasonable vw per tile so N tiles are visible-ish; clamp 8–18vw
  const basis = Math.max(8, Math.min(18, Math.floor(80 / Math.max(1, N))));
  document.documentElement.style.setProperty("--tile-ideal", `${basis}vw`);
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
  return {
    age_group: getCheckedValue("age_group"),
    gender_identity: getCheckedValue("gender_identity"),
    gender_self_describe: demoForm.querySelector("#genderSelfDescribe")?.value || null,
    familiarity: getCheckedValue("familiarity"),
    usage_frequency: getCheckedValue("usage_frequency"),
    experience_depth: getCheckedValues("experience_depth"),
    domain_background: getCheckedValues("domain_background"),
    participant_id: demographicsParticipantId,
  };
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
    safetyIndex.delete(img.dataset.src);
    brick.appendChild(img);

    const overlay = document.createElement("div");
    overlay.className = "loading";
    overlay.innerHTML = `<div class="loading-wrap"><div class="spinner"></div><div class="label">Generating…</div></div>`;
    brick.appendChild(overlay);

    container.appendChild(brick);
  }
}

function slotIndexFromCanonical(canonical) {
  if (typeof canonical !== "string") return null;
  const match = canonical.match(/slot-(\d+)/);
  if (!match) return null;
  const idx = Number(match[1]);
  return Number.isFinite(idx) ? idx : null;
}

function populateBrickWithImage(brick, normalized) {
  if (!brick || !normalized || !normalized.canonical) return;
  let img = brick.querySelector("img");
  if (!img) {
    img = document.createElement("img");
    img.alt = "";
    img.style.maxWidth = "100%";
    img.style.display = "block";
    brick.innerHTML = "";
    brick.appendChild(img);
  }

  const canonical = normalized.canonical;
  img.src = `${canonical}?v=${Date.now()}`;
  img.dataset.basename = canonical.split("/").pop();
  img.dataset.src = canonical;

  rememberImageSafety(canonical, normalized.isSafe);
  applySafetyOverlay(brick, normalized.isSafe);

  const overlay = brick.querySelector(".loading");
  if (overlay) overlay.remove();
}

function renderImageList(images, options = {}) {
  if (!container) return 0;
  const list = Array.isArray(images) ? images : [];
  const expectedCount = Number.isFinite(options.expectedCount) ? Math.max(0, options.expectedCount) : null;
  const normalizedList = [];
  for (const entry of list) {
    const normalized = normalizeImageEntry(entry);
    const canonical = normalized?.canonical || canonicalizeImageSrc(normalized?.url);
    if (!canonical) continue;
    normalizedList.push({ ...normalized, canonical });
  }

  // If we know more images are expected than currently exist, keep placeholders visible.
  if (expectedCount && expectedCount > normalizedList.length) {
    renderPlaceholders(expectedCount);
    normalizedList.forEach((normalized) => {
      const slotIdx = slotIndexFromCanonical(normalized.canonical);
      let brick = null;
      if (slotIdx !== null) {
        brick = container.querySelector(`.brick[data-slot="${slotIdx}"]`);
      }
      if (!brick) {
        brick = container.querySelector(".brick:not(.filled)") || null;
      }
      if (!brick) {
        brick = document.createElement("div");
        brick.className = "brick";
        container.appendChild(brick);
      }
      brick.classList.add("filled");
      populateBrickWithImage(brick, normalized);
    });
    container.querySelectorAll(".brick.filled").forEach((b) => b.classList.remove("filled"));

    if (rankSection) rankSection.classList.remove("hidden");
    if (nextWrap) nextWrap.classList.remove("hidden");

    setTilesPerRow(expectedCount || 6);
    return expectedCount;
  }

  container.innerHTML = "";
  clearZoomSelection();
  container.style.display = "grid";
  container.style.gap = "16px";
  container.style.overflow = "visible";

  let rendered = 0;
  for (const normalized of normalizedList) {
    const canonical = normalized.canonical;
    const div = document.createElement("div");
    div.className = "brick";
    populateBrickWithImage(div, normalized);
    container.appendChild(div);
    rendered += 1;
  }

  if (rankSection) rankSection.classList.remove("hidden");
  if (nextWrap) nextWrap.classList.remove("hidden");

  setTilesPerRow((expectedCount && expectedCount > rendered) ? expectedCount : (rendered || 6));
  return (expectedCount && expectedCount > rendered) ? expectedCount : rendered;
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
      refreshSafetyFromServer();
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

    const serverExpected = Number.isFinite(data?.expected) ? Number(data.expected) : null;
    if (serverExpected !== null) {
      expected = serverExpected;
      setTilesPerRow(expected || 6);
    }

    if (Number.isFinite(data?.round)) {
      currentRound = Number(data.round);
    }
    if (Number.isFinite(data?.iteration)) {
      setIterationDisplay(Number(data.iteration));
    }

    const imagesFromServer = Array.isArray(data?.images) ? data.images : [];
    received = imagesFromServer.length;

    const targetExpected = serverExpected ?? expected ?? imagesFromServer.length;

    renderImageList(imagesFromServer, { expectedCount: targetExpected });
    refreshSafetyFromServer();

    const inflight = Boolean(data?.inflight) || (targetExpected && received < targetExpected);
    isGenerationInFlight = inflight;
    if (inflight && statusEl && targetExpected) {
      statusEl.textContent = `Loaded ${received}/${targetExpected}`;
    }
  } catch (err) {
    console.warn("stage status refresh failed", err);
  }
}

refreshStageStatus();

async function handleUiRefresh() {
  if (!refreshUiBtn) return;
  if (refreshUiBtn.disabled) return;
  refreshUiBtn.disabled = true;
  const prevStatus = statusEl ? statusEl.textContent : "";
  if (statusEl) statusEl.textContent = "Refreshing images…";
  try {
    await refreshStageStatus();
    if (statusEl) statusEl.textContent = "Refresh requested. Reloading images…";
  } catch (err) {
    if (statusEl) statusEl.textContent = "Refresh failed. Please try again.";
  } finally {
    refreshUiBtn.disabled = false;
    setTimeout(() => {
      if (statusEl && statusEl.textContent && statusEl.textContent.startsWith("Refresh")) {
        statusEl.textContent = prevStatus;
      }
    }, 1800);
  }
}

if (refreshUiBtn) {
  refreshUiBtn.addEventListener("click", handleUiRefresh);
}

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
