// ---------- element refs ----------
const container   = document.getElementById("container");
const startBtn    = document.getElementById("startBtn");
const statusEl    = document.getElementById("status");
const gallery     = document.getElementById("gallery");
const nextBtn     = document.getElementById("nextBtn");
const nextWrap    = document.getElementById("nextWrap");
const rankSection = document.getElementById("rankSection");
const controls    = document.getElementById("controls");

// hide Next/ranking initially
if (nextWrap)    nextWrap.classList.add("hidden");
if (rankSection) rankSection.classList.add("hidden");

// ---------- drag-to-reorder ----------
let draggingElem = null;
let startX = 0;
let startIndex = 0;
let currentIndex = 0;

const SLOTS_BASE = "/slots"; // make sure this matches server

function indexOfElement(el){ return [...container.children].indexOf(el); }
function getClientX(e){ return e.touches ? e.touches[0].clientX : e.clientX; }

function onMouseDown(e){
  const brick = e.target.closest(".brick");
  if (!brick) return;
  e.preventDefault();

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

function onMouseMove(e){
  e.preventDefault();
  if (!draggingElem) return;

  const deltaX = getClientX(e) - startX;
  draggingElem.style.transform = `translateX(${deltaX}px)`;

  const all = [...container.children];
  const others = all.filter(el => el !== draggingElem);

  function middle(el){ const r = el.getBoundingClientRect(); return r.left + r.width/2; }
  const midX = middle(draggingElem);

  let newIndex = others.findIndex(other => midX < middle(other));
  if (newIndex === -1) newIndex = others.length;

  if (newIndex !== currentIndex){
    currentIndex = newIndex;
    reorderGhosts();
  }
}

function reorderGhosts(){
  const bricks = [...container.children].filter(el => el !== draggingElem);
  const spacing = draggingElem.getBoundingClientRect().width;
  const cs = getComputedStyle(container);
  const gap = parseFloat(cs.columnGap || cs.getPropertyValue?.("column-gap") || cs.gap || "0") || 0;
  const offset = spacing + gap;

  bricks.forEach((b, i) => {
    b.style.transform = "";
    if (startIndex < currentIndex && i >= startIndex && i < currentIndex){
      b.style.transform = `translateX(-${offset}px)`;
    } else if (startIndex > currentIndex && i >= currentIndex && i < startIndex){
      b.style.transform = `translateX(${offset}px)`;
    }
  });
}

function onMouseUp(){
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
}

// ---------- layout helpers ----------
function setTilesPerRow(N){
  // choose a reasonable vw per tile so N tiles are visible-ish; clamp 8–18vw
  const basis = Math.max(8, Math.min(18, Math.floor(80 / Math.max(1, N))));
  document.documentElement.style.setProperty("--tile-ideal", `${basis}vw`);
}

// ---------- start (initial images) ----------
async function startProcess(){
  try {
    startBtn.disabled = true;
    statusEl.textContent = "Running...";
    const resp = await fetch("/api/start", { method: "POST" });
    if (!resp.ok) throw new Error("start failed");
    const data = await resp.json();
    const images = data.images || [];

    if (controls) controls.classList.add("hidden");

    container.innerHTML = "";
    container.style.display = "flex";
    container.style.flexDirection = "row";
    container.style.flexWrap = "nowrap";
    container.style.gap = "12px";
    container.style.overflowX = "auto";

    for (const src of images){
      const div = document.createElement("div");
      div.className = "brick";
      const img = document.createElement("img");

      const canonical = src.split("?")[0]; // strip any cache-buster
      img.src = `${canonical}?v=${Date.now()}`;
      img.alt = "";
      img.style.maxWidth = "100%";
      img.style.display = "block";

      // keep real filename metadata for ranking (no blob:)
      img.dataset.basename = canonical.split("/").pop();
      img.dataset.src = canonical;

      div.appendChild(img);
      container.appendChild(div);
    }

    if (rankSection) rankSection.classList.remove("hidden");
    if (nextWrap) nextWrap.classList.remove("hidden");

    statusEl.textContent = images.length ? `Loaded ${images.length} images` : "No images returned";

    setTilesPerRow(images.length || 6);
  } catch (err){
    statusEl.textContent = "Error: " + (err && err.message ? err.message : "unknown");
  } finally {
    startBtn.disabled = false;
  }
}

if (startBtn) startBtn.addEventListener("click", startProcess);

// ---------- placeholders + per-slot reveal ----------
function renderPlaceholders(n){
  container.innerHTML = "";
  container.style.display = "flex";
  container.style.gap = "12px";
  container.style.overflowX = "auto";

  for (let i = 0; i < n; i++){
    const brick = document.createElement("div");
    brick.className = "brick";
    brick.dataset.slot = String(i);

    const img = new Image();
    img.alt = "";
    img.dataset.basename = `slot-${i}.png`;
    img.dataset.src      = `${SLOTS_BASE}/slot-${i}.png`;
    brick.appendChild(img);

    const overlay = document.createElement("div");
    overlay.className = "loading";
    overlay.innerHTML = `<div class="loading-wrap"><div class="spinner"></div><div class="label">Generating…</div></div>`;
    brick.appendChild(overlay);

    container.appendChild(brick);
  }
}

async function setSlotImage(slot, round){
  const brick   = container.querySelector(`.brick[data-slot="${slot}"]`);
  if (!brick) return;
  const img     = brick.querySelector("img");
  const overlay = brick.querySelector(".loading");

  const canonical = `${SLOTS_BASE}/slot-${slot}.png`;
  img.dataset.basename = `slot-${slot}.png`;
  img.dataset.src      = canonical;
  img.src = `${canonical}?v=${round}&t=${Date.now()}`;

  if (img.decode){ try { await img.decode(); } catch {} }
  overlay.style.display = "none";
}

// ---------- SSE (push) ----------
const es = new EventSource("/api/events");
let currentRound = null;
let expected = 0;
let received = 0;

es.addEventListener("begin", (ev) => {
  const { round, n } = JSON.parse(ev.data);
  currentRound = round;
  expected = n;
  received = 0;

  if (nextBtn) nextBtn.disabled = true;
  // if (rankSection) rankSection.classList.add("hidden");

  renderPlaceholders(n);
  setTilesPerRow(n);

  statusEl.textContent = `Generating… (0/${expected})`;
});
window.addEventListener("resize", () => setTilesPerRow(expected || 6));

es.addEventListener("slot", (ev) => {
  const { round, slot } = JSON.parse(ev.data);
  if (round !== currentRound) return;
  setSlotImage(slot, round);
  received += 1;
  statusEl.textContent = `Loaded ${received}/${expected}`;
  // if (rankSection) rankSection.classList.remove("hidden");
});

es.addEventListener("done", (ev) => {
  const { round } = JSON.parse(ev.data);
  if (round !== currentRound) return;

  statusEl.textContent = `Loaded ${received}/${expected}`;
  if (nextBtn) nextBtn.disabled = false;
});

es.onerror = () => {
  statusEl.textContent = "Stream error.";
  if (nextBtn) nextBtn.disabled = false;
};

// ---------- ranking ----------
function getRanking(){
  const order = [...container.querySelectorAll(".brick img")].map(img =>
    img.dataset.basename ||
    new URL(img.src, location.href).pathname.split("/").pop()
  ).filter(Boolean);

  // sanity check
  console.log("ranking to send:", order);
  return order;
}

// ---------- Next button ----------

if (nextBtn){
  nextBtn.addEventListener("click", async () => {
    const order = getRanking();
    nextBtn.disabled = true;
    statusEl.textContent = "Starting…";

    // show skeleton now (guess N from current tiles or default)
    const nGuess = container.querySelectorAll(".brick").length || 6;
    renderPlaceholders(nGuess);
    setTilesPerRow(nGuess);

    try{
      await fetch("/api/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ranking: order })
      });
      // SSE will drive begin/slot/done
    } catch {
      statusEl.textContent = "Next failed.";
      nextBtn.disabled = false;
    }
  });
}
