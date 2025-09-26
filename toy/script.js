const container = document.getElementById("container");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const gallery = document.getElementById('gallery');
const nextBtn = document.getElementById('nextBtn');
const nextWrap = document.getElementById('nextWrap');
const rankSection = document.getElementById('rankSection');
const controls = document.getElementById('controls');
// Ensure Next and ranking are hidden on initial load
if (typeof nextWrap !== 'undefined' && nextWrap) nextWrap.classList.add('hidden');
if (typeof rankSection !== 'undefined' && rankSection) rankSection.classList.add('hidden');
let draggingElem = null;
let startX = 0;
let startIndex = 0;
let currentIndex = 0;
let imagesPollHandle = null;

function indexOfElement(el) {
  return [...container.children].indexOf(el);
}

function getClientX(e) {
  return e.touches ? e.touches[0].clientX : e.clientX;
}

function onMouseDown(e)
{
  const brick = e.target.closest('.brick');
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
container.addEventListener("touchstart", onMouseDown, { passive: false }); // passive: false allows preventDefault()

function onMouseMove(e) {
  e.preventDefault();
  if (!draggingElem) return;

  const deltaX = getClientX(e) - startX;
  draggingElem.style.transform = `translateX(${deltaX}px)`;

  const all = [...container.children];
  const others = all.filter(el => el !== draggingElem);

  
  // Get the middle X position of the dragging element
  function middle(el) {
    const rect = el.getBoundingClientRect();
    return rect.left + (rect.width / 2);
  }
  const midX = middle(draggingElem);

  let newIndex = others.findIndex((other) => {
    const other_midX = middle(other);
    return midX < other_midX;
  });
  if (newIndex === -1) newIndex = others.length;

  if (newIndex !== currentIndex) {
    currentIndex = newIndex;
    reorderGhosts();
    // trigger debugger
  }
}

function reorderGhosts() {
  const bricks = [...container.children].filter(el => el !== draggingElem);

  // Dynamically compute total space a brick takes up (including margins)
  const spacing = draggingElem.getBoundingClientRect().width;
  const containerStyles = getComputedStyle(container);
  const columnGap = parseFloat(containerStyles.columnGap || containerStyles.getPropertyValue?.('column-gap') || containerStyles.gap || '0') || 0;
  const offset = spacing + columnGap;


  bricks.forEach((brick, i) => {
    brick.style.transform = "";
    if (startIndex < currentIndex && i >= startIndex && i < currentIndex) {
      // dragging rightward
      brick.style.transform = `translateX(-${offset}px)`;
    } else if (startIndex > currentIndex && i >= currentIndex && i < startIndex) {
      // dragging leftward
      brick.style.transform = `translateX(${offset}px)`;
    }
  });
}


function onMouseUp() {

  const bricks = [...container.children].filter(el => el !== draggingElem);
  bricks.forEach((b) => {
    b.style.transform = "";
    b.classList.add("resetting");
  }
  );

  draggingElem.style.transform = "";
  draggingElem.classList.remove("dragging");

  const referenceNode = bricks[currentIndex] ?? null; // null means append
  container.insertBefore(draggingElem, referenceNode);

  draggingElem = null;
  document.removeEventListener("mousemove", onMouseMove);
  document.removeEventListener("mouseup", onMouseUp);
  document.removeEventListener("touchmove", onMouseMove);
  document.removeEventListener("touchend", onMouseUp);

  // Let the browser repaint first, and then really let it and then finish reset
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      // Then remove the class
      bricks.forEach((b) => {
        b.classList.remove("resetting");
      });
    });
  });

}

async function startProcess() {
  try {
    startBtn.disabled = true;
    statusEl.textContent = 'Running...';
    const resp = await fetch('/api/start', { method: 'POST' });
    if (!resp.ok) throw new Error('start failed');
    const data = await resp.json();
    const images = data.images || [];
    // Hide Start controls once we have results
    if (controls) controls.classList.add('hidden');
    // Build draggable ranking immediately
    container.innerHTML = '';
    container.style.display = 'flex';
    container.style.flexDirection = 'row';
    container.style.flexWrap = 'nowrap';
    container.style.gap = '12px';
    container.style.overflowX = 'auto';
    for (const src of images) {
      const div = document.createElement('div');
      div.className = 'brick';
      const img = document.createElement('img');
      img.src = `${src}?v=${Date.now()}`;
      img.alt = '';
      img.style.maxWidth = '100%';
      img.style.display = 'block';
      div.appendChild(img);
      container.appendChild(div);
    }
    // Show ranking section and Next button
    rankSection.classList.remove('hidden');
    nextWrap.classList.remove('hidden');
    statusEl.textContent = images.length ? `Loaded ${images.length} images` : 'No images returned';
  } catch (err) {
    statusEl.textContent = 'Error: ' + (err && err.message ? err.message : 'unknown');
  } finally {
    startBtn.disabled = false;
  }
}

function renderPlaceholders(n) {
  container.innerHTML = '';
  container.style.display = 'flex';
  container.style.flexDirection = 'row';
  container.style.flexWrap = 'nowrap';
  container.style.gap = '12px';
  container.style.overflowX = 'auto';

  for (let i = 0; i < n; i++) {
    const brick = document.createElement('div');
    brick.className = 'brick';
    brick.dataset.slot = String(i);

    const img = new Image(); // empty for now
    img.alt = '';
    brick.appendChild(img);

    const overlay = document.createElement('div');
    overlay.className = 'loading';
    overlay.innerHTML = `<div style="display:flex;align-items:center;">
        <div class="spinner"></div> Loading…
      </div>`;
    brick.appendChild(overlay);

    container.appendChild(brick);
  }
}

function getSlotEl(slot) {
  return container.querySelector(`.brick[data-slot="${slot}"]`);
}

function setSlotImage(slot, round) {
  const brick = getSlotEl(slot);
  if (!brick) return;
  const img = brick.querySelector('img');
  const overlay = brick.querySelector('.loading');

  const url = `/outputs/slots/slot-${slot}.png?v=${round}`;
  img.src = url;

  // ensure the bitmap is decoded before hiding the overlay
  const onReady = () => { overlay.style.display = 'none'; };
  if (img.decode) {
    img.decode().then(onReady).catch(() => onReady());
  } else {
    img.onload = onReady;
    img.onerror = () => { overlay.textContent = 'Failed'; };
  }
}

// ---- SSE push channel (no polling) ---------------------------------
const es = new EventSource('/api/events');

let currentRound = null;
let expected = 0;
let received = 0;

es.addEventListener('begin', (ev) => {
  const { round, n } = JSON.parse(ev.data);
  currentRound = round;
  expected = n;
  received = 0;

  renderPlaceholders(n);              // <-- create 6 loading slots
  if (rankSection) rankSection.classList.add('hidden');
  statusEl.textContent = `Generating… (0/${expected})`;
});

es.addEventListener('slot', (ev) => {
  const { round, slot } = JSON.parse(ev.data);
  if (round !== currentRound) return; // ignore stale
  setSlotImage(slot, round);          // <-- fill this slot only
  received += 1;
  statusEl.textContent = `Loaded ${received}/${expected}`;
});

es.addEventListener('done', (ev) => {
  const { round } = JSON.parse(ev.data);
  if (round !== currentRound) return;
  statusEl.textContent = `Loaded ${received}/${expected}`;
  if (nextBtn) nextBtn.disabled = false;
  if (rankSection) rankSection.classList.remove('hidden');
});

es.onerror = () => {
  statusEl.textContent = 'Stream error.';
  if (nextBtn) nextBtn.disabled = false;
};

if (startBtn) {
  startBtn.addEventListener('click', startProcess);
}

function buildRankingFromImages() {
  const images = JSON.parse(container.dataset.images || '[]');
  container.innerHTML = '';
  for (const src of images) {
    const div = document.createElement('div');
    div.className = 'brick';
    const img = document.createElement('img');
    img.src = src;
    img.alt = '';
    img.style.maxWidth = '100%';
    img.style.display = 'block';
    div.appendChild(img);
    container.appendChild(div);
  }
}

if (nextBtn) {
  nextBtn.addEventListener('click', async () => {
    // Build current ranking from the draggable container
    const order = [...container.children]
      .map(div => (div.querySelector('img') || {}).src)
      .filter(Boolean);

    nextBtn.disabled = true;
    statusEl.textContent = 'Starting…';

    // Kick off a new round; server will emit begin/slot/done over SSE
    try {
      await fetch('/api/next', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ranking: order, n: undefined }) // or set n explicitly
      });
      // no need to do anything else—SSE handlers will update the UI
    } catch (err) {
      console.error('Next failed', err);
      statusEl.textContent = 'Next failed.';
      nextBtn.disabled = false;
    }
  });
}
