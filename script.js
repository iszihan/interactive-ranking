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
      img.src = src;
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
  nextBtn.addEventListener('click', () => {
    // Capture current ranking (left-to-right)
    const order = [...container.children]
      .map(div => (div.querySelector('img') || {}).src)
      .filter(Boolean);
    fetch('/api/next', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ranking: order })
    }).then(r => r.json()).then(data => {
      const images = data.images || [];
      if (images.length) {
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
    }).catch(err => {
      console.error('Next failed', err);
    });
  });
}
