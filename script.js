const container = document.getElementById("container");
let draggingElem = null;
let startY = 0;
let startIndex = 0;
let currentIndex = 0;

function indexOfElement(el) {
  return [...container.children].indexOf(el);
}

container.addEventListener("mousedown", (e) => {
  if (!e.target.classList.contains("brick")) return;
  
  draggingElem = e.target;
  startY = e.clientY;
  startIndex = indexOfElement(draggingElem);
  currentIndex = startIndex;

  draggingElem.classList.add("dragging");
  document.addEventListener("mousemove", onMouseMove);
  document.addEventListener("mouseup", onMouseUp);
});

function onMouseMove(e) {
  if (!draggingElem) return;

  const deltaY = e.clientY - startY;
  draggingElem.style.transform = `translateY(${deltaY}px)`;

  const all = [...container.children];
  const others = all.filter(el => el !== draggingElem);

  
  // Get the middle Y position of the dragging element
  function middle(el) {
    const rect = el.getBoundingClientRect();
    return rect.top + (rect.height / 2);
  }
  const midY = middle(draggingElem);

  let newIndex = others.findIndex((other) => {
    const other_midY = middle(other);
    return midY < other_midY;
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
  const spacing = draggingElem.getBoundingClientRect().height;
  const styles = getComputedStyle(draggingElem);
  const marginBottom = parseFloat(styles.marginBottom) || 0;
  const marginTop = parseFloat(styles.marginTop) || 0;


  bricks.forEach((brick, i) => {
    brick.style.transform = "";
    if (startIndex < currentIndex && i >= startIndex && i < currentIndex) {
      // dragging downward
      brick.style.transform = `translateY(-${spacing + marginTop}px)`;
    } else if (startIndex > currentIndex && i >= currentIndex && i < startIndex) {
      // dragging upward
      brick.style.transform = `translateY(${spacing + marginBottom}px)`;
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

  // Let the browser repaint first
  requestAnimationFrame(() => {
    // Then remove the class
    bricks.forEach((b) => {
      b.classList.remove("resetting");
    });
  });

}
