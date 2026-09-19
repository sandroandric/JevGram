import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";

// Column order everywhere: AI, Assisted, Human.
const LABEL_ORDER = ["ai_generated", "ai_assisted", "human_written"];
const VIEW_NAMES = [
  ["alone", "Sentence alone"],
  ["neighbors", "With neighbours"],
  ["paragraph", "In its paragraph"],
];

const $ = (id) => document.getElementById(id);
const views = { upload: $("upload-view"), loading: $("loading-view"), result: $("result-view") };

const state = {
  result: null,
  hidden: new Set(),
  minConfidence: 0,
  selected: null,
  highlights: new Map(), // sentence index -> highlight elements
  observer: null,
};

function show(view) {
  for (const [name, element] of Object.entries(views)) element.hidden = name !== view;
  $("reset").hidden = view !== "result";
  window.scrollTo(0, 0);
}

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function topLabel(probabilities) {
  return LABEL_ORDER.reduce((a, b) => (probabilities[a] >= probabilities[b] ? a : b));
}

function stackedBar(bar, probabilities, names) {
  bar.replaceChildren(
    ...LABEL_ORDER.map((key) => {
      const segment = element("span");
      segment.style.width = `${probabilities[key] * 100}%`;
      segment.style.background = `var(--${key})`;
      segment.title = `${names[key]}: ${percent(probabilities[key])}`;
      return segment;
    }),
  );
  return bar;
}

// Upload

const dropzone = $("dropzone");
const fileInput = $("file-input");

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) analyze(fileInput.files[0]);
});
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, () => dropzone.classList.remove("dragging"));
}
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) analyze(file);
});
$("reset").addEventListener("click", reset);

function showUploadError(message) {
  $("upload-error").textContent = message;
  $("upload-error").hidden = false;
  show("upload");
}

async function analyze(file) {
  $("upload-error").hidden = true;
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    showUploadError(`${file.name} is not a PDF. Choose a PDF file.`);
    return;
  }

  $("loading-file").textContent = file.name;
  $("loading-elapsed").textContent = "0 s";
  show("loading");
  const started = performance.now();
  const timer = setInterval(() => {
    $("loading-elapsed").textContent = `${Math.round((performance.now() - started) / 1000)} s`;
  }, 500);

  try {
    const body = new FormData();
    body.append("file", file);
    const [response, pdf] = await Promise.all([
      fetch("/api/analyze", { method: "POST", body }),
      file.arrayBuffer().then((data) => pdfjsLib.getDocument({ data }).promise),
    ]);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `The server returned ${response.status}.`);
    }
    renderResult(payload, pdf);
    show("result");
  } catch (error) {
    showUploadError(error.message || String(error));
  } finally {
    clearInterval(timer);
    fileInput.value = "";
  }
}

function reset() {
  state.observer?.disconnect();
  state.result = null;
  state.selected = null;
  state.hidden.clear();
  state.highlights.clear();
  state.minConfidence = 0;
  $("min-confidence").value = 0;
  $("min-confidence-value").textContent = "0%";
  $("pages").replaceChildren();
  $("detail-body").hidden = true;
  $("detail-empty").hidden = false;
  show("upload");
}

// Result

function renderResult(result, pdf) {
  state.result = result;
  renderVerdict();
  renderFileFacts();
  renderLabelFilter();
  renderPages(pdf);
  applyFilters();
}

function renderVerdict() {
  const { overview, labels } = state.result;
  const overall = overview.overall;

  const name = element("span", "name", labels[overall.label]);
  const share = element("span", "share", percent(overall.probabilities[overall.label]));
  $("verdict-label").replaceChildren(name, share);
  stackedBar($("verdict-bar"), overall.probabilities, labels);

  const paperBasis =
    overview.paper.parts === 1
      ? "One question on the whole text"
      : `The whole text in ${overview.paper.parts} parts, averaged`;
  const levels = [
    ["sentences", "Sentences", `Mean of ${overview.sentences.count} sentences`],
    ["paragraphs", "Paragraphs", `Mean of ${overview.paragraphs.count} paragraphs`],
    ["paper", "Full paper", paperBasis],
    ["overall", "Average", "Overall verdict", "average"],
  ];
  $("verdict-rows").replaceChildren(
    ...levels.map(([key, title, basis, rowClass]) => {
      const probabilities = overview[key].probabilities;
      const row = element("tr", rowClass);
      const head = element("td");
      head.append(element("span", "level", title), element("span", "basis", basis));
      const bar = element("td", "bar");
      bar.append(stackedBar(element("div", "stacked-bar"), probabilities, labels));
      row.append(head, bar);
      const lead = topLabel(probabilities);
      for (const label of LABEL_ORDER) {
        row.append(element("td", label === lead ? "lead" : "", percent(probabilities[label])));
      }
      return row;
    }),
  );
}

function renderFileFacts() {
  const { filename, sentences, pages, model } = state.result;
  $("file-name").textContent = filename;
  const facts = [
    ["sentences", sentences.length],
    [pages.length === 1 ? "page" : "pages", pages.length],
    ["model", model],
  ];
  $("facts").replaceChildren(
    ...facts.map(([term, value]) => {
      const item = element("div");
      item.append(element("dt", "", term), element("dd", "", String(value)));
      return item;
    }),
  );
}

function renderLabelFilter() {
  const { summary } = state.result;
  $("label-filter").replaceChildren(
    ...LABEL_ORDER.map((key) => {
      const item = summary[key];
      const row = element("li");
      const label = element("label");
      const toggle = element("input");
      toggle.type = "checkbox";
      toggle.checked = true;
      const count = element("span", "count", `${item.count} (${percent(item.share)})`);
      count.append(
        element(
          "small",
          "",
          item.mean_confidence === null
            ? "No sentences"
            : `Average confidence ${percent(item.mean_confidence)}`,
        ),
      );
      label.append(toggle, element("span", `chip ${key}`, item.name), count);
      toggle.addEventListener("change", () => {
        if (toggle.checked) state.hidden.delete(key);
        else state.hidden.add(key);
        label.classList.toggle("off", !toggle.checked);
        applyFilters();
      });
      row.append(label);
      return row;
    }),
  );
}

function renderPages(pdf) {
  const container = $("pages");
  const { pages, sentences } = state.result;
  const byPage = pages.map(() => []);
  for (const sentence of sentences) {
    for (const rect of sentence.rects) byPage[rect.page].push([sentence, rect]);
  }

  state.observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting || entry.target.dataset.rendered) continue;
        entry.target.dataset.rendered = "1";
        drawPage(pdf, entry.target);
      }
    },
    { rootMargin: "800px 0px" },
  );

  pages.forEach((page, pageIndex) => {
    const wrapper = element("div", "page");
    wrapper.dataset.page = pageIndex;
    wrapper.style.aspectRatio = `${page.width} / ${page.height}`;

    const overlay = element("div", "page-overlay");
    for (const [sentence, rect] of byPage[pageIndex]) {
      overlay.append(createHighlight(sentence, rect, page));
    }
    const number = element("span", "page-number", `${pageIndex + 1} of ${pages.length}`);

    wrapper.append(element("canvas"), overlay, number);
    container.append(wrapper);
    state.observer.observe(wrapper);
  });
}

async function drawPage(pdf, wrapper) {
  const page = await pdf.getPage(Number(wrapper.dataset.page) + 1);
  const canvas = wrapper.querySelector("canvas");
  const base = page.getViewport({ scale: 1 });
  const scale = (wrapper.clientWidth / base.width) * (window.devicePixelRatio || 1);
  const viewport = page.getViewport({ scale });
  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
}

// Rectangles arrive in PDF points; percentages keep them aligned at any width.
const PADDING = 1;

function createHighlight(sentence, rect, page) {
  const highlight = element("div", `hl ${sentence.label}`);
  highlight.dataset.index = sentence.index;
  highlight.style.left = `${((rect.x0 - PADDING) / page.width) * 100}%`;
  highlight.style.top = `${((rect.y0 - PADDING) / page.height) * 100}%`;
  highlight.style.width = `${((rect.x1 - rect.x0 + 2 * PADDING) / page.width) * 100}%`;
  highlight.style.height = `${((rect.y1 - rect.y0 + 2 * PADDING) / page.height) * 100}%`;
  // Stronger colour for more confident answers.
  highlight.style.setProperty("--strength", (0.16 + 0.34 * sentence.confidence).toFixed(3));

  if (!state.highlights.has(sentence.index)) state.highlights.set(sentence.index, []);
  state.highlights.get(sentence.index).push(highlight);

  highlight.addEventListener("mouseenter", () => setHover(sentence, true));
  highlight.addEventListener("mouseleave", () => setHover(sentence, false));
  highlight.addEventListener("mousemove", moveTooltip);
  highlight.addEventListener("click", () => select(sentence));
  return highlight;
}

function applyFilters() {
  let visible = 0;
  for (const sentence of state.result.sentences) {
    const hidden =
      state.hidden.has(sentence.label) || sentence.confidence < state.minConfidence;
    if (!hidden) visible += 1;
    for (const highlight of state.highlights.get(sentence.index) || []) {
      highlight.classList.toggle("filtered", hidden);
    }
  }
  $("visible-count").textContent =
    `Showing ${visible} of ${state.result.sentences.length} sentences`;
}

$("min-confidence").addEventListener("input", (event) => {
  state.minConfidence = Number(event.target.value) / 100;
  $("min-confidence-value").textContent = `${event.target.value}%`;
  if (state.result) applyFilters();
});

// Hover and selection

const tooltip = $("tooltip");

function setHover(sentence, on) {
  for (const highlight of state.highlights.get(sentence.index)) {
    highlight.classList.toggle("hover", on);
  }
  if (!on) {
    tooltip.hidden = true;
    return;
  }
  const names = state.result.labels;
  tooltip.replaceChildren(
    tooltipRow(names[sentence.label], `${percent(sentence.confidence)} sure`, null, "main"),
    ...LABEL_ORDER.map((key) =>
      tooltipRow(names[key], percent(sentence.probabilities[key]), key),
    ),
  );
  tooltip.hidden = false;
}

function tooltipRow(left, right, swatch, rowClass = "") {
  const row = element("div", `row ${rowClass}`);
  const name = element("span");
  if (swatch) {
    const mark = element("span", "swatch");
    mark.style.background = `var(--${swatch})`;
    name.append(mark);
  }
  name.append(left);
  row.append(name, element("span", "", right));
  return row;
}

function moveTooltip(event) {
  const margin = 14;
  const box = tooltip.getBoundingClientRect();
  let x = event.clientX + margin;
  let y = event.clientY + margin;
  if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - margin;
  if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - margin;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function viewRow(name, probabilities, rowClass = "") {
  const row = element("tr", rowClass);
  row.append(element("td", "", name));
  const lead = topLabel(probabilities);
  for (const key of LABEL_ORDER) {
    row.append(element("td", key === lead ? "lead" : "", percent(probabilities[key])));
  }
  return row;
}

function select(sentence) {
  if (state.selected !== null) {
    for (const highlight of state.highlights.get(state.selected)) {
      highlight.classList.remove("selected");
    }
  }
  state.selected = sentence.index;
  for (const highlight of state.highlights.get(sentence.index)) {
    highlight.classList.add("selected");
  }

  const names = state.result.labels;
  $("detail-empty").hidden = true;
  $("detail-body").hidden = false;
  const chip = $("detail-chip");
  chip.className = `chip ${sentence.label}`;
  chip.textContent = names[sentence.label];
  $("detail-confidence").textContent = `${percent(sentence.confidence)} sure`;
  $("detail-text").textContent = sentence.text;
  $("detail-views").replaceChildren(
    ...VIEW_NAMES.map(([key, name]) => viewRow(name, sentence.views[key])),
    viewRow("Average", sentence.probabilities, "average"),
  );
  $("detail-location").textContent =
    `Sentence ${sentence.index + 1}, page ${sentence.page + 1}`;
}
