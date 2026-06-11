const form = document.querySelector("#research-form");
const queryInput = document.querySelector("#query");
const runButton = document.querySelector("#run-button");
const sampleButton = document.querySelector("#sample-button");
const serviceStatus = document.querySelector("#service-status");
const resultState = document.querySelector("#result-state");
const reportOutput = document.querySelector("#report-output");
const citationsList = document.querySelector("#citations-list");
const metricsGrid = document.querySelector("#metrics");
const runId = document.querySelector("#run-id");
const stages = Array.from(document.querySelectorAll(".stage"));

const metricLabels = {
  model: "Model",
  latency_seconds: "Latency",
  retry_count: "Retries",
  num_sub_questions: "Sub-questions",
  num_sources: "Sources",
  num_citations: "Citations",
  total_tokens: "Tokens"
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInline(text) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, '<span class="cite-token">[$1]</span>');
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const blocks = [];
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      continue;
    }
    if (trimmed.startsWith("# ")) {
      flushParagraph();
      blocks.push(`<h1>${renderInline(trimmed.slice(2))}</h1>`);
      continue;
    }
    if (trimmed.startsWith("## ")) {
      flushParagraph();
      blocks.push(`<h2>${renderInline(trimmed.slice(3))}</h2>`);
      continue;
    }
    paragraph.push(trimmed);
  }
  flushParagraph();

  return blocks.join("");
}

function setStageState(state) {
  stages.forEach((stage, index) => {
    stage.classList.remove("active", "done");
    if (state === "running" && index === 0) {
      stage.classList.add("active");
    }
    if (state === "done") {
      stage.classList.add("done");
    }
  });
}

function formatMetricValue(key, value) {
  if (value === undefined || value === null || value === "") return "-";
  if (key === "latency_seconds") return `${value}s`;
  if (key === "total_tokens" && Number.isFinite(Number(value))) {
    return Number(value).toLocaleString();
  }
  return String(value);
}

function renderMetrics(metadata = {}) {
  const flattened = {
    ...metadata,
    total_tokens: metadata.token_usage?.total_tokens
  };
  const keys = [
    "model",
    "latency_seconds",
    "retry_count",
    "num_sub_questions",
    "num_sources",
    "num_citations",
    "total_tokens"
  ];
  metricsGrid.innerHTML = keys
    .map((key) => {
      const value = formatMetricValue(key, flattened[key]);
      return `<div class="metric"><span>${metricLabels[key]}</span><strong>${escapeHtml(value)}</strong></div>`;
    })
    .join("");
}

function renderCitations(citations = []) {
  if (!citations.length) {
    citationsList.innerHTML = '<li>No citations returned.</li>';
    return;
  }
  citationsList.innerHTML = citations
    .map((url) => {
      const safeUrl = escapeHtml(url);
      return `<li><a href="${safeUrl}" target="_blank" rel="noreferrer">${safeUrl}</a></li>`;
    })
    .join("");
}

function renderResponse(payload, stateText) {
  reportOutput.innerHTML = renderMarkdown(payload.report);
  renderCitations(payload.citations);
  renderMetrics(payload.metadata);
  runId.textContent = payload.run_id || "run";
  resultState.textContent = stateText;
  resultState.classList.remove("error", "muted");
  resultState.classList.add("muted");
  setStageState("done");
}

function setBusy(isBusy) {
  runButton.disabled = isBusy;
  sampleButton.disabled = isBusy;
  resultState.textContent = isBusy ? "running" : resultState.textContent;
  if (isBusy) setStageState("running");
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    serviceStatus.textContent = `Online: ${health.model}`;
    serviceStatus.classList.remove("error", "muted");
  } catch (error) {
    serviceStatus.textContent = "Service offline";
    serviceStatus.classList.add("error");
  }
}

async function loadSample() {
  setBusy(true);
  try {
    const response = await fetch("/sample-response");
    if (!response.ok) throw new Error(`Sample failed with HTTP ${response.status}`);
    const payload = await response.json();
    renderResponse(payload, "sample loaded");
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function runResearch(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (query.length < 3) return;

  setBusy(true);
  resultState.classList.remove("error", "muted");
  resultState.textContent = "running";

  try {
    const response = await fetch("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Research failed with HTTP ${response.status}`);
    }
    renderResponse(payload, "complete");
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

function showError(error) {
  resultState.textContent = "error";
  resultState.classList.add("error");
  reportOutput.innerHTML = `<div class="empty-state">${escapeHtml(error.message || error)}</div>`;
  citationsList.innerHTML = "";
  setStageState("");
}

form.addEventListener("submit", runResearch);
sampleButton.addEventListener("click", loadSample);

refreshHealth();
loadSample();
