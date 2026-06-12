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

// Client-side guard so a stalled /research request never leaves the console stuck.
const RESEARCH_TIMEOUT_MS = 10 * 60 * 1000;

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

// Turns [n] markers in ALREADY-ESCAPED text into anchor pills that jump to the
// matching entry in the citations panel (renderCitations assigns the ids).
function renderCiteTokens(escapedText) {
  return escapedText.replace(
    /\[(\d+)\]/g,
    '<a class="cite-token" href="#citation-$1" title="Jump to citation $1">[$1]</a>'
  );
}

// Escapes the text, then linkifies bare http(s) URLs and renders [n] citation
// pills. URL detection runs on escaped text so the href can never break out of
// its attribute, and the scheme is restricted to http(s).
function renderInline(text) {
  const escaped = escapeHtml(text);
  const urlPattern = /\bhttps?:\/\/\S+/g;
  let html = "";
  let cursor = 0;
  for (const match of escaped.matchAll(urlPattern)) {
    // Keep sentence punctuation out of the link target.
    const trailing = (match[0].match(/[.,)]+$/) || [""])[0];
    const url = match[0].slice(0, match[0].length - trailing.length);
    html += renderCiteTokens(escaped.slice(cursor, match.index));
    html += `<a href="${url}" target="_blank" rel="noreferrer">${url}</a>${trailing}`;
    cursor = match.index + match[0].length;
  }
  html += renderCiteTokens(escaped.slice(cursor));
  return html;
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const blocks = [];
  let paragraph = [];
  let listTag = null; // "ol" | "ul" while a list is open
  let listStart = 1;
  let listItems = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    blocks.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!listItems.length) return;
    const startAttr = listTag === "ol" && listStart !== 1 ? ` start="${listStart}"` : "";
    blocks.push(`<${listTag}${startAttr}>${listItems.join("")}</${listTag}>`);
    listItems = [];
    listTag = null;
    listStart = 1;
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const ordered = trimmed.match(/^(\d+)[.)]\s+(.*)$/);
    if (ordered) {
      flushParagraph();
      if (listTag !== "ol") {
        flushList();
        listTag = "ol";
        listStart = Number(ordered[1]) || 1;
      }
      listItems.push(`<li>${renderInline(ordered[2])}</li>`);
      continue;
    }

    const unordered = trimmed.match(/^[-*+]\s+(.*)$/);
    if (unordered) {
      flushParagraph();
      if (listTag !== "ul") {
        flushList();
        listTag = "ul";
      }
      listItems.push(`<li>${renderInline(unordered[1])}</li>`);
      continue;
    }

    // A plain line ends any open list before joining the running paragraph.
    flushList();
    paragraph.push(trimmed);
  }
  flushParagraph();
  flushList();

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

function setResultState(text, variant) {
  resultState.textContent = text;
  resultState.classList.remove("error", "muted", "success", "running");
  if (variant) resultState.classList.add(variant);
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
    citationsList.innerHTML = "<li>No citations returned.</li>";
    return;
  }
  citationsList.innerHTML = citations
    .map((url, index) => {
      const safeUrl = escapeHtml(url);
      // Citation URLs are LLM output synthesised from scraped web content, so
      // only linkify http(s) schemes (mirroring renderInline); anything else
      // (e.g. javascript:) renders as inert text instead of a clickable link.
      const body = /^https?:\/\//i.test(String(url).trim())
        ? `<a href="${safeUrl}" target="_blank" rel="noreferrer">${safeUrl}</a>`
        : safeUrl;
      return `<li id="citation-${index + 1}">${body}</li>`;
    })
    .join("");
}

function renderResponse(payload, stateText, variant = "muted") {
  reportOutput.innerHTML = renderMarkdown(payload.report);
  renderCitations(payload.citations);
  renderMetrics(payload.metadata);
  runId.textContent = payload.run_id || "run";
  setResultState(stateText, variant);
  setStageState("done");
}

function setBusy(isBusy) {
  runButton.disabled = isBusy;
  sampleButton.disabled = isBusy;
  if (isBusy) {
    setResultState("running", "running");
    setStageState("running");
  }
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
    renderResponse(payload, "sample loaded", "muted");
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

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), RESEARCH_TIMEOUT_MS);

  try {
    const response = await fetch("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (parseError) {
      payload = null; // Non-JSON body (e.g. a gateway error page).
    }
    if (!response.ok) {
      const detail = typeof payload?.detail === "string" ? payload.detail : "";
      throw new Error(detail || `Research failed with HTTP ${response.status}`);
    }
    if (!payload) {
      throw new Error("The server returned an unreadable response.");
    }
    renderResponse(payload, "complete", "success");
  } catch (error) {
    if (error.name === "AbortError") {
      showError(
        new Error(
          "The request timed out after 10 minutes. The pipeline may still be " +
            "running server-side — please retry, or use Load Sample to view a saved run."
        )
      );
    } else {
      showError(error);
    }
  } finally {
    clearTimeout(timeoutId);
    setBusy(false);
  }
}

function showError(error) {
  setResultState("error", "error");
  reportOutput.innerHTML = `<div class="empty-state">${escapeHtml(error.message || error)}</div>`;
  citationsList.innerHTML = "";
  setStageState("");
}

form.addEventListener("submit", runResearch);
sampleButton.addEventListener("click", loadSample);

refreshHealth();
loadSample();
