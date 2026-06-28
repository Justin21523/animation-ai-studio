import "./styles.css";

const project = {
  "slug": "animation-ai-studio",
  "title": "Animation AI Studio: A Demoable AI Animation Workflow Platform",
  "summary": "Animation AI Studio is a local-first AI animation workflow platform. It has been shaped from a research-heavy script collection into a portfolio-ready demo with a GitHub Pages project site, a runnable mock-safe Web UI, seeded demo scenarios, screenshots, and a WebM walkthrough. Users can understand the project from the public page first, then run the local demo mode to inspect FastAPI, SQLite, job orchestration, result browsing, and system monitoring.",
  "category": "ai-data",
  "year": 2026,
  "status": "completed",
  "technologies": [
    "Python",
    "FastAPI",
    "SQLite",
    "Vanilla JavaScript",
    "Server-Sent Events",
    "Pydantic",
    "PyTorch",
    "Diffusers",
    "ComfyUI",
    "FFmpeg",
    "GitHub Pages"
  ],
  "githubUrl": "https://github.com/Justin21523/animation-ai-studio",
  "readmeUrl": "https://github.com/Justin21523/animation-ai-studio#readme",
  "problem": "AI animation projects often remain hard to evaluate because they depend on GPU hardware, model weights, ComfyUI, API keys, media assets, and many batch scripts. That is weak for portfolio walkthroughs: users cannot quickly understand the value or reliably see a working result. This project solves the presentation problem by turning a research pipeline into a demoable, screenshot-ready, recordable, and honestly scoped portfolio project.",
  "solution": "I built a dual-mode architecture. Demo mode uses repo-local SQLite and outputs/demo with a deterministic mock runner that writes logs, summaries, storyboard manifests, quality reports, and gallery previews without private keys, GPU access, or model files. Full mode preserves the integration path to ComfyUI, image providers, TTS, FFmpeg, and batch scripts. The Web UI is a Vanilla JS SPA served by FastAPI, with Jobs, Action, Image, Creative, Results, and System pages. The backend uses FastAPI routers, a SQLite job database, and a JobService layer for execution and progress tracking. A separate GitHub Pages site presents the product in the first viewport and includes screenshots, a WebM demo, architecture notes, and local run instructions.",
  "architecture": "This case study is generated from the portfolio catalog pipeline using README, Git metadata, package/build configuration, and media signals. The final architecture narrative still needs source-level review. Current detected technology signals include: Python, FastAPI, SQLite, Vanilla JavaScript, Server-Sent Events, Pydantic, PyTorch, Diffusers, ComfyUI, FFmpeg, GitHub Pages.",
  "setupGuide": "This project does not expose a verified runnable web command yet. Review the README/source tree and add exact install, run, test, and build commands before interview use.\nNo verified build command was detected. Treat the current portfolio page as a case-study placeholder until build steps are reviewed.",
  "features": [
    "Detected technical signals: Python, FastAPI, SQLite, Vanilla JavaScript, Server-Sent Events, Pydantic, PyTorch, Diffusers, ComfyUI, FFmpeg, GitHub Pages,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability",
    "I built a dual-mode architecture",
    "Demo mode uses repo-local SQLite and outputs/demo with a deterministic mock runner that writes logs, summaries, storyboard manifests, quality reports, and gallery previews without private keys, GPU access, or model files",
    "Full mode preserves the integration path to ComfyUI, image providers, TTS, FFmpeg, and batch scripts",
    "The Web UI is a Vanilla JS SPA served by FastAPI, with Jobs, Action, Image, Creative, Results, and System pages",
    "The backend uses FastAPI routers, a SQLite job database, and a JobService layer for execution and progress tracking"
  ],
  "metrics": [
    {
      "label": "Demo Modules",
      "value": "6"
    },
    {
      "label": "Tech Stack",
      "value": "11"
    },
    {
      "label": "Mode",
      "value": "Fixture"
    },
    {
      "label": "Status",
      "value": "completed"
    }
  ],
  "records": [
    {
      "id": "flow-01",
      "name": "Detected technical signals: Python, FastAPI, SQLite, Vanilla JavaScript, Server-Sent Events, Pydantic, PyTorch, Diffusers, ComfyUI, FFmpeg, GitHub Pages,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability",
      "status": "Ready",
      "owner": "Frontend"
    },
    {
      "id": "flow-02",
      "name": "I built a dual-mode architecture",
      "status": "Review",
      "owner": "Data"
    },
    {
      "id": "flow-03",
      "name": "Demo mode uses repo-local SQLite and outputs/demo with a deterministic mock runner that writes logs, summaries, storyboard manifests, quality reports, and gallery previews without private keys, GPU access, or model files",
      "status": "Queued",
      "owner": "Automation"
    },
    {
      "id": "flow-04",
      "name": "Full mode preserves the integration path to ComfyUI, image providers, TTS, FFmpeg, and batch scripts",
      "status": "Ready",
      "owner": "Product"
    },
    {
      "id": "flow-05",
      "name": "The Web UI is a Vanilla JS SPA served by FastAPI, with Jobs, Action, Image, Creative, Results, and System pages",
      "status": "Review",
      "owner": "Quality"
    }
  ]
};

const state = {
  tab: "overview",
  query: "",
  selected: project.records[0]?.id ?? "",
};

function matches(record) {
  const q = state.query.trim().toLowerCase();
  if (!q) return true;
  return [record.name, record.status, record.owner].join(" ").toLowerCase().includes(q);
}

function renderMetrics() {
  return project.metrics.map((metric) => `
    <div class="metric">
      <span>${metric.label}</span>
      <strong>${metric.value}</strong>
    </div>
  `).join("");
}

function renderTabs() {
  return ["overview", "workflow", "data", "architecture"].map((tab) => `
    <button class="tab ${state.tab === tab ? "active" : ""}" data-tab="${tab}">${tab}</button>
  `).join("");
}

function renderOverview() {
  return `
    <section class="panel hero-panel">
      <div>
        <p class="eyebrow">${project.category} · ${project.year}</p>
        <h1>${project.title}</h1>
        <p class="lead">${project.summary}</p>
      </div>
      <div class="metrics">${renderMetrics()}</div>
    </section>
    <section class="panel split">
      <div>
        <h2>Problem</h2>
        <p>${project.problem}</p>
      </div>
      <div>
        <h2>Solution</h2>
        <p>${project.solution}</p>
      </div>
    </section>
  `;
}

function renderWorkflow() {
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Demo workflow</p>
          <h2>Interactive Review Flow</h2>
        </div>
        <button id="runDemo" class="primary">Run demo pass</button>
      </div>
      <div class="timeline">
        ${project.features.map((feature, index) => `
          <article class="step">
            <span>${String(index + 1).padStart(2, "0")}</span>
            <p>${feature}</p>
          </article>
        `).join("")}
      </div>
      <output id="demoOutput" class="output">Ready to run the guided demo.</output>
    </section>
  `;
}

function renderData() {
  const rows = project.records.filter(matches);
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Fixture data</p>
          <h2>Sample Records</h2>
        </div>
        <input id="search" value="${state.query}" placeholder="Filter records" />
      </div>
      <div class="table">
        ${rows.map((record) => `
          <button class="row ${state.selected === record.id ? "selected" : ""}" data-record="${record.id}">
            <span>${record.id}</span>
            <strong>${record.name}</strong>
            <em>${record.owner}</em>
            <b>${record.status}</b>
          </button>
        `).join("") || `<p class="empty">No records match this filter.</p>`}
      </div>
    </section>
  `;
}

function renderArchitecture() {
  return `
    <section class="panel split">
      <div>
        <p class="eyebrow">Architecture</p>
        <h2>How the demo is organized</h2>
        <p>${project.architecture}</p>
        <pre>demo-app/
  src/main.js
  src/styles.css
  index.html
  package.json</pre>
      </div>
      <div>
        <p class="eyebrow">Run guide</p>
        <h2>Local commands</h2>
        <pre>${project.setupGuide}</pre>
        <div class="chips">${project.technologies.slice(0, 12).map((tech) => `<span>${tech}</span>`).join("")}</div>
      </div>
    </section>
  `;
}

function render() {
  const views = {
    overview: renderOverview,
    workflow: renderWorkflow,
    data: renderData,
    architecture: renderArchitecture,
  };
  document.querySelector("#app").innerHTML = `
    <header class="topbar">
      <a href="${project.githubUrl}" class="brand">${project.title}</a>
      <nav>${renderTabs()}</nav>
      <a class="repo" href="${project.readmeUrl}">README</a>
    </header>
    <main>${views[state.tab]()}</main>
  `;

  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.tab = button.dataset.tab;
      render();
    });
  });
  document.querySelector("#search")?.addEventListener("input", (event) => {
    state.query = event.target.value;
    render();
    document.querySelector("#search")?.focus();
  });
  document.querySelectorAll("[data-record]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selected = button.dataset.record;
      render();
    });
  });
  document.querySelector("#runDemo")?.addEventListener("click", () => {
    const output = document.querySelector("#demoOutput");
    if (output) output.textContent = `${project.title}: ${project.records.length} fixture records processed and ${project.features.length} workflow checks completed.`;
  });
}

render();
