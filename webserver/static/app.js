// RFoldseek search page: form -> POST /api/search -> /result/<id>.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let health = null;
let fileBlob = null;          // uploaded File, or an example fetched as a Blob
let fileName = null;
let job = null;               // current result record
let activeDb = null;
let openRank = null;
let viewer = null;

// ---------------------------------------------------------------- errors
function showError(msg) {
  const el = $("error");
  el.textContent = msg;
  el.hidden = !msg;
  if (msg) window.scrollTo({ top: 0, behavior: "smooth" });
}

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) msg = j.detail; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return r;
}

// ---------------------------------------------------------------- search form
function setFile(blob, name) {
  fileBlob = blob; fileName = name;
  $("chip").hidden = !blob;
  $("chip-name").textContent = name || "";
  $("text").value = "";
  $("text").disabled = !!blob;
  $("text").placeholder = blob ? "" : $("text").dataset.placeholder;
  updateGo();
}

function selectedDbs() {
  return [...document.querySelectorAll("#dbs input:checked")].map((i) => i.value);
}

function updateGo() {
  const hasQuery = !!fileBlob || $("text").value.trim().length > 0;
  const hasDb = selectedDbs().length > 0;
  $("go").disabled = !(health && health.ready && hasQuery && hasDb);
  $("go-hint").textContent = !health ? "" : !health.ready ? "Server is still loading databases…"
    : !hasQuery ? "Add a query structure." : !hasDb ? "Select at least one database." : "";
}

function renderDbs() {
  const box = $("dbs");
  if (!health.ready) {
    box.innerHTML = `<div class="hint"><span class="spinner"></span>${esc(health.error || "Loading databases…")}</div>`;
    return;
  }
  box.innerHTML = health.databases.map((d, i) => `
    <label class="db ${i === 0 ? "on" : ""}">
      <input type="checkbox" value="${esc(d.key)}" ${i === 0 ? "checked" : ""}>
      <div>
        <div class="name">${esc(d.label)}</div>
        <div class="meta">${d.n_chains.toLocaleString()} chains${d.built ? ` · built ${esc(d.built)}` : ""}
          ${d.evalue_calibrated ? "" : '<span class="badge warn" title="This database has no empirical E-value calibration yet; E-values rank hits but are not calibrated significance.">E-value uncalibrated</span>'}</div>
        ${d.source ? `<div class="meta">${esc(d.source)}</div>` : ""}
      </div>
    </label>`).join("");
  box.querySelectorAll("input").forEach((inp) => inp.addEventListener("change", () => {
    inp.closest(".db").classList.toggle("on", inp.checked);
    updateGo();
  }));
}

async function loadHealth() {
  try {
    health = await (await api("/api/health")).json();
  } catch (e) {
    $("dbs").innerHTML = `<div class="hint">Server unreachable: ${esc(e.message)}</div>`;
    return;
  }
  renderDbs();
  $("max-upload").textContent = `up to ${health.max_upload_mb} MB`;
  if (!health.usalign) {
    const tm = $("mode-tm");
    tm.classList.add("off");
    tm.title = "US-align is not installed on this server";
    tm.querySelector("input").disabled = true;
  }
  $("top_n").max = health.top_n_cap;
  updateGo();
  if (!health.ready) setTimeout(loadHealth, 4000);
}

async function loadExamples() {
  let list = [];
  try { list = await (await api("/api/examples")).json(); } catch (_) { /* optional */ }
  if (!list.length) { $("ex-btn").hidden = true; return; }
  $("ex-list").innerHTML = list.map((e, i) =>
    `<button type="button" data-i="${i}">${esc(e.name)}<small>${esc(e.detail)}</small></button>`).join("");
  $("ex-list").querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    const e = list[+b.dataset.i];
    $("ex-list").hidden = true;
    const blob = await (await api(`/static/examples/${e.file}`)).blob();
    setFile(blob, `${e.file} — ${e.name}`);
  }));
}

function wireForm() {
  const t = $("text");
  t.dataset.placeholder = t.placeholder;
  t.addEventListener("input", updateGo);
  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) setFile(f, f.name);
    e.target.value = "";
  });
  $("chip-x").addEventListener("click", () => setFile(null, null));
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f, f.name);
  });
  $("ex-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    $("ex-list").hidden = !$("ex-list").hidden;
  });
  document.addEventListener("click", () => { $("ex-list").hidden = true; });
  document.querySelectorAll("#mode label").forEach((l) => l.addEventListener("click", () => {
    if (l.classList.contains("off")) return;
    document.querySelectorAll("#mode label").forEach((x) => x.classList.toggle("on", x === l));
  }));
  $("form").addEventListener("submit", submit);
}

async function submit(e) {
  e.preventDefault();
  showError("");
  const fd = new FormData();
  if (fileBlob) {
    // examples arrive as Blobs; name them by their file so the server
    // picks the right parser from the extension
    fd.append("structure", fileBlob, fileName.split(" — ")[0]);
  } else {
    fd.append("text", $("text").value);
  }
  const dbs = selectedDbs();
  fd.append("databases", dbs.join(","));
  fd.append("rerank", document.querySelector('#mode input:checked').value);
  for (const k of ["top_k", "top_n", "min_diag"]) fd.append(k, $(k).value);

  showView("result");
  $("res").hidden = true;
  $("res-loading").hidden = false;
  $("res-loading-text").textContent =
    `Searching ${dbs.length} database${dbs.length > 1 ? "s" : ""}…`;
  $("go").disabled = true;
  try {
    job = await (await api("/api/search", { method: "POST", body: fd })).json();
    history.pushState({}, "", `/result/${job.id}`);
    renderResult();
  } catch (err) {
    showView("search");
    showError(`Search failed: ${err.message}`);
  } finally {
    $("res-loading").hidden = true;
    updateGo();
  }
}

// ---------------------------------------------------------------- results
function showView(which) {
  $("view-search").hidden = which !== "search";
  $("view-result").hidden = which !== "result";
}

async function loadResult(id) {
  showView("result");
  $("res").hidden = true;
  $("res-loading").hidden = false;
  $("res-loading-text").textContent = "Loading result…";
  try {
    job = await (await api(`/api/result/${encodeURIComponent(id)}`)).json();
    readHash();
    renderResult();
  } catch (err) {
    showError(`Could not load this result: ${err.message}`);
  } finally {
    $("res-loading").hidden = true;
  }
}

const fmtE = (e) => e == null ? "—" : e === 0 ? "0" : e < 1e-3 || e >= 1e4
  ? e.toExponential(1).replace("e", "e") : e < 1 ? e.toPrecision(2) : e.toFixed(1);
const pct = (x) => x == null ? "—" : `${Math.round(100 * x)}%`;

// "#<db>/<rank>" names an open hit, so a single alignment can be shared
function setHash() {
  const h = openRank ? `#${activeDb}/${openRank}` : "";
  history.replaceState(history.state, "", location.pathname + h);
}

function readHash() {
  const m = location.hash.match(/^#([^/]+)\/(\d+)$/);
  if (m && job.results[decodeURIComponent(m[1])]) {
    activeDb = decodeURIComponent(m[1]);
    openRank = +m[2];
  }
}

function renderResult() {
  const q = job.query;
  $("res-title").textContent = q.filename;
  const p = job.params;
  $("res-sub").textContent = `${q.n_tokens} nucleotides encoded · ` +
    `${p.rerank === "tm" ? "RFoldseek-TM" : "RFoldseek"} · searched ${job.created.replace("T", " ").slice(0, 16)}`;
  $("dl-tsv").href = `/api/result/${job.id}/download?format=tsv`;
  $("dl-json").href = `/api/result/${job.id}/download?format=json`;
  const keys = Object.keys(job.results);
  if (!keys.includes(activeDb)) activeDb = keys[0];
  $("tabs").innerHTML = keys.map((k) => {
    const r = job.results[k];
    return `<button class="tab ${k === activeDb ? "on" : ""}" data-k="${esc(k)}">${esc(r.database.label)}<span class="n">${r.hits.length}</span></button>`;
  }).join("");
  $("tabs").querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
    activeDb = b.dataset.k; openRank = null; setHash(); renderResult();
  }));
  $("res").hidden = false;
  renderTable();
}

function covBar(a, qlen) {
  if (!a || !a.n_aligned) return "";
  const W = 150, H = 12, x0 = ((a.q_start - 1) / qlen) * W, x1 = (a.q_end / qlen) * W;
  const op = 0.35 + 0.65 * Math.min(1, a.seq_identity ?? a.token_identity ?? 0);
  return `<svg width="${W}" height="${H}" aria-label="query ${a.q_start}–${a.q_end} of ${qlen}">
    <rect x="0" y="4" width="${W}" height="4" rx="2" fill="#efebe6"/>
    <rect x="${x0.toFixed(1)}" y="1.5" width="${Math.max(2, x1 - x0).toFixed(1)}" height="9" rx="3"
          fill="#d9703a" fill-opacity="${op.toFixed(2)}"/></svg>`;
}

// NuFold / RNAcentral models are named URS<id>_<taxid>_rank_<n>
const URS = /^(URS[0-9A-F]{10})_(\d+)(?:_rank_(\d+))?/;

function targetCell(h) {
  const u = !h.pdb && h.target.match(URS);
  if (u) {
    return `<a class="tgt" href="https://rnacentral.org/rna/${u[1]}/${u[2]}" target="_blank"
      onclick="event.stopPropagation()">${u[1]}</a><span class="tgt"> · taxon ${u[2]}${u[3] ? ` · model ${u[3]}` : ""}</span>`;
  }
  if (h.pdb) {
    const chain = h.auth_asym && h.auth_asym !== h.chain ? `${h.chain} (auth ${h.auth_asym})` : h.chain;
    return `<a class="tgt" href="https://www.rcsb.org/structure/${esc(h.pdb)}" target="_blank"
      onclick="event.stopPropagation()">${esc(h.pdb)}</a><span class="tgt"> · ${esc(chain)}</span>`;
  }
  return `<span class="tgt">${esc(h.target)}</span>`;
}

function renderTable() {
  const res = job.results[activeDb];
  const tm = res.rerank === "tm";
  const cal = res.database.evalue_calibrated;
  $("tab-meta").textContent = `${res.database.n_chains.toLocaleString()} chains searched in ` +
    `${res.seconds.toFixed(2)} s` + (res.database.source ? ` · ${res.database.source}` : "") +
    (res.database.built ? ` · built ${res.database.built}` : "");
  const head = `<tr><th>#</th><th>Target</th><th>Description</th>
    <th class="num">Seq. id.</th><th class="num">RMSD</th>
    ${tm ? '<th class="num">TM-score</th>' : `<th class="num">E-value${cal ? "" : "*"}</th><th class="num">Bits</th>`}
    <th>Query coverage</th></tr>`;
  const rows = res.hits.map((h) => {
    const a = h.aln || {};
    return `<tr class="hit ${h.rank === openRank ? "open" : ""}" data-rank="${h.rank}">
      <td class="rank">${h.rank}</td>
      <td>${targetCell(h)}</td>
      <td class="desc"><div class="t" title="${esc(h.title || "")}">${esc(h.desc || h.title || (URS.test(h.target) ? "Predicted structure (NuFold)" : "—"))}</div>
        ${h.organism ? `<div class="o">${esc(h.organism)}</div>` : ""}</td>
      <td class="num">${pct(a.seq_identity)}</td>
      <td class="num">${a.rmsd == null ? "—" : a.rmsd.toFixed(2) + " Å"}</td>
      ${tm ? `<td class="num">${h.tm_mean.toFixed(3)}</td>`
           : `<td class="num">${fmtE(h.evalue)}</td><td class="num">${h.bits == null ? h.sw_score.toFixed(1) : h.bits.toFixed(1)}</td>`}
      <td class="cov">${covBar(h.aln, job.query.n_tokens)}</td></tr>
      ${h.rank === openRank ? `<tr><td class="detail" colspan="${tm ? 7 : 8}">${detailHtml(h)}</td></tr>` : ""}`;
  }).join("");
  $("hits").innerHTML = head + (rows || `<tr><td colspan="8" class="hint" style="padding:18px">No hits passed the prefilter.</td></tr>`);
  $("hits").querySelectorAll("tr.hit").forEach((tr) => tr.addEventListener("click", () => {
    const r = +tr.dataset.rank;
    openRank = openRank === r ? null : r;
    setHash();
    renderTable();
  }));
  $("foot").textContent = tm ? "Ranked by the mean of the query- and target-normalised US-align TM-scores."
    : `Ranked by 3Dr Smith–Waterman score.${cal ? "" : " * E-values for this database are not yet empirically calibrated; use them to rank hits, not as significance."}`;
  if (openRank != null) openDetail(res.hits[openRank - 1]);
}

function detailHtml(h) {
  return `<div class="detail-grid">
    <div class="viewer-wrap"><div class="viewer" id="viewer"></div>
      <div class="viewer-legend"><span><span class="sw" style="background:#d9703a"></span>Query</span>
        <span><span class="sw" style="background:#3f6fb5"></span>${esc(h.pdb ? `${h.pdb} ${h.chain}` : h.target)}</span>
        <span class="hint">pale = unaligned</span></div></div>
    <div class="detail-body" id="detail-body"><span class="spinner"></span>Aligning…</div></div>`;
}

async function openDetail(h) {
  let a = h.aln;
  try {
    if (!a) {
      a = await (await api(`/api/result/${job.id}/${activeDb}/${h.rank}/alignment`)).json();
      h.aln = a;
    }
    $("detail-body").innerHTML = alignmentHtml(h, a);
  } catch (err) {
    $("detail-body").textContent = `Alignment unavailable: ${err.message}`;
  }
  drawStructure(h);
}

function alignmentHtml(h, a) {
  const stats = [
    ["Aligned", `${a.n_aligned} nt`],
    ["Seq. identity", pct(a.seq_identity)],
    ["RMSD (C4′)", a.rmsd == null ? "—" : `${a.rmsd.toFixed(2)} Å`],
    ["TM (aligned)", a.tm_aln == null ? "—" : a.tm_aln.toFixed(3)],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const W = 60, lines = [];
  if (a.q_aln) {
    let qi = a.q_start, ti = a.t_start;
    const pad = String(Math.max(a.q_end, a.t_end)).length;
    for (let i = 0; i < a.q_aln.length; i += W) {
      const qs = a.q_aln.slice(i, i + W), ms = a.tok_mid.slice(i, i + W), ts = a.t_aln.slice(i, i + W);
      const qn = qs.replace(/-/g, "").length, tn = ts.replace(/-/g, "").length;
      lines.push(`<span class="m">Query  </span><span class="pos">${String(qi).padStart(pad)} </span><span class="q">${esc(qs)}</span><span class="pos"> ${qi + qn - 1}</span>`);
      lines.push(`<span class="m">3Dr    ${" ".repeat(pad)} ${esc(ms)}</span>`);
      lines.push(`<span class="m">Target </span><span class="pos">${String(ti).padStart(pad)} </span><span class="t">${esc(ts)}</span><span class="pos"> ${ti + tn - 1}</span>`);
      lines.push("");
      qi += qn; ti += tn;
    }
  }
  return `<div class="stats">${stats}</div>
    ${lines.length ? `<pre class="aln">${lines.join("\n")}</pre>
    <div class="aln-note">3Dr line: <b>|</b> same structural token, <b>+</b> positive Structure-BLOSUM substitution, <b>.</b> other aligned pair. Positions count nucleotides with a complete backbone.${a.t_seq_known === false ? " This database carries no sequences, so target residues are shown as ·." : ""}</div>`
      : '<div class="hint">No sequence available for this database.</div>'}
    <div class="aln-note">Query ${a.q_start}–${a.q_end} of ${a.q_len} · target ${a.t_start}–${a.t_end} of ${a.t_len} · 3Dr identity ${pct(a.token_identity)}${h.title ? ` · ${esc(h.title)}` : ""}${h.method ? ` · ${esc(h.method)}` : ""}${h.resolution ? ` ${h.resolution} Å` : ""}</div>`;
}

async function drawStructure(h) {
  const el = $("viewer");
  if (!window.$3Dmol) { el.innerHTML = '<div class="hint" style="padding:16px">3D viewer unavailable (3Dmol.js did not load).</div>'; return; }
  let pdb;
  try {
    pdb = await (await api(`/api/result/${job.id}/${activeDb}/${h.rank}/structure`)).text();
  } catch (err) {
    el.innerHTML = `<div class="hint" style="padding:16px">${esc(err.message)}</div>`;
    return;
  }
  el.innerHTML = "";
  viewer = $3Dmol.createViewer(el, { backgroundColor: "white", antialias: true });
  viewer.addModel(pdb, "pdb");
  const colour = (chain) => (atom) => chain === "A"
    ? (atom.b > 0.5 ? "#d9703a" : "#f2cdb6")
    : (atom.b > 0.5 ? "#3f6fb5" : "#c9d6ea");
  for (const chain of ["A", "B"]) {
    viewer.setStyle({ chain }, { stick: { radius: 0.28, colorfunc: colour(chain) },
                                 sphere: { radius: 0.35, colorfunc: colour(chain) } });
  }
  viewer.zoomTo({ chain: "A" });
  viewer.render();
}

// ---------------------------------------------------------------- boot
function route() {
  const m = location.pathname.match(/^\/result\/([^/]+)/);
  if (m) loadResult(m[1]);
  else { showView("search"); job = null; }
}

$("copy-link").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(location.href); $("copy-link").textContent = "Copied"; }
  catch (_) { $("copy-link").textContent = location.href; }
  setTimeout(() => { $("copy-link").textContent = "Copy link"; }, 1600);
});
window.addEventListener("popstate", route);
wireForm();
loadHealth();
loadExamples();
route();
