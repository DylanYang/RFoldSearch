// RFoldSearch web page: form -> POST /api/search -> /result/<id>.
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
let markNt = null;

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
  updateSummaries();
  $("go-hint").textContent = !health ? "" : !health.ready ? "Server is still loading databases…"
    : !hasQuery ? "" : !hasDb ? "Select at least one database." : "";
}

// ---- display names: short, with a version or the last update date
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function updated(d) {
  const rel = (d.source || "").match(/(\d{1,2})-(\d{1,2})-(\d{4})/);        // e.g. release 9-14-2026
  if (rel) return `${MONTHS[+rel[1] - 1]} ${rel[3]}`;
  const b = (d.built || "").match(/^(\d{4})-(\d{2})/);
  return b ? `${MONTHS[+b[2] - 1]} ${b[1]}` : "";
}
function shortName(d) {
  const known = { "pdb-rna": "PDB-RNA", "rnasolo": "RNAsolo 2.0", "nufold": "NuFold" };
  return known[d.key] || String(d.label).split(" (")[0].replace(/ predicted structures$/i, "");
}
function dbLine(d) {
  const n = d.n_chains.toLocaleString();
  const predicted = /predicted|nufold/i.test(`${d.key} ${d.label} ${d.source || ""}`);
  const bgsu = `${d.source || ""} ${d.label || ""}`.match(/BGSU\s*([\d.]+)/);
  if (predicted) return `${n} predicted RNA structures (Updated ${updated(d)})`;
  if (d.key === "rnasolo" || bgsu) return `${n} non-redundant experimental structures${bgsu ? ` (BGSU ${bgsu[1]})` : ""}`;
  return `${n} experimental RNA chains (Updated ${updated(d)})`;
}

// ---- tidy text coming from the databases
const ACRONYMS = { rna: "RNA", dna: "DNA", trna: "tRNA", mrna: "mRNA", rrna: "rRNA", snrna: "snRNA",
                   snorna: "snoRNA", mirna: "miRNA", ncrna: "ncRNA", lncrna: "lncRNA" };
function niceDesc(t) {
  t = String(t ?? "").trim();
  if (!t) return t;
  if (/[a-z]/.test(t) || !/\s/.test(t)) return t;                            // already mixed case, or one word
  return t.toLowerCase()
    .replace(/(^|[\s(\-/])([a-z])/g, (_, a, b) => a + b.toUpperCase())
    .replace(/\b(Rna|Dna|Trna|Mrna|Rrna|Snrna|Snorna|Mirna|Ncrna|Lncrna)\b/g, (m) => ACRONYMS[m.toLowerCase()]);
}
function speciesOf(h) {                                                      // {name, taxon}
  const raw = String(h.organism ?? "");
  const taxon = (raw.match(/taxon (\d+)/) || [])[1] || null;
  let name = raw.replace(/\s*·\s*taxon \d+/, "").trim();
  if (name && name === name.toUpperCase()) name = name.charAt(0) + name.slice(1).toLowerCase();
  return { name, taxon };
}
function rnaName(h) {                                                        // description without a repeated species
  let d = niceDesc(h.desc || h.title || "");
  if (URS.test(h.target)) {                                                  // predicted: drop "(house mouse)" copies of the organism column
    const parts = d.split(";").map((x) => x.replace(/^\s*\([^)]*\)\s*/, "").replace(/\s*\([^)]*\)\s*$/, "").trim()).filter(Boolean);
    d = [...new Set(parts)].join("; ");
  }
  const sp = speciesOf(h).name.replace(/\s*\(.*$/, "");
  if (sp && d.toLowerCase().startsWith(sp.toLowerCase() + " ")) {
    d = d.slice(sp.length + 1).trim();
    if (/^[a-z]{3,}\b/.test(d)) d = d.charAt(0).toUpperCase() + d.slice(1);
  }
  return d || (URS.test(h.target) ? "Predicted structure" : "—");
}

// source-database logos (from the RNAsolo, RCSB PDB and NuFold sites)
const LOGOS = { rnasolo: "rnasolo.png", "pdb-rna": "pdb.png", nufold: "nufold.png" };
const logo = (key, cls) => (LOGOS[key] ? `<img class="${cls}" src="/static/logos/${LOGOS[key]}" alt="">` : "");

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
        <div class="name">${esc(shortName(d))}${logo(d.key, "dblogo")}</div>
        <div class="meta">${esc(dbLine(d))}</div>
        ${d.notice ? `<div class="notice">${esc(d.notice)}</div>` : ""}
      </div>
    </label>`).join("");
  box.querySelectorAll("input").forEach((inp) => inp.addEventListener("change", () => {
    inp.closest(".db").classList.toggle("on", inp.checked);
    updateGo();
  }));
  updateSummaries();
}

// the one-line summaries of the folded Databases / Mode / Settings sections
function updateSummaries() {
  if (!health || !health.ready) return;
  const names = health.databases.filter((d) => selectedDbs().includes(d.key)).map(shortName);
  $("sum-db").textContent = names.length ? names.join(", ") : "none selected";
  const mode = document.querySelector("#mode input:checked");
  $("sum-mode").textContent = mode && mode.value === "tm" ? "RFoldSearch-TM" : "RFoldSearch";
  $("sum-set").textContent = `${$("top_k").value} hits shown`;
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
    setTimeout(updateSummaries, 0);
  }));
  ["top_k", "top_n", "min_diag"].forEach((id) => $(id).addEventListener("input", updateSummaries));
  document.querySelectorAll(".num-step button").forEach((b) => b.addEventListener("click", () => {
    const inp = $(b.dataset.for);
    const max = +inp.max || Infinity;
    inp.value = Math.min(max, (+inp.value || 0) + +b.dataset.step);
    updateSummaries();
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
  $("nav-search").classList.toggle("on", which !== "tutorial");
  $("nav-tutorial").classList.toggle("on", which === "tutorial");
  $("view-search").hidden = which !== "search";
  $("view-result").hidden = which !== "result";
  $("view-tutorial").hidden = which !== "tutorial";
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
  const mode = p.rerank === "tm" ? "RFoldSearch-TM" : "RFoldSearch";
  $("res-sub").textContent = `${q.n_tokens} nucleotides encoded · ` +
    `${mode} · searched ${job.created.replace("T", " ").slice(0, 16)}`;
  $("dl-tsv").href = `/api/result/${job.id}/download?format=tsv`;
  $("dl-json").href = `/api/result/${job.id}/download?format=json`;
  const keys = Object.keys(job.results);
  if (!keys.includes(activeDb)) activeDb = keys[0];
  $("tabs").innerHTML = keys.map((k) => {
    const r = job.results[k];
    return `<button class="tab ${k === activeDb ? "on" : ""}" data-k="${esc(k)}">${logo(k, "tablogo")}${esc(shortName(r.database))}<span class="n">${r.hits.length}</span></button>`;
  }).join("");
  $("tabs").querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
    activeDb = b.dataset.k; openRank = null; sortKey = null; setHash(); renderResult();
  }));
  $("res").hidden = false;
  renderBest();
  renderTable();
}

function renderBest() {
  const res = job.results[activeDb];
  const h = res.hits[0];
  if (!h) { $("best").innerHTML = ""; return; }
  const a = h.aln || {};
  const tm = res.rerank === "tm";
  const kv = [
    ["Aligned", a.n_aligned == null ? "—" : `${a.n_aligned} nt`],
    noSequence(res) ? ["3Dr identity", pct(a.token_identity)] : ["Seq identity", pct(a.seq_identity)],
    ["RMSD", a.rmsd == null ? "—" : `${a.rmsd.toFixed(2)} Å`],
    tm ? ["TM-score", h.tm_mean.toFixed(3)] : ["TM-score", h.tm_a != null ? h.tm_a.toFixed(3) : "—"],
    ...(tm || h.evalue == null ? [] : [["E-value", fmtE(h.evalue)]]),
  ].map(([k, v]) => `<div>${k}<b>${v}</b></div>`).join("");
  const sp = speciesOf(h).name;
  $("best").innerHTML = `<div class="best">
    <div class="who"><div class="tag">Best hit</div>
      <div class="main">${esc(shortTarget(h))}</div>
      <div class="sub">${esc(rnaName(h))}${sp ? ` · <i>${esc(sp)}</i>` : ""}</div></div>
    <div class="kv">${kv}</div>
    <button class="btn small" id="best-open">Show alignment</button></div>`;
  $("best-open").addEventListener("click", () => {
    openRank = h.rank; setHash(); renderTable();
    $("hits").querySelector("tr.hit.open")?.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

// query / target coverage: the stretch of each chain spanned by the alignment
const covQ = (a) => (a && a.n_aligned ? (a.q_end - a.q_start + 1) / a.q_len : null);
const covT = (a) => (a && a.n_aligned ? (a.t_end - a.t_start + 1) / a.t_len : null);
function covBar(frac) {
  if (frac == null) return "—";
  const w = Math.max(2, Math.round(100 * frac));
  return `<span class="covbar"><span class="cbar"><i style="width:${Math.round(w * 0.64)}px"></i></span><span class="pc">${pct(frac)}</span></span>`;
}

// NuFold / RNAcentral models are named URS<id>_<taxid>_rank_<n>
const URS = /^(URS[0-9A-F]{10})_(\d+)(?:_rank_(\d+))?/;

const rfamOf = (h) => (h.rfam && h.rfam !== "N/A" ? h.rfam : null);

function shortTarget(h) {
  const u = !h.pdb && h.target.match(URS);
  if (u) return u[1];
  return h.pdb ? `${h.pdb} ${h.chain}` : h.target;
}

function noSequence(res) {
  const h = res.hits.find((x) => x.aln);
  return !!h && h.aln.t_seq_known === false;
}

// NuFold models are named URS<id>_<taxid>_rank_<n>; the NuFold database page of an
// entry is /rna/URS<id>_<taxid>, RNAcentral's is /rna/URS<id>/<taxid>
function nufoldLink(u, text) {
  return `<a href="https://nufold.kiharalab.org/rna/${u[1]}_${u[2]}" target="_blank" rel="noopener"
    onclick="event.stopPropagation()">${text}</a>`;
}
function rnacentralLink(u, text) {
  return `<a href="https://rnacentral.org/rna/${u[1]}/${u[2]}" target="_blank" rel="noopener"
    onclick="event.stopPropagation()">${text}</a>`;
}

function targetCell(h) {
  const u = !h.pdb && h.target.match(URS);
  if (u) {
    const sp = speciesOf(h);
    return `<span class="tgt" title="${sp.taxon ? `taxon ${sp.taxon}` : ""}">${nufoldLink(u, u[1])}</span>`;
  }
  if (h.pdb) {
    const chain = h.auth_asym && h.auth_asym !== h.chain ? `${h.chain} (auth ${h.auth_asym})` : h.chain;
    return `<a class="tgt" href="https://www.rcsb.org/structure/${esc(h.pdb)}" target="_blank" rel="noopener"
      onclick="event.stopPropagation()">${esc(h.pdb)}</a><span class="chain">Chain ${esc(chain)}</span>`;
  }
  return `<span class="tgt">${esc(h.target)}</span>`;
}

// columns of the hit table; `get` gives the value used for sorting
let sortKey = null, sortDir = -1;
function tableColumns(res, noSeq) {
  const tm = res.rerank === "tm", sw = res.rerank === "sw";
  const num = (key, label, get, text, title) => ({ key, label, num: true, get, text, title });
  const cols = [];
  if (noSeq) cols.push(num("plddt", "pLDDT", (h) => h.plddt, (h) => h.plddt == null ? "—" : h.plddt.toFixed(1),
                           "Mean pLDDT of the predicted model"));
  cols.push(noSeq
    ? num("tid", "3Dr identity", (h) => h.aln?.token_identity, (h) => pct(h.aln?.token_identity),
          "Fraction of aligned pairs with the same 3Dr structural token")
    : num("sid", "Seq identity", (h) => h.aln?.seq_identity, (h) => pct(h.aln?.seq_identity),
          "Nucleotide identity over the aligned positions"));
  cols.push(num("rmsd", "RMSD (Å)", (h) => h.aln?.rmsd, (h) => h.aln?.rmsd == null ? "—" : h.aln.rmsd.toFixed(2),
                "Root-mean-square deviation of the superposed aligned nucleotides"));
  if (tm) {
    cols.push(num("tm", "TM-score", (h) => h.tm_mean, (h) => h.tm_mean.toFixed(3),
                  "Mean of the query- and target-normalised US-align TM-scores"));
  } else if (sw) {
    cols.push(num("evalue", "E-value", (h) => h.evalue, (h) => fmtE(h.evalue)),
              num("bits", "Bits", (h) => h.bits ?? h.sw_score, (h) => (h.bits ?? h.sw_score).toFixed(1)));
  } else {
    cols.push(num("rstr", "Structure score", (h) => h.structure_score, (h) => h.structure_score.toFixed(2),
                  "S_SW × TM-score"),
              num("tma", "TM-score", (h) => h.tm_a, (h) => h.tm_a.toFixed(3),
                  "TM-score of the token alignment after superposition"),
              num("evalue", "E-value", (h) => h.evalue, (h) => fmtE(h.evalue)));
  }
  cols.push({ key: "qc", label: "Query cov", num: true, get: (h) => covQ(h.aln), html: (h) => covBar(covQ(h.aln)),
              title: "Share of the query spanned by the alignment" },
            num("tc", "Target cov", (h) => covT(h.aln), (h) => pct(covT(h.aln)),
                "Share of the target spanned by the alignment"));
  return cols;
}

function renderTable() {
  const res = job.results[activeDb];
  $("tab-meta").innerHTML = res.database.notice ? `<div class="notice">${esc(res.database.notice)}</div>` : "";
  const noSeq = noSequence(res);
  const cols = tableColumns(res, noSeq);
  const ncol = 3 + cols.length;
  let hits = res.hits;
  if (sortKey) {
    const col = cols.find((c) => c.key === sortKey);
    if (col) hits = [...hits].sort((x, y) => {
      const a = col.get(x), b = col.get(y);
      return (a == null) - (b == null) || (a == null ? 0 : sortDir * (a - b));
    });
  }
  const arrow = (c) => (c.key === sortKey ? `<span class="arrow">${sortDir < 0 ? "▾" : "▴"}</span>` : "");
  const head = `<tr><th></th><th>Target</th><th>RNA / organism</th>${cols.map((c) =>
    `<th class="num sortable" data-sort="${c.key}"${c.title ? ` title="${esc(c.title)} (click to sort)"` : ' title="Click to sort"'}>${esc(c.label)}${arrow(c)}</th>`).join("")}</tr>`;
  const rows = hits.map((h) => {
    const sp = speciesOf(h);
    const tags = noSeq && rfamOf(h) ? `<div class="tags"><span class="pill">Rfam ${esc(rfamOf(h))}</span></div>` : "";
    return `<tr class="hit ${h.rank === openRank ? "open" : ""}" data-rank="${h.rank}">
      <td class="rank">${h.rank}</td>
      <td>${targetCell(h)}</td>
      <td class="desc"><div class="t" title="${esc(h.title || h.desc || "")}">${esc(rnaName(h))}</div>
        ${sp.name ? `<div class="o" title="${sp.taxon ? `taxon ${sp.taxon}` : ""}">${esc(sp.name)}</div>` : ""}${tags}</td>
      ${cols.map((c) => `<td class="num">${c.html ? c.html(h) : c.text(h)}</td>`).join("")}</tr>
      ${h.rank === openRank ? `<tr><td class="detail" colspan="${ncol}">${detailHtml(h)}</td></tr>` : ""}`;
  }).join("");
  $("hits").innerHTML = head + (rows || `<tr><td colspan="${ncol}" class="hint" style="padding:18px">No hits passed the prefilter.</td></tr>`);
  $("hits").querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.sort;
    if (sortKey !== k) { sortKey = k; sortDir = -1; } else if (sortDir < 0) sortDir = 1; else sortKey = null;
    renderTable();
  }));
  $("hits").querySelectorAll("tr.hit").forEach((tr) => tr.addEventListener("click", () => {
    const r = +tr.dataset.rank;
    openRank = openRank === r ? null : r;
    setHash();
    renderTable();
  }));
  $("foot").textContent = "";
  if (openRank != null) openDetail(res.hits[openRank - 1]);
}

function detailHtml(h) {
  return `<div class="detail-grid">
    <div class="viewer-wrap"><div class="viewer" id="viewer"></div>
      <div class="viewer-legend"><span><span class="sw" style="background:#d9703a"></span>Query</span>
        <span title="${esc(h.target)}"><span class="sw" style="background:#3f6fb5"></span>${esc(shortTarget(h))}</span>
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
  const noSeq = a.t_seq_known === false;
  const cq = a.n_aligned ? (a.q_end - a.q_start + 1) / a.q_len : null;
  const ct = a.n_aligned ? (a.t_end - a.t_start + 1) / a.t_len : null;
  const stat = (k, v, sub) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
  const stats = [
    stat("Aligned-region TM-score", a.tm_aln == null ? "—" : a.tm_aln.toFixed(3)),
    stat("RMSD", a.rmsd == null ? "—" : `${a.rmsd.toFixed(2)} Å`, `${a.n_aligned} aligned nt`),
    stat("Query / target coverage", `${pct(cq)} / ${pct(ct)}`, `Query ${a.q_start}–${a.q_end}/${a.q_len} · Target ${a.t_start}–${a.t_end}/${a.t_len}`),
    noSeq ? stat("3Dr identity", pct(a.token_identity))
          : stat("Seq identity", pct(a.seq_identity), `3Dr identity ${pct(a.token_identity)}`),
  ].join("");
  const W = 60, lines = [];
  const cell = (ch, chain, pos) => (ch === "-" ? ch : `<span class="nt" data-c="${chain}" data-p="${pos}">${esc(ch)}</span>`);
  if (a.q_aln) {
    let qi = a.q_start, ti = a.t_start;
    const pad = String(Math.max(a.q_end, a.t_end)).length;
    for (let i = 0; i < a.q_aln.length; i += W) {
      const qs = a.q_aln.slice(i, i + W), ms = a.tok_mid.slice(i, i + W), ts = a.t_aln.slice(i, i + W);
      const qn = qs.replace(/-/g, "").length, tn = ts.replace(/-/g, "").length;
      let qp = qi, tp = ti;
      const qh = [...qs].map((c) => (c === "-" ? c : cell(c, "A", qp++))).join("");
      const th = [...ts].map((c) => (c === "-" ? c : cell(c, "B", tp++))).join("");
      lines.push(`<span class="m">Query  </span><span class="pos">${String(qi).padStart(pad)} </span><span class="q">${qh}</span><span class="pos"> ${qi + qn - 1}</span>`);
      lines.push(`<span class="m">3Dr    ${" ".repeat(pad)} ${esc(ms)}</span>`);
      lines.push(`<span class="m">Target </span><span class="pos">${String(ti).padStart(pad)} </span><span class="t">${th}</span><span class="pos"> ${ti + tn - 1}</span>`);
      lines.push("");
      qi += qn; ti += tn;
    }
  }
  const structure = h.title ? `<div class="r"><span>Structure</span><span>${esc(h.title)}</span></div>` : "";
  const urs = !h.pdb && h.target.match(URS);
  const links = urs ? `<div class="r"><span>Links</span><span>${nufoldLink(urs, "NuFold entry ↗")} · ${rnacentralLink(urs, "RNAcentral ↗")}</span></div>` : "";
  const method = h.method ? `<div class="r"><span>Method</span><span>${esc(h.method)}${h.resolution ? ` · ${Number(h.resolution).toFixed(2)} Å` : ""}</span></div>` : "";
  const top = structure + method + links;
  return `${top ? `<div class="summary">${top}</div>` : ""}
    <div class="stats">${stats}</div>
    ${lines.length ? `<pre class="aln">${lines.join("\n")}</pre><div class="aln-note">Click a nucleotide to locate it, and its aligned partner, in the 3D view.</div>`
      : '<div class="hint">No sequence available for this database.</div>'}`;
}

// token index -> the aligned partner's token index, from the gapped strings
function pairMap(a) {
  if (!a || !a.q_aln) return null;
  const q2t = {}, t2q = {};
  let qi = a.q_start, ti = a.t_start;
  for (let i = 0; i < a.q_aln.length; i++) {
    const qg = a.q_aln[i] === "-", tg = a.t_aln[i] === "-";
    if (!qg && !tg) { q2t[qi] = ti; t2q[ti] = qi; }
    if (!qg) qi++;
    if (!tg) ti++;
  }
  return { q2t, t2q };
}

// Turn the coordinates so the molecule's long axis lies along the long side of the
// panel and its thinnest axis points at the viewer, the view a figure would use.
function orient(atoms, wide) {
  if (atoms.length < 3) return;
  const c = [0, 0, 0];
  for (const a of atoms) { c[0] += a.x; c[1] += a.y; c[2] += a.z; }
  c.forEach((_, i) => { c[i] /= atoms.length; });
  const m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const a of atoms) {
    const d = [a.x - c[0], a.y - c[1], a.z - c[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) m[i][j] += d[i] * d[j];
  }
  // Jacobi rotations: columns of v become the principal axes
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(m[p][q]) < 1e-9) continue;
      const t = (m[q][q] - m[p][p]) / (2 * m[p][q]);
      const tn = Math.sign(t || 1) / (Math.abs(t) + Math.sqrt(t * t + 1));
      const cs = 1 / Math.sqrt(tn * tn + 1), sn = tn * cs;
      for (let k = 0; k < 3; k++) {
        const a = m[k][p], b = m[k][q]; m[k][p] = cs * a - sn * b; m[k][q] = sn * a + cs * b;
      }
      for (let k = 0; k < 3; k++) {
        const a = m[p][k], b = m[q][k]; m[p][k] = cs * a - sn * b; m[q][k] = sn * a + cs * b;
      }
      for (let k = 0; k < 3; k++) {
        const a = v[k][p], b = v[k][q]; v[k][p] = cs * a - sn * b; v[k][q] = sn * a + cs * b;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => m[j][j] - m[i][i]);         // largest spread first
  const ax = order.map((i) => [v[0][i], v[1][i], v[2][i]]);
  const z = [ax[0][1] * ax[1][2] - ax[0][2] * ax[1][1],                // right-handed, so no mirror image
             ax[0][2] * ax[1][0] - ax[0][0] * ax[1][2],
             ax[0][0] * ax[1][1] - ax[0][1] * ax[1][0]];
  const [ex, ey] = wide ? [ax[0], ax[1]] : [ax[1], ax[0]];
  const ez = wide ? z : z.map((x) => -x);
  for (const a of atoms) {
    const d = [a.x - c[0], a.y - c[1], a.z - c[2]];
    const dot = (e) => e[0] * d[0] + e[1] * d[1] + e[2] * d[2];
    [a.x, a.y, a.z] = [dot(ex), dot(ey), dot(ez)];
  }
}

async function drawStructure(h) {
  const el = $("viewer");
  if (!window.$3Dmol) { el.innerHTML = '<div class="hint" style="padding:16px">3D viewer unavailable (3Dmol.js did not load).</div>'; return; }
  let pdb;
  el.innerHTML = '<div class="hint" style="padding:16px">Loading the 3D structures… (a large chain fetched from the RCSB for the first time can take half a minute)</div>';
  try {
    pdb = await (await api(`/api/result/${job.id}/${activeDb}/${h.rank}/structure`)).text();
  } catch (err) {
    el.innerHTML = `<div class="hint" style="padding:16px">${esc(err.message)}</div>`;
    return;
  }
  el.innerHTML = "";
  viewer = $3Dmol.createViewer(el, { backgroundColor: "white", antialias: true });
  viewer.addModel(pdb, "pdb");
  orient(viewer.selectedAtoms({}), el.clientWidth >= el.clientHeight);
  // aligned nucleotides carry B = 1; colours are the query's orange and the hit's blue, pale where unaligned
  const PAL = { A: { on: "#d9703a", off: "#f0cfb8", base: "#e8955f" }, B: { on: "#3f6fb5", off: "#c4d3ea", base: "#6a93cf" } };
  const col = (chain, atom, kind) =>
    atom.b > 0.5 ? (kind === "base" ? PAL[chain].base : PAL[chain].on) : PAL[chain].off;
  // Chains the server sent with all their atoms are drawn as a cartoon: a ribbon along the
  // backbone, the bases as filled rings. The others carry only the five stored atoms per
  // nucleotide: a smooth tube through the C4' atoms, the base a short stick ending in a bead.
  const full = new Set(((pdb.match(/^REMARK {3}1 FULLATOM (.*)$/m) || [])[1] || "").trim().split(/\s+/).filter(Boolean));
  const coarse = ["A", "B"].filter((c) => !full.has(c));
  const baseStyle = () => {
    for (const chain of full) {
      viewer.setStyle({ chain }, { cartoon: { colorfunc: (a) => col(chain, a, "bb"), thickness: 0.6, width: 1.6, ribbon: false, style: "oval" } });
    }
    for (const chain of coarse) {
      viewer.setStyle({ chain }, {});
      viewer.setStyle({ chain, atom: "C4'" }, { sphere: { radius: 0.62, colorfunc: (a) => col(chain, a, "bb") } });
      viewer.setStyle({ chain, atom: "N1" }, { sphere: { radius: 0.34, colorfunc: (a) => col(chain, a, "base") } });
    }
  };
  baseStyle();
  for (const chain of coarse) {
    const atoms = viewer.selectedAtoms({ chain, atom: "C4'" }).sort((x, y) => x.resi - y.resi);
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        viewer.addCurve({ points: run.map((x) => ({ x: x.x, y: x.y, z: x.z })), radius: 0.6, smooth: 6,
                          fill: true, color: run[0].b > 0.5 ? PAL[chain].on : PAL[chain].off });
      }
      run = [];
    };
    for (const at of atoms) {
      const last = run[run.length - 1];
      if (last && (at.resi !== last.resi + 1 || (at.b > 0.5) !== (last.b > 0.5))) {
        // carry the tube across a change of colour so the backbone has no gap
        const carry = last && at.resi === last.resi + 1 ? [last] : [];
        flush(); run = carry;
      }
      run.push(at);
    }
    flush();
  }
  for (const chain of coarse) {                                              // base: C4' -> C1' -> N1
    const byRes = {};
    for (const at of viewer.selectedAtoms({ chain })) (byRes[at.resi] ||= {})[at.atom] = at;
    for (const r of Object.values(byRes)) {
      const c = r["C4'"], o = r["C1'"], n = r["N1"];
      if (!c || !o || !n) continue;
      const colr = c.b > 0.5 ? PAL[chain].base : PAL[chain].off;
      for (const [x, y] of [[c, o], [o, n]]) {
        viewer.addCylinder({ start: { x: x.x, y: x.y, z: x.z }, end: { x: y.x, y: y.y, z: y.z },
                             radius: 0.32, color: colr, fromCap: 1, toCap: 1 });
      }
    }
  }
  viewer.setViewStyle({ style: "outline", color: "#3a3330", width: 0.025 });
  viewer.zoomTo();
  const map = pairMap(h.aln);
  const picked = document.createElement("div");
  picked.className = "picked";
  picked.textContent = "Click a nucleotide in the 3D view, or in the alignment, to see its aligned partner.";
  el.closest(".viewer-wrap").appendChild(picked);
  // mark one nucleotide (and its aligned partner) in the 3D view; `centre` moves the view to it
  const mark = (chain, resi, centre) => {
    const atom = viewer.selectedAtoms({ chain, resi, atom: "C4'" })[0] || viewer.selectedAtoms({ chain, resi })[0];
    if (!atom) { picked.textContent = `${chain === "A" ? "Query" : shortTarget(h)} nucleotide ${resi} is not in the 3D model`; return; }
    const mine = chain === "A" ? "Query" : shortTarget(h);
    const other = map ? (chain === "A" ? map.q2t[resi] : map.t2q[resi]) : null;
    const otherChain = chain === "A" ? "B" : "A";
    const partner = map ? (other ? ` is aligned to ${chain === "A" ? shortTarget(h) : "Query"} nucleotide ${other}` : " is not aligned") : "";
    picked.textContent = `${mine} nucleotide ${resi}${atom.resn && atom.resn !== "N" ? ` (${atom.resn})` : ""}${partner}`;
    viewer.removeAllLabels();
    baseStyle();
    viewer.addLabel(`${chain === "A" ? "Q" : "T"}${resi}`, { position: atom, fontSize: 12,
      backgroundColor: "#26211d", backgroundOpacity: 0.85, fontColor: "white" });
    const markStyle = (c, r, radius) => full.has(c)
      ? viewer.addStyle({ chain: c, resi: r }, { stick: { radius: 0.32, color: "#e5322d" } })
      : viewer.setStyle({ chain: c, resi: r, atom: ["C4'", "N1"] }, { sphere: { radius, color: "#e5322d" } });
    markStyle(chain, resi, 0.62);
    if (other) {
      const pa = viewer.selectedAtoms({ chain: otherChain, resi: other, atom: "C4'" })[0];
      if (pa) {
        viewer.addLabel(`${otherChain === "A" ? "Q" : "T"}${other}`, { position: pa, fontSize: 12,
          backgroundColor: "#7a2a27", backgroundOpacity: 0.85, fontColor: "white" });
        markStyle(otherChain, other, 0.55);
      }
    }
    if (centre) viewer.center({ chain, resi }, 400);
    viewer.render();
  };
  markNt = mark;
  viewer.setClickable({}, true, (atom) => mark(atom.chain, atom.resi, false));
  viewer.render();
  document.querySelectorAll("#detail-body .nt").forEach((el2) => el2.addEventListener("click", () => {
    document.querySelectorAll("#detail-body .nt.sel").forEach((x) => x.classList.remove("sel"));
    const c = el2.dataset.c, p = +el2.dataset.p;
    el2.classList.add("sel");
    const o = map ? (c === "A" ? map.q2t[p] : map.t2q[p]) : null;
    if (o) document.querySelector(`#detail-body .nt[data-c="${c === "A" ? "B" : "A"}"][data-p="${o}"]`)?.classList.add("sel");
    mark(c, p, true);
  }));
}

// ---------------------------------------------------------------- boot
function route() {
  const m = location.pathname.match(/^\/result\/([^/]+)/);
  if (location.pathname === "/tutorial") { showView("tutorial"); job = null; }
  else if (m) loadResult(m[1]);
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
