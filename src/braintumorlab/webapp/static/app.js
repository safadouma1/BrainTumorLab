// BrainTumorLab front-end (vanilla JS, no build step). All API URLs are relative so the app
// also works behind the Google Colab port proxy.
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

/* ------------------------------------------------------------------ icons */
const P = (d) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
const ICONS = {
  home: P('<path d="M3 11 12 4l9 7"/><path d="M5 10v10h14V10"/>'),
  scan: P('<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="4"/>'),
  brain: P('<path d="M12 5a3 3 0 0 0-5.6-1.5A3 3 0 0 0 4 8a3 3 0 0 0 0 5 3 3 0 0 0 3 4 3 3 0 0 0 5 2z"/><path d="M12 5a3 3 0 0 1 5.6-1.5A3 3 0 0 1 20 8a3 3 0 0 1 0 5 3 3 0 0 1-3 4 3 3 0 0 1-5 2z"/>'),
  cube: P('<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/><path d="m12 12 8-4.5M12 12v9M12 12 4 7.5"/>'),
  chart: P('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'),
  sparkles: P('<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>'),
  doc: P('<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h6"/>'),
  upload: P('<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v4h16v-4"/>'),
  play: P('<path d="M7 4v16l13-8z"/>'),
  clock: P('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
  pie: P('<path d="M12 3v9h9A9 9 0 1 1 12 3z"/><path d="M15 3.5A9 9 0 0 1 20.5 9H15z"/>'),
  bars: P('<path d="M5 20V12M10 20V6M15 20v-9M20 20V4"/>'),
  cpu: P('<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'),
  users: P('<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18 14a6 6 0 0 1 3.5 6"/>'),
  target: P('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>'),
  layers: P('<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>'),
  refresh: P('<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>'),
  camera: P('<path d="M4 7h4l2-3h4l2 3h4v13H4z"/><circle cx="12" cy="13" r="4"/>'),
  download: P('<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>'),
  sliders: P('<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>'),
  image: P('<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>'),
  table: P('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/>'),
  trend: P('<path d="m3 17 6-6 4 4 8-8"/><path d="M14 7h7v7"/>'),
  dots: P('<circle cx="6" cy="16" r="1.5"/><circle cx="10" cy="9" r="1.5"/><circle cx="15" cy="13" r="1.5"/><circle cx="19" cy="6" r="1.5"/>'),
  print: P('<path d="M7 9V3h10v6"/><rect x="4" y="9" width="16" height="8" rx="2"/><path d="M7 14h10v7H7z"/>'),
  ruler: P('<path d="m3 17 14-14 4 4L7 21z"/><path d="m7 13 2 2M10 10l2 2M13 7l2 2"/>'),
  vol: P('<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/>'),
};
const NAV = [
  ["dashboard", "Dashboard", "home"], ["analysis", "MRI Analysis", "scan"], ["segmentation", "AI Segmentation", "brain"],
  ["3d", "3D Visualization", "cube"], ["radiomics", "Radiomics", "chart"], ["generation", "MRI Generation", "sparkles"],
  ["models", "Models & Reports", "doc"],
];

/* ------------------------------------------------------------------ helpers */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
let busy = 0;
function loading(on, txt = "Working…") {
  busy += on ? 1 : -1;
  $("#loading-txt").textContent = txt;
  $("#loading").classList.toggle("on", busy > 0);
}
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.style.display = "block";
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.style.display = "none"), 6000);
}
async function api(url, opts = {}, txt) {
  loading(true, txt);
  try {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    return j;
  } finally {
    loading(false);
  }
}
const post = (url, body, txt) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }, txt);
function gradeBadge(g) {
  const k = (g || "").toLowerCase();
  return `<span class="badge ${k === "hgg" ? "hgg" : k === "lgg" ? "lgg" : "unk"}">${esc(g || "n/a")}</span>`;
}
function icons(root = document) {
  $$("[data-icon]", root).forEach((el) => {
    if (el.dataset.iconDone) return;
    el.insertAdjacentHTML("afterbegin", ICONS[el.dataset.icon] || "");
    el.dataset.iconDone = 1;
  });
}
function empty(msg) {
  return `<div class="empty">${msg}</div>`;
}

/* ------------------------------------------------------------------ charts */
Chart.defaults.color = "#8a93b9";
Chart.defaults.borderColor = "rgba(110,134,255,.12)";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
const COL = { blue: "#4f7cff", violet: "#8b5cf6", teal: "#14d3c4", amber: "#fbbf24", rose: "#f43f5e", green: "#22d3a6", slate: "#8a93b9" };
const SERIES = [COL.blue, COL.teal, COL.violet, COL.amber, COL.rose, COL.green];
const charts = {};
function chart(id, cfg) {
  charts[id]?.destroy();
  const el = document.getElementById(id);
  if (!el) return;
  cfg.options = { responsive: true, maintainAspectRatio: false, ...(cfg.options || {}) };
  charts[id] = new Chart(el, cfg);
}

/* ------------------------------------------------------------------ state */
const S = {
  pid: null, info: null, status: null, patients: [], mod: "flair", cursor: [0, 0, 0], overlay: "none", alpha: 0.45, win: 255,
  segPlane: "axial", segMod: "flair", segIdx: 0, segRes: null, page: "dashboard",
};
const volCache = new Map();
async function getVol(pid, name) {
  const key = pid + "|" + name;
  if (volCache.has(key)) return volCache.get(key);
  loading(true, `Loading ${name}…`);
  try {
    const r = await fetch(`api/patient/${encodeURIComponent(pid)}/volume/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const shape = r.headers.get("X-Shape").split(",").map(Number);
    const v = { data: new Uint8Array(await r.arrayBuffer()), shape };
    volCache.set(key, v);
    if (volCache.size > 14) volCache.delete(volCache.keys().next().value);
    return v;
  } finally {
    loading(false);
  }
}

/* ------------------------------------------------------------------ slice rendering */
// Arrays are RAS-oriented [x][y][z] (flat index (x*Y + y)*Z + z). Images are shown rot90:
// axial = slice z (anterior up), coronal = slice y (superior up), sagittal = slice x (superior up).
const LABEL_RGB = { 1: [244, 63, 94], 2: [34, 211, 166], 4: [251, 191, 36] };
function planeDims(shape, plane) {
  const [X, Y, Z] = shape;
  return plane === "axial" ? [X, Y] : plane === "coronal" ? [X, Z] : [Y, Z];
}
function planeIndex(plane) {
  return plane === "axial" ? 2 : plane === "coronal" ? 1 : 0;
}
function flatIdx(shape, plane, idx, col, row) {
  const [X, Y, Z] = shape;
  let x, y, z;
  if (plane === "axial") { x = col; y = Y - 1 - row; z = idx; }
  else if (plane === "coronal") { x = col; y = idx; z = Z - 1 - row; }
  else { x = idx; y = col; z = Z - 1 - row; }
  return (x * Y + y) * Z + z;
}
function renderSlice(canvas, vol, plane, idx, o = {}) {
  const [w, h] = planeDims(vol.shape, plane);
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const d = img.data, win = o.win || 255, a = o.alpha ?? 0.45;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const i = flatIdx(vol.shape, plane, idx, c, r);
    let g = Math.min(255, (vol.data[i] * 255) / win) * (o.dim ?? 1);
    let R = g, G = g, B = g;
    if (o.diff) {
      const gt = o.diff.gt.data[i] > 0, pr = o.diff.pred.data[i] > 0;
      const col = gt && pr ? [20, 211, 196] : pr ? [244, 63, 94] : gt ? [79, 124, 255] : null;
      if (col) { R = (1 - 0.7) * R + 0.7 * col[0]; G = (1 - 0.7) * G + 0.7 * col[1]; B = (1 - 0.7) * B + 0.7 * col[2]; }
    } else if (o.lab) {
      const col = LABEL_RGB[o.lab.data[i]];
      if (col) { R = (1 - a) * R + a * col[0]; G = (1 - a) * G + a * col[1]; B = (1 - a) * B + a * col[2]; }
    }
    const p = (r * w + c) * 4;
    d[p] = R; d[p + 1] = G; d[p + 2] = B; d[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  if (o.cross) {
    ctx.strokeStyle = "rgba(20,211,196,.55)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(o.cross[0] + 0.5, 0); ctx.lineTo(o.cross[0] + 0.5, h);
    ctx.moveTo(0, o.cross[1] + 0.5); ctx.lineTo(w, o.cross[1] + 0.5); ctx.stroke();
  }
}
function crossFor(shape, plane, cur) {
  const [, Y, Z] = shape;
  const [x, y, z] = cur;
  return plane === "axial" ? [x, Y - 1 - y] : plane === "coronal" ? [x, Z - 1 - z] : [y, Z - 1 - z];
}
function canvasPos(canvas, ev) {
  const rect = canvas.getBoundingClientRect();
  const s = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const ox = (rect.width - canvas.width * s) / 2, oy = (rect.height - canvas.height * s) / 2;
  return [Math.floor((ev.clientX - rect.left - ox) / s), Math.floor((ev.clientY - rect.top - oy) / s)];
}
const ORIENT = { axial: ["A", "P", "R", "L"], coronal: ["S", "I", "R", "L"], sagittal: ["S", "I", "A", "P"] };

/* ------------------------------------------------------------------ router */
function buildNav() {
  $("#nav").innerHTML = NAV.map(([id, label, ic]) => `<a href="#/${id}" data-page="${id}">${ICONS[ic]}<span>${label}</span></a>`).join("");
}
async function route() {
  const [pagePart, query] = location.hash.replace("#/", "").split("?");
  const page = pagePart || "dashboard";
  const wanted = new URLSearchParams(query || "").get("pid");
  if (wanted && wanted !== S.pid) return selectPatient(wanted).catch((e) => toast(e.message));
  S.page = NAV.some((n) => n[0] === page) ? page : "dashboard";
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + S.page));
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === S.page));
  try {
    await PAGES[S.page]?.();
  } catch (e) {
    toast(e.message);
  }
}

/* ------------------------------------------------------------------ patient selection */
async function loadPatients() {
  const r = await api("api/patients?limit=5000");
  S.patients = r.patients;
  return r;
}
async function selectPatient(pid) {
  S.info = await api(`api/patient/${encodeURIComponent(pid)}`, {}, "Loading patient…");
  S.pid = pid;
  S.cursor = [...S.info.tumor_center];
  S.segRes = null;
  S.overlay = S.info.has_seg ? "gt" : "none";
  const ch = $("#cur-patient");
  ch.classList.add("on");
  ch.innerHTML = `<span class="dot ok"></span>${esc(S.info.name || pid)} · ${esc(S.info.split)} ${S.info.grade ? "· " + esc(S.info.grade) : ""}`;
  await route();
}
function needPatient(el) {
  if (S.pid) return false;
  el.innerHTML = empty(`<b>No patient selected</b>Use the search bar or the <a href="#/analysis">MRI Analysis</a> page.`);
  return true;
}
function setupSearch() {
  const inp = $("#search"), box = $("#search-results");
  inp.addEventListener("input", () => {
    const q = inp.value.trim().toLowerCase();
    if (!q) { box.style.display = "none"; return; }
    const hits = S.patients.filter((p) => p.patient_id.toLowerCase().includes(q) || (p.name || "").toLowerCase().includes(q)).slice(0, 12);
    box.innerHTML = hits.length ? hits.map((p) => `<div data-pid="${esc(p.patient_id)}"><span>${esc(p.name || p.patient_id)}</span><span>${gradeBadge(p.grade_name)} <span class="badge split">${esc(p.split)}</span></span></div>`).join("") : `<div>No match</div>`;
    box.style.display = "block";
  });
  box.addEventListener("click", (e) => {
    const pid = e.target.closest("[data-pid]")?.dataset.pid;
    if (!pid) return;
    box.style.display = "none"; inp.value = "";
    selectPatient(pid).catch((er) => toast(er.message));
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) box.style.display = "none"; });
}

/* ------------------------------------------------------------------ 1. dashboard */
async function pageDashboard() {
  const [d, st] = await Promise.all([api("api/dashboard"), S.status ? S.status : api("api/status")]);
  S.status = st;
  const g = d.grades || {};
  const v = d.volumes || {};
  const best = d.best_segmentation;
  const kpi = (ic, cls, lbl, val, sub) => `<div class="panel kpi"><div class="ic ${cls}">${ICONS[ic]}</div><div><div class="lbl">${lbl}</div><div class="val">${val}</div><div class="sub">${sub}</div></div></div>`;
  $("#kpis").innerHTML =
    kpi("users", "blue", "Patients (BraTS 2020)", d.n_patients, Object.entries(d.splits || {}).map(([k, n]) => `${k} ${n}`).join(" · ") + (d.n_uploads ? ` · ${d.n_uploads} uploaded` : "")) +
    kpi("brain", "violet", "Grade HGG / LGG", `${g.HGG || 0} / ${g.LGG || 0}`, g.unknown ? `${g.unknown} unknown` : "from name_mapping.csv") +
    kpi("vol", "teal", "Median whole-tumor volume", v.WT ? `${f(v.WT.median, 1)} mL` : "…", v.WT ? `TC ${f(v.TC.median, 1)} · ET ${f(v.ET.median, 1)} mL (n=${v.WT.n})` : `computing (${d.volumes_ready}/${d.n_patients})`) +
    kpi("target", "amber", "Best segmentation (test)", best ? f(best.mean_dice, 3) : "—", best ? `${esc(best.model)} · mean Dice WT/TC/ET · n=${best.n}` : "no model evaluated yet");
  const ev = d.recent || [];
  $("#recent-count").textContent = ev.length ? `${ev.length} latest` : "";
  $("#recent").innerHTML = ev.length
    ? `<table><tr><th>Patient</th><th>Analysis</th><th>Model</th><th>Result</th><th class="num">When</th></tr>${ev.map((e) => `<tr class="click" data-pid="${esc(e.patient_id)}"><td>${esc(e.patient_id)}</td><td>${esc(e.kind)}</td><td>${esc(e.model || "—")}</td><td class="mono">${esc(Object.entries(e.summary || {}).map(([k, x]) => `${k} ${x}`).join(" · ") || "—")}</td><td class="num note">${new Date(e.ts * 1000).toLocaleString()}</td></tr>`).join("")}</table>`
    : empty("<b>No analysis yet</b>Open a patient in MRI Analysis to start.");
  $$("#recent tr[data-pid]").forEach((tr) => tr.addEventListener("click", () => selectPatient(tr.dataset.pid).then(() => (location.hash = "#/analysis"))));
  chart("ch-grade", { type: "doughnut", data: { labels: Object.keys(g), datasets: [{ data: Object.values(g), backgroundColor: [COL.rose, COL.teal, COL.slate], borderWidth: 0 }] }, options: { cutout: "68%", plugins: { legend: { position: "bottom" } } } });
  $("#vol-status").textContent = v.WT ? `n = ${v.WT.n}` : `computing ${d.volumes_ready}/${d.n_patients}`;
  if (v.WT_hist) {
    const e = v.WT_hist.edges;
    chart("ch-vol", { type: "bar", data: { labels: v.WT_hist.counts.map((_, i) => `${Math.round(e[i])}–${Math.round(e[i + 1])}`), datasets: [{ data: v.WT_hist.counts, backgroundColor: COL.blue, borderRadius: 5 }] }, options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 50, font: { size: 10 } } }, y: { title: { display: true, text: "patients" }, ticks: { precision: 0 } } } } });
    const c = v.composition;
    chart("ch-comp", { type: "doughnut", data: { labels: ["Necrosis NCR", "Edema ED", "Enhancing ET"], datasets: [{ data: [c.NCR, c.ED, c.ET], backgroundColor: [COL.rose, COL.green, COL.amber], borderWidth: 0 }] }, options: { cutout: "62%", plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (x) => ` ${x.label}: ${f(x.raw, 1)} mL` } } } } });
  }
  const segm = st.segmentation_models || [];
  $("#model-status").innerHTML =
    `<table><tr><th>Model</th><th>Status</th><th class="num">Score</th></tr>` +
    (segm.length ? segm.map((m) => `<tr><td>${esc(m.name)}</td><td><span class="dot ok" style="display:inline-block"></span> trained${m.epochs_trained ? ` (${m.epochs_trained} ep)` : ""}</td><td class="num mono">${m.test ? "Dice WT " + f(m.test.WT.dice?.mean, 3) : m.best_val_dice ? "val " + f(m.best_val_dice, 3) : "—"}</td></tr>`).join("")
      : `<tr><td>3D U-Net</td><td><span class="dot warn" style="display:inline-block"></span> not trained</td><td class="num note">notebook 03</td></tr>`) +
    `<tr><td>3D CNN classifier</td><td>${st.classifier ? `<span class="dot ok" style="display:inline-block"></span> loaded${st.classifier.legacy ? " (legacy .pkl)" : ""}` : `<span class="dot warn" style="display:inline-block"></span> missing`}</td><td class="num note">grade LGG/HGG</td></tr>` +
    `<tr><td>Med-DDPM (T2)</td><td>${st.ddpm ? `<span class="dot ok" style="display:inline-block"></span> loaded` : `<span class="dot warn" style="display:inline-block"></span> unavailable`}</td><td class="num mono">${st.ddpm ? `epoch ${st.ddpm.epoch ?? "?"}` : ""}</td></tr></table>`;
}

/* ------------------------------------------------------------------ 2. MRI analysis */
const PLANES = ["axial", "coronal", "sagittal"];
function buildViewer(container, prefix) {
  container.innerHTML = PLANES.map((p) => `<div class="view" data-plane="${p}"><canvas></canvas><span class="tag">${p[0].toUpperCase() + p.slice(1)}</span><span class="idx"></span>
    <span class="orient" style="top:30px;left:50%">${ORIENT[p][0]}</span><span class="orient" style="bottom:30px;left:50%">${ORIENT[p][1]}</span>
    <span class="orient" style="top:50%;right:8px">${ORIENT[p][2]}</span><span class="orient" style="top:50%;left:8px">${ORIENT[p][3]}</span>
    <input type="range" id="${prefix}-${p}" min="0" max="1" value="0"></div>`).join("");
}
async function drawAnalysis() {
  if (!S.pid) return;
  const vol = await getVol(S.pid, S.mod);
  let lab = null;
  if (S.overlay !== "none") {
    try { lab = await getVol(S.pid, S.overlay === "gt" ? "seg" : "pred:" + S.overlay); } catch (e) { toast(e.message); }
  }
  for (const p of PLANES) {
    const view = $(`#viewer-a .view[data-plane="${p}"]`);
    const ax = planeIndex(p);
    const sl = $("input", view);
    sl.max = vol.shape[ax] - 1;
    sl.value = S.cursor[ax];
    $(".idx", view).textContent = `${S.cursor[ax] + 1} / ${vol.shape[ax]}`;
    renderSlice($("canvas", view), vol, p, S.cursor[ax], { lab, alpha: S.alpha, win: S.win, cross: crossFor(vol.shape, p, S.cursor) });
  }
}
function setupAnalysis() {
  buildViewer($("#viewer-a"), "va");
  $("#mod-seg").innerHTML = ["flair", "t1", "t1ce", "t2"].map((m) => `<button data-m="${m}" class="${m === S.mod ? "on" : ""}">${{ flair: "FLAIR", t1: "T1", t1ce: "T1ce", t2: "T2" }[m]}</button>`).join("");
  $("#mod-seg").addEventListener("click", (e) => {
    const m = e.target.dataset.m;
    if (!m) return;
    S.mod = m;
    $$("#mod-seg button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
    drawAnalysis().catch((er) => toast(er.message));
  });
  $("#ov-source").addEventListener("change", (e) => { S.overlay = e.target.value; drawAnalysis().catch((er) => toast(er.message)); });
  $("#ov-alpha").addEventListener("input", (e) => { S.alpha = +e.target.value; $("#ov-alpha-v").textContent = S.alpha.toFixed(2); drawAnalysis(); });
  $("#win").addEventListener("input", (e) => { S.win = +e.target.value; $("#win-v").textContent = `0-${S.win}`; drawAnalysis(); });
  for (const p of PLANES) {
    const view = $(`#viewer-a .view[data-plane="${p}"]`);
    $("input", view).addEventListener("input", (e) => { S.cursor[planeIndex(p)] = +e.target.value; drawAnalysis(); });
    $("canvas", view).addEventListener("click", (ev) => {
      if (!S.info) return;
      const [c, r] = canvasPos(ev.target, ev);
      const [, Y, Z] = S.info.shape;
      if (p === "axial") { S.cursor[0] = c; S.cursor[1] = Y - 1 - r; }
      else if (p === "coronal") { S.cursor[0] = c; S.cursor[2] = Z - 1 - r; }
      else { S.cursor[1] = c; S.cursor[2] = Z - 1 - r; }
      S.cursor = S.cursor.map((v, i) => Math.max(0, Math.min(S.info.shape[i] - 1, v)));
      drawAnalysis();
    });
  }
  $("#plist-split").addEventListener("change", renderPatientList);
  $("#btn-upload").addEventListener("click", () => $("#upload-modal").classList.add("on"));
  $("#up-cancel").addEventListener("click", () => $("#upload-modal").classList.remove("on"));
  $("#up-send").addEventListener("click", async () => {
    const form = $("#upload-form");
    if (!form.flair.files.length) return toast("FLAIR is required.");
    try {
      const r = await api("api/upload", { method: "POST", body: new FormData(form) }, "Importing MRI…");
      $("#upload-modal").classList.remove("on");
      form.reset();
      await loadPatients();
      S.autoAnalyze = r.patient_id;
      location.hash = "#/analysis";
      await selectPatient(r.patient_id);
    } catch (e) { toast(e.message); }
  });
}
function renderPatientList() {
  const split = $("#plist-split").value;
  const rows = S.patients.filter((p) => !split || p.split === split);
  $("#plist-count").textContent = rows.length;
  $("#plist").innerHTML = `<table><tr><th>Patient</th><th>Grade</th><th class="num">WT mL</th></tr>${rows.map((p) => `<tr class="click ${p.patient_id === S.pid ? "sel" : ""}" data-pid="${esc(p.patient_id)}"><td>${esc((p.name || p.patient_id).replace("BraTS20_Training_", "#"))} <span class="badge split">${esc(p.split)}</span></td><td>${gradeBadge(p.grade_name)}</td><td class="num mono">${f(p.vol_WT_ml, 1)}</td></tr>`).join("")}</table>`;
  $$("#plist tr[data-pid]").forEach((tr) => tr.addEventListener("click", () => selectPatient(tr.dataset.pid).catch((e) => toast(e.message))));
}
function infoPanel() {
  const i = S.info;
  const vols = i.volumes;
  const loc = i.localization;
  const bar = (k, lbl, col) => `<div style="margin:8px 0"><div style="display:flex;justify-content:space-between"><span>${lbl}</span><span class="mono">${f(vols[`vol_${k}_ml`], 1)} mL</span></div><div class="bar"><i style="width:${Math.min(100, (100 * vols[`vol_${k}_ml`]) / Math.max(vols.vol_WT_ml, 1e-6))}%;background:${col}"></i></div></div>`;
  return `<div class="ai-box"><button class="btn primary" id="btn-full" data-icon="sparkles" style="width:100%">Run full AI analysis</button>
    <p class="note" style="margin:8px 0 0">Automatic segmentation → volumes → localization → shape → grade. No manual segmentation needed.</p><div id="full-out"></div></div>
    <h3 data-icon="doc" style="margin-top:18px">${esc(i.name || i.patient_id)}</h3>
    <dl class="kv"><dt>Split</dt><dd>${esc(i.split)}</dd><dt>Grade</dt><dd>${gradeBadge(i.grade)}</dd><dt>Volume shape</dt><dd>${i.shape.join(" × ")}</dd>
    <dt>Voxel (mm)</dt><dd>${i.spacing.map((s) => f(s, 2)).join(" × ")}</dd><dt>Brain volume</dt><dd>${f(i.brain_volume_ml, 0)} mL</dd>
    ${i.missing?.length ? `<dt>Missing</dt><dd>${esc(i.missing.join(", "))}</dd>` : ""}</dl>
    ${vols ? `<h3 style="margin-top:18px" data-icon="vol">Tumor volumes</h3>${bar("WT", "Whole tumor (WT)", COL.blue)}${bar("TC", "Tumor core (TC)", COL.violet)}${bar("ET", "Enhancing (ET)", COL.amber)}${bar("ED", "Edema (ED)", COL.green)}${bar("NCR", "Necrosis (NCR)", COL.rose)}` : `<p class="note">No ground-truth segmentation.</p>`}
    ${loc ? `<h3 style="margin-top:18px" data-icon="target">Localization</h3><dl class="kv"><dt>Side</dt><dd>${esc(loc.hemisphere)}</dd><dt>Ant./post.</dt><dd>${esc(loc.anterior_posterior)}</dd><dt>Sup./inf.</dt><dd>${esc(loc.superior_inferior)}</dd><dt>Depth</dt><dd>${f(loc.depth_to_surface_mm, 1)} mm (${esc(loc.depth_category)})</dd></dl>` : ""}
    <h3 style="margin-top:18px" data-icon="brain">Grade prediction</h3><div id="cls-out"><button class="btn" id="btn-cls" data-icon="play">Predict with 3D CNN</button></div>`;
}
async function pageAnalysis() {
  if (!S.patients.length) await loadPatients();
  renderPatientList();
  const opts = [`<option value="none">No overlay</option>`];
  if (S.info?.has_seg) opts.push(`<option value="gt">Ground truth</option>`);
  (S.info?.predictions || []).forEach((m) => opts.push(`<option value="${esc(m)}">Prediction · ${esc(m)}</option>`));
  $("#ov-source").innerHTML = opts.join("");
  $("#ov-source").value = S.overlay;
  if (!S.pid) {
    $("#pinfo").innerHTML = empty("<b>Select a patient</b>from the list, the search bar, or import your own MRI.");
    $$("#viewer-a canvas").forEach((c) => { c.width = c.height = 1; });
    return;
  }
  $("#pinfo").innerHTML = infoPanel();
  icons($("#pinfo"));
  $("#btn-cls")?.addEventListener("click", async () => {
    try {
      const r = await api(`api/patient/${encodeURIComponent(S.pid)}/classify`, {}, "Classifying…");
      $("#cls-out").innerHTML = `<div class="metric-big"><div><span>Predicted grade</span><b>${esc(r.grade)}</b></div><div><span>P(HGG)</span><b>${f(r.prob_HGG, 2)}</b></div></div>
        <p class="note">${r.legacy_model ? "Legacy FLAIR-only model (.pkl): grade only." : `Pseudo-profile: ${esc(r.pseudo_profile)} (radiological cluster, not histology).`}</p>`;
    } catch (e) { $("#cls-out").innerHTML = `<p class="note">${esc(e.message)}</p>`; }
  });
  $("#btn-full").addEventListener("click", () => runFullAnalysis().catch((e) => toast(e.message)));
  if (S.autoAnalyze === S.pid) { S.autoAnalyze = null; runFullAnalysis().catch((e) => toast(e.message)); }
  await drawAnalysis();
}
async function runFullAnalysis() {
  const r = await post(`api/patient/${encodeURIComponent(S.pid)}/analyze`, {}, "Full AI analysis (segmentation + classification)…");
  const row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
  let html = "";
  if (r.segmentation) {
    const v = r.segmentation.volumes_pred, loc = r.localization;
    html += `<div class="metric-big" style="margin-top:12px"><div><span>Tumor</span><b style="color:${r.tumor_detected ? "var(--rose)" : "var(--teal)"}">${r.tumor_detected ? "Detected" : "Not detected"}</b></div>
      <div><span>Whole tumor</span><b>${f(v.vol_WT_ml, 1)}</b> <span>mL</span></div></div>
      <dl class="kv" style="margin-top:10px">${row("Model", esc(r.model))}${row("Tumor core (TC)", f(v.vol_TC_ml, 1) + " mL")}${row("Enhancing (ET)", f(v.vol_ET_ml, 1) + " mL")}
      ${row("Edema (ED)", f(v.vol_ED_ml, 1) + " mL")}${row("Necrosis (NCR)", f(v.vol_NCR_ml, 1) + " mL")}
      ${loc ? row("Side", esc(loc.hemisphere)) + row("Position", `${esc(loc.anterior_posterior)}, ${esc(loc.superior_inferior)}`) + row("Depth", `${f(loc.depth_to_surface_mm, 1)} mm`) : ""}
      ${r.shape?.WT?.sphericity ? row("Sphericity (WT)", f(r.shape.WT.sphericity, 2)) + row("Max diameter", f(r.shape.WT.max_3d_diameter_mm, 1) + " mm") : ""}
      ${r.segmentation.metrics ? row("Dice WT vs ground truth", f(r.segmentation.metrics.WT.dice, 3)) : ""}</dl>`;
    if (!S.info.predictions.includes(r.model)) S.info.predictions.push(r.model);
    volCache.delete(S.pid + "|pred:" + r.model);
    S.overlay = r.model;
  } else html += `<div class="warnbox" style="margin-top:12px">Segmentation: ${esc(r.segmentation_error)}</div>`;
  if (r.classification) {
    const c = r.classification;
    html += `<div class="metric-big" style="margin-top:10px"><div><span>Predicted grade</span><b>${esc(c.grade)}</b></div><div><span>P(HGG)</span><b>${f(c.prob_HGG, 2)}</b></div></div>
      <p class="note">${c.legacy_model ? "Grade from the legacy FLAIR-only model (.pkl)." : `Radiological profile: <b>${esc(c.pseudo_profile)}</b> (cluster of tumor volumes, not a histological type).`}</p>`;
  } else html += `<div class="warnbox">Classification: ${esc(r.classification_error)}</div>`;
  html += `<p class="note">LGG = low-grade glioma, HGG = high-grade glioma. Not a medical diagnosis.</p>`;
  $("#full-out").innerHTML = html;
  const sel = $("#ov-source");
  if (r.model && ![...sel.options].some((o) => o.value === r.model)) sel.insertAdjacentHTML("beforeend", `<option value="${esc(r.model)}">Prediction · ${esc(r.model)}</option>`);
  if (r.model) sel.value = r.model;
  await drawAnalysis();
}

/* ------------------------------------------------------------------ 3. segmentation */
function setupSegmentation() {
  $("#seg-plane").innerHTML = PLANES.map((p) => `<button data-p="${p}" class="${p === S.segPlane ? "on" : ""}">${p}</button>`).join("");
  $("#seg-mod").innerHTML = ["flair", "t1ce", "t2", "t1"].map((m) => `<button data-m="${m}" class="${m === S.segMod ? "on" : ""}">${m.toUpperCase().replace("CE", "ce")}</button>`).join("");
  $("#seg-plane").addEventListener("click", (e) => {
    if (!e.target.dataset.p) return;
    S.segPlane = e.target.dataset.p;
    $$("#seg-plane button").forEach((b) => b.classList.toggle("on", b.dataset.p === S.segPlane));
    S.segIdx = S.cursor[planeIndex(S.segPlane)];
    drawSeg();
  });
  $("#seg-mod").addEventListener("click", (e) => {
    if (!e.target.dataset.m) return;
    S.segMod = e.target.dataset.m;
    $$("#seg-mod button").forEach((b) => b.classList.toggle("on", b.dataset.m === S.segMod));
    drawSeg();
  });
  $("#seg-slice").addEventListener("input", (e) => { S.segIdx = +e.target.value; drawSeg(); });
  $("#btn-seg").addEventListener("click", runSeg);
  $("#seg-views").innerHTML = ["Original MRI", "Ground truth", "Prediction", "Error map (WT)"].map((t, i) => `<div class="view"><canvas id="segc${i}"></canvas><span class="tag">${t}</span></div>`).join("");
}
async function runSeg() {
  const model = $("#seg-model").value;
  if (!S.pid || !model) return toast("Select a patient and a trained model.");
  try {
    S.segRes = await post(`api/patient/${encodeURIComponent(S.pid)}/segment?model=${encodeURIComponent(model)}`, {}, "Running 3D segmentation…");
    if (!S.info.predictions.includes(model)) S.info.predictions.push(model);
    volCache.delete(S.pid + "|pred:" + model);
    renderSegMetrics();
    await drawSeg();
  } catch (e) { toast(e.message); }
}
function renderSegMetrics() {
  const r = S.segRes;
  if (!r) return;
  $("#seg-time").textContent = `${esc(r.model)} · ${r.time_s}s`;
  if (!r.metrics) {
    $("#seg-metrics").innerHTML = `<p class="note">No ground truth for this case — predicted volumes only.</p><dl class="kv">${["WT", "TC", "ET"].map((k) => `<dt>${k}</dt><dd>${f(r.volumes_pred[`vol_${k}_ml`], 1)} mL</dd>`).join("")}</dl>`;
    return;
  }
  const m = r.metrics;
  $("#seg-metrics").innerHTML = `<table><tr><th>Region</th><th class="num">Dice</th><th class="num">IoU</th><th class="num">HD95 (mm)</th><th class="num">Sensitivity</th><th class="num">GT mL</th><th class="num">Pred mL</th></tr>
    ${["WT", "TC", "ET"].map((k) => `<tr><td><b>${k}</b> <span class="note">${{ WT: "whole tumor", TC: "tumor core", ET: "enhancing" }[k]}</span></td><td class="num mono">${f(m[k].dice, 3)}</td><td class="num mono">${f(m[k].iou, 3)}</td><td class="num mono">${f(m[k].hd95, 1)}</td><td class="num mono">${f(m[k].sensitivity, 3)}</td><td class="num mono">${f(m[k].volume_gt_ml, 1)}</td><td class="num mono">${f(m[k].volume_pred_ml, 1)}</td></tr>`).join("")}</table>`;
  chart("ch-seg", { type: "bar", data: { labels: ["WT", "TC", "ET"], datasets: [{ label: "Dice", data: ["WT", "TC", "ET"].map((k) => m[k].dice), backgroundColor: [COL.blue, COL.violet, COL.amber], borderRadius: 6 }, { label: "IoU", data: ["WT", "TC", "ET"].map((k) => m[k].iou), backgroundColor: "rgba(20,211,196,.55)", borderRadius: 6 }] }, options: { scales: { y: { min: 0, max: 1 } } } });
}
async function drawSeg() {
  if (!S.pid) return;
  const model = $("#seg-model").value;
  const base = await getVol(S.pid, S.segMod);
  const ax = planeIndex(S.segPlane);
  const sl = $("#seg-slice");
  sl.max = base.shape[ax] - 1;
  sl.value = S.segIdx;
  $("#seg-slice-v").textContent = `${S.segIdx + 1} / ${base.shape[ax]}`;
  const gt = S.info.has_seg ? await getVol(S.pid, "seg") : null;
  const hasPred = model && S.info.predictions.includes(model);
  const pred = hasPred ? await getVol(S.pid, "pred:" + model) : null;
  const o = { win: 255, alpha: 0.5 };
  renderSlice($("#segc0"), base, S.segPlane, S.segIdx, o);
  renderSlice($("#segc1"), base, S.segPlane, S.segIdx, { ...o, lab: gt });
  renderSlice($("#segc2"), base, S.segPlane, S.segIdx, { ...o, lab: pred, dim: pred ? 1 : 0.35 });
  if (gt && pred) renderSlice($("#segc3"), base, S.segPlane, S.segIdx, { ...o, dim: 0.6, diff: { gt, pred } });
  else renderSlice($("#segc3"), base, S.segPlane, S.segIdx, { ...o, dim: 0.25 });
}
async function pageSegmentation() {
  S.status = S.status || (await api("api/status"));
  const models = S.status.segmentation_models || [];
  const sel = $("#seg-model");
  const prev = sel.value;
  sel.innerHTML = models.length ? models.map((m) => `<option value="${esc(m.name)}">${esc(m.name)}</option>`).join("") : `<option value="">No trained model</option>`;
  if (prev) sel.value = prev;
  $("#btn-seg").disabled = !models.length;
  $("#seg-empty").innerHTML = !models.length ? `<div class="warnbox">No segmentation model found in <code>models/segmentation/&lt;name&gt;/best.pt</code>. Train one with <b>notebook 03_segmentation_3d_unet</b> (Colab GPU), then reload this page.</div>` : "";
  if (needPatient($("#seg-metrics"))) { $("#seg-views").style.opacity = 0.3; return; }
  $("#seg-views").style.opacity = 1;
  S.segIdx = S.cursor[planeIndex(S.segPlane)];
  if (!S.segRes) $("#seg-metrics").innerHTML = empty("Run a model to compute Dice, IoU and HD95.");
  await drawSeg();
}

/* ------------------------------------------------------------------ 4. 3D */
const T = { ready: false, meshes: {}, key: null };
function initThree() {
  if (T.ready) return;
  const el = $("#three");
  T.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  T.renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(T.renderer.domElement);
  T.scene = new THREE.Scene();
  T.camera = new THREE.PerspectiveCamera(40, 1, 1, 3000);
  T.camera.position.set(260, 120, 260);
  T.controls = new OrbitControls(T.camera, T.renderer.domElement);
  T.controls.enableDamping = true;
  T.controls.autoRotate = true;
  T.controls.autoRotateSpeed = 1.2;
  T.scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x0a0f25, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, 1.3);
  dl.position.set(200, 300, 150);
  T.scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0x8b5cf6, 0.6);
  dl2.position.set(-200, -100, -200);
  T.scene.add(dl2);
  T.group = new THREE.Group();
  T.group.rotation.x = -Math.PI / 2; // RAS z (superior) -> screen up
  T.scene.add(T.group);
  T.ring = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 128), new THREE.MeshBasicMaterial({ color: 0x4f7cff, transparent: true, opacity: 0.4, side: THREE.DoubleSide }));
  T.ring.rotation.x = -Math.PI / 2;
  T.scene.add(T.ring);
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    T.renderer.setSize(w, h);
    T.camera.aspect = w / h;
    T.camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el);
  resize();
  const loop = () => { requestAnimationFrame(loop); if (S.page === "3d") { T.controls.update(); T.renderer.render(T.scene, T.camera); } };
  loop();
  T.ready = true;
}
function b64(s, Type) {
  const bin = atob(s);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new Type(u.buffer);
}
const LAYERS = [["brain", "Brain", "#9aa7c7"], ["ED", "Edema (ED)", "#22d3a6"], ["NCR", "Necrosis (NCR)", "#f43f5e"], ["ET", "Enhancing (ET)", "#fbbf24"]];
async function load3D() {
  const source = $("#mesh-source").value;
  const key = S.pid + "|" + source;
  if (T.key === key) return;
  const data = await api(`api/patient/${encodeURIComponent(S.pid)}/mesh?source=${encodeURIComponent(source)}`, {}, "Building 3D surfaces…");
  Object.values(T.meshes).forEach((m) => { T.group.remove(m); m.geometry.dispose(); m.material.dispose(); });
  T.meshes = {};
  const stats = [];
  for (const [k] of LAYERS) {
    const d = data[k];
    if (!d) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(b64(d.vertices, Float32Array), 3));
    g.setAttribute("normal", new THREE.BufferAttribute(b64(d.normals, Float32Array).map((v) => -v), 3));
    g.setIndex(new THREE.BufferAttribute(b64(d.faces, Uint32Array), 1));
    const brain = k === "brain", edema = k === "ED";
    const op = brain ? +$("#brain-op").value : edema ? +$("#ed-op").value : 1;
    const mat = new THREE.MeshStandardMaterial({ color: d.color, roughness: brain ? 0.55 : 0.35, metalness: 0.05, transparent: brain || edema, opacity: op, depthWrite: !(brain || edema), side: THREE.DoubleSide, emissive: brain ? 0x000000 : new THREE.Color(d.color).multiplyScalar(0.25) });
    const mesh = new THREE.Mesh(g, mat);
    mesh.renderOrder = brain ? 3 : edema ? 2 : 1;
    mesh.visible = $(`#layer-${k}`)?.checked ?? true;
    T.group.add(mesh);
    T.meshes[k] = mesh;
    stats.push(`${k}: ${d.n_faces.toLocaleString()} faces`);
  }
  $("#mesh-stats").textContent = stats.join(" · ");
  T.key = key;
  fit3D();
}
function fit3D() {
  const ref = T.meshes.brain || Object.values(T.meshes)[0];
  if (!ref) return;
  T.group.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(ref);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const r = sphere.radius;
  T.controls.target.copy(sphere.center);
  T.camera.position.copy(sphere.center).add(new THREE.Vector3(1.25, 0.6, 1.25).multiplyScalar(r * 1.05));
  T.camera.near = r / 50; T.camera.far = r * 30; T.camera.updateProjectionMatrix();
  T.ring.scale.setScalar(r * 0.75);
  T.ring.position.set(sphere.center.x, box.min.y - r * 0.05, sphere.center.z);
  T.home = T.camera.position.clone();
}
function setup3D() {
  $("#mesh-layers").innerHTML = LAYERS.map(([k, l, c]) => `<label class="check"><input type="checkbox" id="layer-${k}" checked><i style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${c}"></i>${l}</label>`).join("");
  LAYERS.forEach(([k]) => $(`#layer-${k}`).addEventListener("change", (e) => { if (T.meshes[k]) T.meshes[k].visible = e.target.checked; }));
  $("#brain-op").addEventListener("input", (e) => { $("#brain-op-v").textContent = (+e.target.value).toFixed(2); if (T.meshes.brain) T.meshes.brain.material.opacity = +e.target.value; });
  $("#autorot").addEventListener("change", (e) => { if (T.controls) T.controls.autoRotate = e.target.checked; });
  $("#ed-op").addEventListener("input", (e) => { $("#ed-op-v").textContent = (+e.target.value).toFixed(2); if (T.meshes.ED) T.meshes.ED.material.opacity = +e.target.value; });
  $("#btn-reset3d").addEventListener("click", fit3D);
  $("#btn-shot").addEventListener("click", () => { const a = document.createElement("a"); a.href = T.renderer.domElement.toDataURL("image/png"); a.download = `${S.pid}_3d.png`; a.click(); });
  $("#mesh-source").addEventListener("change", () => load3D().catch((e) => toast(e.message)));
}
async function page3D() {
  initThree();
  const sel = $("#mesh-source");
  const prev = sel.value;
  const opts = [];
  if (S.info?.has_seg) opts.push(`<option value="gt">Ground truth</option>`);
  (S.info?.predictions || []).forEach((m) => opts.push(`<option value="${esc(m)}">Prediction · ${esc(m)}</option>`));
  sel.innerHTML = opts.join("") || `<option value="gt">Ground truth</option>`;
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  if (!S.pid) { $("#mesh-stats").innerHTML = "Select a patient first."; return; }
  await load3D();
}

/* ------------------------------------------------------------------ 5. radiomics */
let lastRad = null;
function setupRadiomics() {
  $("#btn-rad").addEventListener("click", runRadiomics);
  $("#btn-rad-csv").addEventListener("click", () => {
    if (!lastRad) return toast("Compute the features first.");
    const rows = [["group", "region_or_modality", "feature", "value"]];
    for (const [r, o] of Object.entries(lastRad.shape)) for (const [k, v] of Object.entries(o)) rows.push(["shape", r, k, v]);
    for (const [m, o] of Object.entries(lastRad.first_order)) for (const [reg, fe] of Object.entries(o)) for (const [k, v] of Object.entries(fe)) rows.push(["first_order", `${m}:${reg}`, k, v]);
    for (const [m, o] of Object.entries(lastRad.texture)) for (const [k, v] of Object.entries(o)) rows.push(["texture", m, k, v]);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" }));
    a.download = `${S.pid}_radiomics.csv`;
    a.click();
  });
}
async function runRadiomics() {
  if (!S.pid) return toast("Select a patient first.");
  try {
    lastRad = await api(`api/patient/${encodeURIComponent(S.pid)}/radiomics?source=${encodeURIComponent($("#rad-source").value)}`, {}, "Computing radiomics…");
    renderRadiomics(lastRad);
  } catch (e) { toast(e.message); }
}
function renderRadiomics(r) {
  const sh = r.shape;
  const card = (k, lbl, col) => {
    const s = sh[k] || {};
    return `<div class="panel"><h3><i style="width:10px;height:10px;border-radius:3px;background:${col};display:inline-block"></i>${lbl}</h3>
      <div class="metric-big"><div><span>Volume</span><b>${f(s.volume_ml, 1)}</b> <span>mL</span></div><div><span>Sphericity</span><b>${f(s.sphericity, 2)}</b></div></div>
      <dl class="kv" style="margin-top:12px"><dt>Surface area</dt><dd>${f(s.surface_area_cm2, 1)} cm²</dd><dt>Max 3D diameter</dt><dd>${f(s.max_3d_diameter_mm, 1)} mm</dd>
      <dt>Axes (mm)</dt><dd>${f(s.major_axis_mm, 0)} / ${f(s.minor_axis_mm, 0)} / ${f(s.least_axis_mm, 0)}</dd><dt>Elongation</dt><dd>${f(s.elongation, 2)}</dd><dt>Flatness</dt><dd>${f(s.flatness, 2)}</dd>
      <dt>Surface/volume</dt><dd>${f(s.surface_volume_ratio, 3)}</dd></dl></div>`;
  };
  const mods = Object.keys(r.first_order);
  const fo = ["mean", "std", "median", "p10", "p90", "skewness", "kurtosis", "entropy"];
  const tx = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation"];
  $("#rad-out").innerHTML = `<div class="grid g3">${card("WT", "Whole tumor (WT)", COL.blue)}${card("TC", "Tumor core (TC)", COL.violet)}${card("ET", "Enhancing tumor (ET)", COL.amber)}</div>
    <div class="grid g2" style="margin-top:16px">${card("ED", "Edema (ED)", COL.green)}${card("NCR", "Necrosis (NCR)", COL.rose)}</div>
    <div class="grid g2" style="margin-top:16px">
      <div class="panel"><h3 data-icon="table">First-order intensity — tumor (WT) vs healthy brain<span class="right">z-scored intensities</span></h3>
        <table><tr><th>Feature</th>${mods.map((m) => `<th class="num">${m} tumor</th><th class="num">${m} healthy</th>`).join("")}</tr>
        ${fo.map((k) => `<tr><td>${k}</td>${mods.map((m) => `<td class="num mono">${f(r.first_order[m].WT[k])}</td><td class="num mono note">${f(r.first_order[m].healthy[k])}</td>`).join("")}</tr>`).join("")}</table></div>
      <div class="panel"><h3 data-icon="bars">Mean intensity: tumor vs healthy</h3><div class="chart-box tall"><canvas id="ch-rad"></canvas></div></div>
    </div>
    <div class="panel" style="margin-top:16px"><h3 data-icon="layers">Texture (GLCM, 32 grey levels, WT region)</h3>
      <table><tr><th>Feature</th>${mods.map((m) => `<th class="num">${m}</th>`).join("")}</tr>${tx.map((k) => `<tr><td>${k.replace("glcm_", "")}</td>${mods.map((m) => `<td class="num mono">${f(r.texture[m][k], 3)}</td>`).join("")}</tr>`).join("")}</table>
      <p class="note">Tumor / brain volume ratio: <b class="mono">${f(sh.WT?.tumor_brain_ratio_pct, 2)} %</b> · brain volume ${f(sh.brain?.volume_ml, 0)} mL.</p></div>`;
  icons($("#rad-out"));
  chart("ch-rad", { type: "bar", data: { labels: mods, datasets: [{ label: "Tumor (WT)", data: mods.map((m) => r.first_order[m].WT.mean), backgroundColor: COL.rose, borderRadius: 5 }, { label: "Healthy", data: mods.map((m) => r.first_order[m].healthy.mean), backgroundColor: COL.blue, borderRadius: 5 }] } });
}
async function pageRadiomics() {
  const opts = [`<option value="gt">Ground truth</option>`].concat((S.info?.predictions || []).map((m) => `<option value="${esc(m)}">Prediction · ${esc(m)}</option>`));
  const prev = $("#rad-source").value;
  $("#rad-source").innerHTML = opts.join("");
  if ([...$("#rad-source").options].some((o) => o.value === prev)) $("#rad-source").value = prev;
  if (!S.pid) { $("#rad-out").innerHTML = empty("<b>No patient selected</b>Use the search bar first."); lastRad = null; return; }
  if (lastRad && lastRad._pid !== S.pid) lastRad = null;
  if (!lastRad) $("#rad-out").innerHTML = empty(`Click <b>Compute</b> to extract the features of ${esc(S.pid)}.`);
}

/* ------------------------------------------------------------------ 6. generation */
let genPatients = null;
function setupGeneration() {
  const bind = (id, fmt) => $(id).addEventListener("input", (e) => ($(id + "-v").textContent = fmt(e.target.value)));
  bind("#gen-str", (v) => (+v).toFixed(2));
  bind("#gen-steps", (v) => v);
  bind("#gen-n", (v) => v);
  bind("#gen-res", (v) => (+v).toFixed(2));
  bind("#gen-grow", (v) => v);
  bind("#gen-slice", (v) => (+v < 0 ? "tumor (auto)" : +v + 1));
  const modeUI = () => {
    const m = $("#gen-mode").value;
    $("#gen-res-box").style.display = m === "postop" ? "" : "none";
    $("#gen-grow-box").style.display = m === "progression" ? "" : "none";
    $("#gen-str").closest("label").style.display = m === "synthesis" ? "none" : "";
  };
  $("#gen-mode").addEventListener("change", modeUI);
  modeUI();
  $("#gen-pid").addEventListener("change", genSlices);
  $("#btn-gen").addEventListener("click", runGen);
}
async function genSlices() {
  const pid = $("#gen-pid").value;
  if (!pid) return;
  const r = await api(`api/generation/slices/${encodeURIComponent(pid)}`);
  const s = $("#gen-slice");
  s.min = -1; s.max = r.n_slices - 1; s.value = -1;
  $("#gen-slice-v").textContent = "tumor (auto)";
}
async function runGen() {
  const pid = $("#gen-pid").value;
  if (!pid) return;
  const mode = $("#gen-mode").value;
  const body = { patient_id: pid, mode, slice_index: +$("#gen-slice").value < 0 ? null : +$("#gen-slice").value, strength: +$("#gen-str").value, steps: +$("#gen-steps").value, n_samples: +$("#gen-n").value, seed: +$("#gen-seed").value, residual: +$("#gen-res").value, grow: +$("#gen-grow").value };
  try {
    const r = await post("api/generation/run", body, "Running Med-DDPM sampling…");
    const im = r.images;
    const scen = !["synthesis", "img2img"].includes(mode);
    const card = (src, cap) => (src ? `<div class="img-card"><img src="${src}"><div class="cap">${cap}</div></div>` : "");
    $("#gen-info").textContent = `${pid} · slice ${r.slice_index + 1} · ${mode}`;
    $("#gen-images").innerHTML = `<div class="img-grid">${card(im.condition, "Condition (input channel)")}${card(im.real_seg, "Real T2 + segmentation")}${card(scen ? im.generated_seg : im.generated, scen ? "Simulated T2 + modified segmentation" : "Generated T2")}${card(im.error, "|generated − real| (error map)")}${card(im.uncertainty, "Uncertainty (std over samples)")}</div>
      ${scen ? `<p class="note" style="margin-top:10px">Scenario images are simulations for teaching: BraTS has no follow-up MRI to validate them.</p>` : ""}`;
    const m = r.metrics;
    $("#gen-metrics").innerHTML = scen
      ? `<div class="metric-big"><div><span>Tumor area before</span><b>${f(m.tumor_area_before_px, 0)}</b> px</div><div><span>After</span><b>${f(m.tumor_area_after_px, 0)}</b> px</div><div><span>Change</span><b>${f(m.area_change_pct, 1)} %</b></div></div>`
      : `<div class="metric-big"><div><span>PSNR (dB)</span><b>${f(m.PSNR, 2)}</b></div><div><span>SSIM</span><b>${f(m.SSIM, 3)}</b></div><div><span>MAE</span><b>${f(m.MAE, 3)}</b></div></div>
         <p class="note" style="margin-top:10px">Computed inside the brain on this slice, after scaling both images with the real T2 1–99 % percentiles.${mode === "img2img" ? " Img2img keeps part of the real image: use <b>Synthesis</b> to measure the real generation quality." : ""}</p>`;
  } catch (e) { toast(e.message); }
}
async function pageGeneration() {
  if (!genPatients) {
    const r = await api("api/generation/patients", {}, "Loading Med-DDPM…");
    genPatients = r;
    if (!r.available) {
      $("#gen-warn").innerHTML = `<div class="warnbox">Med-DDPM is not available: ${esc(r.error || "unknown error")}. Expected the npz index in <code>MedDDPM_dataset_T2/</code> and the checkpoint <code>MedDDPM2D_T2_models/best_model_EMA.pt</code> (notebook 04).</div>`;
      $("#btn-gen").disabled = true;
      return;
    }
    if (r.warning) $("#gen-warn").innerHTML = `<div class="warnbox">${esc(r.warning)}</div>`;
    $("#gen-pid").innerHTML = r.patients.map((p) => `<option value="${esc(p.patient_id)}">${esc(p.patient_id)} (${esc(p.split)})</option>`).join("");
    $("#gen-info").textContent = `checkpoint epoch ${r.info.epoch ?? "?"} · cond channels ${r.info.cond_ch} · T=${r.info.num_timesteps}`;
  }
  if (S.pid && [...$("#gen-pid").options].some((o) => o.value === S.pid) && $("#gen-pid").value !== S.pid) $("#gen-pid").value = S.pid;
  await genSlices();
}

/* ------------------------------------------------------------------ 7. models & reports */
async function pageModels() {
  const r = await api("api/models/compare", {}, "Loading evaluations…");
  const seg = r.evaluations.filter((e) => e.task === "segmentation");
  const other = r.evaluations.filter((e) => e.task !== "segmentation");
  const regs = ["WT", "TC", "ET"];
  const ms = (o, k, d = 3) => (o && o[k] ? `${f(o[k].mean, d)} <span class="note">± ${f(o[k].std, d)}</span>` : "—");
  $("#cmp-table").innerHTML = seg.length
    ? `<table><tr><th>Model</th><th class="num">Cases</th>${regs.map((g) => `<th class="num">Dice ${g}</th>`).join("")}${regs.map((g) => `<th class="num">IoU ${g}</th>`).join("")}${regs.map((g) => `<th class="num">HD95 ${g}</th>`).join("")}</tr>
      ${seg.map((e) => `<tr><td><b>${esc(e.model_label || e.model)}</b> <span class="note">${esc(e.split)} · epoch ${e.epoch ?? "?"}</span></td><td class="num mono">${e.n_cases}</td>${regs.map((g) => `<td class="num mono">${ms(e.summary[g], "dice")}</td>`).join("")}${regs.map((g) => `<td class="num mono">${ms(e.summary[g], "iou")}</td>`).join("")}${regs.map((g) => `<td class="num mono">${ms(e.summary[g], "hd95", 1)}</td>`).join("")}</tr>`).join("")}</table>`
    : empty("<b>No segmentation evaluation yet.</b>Run the evaluation cell of notebook 03: it writes <code>results/eval/seg_&lt;model&gt;.json</code>.");
  const ds = (key) => seg.map((e, i) => ({ label: e.model_label || e.model, data: regs.map((g) => e.summary[g][key]?.mean ?? null), backgroundColor: SERIES[i % SERIES.length], borderRadius: 5 }));
  chart("ch-cmp-dice", { type: "bar", data: { labels: regs, datasets: ds("dice") }, options: { scales: { y: { min: 0, max: 1 } } } });
  chart("ch-cmp-hd", { type: "bar", data: { labels: regs, datasets: ds("hd95") } });
  const hist = Object.entries(r.histories);
  chart("ch-cmp-hist", { type: "line", data: { datasets: hist.map(([n, h], i) => ({ label: n, data: h.filter((x) => x.val_dice_mean !== undefined).map((x) => ({ x: x.epoch, y: x.val_dice_mean })), borderColor: SERIES[i % SERIES.length], backgroundColor: SERIES[i % SERIES.length], tension: 0.3, pointRadius: 2 })) }, options: { scales: { x: { type: "linear", title: { display: true, text: "epoch" } }, y: { min: 0, max: 1 } } } });
  chart("ch-cmp-cases", { type: "scatter", data: { datasets: seg.map((e, i) => ({ label: e.model_label || e.model, data: e.per_case.map((c) => ({ x: i + (Math.random() - 0.5) * 0.5, y: c.WT.dice })), backgroundColor: SERIES[i % SERIES.length] + "cc", pointRadius: 3 })) }, options: { scales: { x: { min: -0.7, max: Math.max(seg.length - 0.3, 0.7), ticks: { stepSize: 1, callback: (v) => seg[v]?.model_label || seg[v]?.model || "" } }, y: { min: 0, max: 1, title: { display: true, text: "Dice WT" } } } } });
  $("#cmp-other").innerHTML = other.length
    ? `<table><tr><th>Task</th><th>Model</th><th>Metrics</th></tr>${other.map((e) => `<tr><td>${esc(e.task)}</td><td>${esc(e.model || "")}</td><td class="mono">${esc(Object.entries(e).filter(([k, v]) => typeof v === "number").map(([k, v]) => `${k} ${f(v, 3)}`).join(" · "))}</td></tr>`).join("")}</table>`
    : empty("Classification and generation evaluations (notebooks 02 and 04) appear here.");
  S.status = S.status || (await api("api/status"));
  $("#rep-model").innerHTML = `<option value="">Ground truth only</option>` + (S.status.segmentation_models || []).map((m) => `<option value="${esc(m.name)}">With prediction · ${esc(m.name)}</option>`).join("");
}
async function makeReport() {
  if (!S.pid) return toast("Select a patient in the top bar first.");
  try {
    const r = await api(`api/report/${encodeURIComponent(S.pid)}?model=${encodeURIComponent($("#rep-model").value)}`, {}, "Building report…");
    const p = r.patient, v = p.volumes, l = p.localization, s = r.segmentation, rad = r.radiomics, c = r.classification;
    const row = (a, b) => `<tr><td>${a}</td><td>${b}</td></tr>`;
    $("#report-view").innerHTML = `<h1>BrainTumorLab — analysis report</h1><div>Patient <b>${esc(p.name || p.patient_id)}</b> · split ${esc(p.split)} · generated ${new Date().toLocaleString()}</div>
      <h4>Case</h4><table>${row("Grade (dataset label)", esc(p.grade || "n/a"))}${row("Volume / voxel size", `${p.shape.join("×")} · ${p.spacing.map((x) => f(x, 2)).join("×")} mm`)}${row("Brain volume", f(p.brain_volume_ml, 0) + " mL")}</table>
      ${v ? `<h4>Tumor volumes (ground truth)</h4><table><tr><th>WT</th><th>TC</th><th>ET</th><th>ED</th><th>NCR</th></tr><tr>${["WT", "TC", "ET", "ED", "NCR"].map((k) => `<td>${f(v[`vol_${k}_ml`], 1)} mL</td>`).join("")}</tr></table>` : ""}
      ${l ? `<h4>Localization</h4><table>${row("Side", esc(l.hemisphere))}${row("Anterior / posterior", esc(l.anterior_posterior))}${row("Superior / inferior", esc(l.superior_inferior))}${row("Depth to brain surface", `${f(l.depth_to_surface_mm, 1)} mm (${esc(l.depth_category)})`)}</table>` : ""}
      ${s ? `<h4>Automatic segmentation — ${esc(s.model)}</h4>${s.metrics ? `<table><tr><th>Region</th><th>Dice</th><th>IoU</th><th>HD95 (mm)</th><th>Pred. volume</th></tr>${["WT", "TC", "ET"].map((k) => `<tr><td>${k}</td><td>${f(s.metrics[k].dice, 3)}</td><td>${f(s.metrics[k].iou, 3)}</td><td>${f(s.metrics[k].hd95, 1)}</td><td>${f(s.metrics[k].volume_pred_ml, 1)} mL</td></tr>`).join("")}</table>` : ""}` : r.segmentation_error ? `<h4>Automatic segmentation</h4><p>${esc(r.segmentation_error)}</p>` : ""}
      ${rad ? `<h4>Radiomics (${s ? "prediction" : "ground truth"})</h4><table><tr><th>Region</th><th>Volume</th><th>Surface</th><th>Sphericity</th><th>Max diameter</th><th>Elongation</th></tr>${["WT", "TC", "ET"].map((k) => `<tr><td>${k}</td><td>${f(rad.shape[k].volume_ml, 1)} mL</td><td>${f(rad.shape[k].surface_area_cm2, 1)} cm²</td><td>${f(rad.shape[k].sphericity, 2)}</td><td>${f(rad.shape[k].max_3d_diameter_mm, 1)} mm</td><td>${f(rad.shape[k].elongation, 2)}</td></tr>`).join("")}</table>` : ""}
      ${c ? `<h4>Grade prediction (3D CNN)</h4><table>${row("Predicted grade", `${esc(c.grade)} (P(HGG) = ${f(c.prob_HGG, 2)})`)}</table>` : ""}
      <p class="disc">Mini project built on the BraTS 2020 dataset. Results come from machine-learning models and are not a medical diagnosis.</p>`;
    $("#report-view").style.display = "block";
    $("#btn-print").disabled = false;
  } catch (e) { toast(e.message); }
}

/* ------------------------------------------------------------------ boot */
const PAGES = { dashboard: pageDashboard, analysis: pageAnalysis, segmentation: pageSegmentation, "3d": page3D, radiomics: pageRadiomics, generation: pageGeneration, models: pageModels };
async function boot() {
  buildNav();
  icons();
  setupSearch();
  setupAnalysis();
  setupSegmentation();
  setup3D();
  setupRadiomics();
  setupGeneration();
  $("#btn-report").addEventListener("click", makeReport);
  $("#btn-print").addEventListener("click", () => window.print());
  window.addEventListener("hashchange", route);
  try {
    S.status = await api("api/status");
    const gpu = S.status.device.startsWith("cuda");
    $("#device-chip").innerHTML = `<span class="dot ${gpu ? "ok" : "warn"}"></span>${gpu ? "GPU" : "CPU"} · ${(S.status.segmentation_models || []).length} seg. model(s)`;
    await loadPatients();
  } catch (e) { toast(e.message); }
  await route();
}
boot();
