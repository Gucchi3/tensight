// @file tensight.js
// @brief Viewer for tensight record folders (C-06)
//
// The page is opened from file:// without a server, so it uses classic scripts only.
// Each data file holds one call per line, like tensight.rec({...}); (02c, EXT-01).
// The functions that do not use the DOM are in tensight_core.js (03 U-037 to U-044), which the page loads first.

(function () {
  "use strict";
  
  // ===== Core =====
  
  const core = window.tensightCore;   // functions without the DOM (tensight_core.js)
  
  // ===== Settings =====
  
  const GUT_L      = 64;    // space for labels on the left; the step bar and every step chart share it
  const GUT_R      = 76;    // space on the right of every step chart; the distribution chart puts its color bar there
  const RETRY_MS   = 300;   // wait before reading a file again (02c)
  const STRIPS     = 6;     // element strips shown before "show all"
  const KINDS      = ["output", "input", "output_grad", "value", "grad", "update", "note"];
  const GRAN       = {tensor: "テンソル全体", channel: "チャネルごと", group: "グループごと"};
  const ROT        = {az: -60, el: 28};
  const CMAPS      = core.CMAPS;
  const ICON_CHEV  = '<path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
  const ICON_THEME = '<circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M8 2a6 6 0 0 1 0 12z" fill="currentColor"/>';
  const ICON_MENU  = '<path d="M2.5 4h11M2.5 8h11M2.5 12h11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>';
  const ICON_PLAY  = '<path d="M5 3.2v9.6L12.6 8z" fill="currentColor"/>';
  const ICON_PAUSE = '<path d="M4.5 3.5h2.4v9H4.5zM9.1 3.5h2.4v9H9.1z" fill="currentColor"/>';
  const PLAY_MS    = 400;   // wait between two steps while playing at speed 1.0 (CR-005 proposal)
  const SPEEDS     = [0.5, 0.75, 1, 1.25, 1.5];
  
  // Values that the trend chart can show (FR-021)
  const METRICS = [
    {key: "min"    , get: r => r.min                                     , label: "最小"      },
    {key: "max"    , get: r => r.max                                     , label: "最大"      },
    {key: "mean"   , get: r => r.mean                                    , label: "平均"      },
    {key: "std"    , get: r => r.std                                     , label: "標準偏差"  },
    {key: "l2"     , get: r => r.l2                                      , label: "L2"        },
    {key: "clip"   , get: r => r.quant ? r.quant.clip : undefined        , label: "クリップ率", pct: true  },
    {key: "sqnr"   , get: r => r.quant ? r.quant.sqnr : undefined        , label: "SQNR"      , unit: " dB"},
    {key: "ratio"  , get: r => r.ratio                                   , label: "更新率"    },
    {key: "near"   , get: r => r.boundary ? r.boundary.near : undefined   , label: "境界付近"  , pct: true  },
    {key: "changed", get: r => r.boundary ? r.boundary.changed : undefined, label: "コード変化", pct: true  },
    {key: "back"   , get: r => r.boundary ? r.boundary.back : undefined   , label: "行き来"    , pct: true  },
  ];
  
  // Messages of the viewer (02b, chapter 5)
  const MSG = {
    m101: "まだ記録がありません。学習スクリプトで ts.run_start を呼ぶと、ここに実行が並びます",
    m102: n => n + " 行の記録を読めなかったため、飛ばしました",
    m103: "このフォルダに tensight の記録が見つかりません",
    m104: n => "形が合わない " + n + " 個の観測点を、比較から外しました",
    m105: "この観測点の記録を読めませんでした。学習中なら、少し待ってから再読み込みしてください",
  };
  
  // Palettes for the page and the charts (CR-004), from tensight_core.js
  const PALETTES = core.PALETTES;
  
  // Choices of the viewer; "" means the colormap of the palette, or the setting of the OS
  const ui     = {palette: "std", cmap: "", theme: "", speed: 1, closed: {}};
  let storedUi = false;
  try {
    const saved = JSON.parse(localStorage.getItem("tensight.ui") || "null");
    if (saved) {
      Object.assign(ui, saved);
      delete ui.mapScale;   // the scale of the maps that an older viewer saved; the maps are linear only (CR-022)
      storedUi = true;
    }
  } catch (e) {
    // Storage is blocked; the defaults stay
  }
  
  // The current colors, the current colormap (256 colors as r, g, b bytes), the fonts and the colors of the charts
  let colors = PALETTES[0].light;
  let lut    = null;
  let fonts  = {sans: "sans-serif", mono: "monospace"};
  let fig    = {main: PALETTES[0].light.signal, runs: PALETTES[0].light.runs};
  
  
  function saveUi() {
    //* Save the choices of the viewer in this browser.
    
    try {
      localStorage.setItem("tensight.ui", JSON.stringify(ui));
    } catch (e) {
      // Storage is blocked; the choice lasts until the page closes
    }
  }
  
  
  function palette() {
    //* Return the chosen palette.
    
    return PALETTES.find(p => p.id === ui.palette) || PALETTES[0];
  }
  
  
  function theme() {
    //* Return "light" or "dark": the chosen theme, or the theme of the OS.
    
    return ui.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }
  
  
  function cmapName() {
    //* Return the name of the colormap: the chosen one, or the one of the palette.
    
    return CMAPS.includes(ui.cmap) ? ui.cmap : palette().cmap;
  }
  
  
  function runColor(i) {
    //* Return the color of the i-th shown run in the charts (02b, 2.3).
    //! It follows the chosen colormap (CR-004).
    
    return fig.runs[i % fig.runs.length];
  }
  
  
  function applyColors() {
    //* Put the colors of the palette and the theme into the page, and make the colormap.
    
    // Choose the colors
    const t    = theme();
    const root = document.documentElement;
    colors     = palette()[t];
    lut        = makeLut(cmapName(), colors, t);
    fig        = figureColors(colors, t);
    
    // Put them into the CSS variables
    root.dataset.theme = t;
    for (const key of ["bg", "surface", "text", "weak", "line", "signal", "caution", "alert"]) root.style.setProperty("--" + key, colors[key]);
    root.style.setProperty("--figure", fig.main);
  }
  
  
  function figureColors(c, t) {
    //* Choose the colors of the charts for the chosen colormap (U-038).
    
    return core.figureColors(lut, c, t, cmapName() === palette().cmap || cmapName() === "accent");
  }
  
  
  function hexRgb(hex) {
    //* Turn "#RRGGBB" into [r, g, b] (U-038).
    
    return core.hexRgb(hex);
  }
  
  
  function makeLut(name, c, t) {
    //* Make the colormap of a name (U-038). A name without a table in colormaps.js ("accent") gives the accent ramp of the palette.
    
    const tables = window.TENSIGHT_COLORMAPS || {};
    return core.makeLut(tables[name] || null, c, t);
  }
  
  
  function cmap(x, table) {
    //* Return the color [r, g, b] for x between 0 and 1 (U-038), from the current colormap when table is not given.
    
    return core.cmap(table || lut, x);
  }
  
  
  function rgbCss(rgb) {
    //* Turn [r, g, b] into a CSS color.
    
    return "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
  }
  
  
  function alpha(hex, a) {
    //* Turn "#RRGGBB" and an opacity into a CSS color.
    
    const c = hexRgb(hex);
    return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";
  }
  
  
  function rampCss(name, pal, t) {
    //* Return a CSS gradient that shows a colormap, for the palette menu.
    
    const table = makeLut(name, pal[t], t);
    const stops = [];
    for (let i = 0; i <= 10; i++) stops.push(rgbCss(cmap(i / 10, table)) + " " + i * 10 + "%");
    return "linear-gradient(90deg," + stops.join(",") + ")";
  }
  
  
  // ===== Small helpers =====
  
  function $(id) {
    //* Return the element with this id.
    
    return document.getElementById(id);
  }
  
  
  function wait(ms) {
    //* Wait for ms milliseconds.
    
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  
  function h(tag, attrs, ...kids) {
    //* Make an element with attributes and children. Attributes whose names start with "on" become event listeners.
    //! null, undefined and false are skipped, so "cond ? h(...) : null" can be used for a child.
    
    // Set the attributes
    const el = document.createElement(tag);
    for (const [key, val] of Object.entries(attrs || {})) {
      if (val === null || val === undefined || val === false) continue;
      if (key === "class") el.className = val;
      else if (key === "style") el.style.cssText = val;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), val);
      else el.setAttribute(key, val === true ? "" : val);
    }
    
    // Add the children
    for (const kid of kids.flat(Infinity)) if (kid !== null && kid !== undefined && kid !== false) el.append(kid instanceof Node ? kid : String(kid));
    return el;
  }
  
  
  function icon(paths, size) {
    //* Make a small SVG icon from path data.
    
    const el     = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    el.setAttribute("width"      , size       );
    el.setAttribute("height"     , size       );
    el.setAttribute("viewBox"    , "0 0 16 16");
    el.setAttribute("aria-hidden", "true"     );
    el.innerHTML = paths;
    return el;
  }
  
  
  function fmt(x) {
    //* Show a value with 4 significant digits (U-037).
    
    return core.fmt(x);
  }
  
  
  function fmtPct(x) {
    //* Show a share between 0 and 1 as a percentage (U-037).
    
    return core.fmtPct(x);
  }
  
  
  function fmtInt(x) {
    //* Show a count with commas (U-037).
    
    return core.fmtInt(x);
  }
  
  
  function shortDate(iso) {
    //* Turn an ISO 8601 time into a short form like "10/27 15:30".
    
    const m = /^\d{4}-(\d\d)-(\d\d)T(\d\d):(\d\d)/.exec(iso || "");
    return m ? (+m[1]) + "/" + (+m[2]) + " " + m[3] + ":" + m[4] : "";
  }
  
  
  function nearest(list, s) {
    //* Return the step of the sorted list that is nearest to s (U-037).
    
    return core.stepNearest(list, s);
  }
  
  
  function sortedSteps(m) {
    //* Return the steps of a map of records, in order.
    
    return [...m.keys()].sort((a, b) => a - b);
  }
  
  
  function niceTicks(lo, hi, count) {
    //* Return round numbers between lo and hi, for the labels of an axis (U-037).
    
    return core.niceTicks(lo, hi, count);
  }
  
  
  function tipShow(e, text) {
    //* Show the tooltip near the mouse.
    
    const t         = $("tip");
    t.textContent   = text;
    t.style.display = "block";
    let x           = e.clientX + 14;
    let y           = e.clientY + 14;
    if (x + t.offsetWidth > innerWidth - 8) x = e.clientX - t.offsetWidth - 14;
    if (y + t.offsetHeight > innerHeight - 8) y = e.clientY - t.offsetHeight - 14;
    t.style.left    = x + "px";
    t.style.top     = y + "px";
  }
  
  
  function tipHide() {
    //* Hide the tooltip.
    
    const t = $("tip");
    if (t) t.style.display = "none";
  }
  
  
  function toast(text, kind) {
    //* Show a short message at the bottom for 9 seconds. kind "alert" is for errors.
    
    const close = h("button", {class: "x", "aria-label": "閉じる"}, "×");
    const el    = h("div", {class: "toast" + (kind === "alert" ? " alert" : ""), role: kind === "alert" ? "alert" : "status"}, h("span", null, text), close);
    close.addEventListener("click", () => el.remove());
    $("toasts").append(el);
    setTimeout(() => el.remove(), 9000);
  }
  
  
  function download(name, text, type) {
    //* Let the browser save text as a file.
    
    const url = URL.createObjectURL(new Blob([text], {type}));
    const a   = h("a", {href: url, download: name, style: "display:none"});
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  
  
  // ===== Data store =====
  
  const store  = {runs: new Map(), order: []};
  let bundle   = null;
  let badLines = 0;
  
  
  function newRun(r, src) {
    //* Make an empty run. src holds the files when the run comes from a folder that the user picked.
    
    return {folder: r.folder, name: r.name || r.folder, started: r.started || "", src: src || null, info: null, points: new Map(), byName: new Map(), alerts: new Map(), recs: new Map(), cmp: new Map(), loads: new Map(), status: new Map()};
  }
  
  
  // Functions that the data files call. Each one keeps the data in the store (02c).
  const handlers = {
    runs(o) {
      //* Add the runs listed in runs.js.
      
      for (const r of o.runs || []) {
        if (!r || !r.folder || store.runs.has(r.folder)) continue;
        store.runs.set(r.folder, newRun(r, null));
        store.order.push(r.folder);
      }
    },
    run(o) {
      //* Keep the information and the settings of a run (run.js).
      
      const run = store.runs.get(o.folder);
      if (run) run.info = o;
    },
    point(o) {
      //* Keep one point of the catalog.
      
      const run = store.runs.get(o.run);
      if (!run) return;
      run.points.set(o.id  , o);
      run.byName.set(o.name, o);
    },
    alert(o) {
      //* Keep a mark that a point has NaN or Inf (proposal for FR-022).
      
      const run  = store.runs.get(o.run);
      const list = run ? run.alerts.get(o.id) || [] : [];
      if (!run || list.some(a => a.step === o.step)) return;
      list.push(o);
      run.alerts.set(o.id, list);
    },
    rec(o) {
      //* Keep one record, by point and by step.
      
      const run = store.runs.get(o.run);
      if (!run) return;
      if (!run.recs.has(o.id)) run.recs.set(o.id, new Map());
      run.recs.get(o.id).set(o.step, o);
    },
    cmp(o) {
      //* Keep a full tensor for comparing.
      
      const run = store.runs.get(o.run);
      if (run) run.cmp.set(o.step + "/" + o.id, o);
    },
    bundle(o) {
      //* Keep the sources of the viewer for the HTML export.
      
      bundle = o;
    },
  };
  
  // The functions on window.tensight skip the lines of other format versions (02c)
  const api       = {};
  for (const name of Object.keys(handlers)) api[name] = o => { if (o && o.format === 1) handlers[name](o); };
  window.tensight = api;
  
  // Script tags load one at a time, so that an error event can be blamed on the file that is loading.
  // A file with a half-written last line cannot be parsed, so it is read again once after 300 ms (02c).
  let queue     = Promise.resolve();
  let loadCount = 0;
  
  
  function scriptOnce(src) {
    //* Load one data file with a script tag. The promise fails when the file is missing or cannot be parsed.
    
    return new Promise((resolve, reject) => {
      // Make the tag, and watch for an error while it loads
      const el      = document.createElement("script");
      let broken    = false;
      const onError = () => { broken = true; };
      const finish  = ok => {
        window.removeEventListener("error", onError);
        el.remove();
        if (ok && !broken) resolve();
        else reject(new Error(src));
      };
      
      // Start loading
      window.addEventListener("error", onError);
      el.onload  = () => finish(true);
      el.onerror = () => finish(false);
      el.src     = src + "?v=" + (++loadCount);
      document.head.appendChild(el);
    });
  }
  
  
  function loadScript(src) {
    //* Load a data file of this record folder, and try once more after 300 ms when it fails.
    
    const job = queue.then(() => scriptOnce(src)).catch(() => wait(RETRY_MS).then(() => scriptOnce(src)));
    queue     = job.catch(() => undefined);
    return job;
  }
  
  
  function flushBad() {
    //* Show MSG-102 when lines were skipped while a picked folder was read.
    
    if (!badLines) return;
    toast(MSG.m102(badLines));
    badLines = 0;
  }
  
  
  async function readData(run, rel) {
    //* Read one data file of a run: with a script tag, or from the files of a folder that the user picked.
    //! Lines of a picked folder that cannot be read are skipped and counted (IF-06).
    
    // Files of this record folder are loaded as scripts
    if (!run || !run.src) {
      await loadScript(rel);
      return;
    }
    
    // Files of a picked folder are read as text, line by line
    const file   = run.src.files.get(rel);
    if (!file) throw new Error("missing " + rel);
    const parsed = core.parseLines(await file.text());
    for (const call of parsed.calls) {
      if (handlers[call.fn]) api[call.fn](call.arg);
      else badLines++;
    }
    badLines    += parsed.bad;
  }
  
  
  function runBase(run) {
    //* Return the folder of a run inside the record folder.
    
    return "runs/" + run.folder + "/";
  }
  
  
  function once(run, key, job) {
    //* Run a load only once for each run and key. The promise tells whether it worked.
    
    if (!run.loads.has(key)) {
      run.status.set(key, "loading");
      run.loads.set(key, job().then(() => { run.status.set(key, "ok"); return true; }, () => { run.status.set(key, "error"); return false; }).then(ok => { flushBad(); return ok; }));
    }
    return run.loads.get(key);
  }
  
  
  function ensureCatalog(run) {
    //* Load run.js (when it is not loaded yet) and catalog.js of a run.
    
    return once(run, "catalog", async () => {
      if (!run.info) await readData(run, runBase(run) + "run.js").catch(() => undefined);
      await readData(run, runBase(run) + "catalog.js");
    });
  }
  
  
  function ensurePoint(run, id) {
    //* Load the records of one point.
    
    return once(run, "p" + id, () => readData(run, runBase(run) + "points/" + id + ".js"));
  }
  
  
  // ===== State =====
  
  // What is shown. The runs, the point and the step are also kept after # in the URL (FR-023).
  const state   = {
    primary : "",
    also    : [],
    point   : "",
    step    : null,
    metric  : "mean",
    hist    : "value",
    all     : false,
    view    : "observe",
    base    : "",
    target  : "",
    cstep   : null,
    sort    : {key: "", dir: 1},
    filter  : "",
    embedded: false,
    loading : true,
    playing : false,
    rot     : {az: ROT.az, el: ROT.el},
    hidden  : {values: [], maps: [], trends: []},
  };
  let token     = 0;
  let view      = {charts: []};
  let playTimer = 0;
  
  
  function writeHash() {
    //* Keep the state after # in the URL, so that reloading the page shows the same view (FR-023).
    
    // Collect the values that differ from the defaults
    if (state.embedded) return;
    const q = new URLSearchParams();
    if (state.primary)           q.set("run"   , state.primary       );
    if (state.also.length)       q.set("also"  , state.also.join(","));
    if (state.point)             q.set("point" , state.point         );
    if (state.step !== null)     q.set("step"  , String(state.step)  );
    if (state.metric !== "mean") q.set("metric", state.metric        );
    if (state.hist !== "value")  q.set("hist"  , state.hist          );
    if (state.view === "compare") {
      q.set("view", "compare");
      if (state.base)           q.set("base"  , state.base         );
      if (state.target)         q.set("target", state.target       );
      if (state.cstep !== null) q.set("cstep" , String(state.cstep));
    }
    
    // Replace the URL without adding to the history
    try {
      history.replaceState(null, "", "#" + q.toString());
    } catch (e) {
      location.hash = q.toString();
    }
  }
  
  
  function primaryRun() {
    //* Return the main run (the first of the shown runs).
    
    return store.runs.get(state.primary) || null;
  }
  
  
  function currentPoint() {
    //* Return the chosen point of the main run.
    
    const run = primaryRun();
    return run ? run.byName.get(state.point) || null : null;
  }
  
  
  function shownRuns() {
    //* Return the folders of the shown runs (the main run and the overlaid runs) in the order of the list of runs, so that a run keeps its color whichever run is the main one (FR-034, 02b 2.3).
    
    return store.order.filter(f => store.runs.has(f) && (f === state.primary || state.also.includes(f)));
  }
  
  
  function recsOf(run, p) {
    //* Return the records of the point in run that has the same name as p.
    
    const q = run && p ? run.byName.get(p.name) : null;
    return q ? run.recs.get(q.id) || null : null;
  }
  
  
  function pathOf(p) {
    //* Return the layer (module) path of a point.
    
    return p.layer || p.target;
  }
  
  
  function restOf(p) {
    //* Return the part of the target after the layer path, like "weight".
    
    const path = pathOf(p);
    return p.target === path ? "" : p.target.slice(path.length).replace(/^\./, "");
  }
  
  
  function tabLabel(p) {
    //* Return the label of the tab of a point, like "output", "weight" or "weight:grad".
    
    const rest = restOf(p);
    if (!rest) return p.kind;
    return p.kind === "value" ? rest : rest + ":" + p.kind;
  }
  
  
  function pointOrder(a, b) {
    //* Order the points of one layer: the layer itself, then the weight, then other parameters, each by kind.
    
    const ra = restOf(a);
    const rb = restOf(b);
    const ga = ra === "" ? 0 : ra === "weight" ? 1 : 2;
    const gb = rb === "" ? 0 : rb === "weight" ? 1 : 2;
    return ga - gb || (ra < rb ? -1 : ra > rb ? 1 : 0) || KINDS.indexOf(a.kind) - KINDS.indexOf(b.kind);
  }
  
  
  function modulePoints(run, path) {
    //* Return the points of one layer, in order.
    
    return [...run.points.values()].filter(p => pathOf(p) === path).sort(pointOrder);
  }
  
  
  function treeRows(run, filter) {
    //* Return the rows of the list: the layers nested by their names, each with its points (U-040, FR-018, FR-033).
    
    return core.treeRows([...run.points.values()], filter, pathOf, pointOrder);
  }
  
  
  // ===== List =====
  
  function toggleSide() {
    //* Open or close the list on narrow screens.
    
    $("app").classList.toggle("side-open");
  }
  
  
  function closeSide() {
    //* Close the list on narrow screens.
    
    $("app").classList.remove("side-open");
  }
  
  
  function renderRunButton() {
    //* Draw the button of the main run at the top of the list.
    
    const btn       = $("runbtn");
    const run       = primaryRun();
    const shown     = shownRuns();
    const sw        = shown.length > 1 ? h("span", {class: "rb-sw"}, shown.map((f, i) => h("i", {style: "background:" + runColor(i)}))) : null;
    btn.textContent = "";
    if (!run) {
      btn.append(h("span", {class: "rb-name"}, "実行なし"));
      return;
    }
    btn.append(...[h("span", {class: "rb-name"}, run.name), h("span", {class: "rb-date"}, shortDate(run.started)), sw, h("span", {class: "chev"}, icon(ICON_CHEV, 16))].filter(Boolean));
  }
  
  
  function renderTree() {
    //* Draw the list of layers, filtered by the text in the filter box (FR-033).
    //! A red dot marks a layer that has a point with NaN or Inf (FR-022).
    
    // Clear the list
    const tree       = $("tree");
    const run        = primaryRun();
    tree.textContent = "";
    if (!run || !run.points.size) return;
    
    // Add one row for each layer; rows with points can be chosen
    const rows    = treeRows(run, state.filter.trim().toLowerCase());
    const cur     = currentPoint();
    const curPath = cur ? pathOf(cur) : "";
    for (const r of rows) {
      const n    = r.node;
      const has  = n.points.length > 0;
      const sel  = has && n.path === curPath;
      const bad  = has && modulePoints(run, n.path).some(p => run.alerts.has(p.id));
      const pick = () => {
        const label = cur ? tabLabel(cur) : "";
        selectPoint((n.points.find(p => tabLabel(p) === label) || n.points[0]).name);
        closeSide();
      };
      tree.append(h("div", {class: "row" + (has ? " has" : "") + (sel ? " sel" : ""), role: "treeitem", "aria-level": r.depth + 1, "aria-selected": sel ? "true" : "false", style: "--d:" + r.depth, title: n.path, onclick: has ? pick : null},
                    h("span", {class: "rn"}, n.name), bad ? h("span", {class: "dot", "aria-label": "NaN か Inf を含む"}) : null));
    }
    
    // Show a note when nothing matches, and keep the chosen row in sight
    if (!rows.length) tree.append(h("div", {class: "tree-empty"}, "一致する観測点がありません"));
    const selEl = tree.querySelector(".sel");
    if (selEl) selEl.scrollIntoView({block: "nearest"});
  }
  
  
  function selectPoint(name) {
    //* Choose a point by its name and show it.
    
    if (name === state.point && state.view === "observe") return;
    state.point = name;
    state.all   = false;
    state.view  = "observe";
    writeHash();
    renderTree();
    loadAndRender();
  }
  
  
  // ===== Observe view (SC-01) =====
  
  async function loadAndRender() {
    //* Load what the chosen point needs, then draw the right side.
    //! A newer call stops an older one, so quick clicks do not draw old data.
    
    // Load the catalog of the main run and choose a point
    const my  = ++token;
    const run = primaryRun();
    renderMain();
    if (!run) return;
    await ensureCatalog(run);
    if (my !== token) return;
    if (!currentPoint()) {
      const maps  = run.info && run.info.settings && run.info.settings.maps;
      const names = [...run.points.values()].map(q => q.name);
      const named = core.pointFirst(maps, names);
      const first = treeRows(run, "").find(r => r.node.points.length);
      state.point = named || (first ? first.node.points[0].name : "");
      writeHash();
    }
    renderTree();
    renderMain();
    
    // Load the records of the point in the main run and in the overlaid runs
    const p = currentPoint();
    if (!p) return;
    await ensurePoint(run, p.id);
    for (const f of state.also) {
      const other = store.runs.get(f);
      if (!other) continue;
      await ensureCatalog(other);
      const q     = other.byName.get(p.name);
      if (q) await ensurePoint(other, q.id);
    }
    if (my !== token) return;
    fixStep();
    renderMain();
  }
  
  
  function fixStep() {
    //* Keep the chosen step on a step of the step bar: the nearest one, or the latest step of the main run when no step is chosen yet (U-047, CR-017).
    
    const m     = recsOf(primaryRun(), currentPoint());
    const steps = runSteps(currentPoint());
    if (!m || !m.size || (state.step !== null && steps.includes(state.step))) return;
    state.step  = state.step === null ? sortedSteps(m).pop() : nearest(steps, state.step);
    writeHash();
  }
  
  
  function runSteps(p) {
    //* Return the steps that the shown runs have recorded for the point, from small to large: the steps where the step bar stops (CR-017).
    
    const all = new Set();
    for (const f of shownRuns()) for (const s of (p ? recsOf(store.runs.get(f), p) || new Map() : new Map()).keys()) all.add(s);
    return [...all].sort((a, b) => a - b);
  }
  
  
  function setView(v) {
    //* Switch between the observe view and the compare view.
    
    state.view = v;
    writeHash();
    if (v === "observe") loadAndRender();
    else renderMain();
  }
  
  
  function renderMain() {
    //* Draw the right side again from the state: the observe view (SC-01) or the compare view (SC-02).
    
    // Clear the old view
    const main       = $("content");
    main.textContent = "";
    view             = {charts: []};
    tipHide();
    if (state.view === "compare") {
      renderCompare(main);
      return;
    }
    if (!store.order.length) {
      main.append(h("div", {class: "empty"}, state.loading ? "" : MSG.m101));
      return;
    }
    
    // Show the name of the layer, and the type, the shape and the device of its tensor: the record of the main run at the chosen step or the nearest earlier one (BUG-019, CR-017)
    const run   = primaryRun();
    const p     = currentPoint();
    const m     = run && p ? run.recs.get(p.id) : null;
    fixStep();
    const menu  = h("button", {class: "btn icon menu", "aria-label": "一覧を開く", onclick: toggleSide}, icon(ICON_MENU, 18));
    const title = h("h1", {class: "title mono"}, p ? pathOf(p) : run ? run.name : "");
    const parts = r => [h("span", null, r.dtype), h("span", null, r.shape.length ? r.shape.join("×") : "スカラー"), h("span", null, r.device)];
    const meta  = m && m.size ? h("span", {class: "meta"}) : null;
    view.meta   = () => { const at = p ? recAt(state.primary, p) : null; if (meta) meta.replaceChildren(...(at ? parts(at.rec) : [])); };
    view.meta();
    main.append(h("header", {class: "head"}, h("div", {class: "title-row"}, menu, title, meta, renderActions(true))));
    if (!run || !p) {
      if (run && run.status.get("catalog") === "error") main.append(h("p", {class: "note error"}, MSG.m105));
      return;
    }
    if (!m || !m.size) {
      const failed = run.status.get("p" + p.id) === "error";
      main.append(renderTabs(run, p), h("p", {class: "note" + (failed ? " error" : "")}, failed ? MSG.m105 : "読み込み中"));
      return;
    }
    
    // The step bar stops at the steps that any shown run has recorded (CR-017); map steps to x the same way in the step bar and in every step chart, so that one step lines up across all of them
    const steps = runSteps(p);
    const s0    = steps[0];
    const s1    = steps[steps.length - 1];
    const axis  = {
      s0   : s0,
      s1   : s1,
      ticks: s1 > s0 ? niceTicks(s0, s1, Math.min(6, s1 - s0)).filter(t => t >= s0 && t <= s1) : [s0],
      frac : step => s1 === s0 ? 0.5 : (step - s0) / (s1 - s0),
      x    : (step, w) => core.stepX(step, s0, s1, w),
      step : (x, w) => s0 + (x - GUT_L) / Math.max(1, w - GUT_L - GUT_R) * (s1 - s0),
    };
    Object.assign(view, {run, point: p, recs: m, steps, axis});
    
    // Make the step bar: a tick for each step (red when a record of a shown run at that step has NaN or Inf) and labels at round steps
    const numEl  = h("div", {class: "step-num"});
    const thumb  = h("span", {class: "thumb"});
    const bad    = s => shownRuns().some(f => { const r = (recsOf(store.runs.get(f), p) || new Map()).get(s); return r && (r.nan > 0 || r.inf > 0); });
    const ticks  = steps.map(s => h("span", {class: "tick" + (bad(s) ? " bad" : ""), style: "left:" + axis.frac(s) * 100 + "%"}));
    const track  = h("div", {class: "track", role: "slider", tabindex: "0", "aria-label": "ステップ", "aria-valuemin": steps[0], "aria-valuemax": steps[steps.length - 1]}, ticks, thumb);
    const labels = h("div", {class: "step-labels", "aria-hidden": "true"}, axis.ticks.map(t => h("span", {style: "left:" + axis.frac(t) * 100 + "%"}, fmtInt(t))));
    const fromX  = x => {
      const rc = track.getBoundingClientRect();
      setStep(nearest(steps, s0 + (x - rc.left) / Math.max(1, rc.width) * (s1 - s0)));
    };
    view.slider  = () => {
      thumb.style.left  = axis.frac(state.step) * 100 + "%";
      numEl.textContent = fmtInt(state.step);
      track.setAttribute("aria-valuenow", state.step);
    };
    
    // Move the step by dragging on the bar, or with Home and End
    track.addEventListener("pointerdown", e => {
      e.preventDefault();
      stopPlay();
      track.focus();
      track.setPointerCapture(e.pointerId);
      fromX(e.clientX);
      const move = ev => fromX(ev.clientX);
      const up   = () => {
        track.removeEventListener("pointermove"  , move);
        track.removeEventListener("pointerup"    , up  );
        track.removeEventListener("pointercancel", up  );
      };
      track.addEventListener("pointermove"  , move);
      track.addEventListener("pointerup"    , up  );
      track.addEventListener("pointercancel", up  );
    });
    track.addEventListener("keydown", e => {
      if (e.key !== "Home" && e.key !== "End") return;
      e.preventDefault();
      setStep(e.key === "Home" ? steps[0] : steps[steps.length - 1]);
    });
    view.slider();
    
    // Make the player: one button plays and stops, and another chooses the speed (CR-005 proposal)
    const playBtn = h("button", {class: "btn icon play", onclick: togglePlay});
    const speed   = h("button", {class: "btn speed", "aria-haspopup": "menu", "aria-expanded": "false", title: "再生速度", onclick: e => openSpeed(e.currentTarget)});
    view.player   = () => {
      playBtn.replaceChildren(icon(state.playing ? ICON_PAUSE : ICON_PLAY, 16));
      playBtn.setAttribute("aria-label", state.playing ? "停止" : "再生");
      playBtn.setAttribute("title", (state.playing ? "停止" : "再生") + "（スペース）");
      playBtn.classList.toggle("on", state.playing);
      speed.textContent = speedText(ui.speed);
    };
    view.player();
    
    // Put the step bar, then the groups (FR-041): the values, the maps when a shown run has them, and the trends
    main.append(renderTabs(run, p), h("div", {class: "stepbar"}, numEl, h("div", null, track, labels), h("div", {class: "player"}, playBtn, speed)));
    const hasMap = shownRuns().some(f => [...(recsOf(store.runs.get(f), p) || new Map()).values()].some(r => r.map));
    const groups = [["values", "値と分布", () => renderValues(p)], ["maps", "値の大きさ", () => renderMaps(p)], ["trends", "推移", () => renderTrends(p, axis)]];
    view.groups  = {};
    view.makers  = {};
    view.map3d   = [];
    for (const [key, title, body] of groups) {
      if (key === "maps" && !hasMap) continue;
      view.makers[key] = () => renderGroup(key, title, p, body);
      view.groups[key] = view.makers[key]();
      main.append(view.groups[key]);
    }
    drawAll();
  }
  
  
  function renderActions(observe) {
    //* Make the buttons at the top right. observe is false in the compare view.
    
    const flip = () => {
      ui.theme = theme() === "dark" ? "light" : "dark";
      saveUi();
      recolor();
    };
    const head = state.embedded ? null : observe ? h("button", {class: "btn", onclick: () => setView("compare")}, "比べる") : h("button", {class: "btn", onclick: () => setView("observe")}, "閉じる");
    const out  = state.embedded || !observe ? null : h("button", {class: "btn", "aria-haspopup": "menu", "aria-expanded": "false", onclick: e => openExport(e.currentTarget)}, "書き出す");
    const pal  = h("button", {class: "btn", "aria-haspopup": "dialog", "aria-expanded": "false", onclick: e => openPalette(e.currentTarget)}, "配色");
    const mode = h("button", {class: "btn icon", "aria-label": "明るさを切り替える", title: "明るさ", onclick: flip}, icon(ICON_THEME, 16));
    return h("div", {class: "actions"}, head, out, pal, mode);
  }
  
  
  function renderTabs(run, p) {
    //* Make the tabs of the points of one layer, like output, weight and weight:grad.
    
    const tabs = h("div", {class: "tabs", role: "tablist", "aria-label": "観測点"});
    for (const q of modulePoints(run, pathOf(p))) {
      const sel = q.name === p.name;
      const dot = run.alerts.has(q.id) ? h("span", {class: "dot", "aria-label": "NaN か Inf を含む"}) : null;
      tabs.append(h("button", {class: "tab mono" + (sel ? " sel" : ""), role: "tab", "aria-selected": sel ? "true" : "false", title: q.name, onclick: () => selectPoint(q.name)}, tabLabel(q), dot));
    }
    return tabs;
  }
  
  
  function setStep(s) {
    //* Choose a step: move the step bar, show the type, the shape and the values of the step and draw the charts again (BUG-019).
    
    if (s === state.step || !view.steps || !view.steps.includes(s)) return;
    state.step = s;
    writeHash();
    view.slider();
    view.meta();
    redrawGroup("values");
    redrawGroup("maps");
    drawAll();
  }
  
  
  function speedText(v) {
    //* Show a playing speed, like "1.0×" or "0.75×".
    
    return (v % 1 === 0 ? v.toFixed(1) : String(v)) + "×";
  }
  
  
  function stopPlay() {
    //* Stop playing the steps, and show the play button again.
    
    state.playing = false;
    clearTimeout(playTimer);
    if (view.player) view.player();
  }
  
  
  function togglePlay() {
    //* Start or stop playing the steps one after another (CR-005 proposal).
    //! Playing from the last step starts again from the first step.
    
    if (state.playing) {
      stopPlay();
      return;
    }
    if (!view.steps || view.steps.length < 2) return;
    if (view.steps.indexOf(state.step) >= view.steps.length - 1) setStep(view.steps[0]);
    state.playing = true;
    view.player();
    playNext();
  }
  
  
  function playNext() {
    //* Move to the next recorded step after a short wait, while the steps are playing.
    //! The wait is PLAY_MS at speed 1.0, divided by the speed. Playing stops at the last step.
    
    clearTimeout(playTimer);
    playTimer = setTimeout(() => {
      const i = view.steps ? view.steps.indexOf(state.step) : -1;
      if (!state.playing || i < 0 || i >= view.steps.length - 1) {
        stopPlay();
        return;
      }
      setStep(view.steps[i + 1]);
      if (i + 1 >= view.steps.length - 1) stopPlay();
      else playNext();
    }, PLAY_MS / ui.speed);
  }
  
  
  function openSpeed(anchor) {
    //* Open the menu of the playing speeds (CR-005 proposal).
    
    let pop     = null;
    const pick  = v => {
      ui.speed = v;
      saveUi();
      if (pop) pop.close();
      if (view.player) view.player();
      if (state.playing) playNext();
    };
    const items = SPEEDS.map(v => h("button", {class: "menuitem" + (v === ui.speed ? " sel" : ""), role: "menuitemradio", "aria-checked": v === ui.speed ? "true" : "false", onclick: () => pick(v)}, speedText(v)));
    pop         = popover(anchor, h("div", {role: "menu"}, items), "right");
    if (pop) pop.el.style.minWidth = "96px";
  }
  
  
  function kv(label, content, cls) {
    //* Make one label and value of the values table.
    
    const none = content === "なし";
    return h("div", {class: "kv"}, h("span", {class: "k"}, label), h("span", {class: "v" + (none ? " none" : "") + (cls ? " " + cls : "")}, content));
  }
  
  
  function rangeText(o) {
    //* Show a summary of scales or zero points: one value, or "min〜max".
    
    if (!o) return "なし";
    const show = v => Number.isInteger(v) ? fmtInt(v) : fmt(v);
    return o.count === 1 || o.min === o.max ? show(o.min) : show(o.min) + "〜" + show(o.max);
  }
  
  
  function renderNow(rec) {
    //* Make the part for the chosen step: the values (FR-019) and the distribution or the code usage (FR-020).
    
    // Statistics (FR-008, FR-028)
    const box = h("div", {class: "stats"});
    box.append(h("div", {class: "grid"}, kv("最小", fmt(rec.min)), kv("最大", fmt(rec.max)), kv("平均", fmt(rec.mean)), kv("標準偏差", fmt(rec.std)), kv("L2", fmt(rec.l2)), kv("0の割合", fmtPct(rec.zero)),
                 kv("NaN", fmtInt(rec.nan), rec.nan > 0 ? "bad" : ""), kv("Inf", fmtInt(rec.inf), rec.inf > 0 ? "bad" : ""), kv("要素数", fmtInt(rec.numel)), "ratio" in rec ? kv("更新率", fmt(rec.ratio)) : null));
    
    // Quantization info (FR-012 to FR-015) and the moves around the boundaries (FR-031)
    const q = rec.quant;
    const b = rec.boundary;
    if (q) {
      const sqnr = q.sqnr === "inf" ? "∞" : typeof q.sqnr === "number" ? [fmt(q.sqnr), h("span", {class: "u"}, "dB")] : "なし";
      box.append(h("div", {class: "grid"}, kv("形式", q.format), kv("粒度", GRAN[q.granularity] || q.granularity), kv("スケール", rangeText(q.scale), "wrap"), kv("ゼロ点", rangeText(q.zero_point), "wrap"),
                   kv("コード", q.codes ? fmtInt(q.codes.used) + " / " + fmtInt(q.codes.total) : "なし"), kv("クリップ率", fmtPct(q.clip)), kv("SQNR", sqnr), kv("平均二乗誤差", fmt(q.mse)), kv("最大絶対誤差", fmt(q.max_abs))));
    }
    if (b) box.append(h("div", {class: "grid"}, kv("境界付近", fmtPct(b.near)), kv("コード変化", fmtPct(b.changed)), kv("行き来", fmtPct(b.back))));
    
    // The first values; NaN and Inf are red (FR-010)
    box.append(h("div", {class: "sample"}, h("span", {class: "k"}, "値の一部"), h("span", {class: "vals"}, (rec.sample || []).map(v => h("span", {class: typeof v === "string" ? "bad" : null}, fmt(v))))));
    
    // The histogram can switch to the code usage when the tensor is quantized
    const codes = q && q.codes && Array.isArray(q.codes.counts) ? q.codes : null;
    const mode  = codes && state.hist === "code" ? "code" : "value";
    const cv    = h("canvas", {role: "img", "aria-label": mode === "code" ? "コードの使われ方" : "値の分布"});
    const swap  = v => {
      state.hist = v;
      writeHash();
      redrawGroup("values");
      drawAll();
    };
    addChart(cv, () => drawHist(cv, rec, mode));
    return h("section", {class: "now", "aria-label": "このステップの値"}, box, h("div", {class: "hist"}, h("div", {class: "sec-head"}, codes ? seg([["value", "値"], ["code", "コード"]], mode, swap, "分布の表し方") : h("span", {class: "sec-title"}, "分布")), cv));
  }
  
  
  function seg(options, current, onPick, label) {
    //* Make a row of toggle buttons. The button for current is pressed.
    
    return h("div", {class: "seg", role: "group", "aria-label": label || null}, options.map(([key, text]) => h("button", {"aria-pressed": key === current ? "true" : "false", onclick: () => { if (key !== current) onPick(key); }}, text)));
  }
  
  
  function renderTrend(p, axis) {
    //* Make the trend part: buttons to choose a value, the legend of the runs and the chart (FR-021, FR-034).
    
    // Offer only the values that a shown run has
    const shown = groupRuns("trends", p);
    const lists = shown.map(f => [...(recsOf(store.runs.get(f), p) || new Map()).values()]);
    const avail = METRICS.filter(mt => lists.some(list => list.some(r => typeof mt.get(r) === "number")));
    if (!avail.some(mt => mt.key === state.metric)) state.metric = (avail.find(mt => mt.key === "mean") || avail[0] || METRICS[2]).key;
    
    // Make the part; choosing another value draws only this part again
    const cv     = h("canvas", {role: "img", "aria-label": "ステップごとの変化"});
    const sec    = h("section", {class: "sec"});
    const legend = shown.length > 1 ? h("span", {class: "legend"}, shown.map(f => h("span", null, h("i", {style: "background:" + runColor(shownRuns().indexOf(f))}), store.runs.get(f).name))) : null;
    const pick   = key => {
      state.metric = key;
      writeHash();
      const fresh  = renderTrend(p, axis);
      sec.replaceWith(fresh);
      drawAll();
    };
    sec.append(h("div", {class: "sec-head"}, seg(avail.map(mt => [mt.key, mt.label]), state.metric, pick, "表示する値"), legend), cv);
    addChart(cv, () => drawTrend(cv, p, axis));
    return sec;
  }
  
  
  function renderStrips(runs, axis) {
    //* Make the part with one strip for each followed element (FR-032); several runs share each strip (FR-041).
    
    const keys  = [...new Set(runs.flatMap(r => r.steps.flatMap(s => Object.keys(r.m.get(s).elements || {}))))].sort((a, b) => a - b);
    const cv    = h("canvas", {role: "img", "aria-label": "要素の動き"});
    const sec   = h("section", {class: "sec"});
    const flip  = () => {
      state.all   = !state.all;
      const fresh = renderStrips(runs, axis);
      sec.replaceWith(fresh);
      drawAll();
    };
    const more  = keys.length > STRIPS ? h("button", {class: "more", onclick: flip}, state.all ? "最初の " + STRIPS + " 個に戻す" : "すべて表示（" + keys.length + "）") : null;
    const shown = state.all ? keys : keys.slice(0, STRIPS);
    sec.append(h("div", {class: "sec-head"}, h("span", {class: "sec-title"}, "要素の動き"), more), cv);
    addChart(cv, () => runs.length === 1 ? drawStrips(cv, store.runs.get(runs[0].f), runs[0].steps, axis, runs[0].m, shown) : drawStripsMany(cv, runs, axis, shown));
    return sec;
  }
  
  
  // ===== Groups of the observe view (CR-011) =====
  
  function groupRuns(key, p) {
    //* Return the runs that a group shows for the point: the shown runs that have the point and that the group does not hide, in the order of the list of runs (FR-041).
    //! When the group hides every shown run that has the point, it shows them all (groupShow, BUG-016).
    
    return core.groupShow(shownRuns().filter(f => recsOf(store.runs.get(f), p)), state.hidden[key]);
  }
  
  
  function recAt(f, p) {
    //* Return the record of the point in run f at the chosen step, or at the nearest earlier step, as {step, rec}; null when there is none (FR-041).
    
    const m = recsOf(store.runs.get(f), p);
    const s = m ? core.stepAtOrBefore(sortedSteps(m), state.step) : null;
    return s === null ? null : {step: s, rec: m.get(s)};
  }
  
  
  function runMark(f, step) {
    //* Make the name of a run with its color, and the step of its record when that is not the chosen step (FR-041).
    
    const late = typeof step === "number" && step !== state.step;
    return h("span", {class: "run-mark"}, h("i", {style: "background:" + runColor(shownRuns().indexOf(f))}), store.runs.get(f).name, late ? h("span", {class: "run-step"}, "ステップ " + fmtInt(step)) : null);
  }
  
  
  function redrawGroup(key) {
    //* Make one group of the observe view again, after the step, a chip or the fold button changed (FR-041).
    //! The caller draws the charts with drawAll().
    
    const old        = view.groups && view.groups[key];
    if (!old || !old.isConnected) return;
    const fresh      = view.makers[key]();
    old.replaceWith(fresh);
    view.groups[key] = fresh;
  }
  
  
  function renderGroup(key, title, p, body) {
    //* Make one group of the observe view: the heading with the name, the chips of the shown runs and the fold button, and the body unless it is folded (FR-041, 02b 2.5).
    
    // The chips: one for each shown run when two or more runs are shown; a run without the point is pale
    const shown  = shownRuns();
    const have   = shown.filter(f => recsOf(store.runs.get(f), p));
    const on     = groupRuns(key, p);
    const closed = !!ui.closed[key];
    const chips  = shown.length < 2 ? [] : shown.map(f => {
      const has   = have.includes(f);
      const lit   = on.includes(f);
      const only  = lit && on.length === 1;
      const tip   = !has ? "この観測点の記録がありません" : only ? "このまとまりに出している実行が1つだけなので消せません" : lit ? "このまとまりで消す" : "このまとまりに出す";
      const click = () => {
        if (!has || only) return;
        state.hidden[key] = core.groupToggle(have, have.filter(g => !on.includes(g)), f);
        redrawGroup(key);
        drawAll();
      };
      return h("button", {class: "chip" + (lit ? " on" : ""), style: "--c:" + runColor(shown.indexOf(f)), "aria-pressed": String(lit), "aria-disabled": !has || only ? "true" : null, title: tip, onclick: click}, h("i"), store.runs.get(f).name);
    });
    
    // The fold button; this browser keeps the folded groups (U-046)
    const fold   = () => {
      ui.closed[key] = !closed;
      saveUi();
      redrawGroup(key);
      drawAll();
    };
    const button = h("button", {class: "btn icon fold" + (closed ? " closed" : ""), "aria-expanded": String(!closed), "aria-label": title + (closed ? "を開く" : "をたたむ"), title: closed ? "開く" : "たたむ", onclick: fold}, icon(ICON_CHEV, 16));
    const head   = h("div", {class: "group-head"}, h("h2", {class: "group-title"}, title), h("span", {class: "chips"}, chips), button);
    return h("section", {class: "group", "data-group": key}, head, closed ? null : body());
  }
  
  
  function renderValues(p) {
    //* Make the body of the values group: the values and the distribution of one run, or the table and the overlaid distributions of several runs (FR-019, FR-020, FR-041).
    
    const runs = groupRuns("values", p).map(f => ({f, at: recAt(f, p)}));
    if (runs.length === 1 && runs[0].at && runs[0].at.step === state.step) return renderNow(runs[0].at.rec);
    if (!runs.some(r => r.at)) return h("p", {class: "note"}, "このステップより前の記録がありません");
    return renderNowMany(runs);
  }
  
  
  function renderNowMany(runs) {
    //* Make the values of several runs side by side, one column for each run and the difference for two runs, and their distributions overlaid (FR-041, 02b 2.5).
    
    // The rows: the statistics, then the update ratio, the quantization and the boundaries when a run has them
    const recs    = runs.map(r => r.at ? r.at.rec : null);
    const any     = get => recs.some(rec => rec && get(rec) !== undefined && get(rec) !== null);
    const qv      = (rec, get) => rec.quant ? get(rec.quant) : null;
    const bv      = (rec, get) => rec.boundary ? get(rec.boundary) : null;
    const rows    = [["最小", r => r.min, "num"], ["最大", r => r.max, "num"], ["平均", r => r.mean, "num"], ["標準偏差", r => r.std, "num"], ["L2", r => r.l2, "num"], ["0の割合", r => r.zero, "pct"],
                     ["NaN", r => r.nan, "int"], ["Inf", r => r.inf, "int"], ["要素数", r => r.numel, "int"]];
    if (any(r => r.ratio)) rows.push(["更新率", r => r.ratio, "num"]);
    if (any(r => r.quant)) {
      rows.push(["形式", r => qv(r, q => q.format), "text"], ["粒度", r => qv(r, q => GRAN[q.granularity] || q.granularity), "text"], ["スケール", r => qv(r, q => rangeText(q.scale)), "text"]);
      rows.push(["ゼロ点", r => qv(r, q => rangeText(q.zero_point)), "text"], ["コード", r => qv(r, q => q.codes ? fmtInt(q.codes.used) + " / " + fmtInt(q.codes.total) : null), "text"]);
      rows.push(["クリップ率", r => qv(r, q => q.clip), "pct"], ["SQNR", r => qv(r, q => q.sqnr === "inf" ? Infinity : q.sqnr), "db"], ["平均二乗誤差", r => qv(r, q => q.mse), "num"], ["最大絶対誤差", r => qv(r, q => q.max_abs), "num"]);
    }
    if (any(r => r.boundary)) rows.push(["境界付近", r => bv(r, b => b.near), "pct"], ["コード変化", r => bv(r, b => b.changed), "pct"], ["行き来", r => bv(r, b => b.back), "pct"]);
    
    // The cells, and the difference of the second run from the first for numbers (02b 2.5)
    const show    = (v, kind) => v === null || v === undefined ? "なし" : kind === "pct" ? fmtPct(v) : kind === "int" ? fmtInt(v) : kind === "db" ? (v === Infinity ? "∞" : fmt(v) + " dB") : kind === "text" ? String(v) : fmt(v);
    const two     = runs.length === 2;
    const diff    = (get, kind) => {
      const d = kind === "text" || !recs[0] || !recs[1] ? null : core.diffValue(get(recs[0]), get(recs[1]));
      return d === null ? "" : (d > 0 ? "+" : "") + show(d, kind);
    };
    const cell    = (rec, get, kind, label) => {
      const v   = rec ? get(rec) : null;
      const bad = (label === "NaN" || label === "Inf") && v > 0;
      return h("td", {class: (rec ? "" : "none") + (bad ? " bad" : "") || null}, rec ? show(v, kind) : "記録なし");
    };
    const head    = h("tr", null, h("th", null), runs.map(r => h("th", {scope: "col"}, runMark(r.f, r.at ? r.at.step : null))), two ? h("th", {scope: "col", class: "diff"}, "差") : null);
    const lines   = rows.map(([label, get, kind]) => h("tr", null, h("th", {scope: "row"}, label), recs.map(rec => cell(rec, get, kind, label)), two ? h("td", {class: "diff"}, diff(get, kind)) : null));
    const table   = h("table", {class: "vtable"}, h("thead", null, head), h("tbody", null, lines));
    const values  = r => r.at ? (r.at.rec.sample || []).map(v => h("span", {class: typeof v === "string" ? "bad" : null}, fmt(v))) : "記録なし";
    const samples = h("div", {class: "samples"}, runs.map(r => h("div", {class: "sample"}, runMark(r.f), h("span", {class: "vals"}, values(r)))));
    
    // The distributions overlaid; the codes when every run has them
    const items   = runs.filter(r => r.at).map(r => ({rec: r.at.rec, color: runColor(shownRuns().indexOf(r.f)), name: store.runs.get(r.f).name}));
    const codes   = items.every(it => it.rec.quant && it.rec.quant.codes && Array.isArray(it.rec.quant.codes.counts));
    const mode    = codes && state.hist === "code" ? "code" : "value";
    const cv      = h("canvas", {role: "img", "aria-label": mode === "code" ? "コードの使われ方" : "値の分布"});
    const swap    = v => {
      state.hist = v;
      writeHash();
      redrawGroup("values");
      drawAll();
    };
    addChart(cv, () => drawHistMany(cv, items, mode));
    const pick    = codes ? seg([["value", "値"], ["code", "コード"]], mode, swap, "分布の表し方") : h("span", {class: "sec-title"}, "分布");
    return h("section", {class: "now many", "aria-label": "このステップの値"}, h("div", {class: "stats"}, table, samples), h("div", {class: "hist"}, h("div", {class: "sec-head"}, pick), cv));
  }
  
  
  function renderMaps(p) {
    //* Make the body of the maps group: the 3D view and the top view of each run, on one scale for colors and heights (FR-036, FR-041, 02b 2.5).
    
    // One scale for all runs: the largest max of their maps; the 3D views of the group made before are dropped (BUG-017)
    const runs  = groupRuns("maps", p).map(f => ({f, at: recAt(f, p)}));
    view.map3d  = [];
    const smax  = Math.max(0, ...runs.map(r => r.at && r.at.rec.map ? r.at.rec.map.max : 0));
    const spin  = () => {
      view.map3d = view.map3d.filter(o => o.cv.isConnected);
      for (const o of view.map3d) o.draw();
    };
    const rows  = runs.map(r => {
      const map  = r.at && r.at.rec.map ? r.at.rec.map : null;
      const head = runs.length > 1 || (r.at && r.at.step !== state.step) ? h("div", {class: "map-run"}, runMark(r.f, r.at ? r.at.step : null)) : null;
      if (!map) return h("div", {class: "map-row"}, head, h("p", {class: "note"}, "このステップより前の地図の記録がありません"));
      const cv3  = h("canvas", {class: "cv3d", role: "img", "aria-label": "値の大きさの3次元の図。ドラッグで回し、ダブルクリックで元に戻します"});
      const cv2  = h("canvas", {role: "img", "aria-label": "値の大きさを上から見た図"});
      const draw = () => drawMap3d(cv3, map, smax);
      addChart(cv3, draw);
      addChart(cv2, () => drawMapFlat(cv2, map, smax));
      view.map3d.push({cv: cv3, draw});
      
      // Turn every 3D view together by dragging one; a double click puts them back
      let last   = null;
      let frame  = 0;
      cv3.addEventListener("pointerdown", e => {
        cv3.setPointerCapture(e.pointerId);
        last     = [e.clientX, e.clientY];
        dragging = true;
        tipHide();
      });
      cv3.addEventListener("pointermove", e => {
        if (!last) return;
        state.rot.az -= (e.clientX - last[0]) * 0.5;
        state.rot.el  = Math.max(4, Math.min(88, state.rot.el + (e.clientY - last[1]) * 0.4));
        last          = [e.clientX, e.clientY];
        if (!frame) frame = requestAnimationFrame(() => { frame = 0; spin(); });
      });
      const end  = () => {
        last     = null;
        dragging = false;
      };
      cv3.addEventListener("pointerup"    , end);
      cv3.addEventListener("pointercancel", end);
      cv3.addEventListener("dblclick", () => {
        state.rot = {az: ROT.az, el: ROT.el};
        spin();
      });
      return h("div", {class: "map-row"}, head, h("div", {class: "maps"}, h("div", null, cv3), h("div", null, cv2)));
    });
    return h("div", {class: "map-rows"}, rows);
  }
  
  
  function renderTrends(p, axis) {
    //* Make the body of the trends group: the trend, the distribution over steps, and the element strips of parameters (FR-021, FR-030, FR-032, FR-041).
    
    // The runs of the group with their records and colors
    const runs  = groupRuns("trends", p).map(f => {
      const m = recsOf(store.runs.get(f), p);
      return {f, m, steps: sortedSteps(m), color: runColor(shownRuns().indexOf(f))};
    });
    const parts = [renderTrend(p, axis)];
    
    // The distribution over steps: one band for each run, on one range of values and one scale of counts
    let lo      = Infinity;
    let hi      = -Infinity;
    let cmax    = 1;
    for (const r of runs) {
      for (const s of r.steps) {
        const hs = r.m.get(s).hist;
        if (!hs || !hs.counts || !hs.counts.length || typeof hs.lo !== "number") continue;
        
        // Widen the range of values and the scale of counts
        lo   = Math.min(lo, hs.lo);
        hi   = Math.max(hi, hs.hi);
        cmax = Math.max(cmax, ...hs.counts);
      }
    }
    const scale = runs.length > 1 && isFinite(lo) ? {lo, hi, cmax} : null;
    const bands = runs.map(r => {
      const cv = h("canvas", {role: "img", "aria-label": "分布の推移"});
      addChart(cv, () => drawDist(cv, r.m, r.steps, axis, scale));
      return runs.length > 1 ? h("div", {class: "band"}, h("div", {class: "band-run"}, runMark(r.f)), cv) : cv;
    });
    parts.push(h("section", {class: "sec"}, h("div", {class: "sec-head"}, h("span", {class: "sec-title"}, "分布の推移")), bands));
    
    // The element strips, when a run follows elements
    if (runs.some(r => r.steps.some(s => r.m.get(s).elements))) parts.push(renderStrips(runs, axis));
    return h("div", {class: "trends"}, parts);
  }
  
  
  // ===== Drawing =====
  
  let dragging = false;
  
  
  function addChart(cv, draw) {
    //* Keep a chart, so that it is drawn again when the step, the size or the colors change.
    
    view.charts.push({cv, draw});
  }
  
  
  function drawAll() {
    //* Draw every chart that is still on the page.
    
    view.charts = view.charts.filter(c => c.cv.isConnected);
    for (const c of view.charts) c.draw();
  }
  
  
  function prep(cv, height) {
    //* Size a canvas for the screen (with the pixel ratio), clear it and return the context and the width.
    
    const dpr        = window.devicePixelRatio || 1;
    const w          = Math.max(10, Math.round(cv.clientWidth || (cv.parentElement && cv.parentElement.clientWidth) || 300));
    const bw         = Math.round(w * dpr);
    const bh         = Math.round(height * dpr);
    cv.style.height  = height + "px";
    if (cv.width !== bw || cv.height !== bh) {
      cv.width  = bw;
      cv.height = bh;
    }
    const ctx        = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, height);
    ctx.font         = "12px " + fonts.sans;
    ctx.textBaseline = "middle";
    ctx.lineJoin     = "round";
    return {ctx, w};
  }
  
  
  function field(ctx, x, y, w, hgt) {
    //* Fill the plot area with the surface color.
    
    ctx.fillStyle = colors.surface;
    ctx.fillRect(x, y, w, hgt);
  }
  
  
  function centerNote(ctx, x, y, text) {
    //* Write a short note in the middle of a chart.
    
    ctx.fillStyle    = colors.weak;
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
  }
  
  
  function stepLine(ctx, x, top, bottom) {
    //* Draw the line of the chosen step.
    
    ctx.save();
    ctx.strokeStyle = colors.text;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, top   );
    ctx.lineTo(Math.round(x) + 0.5, bottom);
    ctx.stroke();
    ctx.restore();
  }
  
  
  function stepGrid(ctx, axis, w, top, bottom) {
    //* Draw thin vertical lines at the round steps that the step bar labels.
    
    ctx.save();
    ctx.strokeStyle = colors.line;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    for (const t of axis.ticks) {
      const x = Math.round(axis.x(t, w)) + 0.5;
      ctx.moveTo(x, top   );
      ctx.lineTo(x, bottom);
    }
    ctx.stroke();
    ctx.restore();
  }
  
  
  function colorBar(ctx, x, top, bottom, colorAt, ticks) {
    //* Draw a vertical color bar with labels on its right, so that a color can be read as a value.
    //! ticks are {frac, text}. frac 0 is the bottom of the bar and 1 is its top; colorAt(frac) gives the color there.
    
    // Paint the bar one pixel row at a time, from the top (frac 1) down
    const hgt = Math.max(1, bottom - top);
    for (let y = 0; y < hgt; y++) {
      ctx.fillStyle = rgbCss(colorAt(1 - y / Math.max(1, hgt - 1)));
      ctx.fillRect(x, top + y, 10, 1);
    }
    
    // Write the labels; a label too close to the one before it is skipped
    let last         = Infinity;
    ctx.save();
    ctx.fillStyle    = colors.weak;
    ctx.textAlign    = "left";
    ctx.textBaseline = "middle";
    for (const tick of ticks) {
      const y = Math.max(top + 6, Math.min(bottom - 6, bottom - tick.frac * hgt));
      if (Math.abs(y - last) < 14) continue;
      ctx.fillText(tick.text, x + 16, y);
      last    = y;
    }
    ctx.restore();
  }
  
  
  function stepLabels(ctx, axis, w, y) {
    //* Write the round steps under a step chart, at the same places as on the step bar.
    
    ctx.save();
    ctx.fillStyle    = colors.weak;
    ctx.textAlign    = "center";
    ctx.textBaseline = "top";
    for (const s of axis.ticks) {
      const text = fmtInt(s);
      const half = ctx.measureText(text).width / 2;
      ctx.fillText(text, Math.max(GUT_L + half, Math.min(w - GUT_R - half, axis.x(s, w))), y);
    }
    ctx.restore();
  }
  
  
  function drawHist(cv, rec, mode) {
    //* Draw the distribution of the values, or the number of values for each code (FR-020).
    //! In code mode, codes that no value uses get an orange mark under the axis (FR-014).
    
    // Choose the counts to draw
    const H         = 150;
    const {ctx, w}  = prep(cv, H);
    const top       = 4;
    const bottom    = H - 28;
    const codes     = mode === "code" ? rec.quant.codes : null;
    const counts    = codes ? codes.counts : (rec.hist && rec.hist.counts) || [];
    const lo        = codes ? codes.offset : rec.hist ? rec.hist.lo : null;
    const hi        = codes ? codes.offset + counts.length - 1 : rec.hist ? rec.hist.hi : null;
    field(ctx, 0, top, w, bottom - top);
    cv.onmousemove  = null;
    cv.onmouseleave = tipHide;
    if (!counts.length) {
      centerNote(ctx, w / 2, (top + bottom) / 2, "NaN と Inf のほかに値がありません");
      return;
    }
    
    // Draw the bars, and the marks of unused codes
    const n       = counts.length;
    const max     = Math.max(1, ...counts);
    const bw      = w / n;
    const gap     = bw >= 4 ? 1 : 0;
    ctx.fillStyle = fig.main;
    for (let i = 0; i < n; i++) {
      const bh = counts[i] ? Math.max(1, counts[i] / max * (bottom - top - 6)) : 0;
      if (bh) ctx.fillRect(i * bw + gap / 2, bottom - bh, Math.max(0.7, bw - gap), bh);
    }
    ctx.fillStyle = colors.caution;
    if (codes) for (let i = 0; i < n; i++) if (!counts[i]) ctx.fillRect(i * bw, bottom + 3, Math.max(1, bw - gap), 4);
    ctx.fillStyle = colors.line;
    ctx.fillRect(0, bottom, w, 1);
    
    // Write the two ends, and 0 when it is inside
    const label      = v => codes ? String(v) : fmt(v);
    const zeroX      = codes ? (0 - lo + 0.5) * bw : hi > lo ? (0 - lo) / (hi - lo) * w : -1;
    ctx.fillStyle    = colors.weak;
    ctx.textBaseline = "top";
    ctx.textAlign    = "left";
    ctx.fillText(label(lo), 0, bottom + 10);
    ctx.textAlign    = "right";
    ctx.fillText(label(hi), w, bottom + 10);
    ctx.textAlign    = "center";
    if (lo < 0 && hi > 0 && zeroX > 40 && zeroX < w - 40) ctx.fillText("0", zeroX, bottom + 10);
    
    // Show the count under the mouse
    cv.onmousemove = e => {
      const i = Math.floor((e.clientX - cv.getBoundingClientRect().left) / bw);
      if (i < 0 || i >= n) tipHide();
      else if (codes) tipShow(e, "コード " + (lo + i) + "\n" + fmtInt(counts[i]) + " 個" + (counts[i] ? "" : "（使われていない）"));
      else tipShow(e, fmt(lo + (hi - lo) * i / n) + " 〜 " + fmt(lo + (hi - lo) * (i + 1) / n) + "\n" + fmtInt(counts[i]) + " 個");
    };
  }
  
  
  function drawTrend(cv, p, axis) {
    //* Draw one value of the point over steps, one line for each shown run (FR-021, FR-034).
    //! Clicking the chart moves to the nearest recorded step.
    
    // Collect the values of each shown run
    const H        = 212;
    const {ctx, w} = prep(cv, H);
    const top      = 6;
    const bottom   = H - 28;
    const L        = GUT_L;
    const R        = w - GUT_R;
    const mt       = METRICS.find(x => x.key === state.metric) || METRICS[2];
    const series   = groupRuns("trends", p).map(f => {
      const m   = recsOf(store.runs.get(f), p);
      const pts = m ? sortedSteps(m).map(s => { const raw = mt.get(m.get(s)); return {s, raw, v: typeof raw === "number" && isFinite(raw) ? raw : null}; }) : [];
      return {name: store.runs.get(f).name, color: runColor(shownRuns().indexOf(f)), pts};
    });
    let lo         = Infinity;
    let hi         = -Infinity;
    for (const sr of series) for (const pt of sr.pts) if (pt.v !== null) {
      lo = Math.min(lo, pt.v);
      hi = Math.max(hi, pt.v);
    }
    field(ctx, L, top, R - L, bottom - top);
    stepGrid(ctx, axis, w, top, bottom);
    stepLabels(ctx, axis, w, bottom + 8);
    cv.onmousemove = null;
    cv.onclick     = null;
    if (!isFinite(lo)) {
      centerNote(ctx, (L + R) / 2, (top + bottom) / 2, "この値は記録されていません");
      return;
    }
    
    // Fix the range of y, and draw the grid and its labels; a spread that 4 digits cannot show is drawn as a flat line
    const big       = Math.max(Math.abs(lo), Math.abs(hi));
    const unit      = big > 0 ? Math.pow(10, Math.floor(Math.log10(big)) - 3) : 0;
    if (hi - lo <= Math.max(4 * unit, 1e-12 * Math.max(1, big))) {
      const d = Math.abs(hi) * 0.1 || 1;
      lo     -= d;
      hi     += d;
    }
    const pad       = (hi - lo) * 0.08;
    const yLo       = lo - pad;
    const yHi       = hi + pad;
    const y         = v => bottom - (v - yLo) / (yHi - yLo) * (bottom - top);
    const ticks     = niceTicks(yLo, yHi, 4);
    ctx.strokeStyle = colors.line;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    for (const t of ticks) {
      ctx.moveTo(L, Math.round(y(t)) + 0.5);
      ctx.lineTo(R, Math.round(y(t)) + 0.5);
    }
    ctx.stroke();
    ctx.fillStyle   = colors.weak;
    ctx.textAlign   = "right";
    for (const t of ticks) ctx.fillText(mt.pct ? fmtPct(t) : fmt(t), L - 8, y(t));
    stepLine(ctx, axis.x(state.step, w), top, bottom);
    
    // Draw the lines; the main run comes last so that it is on top
    for (const sr of series.slice().reverse()) {
      let open        = false;
      ctx.strokeStyle = sr.color;
      ctx.fillStyle   = sr.color;
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      for (const pt of sr.pts) {
        if (pt.v === null) {
          open = false;
          continue;
        }
        if (open) ctx.lineTo(axis.x(pt.s, w), y(pt.v));
        else ctx.moveTo(axis.x(pt.s, w), y(pt.v));
        open = true;
      }
      ctx.stroke();
      if (sr.pts.length <= 60) for (const pt of sr.pts) if (pt.v !== null) {
        ctx.beginPath();
        ctx.arc(axis.x(pt.s, w), y(pt.v), 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // Show the values of the nearest step under the mouse, and move there on click
    const allSteps  = [...new Set(series.flatMap(sr => sr.pts.map(pt => pt.s)))];
    cv.style.cursor = "pointer";
    cv.onmouseleave = tipHide;
    cv.onmousemove  = e => {
      const x     = e.clientX - cv.getBoundingClientRect().left;
      if (x < L - 6 || x > R + 6) {
        tipHide();
        return;
      }
      const s     = nearest(allSteps, axis.step(x, w));
      const lines = ["ステップ " + fmtInt(s)];
      for (const sr of series) {
        const pt = sr.pts.find(q => q.s === s);
        lines.push((series.length > 1 ? sr.name : mt.label) + "  " + (!pt ? "記録なし" : mt.pct ? fmtPct(pt.raw) : fmt(pt.raw) + (mt.unit && typeof pt.raw === "number" ? mt.unit : "")));
      }
      tipShow(e, lines.join("\n"));
    };
    cv.onclick      = e => {
      stopPlay();
      setStep(nearest(view.steps, axis.step(e.clientX - cv.getBoundingClientRect().left, w)));
    };
  }
  
  
  function drawDist(cv, recsMap, steps, axis, scale) {
    //* Draw the distribution of every step side by side: x is the step, y the value and the color the count (FR-030).
    //? scale ({lo, hi, cmax}) is shared by the runs of the trends group; null uses the range of this run (FR-041).
    //! The color uses the log of the count, so that thin tails stay visible. A red dot marks a step with NaN or Inf.
    
    // Find the range of the values over all steps
    const H         = 194;
    const {ctx, w}  = prep(cv, H);
    const top       = 6;
    const bottom    = H - 28;
    const L         = GUT_L;
    const R         = w - GUT_R;
    const recs      = steps.map(s => recsMap.get(s));
    const ok        = recs.filter(r => r.hist && r.hist.counts && r.hist.counts.length && typeof r.hist.lo === "number");
    field(ctx, L, top, R - L, bottom - top);
    stepLabels(ctx, axis, w, bottom + 8);
    cv.onmousemove  = null;
    cv.onmouseleave = tipHide;
    if (!ok.length) {
      centerNote(ctx, (L + R) / 2, (top + bottom) / 2, "分布の記録がありません");
      return;
    }
    let lo          = scale ? scale.lo : Math.min(...ok.map(r => r.hist.lo));
    let hi          = scale ? scale.hi : Math.max(...ok.map(r => r.hist.hi));
    let cmax        = scale ? scale.cmax : 1;
    if (!(hi > lo)) {
      lo -= Math.abs(hi) * 0.1 || 1;
      hi += Math.abs(hi) * 0.1 || 1;
    }
    if (!scale) for (const r of ok) for (const c of r.hist.counts) cmax = Math.max(cmax, c);
    
    // Draw one column for each step; each column reaches halfway to its neighbors
    const y    = v => bottom - (v - lo) / (hi - lo) * (bottom - top);
    const xs   = steps.map(s => axis.x(s, w));
    const half = steps.length > 1 ? Math.min(...xs.slice(1).map((x, i) => x - xs[i])) / 2 : Math.min(24, (R - L) / 2);
    const cols = recs.map((r, i) => ({r, x0: Math.max(L, i > 0 ? (xs[i - 1] + xs[i]) / 2 : xs[i] - half), x1: Math.min(R, i < xs.length - 1 ? (xs[i] + xs[i + 1]) / 2 : xs[i] + half)}));
    for (const [i, col] of cols.entries()) {
      const hst = col.r.hist;
      const n   = hst && hst.counts ? hst.counts.length : 0;
      for (let j = 0; j < n; j++) {
        if (!hst.counts[j]) continue;
        const ya      = y(hst.lo + (hst.hi - hst.lo) * j / n);
        const yb      = y(hst.lo + (hst.hi - hst.lo) * (j + 1) / n);
        ctx.fillStyle = rgbCss(cmap(0.12 + 0.88 * Math.log1p(hst.counts[j]) / Math.log1p(cmax)));
        ctx.fillRect(col.x0, yb, col.x1 - col.x0 + 0.5, Math.max(1, ya - yb + 0.5));
      }
      if (col.r.nan > 0 || col.r.inf > 0) {
        ctx.fillStyle = colors.alert;
        ctx.beginPath();
        ctx.arc(xs[i], top + 7, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // Write the labels of y and draw the line of the chosen step
    ctx.fillStyle = colors.weak;
    ctx.textAlign = "right";
    for (const t of niceTicks(lo, hi, 3)) ctx.fillText(fmt(t), L - 8, y(t));
    stepLine(ctx, axis.x(state.step, w), top, bottom);
    
    // Show which color means how many values (counts), on the right; the colors use the log of the count
    const counts = [];
    for (let c = 1; c < cmax; c *= 10) counts.push({frac: Math.log1p(c) / Math.log1p(cmax), text: c >= 1e4 ? "1e" + Math.round(Math.log10(c)) : fmtInt(c)});
    counts.push({frac: 1, text: cmax >= 1e4 ? fmt(cmax) : fmtInt(cmax)});
    colorBar(ctx, R + 14, top, bottom, frac => cmap(0.12 + 0.88 * frac), counts.reverse());
    
    // Show the bin under the mouse
    cv.onmousemove = e => {
      const rc    = cv.getBoundingClientRect();
      const x     = e.clientX - rc.left;
      const col   = cols.find(c => x >= c.x0 && x <= c.x1);
      const hst   = col ? col.r.hist : null;
      if (!hst || !hst.counts || !hst.counts.length) {
        tipHide();
        return;
      }
      const v     = lo + (bottom - (e.clientY - rc.top)) / (bottom - top) * (hi - lo);
      const n     = hst.counts.length;
      const j     = Math.floor((v - hst.lo) / ((hst.hi - hst.lo) || 1) * n);
      const head  = "ステップ " + fmtInt(col.r.step);
      const extra = col.r.nan > 0 || col.r.inf > 0 ? "\nNaN " + fmtInt(col.r.nan) + "  Inf " + fmtInt(col.r.inf) : "";
      if (j < 0 || j >= n) tipShow(e, head + extra);
      else tipShow(e, head + "\n" + fmt(hst.lo + (hst.hi - hst.lo) * j / n) + " 〜 " + fmt(hst.lo + (hst.hi - hst.lo) * (j + 1) / n) + "\n" + fmtInt(hst.counts[j]) + " 個" + extra);
    };
  }
  
  
  function drawStrips(cv, run, steps, axis, m, keys) {
    //* Draw one strip for each followed element (FR-032). A quantized element is drawn in code units (value / scale + zero point),
    //* over the lines of the levels and the orange bands around the boundaries between them (FR-031).
    //! Levels are in code units with the zero point (02c quant.levels). The boundaries follow quant.round_mode: midpoints (nearest) or the levels (floor).
    
    // Find the levels (the latest ones up to the chosen step) and the width of the bands
    let levels     = null;
    let floor      = false;
    for (const s of steps) {
      const q = m.get(s).quant;
      if (q && Array.isArray(q.levels) && (s <= state.step || !levels)) levels = q.levels.slice().sort((a, b) => a - b);
      if (q && s <= state.step) floor = q.round_mode === "floor";
    }
    const settings = run.info && run.info.settings ? run.info.settings : {};
    const delta    = typeof settings.boundary_width === "number" ? settings.boundary_width : 0.05;
    const shape    = m.get(steps[0]).shape || [];
    const bounds   = levels ? levels.slice(1).map((b, j) => ({at: floor ? b : (levels[j] + b) / 2, gap: b - levels[j]})) : [];
    const near     = u => bounds.some(b => Math.abs(u - b.at) <= delta * b.gap);
    
    // Size the canvas for all strips
    const rowH     = 66;
    const gap      = 12;
    const {ctx, w} = prep(cv, Math.max(10, keys.length * (rowH + gap) - gap + 26));
    const L        = GUT_L;
    const R        = w - GUT_R;
    const geo      = [];
    for (const [i, key] of keys.entries()) {
      
      // Collect the values of the element, and choose the range of y
      const y0    = i * (rowH + gap) + 1;
      const y1    = y0 + rowH;
      const pts   = steps.map(s => { const e = (m.get(s).elements || {})[key]; return e && typeof e.value === "number" ? {s, v: e.value, sc: e.scale, z: typeof e.zero_point === "number" ? e.zero_point : 0} : null; }).filter(Boolean);
      const quant = !!levels && pts.length > 0 && pts.every(q => typeof q.sc === "number" && q.sc > 0);
      const us    = pts.map(q => ({s: q.s, v: q.v, u: quant ? q.v / q.sc + q.z : q.v}));
      let lo      = us.length ? Math.min(...us.map(q => q.u)) : 0;
      let hi      = us.length ? Math.max(...us.map(q => q.u)) : 1;
      if (quant) {
        lo -= 0.7;
        hi += 0.7;
      } else {
        const d = (hi - lo) * 0.12 || Math.abs(hi) * 0.1 || 1;
        lo     -= d;
        hi     += d;
      }
      if (quant && hi - lo < 2.6) {
        const c = (lo + hi) / 2;
        lo      = c - 1.3;
        hi      = c + 1.3;
      }
      const yy    = u => y1 - (u - lo) / (hi - lo) * rowH;
      field(ctx, L, y0, R - L, rowH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(L, y0, R - L, rowH);
      ctx.clip();
      
      // Draw the boundary bands, the area out of range and the level lines
      if (quant) {
        let minGap      = Infinity;
        const mids      = [];
        ctx.fillStyle   = alpha(colors.caution, 0.2);
        for (let j = 0; j + 1 < levels.length; j++) {
          const a   = levels[j];
          const b   = levels[j + 1];
          const gpx = (b - a) / (hi - lo) * rowH;
          if (b < lo || a > hi) continue;
          minGap    = Math.min(minGap, gpx);
          if (gpx < 8) continue;
          const at  = floor ? b : (a + b) / 2;
          mids.push(at);
          ctx.fillRect(L, yy(at + delta * (b - a)), R - L, Math.max(1, yy(at - delta * (b - a)) - yy(at + delta * (b - a))));
        }
        ctx.fillStyle   = alpha(colors.caution, 0.08);
        if (hi > levels[levels.length - 1]) ctx.fillRect(L, y0, R - L, yy(levels[levels.length - 1]) - y0);
        if (lo < levels[0]) ctx.fillRect(L, yy(levels[0]), R - L, y1 - yy(levels[0]));
        ctx.strokeStyle = colors.line;
        ctx.lineWidth   = 1;
        ctx.beginPath();
        for (const l of levels) if (l >= lo && l <= hi) {
          ctx.moveTo(L, Math.round(yy(l)) + 0.5);
          ctx.lineTo(R, Math.round(yy(l)) + 0.5);
        }
        if (minGap >= 3) ctx.stroke();
        
        // Draw each boundary as a dashed line, so that a value going back and forth across it is easy to see
        ctx.strokeStyle = alpha(colors.caution, 0.75);
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        for (const mid of mids) {
          ctx.moveTo(L, Math.round(yy(mid)) + 0.5);
          ctx.lineTo(R, Math.round(yy(mid)) + 0.5);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      
      // Draw the line of the chosen step and the path of the element; dots near a boundary are orange
      stepLine(ctx, axis.x(state.step, w), y0, y1);
      ctx.strokeStyle = fig.main;
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      for (const [j, q] of us.entries()) {
        if (j) ctx.lineTo(axis.x(q.s, w), yy(q.u));
        else ctx.moveTo(axis.x(q.s, w), yy(q.u));
      }
      ctx.stroke();
      for (const q of us) {
        const isNear  = quant && near(q.u);
        ctx.fillStyle = isNear ? colors.caution : fig.main;
        ctx.beginPath();
        ctx.arc(axis.x(q.s, w), yy(q.u), isNear ? 3 : 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      
      // Write the index of the element in the strip, and the code of each level line on the left
      const idx        = [];
      let rest         = +key;
      for (let d = shape.length - 1; d >= 0; d--) {
        idx.unshift(rest % shape[d]);
        rest = Math.floor(rest / shape[d]);
      }
      const label      = shape.length > 1 ? "[" + idx.join(", ") + "]" : "[" + key + "]";
      ctx.font         = "12px " + fonts.mono;
      ctx.textAlign    = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle    = alpha(colors.surface, 0.85);
      ctx.fillRect(L + 4, y0 + 3, ctx.measureText(label).width + 8, 17);
      ctx.fillStyle    = colors.weak;
      ctx.fillText(label, L + 8, y0 + 5);
      ctx.font         = "12px " + fonts.sans;
      ctx.textAlign    = "right";
      ctx.textBaseline = "middle";
      let last         = -Infinity;
      if (quant) for (const l of levels) if (l >= lo && l <= hi && Math.abs(yy(l) - last) >= 14) {
        ctx.fillText(Number.isInteger(l) ? String(l) : fmt(l), L - 8, yy(l));
        last = yy(l);
      }
      if (!quant) {
        ctx.fillText(fmt(hi), L - 8, y0 + 8);
        ctx.fillText(fmt(lo), L - 8, y1 - 8);
      }
      geo.push({y0, y1, us, quant, label});
    }
    stepLabels(ctx, axis, w, keys.length * (rowH + gap) - gap + 9);
    
    // Show the value under the mouse
    cv.onmouseleave = tipHide;
    cv.onmousemove  = e => {
      const rc = cv.getBoundingClientRect();
      const x  = e.clientX - rc.left;
      const yv = e.clientY - rc.top;
      const g  = geo.find(o => yv >= o.y0 && yv <= o.y1);
      if (!g || !g.us.length || x < L - 6 || x > R + 6) {
        tipHide();
        return;
      }
      const s  = axis.step(x, w);
      const q  = g.us.reduce((best, o) => Math.abs(o.s - s) < Math.abs(best.s - s) ? o : best);
      tipShow(e, "要素 " + g.label + "\nステップ " + fmtInt(q.s) + "\n値 " + fmt(q.v) + (g.quant ? "\nコードの位置 " + q.u.toFixed(2) : ""));
    };
  }
  
  
  function drawHistMany(cv, items, mode) {
    //* Draw the distributions of several runs over one axis as outlines in their colors: densities of the values, or shares of the codes (FR-041, U-054).
    //! A distribution whose values are all alike is a line at its value, as tall as the chart.
    
    // Turn each run into bins of (x0, x1, y, count)
    const H         = 150;
    const {ctx, w}  = prep(cv, H);
    const top       = 4;
    const bottom    = H - 28;
    const shapes    = items.map(it => {
      if (mode === "code") {
        const s = core.codeShare(it.rec.quant.codes);
        return s ? {it, bins: s.ys.map((y, i) => ({x0: s.offset + i - 0.5, x1: s.offset + i + 0.5, y, n: it.rec.quant.codes.counts[i]}))} : null;
      }
      const d = core.histDensity(it.rec.hist);
      const n = d ? d.ys.length : 0;
      return d ? {it, bins: d.ys.map((y, i) => ({x0: d.lo + (d.hi - d.lo) * i / n, x1: d.lo + (d.hi - d.lo) * (i + 1) / n, y, n: it.rec.hist.counts[i]}))} : null;
    }).filter(Boolean);
    field(ctx, 0, top, w, bottom - top);
    cv.onmousemove  = null;
    cv.onmouseleave = tipHide;
    if (!shapes.length) {
      centerNote(ctx, w / 2, (top + bottom) / 2, "NaN と Inf のほかに値がありません");
      return;
    }
    
    // One range of x for all runs, and y up to the largest density or share
    const spike     = s => s.bins.length === 1 && s.bins[0].x0 === s.bins[0].x1;
    let lo          = Math.min(...shapes.map(s => s.bins[0].x0));
    let hi          = Math.max(...shapes.map(s => s.bins[s.bins.length - 1].x1));
    if (!(hi > lo)) {
      lo -= 0.5;
      hi += 0.5;
    }
    const ymax      = Math.max(1e-300, ...shapes.filter(s => !spike(s)).flatMap(s => s.bins.map(b => b.y)));
    const x         = v => (v - lo) / (hi - lo) * w;
    const y         = v => bottom - Math.min(1, v / ymax) * (bottom - top - 6);
    
    // Draw the runs from the last to the first, so that the first run of the list is on top
    for (const s of shapes.slice().reverse()) {
      ctx.strokeStyle = s.it.color;
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      if (spike(s)) {
        ctx.moveTo(x(s.bins[0].x0), bottom);
        ctx.lineTo(x(s.bins[0].x0), top + 6);
        ctx.stroke();
        continue;
      }
      ctx.moveTo(x(s.bins[0].x0), bottom);
      for (const b of s.bins) {
        ctx.lineTo(x(b.x0), y(b.y));
        ctx.lineTo(x(b.x1), y(b.y));
      }
      ctx.lineTo(x(s.bins[s.bins.length - 1].x1), bottom);
      ctx.fillStyle   = alpha(s.it.color, 0.14);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle    = colors.line;
    ctx.fillRect(0, bottom, w, 1);
    
    // Write the two ends, and 0 when it is inside
    const label      = v => mode === "code" ? String(Math.round(v)) : fmt(v);
    const ends       = mode === "code" ? [lo + 0.5, hi - 0.5] : [lo, hi];
    ctx.fillStyle    = colors.weak;
    ctx.textBaseline = "top";
    ctx.textAlign    = "left";
    ctx.fillText(label(ends[0]), 0, bottom + 10);
    ctx.textAlign    = "right";
    ctx.fillText(label(ends[1]), w, bottom + 10);
    ctx.textAlign    = "center";
    if (lo < 0 && hi > 0 && x(0) > 40 && x(0) < w - 40) ctx.fillText("0", x(0), bottom + 10);
    
    // Show the count of each run in the bin under the mouse
    cv.onmousemove   = e => {
      const v     = lo + (e.clientX - cv.getBoundingClientRect().left) / w * (hi - lo);
      const lines = [mode === "code" ? "コード " + Math.round(v) : fmt(v)];
      for (const s of shapes) {
        const b = s.bins.find(q => v >= q.x0 && v < q.x1) || (spike(s) && Math.abs(x(v) - x(s.bins[0].x0)) < 4 ? s.bins[0] : null);
        lines.push(s.it.name + "  " + (b ? fmtInt(b.n) + " 個" : "範囲の外"));
      }
      tipShow(e, lines.join("\n"));
    };
  }
  
  
  function drawStripsMany(cv, runs, axis, keys) {
    //* Draw one strip for each followed element with a line for each run, in real values (FR-032, FR-041).
    //! The level lines and the boundary bands come from the first run with levels at the chosen step, turned into real values with the scale and the zero point of its element.
    
    // Find the run that gives the levels, and size the canvas for all strips
    const rowH     = 66;
    const gap      = 12;
    const {ctx, w} = prep(cv, Math.max(10, keys.length * (rowH + gap) - gap + 26));
    const L        = GUT_L;
    const R        = w - GUT_R;
    const ref      = runs.map(r => ({r, s: core.stepAtOrBefore(r.steps, state.step)})).find(o => o.s !== null && o.r.m.get(o.s).quant && Array.isArray(o.r.m.get(o.s).quant.levels));
    const refRec   = ref ? ref.r.m.get(ref.s) : null;
    const levels   = refRec ? refRec.quant.levels.slice().sort((a, b) => a - b) : null;
    const floor    = !!refRec && refRec.quant.round_mode === "floor";
    const info     = ref ? store.runs.get(ref.r.f).info : null;
    const delta    = info && info.settings && typeof info.settings.boundary_width === "number" ? info.settings.boundary_width : 0.05;
    const shape    = (runs[0].m.get(runs[0].steps[0]) || {}).shape || [];
    const geo      = [];
    for (const [i, key] of keys.entries()) {
      // The values of each run, and the levels of the reference run in real values
      const y0    = i * (rowH + gap) + 1;
      const y1    = y0 + rowH;
      const lines = runs.map(r => ({r, pts: r.steps.map(s => { const e = (r.m.get(s).elements || {})[key]; return e && typeof e.value === "number" ? {s, v: e.value} : null; }).filter(Boolean)}));
      const vals  = lines.flatMap(l => l.pts.map(q => q.v));
      const el    = refRec && refRec.elements ? refRec.elements[key] : null;
      const grid  = levels && el && typeof el.scale === "number" && el.scale > 0 ? levels.map(l => (l - (typeof el.zero_point === "number" ? el.zero_point : 0)) * el.scale) : null;
      let lo      = vals.length ? Math.min(...vals) : 0;
      let hi      = vals.length ? Math.max(...vals) : 1;
      const pad   = (hi - lo) * 0.12 || Math.abs(hi) * 0.1 || 1;
      lo         -= pad;
      hi         += pad;
      const yy    = v => y1 - (v - lo) / (hi - lo) * rowH;
      field(ctx, L, y0, R - L, rowH);
      ctx.save();
      ctx.beginPath();
      ctx.rect(L, y0, R - L, rowH);
      ctx.clip();
      
      // The boundary bands, the dashed boundaries and the level lines, where they are far enough apart to read
      if (grid) {
        const mids      = [];
        ctx.fillStyle   = alpha(colors.caution, 0.2);
        for (let j = 0; j + 1 < grid.length; j++) {
          const a  = grid[j];
          const b  = grid[j + 1];
          const at = floor ? b : (a + b) / 2;
          if (b < lo || a > hi || (b - a) / (hi - lo) * rowH < 8) continue;
          mids.push(at);
          ctx.fillRect(L, yy(at + delta * (b - a)), R - L, Math.max(1, yy(at - delta * (b - a)) - yy(at + delta * (b - a))));
        }
        ctx.strokeStyle = colors.line;
        ctx.lineWidth   = 1;
        ctx.beginPath();
        for (const l of grid) if (l >= lo && l <= hi) {
          ctx.moveTo(L, Math.round(yy(l)) + 0.5);
          ctx.lineTo(R, Math.round(yy(l)) + 0.5);
        }
        if (grid.filter(l => l >= lo && l <= hi).length * 3 <= rowH) ctx.stroke();
        ctx.strokeStyle = alpha(colors.caution, 0.75);
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        for (const mid of mids) {
          ctx.moveTo(L, Math.round(yy(mid)) + 0.5);
          ctx.lineTo(R, Math.round(yy(mid)) + 0.5);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      
      // The line of the chosen step, and one line of each run (the first run of the list on top)
      stepLine(ctx, axis.x(state.step, w), y0, y1);
      for (const l of lines.slice().reverse()) {
        ctx.strokeStyle = l.r.color;
        ctx.fillStyle   = l.r.color;
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        for (const [j, q] of l.pts.entries()) {
          if (j) ctx.lineTo(axis.x(q.s, w), yy(q.v));
          else ctx.moveTo(axis.x(q.s, w), yy(q.v));
        }
        ctx.stroke();
        if (l.pts.length <= 60) for (const q of l.pts) {
          ctx.beginPath();
          ctx.arc(axis.x(q.s, w), yy(q.v), 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
      
      // Write the index of the element and the range of the values on the left
      const idx        = [];
      let rest         = +key;
      for (let d = shape.length - 1; d >= 0; d--) {
        idx.unshift(rest % shape[d]);
        rest = Math.floor(rest / shape[d]);
      }
      const label      = shape.length > 1 ? "[" + idx.join(", ") + "]" : "[" + key + "]";
      ctx.font         = "12px " + fonts.mono;
      ctx.textAlign    = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle    = alpha(colors.surface, 0.85);
      ctx.fillRect(L + 4, y0 + 3, ctx.measureText(label).width + 8, 17);
      ctx.fillStyle    = colors.weak;
      ctx.fillText(label, L + 8, y0 + 5);
      ctx.font         = "12px " + fonts.sans;
      ctx.textAlign    = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(fmt(hi), L - 8, y0 + 8);
      ctx.fillText(fmt(lo), L - 8, y1 - 8);
      geo.push({y0, y1, lines, label});
    }
    stepLabels(ctx, axis, w, keys.length * (rowH + gap) - gap + 9);
    
    // Show the value of each run at the step under the mouse
    cv.onmouseleave = tipHide;
    cv.onmousemove  = e => {
      const rc  = cv.getBoundingClientRect();
      const x   = e.clientX - rc.left;
      const yv  = e.clientY - rc.top;
      const g   = geo.find(o => yv >= o.y0 && yv <= o.y1);
      if (!g || x < L - 6 || x > R + 6) {
        tipHide();
        return;
      }
      const s   = axis.step(x, w);
      const out = ["要素 " + g.label];
      for (const l of g.lines) {
        const q = l.pts.length ? l.pts.reduce((best, o) => Math.abs(o.s - s) < Math.abs(best.s - s) ? o : best) : null;
        out.push(store.runs.get(l.r.f).name + "  " + (q ? fmt(q.v) + "（ステップ " + fmtInt(q.s) + "）" : "記録なし"));
      }
      tipShow(e, out.join("\n"));
    };
  }
  
  
  // ===== Value maps (CR-003 proposal) =====
  
  const mapCache = new WeakMap();
  
  
  function mapCells(map) {
    //* Return the cells of a map as bytes (U-041).
    
    return core.mapCells(map);
  }
  
  
  function axisName(names) {
    //* Return the name of an axis of a map (U-041, 02b 2.4).
    
    return core.axisName(names);
  }
  
  
  function cellText(map, r, c) {
    //* Return the text of the tooltip for one cell: the rows and columns it covers, and its absolute value.
    
    const span = core.mapSpan;
    const cols = map.col.length ? axisName(map.col) + " " + span(c, map.cols, map.col_size) + "\n" : "";
    return axisName(map.row) + " " + span(r, map.rows, map.row_size) + "\n" + cols + "絶対値 " + fmt(mapCells(map)[r * map.cols + c] / 255 * map.max);
  }
  
  
  function drawMapFlat(cv, map, smax) {
    //* Draw the map from above: rows go down, columns go right and the color is the absolute value.
    //? smax is the largest max of the maps that the group shows, so that one value has one color in every run (FR-041).
    
    // Find the map of the chosen step; a map of an image, of attention weights or of a matrix keeps square cells (02b 2.4)
    const H         = 340;
    const {ctx, w}  = prep(cv, H);
    const top       = 8;
    const bottom    = H - 34;
    const square    = !!map && core.mapSquare(map.layout);
    const right     = w - 84;
    const L         = square ? Math.max(46, (right - (bottom - top)) / 2) : 46;
    const R         = square ? Math.min(right, L + bottom - top) : right;
    cv.onmousemove  = null;
    cv.onmouseleave = tipHide;
    if (!map) {
      centerNote(ctx, w / 2, H / 2, "このステップには記録がありません");
      return;
    }
    
    // Paint the cells into a small image
    const cells = mapCells(map);
    const img   = document.createElement("canvas");
    img.width   = map.cols;
    img.height  = map.rows;
    const ictx  = img.getContext("2d");
    const data  = ictx.createImageData(map.cols, map.rows);
    for (let i = 0; i < map.rows * map.cols; i++) data.data.set([...cmap(core.mapFrac(cells[i] / 255 * map.max, smax)), 255], i * 4);
    ictx.putImageData(data, 0, 0);
    
    // Stretch the image over the plot area, without smoothing
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, L, top, R - L, bottom - top);
    
    // Write the ends of the axes that have more than one position, and the names of the axes
    ctx.fillStyle    = colors.weak;
    ctx.textAlign    = "right";
    if (map.row_size > 1) {
      ctx.textBaseline = "top";
      ctx.fillText("0", L - 6, top);
      ctx.textBaseline = "bottom";
      ctx.fillText(String(map.row_size - 1), L - 6, bottom);
    }
    ctx.textBaseline = "top";
    if (map.col_size > 1) {
      ctx.textAlign = "left";
      ctx.fillText("0", L, bottom + 6);
      ctx.textAlign = "right";
      ctx.fillText(String(map.col_size - 1), R, bottom + 6);
    }
    ctx.textAlign    = "center";
    ctx.fillText(axisName(map.col), (L + R) / 2, bottom + 6);
    ctx.save();
    ctx.translate(L - 12, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "bottom";
    ctx.fillText(axisName(map.row), 0, 0);
    ctx.restore();
    
    // Show which color means which absolute value, on the right
    colorBar(ctx, R + 14, top, bottom, frac => cmap(frac), core.mapTicks(smax).map(o => ({frac: o.frac, text: o.value === 0 ? "0" : fmt(o.value)})));
    
    // Show the cell under the mouse
    cv.onmousemove = e => {
      const rc = cv.getBoundingClientRect();
      const c  = Math.floor((e.clientX - rc.left - L) / (R - L) * map.cols);
      const r  = Math.floor((e.clientY - rc.top - top) / (bottom - top) * map.rows);
      if (c < 0 || c >= map.cols || r < 0 || r >= map.rows) tipHide();
      else tipShow(e, cellText(map, r, c));
    };
  }
  
  
  function drawMap3d(cv, map, smax) {
    //* Draw the map as a surface, like plot_surface of matplotlib: x is the column axis, y the row axis and the height the absolute value. Each cell has the color of the top view (BUG-011).
    //! The camera turns around the vertical axis (az) and looks down from el degrees. Quads are painted from far to near.
    //? smax is the largest max of the maps that the group shows: the heights and the colors use it, as the top view does (FR-041).
    
    // Make the projection
    const H         = 340;
    const {ctx, w}  = prep(cv, H);
    cv.onmousemove  = null;
    cv.onmouseleave = tipHide;
    if (!map) {
      centerNote(ctx, w / 2, H / 2, "このステップには記録がありません");
      return;
    }
    if (map.rows < 2 || map.cols < 2) {
      centerNote(ctx, w / 2, H / 2, "行と列が2つ以上あるときに描けます");
      return;
    }
    const rows      = map.rows;
    const cols      = map.cols;
    const cells     = mapCells(map);
    const fracs     = Array.from(cells, c => core.mapFrac(c / 255 * map.max, smax));
    const Z         = 0.55;
    const az        = state.rot.az * Math.PI / 180;
    const el        = state.rot.el * Math.PI / 180;
    const d         = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    const r         = [-Math.sin(az), Math.cos(az), 0];
    const u         = [-Math.sin(el) * Math.cos(az), -Math.sin(el) * Math.sin(az), Math.cos(el)];
    const scale     = Math.min(w - 60, H - 30) / (2 * Math.sqrt(0.5 + Z * Z / 4));
    const P         = (x, y, z) => [w / 2 + scale * (x * r[0] + y * r[1]), H / 2 - scale * (x * u[0] + y * u[1] + (z - Z / 2) * u[2]), x * d[0] + y * d[1] + (z - Z / 2) * d[2]];
    
    // Draw the floor and the two back walls, with a few grid lines
    const ax        = d[0] > 0 ? -0.5 : 0.5;
    const ay        = d[1] > 0 ? -0.5 : 0.5;
    const poly      = pts => {
      ctx.beginPath();
      for (const [i, pt] of pts.entries()) {
        if (i) ctx.lineTo(pt[0], pt[1]);
        else ctx.moveTo(pt[0], pt[1]);
      }
      ctx.closePath();
    };
    ctx.lineWidth   = 1;
    ctx.strokeStyle = colors.line;
    ctx.fillStyle   = alpha(colors.line, 0.28);
    for (const pane of [[P(-0.5, -0.5, 0), P(0.5, -0.5, 0), P(0.5, 0.5, 0), P(-0.5, 0.5, 0)], [P(ax, -0.5, 0), P(ax, 0.5, 0), P(ax, 0.5, Z), P(ax, -0.5, Z)], [P(-0.5, ay, 0), P(0.5, ay, 0), P(0.5, ay, Z), P(-0.5, ay, Z)]]) {
      poly(pane);
      ctx.fill();
      ctx.stroke();
    }
    ctx.beginPath();
    for (const f of [0.25, 0.5, 0.75]) {
      for (const [a, b] of [[P(-0.5 + f, -0.5, 0), P(-0.5 + f, 0.5, 0)], [P(-0.5, -0.5 + f, 0), P(0.5, -0.5 + f, 0)], [P(ax, -0.5, Z * f), P(ax, 0.5, Z * f)], [P(-0.5, ay, Z * f), P(0.5, ay, Z * f)]]) {
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
    }
    ctx.stroke();
    
    // Cut the surface between each four cell centers into four parts at the middles of its sides; each part belongs to the nearest center
    const at    = (i, j, z) => P(...core.mapPoint(i, j, rows, cols), z);
    const parts = [];
    for (let i = 0; i + 1 < rows; i++) {
      for (let j = 0; j + 1 < cols; j++) {
        const z00 = fracs[i * cols + j] * Z;
        const z01 = fracs[i * cols + j + 1] * Z;
        const z11 = fracs[(i + 1) * cols + j + 1] * Z;
        const z10 = fracs[(i + 1) * cols + j] * Z;
        const c00 = at(i, j, z00);
        const c01 = at(i, j + 1, z01);
        const c11 = at(i + 1, j + 1, z11);
        const c10 = at(i + 1, j, z10);
        const top = at(i, j + 0.5, (z00 + z01) / 2);
        const rgt = at(i + 0.5, j + 1, (z01 + z11) / 2);
        const bot = at(i + 1, j + 0.5, (z11 + z10) / 2);
        const lft = at(i + 0.5, j, (z10 + z00) / 2);
        const mid = at(i + 0.5, j + 0.5, (z00 + z01 + z11 + z10) / 4);
        for (const [ci, cj, pts] of [[i, j, [c00, top, mid, lft]], [i, j + 1, [top, c01, rgt, mid]], [i + 1, j + 1, [mid, rgt, c11, bot]], [i + 1, j, [lft, mid, bot, c10]]]) {
          parts.push({i: ci, j: cj, pts, depth: pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]});
        }
      }
    }
    parts.sort((a, b) => a.depth - b.depth);
    
    // Paint the parts from far to near with the color of their cell: the same color as the top view and its color bar, without shading (02b 2.4, BUG-011)
    for (const part of parts) {
      ctx.fillStyle   = rgbCss(cmap(fracs[part.i * cols + part.j]));
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth   = 0.6;
      poly(part.pts);
      ctx.fill();
      ctx.stroke();
    }
    
    // Write the names of the axes along the front edges of the floor, and the heights on a side edge (a height whose label would overlap one above it is left out, as when seen from above)
    const centre     = P(0, 0, 0);
    const outward    = (pt, dist) => {
      const dx  = pt[0] - centre[0];
      const dy  = pt[1] - centre[1];
      const len = Math.hypot(dx, dy) || 1;
      return [pt[0] + dx / len * dist, pt[1] + dy / len * dist];
    };
    const colPos     = outward(P(0, -ay, 0), 20);
    const rowPos     = outward(P(-ax, 0, 0), 20);
    const side       = P(ax, -ay, 0)[0] > P(-ax, ay, 0)[0] ? [ax, -ay] : [-ax, ay];
    ctx.fillStyle    = colors.weak;
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(axisName(map.col), colPos[0], colPos[1]);
    ctx.fillText(axisName(map.row), rowPos[0], rowPos[1]);
    ctx.textAlign    = "left";
    const placed     = [];
    for (const o of core.mapTicks(smax)) {
      const pt = P(side[0], side[1], Z * o.frac);
      if (placed.some(y => Math.abs(y - pt[1]) < 14)) continue;
      placed.push(pt[1]);
      ctx.fillText(o.value === 0 ? "0" : fmt(o.value), pt[0] + 8, pt[1]);
    }
    ctx.fillText("絶対値", P(side[0], side[1], Z)[0] + 8, P(side[0], side[1], Z)[1] - 16);
    
    // Show the cell under the mouse: test the parts from near to far
    cv.onmousemove = e => {
      if (dragging) return;
      const rc = cv.getBoundingClientRect();
      const mx = e.clientX - rc.left;
      const my = e.clientY - rc.top;
      for (let n = parts.length - 1; n >= 0; n--) {
        const pts = parts[n].pts;
        let in4   = false;
        for (let a = 0, b = 3; a < 4; b = a++) if ((pts[a][1] > my) !== (pts[b][1] > my) && mx < (pts[b][0] - pts[a][0]) * (my - pts[a][1]) / (pts[b][1] - pts[a][1]) + pts[a][0]) in4 = !in4;
        if (in4) {
          tipShow(e, cellText(map, parts[n].i, parts[n].j));
          return;
        }
      }
      tipHide();
    };
  }
  
  
  // ===== Compare view (SC-02) =====
  
  function f64(b64) {
    //* Turn base64 of little-endian float32 values into numbers (U-042, 02c compare files).
    
    return core.decodeFloat32(b64);
  }
  
  
  function renderCompare(main) {
    //* Draw the compare view: choose two runs and a step, then list the error of each point (FR-016).
    
    // Header, and the runs that saved full tensors
    const menu  = h("button", {class: "btn icon menu", "aria-label": "一覧を開く", onclick: toggleSide}, icon(ICON_MENU, 18));
    main.append(h("header", {class: "head"}, h("div", {class: "title-row"}, menu, h("h1", {class: "title plain"}, "比較"), renderActions(false))));
    const cands = store.order.map(f => store.runs.get(f)).filter(r => r.info && r.info.settings && Array.isArray(r.info.settings.compare_steps) && r.info.settings.compare_steps.length);
    if (cands.length < 2) {
      main.append(h("p", {class: "note"}, "比べるには、比較用のテンソルを保存した実行（run_start の compare_steps）が2つ必要です"));
      return;
    }
    
    // Choose the runs and the step; a run with "float" in its name is the default base
    if (!cands.some(r => r.folder === state.base)) state.base = (cands.find(r => /float/.test(r.name)) || cands[0]).folder;
    if (!cands.some(r => r.folder === state.target) || state.target === state.base) state.target = (cands.find(r => r.folder !== state.base && !/float/.test(r.name)) || cands.find(r => r.folder !== state.base)).folder;
    const baseRun   = store.runs.get(state.base);
    const targetRun = store.runs.get(state.target);
    const common    = baseRun.info.settings.compare_steps.filter(s => targetRun.info.settings.compare_steps.includes(s));
    if (!common.includes(state.cstep)) state.cstep = common.length ? common[common.length - 1] : null;
    writeHash();
    
    // Make the three pickers
    const pick = (label, value, items, onPick) => {
      const options = items.map(([v, text]) => h("option", {value: v, selected: String(v) === String(value)}, text));
      return h("label", null, label, h("select", {onchange: e => onPick(e.target.value)}, options));
    };
    const runs = list => list.map(r => [r.folder, r.name + "  " + shortDate(r.started)]);
    main.append(h("div", {class: "pickers"}, pick("基準"    , state.base  , runs(cands)                                    , v => { state.base = v; renderMain(); }  ),
                                              pick("比べる"  , state.target, runs(cands.filter(r => r.folder !== state.base)), v => { state.target = v; renderMain(); }),
                                              pick("ステップ", state.cstep , common.map(s => [s, String(s)])                 , v => { state.cstep = +v; renderMain(); })));
    if (state.cstep === null) {
      main.append(h("p", {class: "note"}, "2つの実行に共通する比較用のステップがありません"));
      return;
    }
    
    // Compare the points with the same name; the error is computed in float64 (IF-07)
    const holder = h("p", {class: "note"}, "計算中");
    const my     = ++token;
    const step   = state.cstep;
    main.append(holder);
    (async () => {
      await ensureCatalog(baseRun);
      await ensureCatalog(targetRun);
      const rows   = [];
      let mismatch = 0;
      for (const bp of baseRun.points.values()) {
        const tp    = targetRun.byName.get(bp.name);
        const ok1   = tp ? await once(baseRun  , "c" + step + "/" + bp.id, () => readData(baseRun  , runBase(baseRun  ) + "compare/" + step + "/" + bp.id + ".js")) : false;
        const ok2   = tp ? await once(targetRun, "c" + step + "/" + tp.id, () => readData(targetRun, runBase(targetRun) + "compare/" + step + "/" + tp.id + ".js")) : false;
        const a     = ok1 ? baseRun.cmp.get(step + "/" + bp.id) : null;
        const b     = ok2 ? targetRun.cmp.get(step + "/" + tp.id) : null;
        if (!a || !b) continue;
        if (a.shape.join("x") !== b.shape.join("x")) {
          mismatch++;
          continue;
        }
        const stats = core.compareStats(f64(a.data), f64(b.data));
        if (!stats) {
          mismatch++;
          continue;
        }
        rows.push({name: bp.name, mse: stats.mse, sqnr: stats.sqnr, order: rows.length});
      }
      if (my === token && state.view === "compare") holder.replaceWith(compareTable({rows, mismatch}, targetRun));
    })();
  }
  
  
  function compareTable(res, targetRun) {
    //* Make the table of errors. A header sorts the rows, and a row opens that point in the observe view.
    
    // Sort the rows; the network order is the default, so the error can be read from the input to the output
    const rows   = res.rows.slice();
    const key    = state.sort.key;
    const dir    = state.sort.dir;
    const val    = (row, k) => k === "order" ? row.order : row[k] === null ? -Infinity : row[k];
    const finite = rows.map(r => r.sqnr).filter(v => typeof v === "number" && isFinite(v));
    const maxSq  = Math.max(1, ...finite);
    let wrap     = null;
    if (key) rows.sort((a, b) => (val(a, key) < val(b, key) ? -1 : val(a, key) > val(b, key) ? 1 : 0) * dir || a.order - b.order);
    
    // Make the headers and the rows
    const th   = (k, text, cls) => {
      const sort = () => {
        state.sort = {key: k, dir: key === k ? -dir : 1};
        wrap.replaceWith(compareTable(res, targetRun));
      };
      return h("th", {class: cls || null}, h("button", {"aria-sort": key === k ? (dir > 0 ? "ascending" : "descending") : null, onclick: sort}, text + (key === k ? (dir > 0 ? " ↑" : " ↓") : "")));
    };
    const open = name => {
      state.view    = "observe";
      state.primary = targetRun.folder;
      state.also    = state.also.filter(f => f !== targetRun.folder);
      state.point   = name;
      state.step    = state.cstep;
      writeHash();
      renderRunButton();
      loadAndRender();
    };
    const body = rows.map(r => {
      const width = r.sqnr === null ? 0 : isFinite(r.sqnr) ? Math.max(2, Math.max(0, r.sqnr) / maxSq * 260) : 260;
      const sqnr  = r.sqnr === null ? "なし" : isFinite(r.sqnr) ? [fmt(r.sqnr), h("span", {class: "k"}, " dB")] : "∞";
      const cells = [h("td", {class: "mono"}, r.name), h("td", {class: "num"}, fmt(r.mse)), h("td", {class: "barcell"}, h("span", {class: "bar", style: "width:" + width + "px"}), "  ", sqnr)];
      return h("tr", {tabindex: "0", onclick: () => open(r.name), onkeydown: e => { if (e.key === "Enter") open(r.name); }}, cells);
    });
    const head = h("thead", null, h("tr", null, th("order", "観測点"), th("mse", "平均二乗誤差", "num"), th("sqnr", "SQNR")));
    wrap       = h("div", null, h("table", {class: "ctable"}, head, h("tbody", null, body)), res.mismatch ? h("p", {class: "cmsg", role: "status"}, MSG.m104(res.mismatch)) : null);
    return wrap;
  }
  
  
  // ===== Menus =====
  
  let openPop = null;
  
  
  function popover(anchor, content, align) {
    //* Open a small panel under a button. Clicking outside or pressing Esc closes it.
    //! Calling it again for the same button closes the panel instead, and returns null.
    
    // Close the open panel first
    if (openPop) {
      const same = openPop.anchor === anchor;
      openPop.close();
      if (same) return null;
    }
    
    // Place the panel under the button
    const rc     = anchor.getBoundingClientRect();
    const el     = h("div", {class: "pop"}, content);
    el.style.top = Math.round(rc.bottom + 6) + "px";
    if (align === "left") el.style.left = Math.round(Math.max(8, rc.left)) + "px";
    else el.style.right = Math.round(Math.max(8, innerWidth - rc.right)) + "px";
    document.body.append(el);
    
    // Close it on a click outside
    const onDown = e => { if (!el.contains(e.target) && !anchor.contains(e.target)) close(); };
    const close  = () => {
      el.remove();
      document.removeEventListener("pointerdown", onDown, true);
      anchor.setAttribute("aria-expanded", "false");
      if (openPop && openPop.el === el) openPop = null;
    };
    document.addEventListener("pointerdown", onDown, true);
    anchor.setAttribute("aria-expanded", "true");
    openPop      = {el, close, anchor};
    if (el.querySelector("button")) el.querySelector("button").focus({preventScroll: true});
    return openPop;
  }
  
  
  function openRunPicker(anchor) {
    //* Open the list of runs: a click on the square shows or hides the run, and a click on the name shows only that run (FR-034, U-053).
    //! The last item reads another record folder that the user picks (FR-034, IF-06).
    
    // Read the picked folder: runs.js by hand (to skip runs that are already listed), then run.js and catalog.js of each new run
    const loadFolder = async files => {
      const found  = files.map(f => ({file: f, rel: f.webkitRelativePath || f.name})).filter(o => o.rel.split("/").pop() === "runs.js").sort((a, b) => a.rel.split("/").length - b.rel.split("/").length)[0];
      if (!found) {
        toast(MSG.m103, "alert");
        return;
      }
      const prefix = found.rel.slice(0, found.rel.length - "runs.js".length);
      const src    = {files: new Map(), label: prefix.replace(/\/$/, "").split("/").pop() || "フォルダ"};
      let listed   = [];
      let added    = 0;
      for (const f of files) if ((f.webkitRelativePath || f.name).startsWith(prefix)) src.files.set((f.webkitRelativePath || f.name).slice(prefix.length), f);
      const parsed = core.parseLines(await found.file.text());
      for (const call of parsed.calls) if (call.fn === "runs") listed = listed.concat(call.arg.runs || []);
      badLines    += parsed.bad;
      for (const r of listed) {
        if (!r || !r.folder || store.runs.has(r.folder)) continue;
        const run = newRun(r, src);
        store.runs.set(r.folder, run);
        store.order.push(r.folder);
        await readData(run, runBase(run) + "run.js").catch(() => undefined);
        const ok  = await readData(run, runBase(run) + "catalog.js").then(() => true, () => false);
        run.loads.set("catalog", Promise.resolve(ok));
        run.status.set("catalog", ok ? "ok" : "error");
        added++;
      }
      
      // Tell what happened
      flushBad();
      if (!listed.length) toast(MSG.m103, "alert");
      else if (!added) toast("この記録フォルダの実行は、すでに一覧にあります");
      else toast(added + " 個の実行を読み込みました");
      renderRunButton();
      if (state.view === "compare") renderMain();
    };
    
    // Make the rows of the runs
    let pop     = null;
    const build = () => {
      const box   = h("div", {role: "listbox", "aria-label": "実行"});
      const shown = shownRuns();
      for (const folder of store.order) {
        const run     = store.runs.get(folder);
        const idx     = shown.indexOf(folder);
        const on      = idx >= 0;
        const only    = on && shown.length === 1;
        
        // Apply a click with runsClick (U-053): the tree follows the main run, and the charts follow the shown runs
        const pick    = part => {
          const next    = core.runsClick(store.order, shown, state.primary, folder, part);
          const moved   = next.primary !== state.primary;
          if (!moved && next.shown.join() === shown.join()) return;
          state.primary = next.primary;
          state.also    = next.shown.filter(f => f !== next.primary);
          writeHash();
          renderRunButton();
          if (moved) renderTree();
          if (state.view === "observe") loadAndRender();
        };
        
        // The square shows or hides the run and keeps the list open; the name shows only the run and closes the list
        const overlay = () => {
          pick("box");
          if (pop) pop.el.replaceChildren(build());
        };
        const choose  = () => {
          if (pop) pop.close();
          pick("name");
        };
        const full    = !on && shown.length >= core.RUNS_MAX;
        const off     = only || full ? "true" : null;
        const tip     = only ? "表示している実行が1つだけなので外せません" : full ? "同時に表示できる実行は " + core.RUNS_MAX + " 個までです" : on ? "外す" : "表示する";
        const square  = h("button", {class: "sw" + (on ? " on" : ""), style: on ? "--c:" + runColor(idx) : null, "aria-pressed": String(on), "aria-disabled": off, "aria-label": run.name + " を表示する", title: tip, onclick: overlay});
        const parts   = [h("span", {class: "nm"}, run.name), h("span", {class: "dt"}, shortDate(run.started)), run.src ? h("span", {class: "src"}, run.src.label) : null];
        const label   = h("button", {class: "rn2" + (folder === state.primary ? " primary" : ""), role: "option", "aria-selected": folder === state.primary ? "true" : "false", title: "この実行だけを表示", onclick: choose}, parts);
        box.append(h("div", {class: "runrow"}, square, label));
      }
      
      // The last item picks another record folder
      const pickFolder = () => {
        const input = h("input", {type: "file", webkitdirectory: true, multiple: true, style: "display:none"});
        if (pop) pop.close();
        input.addEventListener("change", () => {
          loadFolder(Array.from(input.files || []));
          input.remove();
        });
        document.body.append(input);
        input.click();
      };
      box.append(h("hr"), h("button", {class: "menuitem", onclick: pickFolder}, "ほかの記録フォルダを読み込む"));
      return box;
    };
    pop = popover(anchor, build(), "left");
  }
  
  
  function openExport(anchor) {
    //* Open the export menu: HTML of the chosen point and step (FR-024, EXT-03), or CSV of the main run (FR-025, EXT-02).
    
    // CSV: one row for each point and step, UTF-8 with BOM and CRLF, so that Excel opens it correctly
    const csv = async () => {
      const run = primaryRun();
      if (pop) pop.close();
      if (!run) return;
      await ensureCatalog(run);
      for (const p of run.points.values()) await ensurePoint(run, p.id);
      download("tensight_" + run.folder + ".csv", core.csvBuild(run.folder, [...run.points.values()], p => run.recs.get(p.id) || new Map()), "text/csv;charset=utf-8");
    };
    
    // HTML: the viewer (from viewer/bundle.js) and the records of the point in the shown runs in one file, without other files (BUG-018)
    const html = async () => {
      const run   = primaryRun();
      const p     = currentPoint();
      if (pop) pop.close();
      if (!run || !p || !run.recs.get(p.id)) return;
      if (!bundle) await loadScript("viewer/bundle.js").catch(() => undefined);
      if (!bundle) {
        toast("書き出しに使うファイル viewer/bundle.js を読めませんでした", "alert");
        return;
      }
      const esc   = s => s.replace(/<\//g, "<\\/");
      const call  = (fn, o) => "tensight." + fn + "(" + esc(JSON.stringify(o)) + ");";
      const tag   = "scr" + "ipt";
      const block = text => "<" + tag + ">\n" + text + "\n</" + tag + ">\n";
      const runs  = shownRuns().map(f => store.runs.get(f)).filter(r => recsOf(r, p));
      const lines = [call("runs", {format: 1, runs: runs.map(r => ({folder: r.folder, name: r.name, started: r.started}))})];
      const embed = {run: run.folder, also: runs.map(r => r.folder).filter(f => f !== run.folder), hidden: state.hidden, point: p.name, step: state.step};
      const opts  = {embedded: embed, ui: {palette: ui.palette, cmap: ui.cmap, theme: ui.theme, closed: ui.closed}};
      for (const r of runs) {
        const q = r.byName.get(p.name);
        if (r.info) lines.push(call("run", r.info));
        lines.push(call("point", q));
        for (const a of r.alerts.get(q.id) || []) lines.push(call("alert", a));
        for (const s of sortedSteps(r.recs.get(q.id))) lines.push(call("rec", r.recs.get(q.id).get(s)));
      }
      const title = (p.name + " " + state.step).replace(/[<>&"]/g, "");
      const head  = "<!doctype html>\n<html lang=\"ja\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>" + title + "</title>\n";
      const body  = "<style>\n" + bundle.css + "\n</style>\n</head>\n<body>\n<div id=\"app\"></div>\n" + bundle.js.map(js => block(esc(js))).join("") + block(lines.join("\n") + "\ntensight.start(" + esc(JSON.stringify(opts)) + ");");
      download("tensight_" + run.folder + "_" + p.id + "_step" + state.step + ".html", head + body + "</body>\n</html>\n", "text/html;charset=utf-8");
    };
    
    // Open the menu
    let pop    = null;
    const menu = h("div", {role: "menu"}, h("button", {class: "menuitem", role: "menuitem", onclick: html}, "HTML に書き出す"), h("button", {class: "menuitem", role: "menuitem", onclick: csv}, "CSV に書き出す"));
    pop        = popover(anchor, menu, "right");
  }
  
  
  function openPalette(anchor) {
    //* Open the palette menu: palettes for the page (with their colormaps) and colormaps for the charts (CR-004).
    
    let pop     = null;
    const build = () => {
      const t     = theme();
      const pals  = PALETTES.map(pal => {
        const c    = pal[t];
        const sel  = pal.id === palette().id;
        const pick = () => {
          ui.palette = pal.id;
          ui.cmap    = "";
          saveUi();
          recolor();
          if (pop) pop.el.replaceChildren(build());
        };
        const sw   = h("span", {class: "pal-sw"}, [c.bg, c.surface, c.signal, c.caution, c.alert].map(x => h("i", {style: "background:" + x})));
        const ramp = h("span", {class: "pal-ramp", style: "background:" + rampCss(pal.cmap, pal, t)});
        return h("button", {class: "pal" + (sel ? " sel" : ""), role: "radio", "aria-checked": sel ? "true" : "false", onclick: pick}, sw, h("span", {class: "pal-name"}, pal.name), ramp);
      });
      const maps  = CMAPS.map(name => {
        const sel  = name === cmapName();
        const pick = () => {
          ui.cmap = name;
          saveUi();
          recolor();
          if (pop) pop.el.replaceChildren(build());
        };
        const ramp = h("span", {class: "cm-ramp", style: "background:" + rampCss(name, palette(), t)});
        return h("button", {class: "cm" + (sel ? " sel" : ""), role: "radio", "aria-checked": sel ? "true" : "false", onclick: pick}, ramp, h("span", null, name === "accent" ? "アクセント" : name));
      });
      const upper = [h("div", {class: "pop-title"}, "配色"), h("div", {role: "radiogroup", "aria-label": "配色"}, pals)];
      const lower = [h("div", {class: "pop-title"}, "図の色"), h("div", {class: "cmaps", role: "radiogroup", "aria-label": "図の色"}, maps)];
      return h("div", {style: "width:340px"}, upper, lower);
    };
    pop         = popover(anchor, build(), "right");
  }
  
  
  function recolor() {
    //* Apply new colors to the page and draw everything again.
    
    applyColors();
    renderRunButton();
    renderMain();
  }
  
  
  // ===== Start =====
  
  api.start = async function (opts) {
    //* Start the viewer. opts.embedded is set in an exported HTML file, which already holds its data.
    
    // Read the saved width of the list and the fonts, and apply the colors
    const options = opts || {};
    const cs      = getComputedStyle(document.documentElement);
    fonts         = {sans: cs.getPropertyValue("--sans").trim() || "sans-serif", mono: cs.getPropertyValue("--mono").trim() || "monospace"};
    if (options.ui && !storedUi) Object.assign(ui, options.ui);
    try {
      const side = parseInt(localStorage.getItem("tensight.side") || "", 10);
      if (side >= 200 && side <= 480) document.documentElement.style.setProperty("--side-w", side + "px");
    } catch (e) {
      // Storage is blocked; the default width stays
    }
    applyColors();
    
    // Build the frame: the list on the left and the content on the right
    const app     = $("app") || document.body.appendChild(h("div", {id: "app"}));
    const runbtn  = h("button", {class: "runbtn", id: "runbtn", "aria-haspopup": "listbox", "aria-expanded": "false", onclick: e => openRunPicker(e.currentTarget)});
    const filter  = h("input", {id: "filter", type: "search", placeholder: "絞り込み", autocomplete: "off", spellcheck: "false", "aria-label": "観測点を名前で絞り込む", title: "絞り込み（/）", oninput: e => { state.filter = e.target.value; renderTree(); }});
    const resizer = h("div", {class: "resizer", id: "resizer", role: "separator", tabindex: "0", "aria-orientation": "vertical", "aria-label": "一覧の幅"});
    const side    = h("aside", {class: "side", id: "side"}, h("div", {class: "side-top"}, runbtn), h("div", {class: "filter"}, filter), h("nav", {class: "tree", id: "tree", role: "tree", "aria-label": "観測点"}), resizer);
    app.className = "app";
    app.append(side, h("main", {class: "main"}, h("div", {class: "content", id: "content"})));
    document.body.append(h("div", {class: "tip", id: "tip", role: "tooltip"}), h("div", {class: "toasts", id: "toasts", "aria-live": "polite"}));
    
    // Keys: / for the filter, up and down for the list, left and right for the step, Esc to clear (02b, chapter 3)
    document.addEventListener("keydown", e => {
      const t      = e.target;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
      if (e.key === "Escape") {
        if (openPop) openPop.close();
        else if (filter.value) {
          filter.value = "";
          state.filter = "";
          renderTree();
        } else if (typing) t.blur();
        closeSide();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey || state.view !== "observe" || openPop) return;
      if (e.key === "/" && !typing && !state.embedded) {
        e.preventDefault();
        app.classList.add("side-open");
        filter.focus();
        filter.select();
      } else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && (!typing || t === filter) && !state.embedded) {
        const rows = [...$("tree").querySelectorAll(".row.has")];
        const idx  = rows.findIndex(r => r.classList.contains("sel"));
        e.preventDefault();
        if (rows.length) rows[Math.max(0, Math.min(rows.length - 1, idx < 0 ? 0 : idx + (e.key === "ArrowDown" ? 1 : -1)))].click();
      } else if (e.key === " " && !typing && t.tagName !== "BUTTON" && view.steps) {
        e.preventDefault();
        togglePlay();
      } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && !typing && t !== resizer && view.steps) {
        const i = view.steps.indexOf(state.step) + (e.key === "ArrowRight" ? 1 : -1);
        e.preventDefault();
        stopPlay();
        setStep(view.steps[Math.max(0, Math.min(view.steps.length - 1, i))]);
      }
    });
    
    // The list can be made wider or narrower, between 200 px and 480 px
    const setWidth = px => {
      const v = Math.max(200, Math.min(480, Math.round(px)));
      document.documentElement.style.setProperty("--side-w", v + "px");
      try {
        localStorage.setItem("tensight.side", String(v));
      } catch (err) {
        // Storage is blocked; the width lasts until the page closes
      }
    };
    resizer.addEventListener("pointerdown", e => {
      const start = e.clientX;
      const width = $("side").getBoundingClientRect().width;
      const move  = ev => setWidth(width + ev.clientX - start);
      const up    = () => {
        resizer.removeEventListener("pointermove", move);
        resizer.removeEventListener("pointerup"  , up  );
      };
      e.preventDefault();
      resizer.setPointerCapture(e.pointerId);
      resizer.addEventListener("pointermove", move);
      resizer.addEventListener("pointerup"  , up  );
    });
    resizer.addEventListener("keydown", e => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      setWidth($("side").getBoundingClientRect().width + (e.key === "ArrowRight" ? 16 : -16));
    });
    
    // Draw again when the size or the theme of the OS changes
    let frame = 0;
    new ResizeObserver(() => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; drawAll(); }); }).observe($("content"));
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!ui.theme) recolor(); });
    
    // An exported HTML file shows its one point in the runs that were shown, with the runs that the groups hid (BUG-018)
    if (options.embedded) {
      const e        = options.embedded;
      state.embedded = true;
      state.loading  = false;
      state.primary  = e.run;
      state.also     = Array.isArray(e.also) ? e.also.filter(f => store.runs.has(f) && f !== e.run).slice(0, core.RUNS_MAX - 1) : [];
      state.point    = e.point;
      state.step     = e.step;
      for (const key of Object.keys(state.hidden)) state.hidden[key] = e.hidden && Array.isArray(e.hidden[key]) ? e.hidden[key].filter(f => typeof f === "string") : [];
      app.classList.add("embedded");
      for (const f of [e.run, ...state.also]) {
        const run = store.runs.get(f);
        const p   = run ? run.byName.get(e.point) : null;
        if (run) run.loads.set("catalog", Promise.resolve(true));
        if (run && p) run.loads.set("p" + p.id, Promise.resolve(true));
      }
      renderMain();
      return;
    }
    
    // Read the state from the URL (FR-023), and read it again when the part after # changes (U-046)
    readHash();
    addEventListener("hashchange", () => {
      readHash();
      showHash();
    });
    
    // Load the run list and the information of every run, and show the state of the URL
    renderMain();
    await loadScript("runs.js").catch(() => undefined);
    state.loading = false;
    for (const f of store.order) await once(store.runs.get(f), "info", () => readData(store.runs.get(f), runBase(store.runs.get(f)) + "run.js"));
    await showHash();
  };
  
  
  function readHash() {
    //* Put the state after # in the URL into the state of the page (FR-023, U-046).
    
    const q       = new URLSearchParams(location.hash.slice(1));
    const int     = k => { const v = parseInt(q.get(k) || "", 10); return isNaN(v) ? null : v; };
    state.primary = q.get("run") || "";
    state.also    = (q.get("also") || "").split(",").filter(Boolean);
    state.point   = q.get("point") || "";
    state.step    = int("step");
    state.metric  = q.get("metric") || "mean";
    state.hist    = q.get("hist") === "code" ? "code" : "value";
    state.view    = q.get("view") === "compare" ? "compare" : "observe";
    state.base    = q.get("base") || "";
    state.target  = q.get("target") || "";
    state.cstep   = int("cstep");
  }
  
  
  async function showHash() {
    //* Show the state read from the URL: the newest run is the default, and runs that are not loaded are left out.
    
    if (state.loading) return;
    if (!store.runs.has(state.primary)) state.primary = store.order.length ? store.order[store.order.length - 1] : "";
    state.also = state.also.filter(f => store.runs.has(f) && f !== state.primary).slice(0, core.RUNS_MAX - 1);
    renderRunButton();
    if (state.view === "observe" && store.order.length) {
      await loadAndRender();
      return;
    }
    renderMain();
    if (primaryRun()) await ensureCatalog(primaryRun());
    renderTree();
  }
})();
