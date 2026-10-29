// @file tensight_core.js
// @brief Functions of the viewer that do not use the DOM (03 U-037 to U-044, U-053, U-054); Node.js tests them
// @author Kohei Yamaguchi, Claude
//
// The page loads this file with a classic script tag before tensight.js (ADR-001).
// In Node.js, require() returns the same functions (module.exports).

(function (root) {
  "use strict";
  
  // ===== Settings =====
  
  const GUT_L    = 64;   // space on the left of every step chart (02b 2.4)
  const GUT_R    = 76;   // space on the right of every step chart (02b 2.4)
  const AXES     = {batch: "バッチ", channel: "チャネル", length: "長さ", depth: "奥行き", height: "高さ", width: "幅", token: "トークン", dim: "次元", head: "ヘッド", query: "クエリ", key: "キー", out: "出力", in: "入力", row: "行", col: "列", position: "位置", "...": "位置"};
  const SQUARE   = ["conv2d_image", "attention_qk", "matrix"];
  const CSV_HEAD = ["実行", "観測点", "ステップ", "データ型", "形", "要素数", "最小値", "最大値", "平均", "標準偏差", "L2ノルム", "0の割合", "NaNの数", "Infの数", "クリップ率", "平均二乗誤差", "最大絶対誤差", "SQNR", "境界付近の割合", "コードが変わった割合", "行き来した割合", "更新率"];
  const CMAPS    = ["accent", "viridis", "cividis", "magma", "inferno", "plasma", "turbo", "rocket", "mako", "batlow"];   // "accent" comes from the palette; the others are tables of colormaps.js
  const cellsOf  = new WeakMap();
  const RUN_GAP  = 0.08;   // the smallest OKLab distance between two run colors (02b 2.3.3, CR-016)
  const RUNS_MAX = 5;      // the most runs that are shown at once: the number of run colors (FR-034, CR-020)
  
  // Palettes for the page and the charts (CR-004). Each one has colors for the light and the dark theme.
  const PALETTES = [
    {
      id   : "std",
      name : "標準",
      cmap : "accent",
      light: {bg: "#F4F6F8", surface: "#FFFFFF", text: "#15181C", weak: "#667085", line: "#DDE2E8", signal: "#0B7A75", caution: "#B86E00", alert: "#D23B3B", runs: ["#0B7A75", "#5B4BDB", "#D9731A", "#2F6FB5", "#B83280"]},
      dark : {bg: "#0E1114", surface: "#151A1F", text: "#E7EBEF", weak: "#93A0AD", line: "#26303A", signal: "#3CC7BE", caution: "#F0A93B", alert: "#FF6B6B", runs: ["#3CC7BE", "#9488F5", "#F29B4F", "#8AB4F8", "#E879B9"]},
    },
    {
      id   : "viridis",
      name : "Viridis",
      cmap : "viridis",
      light: {bg: "#F3F5F7", surface: "#FFFFFF", text: "#161A22", weak: "#5E6676", line: "#DCE1E7", signal: "#31688E", caution: "#B86E00", alert: "#D23B3B", runs: ["#31688E", "#440154", "#1F8A70", "#5C8E1E", "#8C6BB1"]},
      dark : {bg: "#0D1016", surface: "#141922", text: "#E5E9EF", weak: "#949DAD", line: "#252C38", signal: "#5EC962", caution: "#F0A93B", alert: "#FF6B6B", runs: ["#5EC962", "#29AF7F", "#FDE725", "#6E9BD6", "#B08AE0"]},
    },
    {
      id   : "inferno",
      name : "Inferno",
      cmap : "inferno",
      light: {bg: "#F7F4F2", surface: "#FFFFFF", text: "#1F1416", weak: "#6F6164", line: "#E8DEDC", signal: "#6A176E", caution: "#C25E00", alert: "#C8102E", runs: ["#6A176E", "#2C5F9E", "#D86A00", "#9E2A57", "#3E7D3A"]},
      dark : {bg: "#0F0B0C", surface: "#181113", text: "#F1E9E6", weak: "#A99B9C", line: "#2E2325", signal: "#F7D13D", caution: "#F57D15", alert: "#FF4F5E", runs: ["#F7D13D", "#7EB6F2", "#F57D15", "#E5689A", "#94CF73"]},
    },
    {
      id   : "okabe",
      name : "Okabe–Ito",
      cmap : "cividis",
      light: {bg: "#F4F6F8", surface: "#FFFFFF", text: "#15181C", weak: "#667085", line: "#DDE2E8", signal: "#0072B2", caution: "#B87A00", alert: "#D55E00", runs: ["#0072B2", "#009E73", "#CC79A7", "#15181C", "#56B4E9"]},
      dark : {bg: "#0E1114", surface: "#151A1F", text: "#E7EBEF", weak: "#93A0AD", line: "#26303A", signal: "#56B4E9", caution: "#E69F00", alert: "#FF7A3D", runs: ["#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#E69F00"]},
    },
    {
      id   : "nord",
      name : "Nord",
      cmap : "accent",
      light: {bg: "#E5E9F0", surface: "#F2F4F8", text: "#2E3440", weak: "#4C566A", line: "#D8DEE9", signal: "#5E81AC", caution: "#B26B4F", alert: "#BF616A", runs: ["#5E81AC", "#8E6C89", "#5F7F4B", "#B26B4F", "#4C566A"]},
      dark : {bg: "#2E3440", surface: "#3B4252", text: "#ECEFF4", weak: "#AEB7C6", line: "#4C566A", signal: "#88C0D0", caution: "#EBCB8B", alert: "#D9838B", runs: ["#88C0D0", "#B48EAD", "#A3BE8C", "#D08770", "#81A1C1"]},
    },
    {
      id   : "catppuccin",
      name : "Catppuccin",
      cmap : "accent",
      light: {bg: "#E6E9EF", surface: "#EFF1F5", text: "#4C4F69", weak: "#5C5F77", line: "#CCD0DA", signal: "#8839EF", caution: "#B9730F", alert: "#D20F39", runs: ["#8839EF", "#1E66F5", "#40A02B", "#C8520A", "#B9479A"]},
      dark : {bg: "#181825", surface: "#1E1E2E", text: "#CDD6F4", weak: "#A6ADC8", line: "#313244", signal: "#CBA6F7", caution: "#F9E2AF", alert: "#F38BA8", runs: ["#CBA6F7", "#89B4FA", "#A6E3A1", "#FAB387", "#F5C2E7"]},
    },
  ];
  
  
  // ===== Numbers and ticks (U-037) =====
  
  function fmt(x) {
    //* Show a value with 4 significant digits (02b 3章). A missing value is "なし".
    
    if (x === null || x === undefined) return "なし";
    if (x === "inf" || x === Infinity) return "∞";
    if (x === "-inf" || x === -Infinity) return "-∞";
    if (x === "nan" || (typeof x === "number" && Number.isNaN(x))) return "NaN";
    if (typeof x !== "number") return String(x);
    if (x === 0) return "0";
    return (Math.abs(x) < 1e-3 ? x.toExponential(3) : x.toPrecision(4)).replace("e+", "e");
  }
  
  
  function fmtPct(x) {
    //* Show a share between 0 and 1 as a percentage with 4 significant digits.
    
    if (typeof x !== "number") return fmt(x);
    if (x === 0) return "0%";
    return fmt(x * 100) + "%";
  }
  
  
  function fmtInt(x) {
    //* Show a count with commas.
    
    return typeof x === "number" ? x.toLocaleString("en-US") : fmt(x);
  }
  
  
  function niceTicks(lo, hi, count) {
    //* Return the multiples of the smallest step of 1, 2, 5 x 10^k that is not smaller than (hi - lo) / count, from lo up to hi.
    
    // Choose the step
    if (!(hi > lo)) return [lo];
    const raw  = (hi - lo) / Math.max(1, count);
    const mag  = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(f => f * mag).find(v => v >= raw * (1 - 1e-12));
    
    // List the multiples of the step between lo and hi
    const out  = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
    return out;
  }
  
  
  function stepNearest(steps, s) {
    //* Return the step of the sorted list that is nearest to s; the smaller one wins a tie. An empty list gives null.
    
    let best = steps.length ? steps[0] : null;
    for (const t of steps) if (Math.abs(t - s) < Math.abs(best - s)) best = t;
    return best;
  }
  
  
  function stepX(s, first, last, width) {
    //* Return the x position of step s in a chart of the given width, with the shared side spaces (02b 2.4).
    
    const frac = last === first ? 0.5 : (s - first) / (last - first);
    return GUT_L + frac * (width - GUT_L - GUT_R);
  }
  
  
  // ===== Colors (U-038) =====
  
  function hexRgb(hex) {
    //* Turn "#RRGGBB" into [r, g, b].
    
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  
  
  function rgbHex(rgb) {
    //* Turn [r, g, b] into "#RRGGBB".
    
    return "#" + rgb.map(v => v.toString(16).padStart(2, "0")).join("");
  }
  
  
  function oklab(hex) {
    //* Turn a color into OKLab, where equal steps look equal to the eye.
    
    const [r, g, b] = hexRgb(hex).map(c => { const v = c / 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
    const l         = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m         = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s         = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
  }
  
  
  function mix(hexA, hexB, t) {
    //* Mix two colors in OKLab and return [r, g, b]. t = 0 gives the first color, and t = 1 the second.
    
    // Mix in OKLab, then go back to linear RGB
    const a = oklab(hexA);
    const b = oklab(hexB);
    const L = a[0] + (b[0] - a[0]) * t;
    const A = a[1] + (b[1] - a[1]) * t;
    const B = a[2] + (b[2] - a[2]) * t;
    const l = Math.pow(L + 0.3963377774 * A + 0.2158037573 * B, 3);
    const m = Math.pow(L - 0.1055613458 * A - 0.0638541728 * B, 3);
    const s = Math.pow(L - 0.0894841775 * A - 1.2914855480 * B, 3);
    
    // Turn linear RGB into bytes
    const byte = c => Math.max(0, Math.min(255, Math.round((c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(Math.max(0, c), 1 / 2.4) - 0.055) * 255)));
    return [byte(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s), byte(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s), byte(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)];
  }
  
  
  function contrast(a, b) {
    //* Return the contrast ratio of two colors, from their relative luminance (WCAG 2).
    
    const lum = hex => hexRgb(hex).map(v => v / 255).map(v => v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)).reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i], 0);
    return (Math.max(lum(a), lum(b)) + 0.05) / (Math.min(lum(a), lum(b)) + 0.05);
  }
  
  
  function makeLut(table, c, t) {
    //* Make a colormap of 256 colors as r, g, b bytes, from a table of colormaps.js (256 colors written as RRGGBB).
    //! A table of null gives the accent ramp of the palette c and the theme t: from a light tint of the surface, through the signal color, to a stronger one.
    
    // Read the table
    const out = new Uint8Array(768);
    if (table) {
      for (let i = 0; i < 768; i++) out[i] = parseInt(table.substr(i * 2, 2), 16);
      return out;
    }
    
    // Make the accent ramp
    const lo = rgbHex(mix(c.surface, c.signal, t === "dark" ? 0.24 : 0.16));
    const hi = rgbHex(mix(c.signal, c.text, 0.55));
    for (let i = 0; i < 256; i++) {
      const x = i / 255;
      out.set(x < 0.55 ? mix(lo, c.signal, x / 0.55) : mix(c.signal, hi, (x - 0.55) / 0.45), i * 3);
    }
    return out;
  }
  
  
  function cmap(lut, x) {
    //* Return the color [r, g, b] for x from a colormap of makeLut(). x below 0 or above 1 gives the color of the end.
    
    const i = Math.max(0, Math.min(255, Math.round(x * 255))) * 3;
    return [lut[i], lut[i + 1], lut[i + 2]];
  }
  
  
  function figureColors(lut, c, t, own) {
    //* Choose the colors of the charts: one main color and RUNS_MAX colors for the shown runs (02b 2.3.3, CR-004, CR-016).
    //! own means that the colormap is the palette's own one or accent: then the colors of the palette c are used.
    //! A color of the colormap is used only when its contrast with the surface is 3 or more, and its hue is 30 degrees or more from the alert color and 20 degrees or more from the caution color (grays pass).
    //! A run color of the colormap is RUN_GAP or more apart (OKLab) from the run colors chosen before it; the run colors of the palette fill the rest as they are, so a few of them may not keep the rule above (CR-016).
    
    // The colors of the palette already fit its own colormap
    if (own) return {main: c.signal, runs: c.runs};
    
    // Keep the colors of the colormap that are easy to see and are not taken for the alert or the caution color
    const hue   = hex => { const o = oklab(hex); return {h: Math.atan2(o[2], o[1]) * 180 / Math.PI, chroma: Math.hypot(o[1], o[2])}; };
    const apart = (a, b) => { const d = Math.abs(a.h - b.h) % 360; return Math.min(d, 360 - d); };
    const fits  = hex => { const hc = hue(hex); return contrast(hex, c.surface) >= 3 && (hc.chroma < 0.04 || (apart(hc, hue(c.alert)) >= 30 && apart(hc, hue(c.caution)) >= 20)); };
    const far   = (hex, runs) => runs.every(q => { const a = oklab(q), b = oklab(hex); return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) >= RUN_GAP; });
    const cands = [];
    for (let i = 3; i <= 19; i++) {
      const hex = rgbHex(cmap(lut, i / 20));
      if (fits(hex)) cands.push({x: i / 20, hex});
    }
    
    // Without any, use the signal and the run colors of the palette that follow the same rule (BUG-001), then the other run colors of the palette (CR-016)
    if (!cands.length) {
      const kept = c.runs.filter((hex, i) => i === 0 || fits(hex));
      return {main: c.signal, runs: kept.concat(c.runs.filter(hex => !kept.includes(hex))).slice(0, RUNS_MAX)};
    }
    
    // The main color is a darker one on a light theme and a lighter one on a dark theme; the other runs spread over the colormap, RUN_GAP or more apart
    const want   = t === "dark" ? 0.75 : 0.35;
    const chosen = [cands.reduce((best, o) => Math.abs(o.x - want) < Math.abs(best.x - want) ? o : best)];
    const gapTo  = o => Math.min(...chosen.map(q => Math.abs(q.x - o.x)));
    while (chosen.length < RUNS_MAX) {
      const rest = cands.filter(o => far(o.hex, chosen.map(q => q.hex)));
      if (!rest.length) break;
      chosen.push(rest.reduce((best, o) => gapTo(o) > gapTo(best) ? o : best));
    }
    
    // Fill the rest with the run colors of the palette: the ones RUN_GAP or more apart that keep the rule above, the other ones RUN_GAP or more apart, then the rest (CR-016)
    const runs = chosen.map(o => o.hex);
    for (const hex of c.runs) if (runs.length < RUNS_MAX && fits(hex) && far(hex, runs)) runs.push(hex);
    for (const hex of c.runs) if (runs.length < RUNS_MAX && far(hex, runs)) runs.push(hex);
    for (const hex of c.runs) if (runs.length < RUNS_MAX && !runs.includes(hex)) runs.push(hex);
    return {main: runs[0], runs};
  }
  
  
  // ===== Lines of the record files (U-039) =====
  
  function parseLines(text) {
    //* Turn the text of a record file into its calls (IF-06). Lines that cannot be read are counted, and lines of another format are skipped.
    
    const calls = [];
    let bad     = 0;
    for (const raw of String(text).split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      const m    = /^tensight\.([A-Za-z]+)\((.*)\);?$/.exec(line);
      let arg    = null;
      if (m) {
        try {
          arg = JSON.parse(m[2]);
        } catch (e) {
          arg = null;
        }
      }
      if (!arg || typeof arg !== "object") bad++;
      else if (arg.format === 1) calls.push({fn: m[1], arg});
    }
    return {calls, bad};
  }
  
  
  // ===== Tree of points (U-040) =====
  
  function treeRows(points, filter, pathOf, order) {
    //* Return the rows of the list: the layers nested by their names, each with its points (FR-018, FR-033).
    //! Only the points whose names include filter (without telling capital letters apart) are kept, with the layers above them.
    
    // Put each point into its layer
    const text = String(filter || "").toLowerCase();
    const top  = {children: new Map(), points: []};
    for (const p of points) {
      if (text && !p.name.toLowerCase().includes(text)) continue;
      let node = top;
      let acc  = "";
      for (const part of pathOf(p).split(".")) {
        acc  = acc ? acc + "." + part : part;
        if (!node.children.has(part)) node.children.set(part, {name: part, path: acc, children: new Map(), points: []});
        node = node.children.get(part);
      }
      node.points.push(p);
    }
    
    // Walk the layers in the order they came
    const rows = [];
    const walk = (node, depth) => {
      for (const child of node.children.values()) {
        child.points.sort(order);
        rows.push({node: child, depth});
        walk(child, depth + 1);
      }
    };
    walk(top, 0);
    return rows;
  }
  
  
  // ===== Maps (U-041) =====
  
  function mapCells(map) {
    //* Return the cells of a map as bytes (0 to 255 means 0 to map.max); the bytes of one map are made once.
    
    if (!cellsOf.has(map)) {
      const bin   = atob(map.data);
      const cells = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) cells[i] = bin.charCodeAt(i);
      cellsOf.set(map, cells);
    }
    return cellsOf.get(map);
  }
  
  
  function mapValue(map, r, c) {
    //* Return the value of one cell: byte / 255 x max (02c).
    
    return mapCells(map)[r * map.cols + c] / 255 * map.max;
  }
  
  
  function mapSpan(i, n, size) {
    //* Return the range of indices that cell i of n covers, as "a" or "a〜b" (the same groups as map_build of the recorder).
    
    const a = Math.floor(i * size / n);
    const b = Math.max(a, Math.floor((i + 1) * size / n) - 1);
    return a === b ? String(a) : a + "〜" + b;
  }
  
  
  function axisName(names) {
    //* Return the name of an axis: known names in Japanese, joined with "×" when the axis holds more than one (02b 2.4).
    
    return [].concat(names).map(name => AXES[name] || name).join("×");
  }
  
  
  function mapSquare(layout) {
    //* Return whether the cells of a map with this layout are square (02b 2.4).
    
    return SQUARE.includes(layout);
  }
  
  
  function mapFrac(v, max) {
    //* Return where a value of a map sits on its color and height scale, from 0 to 1: v / max (FR-036).
    //! Values above max are 1. A value or a max that is not positive gives 0. The scale is linear only (CR-022).
    
    if (!(max > 0) || !(v > 0)) return 0;
    return Math.min(1, v / max);
  }
  
  
  function mapTicks(max) {
    //* Return the labels of a color bar or of the heights as {frac, value}: max, half and 0 (FR-036).
    
    return [{frac: 1, value: max}, {frac: 0.5, value: max / 2}, {frac: 0, value: 0}];
  }
  
  
  function mapPoint(i, j, rows, cols) {
    //* Return where the center of the cell (row i, column j) sits on the floor of the 3D view: x goes right with the columns and y goes down with the rows, each from -0.5 to 0.5 (BUG-012).
    //! Seen from straight above with the columns to the right, row 0 is at the top, as in the top view. i and j may be halfway between cells.
    
    return [j / (cols - 1) - 0.5, 0.5 - i / (rows - 1)];
  }
  
  
  // ===== Comparison (U-042) =====
  
  function decodeFloat32(text) {
    //* Turn base64 of little-endian float32 values into a Float32Array (02c compare files).
    
    const bin = atob(text);
    const dv  = new DataView(new ArrayBuffer(bin.length));
    const out = new Float32Array(bin.length >> 2);
    for (let i = 0; i < bin.length; i++) dv.setUint8(i, bin.charCodeAt(i));
    for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
    return out;
  }
  
  
  function compareStats(a, b) {
    //* Return the mean squared error and the SQNR of b against a, in float64 (FR-016). Different lengths and empty arrays give null.
    
    if (a.length !== b.length || a.length === 0) return null;
    let se = 0;
    let ss = 0;
    for (let i = 0; i < a.length; i++) {
      se += (b[i] - a[i]) * (b[i] - a[i]);
      ss += a[i] * a[i];
    }
    return {mse: se / a.length, sqnr: se === 0 ? (ss === 0 ? null : Infinity) : 10 * Math.log10(ss / se)};
  }
  
  
  // ===== CSV (U-043) =====
  
  function csvBuild(folder, points, recordsOf) {
    //* Make the CSV of one run: 22 columns, one row for each point and step, UTF-8 with BOM and CRLF (02c EXT-02).
    
    const cell = v => {
      if (v === null || v === undefined) return "";
      const s = String(v);
      return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const rows = [CSV_HEAD.map(cell).join(",")];
    for (const p of points) {
      const m = recordsOf(p);
      for (const s of [...m.keys()].sort((x, y) => x - y)) {
        const r = m.get(s);
        const q = r.quant || {};
        const b = r.boundary || {};
        rows.push([folder, p.name, r.step, r.dtype, (r.shape || []).join("x"), r.numel, r.min, r.max, r.mean, r.std, r.l2, r.zero, r.nan, r.inf, q.clip, q.mse, q.max_abs, q.sqnr, b.near, b.changed, b.back, r.ratio].map(cell).join(","));
      }
    }
    return "﻿" + rows.join("\r\n") + "\r\n";
  }
  
  
  // ===== Name patterns (U-044) =====
  
  function globRegExp(pattern) {
    //* Turn a name pattern into a regular expression: "*" is any text, "?" is one character, and the rest stays as it is (U-013).
    
    return new RegExp("^" + String(pattern).replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".") + "$");
  }
  
  
  function pointFirst(maps, names) {
    //* Return the first point that the map patterns of run.js match, in the order of the patterns, or null.
    
    const keys = Array.isArray(maps) ? maps : maps && typeof maps === "object" ? Object.keys(maps) : [];
    for (const key of keys) {
      const re    = globRegExp(key);
      const found = names.find(name => re.test(name));
      if (found) return found;
    }
    return null;
  }
  
  
  // ===== Runs (U-053) =====
  
  function runsClick(order, shown, primary, folder, part) {
    //* Return the main run and the shown runs after a click on a row of the list of runs (FR-034, BUG-010): the square ("box") shows or hides the run, and the name ("name") shows only that run.
    //! One run is always shown, so the square of the only shown run does nothing. Hiding the main run makes the first other shown run in the list the main run.
    //! At most RUNS_MAX runs are shown, so the square of a run that is not shown does nothing while RUNS_MAX runs are shown (CR-020).
    
    // A name shows only that run
    if (part === "name") return {primary: folder, shown: [folder]};
    
    // A square adds the run while fewer than RUNS_MAX runs are shown, or takes it away while another run stays shown; the shown runs keep the order of the list
    const on   = shown.includes(folder);
    if ((on && shown.length === 1) || (!on && shown.length >= RUNS_MAX)) return {primary, shown: [...shown]};
    const next = order.filter(f => f === folder ? !on : shown.includes(f));
    return {primary: next.includes(primary) ? primary : next[0], shown: next};
  }
  
  
  // ===== Comparing runs (U-054) =====
  
  function stepAtOrBefore(steps, step) {
    //* Return the largest of the sorted steps that is not after step, or null (FR-041).
    
    let found = null;
    for (const s of steps) if (s <= step) found = s;
    return found;
  }
  
  
  function histDensity(hist) {
    //* Return a distribution as densities {lo, hi, ys}: count / (all counts x the width of a bin); null without counts (FR-041).
    //! When all values are alike (hi <= lo), ys is [1]: the chart draws it as a line at lo.
    
    const counts = hist && Array.isArray(hist.counts) ? hist.counts : [];
    const total  = counts.reduce((a, c) => a + c, 0);
    if (!total || typeof hist.lo !== "number" || typeof hist.hi !== "number") return null;
    if (!(hist.hi > hist.lo)) return {lo: hist.lo, hi: hist.lo, ys: [1]};
    const width  = (hist.hi - hist.lo) / counts.length;
    return {lo: hist.lo, hi: hist.hi, ys: counts.map(c => c / (total * width))};
  }
  
  
  function codeShare(codes) {
    //* Return a code usage as shares {offset, ys}: count / all counts; null without counts (FR-041).
    
    const counts = codes && Array.isArray(codes.counts) ? codes.counts : null;
    const total  = counts ? counts.reduce((a, c) => a + c, 0) : 0;
    return total ? {offset: codes.offset, ys: counts.map(c => c / total)} : null;
  }
  
  
  function diffValue(a, b) {
    //* Return b - a when both are finite numbers, or null (the difference column of FR-041).
    
    return typeof a === "number" && typeof b === "number" && isFinite(a) && isFinite(b) ? b - a : null;
  }
  
  
  function groupToggle(shown, hidden, folder) {
    //* Return the runs to hide in a group after a click on the chip of folder, in the order of shown (FR-041).
    //! A hidden run comes back. A shown run is hidden only while another run of shown stays.
    
    if (hidden.includes(folder)) return shown.filter(f => hidden.includes(f) && f !== folder);
    const left = shown.filter(f => !hidden.includes(f) && f !== folder);
    return shown.filter(f => hidden.includes(f) || (f === folder && left.length > 0));
  }
  
  
  function groupShow(have, hidden) {
    //* Return the runs that a group shows: the runs of have that the group does not hide, or all of have when it hides every one (FR-041, BUG-016).
    //! have holds the shown runs that have the point, in the order of the list of runs; the result keeps that order.
    
    const on = have.filter(f => !hidden.includes(f));
    return on.length ? on : have.slice();
  }
  
  
  // ===== Export =====
  
  const core        = {
    GUT_L, GUT_R, AXES, CMAPS, PALETTES, RUN_GAP, RUNS_MAX,
    fmt, fmtPct, fmtInt, niceTicks, stepNearest, stepX,
    hexRgb, rgbHex, oklab, mix, contrast, makeLut, cmap, figureColors,
    parseLines, treeRows,
    mapCells, mapValue, mapSpan, axisName, mapSquare, mapPoint, mapFrac, mapTicks,
    decodeFloat32, compareStats, csvBuild,
    globRegExp, pointFirst,
    runsClick,
    stepAtOrBefore, histDensity, codeShare, diffValue, groupToggle, groupShow,
  };
  root.tensightCore = core;
  if (typeof module === "object" && module.exports) module.exports = core;
})(typeof window === "object" ? window : globalThis);
