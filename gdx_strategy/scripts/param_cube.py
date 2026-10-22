"""Interactive 3-D explorer: lookback x k x H, coloured by net Sharpe.

  python scripts/param_cube.py      # -> results/param_cube/param_cube.html

Data: every selectable config of the two all-factors runs (their all_configs_summary
files), development and validation net Sharpe at 2 bps. The original grid's three
factor sets are inside the 2008-window run with identical numbers. The page loads
Plotly from a CDN and embeds the data, so it is one self-contained file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "param_cube" / "param_cube.html"
RUNS = [("2008", "All SPY/IAU/GLD/TLT sets · 2008 window", ROOT / "results" / "variants" / "all_factors_2008"),
        ("common", "All 31 sets incl. GDXJ · common window (from Oct 2011)",
         ROOT / "results" / "variants" / "all_factors_common")]
ORIGINAL_FROZEN = ROOT / "results" / "frozen_config.json"
AXES = {"lookback": [20, 40, 60, 120, 250], "k": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], "H": [1, 2, 3, 5, 10],
        "m": [1, 2, 3, 5]}
POLICIES = ["A_run_out", "B_reset_flip", "C_reset_flat"]
DIRS = ["reversion", "estimated"]


def build_data() -> dict:
    data = {"axes": AXES, "policies": POLICIES, "dirs": DIRS, "runs": []}
    orig = json.loads(ORIGINAL_FROZEN.read_text())
    for key, label, d in RUNS:
        s = pd.read_csv(d / "all_configs_summary.csv.gz")
        s = s[(s.policy != "D_stack_2x") & s.period.isin(["development", "validation"])]
        w = s.pivot_table(index=["config_id", "factors", "lookback", "m", "k", "H", "policy", "direction_mode"],
                          columns="period", values="sharpe").reset_index()
        sets = sorted(w["factors"].unique(), key=lambda f: (f.count("+"), f))
        enc = lambda v, axis: AXES[axis].index(v)  # noqa: E731
        rows = [[sets.index(r.factors), enc(r.lookback, "lookback"), enc(r.m, "m"), enc(float(r.k), "k"),
                 enc(r.H, "H"), POLICIES.index(r.policy), DIRS.index(r.direction_mode),
                 round(r.development * 1000), round(r.validation * 1000)]
                for r in w.itertuples()]
        frozen = json.loads((d / "frozen_config.json").read_text())
        marks = [{"label": "Frozen primary (this run)", "id": frozen["primary"]["config_id"]}]
        if key == "2008":
            marks.insert(0, {"label": "Frozen primary (original grid)", "id": orig["primary"]["config_id"]})
        dev = frozen["splits"]["development"]
        data["runs"].append({"key": key, "label": label, "sets": sets, "rows": rows, "marks": marks,
                             "dev": dev, "val": frozen["splits"]["validation"], "n": len(rows)})
    return data


PAGE = r"""<title>GDX Parameter Cube</title>
<style>
/* Layout: one working screen; controls row, then the cube beside a slice summary, then per-H tables. */
:root {
  --bg: #fbfbf9; --panel: #ffffff; --fg: #141413; --muted: #5d5c57; --line: #e3e2dc;
  --neg: #c03a39; --neg-deep: #7c1d1d; --mid: #f0efec; --pos: #2a78d6; --pos-deep: #0d4a8f;
  --mark: #141413; --focus: #2a78d6;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #161615; --panel: #1d1d1c; --fg: #f3f2ee; --muted: #b3b2aa; --line: #34342f;
  --neg: #e66767; --neg-deep: #b13a3a; --mid: #383835; --pos: #3987e5; --pos-deep: #8ab8f0;
  --mark: #ffffff; --focus: #3987e5; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #161615; --panel: #1d1d1c; --fg: #f3f2ee; --muted: #b3b2aa; --line: #34342f;
  --neg: #e66767; --neg-deep: #b13a3a; --mid: #383835; --pos: #3987e5; --pos-deep: #8ab8f0;
  --mark: #ffffff; --focus: #3987e5; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font: 14px/1.5 var(--f-body); }
.wrap { max-width: 1240px; margin: 0 auto; padding-inline: 20px; padding-block: 20px 40px; display: grid; gap: 16px; }
header h1 { font: 600 26px/1.15 var(--f-display); letter-spacing: .01em; margin: 0; text-wrap: balance; }
header p { margin: 6px 0 0; color: var(--muted); max-width: 72ch; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; padding: 12px 14px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.controls label { display: grid; gap: 3px; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  flex: 1 1 150px; min-width: 0; max-width: 100%; }
.controls label:first-child { flex-basis: 300px; }
.controls label:last-child { flex: 0 0 auto; }
.controls select { font: 14px var(--f-body); color: var(--fg); background: var(--bg); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 8px; width: 100%; min-width: 0; text-overflow: ellipsis; }
.controls select:focus-visible, .seg button:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.seg button { font: 13px var(--f-body); color: var(--fg); background: var(--bg); border: 0; padding: 6px 12px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--fg); color: var(--bg); }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; align-items: start; }
@media (max-width: 860px) { .main { grid-template-columns: 1fr; } }
#cube { width: 100%; height: 560px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
@media (max-width: 600px) { #cube { height: 420px; } }
aside { display: grid; gap: 12px; }
.stat { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; display: grid; gap: 8px; min-width: 0; }
.stat h2 { font: 600 12px var(--f-body); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0; }
.kv { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; font-variant-numeric: tabular-nums; }
.kv dt { color: var(--muted); } .kv dd { margin: 0; font-family: var(--f-mono); text-align: right; white-space: nowrap; }
.legend { display: grid; gap: 6px; }
.bar { height: 10px; border-radius: 2px; background: linear-gradient(90deg, var(--neg-deep), var(--neg), var(--mid), var(--pos), var(--pos-deep)); }
.ticks { display: flex; justify-content: space-between; font: 12px var(--f-mono); color: var(--muted); }
.note { color: var(--muted); font-size: 12.5px; margin: 0; }
.tables { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 12px; }
.tbox { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; overflow-x: auto; min-width: 0; }
.tbox h3 { margin: 0 0 6px; font: 600 13px var(--f-body); }
table { border-collapse: separate; border-spacing: 2px; font: 12px var(--f-mono); font-variant-numeric: tabular-nums; width: 100%; }
th { color: var(--muted); font-weight: 500; padding: 2px 4px; text-align: center; }
td { text-align: center; padding: 4px 2px; border-radius: 3px; color: var(--fg); }
td.prim { outline: 2px solid var(--mark); outline-offset: -2px; }
.status { color: var(--muted); font-size: 13px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">

<div class="wrap">
  <header>
    <h1>GDX parameter cube</h1>
    <p>Each point is one strategy configuration: lookback L (days in the factor regression), entry threshold k (|z| needed to trade)
      and holding period H (days). Colour is net Sharpe after 2 bps per unit traded. Drag to rotate, scroll to zoom, hover for values.
      The tables below give the same slice as numbers, one table per H.</p>
  </header>

  <div class="controls" role="group" aria-label="Slice">
    <label for="run">Grid run<select id="run"></select></label>
    <label for="fs">Factor set<select id="fs"></select></label>
    <label for="pol">Policy<select id="pol"></select></label>
    <label for="dir">Direction<select id="dir"></select></label>
    <label for="m">m (residual days)<select id="m"></select></label>
    <label>Period<span class="seg" id="period">
      <button type="button" data-p="0" aria-pressed="true">Development</button>
      <button type="button" data-p="1" aria-pressed="false">Validation</button></span></label>
  </div>

  <div class="main">
    <div id="cube" role="img" aria-label="3-D scatter of net Sharpe by lookback, k and H"><p class="status" id="status">Loading chart…</p></div>
    <aside>
      <section class="stat" aria-live="polite">
        <h2 id="slice-h">This slice</h2>
        <dl class="kv" id="kv"></dl>
      </section>
      <section class="stat legend">
        <h2>Net Sharpe</h2>
        <div class="bar" aria-hidden="true"></div>
        <div class="ticks"><span>≤ −1.0</span><span>0</span><span>≥ +1.0</span></div>
        <p class="note">Red loses, blue makes money, grey is near zero. ◆ marks a frozen primary when it sits in the slice.</p>
      </section>
      <section class="stat">
        <h2>Windows</h2>
        <p class="note" id="windows"></p>
      </section>
    </aside>
  </div>

  <div class="tables" id="tables"></div>
</div>

<script id="cube-data" type="application/json">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("cube-data").textContent);
  const $ = (id) => document.getElementById(id);
  const PL = { A_run_out: "A · run out", B_reset_flip: "B · reset / flip", C_reset_flat: "C · reset / flat" };
  const state = { run: 0, fs: 0, pol: 0, dir: 0, m: 0, period: 0 };

  function parseId(id) {
    const p = id.split("|");
    return { fs: p[0], L: +p[1].slice(1), m: +p[2].slice(1), k: +p[3].slice(1), H: +p[4].slice(1),
             pol: { A: "A_run_out", B: "B_reset_flip", C: "C_reset_flat" }[p[5]],
             dir: { rev: "reversion", est: "estimated" }[p[6]] };
  }
  function fill(sel, items, label) {
    sel.innerHTML = items.map((v, i) => `<option value="${i}">${label ? label(v) : v}</option>`).join("");
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function fmt(x) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2); }

  function setRun(i, keepSlice) {
    state.run = i;
    const R = D.runs[i];
    fill($("fs"), R.sets);
    $("windows").textContent = `Development ${R.dev[0]} → ${R.dev[1]}; validation ${R.val[0]} → ${R.val[1]}. ` +
      `${R.n.toLocaleString()} configurations in this run.`;
    if (!keepSlice) {  // open on the slice of the run's first marked primary
      const p = parseId(R.marks[0].id);
      state.fs = Math.max(0, R.sets.indexOf(p.fs)); state.pol = D.policies.indexOf(p.pol);
      state.dir = D.dirs.indexOf(p.dir); state.m = D.axes.m.indexOf(p.m);
    } else if (state.fs >= R.sets.length) state.fs = 0;
    $("fs").value = state.fs; $("pol").value = state.pol; $("dir").value = state.dir; $("m").value = state.m;
  }

  function slice() {
    const R = D.runs[state.run];
    return R.rows.filter(r => r[0] === state.fs && r[2] === state.m && r[5] === state.pol && r[6] === state.dir);
  }

  function colorFor(v) {  // v in Sharpe units; same scale as the chart
    const t = Math.max(-1, Math.min(1, v));
    const stops = t < 0 ? [css("--mid"), css("--neg"), css("--neg-deep")] : [css("--mid"), css("--pos"), css("--pos-deep")];
    const a = Math.abs(t);
    const mix = (c1, c2, w) => { const h = (c) => [1, 3, 5].map(j => parseInt(c.slice(j, j + 2), 16));
      const x = h(c1), y = h(c2); return `rgb(${x.map((q, j) => Math.round(q + (y[j] - q) * w)).join(",")})`; };
    return a < 0.5 ? mix(stops[0], stops[1], a / 0.5) : mix(stops[1], stops[2], (a - 0.5) / 0.5);
  }

  function draw() {
    const R = D.runs[state.run], A = D.axes, S = slice(), P = state.period;
    const val = (r) => r[7 + P] / 1000, other = (r) => r[8 - P] / 1000;
    const per = P ? "validation" : "development", oth = P ? "development" : "validation";
    const fg = css("--fg"), muted = css("--muted"), line = css("--line"), panel = css("--panel");
    const scale = [[0, css("--neg-deep")], [0.25, css("--neg")], [0.5, css("--mid")], [0.75, css("--pos")], [1, css("--pos-deep")]];
    const pts = {
      type: "scatter3d", mode: "markers", name: "configs",
      x: S.map(r => r[1]), y: S.map(r => r[3]), z: S.map(r => r[4]),
      marker: { size: 7, color: S.map(val), colorscale: scale, cmin: -1, cmax: 1, opacity: 0.95,
                line: { color: muted, width: 0.6 } },
      text: S.map(r => `L ${A.lookback[r[1]]} · k ${A.k[r[3]]} · H ${A.H[r[4]]}<br>${per} Sharpe <b>${fmt(val(r))}</b>` +
                       `<br>${oth} Sharpe ${fmt(other(r))}`),
      hovertemplate: "%{text}<extra></extra>",
    };
    const traces = [pts];
    const fsName = R.sets[state.fs];
    R.marks.forEach((mk, j) => {
      const p = parseId(mk.id);
      if (p.fs !== fsName || p.m !== A.m[state.m] || p.pol !== D.policies[state.pol] || p.dir !== D.dirs[state.dir]) return;
      const r = S.find(q => A.lookback[q[1]] === p.L && A.k[q[3]] === p.k && A.H[q[4]] === p.H);
      if (!r) return;
      traces.push({ type: "scatter3d", mode: "markers", name: mk.label, x: [r[1]], y: [r[3]], z: [r[4]],
        marker: { size: 13, symbol: "diamond-open", color: css("--mark"), line: { color: css("--mark"), width: 3 } },
        hovertemplate: `${mk.label}<br>${per} Sharpe ${fmt(val(r))}<extra></extra>` });
    });
    const ax = (title, vals) => ({ title: { text: title, font: { color: fg, size: 12 } }, tickvals: vals.map((_, i) => i),
      ticktext: vals.map(String), tickfont: { color: muted, size: 11 }, gridcolor: line, zerolinecolor: line,
      backgroundcolor: panel, showbackground: true, linecolor: line });
    const layout = {
      paper_bgcolor: panel, plot_bgcolor: panel, margin: { l: 0, r: 0, t: 8, b: 0 }, showlegend: false,
      font: { family: css("--f-body"), color: fg },
      scene: { xaxis: ax("lookback L (days)", A.lookback), yaxis: ax("threshold k", A.k), zaxis: ax("hold H (days)", A.H),
               camera: { eye: { x: 1.55, y: -1.55, z: 0.95 } }, aspectmode: "cube" },
      hoverlabel: { bgcolor: panel, bordercolor: line, font: { color: fg, family: css("--f-mono") } },
    };
    if (window.Plotly) {
      $("status") && $("status").remove();
      Plotly.react("cube", traces, layout, { displaylogo: false, responsive: true,
        modeBarButtonsToRemove: ["toImage", "resetCameraLastSave3d"] });
    }
    // slice summary
    const v = S.map(val).sort((a, b) => a - b), med = v.length ? v[Math.floor(v.length / 2)] : NaN;
    const best = S.reduce((b, r) => (val(r) > val(b) ? r : b), S[0]);
    const pos = v.filter(x => x > 0).length;
    $("slice-h").textContent = `This slice · ${per}`;
    const kv = [["Configs", S.length], ["Median Sharpe", fmt(med)], ["Above zero", `${Math.round(100 * pos / v.length)}%`],
                ["Best cell", `L${A.lookback[best[1]]} k${A.k[best[3]]} H${A.H[best[4]]}`], ["Best Sharpe", fmt(val(best))],
                [`Best cell, ${oth}`, fmt(other(best))]];
    $("kv").innerHTML = kv.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("");
    // per-H tables (accessible numeric view)
    const markIds = new Set(R.marks.map(m => m.id));
    $("tables").innerHTML = A.H.map((h, hi) => {
      const head = `<tr><th scope="col">L \\ k</th>${A.k.map(k => `<th scope="col">${k}</th>`).join("")}</tr>`;
      const body = A.lookback.map((L, li) => `<tr><th scope="row">${L}</th>` + A.k.map((k, ki) => {
        const r = S.find(q => q[1] === li && q[3] === ki && q[4] === hi);
        if (!r) return "<td>–</td>";
        const x = val(r), id = `${fsName}|L${L}|m${A.m[state.m]}|k${+k}|H${h}|${D.policies[state.pol][0]}|${D.dirs[state.dir].slice(0, 3)}`;
        const ink = Math.abs(x) > 0.55 ? "#ffffff" : css("--fg");
        return `<td class="${markIds.has(id) ? "prim" : ""}" style="background:${colorFor(x)};color:${ink}" title="L ${L}, k ${k}, H ${h}: ${fmt(x)}">${x.toFixed(2)}</td>`;
      }).join("") + "</tr>").join("");
      return `<div class="tbox"><h3>H = ${h} day${h > 1 ? "s" : ""}</h3><table aria-label="${per} Sharpe, H = ${h}"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
    }).join("");
  }

  fill($("run"), D.runs.map(r => r.label));
  fill($("pol"), D.policies, v => PL[v]);
  fill($("dir"), D.dirs);
  fill($("m"), D.axes.m);
  setRun(0, false);
  $("run").addEventListener("change", e => { setRun(+e.target.value, true); draw(); });
  ["fs", "pol", "dir", "m"].forEach(id => $(id).addEventListener("change", e => { state[id] = +e.target.value; draw(); }));
  $("period").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    state.period = +b.dataset.p;
    $("period").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    draw();
  });
  const redraw = () => draw();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.__drawCube = draw;
  draw();
})();
</script>
<script>
(function () {
  function ready() { if (window.__drawCube) window.__drawCube(); }
  function load(src, onerror) { const s = document.createElement("script"); s.src = src; s.onload = ready; s.onerror = onerror; document.head.appendChild(s); }
  load("https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js", function () {
    load("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js", function () {
      const st = document.getElementById("status");
      if (st) st.textContent = "The 3-D chart could not load (no access to the charting library). The tables below show the same slice.";
    });
  });
})();
</script>
"""


def main():
    data = build_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", blob))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {sum(r['n'] for r in data['runs']):,} configs)")


if __name__ == "__main__":
    main()
