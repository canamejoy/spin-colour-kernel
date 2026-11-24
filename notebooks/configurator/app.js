/* SCK run configurator: the form, the Hamiltonian, the model view, the parts tray and the download.
 * Every physical fact on screen comes from SCK (core.cjs), which the tests hold equal to the Python package. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const DEF = TEMPLATE.defaults;
  const STORE = "sck-configurator-v1";

  // kit colours: the first classes look like the balls of a model kit, the rest continue round the hue circle
  const KIT = ["#d7263d", "#1b5bd6", "#f2b705", "#1e9e5a", "#f07c1a", "#7b3fbf", "#e9ecef", "#2a2d33",
               "#18a8c9", "#8a5a2b", "#e86aa6", "#93c13e"];
  const classColour = (k) => (k < KIT.length ? KIT[k] : `hsl(${(k * 137.5) % 360} 62% 52%)`);
  const ACTIVE = "#1b5bd6", SURFACE = "#f2b705", BULK = "#1b5bd6";

  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode: no memory */ } },
  };

  // ------------------------------------------------------------------ state
  const clone = (x) => JSON.parse(JSON.stringify(x));
  let cfg = Object.assign(clone(DEF), (store.get(STORE) || {}).cfg || {});
  let ui = Object.assign({ format: "ipynb", fname: "spin_simulation", view: "colour", shown: 0, bonds: true,
                           ghost: true, theme: "auto" }, (store.get(STORE) || {}).ui || {});
  let res = null;           // the last analysis
  if (!cfg.mode) cfg.mode = "states";
  if (!cfg.sweep) cfg.sweep = { base: { J: { 1: 1.0 } }, rows: [{ p: "D", from: 0, to: 1, step: 0.1 }], prefix: "sweep" };
  if (cfg.box.length < 3) cfg.box = cfg.box.concat([5, 5, 5]).slice(0, 3);
  if (cfg.periodic.length < 3) cfg.periodic = cfg.periodic.concat([false, false, false]).slice(0, 3);

  const save = () => store.set(STORE, { cfg, ui });
  const fmt = (x, d) => (x === undefined || x === null || Number.isNaN(x) ? "–" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d === undefined ? 0 : d }).replace(/,/g, " "));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const dims = () => (SCK.makeCrystal(cfg.crystal, [1]).D);

  // ------------------------------------------------------------------ segmented controls
  function seg(id, value, onPick) {
    const el = $(id);
    const sync = (v) => el.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === v)));
    sync(value);
    el.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      sync(b.dataset.v);
      onPick(b.dataset.v);
    });
    el.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      const bs = [...el.querySelectorAll("button")];
      const i = bs.indexOf(document.activeElement);
      if (i < 0) return;
      const j = (i + (e.key === "ArrowRight" ? 1 : -1) + bs.length) % bs.length;
      bs[j].focus(); bs[j].click(); e.preventDefault();
    });
    return sync;
  }

  // ------------------------------------------------------------------ theme
  function applyTheme(v) {
    if (v === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", v);
    view.sprites = {};
    view.dirty = true;
  }

  // ------------------------------------------------------------------ the form
  const NUM_FIELDS = ["T_max", "T_min", "T_step", "thermalization", "measurement", "measure_every", "replicas", "seed", "config_every"];
  let syncModel;

  function bindForm() {
    syncModel = seg("model", cfg.model, (v) => { cfg.model = v; renderStates(); if (cfg.mode === "sweep") renderSweep(); changed(); });
    $("units").value = cfg.units;
    $("units").addEventListener("change", (e) => { cfg.units = e.target.value; changed(); });
    $("crystal").value = cfg.crystal;
    $("crystal").addEventListener("change", (e) => { cfg.crystal = e.target.value; renderAxes(); renderPresets(); changed(); });
    $("coords").value = cfg.coords;
    $("coords").addEventListener("change", (e) => { cfg.coords = e.target.value; shapeHelp(); changed(); });
    $("shape").value = cfg.shape;
    $("shape").addEventListener("input", (e) => { cfg.shape = e.target.value; changed(); });
    for (const k of NUM_FIELDS) {
      $(k).value = cfg[k];
      $(k).addEventListener("input", (e) => { cfg[k] = e.target.value.trim() === "" ? "" : Number(e.target.value); changed(); });
    }
    $("output_dir").value = cfg.output_dir;
    $("output_dir").addEventListener("input", (e) => { cfg.output_dir = e.target.value; changed(); });
    $("gpus").value = cfg.gpus === "all" ? "all" : cfg.gpus.join(", ");
    $("gpus").addEventListener("input", (e) => {
      const t = e.target.value.trim().toLowerCase();
      cfg.gpus = t === "" || t === "all" ? "all" : t.split(/[\s,]+/).filter(Boolean).map(Number);
      changed();
    });
    $("add-state").addEventListener("click", () => {
      const n = cfg.states.length + 1;
      cfg.states.push({ name: `state-${n}`, J: { 1: 1.0 } });
      ui.shown = cfg.states.length - 1;
      renderStates(); changed();
      const inputs = $("states").querySelectorAll(".state-top input");
      inputs[inputs.length - 1].select();
    });
    $("reset").addEventListener("click", () => {
      const mode = cfg.mode;
      cfg = clone(DEF);
      cfg.mode = mode;
      cfg.sweep = { base: { J: { 1: 1.0 } }, rows: [{ p: "D", from: 0, to: 1, step: 0.1 }], prefix: "sweep" };
      cfg.box = cfg.box.concat([5, 5, 5]).slice(0, 3);
      cfg.periodic = cfg.periodic.concat([false, false, false]).slice(0, 3);
      ui.shown = 0;
      bindValues(); if (cfg.mode === "sweep") renderSweep(); changed();
    });
    renderAxes(); renderPresets(); renderStates(); shapeHelp();
  }

  function bindValues() {
    syncModel(cfg.model);
    for (const k of ["units", "crystal", "coords", "shape", "output_dir"]) $(k).value = cfg[k];
    for (const k of NUM_FIELDS) $(k).value = cfg[k];
    $("gpus").value = cfg.gpus === "all" ? "all" : cfg.gpus.join(", ");
    renderAxes(); renderPresets(); renderStates(); shapeHelp();
  }

  function renderAxes() {
    const D = dims();
    const names = ["x", "y", "z"];
    $("axes").innerHTML = names.slice(0, D).map((a, k) => `
      <div class="axis"><label class="col" style="margin:0">L${a}
        <input data-k="${k}" class="box" inputmode="numeric" value="${esc(cfg.box[k])}" aria-label="Box size along ${a}, in unit cells"></label>
        <label class="check"><input type="checkbox" data-k="${k}" class="per" ${cfg.periodic[k] ? "checked" : ""}> periodic</label></div>`).join("");
    $("axes").querySelectorAll("input.box").forEach((el) => el.addEventListener("input", (e) => {
      cfg.box[+e.target.dataset.k] = e.target.value.trim() === "" ? "" : Number(e.target.value); renderPresets(); changed();
    }));
    $("axes").querySelectorAll("input.per").forEach((el) => el.addEventListener("change", (e) => {
      cfg.periodic[+e.target.dataset.k] = e.target.checked; changed();
    }));
  }

  function renderPresets() {
    const D = dims();
    const L = cfg.box.slice(0, D).map(Number).filter((v) => v > 0);
    const R = Math.max(1, Math.floor(Math.min(...L.slice(0, 2)) / 2) - 0.5);
    const Lz = D === 3 ? Number(cfg.box[2]) || 1 : 1;
    const half = Math.max(0, (Lz - 1) / 2);
    const P = [["Every site", "", ""]];
    if (D === 3) {
      P.push(["Disc", "cylindrical", `(rho <= ${R}) & (abs(z) <= ${half})`]);
      P.push(["Sphere", "spherical", `r <= ${R}`]);
      P.push(["Ring", "cylindrical", `(rho <= ${R}) & (rho >= ${Math.round(R / 2 * 10) / 10})`]);
      P.push(["Thin film", "cartesian", `abs(z) <= ${Math.min(1, half)}`]);
    } else {
      P.push(["Disc", "cylindrical", `rho <= ${R}`]);
      P.push(["Ring", "cylindrical", `(rho <= ${R}) & (rho >= ${Math.round(R / 2 * 10) / 10})`]);
      P.push(["Square hole", "cartesian", `(abs(x) > ${Math.round(R / 3)}) | (abs(y) > ${Math.round(R / 3)})`]);
    }
    $("presets").innerHTML = P.map((p, i) => `<button type="button" class="chip" data-i="${i}">${p[0]}</button>`).join("");
    $("presets").onclick = (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      const p = P[+b.dataset.i];
      cfg.coords = p[1]; cfg.shape = p[2];
      $("coords").value = p[1]; $("shape").value = p[2];
      if (p[2]) { cfg.periodic = [false, false, false]; renderAxes(); }
      shapeHelp(); changed();
    };
  }

  function shapeHelp() {
    const c = cfg.coords;
    $("shape").disabled = !c;
    $("shape-help").innerHTML = c
      ? `Names: <code>${SCK.COORDS[c].join(", ")}</code>. Origin at the centre of the box, lengths in lattice units, angles in radians. Join conditions with <code>&amp;</code> (and), <code>|</code> (or), <code>~</code> (not), each in parentheses. Functions: <code>${SCK.FUNCS.join(", ")}</code>, and <code>pi</code>.`
      : "Every site of the box carries a spin. Pick a coordinate system to cut a shape out of the box.";
  }

  function summary(st) {
    return SCK.PARAMS.filter(([p]) => { const v = SCK.getParam(st, p); return v !== undefined && v !== "" && Number(v) !== 0; })
      .map(([p, l]) => `${l} ${fmtNum(SCK.getParam(st, p))}`).join(" · ") || "all zero";
  }

  function renderStates() {
    const m = cfg.model;
    $("model-help").textContent = { ising: "Spins s = ±1. Carries exchange J and a field H.",
      xy: "Unit spins in the plane. Carries exchange J and an in-plane field H.",
      heisenberg: "Unit spins in 3D. Carries exchange, DMI, cubic and surface anisotropy, and a field." }[m] || "";
    const field = (i, st, k, label, title, off) => `<label class="${off ? "off" : ""}" title="${esc(title)}">${label}
      <input data-i="${i}" data-k="${k}" inputmode="decimal" value="${st[k] === undefined ? "" : esc(st[k])}" ${off ? "disabled" : ""}></label>`;
    const compact = cfg.states.length > 3;
    const del = (i, st) => (cfg.states.length > 1 ? `<button type="button" class="ghost" data-del="${i}" aria-label="Remove ${esc(st.name || "state")}">Remove</button>` : "");
    $("states").innerHTML = (cfg.states.length > 1 ? `<div class="states-tools"><span>${cfg.states.length} states</span><button type="button" class="ghost" data-clear>Keep only the open state</button></div>` : "") +
      cfg.states.map((st, i) => {
      if (compact && i !== ui.shown)
        return `<div class="state compact" data-i="${i}"><button type="button" class="state-sum" data-open="${i}" aria-label="Open ${esc(st.name || "")}">
          <b>${esc(st.name || `state-${i + 1}`)}</b><span>${esc(summary(st))}</span></button>${del(i, st)}<p class="err" data-for="state.${i}"></p></div>`;
      const J = st.J || {};
      return `<div class="state ${i === ui.shown ? "on" : ""}" data-i="${i}">
        <div class="state-top"><input data-i="${i}" data-k="name" value="${esc(st.name || "")}" aria-label="Name of state ${i + 1}">${del(i, st)}</div>
        <p class="sub-h">Exchange per neighbour shell (J &gt; 0 ferromagnetic)</p>
        <div class="pgrid">${[1, 2, 3, 4].map((s) => `<label title="Shell ${s}: ${["nearest", "next-nearest", "third", "fourth"][s - 1]} neighbours">J${["₁", "₂", "₃", "₄"][s - 1]}
          <input data-i="${i}" data-j="${s}" inputmode="decimal" value="${J[s] === undefined ? "" : esc(J[s])}"></label>`).join("")}</div>
        <p class="sub-h">DMI and anisotropy${m === "heisenberg" ? "" : " (Heisenberg only)"}</p>
        <div class="pgrid">
          ${field(i, st, "D", "D", "Bulk-type (Bloch) DMI on in-plane nearest-neighbour bonds", m !== "heisenberg")}
          ${field(i, st, "K1", "K₁", "Cubic anisotropy on bulk sites", m !== "heisenberg")}
          ${field(i, st, "Ks", "Kₛ", "Cubic anisotropy on surface sites (empty = K₁)", m !== "heisenberg")}
          ${field(i, st, "K2", "K₂", "Second-order cubic anisotropy", m !== "heisenberg")}
        </div>
        <p class="sub-h">Field</p>
        <div class="pgrid">
          ${field(i, st, "H", "H", "Field magnitude", false)}
          ${field(i, st, "H_theta", "θ (°)", "Polar angle from +z (Heisenberg)", m !== "heisenberg")}
          ${field(i, st, "H_phi", "φ (°)", "Azimuth from +x (XY, Heisenberg)", m === "ising")}
          <span></span>
        </div>
        <p class="err" data-for="state.${i}"></p>
      </div>`;
    }).join("");
    const box = $("states");
    box.oninput = (e) => {
      const t = e.target;
      const i = +t.dataset.i;
      if (Number.isNaN(i)) return;
      const st = cfg.states[i];
      const v = t.value.trim();
      if (t.dataset.k === "name") st.name = t.value;
      else if (t.dataset.j) {
        st.J = st.J || {};
        if (v === "") delete st.J[t.dataset.j]; else st.J[t.dataset.j] = Number(v);
      } else if (v === "") delete st[t.dataset.k];
      else st[t.dataset.k] = Number(v);
      ui.shown = i;
      changed();
    };
    box.onfocusin = (e) => {
      const card = e.target.closest(".state");
      if (card && !card.classList.contains("compact") && +card.dataset.i !== ui.shown) { ui.shown = +card.dataset.i; markShown(); renderHamiltonian(); }
    };
    box.onclick = (e) => {
      const open = e.target.closest("[data-open]");
      if (open) { ui.shown = +open.dataset.open; renderStates(); renderHamiltonian(); save(); return; }
      if (e.target.closest("[data-clear]")) {
        cfg.states = [cfg.states[ui.shown]]; ui.shown = 0; renderStates(); changed(); return;
      }
      const b = e.target.closest("[data-del]");
      if (!b) return;
      cfg.states.splice(+b.dataset.del, 1);
      ui.shown = Math.min(ui.shown, cfg.states.length - 1);
      renderStates(); changed();
    };
  }
  function markShown() {
    $("states").querySelectorAll(".state").forEach((el) => el.classList.toggle("on", +el.dataset.i === ui.shown));
  }

  // ------------------------------------------------------------------ the Hamiltonian, in MathML
  const mi = (s, v) => `<mi${v ? ' mathvariant="bold"' : ""}>${s}</mi>`;
  const mn = (x) => `<mn>${fmtNum(x)}</mn>`;
  const mo = (s) => `<mo>${s}</mo>`;
  const sub = (a, b) => `<msub>${a}${b}</msub>`;
  const sup = (a, b) => `<msup>${a}${b}</msup>`;
  const sum = (under) => `<munder>${mo("∑")}<mrow>${under}</mrow></munder>`;
  const pair = (s) => `<mrow>${mo("⟨")}<mi>i</mi><mi>j</mi>${mo("⟩")}${s ? `<msub><mrow></mrow><mn>${s}</mn></msub>` : ""}</mrow>`;
  function fmtNum(x) { const v = Number(x); return Number.isInteger(v) ? v.toFixed(1) : String(+v.toPrecision(6)); }
  const spin = (i, model) => (model === "ising" ? sub(mi("s"), mi(i)) : sub(mi("S", true), mi(i)));
  function signed(first, v, rest) {          // "− v rest" with the sign folded in
    const neg = v < 0;
    const op = first ? (neg ? mo("−") : "") : mo(neg ? "−" : "+");
    return `${op}${mn(Math.abs(v))}${rest}`;
  }

  function renderHamiltonian() {
    view.dirty = true;                                  // the field arrow follows the state shown
    const pick = $("state-pick");
    const sweep = cfg.mode === "sweep";
    const label = (s, i) => esc(s.name || `state-${i + 1}`);
    if (sweep) {
      pick.removeAttribute("role"); pick.onclick = pick.onchange = null;
      pick.innerHTML = `<span class="count">${esc(sweepNote())}</span>`;
    } else if (cfg.states.length > 6) {
      pick.removeAttribute("role");
      pick.innerHTML = `<select aria-label="State shown">${cfg.states.map((s, i) => `<option value="${i}" ${i === ui.shown ? "selected" : ""}>${label(s, i)}</option>`).join("")}</select>`;
      pick.onclick = null;
      pick.onchange = (e) => { ui.shown = +e.target.value; renderStates(); renderHamiltonian(); save(); };
    } else {
      pick.setAttribute("role", "radiogroup");
      pick.innerHTML = cfg.states.map((s, i) => `<button type="button" class="chip" role="radio" aria-checked="${i === ui.shown}" data-i="${i}">${label(s, i)}</button>`).join("");
      pick.onchange = null;
      pick.onclick = (e) => { const b = e.target.closest("button"); if (b) { ui.shown = +b.dataset.i; cfg.states.length > 3 ? renderStates() : markShown(); renderHamiltonian(); save(); } };
    }
    const st = sweep ? cfg.sweep.base : (cfg.states[ui.shown] || cfg.states[0]);
    if (!st) { $("ham-math").innerHTML = ""; $("ham-values").innerHTML = ""; $("ham-terms").innerHTML = ""; return; }
    // in a sweep, a swept parameter is written as its symbol and its values are listed below
    const rows = sweep ? cfg.sweep.rows : [];
    const swept = (k) => rows.some((r) => r.p === k);
    const range = (k) => {
      const r = rows.find((x) => x.p === k), v = r && SCK.series(r.from, r.to, r.step);
      if (!v) return "a range still to complete";
      return v.length > 5 ? `{${v.slice(0, 3).map(fmtNum).join(", ")}, …, ${fmtNum(v[v.length - 1])}} (${v.length} values)` : `{${v.map(fmtNum).join(", ")}}`;
    };
    const m = cfg.model, heis = m === "heisenberg";
    const terms = [], notes = [];
    const n = (k) => { const v = SCK.getParam(st, k); return v === undefined || v === "" ? 0 : Number(v); };
    const on = (k) => swept(k) || n(k) !== 0;
    // coefficient: "−D", "+K₂" for a swept symbol, else the signed number
    const coef = (k, symbol, sign, rest) => (swept(k)
      ? `${!terms.length ? (sign < 0 ? mo("−") : "") : mo(sign < 0 ? "−" : "+")}${symbol}${rest}`
      : signed(!terms.length, sign * n(k), rest));
    const value = (k, sym) => (swept(k) ? `${sym} ∈ ${range(k)}` : `${sym} = ${fmtNum(n(k))}`);
    const dot = m === "ising" ? `${spin("i", m)}${spin("j", m)}` : `${spin("i", m)}${mo("·")}${spin("j", m)}`;
    const shellName = ["nearest", "next-nearest", "third", "fourth"];
    const shells = [1, 2, 3, 4].filter((s) => on(`J${s}`));
    for (const s of shells) {
      // H = − J Σ S·S: a positive J gives a leading minus
      terms.push(coef(`J${s}`, sub(mi("J"), `<mn>${s}</mn>`), -1, `${sum(pair(s))}${dot}`));
    }
    if (shells.length) {
      // one line for every exchange term: the shells differ only in their neighbours
      const who = shells.length === 1 ? `${shellName[shells[0] - 1]} neighbours`
        : `${shells.map((s) => shellName[s - 1]).join(", ").replace(/, ([^,]*)$/, " and $1")} neighbours, by shell`;
      notes.push([shells.map((s) => value(`J${s}`, `J${"₁₂₃₄"[s - 1]}`)).join(", "), `exchange with ${who}; each pair once, J > 0 ferromagnetic`, shells.map((s) => `J${"₁₂₃₄"[s - 1]}`).join(" ")]);
    }
    if (heis && on("D")) {
      terms.push(coef("D", mi("D"), -1, `${sum(`${pair(1)}${mo("∥")}`)}${sup(spin("i", m), mi("T"))}${sub(mi("A"), `<mrow><mi>i</mi><mi>j</mi></mrow>`)}${spin("j", m)}`));
      notes.push([value("D", "D"), "bulk-type (Bloch) DMI on in-plane nearest-neighbour bonds: A_lm = Σ_k r̂_k ε_klm, the DM vector along the bond", "D"]);
    }
    const ksSet = SCK.getParam(st, "Ks") !== undefined && SCK.getParam(st, "Ks") !== "";
    if (heis && (on("K1") || on("Ks"))) {
      const sq = (a, b) => `${sup(sub(mi("s"), mi(a)), "<mn>2</mn>")}${sup(sub(mi("s"), mi(b)), "<mn>2</mn>")}`;
      terms.push(`${terms.length ? mo("+") : ""}${sum(mi("i"))}${sub(mi("K"), `<mrow><mn>1</mn><mo>,</mo><mi>i</mi></mrow>`)}${mo("(")}${sq("x", "y")}${mo("+")}${sq("x", "z")}${mo("+")}${sq("y", "z")}${mo(")")}`);
      notes.push([`${value("K1", "K₁")}, ${swept("Ks") || ksSet ? value("Ks", "Kₛ") : "Kₛ = K₁"}`, "cubic anisotropy: K₁ on bulk sites, Kₛ on surface sites (fewer active nearest neighbours than the bulk)", "K₁ Kₛ"]);
    }
    if (heis && on("K2")) {
      terms.push(coef("K2", sub(mi("K"), "<mn>2</mn>"), 1, `${sum(mi("i"))}${sup(`<mrow>${mo("(")}${sub(mi("s"), mi("x"))}${sub(mi("s"), mi("y"))}${sub(mi("s"), mi("z"))}${mo(")")}</mrow>`, "<mn>2</mn>")}`));
      notes.push([value("K2", "K₂"), "second-order cubic anisotropy", "K₂"]);
    }
    if (on("H")) {
      const angleSwept = swept("H_theta") || swept("H_phi");
      const th = n("H_theta") * Math.PI / 180, ph = n("H_phi") * Math.PI / 180;
      const h = m === "ising" ? [1] : m === "xy" ? [Math.cos(ph), Math.sin(ph)] : [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
      const hv = angleSwept ? `<mover>${mi("h")}${mo("^")}</mover>`
        : `<mrow>${mo("(")}${h.map((x) => mn(Math.round(x * 1000) / 1000)).join(mo(","))}${mo(")")}</mrow>`;
      terms.push(coef("H", mi("H"), -1, `${sum(mi("i"))}${m === "ising" ? "" : `${hv}${mo("·")}`}${spin("i", m)}`));
      const dir = m === "ising" ? "along +1" : angleSwept
        ? `along ĥ(θ, φ), ${heis ? value("H_theta", "θ") + "°, " : ""}${value("H_phi", "φ")}°`
        : `along (${h.map((x) => (Math.round(x * 1000) / 1000)).join(", ")})${heis ? ` (θ = ${fmtNum(n("H_theta"))}°, φ = ${fmtNum(n("H_phi"))}°)` : ` (φ = ${fmtNum(n("H_phi"))}°)`}`;
      notes.push([value("H", "H"), `Zeeman field ${dir}`, "H"]);
    }
    const parts = terms.length ? terms : [mn(0)];
    $("ham-math").innerHTML = parts.map((t, k) => `<math displaystyle="true">${k ? "" : `${mi("H")}${mo("=")}`}<mrow>${t}</mrow></math>`).join("");
    const unit = cfg.units === "meV" ? "Energies in meV, temperatures in kelvin." : "Energies in units of J, k_B = 1.";
    // the values stay in sight on one line; what each term means folds away
    notes.push([{ ising: "s = ±1", xy: "|S| = 1, 2D", heisenberg: "|S| = 1, 3D" }[m] || "", `${{ ising: "Ising spins, s = ±1", xy: "unit spins in the plane", heisenberg: "unit spins in 3D" }[m] || ""}. ${unit}`, m === "ising" ? "s" : "S"]);
    $("ham-values").innerHTML = notes.map(([a]) => `<span>${esc(a)}</span>`).join("");
    $("ham-terms").innerHTML = notes.map(([, b, c]) => `<dt>${esc(c)}</dt><dd>${esc(b)}</dd>`).join("");
  }

  // ------------------------------------------------------------------ parameter sweep (its own tab)
  const PLABEL = Object.fromEntries(SCK.PARAMS.map(([k, l]) => [k, l]));
  const allowed = (k) => SCK.PARAMS.find((x) => x[0] === k)[2].includes(cfg.model);
  function sweepResult() { return SCK.sweepStates(cfg.sweep.base, cfg.sweep.rows.filter((r) => allowed(r.p)), cfg.sweep.prefix, 1000); }
  function sweepNote() { const r = sweepResult(); return r.states ? `${r.states.length} states` : "sweep incomplete"; }
  /* The states the downloaded file runs: the list of the States tab, or every combination of the sweep. */
  function runStates() {
    if (cfg.mode !== "sweep") return cfg.states.map((s, i) => Object.assign({}, s, { name: s.name || `state-${i + 1}` }));
    const r = sweepResult();
    return r.states || [];
  }
  function renderSweep() {
    const sw = cfg.sweep, m = cfg.model;
    const groups = [["Exchange per neighbour shell (J > 0 ferromagnetic)", ["J1", "J2", "J3", "J4"]],
                    [`DMI and anisotropy${m === "heisenberg" ? "" : " (Heisenberg only)"}`, ["D", "K1", "Ks", "K2"]],
                    ["Field", ["H", "H_theta", "H_phi"]]];
    $("sweep-base").innerHTML = groups.map(([h, keys]) => `<p class="sub-h">${esc(h)}</p><div class="pgrid">${keys.map((k) => {
      const swept = sw.rows.some((r) => r.p === k), off = !allowed(k);
      const v = SCK.getParam(sw.base, k);
      return `<label class="${off ? "off" : ""}">${PLABEL[k]}${k === "H_theta" || k === "H_phi" ? " (°)" : ""}
        <input data-bk="${k}" inputmode="decimal" value="${swept ? "" : v === undefined ? "" : esc(v)}" placeholder="${swept ? "swept" : ""}" ${off || swept ? "disabled" : ""}></label>`;
    }).join("")}</div>`).join("");
    const used = (k, i) => sw.rows.some((r, j) => j !== i && r.p === k);
    $("sweep-rows").innerHTML = sw.rows.map((r, i) => {
      const vals = SCK.series(r.from, r.to, r.step);
      return `<div class="sweep-row" data-i="${i}">
        <label>Parameter <select data-rk="p">${SCK.PARAMS.filter(([k]) => allowed(k) && !used(k, i)).map(([k, l]) => `<option value="${k}" ${k === r.p ? "selected" : ""}>${l}</option>`).join("")}</select></label>
        <label>From <input data-rk="from" inputmode="decimal" value="${esc(r.from)}"></label>
        <label>To <input data-rk="to" inputmode="decimal" value="${esc(r.to)}"></label>
        <label>Step <input data-rk="step" inputmode="decimal" value="${esc(r.step)}"></label>
        <span class="n">${vals ? `${vals.length} value${vals.length > 1 ? "s" : ""}` : "–"}</span>
        <button type="button" class="ghost" data-rdel="${i}" aria-label="Stop sweeping ${PLABEL[r.p]}">Remove</button>
      </div>`;
    }).join("") || `<p class="help">No parameter is swept yet.</p>`;
    $("sweep-add").disabled = sw.rows.length >= 3 || SCK.PARAMS.every(([k]) => !allowed(k) || sw.rows.some((r) => r.p === k));
    $("sweep-prefix").value = sw.prefix;
    renderSweepTable();
  }
  function renderSweepTable() {
    const r = sweepResult();
    const rows = cfg.sweep.rows.filter((x) => allowed(x.p));
    $("sweep-count").textContent = r.states ? `${fmt(r.states.length)} states` : "";
    if (!r.states) { $("sweep-table").innerHTML = `<p class="help">${esc(r.error)}.</p>`; return; }
    const shown = r.states.slice(0, 150);
    $("sweep-table").innerHTML = `<table><thead><tr><th>#</th>${rows.map((x) => `<th>${PLABEL[x.p]}</th>`).join("")}<th>name in the files</th></tr></thead><tbody>${
      shown.map((s, i) => `<tr><td>${i + 1}</td>${rows.map((x) => `<td>${fmtNum(SCK.getParam(s, x.p))}</td>`).join("")}<td class="nm">${esc(s.name)}</td></tr>`).join("")}</tbody></table>${
      r.states.length > shown.length ? `<p class="help">… and ${fmt(r.states.length - shown.length)} more.</p>` : ""}`;
  }
  function bindSweep() {
    $("sweep-base").addEventListener("input", (e) => {
      const k = e.target.dataset.bk; if (!k) return;
      const v = e.target.value.trim();
      if (v === "") { if (/^J/.test(k)) { const J = Object.assign({}, cfg.sweep.base.J); delete J[k[1]]; cfg.sweep.base.J = J; } else delete cfg.sweep.base[k]; }
      else SCK.setParam(cfg.sweep.base, k, Number(v));
      renderSweepTable(); changed();
    });
    const rowsEl = $("sweep-rows");
    rowsEl.addEventListener("input", (e) => {
      const k = e.target.dataset.rk, row = e.target.closest(".sweep-row"); if (!k || !row) return;
      const r = cfg.sweep.rows[+row.dataset.i];
      if (k === "p") return;
      r[k] = e.target.value.trim() === "" ? "" : Number(e.target.value);
      const vals = SCK.series(r.from, r.to, r.step);
      row.querySelector(".n").textContent = vals ? `${vals.length} value${vals.length > 1 ? "s" : ""}` : "–";
      renderSweepTable(); changed();
    });
    rowsEl.addEventListener("change", (e) => {
      if (e.target.dataset.rk !== "p") return;
      cfg.sweep.rows[+e.target.closest(".sweep-row").dataset.i].p = e.target.value;
      renderSweep(); changed();
    });
    rowsEl.addEventListener("click", (e) => {
      const b = e.target.closest("[data-rdel]"); if (!b) return;
      cfg.sweep.rows.splice(+b.dataset.rdel, 1); renderSweep(); changed();
    });
    $("sweep-add").addEventListener("click", () => {
      const free = SCK.PARAMS.find(([k]) => allowed(k) && !cfg.sweep.rows.some((r) => r.p === k));
      if (!free) return;
      cfg.sweep.rows.push({ p: free[0], from: 0, to: 1, step: 0.1 });
      renderSweep(); changed();
    });
    $("sweep-prefix").addEventListener("input", (e) => { cfg.sweep.prefix = e.target.value; renderSweepTable(); changed(); });
  }
  function setMode(v) {
    cfg.mode = v;
    document.body.classList.toggle("mode-sweep", v === "sweep");
    $("sweep-panel").hidden = v !== "sweep";
    if (v === "sweep") renderSweep(); else view.resize();
    changed();
  }

  // ------------------------------------------------------------------ analysis and error display
  let timer = 0;
  function changed() {
    save();
    renderHamiltonian();
    clearTimeout(timer);
    timer = setTimeout(analyse, 120);
  }

  function analyse() {
    const D = dims();
    const c = clone(cfg);
    c.box = cfg.box.slice(0, D); c.periodic = cfg.periodic.slice(0, D);
    c.states = runStates();
    try { res = SCK.analyse(c); } catch (e) { res = { errors: [{ field: "crystal", msg: e.message }], warnings: [] }; }
    for (const el of document.querySelectorAll("[data-for]")) el.textContent = "";
    for (const el of document.querySelectorAll("[aria-invalid]")) el.removeAttribute("aria-invalid");
    const put = (field, msg) => {
      let el = document.querySelector(`[data-for="${field}"]`);
      if (!el && field.startsWith("state.")) el = document.querySelector(`[data-for="${field.split(".").slice(0, 2).join(".")}"]`);
      if (!el) el = document.querySelector('[data-for="states"]');
      el.textContent = el.textContent ? `${el.textContent} ${msg}` : msg;
    };
    for (const e of res.errors) put(e.field, e.msg);
    for (const w of res.warnings) put(w.field, w.msg);
    if (res.errors.some((e) => e.field === "shape")) $("shape").setAttribute("aria-invalid", "true");
    if (res.errors.some((e) => e.field === "temperature")) ["T_max", "T_min", "T_step"].forEach((k) => $(k).setAttribute("aria-invalid", "true"));
    for (const k of ["thermalization", "measurement", "measure_every", "replicas", "seed", "config_every"]) {
      const v = cfg[k];
      const bad = !(Number.isInteger(v) && v >= (k === "seed" || k === "thermalization" ? 0 : 1));
      if (bad) $(k).setAttribute("aria-invalid", "true");
    }
    const t = res.temperatures;
    $("t-help").textContent = t ? `${t.length} temperatures, from ${t[0]} down to ${t[t.length - 1]}${cfg.units === "meV" ? " K" : ""}.` : "";
    renderTray();
    renderDownload();
    view.load(res);
  }

  function formErrors() {
    const out = (res ? res.errors.map((e) => e.msg) : ["checking…"]);
    for (const k of ["thermalization", "measurement", "measure_every", "replicas", "seed", "config_every"]) {
      const v = cfg[k];
      if (!(Number.isInteger(v) && v >= (k === "seed" || k === "thermalization" ? 0 : 1)))
        out.push(`${k.replace("_", " ")} must be a whole number${k === "seed" || k === "thermalization" ? "" : " of at least 1"}`);
    }
    if (cfg.gpus !== "all" && (!cfg.gpus.length || cfg.gpus.some((g) => !Number.isInteger(g) || g < 0))) out.push('GPUs must be "all" or GPU numbers such as 0, 1');
    if (!String(cfg.output_dir || "").trim()) out.push("name the output folder");
    if (cfg.mode === "sweep") { const r = sweepResult(); if (!r.states) out.unshift(`sweep: ${r.error}`); }
    return out;
  }

  // ------------------------------------------------------------------ parts tray
  let isolate = -1, hoverClass = -1;
  function renderTray() {
    const ok = res && res.colour;
    $("ncol").textContent = ok ? `${res.nColours} colour${res.nColours > 1 ? "s" : ""}` : "";
    if (!ok) {
      $("group").textContent = "Fix the settings marked in red to see the colouring.";
      $("classes").innerHTML = ""; $("sites").innerHTML = ""; $("budget").innerHTML = "";
      return;
    }
    const counts = new Array(res.nColours).fill(0);
    for (let i = 0; i < res.lattice.n; i++) if (res.mask[i]) counts[res.colour[i]]++;
    const g = res.colouring.m.length ? res.colouring.m.map((x) => `ℤ${x}`).join(" × ") : "one class";
    $("group").textContent = `Sites in one class share no bond, so each class updates in parallel. Colouring group ${g}, checked on every bond${res.periodicUsed.some(Boolean) ? " and across the periodic seam" : ""}.`;
    if (isolate >= res.nColours) isolate = -1;
    $("classes").innerHTML = counts.map((c, k) => `<li><button type="button" class="cmp${isolate >= 0 && isolate !== k ? " dim" : ""}" data-k="${k}" aria-pressed="${isolate === k}">
      <span class="ball" style="background:${classColour(k)}"></span><span><span class="lbl">colour ${k + 1}</span><span class="n">${fmt(c)}</span></span></button></li>`).join("");
    const cl = $("classes");
    cl.onclick = (e) => {
      const b = e.target.closest("button"); if (!b) return;
      const k = +b.dataset.k; isolate = isolate === k ? -1 : k;
      if (ui.view !== "colour") setView("colour");
      renderTray(); view.dirty = true;
    };
    cl.onpointerover = (e) => { const b = e.target.closest("button"); hoverClass = b ? +b.dataset.k : -1; view.dirty = true; };
    cl.onpointerleave = () => { hoverClass = -1; view.dirty = true; };
    cl.onfocusin = (e) => { const b = e.target.closest("button"); hoverClass = b ? +b.dataset.k : -1; view.dirty = true; };
    cl.onfocusout = () => { hoverClass = -1; view.dirty = true; };

    const n = res.lattice.n;
    $("sites").innerHTML = [["active", res.nActive], ["surface", res.nSurface], ["bulk", res.nActive - res.nSurface], ["box", n]]
      .map(([l, v]) => `<li><span class="stat"><span class="lbl">${l}</span><span class="n">${fmt(v)}</span></span></li>`).join("");

    const T = res.temperatures || [];
    const sw = (Number(cfg.thermalization) || 0) + (Number(cfg.measurement) || 0);
    const R = Number(cfg.replicas) || 0, S = runStates().length;
    const updates = T.length * sw * R * S * res.nActive;
    const gb = res.sizeGB || 0;
    $("budget").innerHTML = `<dl>
      <dt>temperatures</dt><dd>${fmt(T.length)}</dd>
      <dt>sweeps per temperature</dt><dd>${fmt(sw)}</dd>
      <dt>chains</dt><dd>${fmt(S)} × ${fmt(R)}</dd>
      <dt>spin updates</dt><dd>${updates ? updates.toExponential(2).replace("e+", " × 10^") : "–"}</dd>
      <dt>configurations saved</dt><dd>${fmt(res.nSaved)} per chain</dd></dl>
      <div class="cap"><div class="cap-top"><span>output size</span><b>${gb < 0.01 ? `${fmt(gb * 1000, 1)} MB` : `${fmt(gb, 2)} GB`}</b></div>
      <p class="help">Spin configurations (float32) written to the output folder; make sure its disk can hold them.</p></div>`;
  }

  // ------------------------------------------------------------------ download
  let syncFormat;
  function bindDownload() {
    syncFormat = seg("format", ui.format, (v) => { ui.format = v; renderDownload(); save(); });
    $("fname").value = ui.fname;
    $("fname").addEventListener("input", (e) => { ui.fname = e.target.value; save(); });
    $("download").addEventListener("click", download);
  }
  function renderDownload() {
    const nb = ui.format === "ipynb";
    $("fext").textContent = nb ? ".ipynb" : ".py";
    $("download-label").textContent = nb ? "Download notebook" : "Download script";
    $("format-help").textContent = nb
      ? "For Kaggle, Colab or Jupyter: upload it and press Run all."
      : "For a terminal or a cluster: python " + (ui.fname || "spin_simulation") + ".py";
    const errs = formErrors();
    $("download").disabled = errs.length > 0;
    $("download-msg").textContent = errs.length ? `Fix first: ${errs[0]}${errs.length > 1 ? ` (and ${errs.length - 1} more)` : ""}.` : "";
  }
  function download() {
    const D = dims();
    const c = clone(cfg);
    c.box = cfg.box.slice(0, D); c.periodic = cfg.periodic.slice(0, D);
    c.states = runStates();
    const nb = ui.format === "ipynb";
    let text;
    try { text = nb ? SCK.notebookFor(TEMPLATE, c) : SCK.scriptFor(TEMPLATE, c); }
    catch (e) { $("download-msg").textContent = e.message; return; }
    const name = (ui.fname || "spin_simulation").replace(/[\\/:*?"<>|]+/g, "_") + (nb ? ".ipynb" : ".py");
    const url = URL.createObjectURL(new Blob([text], { type: nb ? "application/x-ipynb+json" : "text/x-python" }));
    const a = document.createElement("a");
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  // ------------------------------------------------------------------ the model view (canvas, painter's algorithm)
  const HOME = [-0.6, -0.5];                 // yaw, pitch of the home view: front, right and top in sight
  const clampPitch = (p) => Math.max(-Math.PI / 2, Math.min(Math.PI / 2, p));
  // keys 1 to 6: top, front, right, bottom, back, left (the order of the flat view buttons)
  const FACES = [[0, 0, 1], [0, -1, 0], [1, 0, 0], [0, 0, -1], [0, 1, 0], [-1, 0, 0]];

  const view = {
    canvas: null, ctx: null, w: 0, h: 0, dpr: 1, dirty: true, yaw: -0.6, pitch: -0.5, zoom: 1, tween: null, panX: 0, panY: 0,
    sprites: {}, res: null, D: 3, layer: 1, radius: 0.3, extent: 1, order: null, bondPairs: null,

    init() {
      this.canvas = $("model-canvas");
      this.ctx = this.canvas.getContext("2d");
      new ResizeObserver(() => this.resize()).observe($("viewer"));
      let drag = null;
      this.canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || this.D === 2 }; this.canvas.setPointerCapture(e.pointerId); });
      this.canvas.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY;
        if (drag.pan) { this.panX += dx; this.panY += dy; }
        else { this.tween = null; this.yaw += dx * 0.008; this.pitch = clampPitch(this.pitch - dy * 0.008); }
        this.dirty = true;
      });
      const end = () => { drag = null; };
      this.canvas.addEventListener("pointerup", end);
      this.canvas.addEventListener("pointercancel", end);
      this.canvas.addEventListener("wheel", (e) => { e.preventDefault(); this.zoom = Math.max(0.2, Math.min(12, this.zoom * Math.exp(-e.deltaY * 0.0015))); this.dirty = true; }, { passive: false });
      this.canvas.addEventListener("keydown", (e) => {
        const k = e.key;
        if (k === "ArrowLeft") this.yaw -= 0.12; else if (k === "ArrowRight") this.yaw += 0.12;
        else if (k === "ArrowUp") this.pitch = clampPitch(this.pitch + 0.12);
        else if (k === "ArrowDown") this.pitch = clampPitch(this.pitch - 0.12);
        else if (k === "+" || k === "=") this.zoom = Math.min(12, this.zoom * 1.15);
        else if (k === "-") this.zoom = Math.max(0.2, this.zoom / 1.15);
        else if (k === "0") this.reset();
        else if (this.D === 3 && k >= "1" && k <= "6") this.lookFrom(FACES[+k - 1]);
        else if (this.D === 3 && k === "7") this.animateTo(HOME[0], HOME[1]);
        else return;
        e.preventDefault(); this.dirty = true;
      });
      $("reset-view").addEventListener("click", () => this.reset());
      $("gizmo-home").addEventListener("click", () => { this.animateTo(HOME[0], HOME[1]); });
      for (const b of $("views").querySelectorAll("button[data-n]")) {
        const n = b.dataset.n.split(",").map(Number), m = Math.hypot(...n);
        b.addEventListener("click", () => this.lookFrom(n.map((v) => v / m)));
      }
      $("layers").addEventListener("input", (e) => { this.layer = +e.target.value; this.syncLayers(); this.dirty = true; });
      const loop = (now) => { this.step(now); if (this.dirty) { this.dirty = false; this.draw(); } requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    },
    reset() { this.tween = null; this.yaw = this.D === 3 ? HOME[0] : 0; this.pitch = this.D === 3 ? HOME[1] : -Math.PI / 2; this.zoom = 1; this.panX = this.panY = 0; this.dirty = true; },
    /* Turn the camera so that the unit direction n (lattice axes) points at the viewer, in 300 ms. */
    lookFrom(n) {
      const h = Math.hypot(n[0], n[1]);
      let yaw = h < 1e-9 ? Math.round(this.yaw / (Math.PI / 2)) * (Math.PI / 2) : Math.atan2(-n[0], -n[1]);
      const pitch = -Math.asin(Math.max(-1, Math.min(1, n[2])));
      this.animateTo(yaw, pitch);
    },
    animateTo(yaw, pitch) {
      const d = ((yaw - this.yaw + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
      this.tween = { t0: performance.now(), y0: this.yaw, p0: this.pitch, dy: d, dp: pitch - this.pitch, ms: reduce ? 0 : 300 };
      this.dirty = true;
    },
    step(now) {
      const tw = this.tween;
      if (!tw) return;
      const u = tw.ms ? Math.max(0, Math.min(1, (performance.now() - tw.t0) / tw.ms)) : 1;
      const e = 1 - Math.pow(1 - u, 3);
      this.yaw = tw.y0 + tw.dy * e; this.pitch = tw.p0 + tw.dp * e;
      if (u >= 1) this.tween = null;
      this.dirty = true;
    },
    resize() {
      const r = $("viewer").getBoundingClientRect();
      this.dpr = Math.min(2, window.devicePixelRatio || 1);
      this.w = r.width; this.h = r.height;
      this.canvas.width = Math.max(1, Math.round(r.width * this.dpr));
      this.canvas.height = Math.max(1, Math.round(r.height * this.dpr));
      this.dirty = true;
    },
    /* The sites worth drawing: displayed (active, or cut away with ghosts on), under the layer cut, and, in 3D,
       in the outer cells of the model. A cell more than SHELL cells away from any cell that is not completely
       displayed is hidden behind the ones in front of it, whatever the viewing direction. */
    visibleSites(culled) {
      const r = this.res, key = `${this.layer}|${ui.ghost}|${culled}`;
      if (this.vis && this.visKey === key && this.visRes === r) return this.vis;
      const [b, L3, L2, L1] = r.lattice.dims, nc = L1 * L2 * L3, cellZ = r.lattice.cell;
      const shown = (i) => (this.D === 2 || cellZ[3 * i + 2] < this.layer) && (r.mask[i] || ui.ghost);
      let depth = null;
      if (this.D === 3 && culled) {
        const SHELL = b > 1 ? 0 : 1;
        depth = new Int16Array(nc).fill(-1);
        const full = new Uint8Array(nc).fill(1);
        for (let i = 0; i < r.lattice.n; i++) if (!shown(i)) full[i % nc] = 0;
        let front = [];
        for (let z = 0; z < L3; z++) for (let y = 0; y < L2; y++) for (let x = 0; x < L1; x++) {
          const c = (z * L2 + y) * L1 + x;
          // a cell on the box border or next to a cell that is not full can be seen
          let open = !full[c] || x === 0 || y === 0 || z === 0 || x === L1 - 1 || y === L2 - 1 || z === L3 - 1;
          for (let dz = -1; dz <= 1 && !open; dz++) for (let dy = -1; dy <= 1 && !open; dy++) for (let dx = -1; dx <= 1 && !open; dx++)
            if (!full[((z + dz) * L2 + y + dy) * L1 + x + dx]) open = true;
          if (open) { depth[c] = 0; front.push(c); }
        }
        for (let d = 1; d <= SHELL && front.length; d++) {
          const next = [];
          for (const c of front) {
            const x = c % L1, y = ((c - x) / L1) % L2, z = (c - x - y * L1) / (L1 * L2);
            for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
              const X = x + dx, Y = y + dy, Z = z + dz;
              if (X < 0 || Y < 0 || Z < 0 || X >= L1 || Y >= L2 || Z >= L3) continue;
              const k = (Z * L2 + Y) * L1 + X;
              if (depth[k] < 0) { depth[k] = d; next.push(k); }
            }
          }
          front = next;
        }
      }
      const out = [];
      for (let i = 0; i < r.lattice.n; i++) if (shown(i) && (!depth || depth[i % nc] >= 0)) out.push(i);
      this.vis = Int32Array.from(out); this.visKey = key; this.visRes = r;
      return this.vis;
    },
    syncLayers() {
      const L3 = this.res ? this.res.lattice.dims[1] : 1;
      $("layers-out").textContent = L3 > 1 ? `1–${this.layer} of ${L3}` : "";
    },
    load(r) {
      const msg = $("view-msg");
      if (!r || !r.colour) {
        this.res = null;
        msg.hidden = false;
        msg.textContent = r && r.errors.length ? "The model appears once the settings marked in red are fixed." : "";
        this.dirty = true;
        return;
      }
      const sameBox = this.res && this.res.lattice.n === r.lattice.n && this.D === r.crystal.D && this.res.nActive === r.nActive;
      const firstLoad = !this.res;
      this.res = r;
      const tooMany = r.lattice.n > 400000;
      msg.hidden = !tooMany;
      if (tooMany) msg.textContent = `${fmt(r.lattice.n)} sites are too many to draw here. The counts, colouring and files are still exact.`;
      this.D = r.crystal.D;
      const b0 = r.crystal.bonds.find((b) => b.shell === 1);
      const nn = SCK.bondVector(r.crystal.vectors, r.crystal.basis, b0.alpha, b0.alpha2, b0.delta).reduce((s, x) => s + x * x, 0) ** 0.5;
      this.radius = nn * 0.3;
      let ext = 0;
      const P = r.lattice.pos;
      for (let i = 0; i < r.lattice.n; i++) if (r.mask[i]) ext = Math.max(ext, Math.hypot(P[3 * i], P[3 * i + 1], P[3 * i + 2]));
      this.extent = ext + nn;
      const L3 = r.lattice.dims[1];
      const lw = $("layers-wrap"), lr = $("layers");
      lw.hidden = L3 < 2;
      lr.max = L3;
      if (!sameBox || +lr.value > L3) { this.layer = L3; lr.value = L3; }
      this.syncLayers();
      // first-shell bonds among active sites, for the sticks
      this.bondPairs = null;
      if (r.nActive <= 6000 && r.lattice.n <= 60000) {
        const dims = r.lattice.dims, box = [dims[3], dims[2], dims[1]], per = r.periodicUsed;
        const pairs = [];
        for (let i = 0; i < r.lattice.n; i++) {
          if (!r.mask[i]) continue;
          const a = r.lattice.alpha[i], c = [r.lattice.cell[3 * i], r.lattice.cell[3 * i + 1], r.lattice.cell[3 * i + 2]];
          for (const bd of r.crystal.bonds) {
            if (bd.shell !== 1 || bd.alpha !== a) continue;
            const c2 = c.slice(); let ok = true, wrapped = false;
            for (let k = 0; k < this.D; k++) {
              const t = c[k] + bd.delta[k];
              if (t < 0 || t >= box[k]) { if (per[k]) { wrapped = true; c2[k] = ((t % box[k]) + box[k]) % box[k]; } else ok = false; }
              else c2[k] = t;
            }
            if (!ok || wrapped) continue;                     // seam bonds would cross the whole model
            const j = ((bd.alpha2 * dims[1] + c2[2]) * dims[2] + c2[1]) * dims[3] + c2[0];
            if (j > i && r.mask[j]) pairs.push(i, j);
          }
        }
        this.bondPairs = new Int32Array(pairs);
      }
      $("opt-bonds").disabled = !this.bondPairs;
      $("gizmo").hidden = this.D !== 3;
      if (firstLoad || !sameBox) this.reset();
      this.legend();
      this.dirty = true;
    },
    sprite(col, alpha) {
      const key = col + alpha;
      if (this.sprites[key]) return this.sprites[key];
      const S = 64, c = document.createElement("canvas");
      c.width = c.height = S;
      const g = c.getContext("2d");
      g.globalAlpha = alpha;
      const grad = g.createRadialGradient(S * 0.36, S * 0.32, S * 0.04, S * 0.5, S * 0.5, S * 0.5);
      grad.addColorStop(0, "rgba(255,255,255,0.95)");
      grad.addColorStop(0.18, col);
      grad.addColorStop(0.75, col);
      grad.addColorStop(1, "rgba(0,0,0,0.55)");
      g.fillStyle = col; g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 1, 0, 2 * Math.PI); g.fill();
      g.fillStyle = grad; g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 1, 0, 2 * Math.PI); g.fill();
      return (this.sprites[key] = c);
    },
    styleOf(i) {
      const r = this.res;
      if (!r.mask[i]) return null;
      if (ui.view === "colour") {
        const k = r.colour[i];
        const focus = hoverClass >= 0 ? hoverClass : isolate;
        if (focus >= 0 && k !== focus) return "ghost";
        return classColour(k);
      }
      if (ui.view === "surface") return r.surface[i] ? SURFACE : BULK;
      return ACTIVE;
    },
    legend() {
      const el = $("legend");
      const dot = (c, t) => `<span><span class="ball" style="width:12px;height:12px;background:${c}"></span>${t}</span>`;
      el.innerHTML = !this.res ? "" : ui.view === "surface" ? dot(SURFACE, "surface (Kₛ)") + dot(BULK, "bulk (K₁)")
        : ui.view === "active" ? dot(ACTIVE, "active") + (ui.ghost ? `<span>· cut away</span>` : "")
        : "";
      $("hint").textContent = this.D === 3 ? "Drag to rotate · shift-drag to move · scroll to zoom" : "Drag to move · scroll to zoom";
    },
    draw() {
      const ctx = this.ctx, r = this.res;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (!r || r.lattice.n > 400000) return;
      // while the view moves, only the outer cells are drawn; once it rests, everything is drawn again
      const now = performance.now(), moving = now - (this.lastDraw || 0) < 250;
      this.lastDraw = now;
      clearTimeout(this.restTimer);
      if (moving) this.restTimer = setTimeout(() => { this.lastDraw = 0; this.dirty = true; }, 260);
      const css = getComputedStyle(document.documentElement);
      const ghostCol = css.getPropertyValue("--ghost").trim(), stickCol = css.getPropertyValue("--stick").trim();
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      const P = r.lattice.pos, n = r.lattice.n, cellZ = r.lattice.cell;
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const scale = (Math.min(this.w, this.h) * 0.46 / this.extent) * this.zoom;
      const ox = this.w / 2 + this.panX, oy = this.h / 2 + this.panY;
      if (!this.X || this.X.length !== n) { this.X = new Float32Array(n); this.Y = new Float32Array(n); this.Z = new Float32Array(n); }
      const X = this.X, Y = this.Y, Z = this.Z;
      const vis = this.visibleSites(moving && this.D === 3);
      let zmin = Infinity, zmax = -Infinity;
      for (const i of vis) {
        const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
        // yaw about z, then pitch about the screen x axis; z of the lattice points up on screen
        const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
        const depth = y1 * cp + z * sp;          // larger = farther
        const up = z * cp - y1 * sp;
        X[i] = ox + x1 * scale; Y[i] = oy - up * scale; Z[i] = depth;
        if (depth < zmin) zmin = depth;
        if (depth > zmax) zmax = depth;
      }
      // a bond is drawn only between two sites projected in this frame: the others hold stale positions
      if (!this.drawn || this.drawn.length !== n) this.drawn = new Uint8Array(n);
      const drawn = this.drawn;
      drawn.fill(0);
      for (const i of vis) drawn[i] = 1;
      const visible = (i) => drawn[i] === 1;
      // far to near, by a counting sort on depth (exact enough: a bucket is a small fraction of a ball)
      const NB = 4096, span = zmax - zmin || 1, count = new Int32Array(NB + 1);
      const bucket = (i) => NB - 1 - Math.min(NB - 1, Math.floor(((Z[i] - zmin) / span) * NB));
      for (const i of vis) count[bucket(i) + 1]++;
      for (let k = 0; k < NB; k++) count[k + 1] += count[k];
      const idx = new Int32Array(vis.length);
      for (const i of vis) idx[count[bucket(i)]++] = i;
      const rad = this.radius * scale;
      if (ui.bonds && this.bondPairs && rad > 2.5) {
        ctx.strokeStyle = stickCol; ctx.lineWidth = Math.max(1, rad * 0.28); ctx.lineCap = "round";
        ctx.beginPath();
        const bp = this.bondPairs;
        for (let k = 0; k < bp.length; k += 2) {
          const i = bp[k], j = bp[k + 1];
          if (!visible(i) || !visible(j)) continue;
          ctx.moveTo(X[i], Y[i]); ctx.lineTo(X[j], Y[j]);
        }
        ctx.stroke();
      }
      for (const i of idx) {
        const s = this.styleOf(i);
        if (s === null) {                               // a cut-away site
          ctx.fillStyle = ghostCol;
          ctx.beginPath(); ctx.arc(X[i], Y[i], Math.max(0.8, rad * 0.22), 0, 2 * Math.PI); ctx.fill();
        } else if (s === "ghost") {
          ctx.fillStyle = ghostCol;
          ctx.beginPath(); ctx.arc(X[i], Y[i], Math.max(1, rad * 0.45), 0, 2 * Math.PI); ctx.fill();
        } else if (rad < 2) {
          ctx.fillStyle = s; ctx.fillRect(X[i] - rad, Y[i] - rad, 2 * rad + 0.5, 2 * rad + 0.5);
        } else {
          ctx.drawImage(this.sprite(s, 1), X[i] - rad, Y[i] - rad, 2 * rad, 2 * rad);
        }
      }
      this.drawField(ctx, { cy, sy, cp, sp, scale, ox, oy }, css);
    },
    /* The field of the state shown: an arrow outside the model, pointing at it along H. Ising spins have no
       direction in space, so there is no arrow for them. */
    fieldDirection() {
      if (cfg.mode === "sweep" || cfg.model === "ising") return null;
      const st = cfg.states[ui.shown] || cfg.states[0];
      if (!st) return null;
      const num = (k) => { const v = Number(SCK.getParam(st, k)); return Number.isFinite(v) ? v : 0; };
      const H = num("H");
      if (!H) return null;
      const t = num("H_theta") * Math.PI / 180, f = num("H_phi") * Math.PI / 180;
      const h = cfg.model === "xy" ? [Math.cos(f), Math.sin(f), 0] : [Math.sin(t) * Math.cos(f), Math.sin(t) * Math.sin(f), Math.cos(t)];
      return H < 0 ? h.map((x) => -x) : h;
    },
    /* A fixed dial in the lower left corner: the field's direction as seen from the camera. The arrow shortens as
       the field turns towards or away from the viewer; along the line of sight it becomes ⊙ (towards) or ⊗ (away). */
    drawField(ctx, c, css) {
      const h = this.fieldDirection();
      if (!h) return;
      const x1 = h[0] * c.cy - h[1] * c.sy, y1 = h[0] * c.sy + h[1] * c.cy;
      const u = [x1, -(h[2] * c.cp - y1 * c.sp)], depth = y1 * c.cp + h[2] * c.sp;   // screen direction, depth
      const R = 30, cx = 18 + R, cy = this.h - 44 - R, len = Math.hypot(u[0], u[1]);
      const col = css.getPropertyValue("--field").trim() || "#14171a";
      ctx.save();
      ctx.strokeStyle = css.getPropertyValue("--line-strong").trim(); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.5; ctx.lineCap = "round";
      ctx.font = "italic 600 15px Georgia, 'Times New Roman', serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("H", cx + R + 6, cy - R + 6);
      if (len < 0.2) {
        if (depth < 0) { ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill(); }      // towards the viewer
        else { const k = 7; ctx.beginPath(); ctx.moveTo(cx - k, cy - k); ctx.lineTo(cx + k, cy + k); ctx.moveTo(cx + k, cy - k); ctx.lineTo(cx - k, cy + k); ctx.stroke(); }
        ctx.beginPath(); ctx.arc(cx, cy, 11, 0, 2 * Math.PI); ctx.stroke();
      } else {
        const L = (R - 4) * len, ux = u[0] / len, uy = u[1] / len, head = 10;
        const tx = cx - ux * L, ty = cy - uy * L, hx = cx + ux * L, hy = cy + uy * L;
        ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(hx - ux * head * 0.6, hy - uy * head * 0.6); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(hx, hy);
        ctx.lineTo(hx - ux * head - uy * head * 0.55, hy - uy * head + ux * head * 0.55);
        ctx.lineTo(hx - ux * head + uy * head * 0.55, hy - uy * head - ux * head * 0.55);
        ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    },
  };

  function setView(v) {
    ui.view = v;
    $("tabs").querySelectorAll("button").forEach((b) => { b.setAttribute("aria-selected", String(b.dataset.v === v)); b.tabIndex = b.dataset.v === v ? 0 : -1; });
    view.legend(); view.dirty = true; save();
  }

  // ------------------------------------------------------------------ start
  function start() {
    $("commit").textContent = `commit ${TEMPLATE.commit}`;
    seg("theme", ui.theme, (v) => { ui.theme = v; applyTheme(v); save(); });
    applyTheme(ui.theme);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { view.sprites = {}; view.dirty = true; });
    const tabs = $("tabs");
    tabs.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setView(b.dataset.v); });
    tabs.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      const bs = [...tabs.querySelectorAll("button")];
      const i = bs.findIndex((b) => b.dataset.v === ui.view);
      const j = (i + (e.key === "ArrowRight" ? 1 : -1) + bs.length) % bs.length;
      setView(bs[j].dataset.v); bs[j].focus(); e.preventDefault();
    });
    $("opt-bonds").checked = ui.bonds;
    $("opt-ghost").checked = ui.ghost;
    $("opt-bonds").addEventListener("change", (e) => { ui.bonds = e.target.checked; view.dirty = true; save(); });
    $("opt-ghost").addEventListener("change", (e) => { ui.ghost = e.target.checked; view.legend(); view.dirty = true; save(); });
    view.init();
    setView(ui.view);
    bindForm();
    bindSweep();
    bindDownload();
    seg("mode", cfg.mode, setMode);
    document.body.classList.toggle("mode-sweep", cfg.mode === "sweep");
    $("sweep-panel").hidden = cfg.mode !== "sweep";
    if (cfg.mode === "sweep") renderSweep();
    renderHamiltonian();
    analyse();
  }
  start();
})();
