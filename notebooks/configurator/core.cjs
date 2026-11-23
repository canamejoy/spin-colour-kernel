/* Core of the run configurator: a line-by-line port of the package's crystals.py, colouring.py, the active-site
 * shape and surface rules of simulate.py, and the writer of the notebook parameters cell. It runs in the browser
 * and under Node (tests/test_configurator.py compares every result with the Python package). No dependencies. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SCK = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------------ small helpers
  function product(ranges) {            // itertools.product(*ranges): last index varies fastest
    const out = [];
    const n = ranges.length;
    if (n === 0) return [[]];
    const idx = new Array(n).fill(0);
    if (ranges.some((r) => r.length === 0)) return out;
    for (;;) {
      out.push(idx.map((i, k) => ranges[k][i]));
      let k = n - 1;
      while (k >= 0) {
        idx[k]++;
        if (idx[k] < ranges[k].length) break;
        idx[k] = 0;
        k--;
      }
      if (k < 0) return out;
    }
  }
  const range = (a, b) => { const r = []; for (let i = a; i < b; i++) r.push(i); return r; };
  const pmod = (a, m) => ((a % m) + m) % m;
  const round9 = (x) => Math.round(x * 1e9) / 1e9;

  // ------------------------------------------------------------------ crystals.py
  const S3 = Math.sqrt(3.0);
  const TRI = [[1.0, 0.0], [0.5, S3 / 2]];
  const I2 = [[1, 0], [0, 1]], I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const TABLE = {
    square: [I2, [[0, 0]]],
    triangular: [TRI, [[0, 0]]],
    honeycomb: [TRI, [[0, 0], [1 / 3, 1 / 3]]],
    kagome: [TRI, [[0, 0], [0.5, 0], [0, 0.5]]],
    sc: [I3, [[0, 0, 0]]],
    bcc: [I3, [[0, 0, 0], [0.5, 0.5, 0.5]]],
    fcc: [I3, [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]],
  };
  const CRYSTALS = Object.keys(TABLE);

  function cart(frac, vectors) {        // frac @ vectors
    const D = vectors.length;
    const r = new Array(D).fill(0);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) r[j] += frac[i] * vectors[i][j];
    return r;
  }
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

  function bondVector(vectors, basis, a, a2, delta) {
    return cart(delta.map((d, k) => d + basis[a2][k] - basis[a][k]), vectors);
  }

  function shellDistances(vectors, basis, nshell, reach) {
    const D = vectors.length;
    const ds = new Set();
    const deltas = product(new Array(D).fill(range(-reach, reach + 1)));
    for (let a = 0; a < basis.length; a++)
      for (let a2 = 0; a2 < basis.length; a2++)
        for (const delta of deltas) {
          const d = norm(bondVector(vectors, basis, a, a2, delta));
          if (d > 1e-9) ds.add(round9(d));
        }
    return [...ds].sort((x, y) => x - y).slice(0, nshell);
  }

  function buildBonds(vectors, basis, shells, reach) {
    const dist = shellDistances(vectors, basis, Math.max(...shells), reach);
    const D = vectors.length;
    const deltas = product(new Array(D).fill(range(-reach, reach + 1)));
    const out = [];
    for (const s of [...shells].sort((x, y) => x - y)) {
      const target = dist[s - 1];
      for (let a = 0; a < basis.length; a++)
        for (let a2 = 0; a2 < basis.length; a2++)
          for (const delta of deltas) {
            const d = norm(bondVector(vectors, basis, a, a2, delta));
            if (Math.abs(d - target) < 1e-7) out.push({ alpha: a, alpha2: a2, delta: delta.slice(), shell: s });
          }
    }
    return out;
  }

  function makeCrystal(name, shells) {
    shells = shells || [1];
    if (!(name in TABLE)) throw new Error(`unknown crystal "${name}"; known: ${CRYSTALS.slice().sort().join(", ")}`);
    const [vectors, basis] = TABLE[name];
    const reach = Math.max(...shells) <= 2 ? 3 : 4;
    return { name, vectors, basis, D: vectors.length, b: basis.length, shells: shells.slice(),
             bonds: buildBonds(vectors, basis, shells, reach) };
  }

  // ------------------------------------------------------------------ colouring.py
  function abelianGroups(n) {
    if (n === 1) return [[]];
    const out = [];
    (function rec(rem, prev, acc) {
      if (rem === 1) { out.push(acc); return; }
      for (let d = 2; d <= rem; d++)
        if (rem % d === 0 && (prev === null || d % prev === 0)) rec(rem / d, d, acc.concat([d]));
    })(n, null, []);
    return out.filter((g) => g.every((x, i) => i === g.length - 1 || g[i + 1] % x === 0));
  }

  function nColours(col) { return col.m.reduce((p, x) => p * x, 1); }

  function colourOf(col, c, alpha) {
    let idx = 0;
    for (let r = 0; r < col.m.length; r++) {
      let s = col.o[alpha][r];
      for (let k = 0; k < c.length; k++) s += col.K[r][k] * c[k];
      idx = idx * col.m[r] + pmod(s, col.m[r]);
    }
    return idx;
  }

  function admissible(col, crystal) {
    for (const bd of crystal.bonds) {
      let allZero = true;
      for (let r = 0; r < col.m.length; r++) {
        let s = col.o[bd.alpha2][r] - col.o[bd.alpha][r];
        for (let k = 0; k < crystal.D; k++) s += col.K[r][k] * bd.delta[k];
        if (pmod(s, col.m[r]) !== 0) { allZero = false; break; }
      }
      if (allZero) return false;
    }
    return true;
  }

  function seamOk(col, box, periodic) {
    for (let k = 0; k < box.length; k++)
      if (periodic[k]) for (let r = 0; r < col.m.length; r++) if ((col.K[r][k] * box[k]) % col.m[r]) return false;
    return true;
  }

  function search(crystal, box, periodic, maxColours) {
    maxColours = maxColours || 32;
    const D = crystal.D, b = crystal.b;
    for (let n = 1; n <= maxColours; n++)
      for (const m of abelianGroups(n)) {
        const r = m.length;
        const elems = r ? product(m.map((mi) => range(0, mi))) : [[]];
        for (const cols of product(new Array(D).fill(elems))) {
          const K = range(0, r).map((i) => range(0, D).map((k) => cols[k][i]));
          for (const offs of product(new Array(b - 1).fill(elems))) {
            const col = { m, K, o: [new Array(r).fill(0)].concat(offs) };
            if (box && !seamOk(col, box, periodic || new Array(D).fill(true))) continue;
            if (admissible(col, crystal)) return col;
          }
        }
      }
    return null;
  }

  function verifyOnBox(col, crystal, box, periodic) {
    for (const c of product(box.map((L) => range(0, L))))
      for (const bd of crystal.bonds) {
        const c2 = [];
        let inside = true;
        for (let k = 0; k < crystal.D; k++) {
          let x = c[k] + bd.delta[k];
          if (periodic[k]) x = pmod(x, box[k]);
          else if (!(x >= 0 && x < box[k])) inside = false;
          c2.push(x);
        }
        if (inside && colourOf(col, c, bd.alpha) === colourOf(col, c2, bd.alpha2)) return false;
      }
    return true;
  }

  function describeColouring(col) {
    const grp = col.m.length ? col.m.map((x) => "Z" + x).join(" x ") : "trivial";
    return `${grp}; K=${JSON.stringify(col.K)}; o=${JSON.stringify(col.o)}`;
  }

  // ------------------------------------------------------------------ geometry (simulate.py)
  /* Every site of the box in the package's (alpha, z, y, x) order, x fastest, with its cell, Cartesian
   * position centred on the box (3 components) and flat index. */
  function lattice(crystal, box) {
    const D = crystal.D;
    if (box.length !== D) throw new Error(`crystal ${crystal.name} needs a box of ${D} sizes, got [${box.join(", ")}]`);
    const L1 = box[0], L2 = box[1], L3 = D === 3 ? box[2] : 1;
    const n = crystal.b * L3 * L2 * L1;
    const pos = new Float64Array(n * 3), cell = new Int32Array(n * 3), alpha = new Int32Array(n);
    let i = 0;
    for (let a = 0; a < crystal.b; a++)
      for (let z = 0; z < L3; z++)
        for (let y = 0; y < L2; y++)
          for (let x = 0; x < L1; x++) {
            const c = [x, y, z].slice(0, D);
            const p = cart(c.map((v, k) => v + crystal.basis[a][k]), crystal.vectors);
            pos[3 * i] = p[0]; pos[3 * i + 1] = p[1]; pos[3 * i + 2] = D === 3 ? p[2] : 0;
            cell[3 * i] = x; cell[3 * i + 1] = y; cell[3 * i + 2] = z;
            alpha[i] = a;
            i++;
          }
    const mean = [0, 0, 0];
    for (let j = 0; j < n; j++) for (let k = 0; k < 3; k++) mean[k] += pos[3 * j + k];
    for (let k = 0; k < 3; k++) mean[k] /= n;
    for (let j = 0; j < n; j++) for (let k = 0; k < 3; k++) pos[3 * j + k] -= mean[k];
    return { n, pos, cell, alpha, dims: [crystal.b, L3, L2, L1] };
  }

  function flatIndex(dims, a, z, y, x) { return ((a * dims[1] + z) * dims[2] + y) * dims[3] + x; }

  // ------------------------------------------------------------------ the shape expression (numpy semantics)
  const COORDS = { cartesian: ["x", "y", "z"], spherical: ["r", "theta", "phi"], cylindrical: ["rho", "phi", "z"] };
  const FUNCS = {
    sqrt: [1, Math.sqrt], sin: [1, Math.sin], cos: [1, Math.cos], tan: [1, Math.tan], arccos: [1, Math.acos],
    exp: [1, Math.exp], log: [1, Math.log], abs: [1, Math.abs], floor: [1, Math.floor], ceil: [1, Math.ceil],
    arctan2: [2, Math.atan2], minimum: [2, Math.min], maximum: [2, Math.max],
    mod: [2, (a, b) => a - Math.floor(a / b) * b],
  };

  class ShapeError extends Error {}

  function tokenize(src) {
    const toks = [];
    const re = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|(\*\*|\/\/|<=|>=|==|!=|[-+*\/%~&|^<>(),]))/y;
    let i = 0;
    while (i < src.length) {
      if (/^\s*$/.test(src.slice(i))) break;
      re.lastIndex = i;
      const m = re.exec(src);
      if (!m) throw new ShapeError(`cannot read "${src.slice(i).trim().slice(0, 12)}" in the shape`);
      if (m[1] !== undefined) toks.push({ t: "num", v: parseFloat(m[1]) });
      else if (m[2] !== undefined) toks.push({ t: "name", v: m[2] });
      else toks.push({ t: "op", v: m[3] });
      i = re.lastIndex;
    }
    return toks;
  }

  /* Python's precedence, lowest first: comparisons (chained), |, ^, &, + -, * / // %, unary + - ~, **. */
  function parseShape(src) {
    const toks = tokenize(src);
    let p = 0;
    const peek = () => toks[p];
    const isOp = (v) => toks[p] && toks[p].t === "op" && toks[p].v === v;
    const expect = (v) => {
      if (!isOp(v)) throw new ShapeError(`expected "${v}" in the shape`);
      p++;
    };
    function comparison() {
      const first = bor();
      const ops = [], rest = [];
      while (toks[p] && toks[p].t === "op" && ["<", "<=", ">", ">=", "==", "!="].includes(toks[p].v)) {
        ops.push(toks[p++].v);
        rest.push(bor());
      }
      return ops.length ? { k: "cmp", first, ops, rest } : first;
    }
    const binary = (next, ops) => function () {
      let left = next();
      while (toks[p] && toks[p].t === "op" && ops.includes(toks[p].v)) {
        const op = toks[p++].v;
        left = { k: "bin", op, left, right: next() };
      }
      return left;
    };
    function unary() {
      if (toks[p] && toks[p].t === "op" && ["-", "+", "~"].includes(toks[p].v)) {
        const op = toks[p++].v;
        return { k: "un", op, arg: unary() };
      }
      return power();
    }
    function power() {
      const base = atom();
      if (isOp("**")) { p++; return { k: "bin", op: "**", left: base, right: unary() }; }
      return base;
    }
    function atom() {
      const t = peek();
      if (!t) throw new ShapeError("the shape ends too early");
      if (t.t === "num") { p++; return { k: "num", v: t.v }; }
      if (t.t === "name") {
        p++;
        if (["and", "or", "not"].includes(t.v))
          throw new ShapeError(`"${t.v}" does not work on arrays: use & (and), | (or), ~ (not), with each condition in parentheses`);
        if (t.v === "pi") return { k: "num", v: Math.PI };
        if (isOp("(")) {
          p++;
          const args = [];
          if (!isOp(")")) { args.push(comparison()); while (isOp(",")) { p++; args.push(comparison()); } }
          expect(")");
          if (!(t.v in FUNCS)) throw new ShapeError(`unknown function "${t.v}"; known: ${Object.keys(FUNCS).join(", ")}`);
          if (FUNCS[t.v][0] !== args.length) throw new ShapeError(`${t.v} takes ${FUNCS[t.v][0]} argument(s)`);
          return { k: "call", f: t.v, args };
        }
        return { k: "var", v: t.v };
      }
      if (t.v === "(") { p++; const e = comparison(); expect(")"); return e; }
      throw new ShapeError(`unexpected "${t.v}" in the shape`);
    }
    const mul = binary(unary, ["*", "/", "//", "%"]);
    const add = binary(mul, ["+", "-"]);
    const band = binary(add, ["&"]);
    const bxor = binary(band, ["^"]);
    const bor = binary(bxor, ["|"]);
    if (!toks.length) throw new ShapeError("the shape is empty");
    const tree = comparison();
    if (p !== toks.length) {
      if (["and", "or", "not"].includes(toks[p].v))
        throw new ShapeError(`"${toks[p].v}" does not work on arrays: use & (and), | (or), ~ (not), with each condition in parentheses`);
      throw new ShapeError(`unexpected "${toks[p].v}" in the shape`);
    }
    return tree;
  }

  function shapeNames(tree, acc) {
    acc = acc || new Set();
    if (tree.k === "var") acc.add(tree.v);
    for (const key of ["first", "left", "right", "arg"]) if (tree[key]) shapeNames(tree[key], acc);
    for (const key of ["rest", "args"]) if (tree[key]) tree[key].forEach((t) => shapeNames(t, acc));
    return acc;
  }

  const num = (v) => (typeof v === "boolean" ? (v ? 1 : 0) : v);
  function logic(op, a, b) {
    if (typeof a !== "boolean" || (b !== undefined && typeof b !== "boolean"))
      throw new ShapeError(`"${op}" combines conditions: put each comparison in parentheses, as in (rho <= 5) & (abs(z) <= 1)`);
    return op === "&" ? a && b : op === "|" ? a || b : op === "^" ? a !== b : !a;
  }
  function evaluate(t, env) {
    switch (t.k) {
      case "num": return t.v;
      case "var": return env[t.v];
      case "call": return FUNCS[t.f][1](...t.args.map((a) => num(evaluate(a, env))));
      case "un": {
        const v = evaluate(t.arg, env);
        return t.op === "~" ? logic("~", v) : t.op === "-" ? -num(v) : num(v);
      }
      case "bin": {
        const a = evaluate(t.left, env), b = evaluate(t.right, env);
        if (t.op === "&" || t.op === "|" || t.op === "^") return logic(t.op, a, b);
        const x = num(a), y = num(b);
        switch (t.op) {
          case "+": return x + y;
          case "-": return x - y;
          case "*": return x * y;
          case "/": return x / y;
          case "//": return Math.floor(x / y);
          case "%": return x - Math.floor(x / y) * y;
          case "**": return Math.pow(x, y);
        }
        break;
      }
      case "cmp": {
        let left = num(evaluate(t.first, env));
        for (let i = 0; i < t.ops.length; i++) {
          const right = num(evaluate(t.rest[i], env));
          const ok = { "<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right,
                       "==": left === right, "!=": left !== right }[t.ops[i]];
          if (!ok) return false;
          left = right;
        }
        return true;
      }
    }
    throw new ShapeError("cannot evaluate the shape");
  }

  /* active_sites(): a Uint8Array mask over lattice(); both coords and shape empty keeps every site. */
  function activeMask(lat, coords, shape) {
    coords = (coords || "").trim().toLowerCase();
    shape = (shape || "").trim();
    if (coords && !(coords in COORDS))
      throw new ShapeError(`unknown coordinate system "${coords}"; known: ${Object.keys(COORDS).join(", ")}`);
    const mask = new Uint8Array(lat.n);
    if (!shape) { mask.fill(1); return mask; }
    if (!coords) throw new ShapeError("a shape needs its coordinate system: choose cartesian, spherical or cylindrical");
    const tree = parseShape(shape);
    for (const v of shapeNames(tree))
      if (!COORDS[coords].includes(v))
        throw new ShapeError(`name '${v}' is not defined; ${coords} coordinates are ${COORDS[coords].join(", ")}`);
    let any = false;
    for (let i = 0; i < lat.n; i++) {
      const x = lat.pos[3 * i], y = lat.pos[3 * i + 1], z = lat.pos[3 * i + 2];
      const r = Math.sqrt(x * x + y * y + z * z), phi = Math.atan2(y, x);
      const env = coords === "cartesian" ? { x, y, z }
        : coords === "spherical" ? { r, theta: Math.acos(r > 0 ? z / r : 1), phi }
        : { rho: Math.hypot(x, y), phi, z };
      const v = evaluate(tree, env);
      mask[i] = (typeof v === "boolean" ? v : v !== 0 && !Number.isNaN(v)) ? 1 : 0;
      any = any || mask[i] === 1;
    }
    if (!any) throw new ShapeError(`the shape "${shape}" contains no site of the box`);
    return mask;
  }

  /* _surface(): active sites with fewer active first-shell neighbours than a bulk site of their basis index. */
  function surfaceMask(crystal, box, periodic, lat, mask) {
    const dims = lat.dims;
    const full = new Array(crystal.b).fill(0);
    const count = new Int32Array(lat.n);
    const L = [box[0], box[1], crystal.D === 3 ? box[2] : 1];
    for (const bd of crystal.bonds) {
      if (bd.shell !== 1) continue;
      full[bd.alpha]++;
      for (let z = 0; z < dims[1]; z++)
        for (let y = 0; y < dims[2]; y++)
          for (let x = 0; x < dims[3]; x++) {
            const c = [x, y, z];
            let ok = true;
            for (let k = 0; k < crystal.D; k++) {
              const t = c[k] + bd.delta[k];
              if (periodic[k]) c[k] = pmod(t, L[k]);
              else if (t < 0 || t >= L[k]) { ok = false; break; }
              else c[k] = t;
            }
            if (ok && mask[flatIndex(dims, bd.alpha2, c[2], c[1], c[0])])
              count[flatIndex(dims, bd.alpha, z, y, x)]++;
          }
    }
    const out = new Uint8Array(lat.n);
    for (let i = 0; i < lat.n; i++) out[i] = mask[i] && count[i] < full[lat.alpha[i]] ? 1 : 0;
    return out;
  }

  // ------------------------------------------------------------------ the whole system (build_system)
  const MODELS = { ising: 1, xy: 2, heisenberg: 3 };
  const PARAM_KEYS = ["name", "J", "D", "K1", "Ks", "K2", "H", "H_theta", "H_phi"];

  function shellsOf(states) {           // the union over states of the shells with J != 0, plus 1
    const s = new Set([1]);
    for (const st of states) for (const [k, v] of Object.entries(st.J || { 1: 1 })) if (Number(v) !== 0) s.add(Number(k));
    return [...s].sort((a, b) => a - b);
  }

  /* Everything the preview shows for one configuration: lattice, active and surface sites, colouring, ladder. */
  function analyse(cfg) {
    const res = { errors: [], warnings: [] };
    const model = String(cfg.model || "").toLowerCase();
    if (!(model in MODELS)) res.errors.push({ field: "model", msg: `unknown model "${cfg.model}"` });
    const states = cfg.states || [];
    if (!states.length) res.errors.push({ field: "states", msg: "add at least one state" });
    const names = states.map((s, i) => s.name || `state-${i + 1}`);
    if (new Set(names).size !== names.length) res.errors.push({ field: "states", msg: "state names must be distinct" });
    states.forEach((st, i) => {
      if (model !== "heisenberg")
        for (const k of ["D", "K1", "Ks", "K2"])
          if (Number(st[k] || 0)) res.errors.push({ field: `state.${i}.${k}`, msg: `${k} needs the Heisenberg model; ${model} carries J and H only` });
      for (const [s, v] of Object.entries(st.J || {}))
        if (!Number.isFinite(Number(v))) res.errors.push({ field: `state.${i}.J`, msg: `J of shell ${s} is not a number` });
    });
    let crystal = null;
    try { crystal = makeCrystal(cfg.crystal, shellsOf(states)); } catch (e) { res.errors.push({ field: "crystal", msg: e.message }); }
    const box = (cfg.box || []).map(Number);
    if (crystal && (box.length !== crystal.D || box.some((L) => !Number.isInteger(L) || L < 1)))
      res.errors.push({ field: "box", msg: `${crystal.name} needs ${crystal.D} whole sizes of at least 1` });
    const T = ladder(cfg.T_max, cfg.T_min, cfg.T_step);
    if (T.error) res.errors.push({ field: "temperature", msg: T.error });
    else res.temperatures = T.values;
    if (!crystal || res.errors.some((e) => e.field === "box")) return res;

    res.crystal = crystal;
    const lat = lattice(crystal, box);
    res.lattice = lat;
    let mask;
    try { mask = activeMask(lat, cfg.coords, cfg.shape); } catch (e) { res.errors.push({ field: "shape", msg: e.message }); return res; }
    res.mask = mask;
    res.nActive = mask.reduce((s, v) => s + v, 0);

    let periodic = (cfg.periodic || []).slice(0, crystal.D).map(Boolean);
    while (periodic.length < crystal.D) periodic.push(false);
    let col = search(crystal, box, periodic);
    if (col && !verifyOnBox(col, crystal, box, periodic)) col = null;
    if (!col && periodic.some(Boolean)) {
      res.warnings.push({ field: "periodic", msg: `no admissible colouring fits the periodic box [${box.join(", ")}]: the run uses open borders` });
      periodic = periodic.map(() => false);
      col = search(crystal, box, periodic);
    }
    res.periodicUsed = periodic;
    if (!col) { res.errors.push({ field: "crystal", msg: "no linear colouring found within 32 colours" }); return res; }
    res.colouring = col;
    res.nColours = nColours(col);
    const colour = new Int32Array(lat.n);
    for (let i = 0; i < lat.n; i++)
      colour[i] = colourOf(col, [lat.cell[3 * i], lat.cell[3 * i + 1], lat.cell[3 * i + 2]].slice(0, crystal.D), lat.alpha[i]);
    res.colour = colour;
    res.surface = surfaceMask(crystal, box, periodic, lat, mask);
    res.nSurface = res.surface.reduce((s, v) => s + v, 0);
    if (res.temperatures) {
      const ce = Math.max(1, Math.floor(Number(cfg.config_every) || 1));
      const nSaved = res.temperatures.filter((_, t) => t % ce === 0).length;
      res.nSaved = nSaved;
      res.sizeGB = states.length * Number(cfg.replicas || 0) * nSaved * res.nActive * 3 * 4 / 1e9;
    }
    return res;
  }

  function ladder(Tmax, Tmin, Tstep) {
    Tmax = Number(Tmax); Tmin = Number(Tmin); Tstep = Number(Tstep);
    if (![Tmax, Tmin, Tstep].every(Number.isFinite)) return { error: "the temperatures must be numbers" };
    if (Tstep <= 0 || Tmin <= 0 || Tmax < Tmin) return { error: "need T max ≥ T min > 0 and a step > 0" };
    const k = Math.floor((Tmax - Tmin) / Tstep + 1e-9);
    if (k > 100000) return { error: "more than 100 000 temperatures: use a larger step" };
    const values = [];
    for (let i = 0; i <= k; i++) values.push(Math.round((Tmax - Tstep * i) * 1e10) / 1e10);
    return { values };
  }

  /* The values from..to by step, rounded so that 0.1 + 0.2 prints as 0.3; null when the range is unusable. */
  function series(from, to, step) {
    from = Number(from); to = Number(to); step = Number(step);
    if (![from, to, step].every(Number.isFinite) || step <= 0) return null;
    const k = Math.floor(Math.abs(to - from) / step + 1e-9);
    const dir = to >= from ? 1 : -1;
    const out = [];
    for (let i = 0; i <= k; i++) out.push(Math.round((from + dir * step * i) * 1e10) / 1e10);
    return out;
  }

  // ------------------------------------------------------------------ the parameters cell
  function pyStr(s) { return JSON.stringify(String(s)); }
  function pyNum(v) {
    const x = Number(v);
    if (!Number.isFinite(x)) throw new Error(`not a number: ${v}`);
    return Number.isInteger(x) ? x.toFixed(1) : String(x);
  }
  function pyInt(v) { return String(Math.trunc(Number(v))); }
  function pyBool(v) { return v ? "True" : "False"; }

  function stateLiteral(st, i) {
    const parts = [`name=${pyStr(st.name || `state-${i + 1}`)}`];
    const J = Object.entries(st.J || { 1: 1 }).filter(([, v]) => v !== "" && v !== null && v !== undefined);
    parts.push(`J={${J.map(([s, v]) => `${pyInt(s)}: ${pyNum(v)}`).join(", ")}}`);
    for (const k of ["D", "K1", "Ks", "K2", "H", "H_theta", "H_phi"])
      if (st[k] !== undefined && st[k] !== null && st[k] !== "") parts.push(`${k}=${pyNum(st[k])}`);
    return `    dict(${parts.join(", ")}),`;
  }

  /* Replaces the value of every setting in the notebook's own parameters cell, comments untouched. */
  function writeParams(template, cfg) {
    const set = {
      MODEL: pyStr(cfg.model), UNITS: pyStr(cfg.units), CRYSTAL: pyStr(cfg.crystal),
      BOX: `[${cfg.box.map(pyInt).join(", ")}]`, PERIODIC: `[${cfg.periodic.map(pyBool).join(", ")}]`,
      COORDS: pyStr(cfg.coords || ""), SHAPE: pyStr(cfg.shape || ""),
      T_MAX: pyNum(cfg.T_max), T_MIN: pyNum(cfg.T_min), T_STEP: pyNum(cfg.T_step),
      THERMALIZATION: pyInt(cfg.thermalization), MEASUREMENT: pyInt(cfg.measurement),
      MEASURE_EVERY: pyInt(cfg.measure_every), REPLICAS: pyInt(cfg.replicas), SEED: pyInt(cfg.seed),
      CONFIG_EVERY: pyInt(cfg.config_every), OUTPUT_DIR: pyStr(cfg.output_dir || "simulation_output"),
      GPUS: cfg.gpus === "all" || cfg.gpus === undefined ? '"all"' : `[${cfg.gpus.map(pyInt).join(", ")}]`,
    };
    const lines = template.split("\n");
    const out = [];
    let seen = new Set();
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (/^STATES = \[/.test(line)) {
        out.push("STATES = [");
        cfg.states.forEach((st, k) => out.push(stateLiteral(st, k)));
        while (i < lines.length && lines[i] !== "]") {
          if (/^\s+#/.test(lines[i])) out.push(lines[i]);       // the commented examples stay
          i++;
        }
        out.push("]");
        seen.add("STATES");
        continue;
      }
      const m = /^([A-Z_]+)(\s*=\s*)(.*?)(\s+#.*)?$/.exec(line);
      if (m && m[1] in set) {
        out.push(m[1] + m[2] + set[m[1]] + (m[4] || ""));
        seen.add(m[1]);
        continue;
      }
      out.push(line);
    }
    const missing = Object.keys(set).concat(["STATES"]).filter((k) => !seen.has(k));
    if (missing.length) throw new Error(`the template has no line for ${missing.join(", ")}`);
    return out.join("\n");
  }

  function notebookFor(tpl, cfg) {
    const nb = JSON.parse(JSON.stringify(tpl.notebook));
    const src = writeParams(tpl.params, cfg);
    nb.cells[tpl.paramsCell].source = src.split(/(?<=\n)/);
    return JSON.stringify(nb, null, 1) + "\n";
  }

  function scriptFor(tpl, cfg) {
    return tpl.script.replace(tpl.paramsMarker, () => writeParams(tpl.params, cfg));
  }

  return {
    CRYSTALS, MODELS, COORDS, FUNCS: Object.keys(FUNCS), PARAM_KEYS,
    product, makeCrystal, abelianGroups, search, seamOk, verifyOnBox, colourOf, nColours, describeColouring,
    lattice, activeMask, parseShape, surfaceMask, shellsOf, analyse, ladder, ShapeError,
    writeParams, notebookFor, scriptFor, bondVector, series,
  };
});
