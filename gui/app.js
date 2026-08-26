/* Console wiring. The model itself is in model.js; this file is only UI.
   Payload source: an inlined <script id="payload"> (standalone.html) if present,
   otherwise a fetch of payload.json (served build). */

const $ = (id) => document.getElementById(id);
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const fmt = (v, d = 2) =>
  v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (v, d = 2) => (v >= 0 ? "+" : "−") + fmt(Math.abs(v), d);

/* Absolute buckets, not per-frame quantiles: the point of sweeping the week is
   seeing demand actually move, which requantiling every frame would hide. */
const BREAKS = [1, 5, 15, 40, 100, 250];
const bucket = (v) => { let i = 0; while (i < BREAKS.length && v >= BREAKS[i]) i++; return i; };

/* ── Theme: remembered locally, since there is no host to stamp it ─────── */
/* Wrapped: the theme control is chrome, and chrome must never be able to stop
   the console from booting. matchMedia is guarded because embedded and headless
   browsers do not always provide it. */
const prefersDark = () => {
  try {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  } catch { return false; }
};

try {
  const btn = $("theme");
  const stored = (() => {
    try { return localStorage.getItem("console-theme"); } catch { return null; }
  })();
  const root = document.documentElement;
  const apply = (mode) => {
    if (mode) root.setAttribute("data-theme", mode);
    else root.removeAttribute("data-theme");
    btn.textContent = (mode === "dark" || (!mode && prefersDark())) ? "Light" : "Dark";
  };
  apply(stored);
  btn.addEventListener("click", () => {
    const dark = root.getAttribute("data-theme") === "dark" ||
      (!root.hasAttribute("data-theme") && prefersDark());
    const next = dark ? "light" : "dark";
    try { localStorage.setItem("console-theme", next); } catch { /* private mode */ }
    apply(next);
  });
} catch (err) {
  /* No theme toggle in this environment; the page still works. */
}

/* ── Boot ─────────────────────────────────────────────────────────────── */
(async function boot() {
  const inline = $("payload");
  let payload = null;
  if (inline) {
    payload = JSON.parse(inline.textContent);
  } else {
    try {
      const res = await fetch("payload.json");
      if (!res.ok) throw new Error("HTTP " + res.status);
      payload = await res.json();
    } catch (err) {
      $("loadfail").hidden = false;
      return;
    }
  }
  $("console").hidden = false;
  init(payload);
})();

function init(P) {
  P.byId = {};
  for (const z of P.zones) P.byId[z.id] = z;

  const state = {
    zone: 161, date: "2024-11-28", hour: 15,
    rain: 0, rainTouched: false,
    temp: null, tempTouched: false,
    force: false,
  };
  let lastCity = {};

  /* Zone picker, grouped by borough */
  const sel = $("zone"), groups = {};
  for (const z of P.zones) (groups[z.b] = groups[z.b] || []).push(z);
  for (const b of Object.keys(groups).sort()) {
    const og = document.createElement("optgroup");
    og.label = b;
    for (const z of groups[b]) {
      const o = document.createElement("option");
      o.value = z.id;
      o.textContent = `${z.n} · ${z.id}`;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  if (!P.byId[state.zone]) state.zone = P.zones[0].id;
  sel.value = String(state.zone);

  /* Map */
  const svg = $("map");
  svg.setAttribute("viewBox", `0 0 ${P.view.w} ${P.view.h}`);
  const zonePaths = {};
  for (const z of P.zones) {
    const d = P.paths[String(z.id)];
    if (!d) continue;
    const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
    el.setAttribute("d", d);
    el.setAttribute("class", "q0");
    el.dataset.zone = z.id;
    svg.appendChild(el);
    zonePaths[z.id] = el;
  }

  const legend = $("legend");
  ["under 1", "1 – 5", "5 – 15", "15 – 40", "40 – 100", "100 – 250", "250 +"]
    .forEach((t, i) => {
      const row = document.createElement("span");
      row.className = "sw";
      const sw = document.createElement("i");
      sw.style.background = `var(--seq${i})`;
      row.append(sw, document.createTextNode(t));
      legend.appendChild(row);
    });

  const tip = $("tip");
  svg.addEventListener("pointerover", (e) => {
    const zid = e.target.dataset && e.target.dataset.zone;
    if (!zid) return;
    const z = P.byId[zid], ch = P.chars[String(z.c)];
    const how = howOf(parts().dow, state.hour);
    tip.innerHTML = `<b>${z.n}</b><span class="n">${z.b} · zone ${z.id}</span>` +
      `<span class="n">forecast ${fmt(lastCity[zid], 1)} trips</span>` +
      `<span class="n">history ${fmt(P.hist[zid][how], 1)}</span>` +
      `<span class="n">${ch.label}</span>`;
    tip.classList.add("on");
  });
  svg.addEventListener("pointermove", (e) => {
    const pad = 14, r = tip.getBoundingClientRect();
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  });
  svg.addEventListener("pointerout", () => tip.classList.remove("on"));
  svg.addEventListener("click", (e) => {
    const zid = e.target.dataset && e.target.dataset.zone;
    if (!zid) return;
    state.zone = +zid;
    sel.value = zid;
    render();
  });

  function parts() {
    const [y, m, d] = state.date.split("-").map(Number);
    return { y, m, d, dow: dowOf(y, m, d) };
  }

  /* ── Render ─────────────────────────────────────────────────────────── */
  function render() {
    const { dow } = parts();
    const opts = {
      temp: state.tempTouched ? state.temp : null,
      precip: state.rainTouched ? state.rain : null,
      forceEvent: state.force,
    };

    const z = P.byId[state.zone];
    const cond = conditionsModel(P, state.zone, state.date, dow, state.hour, opts);
    const shape = shapeModel(P, state.zone, dow, state.hour);
    const hh = String(state.hour).padStart(2, "0");
    const when = `${DAYS[dow]} ${state.date} · ${hh}:00`;

    $("heroFig").textContent = fmt(cond.predicted, 2);
    $("heroWhere").innerHTML = `<strong>${z.n}</strong> · ${z.b} · zone ${z.id}`;
    $("whenLabel").textContent = when;
    $("mapWhen").textContent = when;
    $("hourLabel").textContent = `${DAYS[dow]} ${hh}:00`;

    const ch = P.chars[String(z.c)];
    $("heroShape").innerHTML =
      `Shape model (what the Kafka/Spark stream serves): <b>${fmt(shape.predicted, 2)}</b><br>` +
      `cluster ${z.c} — ${ch.label}, peaks ${ch.peak} · ` +
      `<b>${fmt(shape.multiple, 2)}×</b> this zone’s flat hourly average`;

    /* Waterfall — cumulative, so each delta visibly stacks onto the base */
    const steps = [
      { k: "History", v: cond.hist, abs: true },
      { k: "Calendar", v: cond.calendar },
      { k: "Weather", v: cond.weather },
      { k: "Events", v: cond.events },
    ];
    const scale = Math.max(cond.hist, cond.predicted,
      ...steps.map((s) => Math.abs(s.v)), 1) * 1.04;
    let cum = 0, html = "";
    for (const s of steps) {
      const from = s.abs ? 0 : cum, to = s.abs ? s.v : cum + s.v;
      cum = to;
      const lo = Math.max(0, Math.min(from, to)), hi = Math.max(from, to);
      const cls = (s.abs || s.v >= 0) ? "p" : "n";
      html += `<div class="r"><span class="k">${s.k}</span><span class="track">` +
        (s.abs ? "" : `<i class="zero" style="left:${(from / scale) * 100}%"></i>`) +
        `<i class="bar ${cls}" style="left:${(lo / scale) * 100}%;` +
        `width:${Math.max(0.4, ((hi - lo) / scale) * 100)}%"></i>` +
        `</span><span class="v">${s.abs ? fmt(s.v, 2) : signed(s.v, 2)}</span></div>`;
    }
    html += `<div class="r total"><span class="k">Forecast</span><span class="track">` +
      `<i class="bar p" style="left:0;width:${(cond.predicted / scale) * 100}%"></i>` +
      `</span><span class="v">${fmt(cond.predicted, 2)}</span></div>`;
    $("fall").innerHTML = html;

    const inp = cond.inputs, flags = [];
    if (inp.fedhol) flags.push(`federal holiday (${inp.holidayName})`);
    else if (inp.holiday) flags.push(`holiday (${inp.holidayName})`);
    if (inp.event) flags.push(`event (${inp.eventName || "forced"})`);
    $("assumed").textContent =
      `Inputs: ${fmt(inp.temp, 1)} °C, ${fmt(inp.precip, 1)} mm rain, ` +
      (flags.length ? flags.join(", ") : "no holiday, no event") +
      ` — ${Object.values(cond.assumed).join("; ")}` +
      (cond.clamped ? " · raw prediction was negative, clamped to 0" : "");
    $("dateFlags").textContent = flags.length ? flags.join(", ") : "ordinary day";

    lastCity = cityAt(P, state.date, dow, state.hour, opts);
    for (const zid in zonePaths) {
      zonePaths[zid].setAttribute("class",
        "q" + bucket(lastCity[zid]) + (+zid === state.zone ? " sel" : ""));
    }
    if (zonePaths[state.zone]) svg.appendChild(zonePaths[state.zone]);

    $("rainVal").textContent = state.rainTouched ? `${fmt(state.rain, 1)} mm` : "0 mm (assumed)";
    $("tempVal").textContent = state.tempTouched
      ? `${fmt(state.temp, 1)} °C`
      : `auto · ${fmt(inp.temp, 1)} °C normal`;
    if (!state.tempTouched) $("temp").value = inp.temp;
  }

  /* ── Controls ───────────────────────────────────────────────────────── */
  sel.addEventListener("change", (e) => { state.zone = +e.target.value; render(); });
  $("date").addEventListener("change", (e) => {
    if (!e.target.value) return;
    state.date = e.target.value;
    render();
  });
  $("hour").addEventListener("input", (e) => { state.hour = +e.target.value; render(); });
  $("rain").addEventListener("input", (e) => {
    state.rain = +e.target.value; state.rainTouched = true; render();
  });
  $("temp").addEventListener("input", (e) => {
    state.temp = +e.target.value; state.tempTouched = true; render();
  });
  $("force").addEventListener("change", (e) => { state.force = e.target.checked; render(); });
  $("resetCond").addEventListener("click", () => {
    state.rain = 0; state.rainTouched = false;
    state.temp = null; state.tempTouched = false;
    state.force = false;
    $("rain").value = 0;
    $("force").checked = false;
    render();
  });

  /* Sweep the week: one hour per tick, rolling the date forward */
  let sweep = null;
  const stopSweep = () => { clearInterval(sweep); sweep = null; $("play").textContent = "Sweep week"; };
  $("play").addEventListener("click", () => {
    if (sweep) { stopSweep(); return; }
    $("play").textContent = "Stop";
    let steps = 0;
    sweep = setInterval(() => {
      state.hour += 1;
      if (state.hour > 23) {
        state.hour = 0;
        const [y, m, d] = state.date.split("-").map(Number);
        const nd = new Date(y, m - 1, d + 1);
        if (nd.getFullYear() !== y && nd.getMonth() === 0) { stopSweep(); return; }
        state.date = `${nd.getFullYear()}-${String(nd.getMonth() + 1).padStart(2, "0")}-` +
                     `${String(nd.getDate()).padStart(2, "0")}`;
        $("date").value = state.date;
      }
      $("hour").value = state.hour;
      render();
      if (++steps >= 168) stopSweep();
    }, 420);
  });

  /* ── Presets: the moments worth showing ─────────────────────────────── */
  const PRESETS = [
    { t: "Thanksgiving, Midtown", s: "history overshoots by ~130 trips",
      z: 161, d: "2024-11-28", h: 15, rain: null },
    { t: "Saturday 01:00, East Village", s: "the nightlife cluster, 5 mm of rain",
      z: 79, d: "2024-01-13", h: 1, rain: 5 },
    { t: "Rainy Friday rush, Midtown", s: "9 mm on the busiest hour of the week",
      z: 161, d: "2024-11-22", h: 18, rain: 9 },
    { t: "New Year’s Eve, Times Sq", s: "an event day in the held-out window",
      z: 230, d: "2024-12-31", h: 22, rain: null },
  ];
  const presetWrap = $("presets");
  PRESETS.forEach((p) => {
    if (!P.byId[p.z]) return;                 // zone not in this modelling set
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset-btn";
    b.innerHTML = `<b>${p.t}</b><span>${p.s}</span>`;
    b.addEventListener("click", () => {
      if (sweep) stopSweep();
      state.zone = p.z; state.date = p.d; state.hour = p.h; state.force = false;
      state.rainTouched = p.rain !== null;
      state.rain = p.rain === null ? 0 : p.rain;
      state.tempTouched = false; state.temp = null;
      sel.value = String(p.z);
      $("date").value = p.d;
      $("hour").value = p.h;
      $("rain").value = state.rain;
      $("force").checked = false;
      render();
    });
    presetWrap.appendChild(b);
  });

  /* ── Evidence: baselines ────────────────────────────────────────────── */
  if (P.ladder.length) {
    const all = P.sig.find((s) => s.s === "all");
    const rows = P.ladder.map((r) => ({ ...r, ours: false }));
    if (all) {
      rows.push({ m: "Conditions model (linear + weather + events)",
                  wape: all.full / 100, ours: true });
    }
    rows.sort((a, b) => b.wape - a.wape);
    const max = Math.max(...rows.map((r) => r.wape));
    $("ladder").innerHTML = rows.map((r) => {
      const peeks = /Moving avg|Weighted MA|EWMA/.test(r.m);
      const name = r.m.replace(/\s*\(([^)]*)\)\s*$/, " <small>$1</small>");
      return `<div class="r${r.ours ? " ours" : ""}">` +
        `<span class="m">${name}` +
        (peeks ? ' <span class="peek">sees recent actuals</span>' : "") +
        `</span><span class="track"><i style="width:${(r.wape / max) * 100}%"></i></span>` +
        `<span class="w">${fmt(r.wape * 100, 1)}%</span></div>`;
    }).join("");
  }

  /* ── Evidence: significance ─────────────────────────────────────────── */
  if (P.sig.length) {
    const lo = Math.min(...P.sig.map((s) => s.lo)) - 4;
    const hi = Math.max(...P.sig.map((s) => s.hi)) + 4;
    const x = (v) => ((v - lo) / (hi - lo)) * 100;
    const NAMES = { all: "Full grid", rain_hours: "Rain hours", special_days: "Special days" };
    const order = ["rain_hours", "all", "special_days"];
    $("sig").innerHTML = order.map((key) => {
      const s = P.sig.find((r) => r.s === key);
      if (!s) return "";
      const decisive = s.lo > 0;
      return `<div class="row"><span class="nm">${NAMES[key]}` +
        `<small>${s.d} test days · ${fmt(s.hist, 2)}% → ${fmt(s.full, 2)}%</small></span>` +
        `<span class="ci"><i class="axis" style="left:${x(0)}%"></i>` +
        `<i class="whisk" style="left:${x(s.lo)}%;width:${x(s.hi) - x(s.lo)}%"></i>` +
        `<i class="cap" style="left:${x(s.lo)}%"></i><i class="cap" style="left:${x(s.hi)}%"></i>` +
        `<i class="pt" style="left:calc(${x(s.gain)}% - .3125rem)"></i>` +
        `<span class="tk" style="left:${x(0)}%">0</span>` +
        `<span class="tk" style="left:${x(s.lo)}%">${signed(s.lo, 1)}</span>` +
        `<span class="tk" style="left:${x(s.hi)}%">${signed(s.hi, 1)}</span></span>` +
        `<span class="verdict"><span class="pill ${decisive ? "good" : "soft"}">` +
        `${decisive ? "decisive" : "not decisive"}</span><br>` +
        `<span class="peek">${signed(s.gain, 1)}% · p ${s.p.toFixed(4)}</span></span></div>`;
    }).join("");
  }

  /* ── Evidence: cluster shapes ───────────────────────────────────────── */
  (function clusters() {
    const W = 336, H = 56;
    const keys = Object.keys(P.chars).sort((a, b) => P.chars[b].trips - P.chars[a].trips);
    $("clusters").innerHTML = keys.map((c) => {
      const s = P.share[c], ch = P.chars[c], max = Math.max(...s);
      const pt = (i) => [(i / 167) * W, H - (s[i] / max) * (H - 6) - 3];
      const xy = (i) => pt(i).map((v) => v.toFixed(1)).join(" ");
      const line = s.map((_, i) => (i ? "L" : "M") + xy(i)).join("");
      const area = `M0 ${H}L` + s.map((_, i) => xy(i)).join("L") + `L${W} ${H}Z`;
      const peak = s.indexOf(max), pp = pt(peak);
      const grid = [1, 2, 3, 4, 5, 6].map((d) => {
        const gx = ((d * 24) / 167) * W;
        return `<line x1="${gx}" y1="0" x2="${gx}" y2="${H}" stroke="var(--rule)" stroke-width="1"/>`;
      }).join("");
      return `<div class="cl"><div class="hd">` +
        `<i class="chip" style="background:var(--c${c})"></i>` +
        `<span class="nm">Cluster ${c}</span>` +
        `<span class="sub">${ch.n} zones · ${(ch.trips / 1e6).toFixed(1)}M trips</span></div>` +
        `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" ` +
        `aria-label="${ch.label}, peaks ${ch.peak}">${grid}` +
        `<path class="area" d="${area}"/>` +
        `<path class="line" d="${line}" vector-effect="non-scaling-stroke"/>` +
        `<circle class="pk" cx="${pp[0].toFixed(1)}" cy="${pp[1].toFixed(1)}" r="3"/></svg>` +
        `<div class="days">${DAYS.map((d) => `<span>${d}</span>`).join("")}</div>` +
        `<div class="sub">${ch.label} · peaks ${ch.peak}</div></div>`;
    }).join("");
  })();

  /* ── Evidence: scale + flows ────────────────────────────────────────── */
  $("bench").querySelector("tbody").innerHTML = P.bench.map((b) =>
    `<tr><td>${b.mo} month${b.mo > 1 ? "s" : ""}</td>` +
    `<td class="r">${(b.rin / 1e6).toFixed(2)}M</td>` +
    `<td class="r">${b.sec.toFixed(2)} s</td>` +
    `<td class="r">${(b.rps / 1e6).toFixed(2)}M rows/s</td></tr>`).join("");
  $("rules").querySelector("tbody").innerHTML = P.rules.map((r) =>
    `<tr><td>${r.pu}</td><td>${r.do}</td>` +
    `<td class="r">${r.lift.toFixed(2)}</td>` +
    `<td class="r">${(r.conf * 100).toFixed(1)}%</td></tr>`).join("");

  $("metaZones").textContent = `${P.meta.zones} zones`;
  $("metaK").textContent =
    `K=${P.meta.k} clusters (elbow ${P.meta.elbow}, silhouette ${P.meta.sil})`;

  /* ── Tabs ───────────────────────────────────────────────────────────── */
  document.querySelectorAll('[role="tab"]').forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll('[role="tab"]').forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        $(t.getAttribute("aria-controls")).hidden = !on;
      });
    });
  });

  render();
}
