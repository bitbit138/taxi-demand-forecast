/* Live stream page. Loads payload.json once (zone shapes, names, clusters — the
   same payload the model console uses) and then polls stream_state.json, which
   src/stream/spark_stream.py rewrites every few seconds. No model arithmetic
   happens here: every count and forecast shown is what the stream wrote. */

const $ = (id) => document.getElementById(id);
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const POLL_MS = 2000;
const FEED_ROWS = 600;   // rendered rows; the count in the header is always the total
const fmt = (v, d = 2) =>
  v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (v, d = 2) => (v >= 0 ? "+" : "−") + fmt(Math.abs(v), d);

/* Same absolute buckets as the model console, so the two choropleths read alike. */
const BREAKS = [1, 5, 15, 40, 100, 250];
const bucket = (v) => { let i = 0; while (i < BREAKS.length && v >= BREAKS[i]) i++; return i; };

/* ── Theme: shared key with the model console ─────────────────────────── */
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
} catch (err) { /* chrome only */ }

/* "2024-01-10 17:00" -> "Wed 2024-01-10 · 17:00" */
function whenLabel(stamp) {
  if (!stamp) return "—";
  const [d, t] = stamp.split(" ");
  const [y, m, day] = d.split("-").map(Number);
  return `${DAYS[new Date(y, m - 1, day).getDay()]} ${d} · ${t}`;
}
const hh = (stamp) => (stamp ? stamp.slice(11) : "—");
const stampSeconds = (stamp) => {
  const [d, t] = stamp.split(" ");
  const [y, m, day] = d.split("-").map(Number);
  const [H, M] = t.split(":").map(Number);
  return Date.UTC(y, m - 1, day, H, M) / 1000;
};
const dash = (v) => v.replace(/^(\d+) (\w+?)s?$/, "$1-$2");   // "2 hours" -> "2-hour"

/* ── Boot ─────────────────────────────────────────────────────────────── */
(async function boot() {
  let payload = null;
  try {
    const res = await fetch("payload.json", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    payload = await res.json();
  } catch (err) {
    $("loadfail").hidden = false;
    return;
  }
  $("stream").hidden = false;
  init(payload);
})();

function init(P) {
  P.byId = {};
  for (const z of P.zones) P.byId[z.id] = z;
  $("metaZones").textContent = `${P.zones.length} zones`;

  /* ── Map from the shared payload ────────────────────────────────────── */
  const svg = $("map");
  svg.setAttribute("viewBox", `0 0 ${P.view.w} ${P.view.h}`);
  const zonePaths = {};
  for (const z of P.zones) {
    const d = P.paths[String(z.id)];
    if (!d) continue;
    const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
    el.setAttribute("d", d);
    el.setAttribute("class", "none");
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
  const none = document.createElement("span");
  none.className = "sw";
  const nsw = document.createElement("i");
  nsw.style.background = "var(--surface)";
  nsw.style.border = "1px solid var(--rule-hi)";
  none.append(nsw, document.createTextNode("no pickups"));
  legend.appendChild(none);

  /* ── Page state ─────────────────────────────────────────────────────── */
  const ui = { mode: null, zone: null };   // mode: null = auto, "fill" | "closed"
  let S = null;                            // latest snapshot
  let view = { closed: [], open: [], filling: null, latest: null };

  const derive = (snap) => {
    const closed = snap.closed || [], open = snap.open || [];
    return {
      closed, open,
      latest: closed.length ? closed[closed.length - 1] : null,
      filling: open.length ? open[open.length - 1] : null,
    };
  };
  const mapWindow = () => {
    const mode = ui.mode || (view.filling ? "fill" : "closed");
    const w = mode === "fill" ? view.filling : view.latest;
    return { mode, w };
  };

  /* Tooltip: the stream's own numbers for the hovered zone */
  const tip = $("tip");
  svg.addEventListener("pointerover", (e) => {
    const zid = e.target.dataset && e.target.dataset.zone;
    if (!zid) return;
    const z = P.byId[zid], ch = P.chars[String(z.c)];
    let rows = `<b>${z.n}</b><span class="n">${z.b} · zone ${z.id} · cluster ${z.c}</span>`;
    const f = view.filling && view.filling.cells[zid];
    const l = view.latest && view.latest.cells[zid];
    if (view.filling) {
      rows += `<span class="n">${hh(view.filling.start)} filling: ${f ? fmt(f[0], 0) : 0} so far` +
        (f && f[1] != null ? ` · forecast ${fmt(f[1], 1)}` : "") + `</span>`;
    }
    if (view.latest) {
      rows += l
        ? `<span class="n">${hh(view.latest.start)} closed: ${fmt(l[0], 0)} actual · ` +
          `${l[1] == null ? "—" : fmt(l[1], 1)} predicted · error ${l[1] == null ? "—" : signed(l[1] - l[0], 1)}</span>`
        : `<span class="n">${hh(view.latest.start)} closed: no pickups</span>`;
    }
    if (!view.filling && !view.latest) rows += `<span class="n">no window yet</span>`;
    rows += `<span class="n">${ch ? ch.label : ""}</span>`;
    tip.innerHTML = rows;
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
    ui.zone = +zid;
    renderMap();
    renderZone();
  });
  $("modeFill").addEventListener("click", () => { ui.mode = "fill"; renderMap(); });
  $("modeClosed").addEventListener("click", () => { ui.mode = "closed"; renderMap(); });

  /* ── Renderers ──────────────────────────────────────────────────────── */
  function renderTiles() {
    const T = S.totals, pr = S.progress || {};
    const f = view.filling;
    $("tClock").textContent = pr.watermark ? whenLabel(pr.watermark) : "—";
    $("tClockSub").textContent = f
      ? `filling ${hh(f.start)} – ${hh(f.end)} · ${fmt(f.actual, 0)} trips so far`
      : (pr.watermark ? "no window open" : "nothing ingested yet");

    /* Replay speed: simulated seconds the watermark moved per real second. */
    const samples = (pr.samples || []).filter((s) => s[3]);
    let speed = null;
    if (samples.length >= 2) {
      const a = samples[0], b = samples[samples.length - 1];
      const dt = b[0] - a[0];
      if (dt >= 20 && b[3] !== a[3]) speed = (stampSeconds(b[3]) - stampSeconds(a[3])) / dt;
    }
    const mins = speed == null ? null : Math.round(speed) / 60;
    $("tSpeed").textContent = speed == null ? "—" : `≈${fmt(speed, 0)}×`;
    $("tSpeedSub").textContent = speed == null
      ? "measured once the watermark moves"
      : `1 real second ≈ ${fmt(mins, Number.isInteger(mins) ? 0 : 1)} simulated minute${mins >= 1.5 ? "s" : ""}`;

    $("tIngest").textContent = fmt(pr.input_rows || 0, 0);
    $("tIngestSub").textContent = pr.rate
      ? `${fmt(pr.rate, 0)} trips/s in the last micro-batch`
      : "from Kafka, after cleaning";
    renderSpark(pr.samples || []);

    $("tWindows").textContent = T.windows_closed;
    $("tWindowsSub").textContent = `closed · ${T.windows_open || 0} open` +
      (T.windows_open ? " (held back by the watermark)" : "");
    $("tCells").textContent = fmt(T.cells, 0);
    $("tCellsSub").textContent = S.batches.length
      ? `final pairs · ${S.batches.length} closing micro-batch${S.batches.length === 1 ? "" : "es"}`
      : "final (zone, window) pairs";
    $("tMae").textContent = T.mae == null ? "—" : fmt(T.mae, 2);
    $("tWape").textContent = T.wape == null ? "—" : fmt(T.wape * 100, 1) + "%";
    $("tWapeSub").textContent = T.actual ? `${fmt(T.actual, 0)} trips in closed windows` : "Σ|error| / Σactual";
    $("tWatermark").textContent = S.watermark || "2 hours";
    $("metaWindow").textContent =
      `${dash(S.window || "1 hour")} windows · ${dash(S.watermark || "2 hours")} watermark`;
  }

  function renderSpark(samples) {
    const el = $("spark");
    const pts = samples.slice(-120).map((s) => s[2]);
    if (pts.length < 2) { el.innerHTML = ""; return; }
    const max = Math.max(...pts, 1);
    const xy = pts.map((v, i) => [(i / (pts.length - 1)) * 120, 23 - (v / max) * 21]);
    const line = xy.map(([x, y], i) => (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1)).join("");
    el.innerHTML = `<path class="area" d="${line}L120 24L0 24Z"/><path d="${line}"/>`;
  }

  function renderMap() {
    const { mode, w } = mapWindow();
    $("modeFill").setAttribute("aria-pressed", mode === "fill");
    $("modeClosed").setAttribute("aria-pressed", mode === "closed");
    svg.classList.toggle("prov", mode === "fill");
    $("mapProv").hidden = mode !== "fill";
    $("mapTitle").textContent = mode === "fill"
      ? "Filling now · pickups so far" : "Latest closed window · actual demand";
    const cells = w ? w.cells : {};
    for (const zid in zonePaths) {
      const c = cells[zid];
      zonePaths[zid].setAttribute("class",
        (c ? "q" + bucket(c[0]) : "none") + (+zid === ui.zone ? " sel" : ""));
    }
    if (ui.zone && zonePaths[ui.zone]) svg.appendChild(zonePaths[ui.zone]);
    const desc = $("mapDesc");
    if (!w) {
      desc.textContent = mode === "fill"
        ? "No window is open yet — the first trips have not been ingested."
        : "No window has closed yet — the first closes once the watermark passes its end.";
    } else if (mode === "fill") {
      desc.textContent = `${whenLabel(w.start)} – ${hh(w.end)}: ${fmt(w.actual, 0)} pickups counted so far in ` +
        `${w.n} zones. Provisional — this hour keeps filling until the watermark passes ${hh(w.end)} + ` +
        `${S.watermark || "2 hours"}; then it closes, is scored and lands in the feed below.`;
    } else {
      desc.textContent = `${whenLabel(w.start)} – ${hh(w.end)}: ${fmt(w.actual, 0)} pickups in ${w.n} zones, ` +
        `final. The stream predicted ${fmt(w.predicted, 0)}; MAE ${fmt(w.mae, 2)} per zone. Hover for a zone's numbers.`;
    }
  }

  function renderZone() {
    const box = $("zone");
    if (!ui.zone) return;
    const z = P.byId[ui.zone], ch = P.chars[String(z.c)];
    $("zoneWhere").textContent = `${z.b} · zone ${z.id}`;
    const rows = [];
    for (const w of view.closed) {
      const c = w.cells[String(z.id)];
      rows.push({ h: hh(w.start), a: c ? c[0] : 0, p: c ? c[1] : null, open: false });
    }
    for (const w of view.open) {
      const c = w.cells[String(z.id)];
      rows.push({ h: hh(w.start), a: c ? c[0] : 0, p: c ? c[1] : null, open: true });
    }
    const max = Math.max(1, ...rows.map((r) => Math.max(r.a, r.p || 0)));
    let html = `<div class="zname">${z.n}</div>` +
      `<div class="zsub">cluster ${z.c} — ${ch ? ch.label : ""}${ch ? `, peaks ${ch.peak}` : ""} · ` +
      `level <b>${fmt(z.lvl, 2)}</b> trips/h</div>`;
    if (!rows.length) {
      html += `<p class="note" style="margin:.75rem 0 0;">No window has a count for this zone yet.</p>`;
    } else {
      html += `<div class="zrows"><div class="r"><span class="hd">hour</span><span class="hd">actual ▮ · predicted ▏</span>` +
        `<span class="hd v">act</span><span class="hd v">pred</span><span class="hd v">err</span></div>`;
      for (const r of rows.slice(-26).reverse()) {
        const e = r.p == null ? null : r.p - r.a;
        html += `<div class="r${r.open ? " open" : ""}"><span class="h">${r.h}${r.open ? "…" : ""}</span>` +
          `<span class="track"><i class="a" style="width:${(r.a / max) * 100}%"></i>` +
          (r.p == null ? "" : `<i class="p" style="width:${(r.p / max) * 100}%"></i>`) + `</span>` +
          `<span class="v">${fmt(r.a, 0)}</span><span class="v">${r.p == null ? "—" : fmt(r.p, 1)}</span>` +
          `<span class="v ${e == null || r.open ? "" : (e >= 0 ? "pos" : "neg")}">${e == null || r.open ? "—" : signed(e, 1)}</span></div>`;
      }
      html += `</div>`;
    }
    box.innerHTML = html;
  }

  function renderBatches() {
    const ol = $("batches");
    if (!S.batches.length) { ol.innerHTML = '<li class="empty">none yet</li>'; return; }
    ol.innerHTML = S.batches.slice().reverse().slice(0, 40).map((b) => {
      const t = new Date(b.t * 1000);
      const clock = `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}:${String(t.getSeconds()).padStart(2, "0")}`;
      const wins = b.windows.map(hh).join(", ");
      return `<li><span class="id">#${b.id}</span>` +
        `<span class="w">closed ${wins} <small>· ${b.cells} cells · ${clock}</small></span>` +
        `<span class="m">${fmt(b.actual, 0)} vs ${fmt(b.predicted, 0)} trips · MAE ${fmt(b.mae, 2)} trips</span></li>`;
    }).join("");
  }

  function renderChart() {
    const el = $("chart");
    const W = 900, H = 240, L = 48, R = 16, T = 14, B = 30;
    const series = [
      ...view.closed.map((w) => ({ h: hh(w.start), a: w.actual, p: w.predicted, open: false })),
      ...view.open.map((w) => ({ h: hh(w.start), a: w.actual, p: w.predicted, open: true })),
    ];
    if (!series.length) {
      el.innerHTML = `<text class="empty" x="${W / 2}" y="${H / 2}" text-anchor="middle">The curve draws itself as hours close.</text>`;
      return;
    }
    const n = Math.max(series.length, 8);
    const max = Math.max(1, ...series.map((s) => Math.max(s.a, s.p || 0))) * 1.08;
    const x = (i) => L + (i / Math.max(n - 1, 1)) * (W - L - R);
    const y = (v) => T + (1 - v / max) * (H - T - B);
    let out = "";
    const step = Math.pow(10, Math.floor(Math.log10(max))) / 2;
    const tick = max / step > 8 ? step * 2 : step;
    for (let v = 0; v <= max; v += tick) {
      out += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/>` +
        `<text class="axis" x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${fmt(v, 0)}</text>`;
    }
    const every = series.length > 16 ? 3 : (series.length > 8 ? 2 : 1);
    const lastLabelled = Math.floor((series.length - 1) / every) * every;
    series.forEach((s, i) => {
      // Label every N-th hour, plus the last one unless it would sit on a labelled neighbour.
      if (i % every === 0 || (i === series.length - 1 && i - lastLabelled >= every - 1 && every > 1)) {
        out += `<text class="axis" x="${x(i)}" y="${H - 10}" text-anchor="middle">${s.h}</text>`;
      }
    });
    const pred = series.filter((s) => s.p != null);
    if (pred.length) {
      out += `<path class="pred" d="${series.map((s, i) => s.p == null ? "" : (i === 0 ? "M" : "L") + x(i).toFixed(1) + " " + y(s.p).toFixed(1)).join("")}"/>`;
      series.forEach((s, i) => { if (s.p != null) out += `<circle class="pdot" cx="${x(i)}" cy="${y(s.p)}" r="2.5"/>`; });
    }
    const closedN = view.closed.length;
    if (closedN) {
      out += `<path class="act" d="${series.slice(0, closedN).map((s, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(s.a).toFixed(1)).join("")}"/>`;
    }
    if (view.open.length) {
      const from = Math.max(closedN - 1, 0);
      out += `<path class="act open" d="${series.slice(from).map((s, j) => (j ? "L" : "M") + x(from + j).toFixed(1) + " " + y(s.a).toFixed(1)).join("")}"/>`;
    }
    series.forEach((s, i) => {
      out += `<circle class="dot${s.open ? " open" : ""}" cx="${x(i)}" cy="${y(s.a)}" r="4"><title>${s.h}: ${fmt(s.a, 0)} actual${s.open ? " so far" : ""}${s.p == null ? "" : `, ${fmt(s.p, 0)} predicted`}</title></circle>`;
    });
    el.innerHTML = out;
  }

  let lastFeedKey = null, seenBatch = -1;
  function renderFeed() {
    const key = view.closed.map((w) => w.start).join("|");
    if (key === lastFeedKey) return;
    lastFeedKey = key;
    const tb = $("feed").querySelector("tbody");
    if (!view.closed.length) {
      tb.innerHTML = '<tr><td colspan="8" class="empty">Nothing has closed yet.</td></tr>';
      $("feedCount").textContent = "";
      return;
    }
    const rows = [];
    for (const w of view.closed.slice().reverse()) {
      const cells = Object.entries(w.cells).sort((a, b) => b[1][0] - a[1][0]);
      for (const [zid, [a, p]] of cells) rows.push({ w, zid, a, p });
      if (rows.length >= FEED_ROWS) break;
    }
    const total = view.closed.reduce((s, w) => s + w.n, 0);
    const newest = view.closed[view.closed.length - 1].batch;
    tb.innerHTML = rows.slice(0, FEED_ROWS).map(({ w, zid, a, p }) => {
      const z = P.byId[zid] || { n: `zone ${zid}`, b: "", c: "" };
      const e = p == null ? null : p - a;
      const cls = e == null ? "" : (e >= 0 ? "pos" : "neg");
      return `<tr${w.batch === newest && newest !== seenBatch ? ' class="new"' : ""}>` +
        `<td>${w.start} – ${hh(w.end)}</td><td>${z.n}</td><td>${z.b}</td>` +
        `<td class="r"><i class="chip" style="background:var(--c${z.c})"></i>${z.c}</td>` +
        `<td class="r">${fmt(a, 0)}</td><td class="r">${p == null ? "—" : fmt(p, 1)}</td>` +
        `<td class="r ${cls}">${e == null ? "—" : signed(e, 1)}</td>` +
        `<td class="r">${w.batch}</td></tr>`;
    }).join("");
    seenBatch = newest;
    $("feedCount").textContent = total > FEED_ROWS
      ? `latest ${FEED_ROWS} of ${fmt(total, 0)}` : `${fmt(total, 0)} cells`;
  }

  let lastVerdictKey = null;
  function renderVerdict() {
    const v = S.verdict;
    const key = S.status + ":" + JSON.stringify(v);
    if (key === lastVerdictKey) return;
    lastVerdictKey = key;
    const box = $("verdict"), pill = $("vPill");
    $("vFacts").hidden = true;
    $("vNote").textContent = "";
    if (S.status === "validating") {
      box.dataset.state = "validating";
      pill.textContent = "checking";
      $("vTitle").textContent = "Consumer stopped — comparing every streamed cell with demand.parquet…";
      return;
    }
    if (S.status !== "finished") {
      box.dataset.state = "pending";
      pill.textContent = "pending";
      $("vTitle").innerHTML = S.run_seconds
        ? `Runs after the consumer stops (${Math.round(S.run_seconds / 60)} min). Every streamed cell is compared with <code>demand.parquet</code>.`
        : "Runs after the consumer stops. Every streamed cell is compared with <code>demand.parquet</code>.";
      return;
    }
    if (!v) {
      box.dataset.state = "pending";
      pill.textContent = "skipped";
      $("vTitle").textContent = "Consumer finished without --validate.";
      return;
    }
    if (v.reason) {
      box.dataset.state = "fail";
      pill.textContent = "no result";
      $("vTitle").textContent = `Nothing to compare: ${v.reason}.`;
      return;
    }
    box.dataset.state = v.ok ? "pass" : "fail";
    pill.textContent = v.ok ? "pass" : "mismatch";
    $("vTitle").textContent = v.ok
      ? "Every streamed cell matches the batch aggregate exactly."
      : `${fmt(v.mismatched, 0)} cells disagree between the stream and the batch aggregate.`;
    $("vStreamed").textContent = fmt(v.streamed, 0);
    $("vBatch").textContent = fmt(v.batch, 0);
    $("vCells").textContent = fmt(v.cells, 0);
    $("vBad").textContent = fmt(v.mismatched, 0);
    $("vFacts").hidden = false;
    $("vNote").textContent = `${v.windows} closed window${v.windows === 1 ? "" : "s"} compared cell for cell ` +
      "(zone × hour) against the batch demand table, restricted to the modelling zones.";
  }

  function renderStatus(ageS) {
    const st = $("status"), txt = $("statusText"), age = $("statusAge");
    const T = S.totals, pr = S.progress || {};
    let state = "running", text = "";
    if (S.status === "running") {
      if (!pr.input_rows) {
        text = "Consumer running — waiting for the producer's first trips.";
      } else if (!T.cells) {
        text = `Ingesting — ${fmt(pr.input_rows, 0)} trips so far, watermark at ${pr.watermark || "—"}. ` +
          `The first hour closes once the watermark passes its end by ${S.watermark || "2 hours"}.`;
      } else {
        text = `Streaming — watermark ${pr.watermark}, ${T.windows_closed} window${T.windows_closed === 1 ? "" : "s"} closed` +
          (view.filling ? `, ${hh(view.filling.start)} filling (${fmt(view.filling.actual, 0)} trips so far)` : "") + ".";
      }
      if (ageS > 15) {
        state = "stale";
        text += " No update for a while — the consumer may have stopped; check its window.";
      }
    } else if (S.status === "validating") {
      state = "validating";
      text = "Consumer stopped — validating streamed windows against the batch aggregate…";
    } else if (S.status === "finished") {
      const v = S.verdict;
      if (v && v.ok) { state = "pass"; text = `Finished — validation passed: ${fmt(v.streamed, 0)} streamed = ${fmt(v.batch, 0)} batch trips over ${v.windows} windows, 0 cells disagree.`; }
      else if (v && v.reason) { state = "fail"; text = `Finished — nothing to validate (${v.reason}).`; }
      else if (v) { state = "fail"; text = `Finished — validation FAILED: ${fmt(v.mismatched, 0)} cells disagree (${fmt(v.streamed, 0)} streamed vs ${fmt(v.batch, 0)} batch).`; }
      else { state = "stale"; text = "Consumer finished (no --validate)."; }
    }
    st.dataset.state = state;
    txt.textContent = text;
    age.textContent = S.status === "finished" ? "" : `updated ${ageS}s ago`;
    renderElapsed();
  }

  function renderElapsed() {
    const el = $("statusElapsed");
    if (!S || !S.started_at) { el.textContent = ""; return; }
    // Stopwatch in REAL time since the consumer started; frozen at the final
    // snapshot once the run is finished.
    const end = S.status === "finished" ? S.updated_at : Date.now() / 1000;
    const t = Math.max(0, Math.round(end - S.started_at));
    const clock = `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
    const total = S.run_seconds ? ` / ~${Math.round(S.run_seconds / 60)} min` : "";
    el.textContent = S.status === "finished"
      ? `ran ${clock} min` : `⏱ ${clock}${total} elapsed`;
  }

  function renderWaiting(reason) {
    $("status").dataset.state = "waiting";
    $("statusElapsed").textContent = "";
    $("statusText").textContent = reason ||
      "Waiting for the consumer — stream_state.json is not there yet. Start the demo: python run.py demo";
    $("statusAge").textContent = "";
  }

  const ageOf = (snap) => Math.max(0, Math.round(Date.now() / 1000 - snap.updated_at));
  function render(snap) {
    S = snap;
    view = derive(snap);
    renderTiles();
    renderMap();
    renderZone();
    renderBatches();
    renderChart();
    renderFeed();
    renderVerdict();
    renderStatus(ageOf(snap));
  }

  /* ── Poll ───────────────────────────────────────────────────────────── */
  let lastText = null;
  async function poll() {
    try {
      const res = await fetch("stream_state.json", { cache: "no-store" });
      if (res.status === 404) { renderWaiting(); lastText = null; return; }
      if (!res.ok) throw new Error("HTTP " + res.status);
      const text = await res.text();
      if (text !== lastText) {
        lastText = text;
        render(JSON.parse(text));
      } else if (S) {
        renderStatus(ageOf(S));
      }
    } catch (err) {
      renderWaiting("The server did not answer — is python run.py demo (or http.server) still running?");
    }
  }
  poll();
  setInterval(poll, POLL_MS);
  setInterval(() => { if (S) renderElapsed(); }, 1000);   // tick the stopwatch every second
}
