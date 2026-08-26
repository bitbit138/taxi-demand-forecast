/* The fitted model, served in the browser.
   Mirrors src/stream/predict_live.py exactly: the shape model is
   level x cluster_share x 168, the conditions model applies the OLS
   coefficients ablation.py exported to models/conditions_model.json.
   Verified against the Python implementation over random queries. */
const HOW = 168, FOURIER = 3, TWO_PI = 2 * Math.PI;

const dowOf = (y, m, d) => (new Date(y, m - 1, d).getDay() + 6) % 7;   // Mon=0
const howOf = (dow, hour) => dow * 24 + hour;

function shapeModel(P, zoneId, dow, hour) {
  const z = P.byId[zoneId];
  const share = P.share[String(z.c)][howOf(dow, hour)];
  return { predicted: z.lvl * share * HOW, share, level: z.lvl, cluster: z.c,
           multiple: share * HOW };
}

function conditionsModel(P, zoneId, date, dow, hour, opts) {
  const how = howOf(dow, hour), m = P.model, coef = m.coefficients;
  const hist = P.hist[String(zoneId)][how];
  const month = parseInt(date.slice(5, 7), 10);

  const assumed = {};
  let temp = opts.temp;
  if (temp === null || temp === undefined) {
    temp = m.monthly_temp_normals_c[String(month)];
    assumed.temp = `monthly normal for ${date.slice(5, 7)}`;
  }
  let precip = opts.precip;
  if (precip === null || precip === undefined) {
    precip = m.default_precip_mm;
    assumed.precip = "no rain assumed";
  }

  const ev = P.events[date];
  let holiday = ev ? !!ev.h : false, fedhol = ev ? !!ev.f : false,
      event = ev ? !!ev.e : false;
  assumed.flags = ev ? `calendar entry for ${date}` : `${date} is an ordinary day`;
  if (opts.forceEvent && !event) { event = true; assumed.flags += " (event forced)"; }

  const dev = temp - m.train_mean_temp_c;
  const f = {
    hist_avg_demand: hist,
    hour_sin: Math.sin(TWO_PI * hour / 24), hour_cos: Math.cos(TWO_PI * hour / 24),
    dow_sin: Math.sin(TWO_PI * dow / 7), dow_cos: Math.cos(TWO_PI * dow / 7),
    weekend_d: (dow === 5 || dow === 6) ? 1 : 0,
    temp_c: temp, precip_mm: precip,
    temp_dev_x_hist: dev * hist, precip_x_hist: precip * hist,
    holiday_d: holiday ? 1 : 0, fedhol_d: fedhol ? 1 : 0, event_d: event ? 1 : 0,
    fedhol_x_hist: (fedhol ? 1 : 0) * hist, event_x_hist: (event ? 1 : 0) * hist,
  };
  for (let k = 1; k <= FOURIER; k++) {
    const a = TWO_PI * k * how / HOW;
    f["how_sin_" + k] = Math.sin(a);
    f["how_cos_" + k] = Math.cos(a);
  }

  const contrib = {};
  let sum = 0;                       // sum first, then add intercept, as in Python
  for (const name in coef) { const c = coef[name] * f[name]; contrib[name] = c; sum += c; }
  const raw = m.intercept + sum;

  const WEATHER = ["temp_c", "precip_mm", "temp_dev_x_hist", "precip_x_hist"];
  const EVENTS = ["holiday_d", "fedhol_d", "event_d", "fedhol_x_hist", "event_x_hist"];
  const add = (keys) => keys.reduce((t, k) => t + (contrib[k] || 0), 0);
  const weather = add(WEATHER), events = add(EVENTS);

  return {
    predicted: Math.max(0, raw), raw, clamped: raw < 0, hist,
    weather, events, calendar: raw - hist - weather - events,
    inputs: { temp, precip, holiday, fedhol, event,
              holidayName: ev ? ev.hn : "", eventName: ev ? ev.en : "" },
    assumed,
  };
}

/* Every modelling zone at one hour-of-week, under the same conditions —
   this is what the map paints. */
function cityAt(P, date, dow, hour, opts) {
  const out = {};
  for (const z of P.zones) {
    out[z.id] = conditionsModel(P, z.id, date, dow, hour, opts).predicted;
  }
  return out;
}

if (typeof module !== "undefined") {
  module.exports = { shapeModel, conditionsModel, cityAt, dowOf, howOf };
}
