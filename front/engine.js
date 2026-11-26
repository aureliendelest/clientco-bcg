// List building and captured revenue. No DOM here, so that it can be run and checked under Node.
//
// Policies, all under the same K per list and the same rest period:
//   model : highest expected value p x V (or p x revenue of the last 90 days x 4)
//   size  : biggest clients (V), no model
//   oracle: knows who leaves; biggest churners not captured yet (an indicative ceiling, built list by list)
//
// A client listed less than `restDays` days before cannot be listed again. A churner is "captured" when listed at a
// date where he is a churner (no purchase in the next 180 days); he is counted once, with the same fixed value
// whatever the cadence (`loss`: V at the first weekly date where he is a churner).

const DAY_MS = 86400000;

function dayNumber(iso) {
  return Math.round(Date.parse(iso + "T00:00:00Z") / DAY_MS);
}

function listDates(data, cadenceWeeks) {
  return data.dates.filter((_, i) => i % cadenceWeeks === 0);
}

const orderCache = new WeakMap();

// Row indices of one date sorted by decreasing score, ties by client id. For a score ranking, only the first `depth`
// rows are the true top of the whole perimeter (the embedded candidates hold them all); the oracle has every churner.
function order(date, name, depth) {
  let cache = orderCache.get(date);
  if (!cache) orderCache.set(date, (cache = {}));
  if (!cache[name]) {
    const score = name === "oracle" ? date.loss : date[name];
    let idx = date.id.map((_, i) => i);
    if (name === "oracle") idx = idx.filter((i) => date.gone[i] === 1);
    idx.sort((a, b) => score[b] - score[a] || date.id[a] - date.id[b]);
    cache[name] = name === "oracle" ? idx : idx.slice(0, depth);
  }
  return cache[name];
}

function runPolicy(dates, ks, restDays, ranking, depth) {
  const lastListed = new Map();
  const captured = new Set();
  return dates.map((date, n) => {
    const day = dayNumber(date.date);
    const k = ks[n];
    const rows = [];
    for (const i of order(date, ranking, depth)) {
      if (rows.length >= k) break;
      const id = date.id[i];
      const last = lastListed.get(id);
      if (last !== undefined && day - last < restDays) continue;
      if (ranking === "oracle" && captured.has(id)) continue;
      rows.push(i);
    }
    let listV = 0, newV = 0, churners = 0, newChurners = 0, calledBefore = 0;
    const before = [];
    for (const i of rows) {
      const id = date.id[i];
      before.push(lastListed.has(id));
      if (lastListed.has(id)) calledBefore++;
      if (date.gone[i] === 1) {
        churners++;
        listV += date.v[i];
        if (!captured.has(id)) {
          captured.add(id);
          newChurners++;
          newV += date.loss[i];
        }
      }
    }
    for (const i of rows) lastListed.set(date.id[i], day);
    return { date, k, rows, before, calledBefore, churners, newChurners, listV, newV,
             short: ranking !== "oracle" && rows.length < k };
  });
}

// params: {cadenceWeeks, ks (one K per list), restDays, ranking: "ev" | "evr"}
function simulate(data, params) {
  const dates = listDates(data, params.cadenceWeeks);
  return {
    model: runPolicy(dates, params.ks, params.restDays, params.ranking, data.depth[params.ranking]),
    size: runPolicy(dates, params.ks, params.restDays, "v", data.depth.v),
    oracle: runPolicy(dates, params.ks, params.restDays, "oracle"),
  };
}

function totals(lists) {
  const sum = (f) => lists.reduce((s, l) => s + f(l), 0);
  return {
    calls: sum((l) => l.rows.length),
    capturedV: sum((l) => l.newV),
    churners: sum((l) => l.newChurners),
    distinct: new Set(lists.flatMap((l) => l.rows.map((i) => l.date.id[i]))).size,
  };
}

if (typeof module !== "undefined") module.exports = { simulate, totals, listDates };
