// GTA VI trophy rarity forecast V2 (Node 22, standard library only).
//
// Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
// earn rate on Exophase at the case's target, as five quantiles (0.05 / 0.25 /
// 0.5 / 0.75 / 0.95) in percent. Reads every <id>.in in the input directory
// and writes one <id>.out per case. Deterministic: same input, same bytes.
//
// V2 rebuilds V1's single-model forecast as an ENSEMBLE, the fix the
// sister Twitch forecast's top 10 all used (their losses 0.066-0.080 with
// 90% coverage 0.83-1.00, against our single model's 0.199 / 0.48 there).
// V1 here scored 0.170 on the practice mid scenario with coverage 1.00 --
// its median was decent but its band was far too wide, so the loss was
// mostly spread cost. V2 attacks both halves:
//
//  1. Median: seven views of the same trophy, combined in logit space.
//     M1 direct WLS logit trend vs ln(1+hours), slope shrunk to a tier
//     prior (V1's model). M2 kernel-weighted local-linear fit: a Gaussian
//     kernel around the latest history bends with recent flattening that a
//     global line misses. M3 log-ratio to a reference trophy: ratios
//     cancel the common dilution from Exophase's player growth, so the
//     ratio trend is flatter and steadier than any absolute rate. M4 a
//     concave quadratic in ln(1+hours), which can saturate where a straight
//     line keeps climbing past week one. M5 the house-curve OLS over the
//     last 24h. M6 a pooled slope: one slope fitted across all trophies
//     with per-trophy intercepts, which borrows strength when a trophy's
//     own history is thin. M7 the launch-dilution prior (V1's), which
//     carries flat histories where no trend exists at all.
//  2. The MEDIAN comes only from the three members whose extrapolation is
//     sound at 4.5 days -- the direct trend, the log-ratio and the concave
//     quadratic -- weighted by long-horizon rolling-origin scores (each
//     member refitted on history up to each past snapshot, scored
//     predicting the last snapshot). The other three members (local-linear,
//     house curve, pooled slope) do NOT carry the median: origin scoring
//     cannot validate a 4.5-day extrapolation, and on the synthetic final
//     it ranked the steep-growth chasers first while they overshot the
//     flat week-one regime by up to +0.7 logit. Those members instead set
//     the spread, through the ensemble's disagreement.
//  3. No bias correction is applied: the signed rolling-origin error
//     measures early-growth behaviour that does not carry over to the flat
//     week-one regime.
//  3. Spread from honest parts, calibrated to ~0.9 coverage instead of
//     V1's 1.00: the rolling-origin residual scale, the ensemble's own
//     disagreement (when the members disagree the band widens), a smaller
//     random-walk term, the refresh-jump scale from the history's change
//     points, and slope uncertainty. The downside leg still carries the
//     full refresh term (refreshes at launch mostly push rates down) and
//     the asymmetry fades as the checkpoint ages.
//
// Two logical orderings are then enforced on the medians: story chapters
// must not rise (finishing Chapter 2 implies Chapter 1), and the platinum
// -- which needs every other trophy -- must not exceed any other trophy.
// Quantiles shift with their median, so spreads are preserved.
//
// Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
// /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  slot: string;
  at: string;
  players_tracked: number | null;
  trophies_earned: number | null;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const parseTime = (s: string): number => Date.parse(s);

/** Prior slope of logit(rate) against ln(1 + hours), by tier and story spot. */
function priorSlope(t: Trophy): number {
  let base: number;
  if (t.tier === 'platinum') base = 0.3;
  else if (t.tier === 'gold') base = 0.45;
  else if (t.tier === 'silver') base = 0.65;
  else base = 1.05;
  const chapter = /Complete Chapter (\d)/.exec(t.description);
  if (chapter !== null) {
    // Earlier chapters are reached sooner, so they climb faster.
    base += [0, 0.6, 0.5, 0.4, 0.3][Number(chapter[1])] ?? 0.2;
  } else if (/Complete the final mission/.test(t.description)) {
    base += 0.1;
  }
  return Math.min(1.8, Math.max(0.1, base));
}

interface WeightedFit {
  /** Intercept and slope of y against x. */
  a: number;
  b: number;
  /** Weighted mean of x, and the residual scatter around the line. */
  xBar: number;
  residSd: number;
  /** Standard error of the slope (0 when the fit is degenerate). */
  slopeSe: number;
  n: number;
}

/** Weighted least squares of y on x. Returns null when x never varies. */
function weightedFit(xs: number[], ys: number[], ws: number[]): WeightedFit | null {
  const n = xs.length;
  if (n < 2) return null;
  let wSum = 0;
  let xBar = 0;
  let yBar = 0;
  for (let i = 0; i < n; i++) {
    wSum += ws[i];
    xBar += ws[i] * xs[i];
    yBar += ws[i] * ys[i];
  }
  xBar /= wSum;
  yBar /= wSum;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += ws[i] * (xs[i] - xBar) * (xs[i] - xBar);
    sxy += ws[i] * (xs[i] - xBar) * (ys[i] - yBar);
  }
  if (sxx < 1e-12) return null;
  const b = sxy / sxx;
  const a = yBar - b * xBar;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (a + b * xs[i]);
    rss += ws[i] * e * e;
  }
  const residSd = n >= 3 ? Math.sqrt(rss / wSum * (n / (n - 2))) : 0.35;
  const slopeSe = n >= 3 ? residSd / Math.sqrt(sxx / wSum * n) : 0.35;
  return { a, b, xBar, residSd, slopeSe, n };
}

/** Weighted quadratic fit y ~ a + b x + c x^2, with c clipped to <= 0 so the
 *  curve can saturate but never accelerate upwards. Solved by 3x3 Gaussian
 *  elimination on the weighted normal equations. Null when degenerate. */
function weightedQuad(xs: number[], ys: number[], ws: number[]): { a: number; b: number; c: number; xBar: number; residSd: number; n: number } | null {
  const n = xs.length;
  if (n < 3) return null;
  // Normal-equation sums for basis [1, x, x^2].
  const s = [0, 0, 0, 0, 0];
  const t = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const w = ws[i];
    const x = xs[i];
    const xp = [1, x, x * x, x * x * x, x * x * x * x];
    for (let k = 0; k < 5; k++) s[k] += w * xp[k];
    for (let k = 0; k < 3; k++) t[k] += w * ys[i] * xp[k];
  }
  // Matrix [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]] * [a,b,c] = [t0,t1,t2].
  const m = [[s[0], s[1], s[2], t[0]], [s[1], s[2], s[3], t[1]], [s[2], s[3], s[4], t[2]]];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    for (let r = col + 1; r < 3; r++) {
      const f = m[r][col] / m[col][col];
      for (let k = col; k < 4; k++) m[r][k] -= f * m[col][k];
    }
  }
  const coef = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let v = m[r][3];
    for (let k = r + 1; k < 3; k++) v -= m[r][k] * coef[k];
    coef[r] = v / m[r][r];
  }
  let c = Math.min(0, coef[2]);
  // Shrink the curvature halfway toward a straight line: with few points a
  // full quadratic overfits the bend it just saw.
  c *= 0.5;
  const a = coef[0];
  const b = coef[1];
  let rss = 0;
  let wSum = 0;
  let xBar = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (a + b * xs[i] + c * xs[i] * xs[i]);
    rss += ws[i] * e * e;
    wSum += ws[i];
    xBar += ws[i] * xs[i];
  }
  xBar /= wSum;
  const residSd = Math.sqrt(rss / wSum * (n / Math.max(1, n - 3)));
  return { a, b, c, xBar, residSd, n };
}

interface Snapshot {
  /** Hours from release to the first read showing this snapshot. */
  hours: number;
  players: number;
  rates: Array<number | null>;
}

/** Collapse the history to one snapshot per distinct page state, keeping the
 *  first read that showed it (the moment a refresh became visible). */
function distinctSnapshots(c: Case, releaseMs: number): Snapshot[] {
  const out: Snapshot[] = [];
  let prevKey = '';
  for (const h of c.history) {
    const key = `${h.players_tracked}|${h.rates.join(',')}`;
    if (key === prevKey) continue;
    prevKey = key;
    out.push({
      hours: (parseTime(h.at) - releaseMs) / 3_600_000,
      players: typeof h.players_tracked === 'number' && h.players_tracked > 0
        ? h.players_tracked
        : (out.length > 0 ? out[out.length - 1].players : 638),
      rates: h.rates,
    });
  }
  return out;
}

/** Refresh-jump scale from the history's own change points: the scatter of
 *  the median per-trophy logit move at each change. Falls back to a
 *  launch-decaying prior when the history has too few changes. V1's prior
 *  (0.95) made the practice band far too wide -- the first refresh's move
 *  is mostly a COMMON shift, not per-trophy scatter -- so the prior is
 *  0.55 here, and the ensemble's disagreement carries the rest. */
function refreshScale(snaps: Snapshot[], checkpointHours: number): number {
  if (snaps.length < 3) return 0.55 * Math.exp(-checkpointHours / 48);
  const moves: number[] = [];
  for (let j = 1; j < snaps.length; j++) {
    const deltas: number[] = [];
    for (let i = 0; i < snaps[j].rates.length; i++) {
      const a = snaps[j - 1].rates[i];
      const b = snaps[j].rates[i];
      if (a === null || b === null) continue;
      if (a <= 0.05 || a >= 95 || b <= 0.05 || b >= 95) continue;
      deltas.push(logit(b) - logit(a));
    }
    if (deltas.length >= 5) {
      deltas.sort((x, y) => x - y);
      moves.push(deltas[Math.floor(deltas.length / 2)]);
    }
  }
  if (moves.length < 2) return 0.55 * Math.exp(-checkpointHours / 48);
  const sorted = [...moves].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  const absDev = moves.map((x) => Math.abs(x - med)).sort((a, b) => a - b);
  const mad = absDev[Math.floor(absDev.length / 2)];
  return Math.min(0.45, Math.max(0.15, 1.4826 * mad));
}

interface Series {
  xs: number[];
  ys: number[];
  es: number[];
  ws: number[];
  xLast: number;
  lastRate: number;
  nDistinct: number;
}

/** Per-trophy series over the given snapshots. Rates at or above 99% are
 *  censored out of the fit: a 100% rate in a tiny early cohort is not a
 *  measurement of the rate, its clamped logit of 9.21 would poison the
 *  line. Recency weights decay 0.2 per unit of x behind the last point. */
function buildSeries(snaps: Snapshot[], idx: number): Series {
  const last = snaps[snaps.length - 1];
  const lastRate = last.rates[idx] ?? 1.0;
  const xLast = Math.log(1 + Math.max(last.hours, -0.9));
  const xs: number[] = [];
  const ys: number[] = [];
  const es: number[] = [];
  const ws: number[] = [];
  let nDistinct = 0;
  for (const s of snaps) {
    const r = s.rates[idx];
    if (r === null || r === undefined) continue;
    nDistinct++;
    if (r >= 99.0) continue;
    const x = Math.log(1 + Math.max(s.hours, -0.9));
    xs.push(x);
    ys.push(logit(r));
    es.push(Math.log(Math.max((r / 100) * s.players, 0.5)));
    ws.push(Math.exp(-0.2 * (xLast - x)));
  }
  return { xs, ys, es, ws, xLast, lastRate, nDistinct };
}

/** Shrink a fitted slope toward the tier prior: with nEff points of evidence
 *  the slope is (nEff * b + k * prior) / (nEff + k), intercept re-centred so
 *  the line still passes through the weighted mean point. */
function shrunkLine(fit: WeightedFit, ys: number[], ws: number[], prior: number, k: number, xTarget: number): number {
  const nEff = Math.min(fit.n, 12);
  const b = (nEff * fit.b + k * prior) / (nEff + k);
  const yBar = ys.reduce((s, y, i) => s + ws[i] * y, 0) / ws.reduce((s, w) => s + w, 0);
  const a = yBar - b * fit.xBar;
  return a + b * xTarget;
}

/** M1: direct WLS logit trend, slope shrunk to the tier prior. Null when the
 *  series has fewer than 2 usable points. */
function memberDirect(s: Series, prior: number, xTarget: number): number | null {
  const fit = weightedFit(s.xs, s.ys, s.ws);
  if (fit === null) return null;
  return shrunkLine(fit, s.ys, s.ws, prior, 3.5, xTarget);
}

/** M2: kernel-weighted local-linear fit. A Gaussian kernel around the latest
 *  history (bandwidth 0.7 in x) lets the line bend with recent flattening;
 *  slope shrunk lightly (k = 1.5) because a local fit has less data. */
function memberLocal(s: Series, prior: number, xTarget: number): number | null {
  const h = 1.0;
  const ws = s.xs.map((x, i) => s.ws[i] * Math.exp(-((x - s.xLast) * (x - s.xLast)) / (2 * h * h)));
  const fit = weightedFit(s.xs, s.ys, ws);
  if (fit === null) return null;
  // Same strong shrinkage as the direct fit: a local line with a light
  // shrink chases recent steep growth and overshoots the flat regime.
  return shrunkLine(fit, s.ys, ws, prior, 3.5, xTarget);
}

/** M4: concave quadratic in x, which can saturate where a straight line
 *  keeps climbing past week one. Slope shrunk to the prior as in M1. */
function memberQuad(s: Series, prior: number, xTarget: number): number | null {
  const q = weightedQuad(s.xs, s.ys, s.ws);
  if (q === null) return null;
  // Re-centre through the weighted mean point with the shrunk slope, then
  // add the (negative) curvature term beyond the mean.
  const nEff = Math.min(q.n, 12);
  const fit = weightedFit(s.xs, s.ys, s.ws);
  if (fit === null) return null;
  const b = (nEff * fit.b + 3.5 * prior) / (nEff + 3.5);
  const yBar = s.ys.reduce((sum, y, i) => sum + s.ws[i] * y, 0) / s.ws.reduce((sum, w) => sum + w, 0);
  const a = yBar - b * q.xBar - q.c * q.xBar * q.xBar;
  return a + b * xTarget + q.c * xTarget * xTarget;
}

/** M5: house-curve OLS over the last 24h of reads, unweighted, slope shrunk
 *  to the tier prior like M1 (the unshrunk version extrapolates the recent
 *  slope 4.5 days and overshoots the flattening curve). Null with fewer
 *  than 3 reads. `reads` is the case history pre-parsed once (ms,
 *  x = ln(1+hours), rates); only reads at or before cutoffMs count. */
function memberHouseCurve(reads: Array<{ ms: number; x: number; rates: Array<number | null> }>, idx: number, prior: number, xTarget: number, cutoffMs: number): number | null {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const h of reads) {
    if (h.ms > cutoffMs || cutoffMs - h.ms > 24 * 3_600_000) continue;
    const r = h.rates[idx];
    if (r === null || r === undefined || r >= 99.0) continue;
    xs.push(h.x);
    ys.push(logit(r));
  }
  if (xs.length < 3) return null;
  const fit = weightedFit(xs, ys, xs.map(() => 1));
  if (fit === null) return null;
  return shrunkLine(fit, ys, xs.map(() => 1), prior, 3.5, xTarget);
}

/** Indices of the trophies sharing trophy i's tier (for the tier pool). */
function tierIdx(c: Case, i: number): number[] {
  const tier = c.trophies[i].tier;
  const out: number[] = [];
  c.trophies.forEach((t, k) => { if (t.tier === tier) out.push(k); });
  return out;
}

/** M6: pooled slope. One slope fitted across the trophies of the same tier
 *  (demeaned y on demeaned x, pooled), per-trophy intercept. Borrows
 *  strength when a trophy's own history is thin. Pooling across ALL tiers
 *  drags fast story trophies down toward slow grind trophies (on the
 *  synthetic final it sat 0.76 logit below the truth for a story trophy).
 *  Falls back to the global pool when the tier has too few fitted trophies. */
function pooledSlope(all: Series[], tierIdx: number[] | null, trophies: Trophy[]): number {
  const pool = (idxs: number[]): number | null => {
    let sxx = 0;
    let sxy = 0;
    let used = 0;
    for (const k of idxs) {
      const s = all[k];
      if (s.xs.length < 2) continue;
      used++;
      const wSum = s.ws.reduce((a, b) => a + b, 0);
      const xBar = s.xs.reduce((a, x, i) => a + s.ws[i] * x, 0) / wSum;
      const yBar = s.ys.reduce((a, y, i) => a + s.ws[i] * y, 0) / wSum;
      for (let i = 0; i < s.xs.length; i++) {
        sxx += s.ws[i] * (s.xs[i] - xBar) * (s.xs[i] - xBar);
        sxy += s.ws[i] * (s.xs[i] - xBar) * (s.ys[i] - yBar);
      }
    }
    if (used < 3 || sxx < 1e-12) return null;
    return Math.min(1.8, Math.max(0.1, sxy / sxx));
  };
  if (tierIdx !== null) {
    const v = pool(tierIdx);
    if (v !== null) return v;
  }
  const allIdx = all.map((_, k) => k);
  return pool(allIdx) ?? 0.9;
}

function memberPooled(s: Series, bPool: number, xTarget: number): number | null {
  if (s.xs.length < 1) return null;
  const wSum = s.ws.reduce((a, b) => a + b, 0);
  const xBar = s.xs.reduce((a, x, i) => a + s.ws[i] * x, 0) / wSum;
  const yBar = s.ys.reduce((a, y, i) => a + s.ws[i] * y, 0) / wSum;
  return yBar + bPool * (xTarget - xBar);
}

/** M3: log-ratio to a reference trophy. The ratio's trend is fitted (slope
 *  shrunk toward 0 -- ratios drift slowly) and added to the reference
 *  trophy's own M1 median, so common dilution cancels out of the ratio. */
function memberRatio(
  s: Series, ref: Series, refMedian: number | null, prior: number, xTarget: number,
): number | null {
  if (refMedian === null || s.xs.length < 2 || ref.xs.length < 2) return null;
  // Align the two series on shared x values (same snapshots, same order).
  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  const refByX = new Map<number, number>();
  ref.xs.forEach((x, i) => refByX.set(Math.round(x * 1e9), ref.ys[i]));
  s.xs.forEach((x, i) => {
    const ry = refByX.get(Math.round(x * 1e9));
    if (ry === undefined) return;
    xs.push(x);
    ys.push(s.ys[i] - ry);
    ws.push(s.ws[i]);
  });
  if (xs.length < 2) return null;
  const fit = weightedFit(xs, ys, ws);
  if (fit === null) return null;
  // Ratio slope shrunk toward 0 (k = 4 against nEff points).
  const nEff = Math.min(fit.n, 12);
  const b = (nEff * fit.b) / (nEff + 4);
  const yBar = ys.reduce((sum, y, i) => sum + ws[i] * y, 0) / ws.reduce((sum, w) => sum + w, 0);
  const a = yBar - b * fit.xBar;
  return refMedian + (a + b * xTarget);
}

/** Fallback median when no trend fit exists: last rate grown at the tier
 *  prior over the extrapolation in x. */
function fallbackMedian(s: Series, prior: number, xTarget: number): number {
  return logit(s.lastRate) + prior * Math.max(0, xTarget - s.xLast);
}

/** Launch-dilution prior (V1's): players keep growing at the launch rate
 *  over the stale stretch plus the horizon, earners keep up with only a
 *  share of that growth. Two V2 changes: the share's ceiling drops from
 *  0.85 to 0.70 -- V1's 0.78x for the 100% first trophy sat far above every
 *  plausible refresh (0.30-0.65x) -- and earners catch up a little over the
 *  horizon itself, so a later practice checkpoint dilutes less than an
 *  earlier one. Returns [median logit, spread]. */
function dilutionPrior(
  c: Case, s: Series, checkpointHours: number, horizonHours: number,
): { logit: number; spread: number; alpha: number } {
  const firstMs = parseTime(c.history[0].at);
  const staleHours = Math.max(0, (parseTime(c.at) - firstMs) / 3_600_000);
  const g = 0.38 * Math.exp(-checkpointHours / 36);
  const growLog = Math.min(Math.log(8), g * (staleHours + horizonHours));
  let alpha = Math.min(0.7, Math.max(0.18, 0.18 + 0.55 * Math.pow(s.lastRate / 100, 0.3)));
  // Earners catch up over the horizon: the cohort ages into the trophy.
  alpha = alpha + (1 - alpha) * (1 - Math.exp(-horizonHours / 24));
  const rateDil = clampRate(s.lastRate * Math.exp(-(1 - alpha) * growLog));
  return { logit: logit(rateDil), spread: 0.5 * (1 - alpha) * growLog, alpha };
}

/** The six snapshot-based ensemble members, in order M1, M2, M3, M4, M6, M5.
 *  Each returns a median logit at xTarget, or null when it cannot fit.
 *  M5 (house curve) is scored separately on reads; it joins the ensemble
 *  at the mean weight of the scored members. */
const N_MEMBERS = 6;

interface MemberSet {
  /** Median logit per member (fallback substituted where a member is null). */
  medians: number[];
  /** True where the member actually fitted (false = fallback substituted). */
  fitted: boolean[];
  /** M1 slope standard error, for the spread's slope term. */
  slopeSe: number;
  /** M1 fit's weighted mean x, for the extrapolation distance. */
  xBar: number;
}

function memberSet(
  c: Case, snaps: Snapshot[], series: Series[], idx: number, refIdx: number,
  refMedian: number | null, bPool: number, reads: Array<{ ms: number; x: number; rates: Array<number | null> }>,
  cutoffMs: number, computeHouse: boolean, xTarget: number,
): MemberSet {
  const s = series[idx];
  const t = c.trophies[idx];
  const prior = priorSlope(t);
  const fb = fallbackMedian(s, prior, xTarget);
  const m1 = memberDirect(s, prior, xTarget);
  const m2 = memberLocal(s, prior, xTarget);
  const m3 = memberRatio(s, series[refIdx], refMedian, prior, xTarget);
  const m4 = memberQuad(s, prior, xTarget);
  const m6 = memberPooled(s, bPool, xTarget);
  // M5 is not scored in the rolling origin (it fits reads, and its weight
  // is the mean of the scored members), so skip computing it there.
  const m5 = computeHouse ? memberHouseCurve(reads, idx, prior, xTarget, cutoffMs) : null;
  const raw = [m1, m2, m3, m4, m6, m5];
  const fit = weightedFit(s.xs, s.ys, s.ws);
  return {
    medians: raw.map((m) => (m === null ? fb : m)),
    fitted: raw.map((m) => m !== null),
    slopeSe: fit !== null ? fit.slopeSe : 0.35,
    xBar: fit !== null ? fit.xBar : s.xLast,
  };
}

/** Long-horizon rolling-origin evaluation. For each past snapshot j >= 3,
 *  every member is refitted on the snapshots before j and scored predicting
 *  the LAST snapshot's rate -- the same shape as the real task (fit early,
 *  predict late). Scoring the next snapshot instead rewarded the members
 *  that chase steep early growth (local-linear, house curve), which then
 *  overshot the flat week-one regime by +0.24 logit on the synthetic final.
 *  Returns per-member weights (softmax on mean absolute logit error) and a
 *  per-trophy residual scale for the ensemble median. Null when the history
 *  is too short for any origin. No bias correction is applied: the signed
 *  rolling-origin error measures early-growth behaviour that does not carry
 *  over to the flat week-one regime. */
function rollingOrigin(
  c: Case, snaps: Snapshot[], series: Series[], refIdx: number, releaseMs: number,
  reads: Array<{ ms: number; x: number; rates: Array<number | null> }>,
): {
  weights: number[];
  bias: number[];
  sigmaLoo: number[];
  pooledSigma: number;
} | null {
  const nT = c.trophies.length;
  const last = snaps.length - 1;
  const origins: number[] = [];
  for (let j = 3; j <= last - 1; j++) origins.push(j);
  if (origins.length === 0) return null;
  const xLast = Math.log(1 + Math.max(snaps[last].hours, -0.9));
  const cutoffLast = releaseMs + snaps[last].hours * 3_600_000;
  // One helper scores a member set against the last snapshot's rates.
  const evalPrefix = (
    j: number,
    useWeights: number[] | null,
    bias: number[],
    absErr: number[] | null,
    counts: number[] | null,
    resSq: number[] | null,
    resN: number[] | null,
    pool: { sq: number; n: number },
  ): void => {
    const prefix = snaps.slice(0, j);
    const preSeries = series.map((_, i) => buildSeries(prefix, i));
    const prePools = preSeries.map((_, i) => pooledSlope(preSeries, tierIdx(c, i), c.trophies));
    const refPre = memberDirect(preSeries[refIdx], priorSlope(c.trophies[refIdx]), xLast);
    const cutoffJ = releaseMs + snaps[j].hours * 3_600_000;
    for (let i = 0; i < nT; i++) {
      const y = snaps[last].rates[i];
      if (y === null || y === undefined || y >= 99.0 || y <= 0.05) continue;
      const truth = logit(y);
      const ms = memberSet(c, prefix, preSeries, i, refIdx, refPre, prePools[i], reads, cutoffJ, true, xLast);
      if (absErr !== null && counts !== null) {
        for (let m = 0; m < N_MEMBERS; m++) {
          absErr[m] += Math.abs(ms.medians[m] - truth);
          counts[m]++;
        }
      }
      if (useWeights !== null && resSq !== null && resN !== null) {
        let ens = 0;
        for (let m = 0; m < N_MEMBERS; m++) ens += useWeights[m] * (ms.medians[m] - bias[m]);
        const r = ens - truth;
        resSq[i] += r * r;
        resN[i]++;
        pool.sq += r * r;
        pool.n++;
      }
    }
  };
  // Pass 1: member errors against the last snapshot.
  const absErr = new Array<number>(N_MEMBERS).fill(0);
  const counts = new Array<number>(N_MEMBERS).fill(0);
  const pool1 = { sq: 0, n: 0 };
  for (const j of origins) evalPrefix(j, null, [0, 0, 0, 0, 0, 0], absErr, counts, null, null, pool1);
  const meanAbs = absErr.map((v, m) => (counts[m] > 0 ? v / counts[m] : 1));
  const bias = [0, 0, 0, 0, 0, 0];
  // Weights over the median members only (M1, M3, M4 = indices 0, 2, 3).
  // Softmax on mean absolute error; floor keeps each member alive.
  const medIdx = [0, 2, 3];
  const rawMed = medIdx.map((m) => Math.exp(-meanAbs[m] / 0.25));
  const flooredMed = rawMed.map((w) => Math.max(0.05, w));
  const wSumMed = flooredMed.reduce((a, b) => a + b, 0);
  const weights = [0, 0, 0, 0, 0, 0];
  medIdx.forEach((m, k) => { weights[m] = flooredMed[k] / wSumMed; });
  // Pass 2: ensemble residuals per trophy, with weights applied.
  const resSq = new Array<number>(nT).fill(0);
  const resN = new Array<number>(nT).fill(0);
  const pool2 = { sq: 0, n: 0 };
  for (const j of origins) evalPrefix(j, weights, bias, null, null, resSq, resN, pool2);
  const pooledSigma = pool2.n > 0 ? Math.sqrt(pool2.sq / pool2.n) : 0.35;
  const sigmaLoo = resSq.map((v, i) => {
    const own = resN[i] > 0 ? Math.sqrt(v / resN[i]) : pooledSigma;
    return Math.min(0.8, Math.max(0.12, 0.5 * own + 0.5 * pooledSigma));
  });
  return { weights, bias, sigmaLoo, pooledSigma };
}

/** Median logit and spread (down / up, in logit units) for one trophy. */
function forecastTrophy(
  c: Case,
  snaps: Snapshot[],
  series: Series[],
  idx: number,
  refIdx: number,
  xTarget: number,
  horizonHours: number,
  horizonDays: number,
  checkpointHours: number,
  refreshSd: number,
  ro: ReturnType<typeof rollingOrigin>,
  bPool: number,
  reads: Array<{ ms: number; x: number; rates: Array<number | null> }>,
): { medianLogit: number; sigmaDown: number; sigmaUp: number } {
  const t = c.trophies[idx];
  const s = series[idx];
  const prior = priorSlope(t);
  const lastRate = s.lastRate;

  // Fully censored trophy (every rate >= 99%, e.g. the first trophy on
  // launch night): no trend exists, the dilution prior is the forecast.
  if (s.xs.length === 0) {
    const dil = dilutionPrior(c, s, checkpointHours, horizonHours);
    const sd = Math.min(2.2, Math.max(0.35, Math.sqrt(dil.spread * dil.spread + refreshSd * refreshSd + 0.09)));
    return { medianLogit: Math.min(6.9, Math.max(-9.0, dil.logit)), sigmaDown: sd, sigmaUp: 0.7 * sd };
  }

  const refMedian = memberDirect(series[refIdx], priorSlope(c.trophies[refIdx]), xTarget);
  const ms = memberSet(c, snaps, series, idx, refIdx, refMedian, bPool, reads, parseTime(c.at), true, xTarget);

  // Ensemble median: rolling-origin weights over the median members only
  // (M1, M3, M4); equal weights among them when the history is too short
  // to score. The other members set the spread, not the median.
  const medIdx = [0, 2, 3];
  const weights = ro !== null ? ro.weights : [1 / 3, 0, 1 / 3, 1 / 3, 0, 0];
  const bias = ro !== null ? ro.bias : [0, 0, 0, 0, 0, 0];
  let ensemble = 0;
  for (const m of medIdx) ensemble += weights[m] * (ms.medians[m] - bias[m]);
  // Ensemble disagreement: the spread of ALL six members' medians around
  // the ensemble median (equal weights: it measures disagreement, not skill).
  let ensVar = 0;
  for (let m = 0; m < N_MEMBERS; m++) {
    const d = ms.medians[m] - bias[m] - ensemble;
    ensVar += d * d / N_MEMBERS;
  }
  const sigmaEns = Math.min(0.7, Math.sqrt(ensVar));

  // Launch dilution: strong while the history is flat, gone by five
  // distinct snapshots. Blends with the ensemble median.
  const wDil = clamp01((5 - s.nDistinct) / 4);
  let dilSpread = 0;
  let medianLogit = ensemble;
  if (wDil > 0) {
    const dil = dilutionPrior(c, s, checkpointHours, horizonHours);
    dilSpread = dil.spread;
    medianLogit = (1 - wDil) * ensemble + wDil * dil.logit;
  }
  // A trophy still at >= 99% leans on dilution even with some history.
  if (lastRate >= 99.0 && wDil < 0.7) {
    const dil = dilutionPrior(c, s, checkpointHours, horizonHours);
    dilSpread = Math.max(dilSpread, dil.spread);
    medianLogit = 0.3 * medianLogit + 0.7 * dil.logit;
  }
  medianLogit = Math.min(6.9, Math.max(-9.0, medianLogit));

  // Spread: rolling-origin residual + ensemble disagreement + random walk +
  // refresh jumps + slope uncertainty (+ dilution uncertainty while the
  // prior carries weight). Downside leg carries the full refresh term;
  // upside leg only part of it, fading as the checkpoint ages. The scale
  // is calibrated on backtests to ~0.9 coverage (V1's 0.8 scale with a
  // 0.95 refresh prior covered 1.00 -- far too wide).
  const SPREAD_SCALE = 0.45;
  const sigmaLoo = ro !== null ? ro.sigmaLoo[idx] : 0.35;
  const sigmaRw = 0.3 * Math.sqrt(horizonDays);
  const slopeTerm = 0.5 * Math.min(0.4, ms.slopeSe * Math.max(0, xTarget - ms.xBar));
  const upFrac = 0.6 + 0.4 * Math.min(1, checkpointHours / 72);
  const sigmaDown = SPREAD_SCALE * Math.min(
    2.2,
    Math.max(0.35, Math.sqrt(
      sigmaLoo * sigmaLoo + sigmaEns * sigmaEns + sigmaRw * sigmaRw +
      refreshSd * refreshSd + slopeTerm * slopeTerm +
      (wDil * dilSpread) * (wDil * dilSpread),
    )),
  );
  const sigmaUp = SPREAD_SCALE * Math.min(
    2.2,
    Math.max(0.3, Math.sqrt(
      sigmaLoo * sigmaLoo + sigmaEns * sigmaEns + sigmaRw * sigmaRw +
      (upFrac * refreshSd) * (upFrac * refreshSd) + slopeTerm * slopeTerm +
      (0.3 * wDil * dilSpread) * (0.3 * wDil * dilSpread),
    )),
  );
  return { medianLogit, sigmaDown, sigmaUp };
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);
  const releaseMs = c.game?.release_date
    ? Date.parse(`${c.game.release_date}T00:00:00Z`)
    : parseTime(c.history[0].at);
  const snaps = distinctSnapshots(c, releaseMs);
  const checkpointHours = (parseTime(c.at) - releaseMs) / 3_600_000;
  const targetHours = (parseTime(c.target) - releaseMs) / 3_600_000;
  const xTarget = Math.log(1 + Math.max(targetHours, -0.9));
  const horizonHours = c.horizon_minutes / 60;
  const horizonDays = c.horizon_minutes / 1440;

  const series = c.trophies.map((_, i) => buildSeries(snaps, i));
  const pools = series.map((_, i) => pooledSlope(series, tierIdx(c, i), c.trophies));
  const refreshSd = refreshScale(snaps, checkpointHours);
  // Reference trophy for the log-ratio member: the trophy whose last rate
  // sits closest to the median last rate, with a full history -- a mid
  // story/activity trophy with a smooth, monotone climb. The highest-rate
  // trophy is a bad reference: on the synthetic final it was the 100%-then-
  // diluted first trophy, whose non-monotone path made the ratio member the
  // ensemble's worst extrapolator while the origin weights ranked it first.
  const usableRates = c.trophies
    .map((_, i) => series[i].lastRate)
    .filter((r) => r < 99 && r > 0.05)
    .sort((a, b) => a - b);
  const medRate = usableRates.length > 0 ? usableRates[Math.floor(usableRates.length / 2)] : 10;
  let refIdx = 0;
  let refDist = Infinity;
  c.trophies.forEach((t, i) => {
    const r = series[i].lastRate;
    if (t.tier === 'platinum' || r >= 99 || r <= 0.05 || series[i].xs.length < 5) return;
    const d = Math.abs(Math.log(r / medRate));
    if (d < refDist) { refIdx = i; refDist = d; }
  });
  if (refDist === Infinity) {
    c.trophies.forEach((t, i) => {
      const r = series[i].lastRate;
      if (t.tier === 'platinum' || r >= 99 || r <= 0.05) return;
      const d = Math.abs(Math.log(r / medRate));
      if (d < refDist) { refIdx = i; refDist = d; }
    });
  }
  const reads = c.history.map((h) => {
    const hours = (parseTime(h.at) - releaseMs) / 3_600_000;
    return { ms: parseTime(h.at), x: Math.log(1 + Math.max(hours, -0.9)), rates: h.rates };
  });
  const ro = rollingOrigin(c, snaps, series, refIdx, releaseMs, reads);

  // Per-trophy medians and spreads.
  const medians: number[] = [];
  const downs: number[] = [];
  const ups: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const f = forecastTrophy(c, snaps, series, i, refIdx, xTarget, horizonHours, horizonDays, checkpointHours, refreshSd, ro, pools[i], reads);
    medians.push(f.medianLogit);
    downs.push(f.sigmaDown);
    ups.push(f.sigmaUp);
  }

  // Ordering: story chapters must not rise; the platinum needs every other
  // trophy, so it must not exceed any of them. Shift the whole band.
  const chapterIdx: number[] = c.trophies
    .map((t, i) => ({ t, i }))
    .filter(({ t }) => /Complete Chapter (\d)/.test(t.description))
    .sort((a, b) => {
      const na = Number(/Complete Chapter (\d)/.exec(a.t.description)?.[1] ?? 99);
      const nb = Number(/Complete Chapter (\d)/.exec(b.t.description)?.[1] ?? 99);
      return na - nb;
    })
    .map(({ i }) => i);
  const finalIdx = c.trophies.findIndex((t) => /Complete the final mission/.test(t.description));
  if (finalIdx >= 0) chapterIdx.push(finalIdx);
  let runMin = Infinity;
  for (const i of chapterIdx) {
    if (medians[i] > runMin) medians[i] = runMin;
    runMin = Math.min(runMin, medians[i]);
  }
  const platIdx = c.trophies.findIndex((t) => t.tier === 'platinum');
  if (platIdx >= 0) {
    let otherMin = Infinity;
    for (let i = 0; i < c.trophies.length; i++) {
      if (i !== platIdx) otherMin = Math.min(otherMin, medians[i]);
    }
    if (medians[platIdx] > otherMin) medians[platIdx] = otherMin;
  }

  const trophies = c.trophies.map((t, i) => {
    const qs = Z.map((z) => {
      const l = medians[i] + z * (z < 0 ? downs[i] : ups[i]);
      return round4(clampRate(invLogitPct(l)));
    });
    // Keep the row non-decreasing after rounding and clamping.
    for (let j = 1; j < qs.length; j++) qs[j] = Math.max(qs[j], qs[j - 1]);
    return { id: t.id, q: qs };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(forecastCase(c))}\n`);
}
