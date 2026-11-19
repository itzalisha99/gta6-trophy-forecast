// GTA VI trophy rarity forecast (Node 22, standard library only).
//
// Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
// earn rate on Exophase at the case's target, as five quantiles (0.05 / 0.25 /
// 0.5 / 0.75 / 0.95) in percent. Reads every <id>.in in the input directory
// and writes one <id>.out per case. Deterministic: same input, same bytes.
//
// The model, per trophy, in three parts:
//
//  1. Growth. Earn rates follow players' progress through the game, so on
//     the logit scale a rate moves roughly linearly against log time since
//     release, x = ln(1 + hours). The median is a direct weighted fit of
//     logit(rate) against x over the case's distinct page states, with
//     gentle recency weights and slopes shrunk toward a tier prior (early
//     bronze story trophies climb fastest, the platinum slowest), which
//     steadies trophies whose history is short or flat. An earners-over-
//     players fit (log earners and log players tracked each against x,
//     divided at the target) is kept only as a fallback for trophies whose
//     direct fit is degenerate: two extrapolated lines can diverge, so it
//     never carries a median the direct fit can give.
//
//  2. Launch dilution. At launch Exophase's tracked players grow much faster
//     than earners: a refresh after a long stale stretch cuts most rates.
//     When a case's history has barely changed (few distinct snapshots) the
//     trend fit sees a flat line and says nothing, so a dilution prior takes
//     over: players are assumed to keep growing at the launch rate over the
//     stale stretch plus the horizon, earners keep up with only a share of
//     that growth (a share that rises with the trophy's current rate -- the
//     first mission is earned almost as fast as players join, rare trophies
//     much slower). The prior fades out as real history accumulates, and is
//     gone once a case has five distinct snapshots.
//
//  3. Spread. Pinball loss on the logit scale punishes a band that is too
//     narrow when a refresh moves rates and a band that is too wide
//     everywhere, so the spread is built from honest parts: a random-walk
//     term growing with the horizon, the trend fit's residual, the slope's
//     standard error times the extrapolation distance, a refresh-jump term
//     (measured from the history's own change points, or a launch-decaying
//     prior when there are none), and the dilution model's own uncertainty.
//     The band is asymmetric: refreshes at launch mostly push rates down,
//     so the downside leg carries the full refresh term and the upside leg
//     only part of it. The two legs are then scaled by 0.8: backtests on
//     synthetic finals and on the practice cases showed the raw spread about
//     a quarter too wide, with 90% coverage holding at the scaled width.
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

/** Median logit and spread (down / up, in logit units) for one trophy. */
function forecastTrophy(
  c: Case,
  snaps: Snapshot[],
  idx: number,
  xTarget: number,
  horizonHours: number,
  horizonDays: number,
  checkpointHours: number,
  logN: WeightedFit | null,
  refreshSd: number,
): { medianLogit: number; sigmaDown: number; sigmaUp: number } {
  const t = c.trophies[idx];
  const last = snaps[snaps.length - 1];
  const lastRate = last.rates[idx] ?? 1.0;
  const prior = priorSlope(t);

  // Per-trophy series over the distinct snapshots. The whole history is
  // fitted -- launch week is one smooth adoption curve, and the early
  // reads steady the slope -- with gentle recency weights so the recent
  // regime counts most. One guard: rates at or above 99% are censored out
  // of the fit, because a 100% rate in a tiny early cohort is not a
  // measurement of the rate -- its clamped logit of 9.21 would poison the
  // line. The count below still counts every snapshot, so the dilution
  // prior keys on the full history.
  const nDistinct = snaps.filter((s) => s.rates[idx] !== null && s.rates[idx] !== undefined).length;
  const series = snaps;
  const xs: number[] = [];
  const ys: number[] = [];
  const es: number[] = []; // log earners
  const ws: number[] = [];
  const xLast = Math.log(1 + Math.max(snaps[snaps.length - 1].hours, -0.9));
  for (const s of series) {
    const r = s.rates[idx];
    if (r === null || r === undefined || r >= 99.0) continue;
    const x = Math.log(1 + Math.max(s.hours, -0.9));
    xs.push(x);
    ys.push(logit(r));
    es.push(Math.log(Math.max((r / 100) * s.players, 0.5)));
    ws.push(Math.exp(-0.2 * (xLast - x)));
  }

  // Direct trend fit, slope shrunk toward the tier prior.
  const fit = weightedFit(xs, ys, ws);
  let trendLogit: number | null = null;
  let slopeSe = 0.35;
  let residSd = 0.35;
  if (fit !== null) {
    const k = 3.5;
    const nEff = Math.min(fit.n, 12);
    const b = (nEff * fit.b + k * prior) / (nEff + k);
    const a = (ws.reduce((s, w, i) => s + w * ys[i], 0) / ws.reduce((s, w) => s + w, 0)) - b * fit.xBar;
    trendLogit = a + b * xTarget;
    slopeSe = fit.slopeSe;
    residSd = fit.residSd;
  }

  // Earners-over-players fit: log earners against x, slope shrunk toward
  // (players slope + tier prior); divided by the players fit at the target.
  // Two extrapolated lines can diverge wildly -- a small slope gap becomes
  // an exponential rate gap -- so this view is only ever a bounded
  // correction to the direct trend fit, never a free extrapolation.
  let enLogit: number | null = null;
  const fitE = weightedFit(xs, es, ws);
  if (fitE !== null && logN !== null) {
    const k = 3.5;
    const nEff = Math.min(fitE.n, 12);
    const bE = (nEff * fitE.b + k * (logN.b + prior)) / (nEff + k);
    const eBar = ws.reduce((s, w, i) => s + w * es[i], 0) / ws.reduce((s, w) => s + w, 0);
    const aE = eBar - bE * fitE.xBar;
    const logEtarget = aE + bE * xTarget;
    const logNtarget = logN.a + logN.b * xTarget;
    const rateEn = 100 * Math.exp(logEtarget - logNtarget);
    enLogit = logit(clampRate(rateEn));
    if (trendLogit !== null) {
      enLogit = Math.min(trendLogit + 1.0, Math.max(trendLogit - 1.0, enLogit));
    } else {
      // No trend fit to anchor on: cap growth above the last rate.
      enLogit = Math.min(logit(lastRate) + 2.0, Math.max(logit(lastRate) - 2.0, enLogit));
    }
  }

  // The direct trend fit carries the median whenever it exists. The
  // earners-over-players view is only a fallback for trophies whose direct
  // fit is degenerate: its two extrapolated lines can diverge (a small
  // slope gap becomes an exponential rate gap), and on backtests it pulled
  // fast-growing trophies up by about half a logit.
  let medianLogit: number;
  if (trendLogit !== null) {
    medianLogit = trendLogit;
  } else if (enLogit !== null) {
    medianLogit = enLogit;
  } else {
    medianLogit = logit(lastRate) + prior * Math.max(0, xTarget - xLast);
  }

  // Launch dilution prior: strong while the history is flat, gone by five
  // distinct snapshots. Players grow at the launch rate over the stale
  // stretch plus the horizon; earners keep up with a share of that growth.
  const wDil = clamp01((5 - nDistinct) / 4);
  let dilLogit = medianLogit;
  let dilSpread = 0;
  let alpha = 0.3;
  if (wDil > 0) {
    const firstMs = parseTime(c.history[0].at);
    const staleHours = Math.max(0, (parseTime(c.at) - firstMs) / 3_600_000);
    const g = 0.38 * Math.exp(-checkpointHours / 36);
    const growLog = Math.min(Math.log(8), g * (staleHours + horizonHours));
    alpha = Math.min(0.85, Math.max(0.18, 0.18 + 0.55 * Math.pow(lastRate / 100, 0.3)));
    const rateDil = clampRate(lastRate * Math.exp(-(1 - alpha) * growLog));
    dilLogit = logit(rateDil);
    dilSpread = 0.5 * (1 - alpha) * growLog;
    medianLogit = (1 - wDil) * medianLogit + wDil * dilLogit;
  }
  medianLogit = Math.min(6.9, Math.max(-9.0, medianLogit));

  // Spread: random walk + fit residual + slope uncertainty + refresh jumps
  // (+ dilution uncertainty while the prior carries weight). Downside leg
  // carries the full refresh term; upside leg only part of it.
  const sigmaRw = 0.5 * Math.sqrt(horizonDays);
  const fitSd = Math.min(1.0, Math.max(0.15, residSd));
  const slopeTerm = 0.5 * Math.min(0.4, slopeSe * Math.max(0, xTarget - (fit?.xBar ?? xLast)));
  // Backtest calibration (see README): the raw spread runs about a quarter
  // too wide, so both legs are scaled by 0.8.
  const SPREAD_SCALE = 0.8;
  const sigmaDown = SPREAD_SCALE * Math.min(
    2.2,
    Math.max(0.45, Math.sqrt(
      sigmaRw * sigmaRw + fitSd * fitSd + slopeTerm * slopeTerm +
      refreshSd * refreshSd + (wDil * dilSpread) * (wDil * dilSpread),
    )),
  );
  const sigmaUp = SPREAD_SCALE * Math.min(
    2.2,
    Math.max(0.35, Math.sqrt(
      sigmaRw * sigmaRw + fitSd * fitSd + slopeTerm * slopeTerm +
      (0.6 * refreshSd) * (0.6 * refreshSd) + (0.3 * wDil * dilSpread) * (0.3 * wDil * dilSpread),
    )),
  );
  return { medianLogit, sigmaDown, sigmaUp };
}

/** Refresh-jump scale from the history's own change points: the scatter of
 *  the median per-trophy logit move at each change. Falls back to a
 *  launch-decaying prior when the history has too few changes. */
function refreshScale(snaps: Snapshot[], checkpointHours: number): number {
  if (snaps.length < 3) return 0.95 * Math.exp(-checkpointHours / 48);
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
  if (moves.length < 2) return 0.95 * Math.exp(-checkpointHours / 48);
  const sorted = [...moves].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  const absDev = moves.map((x) => Math.abs(x - med)).sort((a, b) => a - b);
  const mad = absDev[Math.floor(absDev.length / 2)];
  return Math.min(0.5, Math.max(0.2, 1.4826 * mad));
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

  // Shared players fit: log(players tracked) against log time.
  const nx: number[] = [];
  const ny: number[] = [];
  const nw: number[] = [];
  const xLast = Math.log(1 + Math.max(snaps[snaps.length - 1].hours, -0.9));
  for (const s of snaps) {
    const x = Math.log(1 + Math.max(s.hours, -0.9));
    nx.push(x);
    ny.push(Math.log(Math.max(s.players, 1)));
    nw.push(Math.exp(-0.2 * (xLast - x)));
  }
  const logN = weightedFit(nx, ny, nw);
  const refreshSd = refreshScale(snaps, checkpointHours);

  // Per-trophy medians and spreads.
  const medians: number[] = [];
  const downs: number[] = [];
  const ups: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const f = forecastTrophy(c, snaps, i, xTarget, horizonHours, horizonDays, checkpointHours, logN, refreshSd);
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
