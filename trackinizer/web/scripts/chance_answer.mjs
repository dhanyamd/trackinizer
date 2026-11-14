// Composes the answer to "what is the chance rate" from the arms' own series,
// read out of the published answer that raised the question.
import { readFileSync, writeFileSync } from "node:fs";

// The three arms' own series, lifted out of the published grokking answer before this
// one replaced it: `blocks-latest.json` is a single slot, and /tmp/grok_*.json and
// Experiment#1658 are both gone, so this frozen copy is the last one standing. The decay
// arm is kept as the row already derived from it (it is off this chart's scale anyway).
const DATA = JSON.parse(readFileSync("scripts/chance_data.json", "utf8"));

const N = 4705;
const P = 1 / 97;
const SD = Math.sqrt((P * (1 - P)) / N);
const LO = P - 1.96 * SD;
const HI = P + 1.96 * SD;
const pct = (x) => Number((x * 100).toFixed(4));
const pLO = pct(LO);
const pHI = pct(HI);
const pP = pct(P);

const stat = (rows) => {
  const v = rows.map(([, a]) => a);
  const best = Math.max(...v);
  return {
    final: v[v.length - 1],
    best,
    bestStep: rows[v.indexOf(best)][0],
    above: v.filter((x) => x > pP).length,
    inBand: v.filter((x) => x >= pLO && x <= pHI).length,
    n: v.length,
  };
};
const decayRow = DATA.table_rows.find((r) => r[0] === "decay");
const S = {
  decay: {
    final: decayRow[1],
    best: decayRow[4],
    bestStep: decayRow[5],
    above: Number(decayRow[6].split(" ")[0]),
    inBand: Number(decayRow[7].split(" ")[0]),
    n: 121,
  },
  no_decay: stat(DATA.points.no_decay),
  clamp: stat(DATA.points.clamp),
};

const points = [];
for (const name of ["clamp", "no_decay"]) {
  for (const [step, acc] of DATA.points[name]) points.push({ step, acc, arm: name });
}

// The theme draws every rule in the hairline ink it draws gridlines in, so a gridline
// at 1.0 beside the chance rule at 1.0309 is two identical lines. Tick where the data
// is not, weight the chance rule, and label all three inside the frame.
const yEnc = (field) => ({
  field,
  type: "quantitative",
  scale: { domain: [0, 1.6] },
  axis: { title: "test accuracy (%)", values: [0, 0.4, 0.8, 1.2, 1.6] },
});

const answer = {
  title: "Chance rate · 1/97 on 4,705 pairs",
  prompt: "what is the chance rate",
  budget: { prose: 1, section: 3, cells: 1, figure: 3, marks: 1, footnote: 1 },
  compose: [
    {
      node: "prose",
      lead: true,
      span: true,
      text: "Chance is 1/97 = 1.0309%, because the task is 97-way: predict the residue of a sum mod 97. On the 4,705 held-out pairs that is 48.5 correct, and one eval carries 0.1473 pp of sampling noise, so a single eval anywhere in 0.742% to 1.320% is indistinguishable from guessing. The clamped arm is not flat at that line: it climbs to it from 0% at memorization, first crosses it at step 1,400, and spends 106 of its 121 evals inside the band. The other 15 matter to the sentence you asked about, because 7 of them sit above the band and all 7 are after step 10,400.",
    },
    { node: "section", label: "where 1.0309 per cent comes from", size: "97 classes, 4,705 held-out pairs" },
    {
      node: "cells",
      cells: [
        "chance",
        "1.0309%",
        "= 48.5 of 4,705",
        "one-eval sd",
        "0.1473 pp",
        "95% band",
        "0.742% to 1.320%",
      ],
    },
    {
      node: "figure",
      figure: {
        kind: "equation",
        display: true,
        tex: "\\text{chance}=\\tfrac{1}{p}=\\tfrac{1}{97}=1.0309\\%,\\qquad \\sigma_{\\text{eval}}=\\sqrt{\\tfrac{\\hat p\\,(1-\\hat p)}{n}}=\\sqrt{\\tfrac{0.010309\\times0.989691}{4705}}=0.1473\\ \\text{pp}",
      },
    },
    { node: "section", label: "the two non-grokking arms against the line", size: "121 evals each, 100-step intervals" },
    {
      node: "figure",
      figure: {
        kind: "chart",
        label:
          "Every eval of the two arms that never generalized, against the chance line. The solid rule is 1.0309%; the dashed pair is the 95% sampling band for one eval on 4,705 pairs. The clamped arm rises out of memorization to the line by step 1,400 and stays in the band for 106 of 121 evals, with 7 above it, all after step 10,400. no_decay sits below the line for the whole run. The decay arm is not drawn: it reaches 100% and would flatten this scale to a baseline.",
        spec: {
          height: 240,
          layer: [
            {
              data: { values: [{ y: pct(LO) }, { y: pct(HI) }] },
              mark: { type: "rule", strokeDash: [4, 3] },
              encoding: { y: yEnc("y") },
            },
            {
              data: { values: [{ y: pct(P) }] },
              mark: { type: "rule", strokeWidth: 2 },
              encoding: { y: yEnc("y") },
            },
            {
              data: {
                values: [
                  { x: 4300, y: pct(HI), t: "+1.96 sd  1.3196%", dy: -5 },
                  { x: 4300, y: pct(P), t: "chance  1.0309%", dy: 11 },
                  { x: 4300, y: pct(LO), t: "-1.96 sd  0.7423%", dy: 11 },
                ],
              },
              mark: { type: "text", align: "left", fontSize: 10.5 },
              encoding: {
                x: { field: "x", type: "quantitative" },
                y: yEnc("y"),
                text: { field: "t", type: "nominal" },
                yOffset: { field: "dy", type: "quantitative", scale: { type: "identity" } },
              },
            },
            {
              data: { values: points },
              mark: { type: "point", filled: true, size: 16 },
              encoding: {
                x: { field: "step", type: "quantitative", axis: { title: "step", tickCount: 7, format: "d" } },
                y: yEnc("acc"),
                color: { field: "arm", type: "nominal", legend: { title: "arm" } },
                tooltip: [
                  { field: "arm", type: "nominal" },
                  { field: "step", type: "quantitative" },
                  { field: "acc", type: "quantitative", title: "test accuracy (%)" },
                ],
              },
            },
          ],
        },
      },
    },
    {
      node: "marks",
      span: true,
      items: [
        {
          kind: "demand",
          text: "Say the band, not `never left chance`. The checkable sentence is that the clamped arm's test accuracy stayed inside 0.742% to 1.320% for 106 of its 121 evals and finished 98.70 pp below the decay arm, which is what rules out generalization. `Never left` also claims the 15 evals outside the band did not happen.",
        },
        {
          kind: "finding",
          text: "The clamped arm did move, upward, and only as far as chance. It is at 0.32% at step 600 when the norm is pinned, 0.85% at 1,000, first above 1.0309% at step 1,400, and averages 1.137% across the 115 evals from step 600 on: 53.5 of 4,705 against 48.5 expected.",
        },
        {
          kind: "flag",
          text: "Seven evals sit above the 95% band and all seven are late: 1.32% at 10,400 and 10,500, 1.38% at 11,200, 1.34% at 11,300, 1.32% at 11,500, 1.36% at 11,700, 1.32% at 11,800. The best of them is 64.9 of 4,705 against 48.5 expected, 2.37 sd out. That is 16 extra pairs, not a circuit, but it is a drift and not a flat line.",
        },
        {
          kind: "note",
          text: "The 121 evals share one test set and successive states of one model, so they are not 121 independent draws. The band is the right test for a single eval. An aggregate computed across the run, 7.70 sd on the post-clamp mean, treats correlated evals as independent and overstates the excess by an unknown factor.",
        },
        {
          kind: "finding",
          text: "no_decay is the arm that is genuinely off the line, and it is below it: final test 0.30%, 14.1 of 4,705, 4.96 sd under chance. Only 1 of its 121 evals lands inside the band, the untrained eval at step 1. A memorizing model is anti-predictive on held-out pairs, so below chance is the memorizing baseline here, not chance itself.",
        },
      ],
    },
    { node: "section", label: "each arm in counts, not percentages", size: "n = 4,705, chance = 48.5" },
    {
      node: "figure",
      figure: {
        kind: "table",
        table: {
          caption:
            "Each arm against the chance line, in counts as well as percentages. n = 4,705 held-out pairs, chance 1.0309% (48.5 correct), one-eval sd 0.1473 pp, 95% band 0.742% to 1.320%. z is (final test minus chance) divided by that sd, so it is the test for one eval and not for the run. Counts are derived from the accuracy the runner filed at 4 decimal places, so each lands within half a pair of the integer.",
          columns: [
            "final test",
            "correct of 4,705",
            "z vs chance",
            "best eval",
            "best eval step",
            "evals above chance",
            "evals inside band",
          ],
          rows: ["decay", "no_decay", "clamp"].map((k) => [
            k,
            S[k].final,
            Number(((S[k].final / 100) * N).toFixed(1)),
            Number(((S[k].final / 100 - P) / SD).toFixed(2)),
            S[k].best,
            S[k].bestStep,
            `${S[k].above} of ${S[k].n}`,
            `${S[k].inBand} of ${S[k].n}`,
          ]),
        },
      },
    },
    {
      node: "footnote",
      text:
        "Chance and the band are computed here, not filed: p = 97 and 4,705 held-out pairs come from the run config quoted on the previous surface, the sd is the binomial one printed above · the three series are the 121-eval curves embedded in that surface, which is now the only copy: /tmp/grok_clamp.json, /tmp/grok_no_decay.json, /tmp/grok_decay.json and /tmp/run_grok.py are gone from this box, and Experiment#1658 and Belief#651 both 404 on this server, whose graph has been reset since they were filed · so every figure here is traceable to that published answer and no further back, and the two words `weight-decay-driven` and `blocks circuit formation` remain untested for the reasons that surface gave.",
    },
  ],
};

writeFileSync("dist/blocks-latest.json", JSON.stringify(answer, null, 1));
writeFileSync("public/blocks-latest.json", JSON.stringify(answer, null, 1));
console.log("stats", JSON.stringify(S, null, 1));
console.log("band", pct(LO), pct(HI), "chance", pct(P), "sd pp", Number((SD * 100).toFixed(4)));
