const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, LevelFormat,
} = require("docx");

const m = JSON.parse(fs.readFileSync("outputs/metrics.json", "utf8"));
const bt = m.backtest;
const folds = Object.keys(bt);

const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, ...opts })] });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun(t)] });
const bullet = (parts) =>
  new Paragraph({
    numbering: { reference: "bul", level: 0 },
    spacing: { after: 60 },
    children: parts.map((x) => (typeof x === "string" ? new TextRun(x) : new TextRun({ text: x.b, bold: true }))),
  });
const img = (file, w, h) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 160 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h } })],
  });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFCBCE" };
const borders = { top: border, bottom: border, left: border, right: border };
const cell = (text, width, o = {}) =>
  new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: o.head ? { fill: "064A56", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({
      alignment: o.right ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!o.head, color: o.head ? "FFFFFF" : undefined, size: 20 })],
    })],
  });

const widths = [3000, 2090, 2090, 2180];
const rows = [
  new TableRow({ tableHeader: true, children: [
    cell("Test window (trained on all earlier data)", widths[0], { head: true }),
    cell("Baseline $/mile", widths[1], { head: true, right: true }),
    cell("LightGBM", widths[2], { head: true, right: true }),
    cell("LightGBM + offset", widths[3], { head: true, right: true }),
  ] }),
  ...folds.map((f) => new TableRow({ children: [
    cell(f.replace("..", " to "), widths[0]),
    cell(bt[f].baseline_rate_per_mile["MAPE_%"].toFixed(2) + "%", widths[1], { right: true }),
    cell(bt[f].lightgbm["MAPE_%"].toFixed(2) + "%", widths[2], { right: true }),
    cell(bt[f].lightgbm_plus_recent_offset["MAPE_%"].toFixed(2) + "%", widths[3], { right: true }),
  ] })),
];
const avg = (k) => (folds.reduce((s, f) => s + bt[f][k]["MAPE_%"], 0) / folds.length).toFixed(2) + "%";
rows.push(new TableRow({ children: [
  cell("Average of the 3 folds", widths[0], {}),
  cell(avg("baseline_rate_per_mile"), widths[1], { right: true }),
  cell(avg("lightgbm"), widths[2], { right: true }),
  cell(avg("lightgbm_plus_recent_offset"), widths[3], { right: true }),
] }));

const dq = m.data_quality;
const children = [
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Freight Rate Prediction: Validation and Split Approach", bold: true, size: 36, color: "064A56" })] }),
  p("Machine Learning Engineer assessment, Spotter. Metric reported: MAPE on posted_rate (the final scoring metric is calculated by Spotter).", { color: "455A60" }),

  h1("1. Data and the key constraint"),
  p("train_test.csv holds 48,000 labelled loads dated 2025-01-01 to 2025-10-31. validation.csv holds 12,000 unlabelled loads dated 2025-11-01 to 2025-12-31. The validation set is therefore entirely in the future relative to the training data. A random train/test split would let the model see loads from the same days it is tested on and would give an optimistic score that will not hold on the real validation set."),

  h1("2. Split and validation approach"),
  bullet([{ b: "Rolling-origin (forward-chaining) backtest. " }, "For each fold the model is trained only on data before the fold and tested on the following two months: test May-Jun (train Jan-Apr), Jul-Aug (train Jan-Jun) and Sep-Oct (train Jan-Aug). This mimics the real task of predicting the next two months."]),
  bullet([{ b: "No random shuffling and no future information. " }, "Features that depend on the day (daily median market_index, daily mean quote_signal) are computed from that day's own rows only, so they use no information from other periods."]),
  bullet([{ b: "Final model. " }, "After choosing the approach on the backtest, the model is retrained on all 48,000 labelled rows and applied to validation.csv."]),
  bullet([{ b: "Out-of-fold residuals by month. " }, "To estimate recent drift I hold out one calendar month at a time inside the training window and measure residuals, so the correction is not computed on rows the model was fitted on."]),
  bullet([{ b: "Metric. " }, "MAPE (also MAE, median APE and RMSE are saved in outputs/metrics.json). MAPE is used because rates range from about $60 to $25,000, so errors are best judged relative to the load size."]),

  h1("3. Data quality findings and handling"),
  bullet([{ b: "Weight: " }, `${dq.negative_weight_rows} negative values with a magnitude matching the normal range, so they are sign errors and are converted with abs(). ${dq.missing_weight_rows} missing values are filled with the median and flagged.`]),
  bullet([{ b: "market_index: " }, `${dq.missing_market_index_rows} missing values. The index behaves like a daily market level (within-day std 0.025 vs between-day std 0.17), so a missing value is filled with the median of that day.`]),
  bullet([{ b: "Corrupted labels: " }, `about ${dq["extreme_label_rows_oof_gt_0.3_log"]} rows (1.4%) have a posted_rate that is roughly 0.2-0.5x or 2.3-5x what their distance and equipment imply, for example 11,377 dollars for a Houston to New York load. The residuals are bimodal (a tight core and a separate cluster), which points to data errors rather than real variation. I did not delete them because the same noise will exist in the validation labels; instead the model uses an L1 objective, which is largely unaffected by them.`]),
  bullet([{ b: "Unseen cities: " }, `${dq.validation_rows_with_unseen_pickup_city} validation rows have a pickup city (and ${dq.validation_rows_with_unseen_delivery_city} a delivery city) that never appears in training (Charlotte, Chicago, Allentown, Jackson, Knoxville, Laredo, Norfolk, San Diego). The model therefore does not use city names or city encodings and relies on distance, coordinates, equipment, weight and market features, which generalise to new cities.`]),
  bullet([{ b: "Coordinates and distance: " }, "coordinates look clipped (several cities share a latitude boundary, for example Los Angeles and Phoenix), and 214 rows have a road distance more than 1.5x the straight-line distance. The provided distance is used as the main signal, and the straight-line distance and the ratio are added as extra features."]),

  img("outputs/figures/eda_target.png", 560, 202),
  img("outputs/figures/market_index_timeline.png", 560, 179),

  h1("4. Model"),
  p("LightGBM gradient boosting on log(posted_rate) with an L1 objective. Boosted trees handle the strong non-linear distance effect (short loads cost far more per mile) and the interactions with equipment type without manual tuning. Predicting the log makes errors relative, matching MAPE. Day-of-week was tested and removed because it reduced backtest accuracy. A linear baseline and a $/mile-by-equipment baseline were clearly worse."),
  p("Most important features by gain: log distance (45%), straight-line distance (14%), quote_signal (9%), distance (9%) and equipment (6%)."),

  h1("5. Results"),
  new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: widths, rows }),
  new Paragraph({ spacing: { after: 100 }, children: [] }),
  img("outputs/figures/backtest_mape.png", 480, 216),
  p("Offset: rates in the later months of training sit about 2% above what a model trained on the earlier months predicts, and I add the median log-residual of the last 56 days (+2.1%) to the final predictions. It helped in two of three folds but hurt in the May-Jun fold (3.95% to 4.29%), so it is a modest, not guaranteed, improvement; the average across folds is slightly better with it."),

  h1("6. Fixed December prediction chart"),
  p("december_chart_inputs.csv contains only the route, distance, equipment, weight and date, with no market columns. For each December date I use that date's median market_index and mean quote_signal taken from validation.csv; these are model inputs available at prediction time, not labels. The curve therefore follows the weekly rhythm of the market index (roughly a 7-day cycle) and stays within about 797 to 834 dollars, which is plausible for a 360-mile dry van load."),
  p("Limitation: there is no December data in training, so any holiday effect on rates cannot be learned and is not reflected in the chart."),
  img("scorer_results/candidate_december.png", 560, 252),

  h1("7. Limitations and next steps"),
  bullet(["The corrupted-label rate (about 1.4%) puts a floor on achievable MAPE and on RMSE; identifying whether those rows are errors or special cases would help."]),
  bullet(["With more time: tune hyperparameters with the same forward-chaining folds, add per-lane statistics with a fallback for unseen cities, and test quantile models for prediction intervals."]),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [{
      id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 28, bold: true, font: "Calibri", color: "064A56" },
      paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 },
    }],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1200, right: 1440, bottom: 1200, left: 1440 } } }, children }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report.docx", buf); console.log("report.docx written"); });
