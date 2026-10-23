(function () {
  "use strict";

  var statusEl = document.getElementById("dashboard-status");
  var errorEl = document.getElementById("dashboard-error");
  var scEl = document.getElementById("sc-content");
  var cmEl = document.getElementById("cm-content");
  var calEl = document.getElementById("cal-content");

  function fmtPct(x, digits) {
    digits = digits === undefined ? 1 : digits;
    return (Number(x) * 100).toFixed(digits) + "%";
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    attrs = attrs || {};
    Object.keys(attrs).forEach(function (k) {
      if (k === "text") node.textContent = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { node.appendChild(c); });
    return node;
  }

  function sourceLink(label, path, url) {
    var p = el("p", { class: "source-note" });
    p.appendChild(document.createTextNode(label + ": "));
    var absoluteUrl = /^https?:\/\//.test(url) ? url : window.PrpdApi.API_BASE_URL + url;
    var a = el("a", { href: absoluteUrl, text: path, target: "_blank", rel: "noopener" });
    p.appendChild(a);
    return p;
  }

  // ---------------------------------------------------------------- sensor comparison

  function renderSensorComparison(data) {
    scEl.innerHTML = "";

    var ensembleRows = data.system_comparison.filter(function (r) { return r.system === "Ensemble"; });
    var chart = el("div", { class: "bar-chart" });
    var sensorLabels = { sensor_1: "UHF Antenna", sensor_2: "Coupling Capacitor", sensor_3: "HFCT" };
    ensembleRows.forEach(function (r) {
      var f1 = parseFloat(r.macro_f1);
      var col = el("div", { class: "bar-col" });
      col.appendChild(el("div", { class: "bar-value", text: f1.toFixed(3) }));
      col.appendChild(el("div", { class: "bar-rect", style: "height:" + Math.max(4, f1 * 160) + "px" }));
      col.appendChild(el("div", { class: "bar-label", text: sensorLabels[r.sensor] || r.sensor }));
      chart.appendChild(col);
    });
    scEl.appendChild(el("p", { style: "font-size:0.85rem;color:var(--color-text-muted);margin:0 0 4px", text: "Ensemble macro-F1 by sensor (ten shared classes)" }));
    scEl.appendChild(chart);

    var table = el("table", { class: "data-table" });
    var thead = el("thead", {}, [el("tr", {}, [
      el("th", { text: "Fault" }),
      el("th", { text: "S1 n" }), el("th", { text: "S1 F1" }),
      el("th", { text: "S2 n" }), el("th", { text: "S2 F1" }),
      el("th", { text: "S3 n" }), el("th", { text: "S3 F1" })
    ])]);
    var tbody = el("tbody");
    var footRow = null;
    data.per_class.forEach(function (r) {
      var row = el("tr", {}, [
        el("td", { text: r.fault }),
        el("td", { text: r.sensor_1_test_images }), el("td", { text: parseFloat(r.sensor_1_f1).toFixed(3) }),
        el("td", { text: r.sensor_2_test_images }), el("td", { text: parseFloat(r.sensor_2_f1).toFixed(3) }),
        el("td", { text: r.sensor_3_test_images }), el("td", { text: parseFloat(r.sensor_3_f1).toFixed(3) })
      ]);
      if (r.fault === "OVERALL") {
        footRow = row;
      } else {
        tbody.appendChild(row);
      }
    });
    table.appendChild(thead);
    table.appendChild(tbody);
    if (footRow) table.appendChild(el("tfoot", {}, [footRow]));
    scEl.appendChild(el("div", { class: "data-table-wrap" }, [table]));
    scEl.appendChild(sourceLink("Source", data.source, data.source_url));
  }

  // ---------------------------------------------------------------- confusion matrix

  function heatColor(pct) {
    var a = 0.08 + (pct / 100) * 0.82;
    return "rgba(11,95,165," + a.toFixed(3) + ")";
  }

  function renderConfusionMatrix(data) {
    cmEl.innerHTML = "";

    var stats = el("div", { class: "stat-row" }, [
      el("div", { class: "stat-card" }, [el("div", { class: "stat-value", text: fmtPct(data.accuracy) }), el("div", { class: "stat-label", text: "Accuracy" })]),
      el("div", { class: "stat-card" }, [el("div", { class: "stat-value", text: data.macro_f1.toFixed(3) }), el("div", { class: "stat-label", text: "Macro-F1" })]),
      el("div", { class: "stat-card" }, [el("div", { class: "stat-value", text: String(data.n) }), el("div", { class: "stat-label", text: "Test images" })])
    ]);
    cmEl.appendChild(stats);

    var displayNames = data.classes.map(function (c) { return window.PrpdApi.displayName(c); });
    var table = el("table", { class: "data-table heatmap" });
    var headRow = el("tr", {}, [el("th", { text: "True \\ Predicted" })].concat(
      displayNames.map(function (n) { return el("th", { text: n }); })
    ));
    table.appendChild(el("thead", {}, [headRow]));
    var tbody = el("tbody");
    data.counts.forEach(function (row, i) {
      var tr = el("tr");
      tr.appendChild(el("th", { class: "row-head", text: displayNames[i] }));
      row.forEach(function (count, j) {
        var pct = data.row_percent[i][j];
        var td = el("td", {
          text: String(count),
          style: "background:" + heatColor(pct) + (pct > 55 ? ";color:#fff" : ""),
          title: displayNames[i] + " predicted as " + displayNames[j] + ": " + pct.toFixed(1) + "%"
        });
        if (i === j) td.classList.add("diag");
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    cmEl.appendChild(el("div", { class: "data-table-wrap" }, [table]));

    var pcHeading = el("h3", { style: "font-size:0.95rem;margin:18px 0 8px", text: "Per-class precision / recall / F1" });
    cmEl.appendChild(pcHeading);
    var pcTable = el("table", { class: "data-table" });
    pcTable.appendChild(el("thead", {}, [el("tr", {}, [
      el("th", { text: "Fault" }), el("th", { text: "Precision" }), el("th", { text: "Recall" }), el("th", { text: "F1" }), el("th", { text: "n" })
    ])]));
    var pcBody = el("tbody");
    data.per_class.forEach(function (r) {
      pcBody.appendChild(el("tr", {}, [
        el("td", { text: window.PrpdApi.displayName(r.fault_type) }),
        el("td", { text: parseFloat(r.precision).toFixed(3) }),
        el("td", { text: parseFloat(r.recall).toFixed(3) }),
        el("td", { text: parseFloat(r.f1).toFixed(3) }),
        el("td", { text: r.support })
      ]));
    });
    pcTable.appendChild(pcBody);
    cmEl.appendChild(el("div", { class: "data-table-wrap" }, [pcTable]));

    cmEl.appendChild(sourceLink("Confusion matrix CSV", data.source_csv, data.source_csv_url));
    cmEl.appendChild(sourceLink("Per-class report CSV", data.source_per_class_csv, data.source_per_class_csv_url));
    cmEl.appendChild(sourceLink("300 DPI figure", data.source_png, data.source_png_url));
    cmEl.appendChild(el("p", { class: "source-note", text: "Generated by: " + data.generated_by }));
  }

  // ---------------------------------------------------------------- calibration

  function buildReliabilitySvg(bins) {
    var W = 300, H = 300, PAD = 36;
    var plot = W - 2 * PAD;
    function x(conf) { return PAD + conf * plot; }
    function y(acc) { return H - PAD - acc * plot; }

    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="rel-title">';
    svg += '<title id="rel-title">Reliability diagram: accuracy versus confidence per bin</title>';
    // axes
    svg += '<line x1="' + x(0) + '" y1="' + y(0) + '" x2="' + x(1) + '" y2="' + y(0) + '" stroke="var(--color-border)" stroke-width="1"/>';
    svg += '<line x1="' + x(0) + '" y1="' + y(0) + '" x2="' + x(0) + '" y2="' + y(1) + '" stroke="var(--color-border)" stroke-width="1"/>';
    // diagonal (perfect calibration)
    svg += '<line x1="' + x(0) + '" y1="' + y(0) + '" x2="' + x(1) + '" y2="' + y(1) + '" stroke="var(--color-text-muted)" stroke-width="1" stroke-dasharray="4 3"/>';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      svg += '<text x="' + x(t) + '" y="' + (y(0) + 14) + '" font-size="8" fill="var(--color-text-muted)" text-anchor="middle">' + t + '</text>';
      svg += '<text x="' + (x(0) - 6) + '" y="' + (y(t) + 3) + '" font-size="8" fill="var(--color-text-muted)" text-anchor="end">' + t + '</text>';
    });
    svg += '<text x="' + (W / 2) + '" y="' + (H - 4) + '" font-size="9" fill="var(--color-text)" text-anchor="middle">Confidence</text>';
    svg += '<text x="10" y="' + (H / 2) + '" font-size="9" fill="var(--color-text)" text-anchor="middle" transform="rotate(-90 10 ' + (H / 2) + ')">Accuracy</text>';

    var points = bins.map(function (b) { return [x(b.confidence), y(b.accuracy)]; });
    if (points.length > 1) {
      svg += '<polyline points="' + points.map(function (p) { return p[0] + "," + p[1]; }).join(" ") +
        '" fill="none" stroke="var(--color-accent)" stroke-width="1.5"/>';
    }
    points.forEach(function (p, i) {
      var r = Math.max(2.5, Math.min(7, Math.sqrt(bins[i].count) * 1.1));
      svg += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="' + r + '" fill="var(--color-accent)" fill-opacity="0.85"/>';
    });
    svg += "</svg>";
    return svg;
  }

  function renderCalibration(data) {
    calEl.innerHTML = "";
    var s = data.summary;

    var stats = el("div", { class: "stat-row" }, [
      ["ECE", s.ECE.toFixed(4)],
      ["MCE", s.MCE.toFixed(4)],
      ["Brier", s.Brier.toFixed(4)],
      ["NLL", s.NLL.toFixed(4)],
      ["Accuracy", fmtPct(s.accuracy)],
      ["Mean confidence", fmtPct(s.mean_confidence)]
    ].map(function (pair) {
      return el("div", { class: "stat-card" }, [
        el("div", { class: "stat-value", text: pair[1] }),
        el("div", { class: "stat-label", text: pair[0] })
      ]);
    }));
    calEl.appendChild(stats);

    var relWrap = el("div", { class: "reliability-svg-wrap", html: buildReliabilitySvg(data.reliability_bins) });
    calEl.appendChild(el("p", { style: "font-size:0.85rem;color:var(--color-text-muted);margin:0 0 4px", text: "Reliability diagram (15 equal-width confidence bins; dashed line = perfect calibration)" }));
    calEl.appendChild(relWrap);

    var binTable = el("table", { class: "data-table" });
    binTable.appendChild(el("thead", {}, [el("tr", {}, [
      el("th", { text: "Bin" }), el("th", { text: "Count" }), el("th", { text: "Mean confidence" }), el("th", { text: "Accuracy" }), el("th", { text: "Gap" })
    ])]));
    var binBody = el("tbody");
    data.reliability_bins.forEach(function (b) {
      binBody.appendChild(el("tr", {}, [
        el("td", { text: b.lo.toFixed(2) + "–" + b.hi.toFixed(2) }),
        el("td", { text: String(b.count) }),
        el("td", { text: fmtPct(b.confidence) }),
        el("td", { text: fmtPct(b.accuracy) }),
        el("td", { text: (b.gap >= 0 ? "+" : "") + fmtPct(b.gap) })
      ]));
    });
    binTable.appendChild(binBody);
    calEl.appendChild(el("div", { class: "data-table-wrap" }, [binTable]));

    var opHeading = el("h3", { style: "font-size:0.95rem;margin:18px 0 8px", text: "Coverage vs. accuracy (confidence threshold)" });
    calEl.appendChild(opHeading);
    var opTable = el("table", { class: "data-table" });
    opTable.appendChild(el("thead", {}, [el("tr", {}, [
      el("th", { text: "Min. confidence" }), el("th", { text: "Coverage" }), el("th", { text: "Accuracy" }), el("th", { text: "n" })
    ])]));
    var opBody = el("tbody");
    data.operating_points.forEach(function (r) {
      var tr = el("tr", {}, [
        el("td", { text: r.min_confidence }),
        el("td", { text: fmtPct(r.coverage) }),
        el("td", { text: fmtPct(r.accuracy) }),
        el("td", { text: r.n })
      ]);
      if (r.min_confidence === "0.9") tr.style.fontWeight = "700";
      opBody.appendChild(tr);
    });
    opTable.appendChild(opBody);
    calEl.appendChild(el("div", { class: "data-table-wrap" }, [opTable]));

    var row09 = data.operating_points.find(function (r) { return r.min_confidence === "0.9"; });
    if (row09) {
      calEl.appendChild(el("p", {
        class: "callout-row",
        text: "This is the basis for the Classify page's “High confidence” tier: at ≥0.9 confidence, " +
          fmtPct(row09.coverage) + " of test images are covered at " + fmtPct(row09.accuracy) + " accuracy."
      }));
    }

    calEl.appendChild(sourceLink("Calibration summary JSON", data.source_json, data.source_json_url));
    calEl.appendChild(sourceLink("Coverage/accuracy CSV", data.source_ops_csv, data.source_ops_csv_url));
    calEl.appendChild(sourceLink("Raw test-split probabilities", data.source_probs_npz, data.source_probs_npz_url));
    calEl.appendChild(el("p", { class: "source-note", text: "Reliability diagram computed with: " + data.reliability_method }));
  }

  // ---------------------------------------------------------------- bootstrap

  async function load() {
    statusEl.textContent = "Loading dashboard data…";
    try {
      var base = window.PrpdApi.API_PREFIX;
      var [scRes, cmRes, calRes] = await Promise.all([
        fetch(base + "/dashboard/sensor-comparison"),
        fetch(base + "/dashboard/confusion-matrix"),
        fetch(base + "/dashboard/calibration")
      ]);
      if (!scRes.ok || !cmRes.ok || !calRes.ok) throw new Error("One or more dashboard endpoints failed.");
      var sc = await scRes.json();
      var cm = await cmRes.json();
      var cal = await calRes.json();

      renderSensorComparison(sc);
      renderConfusionMatrix(cm);
      renderCalibration(cal);
      statusEl.textContent = "";
    } catch (err) {
      statusEl.textContent = "";
      var div = document.createElement("div");
      div.className = "error-banner";
      div.setAttribute("role", "alert");
      div.textContent = "Could not load dashboard data: " + (err.message || err) +
        ". Make sure the backend is running at " + window.PrpdApi.API_BASE_URL + ".";
      errorEl.appendChild(div);
    }
  }

  load();
})();
