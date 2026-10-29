(function () {
  "use strict";

  var MAX_FILE_BYTES = 15 * 1024 * 1024;
  var TIE_MARGIN = 0.15; // matches backend DEFAULT_TIE_MARGIN (test_model.py)

  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var dzFilename = document.getElementById("dz-filename");
  var previewWrap = document.getElementById("preview-wrap");
  var previewImg = document.getElementById("preview-img");
  var classifyBtn = document.getElementById("classify-btn");
  var sensorSelect = document.getElementById("sensor-select");
  var statusLine = document.getElementById("status-line");
  var errorRegion = document.getElementById("error-region");
  var resultsContent = document.getElementById("results-content");

  var currentFile = null;
  var currentObjectUrl = null;
  var lastVoltageUsed = null;

  // ---------------------------------------------------------------- upload

  function resetError() {
    errorRegion.innerHTML = "";
  }

  function showError(message) {
    resetError();
    var div = document.createElement("div");
    div.className = "error-banner";
    div.setAttribute("role", "alert");
    div.textContent = message;
    errorRegion.appendChild(div);
  }

  function setStatus(message, busy) {
    statusLine.innerHTML = "";
    if (busy) {
      var spinner = document.createElement("span");
      spinner.className = "spinner";
      spinner.setAttribute("aria-hidden", "true");
      statusLine.appendChild(spinner);
    }
    if (message) {
      var span = document.createElement("span");
      span.textContent = message;
      statusLine.appendChild(span);
    }
  }

  function resetResults() {
    resultsContent.innerHTML = "";
    var p = document.createElement("p");
    p.className = "results-placeholder";
    p.textContent = "Upload an image and click Classify to see the predicted fault, confidence, and top candidates here.";
    resultsContent.appendChild(p);
  }

  function validateFile(file) {
    var okType = file.type === "image/png" || file.type === "image/jpeg";
    if (!okType) {
      return "Unsupported file type (" + (file.type || "unknown") + "). Please upload a PNG or JPG.";
    }
    if (file.size > MAX_FILE_BYTES) {
      return "File is too large (" + (file.size / (1024 * 1024)).toFixed(1) + " MB). The limit is 15 MB.";
    }
    return null;
  }

  function handleFile(file) {
    resetError();
    resetResults();
    var problem = validateFile(file);
    if (problem) {
      currentFile = null;
      classifyBtn.disabled = true;
      dzFilename.textContent = "";
      previewWrap.hidden = true;
      showError(problem);
      return;
    }
    currentFile = file;
    lastVoltageUsed = null;
    dzFilename.textContent = file.name;
    classifyBtn.disabled = false;

    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = URL.createObjectURL(file);
    previewImg.src = currentObjectUrl;
    previewWrap.hidden = false;
    setStatus("", false);
  }

  dropzone.addEventListener("click", function () {
    fileInput.click();
  });
  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  dropzone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", function () {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length) {
      handleFile(fileInput.files[0]);
    }
  });

  // ---------------------------------------------------------------- classify

  classifyBtn.addEventListener("click", async function () {
    if (!currentFile) return;
    resetError();
    classifyBtn.disabled = true;
    setStatus("Classifying…", true);

    var sensor = sensorSelect.value;
    lastVoltageUsed = null;
    try {
      var result = await window.PrpdApi.predict(currentFile, { sensor: sensor, topK: 3 });
      setStatus("", false);
      renderResult(result, sensor);
    } catch (err) {
      setStatus("", false);
      showError(err.message || "Something went wrong while classifying this image.");
    } finally {
      classifyBtn.disabled = !currentFile;
    }
  });

  // ---------------------------------------------------------------- rendering

  function confidenceTierInfo(tier) {
    if (tier === "accept") return { icon: "✓", cls: "tier-accept" };
    if (tier === "review") return { icon: "!", cls: "tier-review" };
    return { icon: "?", cls: "tier-manual" };
  }

  function faultTitle(code) {
    if (!window.PrpdLibrary) return null;
    var entry = window.PrpdLibrary.find(function (f) { return f.code === code; });
    return entry ? entry.title : null;
  }

  function confidenceExplainer(tier, confidence) {
    // Wording and figures taken only from the thesis's calibration results
    // (chapters/09_calibration.tex; stage2_output/calibration_operating_points_sensor_2.csv).
    var pct = (confidence * 100).toFixed(1);
    if (tier === "accept") {
      return "What this means: at or above 90% confidence, this model was correct on 98.4% of the matching test images " +
        "(covering about two-thirds of the test set). This prediction (" + pct + "%) falls in that band.";
    }
    if (tier === "review") {
      return "What this means: at 70% confidence or above, this model was correct about 93.7% of the time in testing, " +
        "below the 90% band's 98.4%. This prediction (" + pct + "%) falls between 70% and 90% — a second look is recommended.";
    }
    return "What this means: below 70% confidence, correctness drops noticeably in testing. This prediction (" + pct +
      "%) is below that line — verify this result manually before relying on it.";
  }

  function buildTop3(top) {
    var ul = document.createElement("ul");
    ul.className = "top3-list";
    top.forEach(function (entry) {
      var li = document.createElement("li");
      li.className = "top3-row";

      var name = document.createElement("span");
      name.className = "top3-name";
      name.textContent = entry.fault_display;

      var track = document.createElement("span");
      track.className = "top3-bar-track";
      var fill = document.createElement("span");
      fill.className = "top3-bar-fill";
      fill.style.width = Math.max(2, entry.probability * 100) + "%";
      track.appendChild(fill);

      var pct = document.createElement("span");
      pct.className = "top3-pct";
      var pctValue = (entry.probability * 100).toFixed(1) + "%";
      pct.textContent = pctValue;

      li.setAttribute("aria-label", entry.fault_display + ": " + pctValue);
      li.appendChild(name);
      li.appendChild(track);
      li.appendChild(pct);
      ul.appendChild(li);
    });
    return ul;
  }

  function detectTie(top) {
    if (!top || top.length < 2) return null;
    var topP = top[0].probability;
    var tied = top.filter(function (e) {
      return topP - e.probability <= TIE_MARGIN;
    });
    return tied.length > 1 ? tied : null;
  }

  function renderResult(result, sensor) {
    resultsContent.innerHTML = "";

    if (!result.size_matches_training) {
      var warn = document.createElement("div");
      warn.className = "warning-banner";
      warn.setAttribute("role", "alert");
      warn.textContent = "This image is " + result.image_size[0] + "×" + result.image_size[1] +
        " px, but the training plots are 1800×1050 px. The result below is not trustworthy.";
      resultsContent.appendChild(warn);
    }

    var header = document.createElement("div");
    header.className = "result-header";

    var faultName = document.createElement("div");
    faultName.className = "result-fault";
    faultName.textContent = result.fault_display;

    var tierInfo = confidenceTierInfo(result.confidence_tier.tier);
    var badge = document.createElement("span");
    badge.className = "confidence-badge " + tierInfo.cls;
    var badgeIcon = document.createElement("span");
    badgeIcon.className = "badge-icon";
    badgeIcon.setAttribute("aria-hidden", "true");
    badgeIcon.textContent = tierInfo.icon;
    var badgeText = document.createElement("span");
    badgeText.textContent = result.confidence_tier.label + " — " + (result.confidence * 100).toFixed(1) + "%";
    badge.appendChild(badgeIcon);
    badge.appendChild(badgeText);

    header.appendChild(faultName);
    header.appendChild(badge);
    resultsContent.appendChild(header);

    var title = faultTitle(result.fault);
    if (title) {
      var titleEl = document.createElement("p");
      titleEl.className = "result-fault-title";
      titleEl.textContent = title;
      resultsContent.appendChild(titleEl);
    }

    var explainer = document.createElement("p");
    explainer.className = "confidence-explainer";
    explainer.textContent = confidenceExplainer(result.confidence_tier.tier, result.confidence);
    resultsContent.appendChild(explainer);

    var top3Heading = document.createElement("h3");
    top3Heading.style.fontSize = "0.95rem";
    top3Heading.style.margin = "0 0 8px";
    top3Heading.textContent = "Top candidates";
    resultsContent.appendChild(top3Heading);
    resultsContent.appendChild(buildTop3(result.top));

    var agree = document.createElement("p");
    agree.className = "model-agreement" + (result.models_agree ? "" : " disagree");
    agree.textContent = result.models_agree
      ? "Both component models (prototypical network and ResNet-18) agree on this prediction."
      : "The two component models disagree: prototypical network predicts " + window.PrpdApi.displayName(result.prototypical.fault) +
        ", ResNet-18 predicts " + window.PrpdApi.displayName(result.resnet18.fault) + ". Treat this result with extra caution.";
    resultsContent.appendChild(agree);

    var tied = detectTie(result.top);
    if (tied) {
      resultsContent.appendChild(buildVoltagePanel(currentFile, sensor, tied));
    }

    resultsContent.appendChild(buildReportButton(currentFile, sensor));
  }

  function buildReportButton(file, sensor) {
    var wrap = document.createElement("div");
    wrap.style.marginTop = "18px";
    wrap.style.paddingTop = "16px";
    wrap.style.borderTop = "1px solid var(--color-border)";

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-secondary";
    btn.textContent = "Download PDF report";

    var status = document.createElement("span");
    status.style.marginLeft = "10px";
    status.style.fontSize = "0.85rem";
    status.style.color = "var(--color-text-muted)";

    btn.addEventListener("click", async function () {
      btn.disabled = true;
      status.textContent = "Building report…";
      try {
        var options = { sensor: sensor };
        if (lastVoltageUsed !== null) options.voltage = lastVoltageUsed;
        var blob = await window.PrpdApi.downloadReport(file, options);
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "prpd_classification_report.pdf";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        status.textContent = "";
      } catch (e) {
        status.textContent = e.message || "Could not build the report.";
      } finally {
        btn.disabled = false;
      }
    });

    wrap.appendChild(btn);
    wrap.appendChild(status);
    return wrap;
  }

  function buildVoltagePanel(file, sensor, tiedClasses) {
    var wrap = document.createElement("div");
    wrap.className = "voltage-panel";

    var p = document.createElement("p");
    p.style.margin = "0 0 10px";
    p.style.fontSize = "0.9rem";
    var names = tiedClasses.map(function (e) { return e.fault_display; }).join(" and ");
    p.textContent = "These classes are close in confidence (" + names + "). If you know the discharge inception voltage, " +
      "enter it to check which class it is consistent with.";
    wrap.appendChild(p);

    var row = document.createElement("div");
    row.className = "voltage-row";

    var field = document.createElement("div");
    field.className = "field";
    var label = document.createElement("label");
    label.setAttribute("for", "voltage-input");
    label.textContent = "Voltage (kV)";
    var input = document.createElement("input");
    input.type = "number";
    input.id = "voltage-input";
    input.step = "0.1";
    input.min = "0";
    field.appendChild(label);
    field.appendChild(input);

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-secondary";
    btn.textContent = "Check with voltage";

    row.appendChild(field);
    row.appendChild(btn);
    wrap.appendChild(row);

    var outcome = document.createElement("div");
    outcome.id = "voltage-outcome";
    wrap.appendChild(outcome);

    btn.addEventListener("click", async function () {
      var voltage = parseFloat(input.value);
      if (isNaN(voltage) || voltage < 0) {
        outcome.innerHTML = "";
        var err = document.createElement("div");
        err.className = "error-banner";
        err.textContent = "Enter a valid voltage in kV.";
        outcome.appendChild(err);
        return;
      }
      btn.disabled = true;
      outcome.innerHTML = "";
      try {
        var result = await window.PrpdApi.predict(file, { sensor: sensor, voltage: voltage, topK: 3 });
        lastVoltageUsed = voltage;
        renderVoltageOutcome(outcome, result);
      } catch (e) {
        var err2 = document.createElement("div");
        err2.className = "error-banner";
        err2.textContent = e.message || "Could not check voltage.";
        outcome.appendChild(err2);
      } finally {
        btn.disabled = false;
      }
    });

    return wrap;
  }

  function renderVoltageOutcome(container, result) {
    container.innerHTML = "";
    var box = document.createElement("div");
    box.className = "voltage-result";

    var final = document.createElement("div");
    final.innerHTML = "";
    var strong = document.createElement("strong");
    strong.textContent = result.final_fault_display;
    final.appendChild(document.createTextNode("Final: "));
    final.appendChild(strong);
    box.appendChild(final);

    var finalTitle = faultTitle(result.final_fault);
    if (finalTitle) {
      var finalTitleEl = document.createElement("div");
      finalTitleEl.style.marginTop = "2px";
      finalTitleEl.style.color = "var(--color-text-muted)";
      finalTitleEl.style.fontSize = "0.88rem";
      finalTitleEl.textContent = finalTitle;
      box.appendChild(finalTitleEl);
    }

    var action = document.createElement("div");
    action.style.marginTop = "6px";
    action.style.color = "var(--color-text-muted)";
    action.textContent = result.voltage_check ? result.voltage_check.action : "";
    box.appendChild(action);

    container.appendChild(box);
  }

  resetResults();
})();
