/* Shared API access for the PRPD Fault Classifier backend (FastAPI).
 * Reads the backend origin from window.PRPD_CONFIG (see js/config.js); every
 * backend route lives under /api, so API_PREFIX is appended to every path
 * built here. */
(function (global) {
  "use strict";

  var API_BASE_URL = (global.PRPD_CONFIG && global.PRPD_CONFIG.API_BASE_URL !== undefined)
    ? global.PRPD_CONFIG.API_BASE_URL
    : "http://127.0.0.1:8000";
  var API_PREFIX = API_BASE_URL + "/api";

  var FAULT_DISPLAY_FALLBACK = {
    fault_2: "F2 (includes F3)",
    fault_4: "F4",
    fault_5: "F5",
    fault_6: "F6",
    fault_7: "F7",
    fault_8: "F8",
    fault_9: "F9",
    fault_10: "F10",
    fault_12: "F12",
    fault_13: "F13"
  };

  function displayName(code) {
    return FAULT_DISPLAY_FALLBACK[code] || code;
  }

  function libraryImageUrl(filename) {
    return API_PREFIX + "/library-images/" + encodeURIComponent(filename);
  }

  function apiError(message) {
    var err = new Error(message);
    err.isApiError = true;
    return err;
  }

  async function getClasses(sensor) {
    var res = await fetch(API_PREFIX + "/classes?sensor=" + encodeURIComponent(sensor));
    if (!res.ok) throw apiError("Could not load class list (" + res.status + ")");
    return res.json();
  }

  async function predict(file, options) {
    options = options || {};
    var form = new FormData();
    form.append("file", file, file.name);
    form.append("sensor", options.sensor || "2");
    form.append("top_k", String(options.topK || 3));
    if (options.voltage !== undefined && options.voltage !== null && options.voltage !== "") {
      form.append("voltage", String(options.voltage));
    }
    if (options.tieMargin !== undefined && options.tieMargin !== null) {
      form.append("tie_margin", String(options.tieMargin));
    }

    var res;
    try {
      res = await fetch(API_PREFIX + "/predict", { method: "POST", body: form });
    } catch (networkErr) {
      throw apiError(
        "Could not reach the classifier backend at " + API_PREFIX +
        ". Make sure the server is running (see README.md)."
      );
    }

    var body;
    try {
      body = await res.json();
    } catch (parseErr) {
      throw apiError("The server returned an unreadable response (status " + res.status + ").");
    }

    if (!res.ok) {
      throw apiError(body && body.detail ? body.detail : "Request failed (status " + res.status + ").");
    }
    return body;
  }

  async function downloadReport(file, options) {
    options = options || {};
    var form = new FormData();
    form.append("file", file, file.name);
    form.append("sensor", options.sensor || "2");
    if (options.voltage !== undefined && options.voltage !== null && options.voltage !== "") {
      form.append("voltage", String(options.voltage));
    }
    if (options.tieMargin !== undefined && options.tieMargin !== null) {
      form.append("tie_margin", String(options.tieMargin));
    }

    var res;
    try {
      res = await fetch(API_PREFIX + "/report", { method: "POST", body: form });
    } catch (networkErr) {
      throw apiError("Could not reach the classifier backend at " + API_PREFIX + " to build the report.");
    }
    if (!res.ok) {
      var detail = "Report request failed (status " + res.status + ").";
      try {
        var body = await res.json();
        if (body && body.detail) detail = body.detail;
      } catch (e) { /* non-JSON error body: keep generic message */ }
      throw apiError(detail);
    }
    return res.blob();
  }

  global.PrpdApi = {
    API_BASE_URL: API_BASE_URL,
    API_PREFIX: API_PREFIX,
    displayName: displayName,
    libraryImageUrl: libraryImageUrl,
    getClasses: getClasses,
    predict: predict,
    downloadReport: downloadReport
  };
})(window);
