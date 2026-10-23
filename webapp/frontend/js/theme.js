(function () {
  "use strict";

  var STORAGE_KEY = "prpd-theme"; // "light" | "dark" | absent (follow system)

  function apply(theme) {
    if (theme === "light" || theme === "dark") {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }

  function currentEffectiveTheme() {
    var stored = null;
    try {
      stored = localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      /* private browsing / storage blocked: fall through to system preference */
    }
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function setTheme(theme) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (e) {
      /* ignore: toggle still works for this page view */
    }
    apply(theme);
    updateToggleLabel();
  }

  function updateToggleLabel() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var effective = currentEffectiveTheme();
    var next = effective === "dark" ? "light" : "dark";
    btn.querySelector(".theme-toggle-icon").textContent = effective === "dark" ? "☀" : "☽";
    btn.querySelector(".theme-toggle-text").textContent = effective === "dark" ? "Light mode" : "Dark mode";
    btn.setAttribute("aria-label", "Switch to " + next + " mode");
  }

  try {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") apply(stored);
  } catch (e) {
    /* no stored preference available; CSS prefers-color-scheme still applies */
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    updateToggleLabel();
    btn.addEventListener("click", function () {
      var next = currentEffectiveTheme() === "dark" ? "light" : "dark";
      setTheme(next);
    });
  });
})();
