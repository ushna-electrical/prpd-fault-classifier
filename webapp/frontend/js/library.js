(function () {
  "use strict";

  var listEl = document.getElementById("fault-list");
  var detailEl = document.getElementById("library-detail");
  var DATA = window.PrpdLibrary;

  function codeFromHash() {
    var h = window.location.hash.replace("#", "");
    return DATA.some(function (f) { return f.code === h; }) ? h : DATA[0].code;
  }

  function buildList() {
    listEl.innerHTML = "";
    DATA.forEach(function (f) {
      var li = document.createElement("li");
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "fault-list-btn";
      btn.id = "nav-" + f.code;

      var code = document.createElement("span");
      code.className = "code";
      code.textContent = f.display;

      var title = document.createTextNode(f.title);

      var family = document.createElement("span");
      family.className = "family";
      family.textContent = f.defectFamily;

      btn.appendChild(code);
      btn.appendChild(title);
      btn.appendChild(family);
      btn.addEventListener("click", function () {
        window.location.hash = f.code;
      });

      li.appendChild(btn);
      listEl.appendChild(li);
    });
  }

  function renderDetail(code) {
    var f = DATA.find(function (x) { return x.code === code; });
    if (!f) return;

    document.querySelectorAll(".fault-list-btn").forEach(function (btn) {
      var isCurrent = btn.id === "nav-" + code;
      if (isCurrent) {
        btn.setAttribute("aria-current", "true");
      } else {
        btn.removeAttribute("aria-current");
      }
    });

    detailEl.innerHTML = "";

    var header = document.createElement("div");
    header.className = "detail-header";
    var codeSpan = document.createElement("span");
    codeSpan.className = "detail-code";
    codeSpan.textContent = f.display;
    header.appendChild(codeSpan);
    detailEl.appendChild(header);

    var title = document.createElement("h2");
    title.className = "detail-title";
    title.textContent = f.title;
    detailEl.appendChild(title);

    var meta = document.createElement("div");
    meta.className = "detail-meta";
    [
      ["Defect family", f.defectFamily],
      ["Cables", f.cables],
      ["Recordings", f.recordings]
    ].forEach(function (pair) {
      var div = document.createElement("div");
      var span = document.createElement("span");
      span.textContent = pair[0];
      div.appendChild(span);
      div.appendChild(document.createTextNode(pair[1]));
      meta.appendChild(div);
    });
    detailEl.appendChild(meta);

    if (f.note) {
      var note = document.createElement("div");
      note.className = "detail-note";
      note.textContent = f.note;
      detailEl.appendChild(note);
    }

    var text = document.createElement("div");
    text.className = "detail-text";
    f.detail.forEach(function (para) {
      var p = document.createElement("p");
      p.textContent = para;
      text.appendChild(p);
    });
    detailEl.appendChild(text);

    var imagesHeading = document.createElement("h3");
    imagesHeading.style.fontSize = "0.95rem";
    imagesHeading.style.margin = "0 0 4px";
    imagesHeading.textContent = "Example PRPD images";
    detailEl.appendChild(imagesHeading);

    var images = document.createElement("div");
    images.className = "detail-images";
    f.images.forEach(function (img) {
      var figure = document.createElement("figure");
      var image = document.createElement("img");
      image.src = window.PrpdApi.libraryImageUrl(img.file);
      image.alt = f.display + " (" + f.title + "), recorded on Sensor " + img.sensor;
      image.loading = "lazy";
      var caption = document.createElement("figcaption");
      caption.textContent = "Sensor " + img.sensor;
      figure.appendChild(image);
      figure.appendChild(caption);
      images.appendChild(figure);
    });
    detailEl.appendChild(images);

    var source = document.createElement("p");
    source.className = "detail-source";
    source.textContent = "Source: " + f.source;
    detailEl.appendChild(source);
  }

  window.addEventListener("hashchange", function () {
    renderDetail(codeFromHash());
  });

  buildList();
  renderDetail(codeFromHash());
})();
