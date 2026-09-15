/* Antalia 1 sample site. Renders gallery.json into the page and drives the players. No dependencies. */
(function () {
  "use strict";

  var STRINGS = {
    en: {
      raw: "Single seed",
      selected: "Best of 8",
      seed: "seed {seed}",
      candidate: "{candidate}",
      cer: "CER",
      sim: "Sim",
      play: "Play {variant}: {text}",
      pause: "Pause {variant}",
      seek: "Seek in {variant}",
      download: "WAV",
      downloadLabel: "Download {variant} audio for “{text}”",
      heard: "Whisper heard:",
      range: "Across 8 seeds: CER {min}–{max}",
      recovered: "Single seed CER {raw}. Best of 8 recovered it to {selected}.",
      notRecovered: "Not recovered by selection: CER {raw} single seed, {selected} selected.",
      lowerSeedLost: " A lower-CER seed existed but lost on the combined selection score.",
      all: "All",
      unchunked: "Unchunked",
      unchunkedMeta: "370 characters in one pass",
      chunked: "Chunked at ≤120 characters",
      chunkedMeta: "160 ms pauses · 0.085 s/char floor",
      durationLabel: "Rendered duration",
      foundation: "Foundation, unconditioned",
      foundationMeta: "antalia-1-foundation · speaker id 0",
      steps16: "16 Euler steps",
      steps16Meta: "−45.1 % generation time",
      steps32: "32 Euler steps",
      steps32Meta: "release recipe · seed {seed}",
      copy: "Copy",
      copied: "Copied",
      loadError: "Unable to load gallery.json. Serve this folder over HTTP and reload.",
      categories: {
        acknowledgement: "Acknowledgement",
        adversarial_normalization: "Normalization",
        emotional_style: "Emotional style",
        foreign_abbreviations: "Foreign terms",
        general: "General",
        long_form: "Long form",
        names_places: "Names & places",
        numeric: "Numbers",
        questions_confirmations: "Questions",
        voice_agent: "Voice agent",
        foundation: "Foundation model",
        low_latency: "Low latency"
      }
    },
    tr: {
      raw: "Tek tohum",
      selected: "8’de en iyi",
      seed: "tohum {seed}",
      candidate: "{candidate}",
      cer: "CER",
      sim: "Benz.",
      play: "{variant} çal: {text}",
      pause: "{variant} duraklat",
      seek: "{variant} içinde ilerle",
      download: "WAV",
      downloadLabel: "“{text}” için {variant} sesini indir",
      heard: "Whisper’ın duyduğu:",
      range: "8 tohum boyunca: CER {min}–{max}",
      recovered: "Tek tohumda CER {raw}. 8’de en iyi seçimi {selected} değerine düzeltti.",
      notRecovered: "Seçim düzeltemedi: tek tohumda CER {raw}, seçilende {selected}.",
      lowerSeedLost: " Daha düşük CER’li bir tohum vardı ama birleşik seçim puanında geride kaldı.",
      all: "Tümü",
      unchunked: "Parçalanmamış",
      unchunkedMeta: "370 karakter tek geçişte",
      chunked: "≤120 karakterde parçalanmış",
      chunkedMeta: "160 ms duraklama · 0,085 s/karakter tabanı",
      durationLabel: "Üretilen süre",
      foundation: "Temel model, koşulsuz",
      foundationMeta: "antalia-1-foundation · konuşmacı kimliği 0",
      steps16: "16 Euler adımı",
      steps16Meta: "üretim süresi −%45,1",
      steps32: "32 Euler adımı",
      steps32Meta: "yayın tarifi · tohum {seed}",
      copy: "Kopyala",
      copied: "Kopyalandı",
      loadError: "gallery.json yüklenemedi. Bu klasörü HTTP üzerinden sunup sayfayı yenileyin.",
      categories: {
        acknowledgement: "Onaylama",
        adversarial_normalization: "Normalleştirme",
        emotional_style: "Duygusal ton",
        foreign_abbreviations: "Yabancı terimler",
        general: "Genel",
        long_form: "Uzun metin",
        names_places: "İsimler ve yerler",
        numeric: "Sayılar",
        questions_confirmations: "Sorular",
        voice_agent: "Sesli asistan",
        foundation: "Temel model",
        low_latency: "Düşük gecikme"
      }
    }
  };

  var main = document.querySelector("main[data-gallery]");
  if (!main) return;

  var lang = main.getAttribute("data-lang") === "tr" ? "tr" : "en";
  var t = STRINGS[lang];
  var audioBase = main.getAttribute("data-audio-base") || "audio/";
  var galleryUrl = main.getAttribute("data-gallery") || "gallery.json";
  var SVG_NS = "http://www.w3.org/2000/svg";

  /* Helpers ---------------------------------------------------------------- */

  function fmt(template, vars) {
    return template.replace(/\{(\w+)\}/g, function (_, key) {
      return vars[key] == null ? "" : String(vars[key]);
    });
  }

  function num(value, digits) {
    var s = Number(value).toFixed(digits);
    return lang === "tr" ? s.replace(".", ",") : s;
  }

  function clock(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds - m * 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      for (var key in attrs) {
        if (attrs[key] == null) continue;
        if (key === "text") node.textContent = attrs[key];
        else if (key === "class") node.className = attrs[key];
        else node.setAttribute(key, attrs[key]);
      }
    }
    if (children) {
      for (var i = 0; i < children.length; i++) {
        if (children[i] == null) continue;
        node.appendChild(typeof children[i] === "string" ? document.createTextNode(children[i]) : children[i]);
      }
    }
    return node;
  }

  function icon(kind) {
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("fill", "currentColor");
    var path = document.createElementNS(SVG_NS, "path");
    if (kind === "play") {
      path.setAttribute("d", "M7 4.5v15a1 1 0 0 0 1.5.86l12-7.5a1 1 0 0 0 0-1.72l-12-7.5A1 1 0 0 0 7 4.5z");
      svg.setAttribute("class", "i-play");
    } else if (kind === "pause") {
      path.setAttribute("d", "M6 4h4v16H6zM14 4h4v16h-4z");
      svg.setAttribute("class", "i-pause");
    } else if (kind === "download") {
      svg.setAttribute("fill", "none");
      svg.setAttribute("stroke", "currentColor");
      svg.setAttribute("stroke-width", "2");
      svg.setAttribute("stroke-linecap", "round");
      svg.setAttribute("stroke-linejoin", "round");
      path.setAttribute("d", "M12 4v11m0 0 4-4m-4 4-4-4M5 19h14");
    } else if (kind === "copy") {
      svg.setAttribute("fill", "none");
      svg.setAttribute("stroke", "currentColor");
      svg.setAttribute("stroke-width", "2");
      svg.setAttribute("stroke-linecap", "round");
      svg.setAttribute("stroke-linejoin", "round");
      path.setAttribute("d", "M9 9h10v11H9zM5 15V4h10");
    } else if (kind === "check") {
      svg.setAttribute("fill", "none");
      svg.setAttribute("stroke", "currentColor");
      svg.setAttribute("stroke-width", "2.5");
      svg.setAttribute("stroke-linecap", "round");
      svg.setAttribute("stroke-linejoin", "round");
      path.setAttribute("d", "M20 6 9 17l-5-5");
    }
    svg.appendChild(path);
    return svg;
  }

  function categoryName(key) {
    return t.categories[key] || key;
  }

  function cerClass(cer) {
    if (cer == null) return "";
    if (cer <= 0.05) return " metric--good";
    if (cer <= 0.15) return " metric--warn";
    return " metric--bad";
  }

  function metric(label, value, digits, cls) {
    return el("span", { class: "metric" + (cls || "") }, [label + " ", el("b", { text: num(value, digits) })]);
  }

  /* Player ----------------------------------------------------------------- */

  var current = null;

  function take(opts) {
    var src = audioBase + opts.audio;
    var audio = el("audio", { preload: "none" });
    var playBtn = el("button", {
      class: "play",
      type: "button",
      "aria-label": fmt(t.play, { variant: opts.variant, text: opts.text })
    }, [icon("play"), icon("pause")]);
    var seek = el("input", {
      class: "seek", type: "range", min: "0", max: "1000", value: "0", step: "1",
      "aria-label": fmt(t.seek, { variant: opts.variant }),
      "aria-valuetext": clock(0)
    });
    var total = opts.seconds != null ? opts.seconds : 0;
    var time = el("span", { class: "time", text: clock(0) + " / " + clock(total) });
    var stats = el("div", { class: "take-stats" }, (opts.stats || []).concat([
      el("a", {
        class: "dl", href: src, download: "",
        "aria-label": fmt(t.downloadLabel, { variant: opts.variant, text: opts.text })
      }, [icon("download"), t.download])
    ]));
    var root = el("div", { class: "take" }, [
      playBtn,
      el("div", { class: "take-head" }, [
        el("span", { class: "take-name", text: opts.variant }),
        opts.meta ? el("span", { class: "take-meta", text: opts.meta }) : null
      ]),
      el("div", { class: "take-bar" }, [seek, time]),
      stats,
      audio
    ]);

    var loaded = false;
    var scrubbing = false;

    function duration() {
      return isFinite(audio.duration) && audio.duration > 0 ? audio.duration : total;
    }

    function paint() {
      var d = duration();
      var p = d ? audio.currentTime / d : 0;
      if (!scrubbing) seek.value = String(Math.round(p * 1000));
      seek.style.setProperty("--p", (p * 100).toFixed(2) + "%");
      seek.setAttribute("aria-valuetext", clock(audio.currentTime));
      time.textContent = clock(audio.currentTime) + " / " + clock(d);
    }

    function stop() {
      audio.pause();
      audio.currentTime = 0;
      root.classList.remove("is-playing");
      playBtn.setAttribute("aria-label", fmt(t.play, { variant: opts.variant, text: opts.text }));
      paint();
    }

    function play() {
      if (current && current !== stop) current();
      current = stop;
      if (!loaded) {
        audio.src = src;
        loaded = true;
        root.classList.add("is-loading");
      }
      var attempt = audio.play();
      if (attempt && attempt.catch) attempt.catch(function () { root.classList.remove("is-loading"); });
    }

    playBtn.addEventListener("click", function () {
      if (audio.paused) play();
      else {
        audio.pause();
        root.classList.remove("is-playing");
        playBtn.setAttribute("aria-label", fmt(t.play, { variant: opts.variant, text: opts.text }));
      }
    });
    audio.addEventListener("playing", function () {
      root.classList.remove("is-loading");
      root.classList.add("is-playing");
      playBtn.setAttribute("aria-label", fmt(t.pause, { variant: opts.variant }));
    });
    audio.addEventListener("pause", function () {
      root.classList.remove("is-playing");
    });
    audio.addEventListener("timeupdate", paint);
    audio.addEventListener("loadedmetadata", paint);
    audio.addEventListener("ended", stop);
    seek.addEventListener("input", function () {
      scrubbing = true;
      var d = duration();
      var target = (Number(seek.value) / 1000) * d;
      time.textContent = clock(target) + " / " + clock(d);
      seek.style.setProperty("--p", (Number(seek.value) / 10).toFixed(2) + "%");
    });
    seek.addEventListener("change", function () {
      scrubbing = false;
      if (!loaded) { audio.src = src; loaded = true; }
      var apply = function () { audio.currentTime = (Number(seek.value) / 1000) * duration(); paint(); };
      if (audio.readyState >= 1) apply();
      else audio.addEventListener("loadedmetadata", apply, { once: true });
    });
    return root;
  }

  /* Rows --------------------------------------------------------------------- */

  function tags(entry, category) {
    return el("div", { class: "sample-tags" }, [
      el("span", { class: "tag", text: categoryName(category) }),
      entry.id ? el("span", { class: "tag tag--id", text: entry.id }) : null
    ]);
  }

  function row(entry, opts) {
    var lead = [tags(entry, opts.category), el("p", { class: "sentence", lang: "tr", text: entry.text })];
    if (opts.extra) lead = lead.concat(opts.extra);
    return el("article", {
      class: "sample",
      "data-category": opts.category,
      "aria-label": entry.text
    }, [
      el("div", { class: "sample-lead" }, lead),
      el("div", { class: "takes" + (opts.takes.length === 1 ? " takes--single" : "") }, opts.takes)
    ]);
  }

  function pairTakes(entry) {
    return [
      take({
        audio: entry.raw.audio, text: entry.text, variant: t.raw, seconds: entry.raw.seconds,
        meta: fmt(t.seed, { seed: entry.raw.seed }),
        stats: [metric(t.cer, entry.raw.cer, 2, cerClass(entry.raw.cer)), metric(t.sim, entry.raw.speaker_similarity, 3)]
      }),
      take({
        audio: entry.selected.audio, text: entry.text, variant: t.selected, seconds: entry.selected.seconds,
        meta: fmt(t.candidate, { candidate: entry.selected.candidate }),
        stats: [metric(t.cer, entry.selected.cer, 2, cerClass(entry.selected.cer)), metric(t.sim, entry.selected.speaker_similarity, 3)]
      })
    ];
  }

  function failureExtras(entry) {
    var vars = {
      raw: num(entry.raw.cer, 2), selected: num(entry.selected.cer, 2),
      min: num(entry.candidate_cer_min, 2), max: num(entry.candidate_cer_max, 2)
    };
    var recovered = entry.selected.cer < entry.raw.cer - 0.0005;
    var noteText = fmt(recovered ? t.recovered : t.notRecovered, vars);
    if (!recovered && entry.candidate_cer_min < entry.selected.cer - 0.0005) noteText += t.lowerSeedLost;

    var scale = 0.5;
    function pct(v) { return (Math.min(v, scale) / scale * 100).toFixed(1) + "%"; }
    var span = el("div", { class: "cer-span" });
    span.style.left = pct(entry.candidate_cer_min);
    span.style.width = (Math.min(entry.candidate_cer_max, scale) / scale * 100 - Math.min(entry.candidate_cer_min, scale) / scale * 100).toFixed(1) + "%";
    var rawDot = el("i", { class: "cer-dot cer-dot--raw", title: t.raw + " " + vars.raw });
    rawDot.style.left = pct(entry.raw.cer);
    var selDot = el("i", { class: "cer-dot cer-dot--sel", title: t.selected + " " + vars.selected });
    selDot.style.left = pct(entry.selected.cer);

    var out = [el("p", { class: "sample-note", text: noteText })];
    if (entry.raw.asr_text) {
      out.push(el("p", { class: "heard", lang: "tr" }, [el("b", { lang: lang, text: t.heard }), " ", el("q", { text: entry.raw.asr_text })]));
    }
    out.push(el("div", { class: "cer-range", role: "img", "aria-label": fmt(t.range, vars) }, [
      el("div", { class: "cer-track" }, [span, rawDot, selDot]),
      el("small", { text: fmt(t.range, vars) })
    ]));
    return out;
  }

  /* Filters -------------------------------------------------------------------- */

  function buildFilters(container, list, entries) {
    var counts = {};
    var order = [];
    entries.forEach(function (e) {
      if (!counts[e.category]) { counts[e.category] = 0; order.push(e.category); }
      counts[e.category] += 1;
    });
    var buttons = [];
    function apply(key) {
      buttons.forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-key") === key ? "true" : "false"); });
      var rows = list.querySelectorAll(".sample");
      for (var i = 0; i < rows.length; i++) {
        rows[i].hidden = key !== "all" && rows[i].getAttribute("data-category") !== key;
      }
    }
    function button(key, label, count) {
      var b = el("button", { class: "filter", type: "button", "aria-pressed": key === "all" ? "true" : "false", "data-key": key }, [
        label, el("small", { text: String(count) })
      ]);
      b.addEventListener("click", function () { apply(key); });
      buttons.push(b);
      return b;
    }
    container.appendChild(button("all", t.all, entries.length));
    order.forEach(function (key) { container.appendChild(button(key, categoryName(key), counts[key])); });
  }

  /* Render ---------------------------------------------------------------------- */

  function render(data) {
    var i, frag;
    var byId = {};
    data.showcase.forEach(function (e) { byId[e.id] = e; });
    var extras = {};
    data.extras.forEach(function (e) { extras[e.id] = e; });

    var feature = document.querySelector("[data-feature]");
    if (feature) {
      var fe = byId[feature.getAttribute("data-feature")];
      var slot = feature.querySelector("[data-feature-take]");
      if (fe && slot) {
        slot.replaceChildren(take({
          audio: fe.selected.audio, text: fe.text, variant: t.selected, seconds: fe.selected.seconds,
          meta: fmt(t.candidate, { candidate: fe.selected.candidate }),
          stats: [metric(t.cer, fe.selected.cer, 2, cerClass(fe.selected.cer)), metric(t.sim, fe.selected.speaker_similarity, 3)]
        }));
      }
    }

    var showcase = document.getElementById("showcase-list");
    if (showcase) {
      frag = document.createDocumentFragment();
      for (i = 0; i < data.showcase.length; i++) {
        frag.appendChild(row(data.showcase[i], { category: data.showcase[i].category, takes: pairTakes(data.showcase[i]) }));
      }
      showcase.replaceChildren(frag);
      var filters = document.getElementById("showcase-filters");
      if (filters) buildFilters(filters, showcase, data.showcase);
    }

    var failures = document.getElementById("failures-list");
    if (failures) {
      frag = document.createDocumentFragment();
      for (i = 0; i < data.failures.length; i++) {
        frag.appendChild(row(data.failures[i], {
          category: data.failures[i].category,
          extra: failureExtras(data.failures[i]),
          takes: pairTakes(data.failures[i])
        }));
      }
      failures.replaceChildren(frag);
    }

    var longList = document.getElementById("long-list");
    if (longList && extras["long-unchunked"] && extras["long-chunked"]) {
      var lu = extras["long-unchunked"], lc = extras["long-chunked"];
      var luS = lu.seconds || 9.1, lcS = lc.seconds || 34.8;
      var compare = el("div", { class: "duration-compare", role: "img", "aria-label": t.durationLabel + ": " + t.unchunked + " " + num(luS, 1) + " s, " + t.chunked + " " + num(lcS, 1) + " s" }, [
        el("div", { class: "duration-row" }, [el("span", { text: t.unchunked }), el("div", { class: "duration-bar", style: "width:" + (luS / lcS * 100).toFixed(1) + "%" }, [el("b", { text: num(luS, 1) + " s" })])]),
        el("div", { class: "duration-row" }, [el("span", { text: t.chunked }), el("div", { class: "duration-bar duration-bar--full", style: "width: calc(100% - 3.5rem)" }, [el("b", { text: num(lcS, 1) + " s" })])])
      ]);
      longList.replaceChildren(row({ id: "long-input", text: lu.text }, {
        category: "long_form",
        extra: [compare],
        takes: [
          take({ audio: lu.audio, text: lu.text, variant: t.unchunked, meta: t.unchunkedMeta, seconds: lu.seconds }),
          take({ audio: lc.audio, text: lc.text, variant: t.chunked, meta: t.chunkedMeta, seconds: lc.seconds })
        ]
      }));
    }

    var foundationList = document.getElementById("foundation-list");
    if (foundationList) {
      frag = document.createDocumentFragment();
      data.extras.forEach(function (f) {
        if (f.role !== "foundation") return;
        var twin = byId[f.id.replace(/-foundation$/, "")];
        var takes = [take({ audio: f.audio, text: f.text, variant: t.foundation, meta: t.foundationMeta, seconds: f.seconds })];
        if (twin) {
          takes.push(take({
            audio: twin.selected.audio, text: twin.text, variant: "Antalia 1 · " + t.selected, seconds: twin.selected.seconds,
            meta: fmt(t.candidate, { candidate: twin.selected.candidate }),
            stats: [metric(t.cer, twin.selected.cer, 2, cerClass(twin.selected.cer)), metric(t.sim, twin.selected.speaker_similarity, 3)]
          }));
        }
        frag.appendChild(row(f, { category: "foundation", takes: takes }));
      });
      foundationList.replaceChildren(frag);
    }

    var stepsList = document.getElementById("steps-list");
    if (stepsList && extras["general-001-16steps"]) {
      var s16 = extras["general-001-16steps"];
      var s32 = byId["general-001"];
      var stepTakes = [take({ audio: s16.audio, text: s16.text, variant: t.steps16, meta: t.steps16Meta, seconds: s16.seconds })];
      if (s32) {
        stepTakes.unshift(take({
          audio: s32.raw.audio, text: s32.text, variant: t.steps32, seconds: s32.raw.seconds,
          meta: fmt(t.steps32Meta, { seed: s32.raw.seed }),
          stats: [metric(t.cer, s32.raw.cer, 2, cerClass(s32.raw.cer)), metric(t.sim, s32.raw.speaker_similarity, 3)]
        }));
      }
      stepsList.replaceChildren(row(s16, { category: "low_latency", takes: stepTakes }));
    }
  }

  function showError() {
    var containers = main.querySelectorAll("[data-gallery-container]");
    for (var i = 0; i < containers.length; i++) {
      containers[i].replaceChildren(el("p", { class: "gallery-status", role: "status", text: t.loadError }));
    }
  }

  /* Copy buttons on code blocks ---------------------------------------------------- */

  if (navigator.clipboard) {
    var blocks = document.querySelectorAll(".code pre");
    for (var b = 0; b < blocks.length; b++) {
      (function (pre) {
        var btn = el("button", { class: "copy", type: "button", "aria-label": t.copy }, [icon("copy"), el("span", { text: t.copy })]);
        btn.addEventListener("click", function () {
          navigator.clipboard.writeText(pre.textContent.replace(/\s+#.*$/gm, "").trim()).then(function () {
            btn.replaceChildren(icon("check"), el("span", { text: t.copied }));
            setTimeout(function () { btn.replaceChildren(icon("copy"), el("span", { text: t.copy })); }, 1800);
          });
        });
        pre.parentNode.appendChild(btn);
      })(blocks[b]);
    }
  }

  fetch(galleryUrl)
    .then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    })
    .then(render)
    .catch(function (err) {
      console.error("Antalia gallery:", err);
      showError();
    });
})();
