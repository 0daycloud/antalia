/* Antalia 1 showcase — renders gallery.json into the page. No dependencies. */
(function () {
  "use strict";

  var STRINGS = {
    en: {
      raw: "Single seed (raw)",
      selected: "Best of 8 (selected)",
      seed: "seed {seed}",
      candidate: "{candidate}",
      cer: "CER {value}",
      sim: "Sim {value}",
      download: "Download WAV",
      downloadLabel: "Download {variant} audio for “{text}”",
      audioLabel: "{variant}: {text}",
      recovered:
        "Single seed: CER {raw}. Best of 8 recovered it: CER {selected}. Across the 8 seeds, CER ranged {min}–{max}.",
      notRecovered:
        "Not recovered by selection: CER {raw} single seed, {selected} selected. Across the 8 seeds, CER ranged {min}–{max}.",
      lowerSeedLost: " A lower-CER seed existed but lost on the combined selection score.",
      unchunked: "Unchunked",
      unchunkedMeta: "9.1 s · 370 characters in one pass",
      chunked: "Chunked at ≤120 characters",
      chunkedMeta: "34.8 s · 160 ms pauses · 0.085 s/char floor",
      foundation: "Foundation, unconditioned",
      foundationMeta: "antalia-1-foundation · speaker id 0",
      steps16: "16 Euler steps",
      steps16Meta: "−45.1 % generation time vs 32 steps",
      steps32: "32 Euler steps (release recipe)",
      loadError: "Unable to load gallery.json. Serve this folder over HTTP and reload.",
      categories: {
        acknowledgement: "Acknowledgement",
        adversarial_normalization: "Normalization",
        emotional_style: "Emotional style",
        foreign_abbreviations: "Foreign terms & abbreviations",
        general: "General",
        long_form: "Long form",
        names_places: "Names & places",
        numeric: "Numbers",
        questions_confirmations: "Questions & confirmations",
        voice_agent: "Voice agent",
        foundation: "Foundation model",
        low_latency: "Low latency"
      }
    },
    tr: {
      raw: "Tek tohum (ham)",
      selected: "8’de en iyi (seçilmiş)",
      seed: "tohum {seed}",
      candidate: "{candidate}",
      cer: "CER {value}",
      sim: "Benzerlik {value}",
      download: "WAV indir",
      downloadLabel: "“{text}” için {variant} sesini indir",
      audioLabel: "{variant}: {text}",
      recovered:
        "Tek tohum: CER {raw}. 8’de en iyi seçimi düzeltti: CER {selected}. 8 tohum boyunca CER {min}–{max} aralığındaydı.",
      notRecovered:
        "Seçim düzeltemedi: tek tohumda CER {raw}, seçilende {selected}. 8 tohum boyunca CER {min}–{max} aralığındaydı.",
      lowerSeedLost: " Daha düşük CER’li bir tohum vardı ama birleşik seçim puanında geride kaldı.",
      unchunked: "Parçalanmamış",
      unchunkedMeta: "9,1 s · 370 karakter tek geçişte",
      chunked: "≤120 karakterde parçalanmış",
      chunkedMeta: "34,8 s · 160 ms duraklama · 0,085 s/karakter tabanı",
      foundation: "Temel model, koşulsuz",
      foundationMeta: "antalia-1-foundation · konuşmacı kimliği 0",
      steps16: "16 Euler adımı",
      steps16Meta: "32 adıma göre üretim süresi −%45,1",
      steps32: "32 Euler adımı (yayın tarifi)",
      loadError: "gallery.json yüklenemedi. Bu klasörü HTTP üzerinden sunup sayfayı yenileyin.",
      categories: {
        acknowledgement: "Onaylama",
        adversarial_normalization: "Normalleştirme",
        emotional_style: "Duygusal ton",
        foreign_abbreviations: "Yabancı terimler ve kısaltmalar",
        general: "Genel",
        long_form: "Uzun metin",
        names_places: "İsimler ve yerler",
        numeric: "Sayılar",
        questions_confirmations: "Sorular ve onaylar",
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

  function fmt(template, vars) {
    return template.replace(/\{(\w+)\}/g, function (_, key) {
      return vars[key] == null ? "" : String(vars[key]);
    });
  }

  function num(value, digits) {
    var s = Number(value).toFixed(digits);
    return lang === "tr" ? s.replace(".", ",") : s;
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
        if (children[i]) node.appendChild(children[i]);
      }
    }
    return node;
  }

  function svgIcon(path) {
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    var p = document.createElementNS(ns, "path");
    p.setAttribute("d", path);
    svg.appendChild(p);
    return svg;
  }

  var DOWNLOAD_PATH = "M12 3v12m0 0 4-4m-4 4-4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2";

  function cerChip(cer) {
    var cls = "chip";
    if (cer <= 0.0005) cls += " chip--good";
    else if (cer >= 0.15) cls += " chip--bad";
    return el("span", { class: cls, text: fmt(t.cer, { value: num(cer, 2) }) });
  }

  function simChip(sim) {
    return el("span", { class: "chip", text: fmt(t.sim, { value: num(sim, 3) }) });
  }

  /* A single player block: label, native audio, stat chips, download link. */
  function player(opts) {
    var src = audioBase + opts.audio;
    var label = el("div", { class: "player-label" }, [
      el("span", { text: opts.variant }),
      opts.meta ? el("small", { text: opts.meta }) : null
    ]);
    var audio = el("audio", {
      controls: "",
      preload: "none",
      src: src,
      "aria-label": fmt(t.audioLabel, { variant: opts.variant, text: opts.text })
    });
    var stats = opts.stats && opts.stats.length ? el("div", { class: "player-stats" }, opts.stats) : null;
    var download = el("a", {
      class: "player-download",
      href: src,
      download: opts.audio,
      "aria-label": fmt(t.downloadLabel, { variant: opts.variant, text: opts.text })
    }, [svgIcon(DOWNLOAD_PATH), el("span", { text: t.download })]);
    return el("div", { class: "player" }, [label, audio, stats, download]);
  }

  function card(entry, opts) {
    var headingId = "sample-" + entry.id;
    var meta = el("div", { class: "card-meta" }, [
      el("span", { class: "chip chip--category", text: t.categories[opts.category] || opts.category }),
      el("span", { class: "chip chip--id", text: entry.id })
    ]);
    var text = el("p", { class: "card-text", id: headingId, lang: "tr", text: entry.text });
    var children = [meta, text];
    if (opts.note) children.push(el("p", { class: "card-note" }, opts.note));
    children.push(el("div", { class: "players" }, opts.players));
    return el("article", {
      class: "card" + (opts.players.length === 1 ? " card--single" : ""),
      "aria-labelledby": headingId
    }, children);
  }

  function pairCard(entry, note) {
    return card(entry, {
      category: entry.category,
      note: note,
      players: [
        player({
          audio: entry.raw.audio,
          text: entry.text,
          variant: t.raw,
          meta: fmt(t.seed, { seed: entry.raw.seed }),
          stats: [cerChip(entry.raw.cer), simChip(entry.raw.speaker_similarity)]
        }),
        player({
          audio: entry.selected.audio,
          text: entry.text,
          variant: t.selected,
          meta: fmt(t.candidate, { candidate: entry.selected.candidate }),
          stats: [cerChip(entry.selected.cer), simChip(entry.selected.speaker_similarity)]
        })
      ]
    });
  }

  function failureNote(entry) {
    var vars = {
      raw: num(entry.raw.cer, 2),
      selected: num(entry.selected.cer, 2),
      min: num(entry.candidate_cer_min, 2),
      max: num(entry.candidate_cer_max, 2)
    };
    var recovered = entry.selected.cer < entry.raw.cer - 0.0005;
    var textNode = document.createTextNode(fmt(recovered ? t.recovered : t.notRecovered, vars));
    var nodes = [textNode];
    if (!recovered && entry.candidate_cer_min < entry.selected.cer - 0.0005) {
      nodes.push(document.createTextNode(t.lowerSeedLost));
    }
    return nodes;
  }

  function render(data) {
    var showcase = document.getElementById("showcase-grid");
    var failures = document.getElementById("failures-grid");
    var longGrid = document.getElementById("long-grid");
    var foundationGrid = document.getElementById("foundation-grid");
    var stepsGrid = document.getElementById("steps-grid");

    var frag, i;

    if (showcase) {
      frag = document.createDocumentFragment();
      for (i = 0; i < data.showcase.length; i++) frag.appendChild(pairCard(data.showcase[i], null));
      showcase.replaceChildren(frag);
    }

    if (failures) {
      frag = document.createDocumentFragment();
      for (i = 0; i < data.failures.length; i++) {
        frag.appendChild(pairCard(data.failures[i], failureNote(data.failures[i])));
      }
      failures.replaceChildren(frag);
    }

    var extras = {};
    for (i = 0; i < data.extras.length; i++) extras[data.extras[i].id] = data.extras[i];

    if (longGrid && extras["long-unchunked"] && extras["long-chunked"]) {
      var lu = extras["long-unchunked"];
      var lc = extras["long-chunked"];
      longGrid.replaceChildren(card({ id: "long-input", text: lu.text }, {
        category: "long_form",
        players: [
          player({ audio: lu.audio, text: lu.text, variant: t.unchunked, meta: t.unchunkedMeta }),
          player({ audio: lc.audio, text: lc.text, variant: t.chunked, meta: t.chunkedMeta })
        ]
      }));
    }

    if (foundationGrid) {
      frag = document.createDocumentFragment();
      for (i = 0; i < data.extras.length; i++) {
        var f = data.extras[i];
        if (f.role !== "foundation") continue;
        frag.appendChild(card(f, {
          category: "foundation",
          players: [player({ audio: f.audio, text: f.text, variant: t.foundation, meta: t.foundationMeta })]
        }));
      }
      foundationGrid.replaceChildren(frag);
    }

    if (stepsGrid && extras["general-001-16steps"]) {
      var s16 = extras["general-001-16steps"];
      var s32 = null;
      for (i = 0; i < data.showcase.length; i++) {
        if (data.showcase[i].id === "general-001") s32 = data.showcase[i];
      }
      var players = [player({ audio: s16.audio, text: s16.text, variant: t.steps16, meta: t.steps16Meta })];
      if (s32) {
        players.unshift(player({
          audio: s32.raw.audio,
          text: s32.text,
          variant: t.steps32,
          meta: fmt(t.seed, { seed: s32.raw.seed })
        }));
      }
      stepsGrid.replaceChildren(card(s16, { category: "low_latency", players: players }));
    }
  }

  /* Only one audio element plays at a time. */
  document.addEventListener("play", function (event) {
    var target = event.target;
    if (!(target instanceof HTMLMediaElement)) return;
    var all = document.querySelectorAll("audio");
    for (var i = 0; i < all.length; i++) {
      if (all[i] !== target && !all[i].paused) all[i].pause();
    }
  }, true);

  function showError() {
    var containers = main.querySelectorAll("[data-gallery-container]");
    for (var i = 0; i < containers.length; i++) {
      containers[i].replaceChildren(el("p", { class: "gallery-status", role: "status", text: t.loadError }));
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
