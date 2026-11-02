(() => {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), fileInput = $("file"), stage = $("stage"), img = $("img"),
        busy = $("busy"), views = $("views"), go = $("go"), reset = $("reset"),
        errorBox = $("error"), empty = $("empty"), result = $("result"), statusEl = $("status");
  const MAX_BYTES = 10 * 1024 * 1024;
  let file = null, originalURL = "", heatmapURL = "";

  // ---- server status ----
  fetch(`${API_URL}/api/health`)
    .then((r) => r.json())
    .then((d) => {
      statusEl.className = "status " + (d.model_ready ? "up" : "down");
      statusEl.textContent = d.model_ready ? "Server ready" : "Server running, model not trained yet";
    })
    .catch(() => {
      statusEl.className = "status down";
      statusEl.textContent = "Server not reachable";
    });

  // ---- choosing a file ----
  function showError(msg) { errorBox.textContent = msg; errorBox.hidden = !msg; }

  function setFile(f) {
    showError("");
    if (!f) return;
    if (!/\.(png|jpe?g|bmp)$/i.test(f.name)) return showError("Unsupported file type. Use PNG, JPG or BMP.");
    if (f.size > MAX_BYTES) return showError("File is larger than 10 MB. Choose a smaller image.");
    file = f;
    originalURL = URL.createObjectURL(f);
    heatmapURL = "";
    img.src = originalURL;
    drop.hidden = true; stage.hidden = false; views.hidden = false;
    setView("original");
    views.querySelector('[data-view="heatmap"]').disabled = true;
    result.hidden = true; empty.hidden = false;
    go.disabled = false; reset.hidden = false;
  }

  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // ---- original / heat-map toggle ----
  function setView(v) {
    views.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
    img.src = v === "heatmap" && heatmapURL ? heatmapURL : originalURL;
  }
  views.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b && !b.disabled) setView(b.dataset.view);
  });

  // ---- analyse ----
  go.addEventListener("click", async () => {
    if (!file) return;
    showError(""); busy.hidden = false; go.disabled = true;
    const form = new FormData();
    form.append("image", file);
    try {
      const res = await fetch(`${API_URL}/api/predict`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "The server could not analyse this image.");
      render(data);
    } catch (err) {
      showError(err instanceof TypeError
        ? "Cannot reach the server. Start the backend (python app.py) and try again."
        : err.message);
    } finally {
      busy.hidden = true; go.disabled = false;
    }
  });

  function render(d) {
    const bad = d.label === "Abnormal";
    const pct = (x) => (x * 100).toFixed(1) + " %";
    $("verdict").textContent = d.label;
    $("verdict").className = "verdict " + (bad ? "bad" : "ok");
    $("bar").className = bad ? "bad" : "";
    $("bar").style.width = (d.abnormal_probability * 100).toFixed(1) + "%";
    $("prob").textContent = pct(d.abnormal_probability);
    $("conf").textContent = pct(d.confidence);
    $("thr").textContent = d.threshold.toFixed(2);
    $("model").textContent = d.model;
    $("ms").textContent = d.inference_ms + " ms";
    heatmapURL = d.heatmap;
    views.querySelector('[data-view="heatmap"]').disabled = false;
    setView("heatmap");
    empty.hidden = true; result.hidden = false;
  }

  // ---- start over ----
  reset.addEventListener("click", () => {
    file = null; fileInput.value = ""; img.removeAttribute("src");
    stage.hidden = true; views.hidden = true; drop.hidden = false;
    result.hidden = true; empty.hidden = false; showError("");
    go.disabled = true; reset.hidden = true;
  });
})();
