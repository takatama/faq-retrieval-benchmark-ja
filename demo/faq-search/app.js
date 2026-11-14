(() => {
  "use strict";
  const data = window.FAQ_DEMO_DATA;
  if (!data || data.kind !== "recorded-api-replay") throw new Error("Demo recordings are missing.");
  const $ = selector => document.querySelector(selector);
  const faqs = new Map(data.faqs.map(f => [f.id, f]));
  const methods = { vector: "検索のみ", flash: "Flashで確認", decision: "Decisionで確認" };
  let sample = data.samples[0];
  let method = "decision";
  let controller = null;
  let serial = 0;
  const seconds = ms => (ms / 1000).toFixed(2) + "秒";
  const escape = value => String(value).replace(/[&<>"']/g, char => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[char]));
  const star = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="m6 0 1.4 4.5L12 6 7.4 7.5 6 12 4.6 7.5 0 6l4.6-1.5Z"/></svg>';

  function totalFor(selectedMethod, selectedSample = sample) {
    return selectedSample.embedding_ms + selectedSample.retrieval_ms +
      (selectedMethod === "vector" ? 0 : selectedSample.methods[selectedMethod].api_ms);
  }
  function timingTable() {
    const cells = (name, values) => '<tr><th scope="row">' + name + '</th>' + values.map(x => '<td>' + x + '</td>').join("") + '</tr>';
    const shared = seconds(sample.embedding_ms + sample.retrieval_ms);
    $("#timing-body").innerHTML = cells("ベクトル化・検索", [shared, shared, shared]) +
      cells("候補の並べ直し", ["—", seconds(sample.methods.flash.api_ms), seconds(sample.methods.decision.api_ms)]) +
      cells("合計", ["vector","flash","decision"].map(m => seconds(totalFor(m))));
  }
  function setStage(stage, state) {
    document.querySelector('[data-stage="' + stage + '"]').dataset.state = state;
  }
  function completedStages() {
    setStage("embed", "done"); setStage("retrieve", "done");
    setStage("rerank", method === "vector" ? "skipped" : "done");
  }
  function renderResults() {
    const ids = method === "vector" ? sample.candidates : sample.methods[method].ranked;
    $("#results").innerHTML = ids.slice(0, 3).map((id, index) => {
      const faq = faqs.get(id);
      const recommended = index === 0 && method !== "vector";
      return '<details class="faq-card"' + (index === 0 ? ' open' : '') + '><summary>' +
        '<span class="faq-rank" aria-label="' + (index + 1) + '位">' + String(index + 1).padStart(2, "0") + '</span>' +
        '<div class="faq-label">' + (recommended ? '<span class="recommendation">' + star + 'AIのおすすめ</span>' : '') +
        '<h3>' + escape(faq.title) + '</h3></div><span class="faq-arrow" aria-hidden="true">＋</span></summary>' +
        '<div class="faq-answer"><span class="faq-category">' + escape(faq.category) + '</span>' + escape(faq.answer) + '</div></details>';
    }).join("");
    $("#results").setAttribute("aria-busy", "false");
    $("#result-meta").textContent = "記録 " + seconds(totalFor(method));
    $("#results-footnote").textContent = method === "vector" ? "ベクトル検索の上位3件です。回答文は用意してあるFAQの本文です。" :
      "候補10件を" + (method === "decision" ? "Decision" : "Flash") + "で確認。回答文は用意してあるFAQの本文です。";
    completedStages();
  }
  function cancel() {
    serial += 1;
    if (controller) controller.abort();
    controller = null;
    $("#results").setAttribute("aria-busy", "false");
    $(".search-button").removeAttribute("data-running");
  }
  function pending() {
    $("#results").innerHTML = '<div class="empty-state"><svg viewBox="0 0 40 40" aria-hidden="true"><rect x="7" y="6" width="26" height="29" rx="5"/><path d="M14 14h12M14 21h12M14 28h7"/></svg><strong>この質問のFAQを探す</strong></div>';
    $("#result-meta").textContent = "";
    ["embed","retrieve","rerank"].forEach(s => setStage(s, "idle"));
    $("#results-footnote").textContent = "「FAQを探す」で、収録した検索の流れを再生します。";
  }
  function pick(next) {
    cancel();
    sample = next;
    $("#query").value = sample.query;
    document.querySelectorAll(".sample-button").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.id === sample.id)));
    timingTable(); pending();
    $("#search-status").textContent = sample.label + "の質問を選びました。";
  }
  function wait(ms, signal) {
    return new Promise((resolve, reject) => {
      if (signal.aborted) return reject(new DOMException("Aborted", "AbortError"));
      const finish = () => { signal.removeEventListener("abort", abort); resolve(); };
      const timer = setTimeout(finish, ms);
      const abort = () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); };
      signal.addEventListener("abort", abort, { once:true });
    });
  }
  function loading(text) {
    $("#results").innerHTML = '<div class="empty-state"><span class="spinner" aria-hidden="true"></span><strong>' + escape(text) + '</strong></div>';
    $("#search-status").textContent = text;
  }
  async function run() {
    cancel();
    const current = serial;
    controller = new AbortController();
    const signal = controller.signal;
    $("#results").setAttribute("aria-busy", "true");
    $(".search-button").setAttribute("data-running", "true");
    $("#result-meta").textContent = "記録を再生中";
    $("#results-footnote").textContent = "API通信は行わず、収録時の待ち時間と結果を再生しています。";
    ["embed","retrieve","rerank"].forEach(s => setStage(s, "idle"));
    try {
      setStage("embed", "active"); loading("質問を、検索できる形に変換しています");
      await wait(sample.embedding_ms, signal);
      if (current !== serial) return;
      setStage("embed", "done"); setStage("retrieve", "active"); loading("似ているFAQを10件探しています");
      await wait(Math.max(sample.retrieval_ms, 1), signal);
      if (current !== serial) return;
      setStage("retrieve", "done");
      if (method !== "vector") {
        setStage("rerank", "active"); loading(methods[method] + "しています");
        await wait(sample.methods[method].api_ms, signal);
        if (current !== serial) return;
      }
      renderResults();
      $("#search-status").textContent = "FAQ候補を3件表示しました。" + (method === "vector" ? "" : methods[method] + "しました。");
    } catch (error) {
      if (error.name !== "AbortError") {
        loading("再生できませんでした。もう一度お試しください。");
        console.error(error);
      }
    } finally {
      if (current === serial) {
        $("#results").setAttribute("aria-busy", "false");
        $(".search-button").removeAttribute("data-running");
        controller = null;
      }
    }
  }

  $("#sample-list").innerHTML = data.samples.map(s => '<button type="button" class="sample-button" data-id="' + escape(s.id) + '" aria-pressed="' + (s.id === sample.id) + '">' + escape(s.label) + '</button>').join("");
  $("#sample-list").addEventListener("click", event => {
    const button = event.target.closest("button[data-id]");
    if (button) pick(data.samples.find(s => s.id === button.dataset.id));
  });
  $("#search-form").addEventListener("submit", event => { event.preventDefault(); run(); });
  document.querySelectorAll('input[name="method"]').forEach(input => input.addEventListener("change", () => {
    cancel(); method = input.value; pending();
    $("#search-status").textContent = methods[method] + "を選びました。FAQを探すボタンで再生できます。";
  }));
  $("#about-button").addEventListener("click", () => $("#about-dialog").showModal());
  $("#close-about").addEventListener("click", () => $("#about-dialog").close());
  $("#recorded-at").textContent = new Intl.DateTimeFormat("ja-JP", { year:"numeric", month:"long", day:"numeric", timeZone:"Asia/Tokyo" }).format(new Date(data.recorded_at));
  $("#codepen-button").addEventListener("click", () => {
    if (!window.FAQ_CODEPEN_PAYLOAD) return;
    const form = document.createElement("form");
    form.method = "POST"; form.action = "https://codepen.io/pen/define/"; form.target = "_blank";
    const input = document.createElement("input");
    input.type = "hidden"; input.name = "data"; input.value = JSON.stringify(window.FAQ_CODEPEN_PAYLOAD);
    form.append(input); document.body.append(form); form.submit(); form.remove();
    $("#codepen-status").textContent = "CodePenの編集画面を開きます。公開・保存はCodePen側で行えます。";
  });
  if (!window.FAQ_CODEPEN_PAYLOAD) $("#codepen-button").hidden = true;
  window.addEventListener("pagehide", cancel);
  $("#query").value = sample.query; timingTable(); renderResults();
  document.documentElement.dataset.ready = "true";
})();
