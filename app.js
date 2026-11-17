/* CredX MVP – alternative credit scoring for young people
 * User journey: choose profile → answer 3 short steps → get ML-based score,
 * default-risk assessment, alerts, action plan and what-if simulator.
 * The model (logistic regression) is trained in model/train_model.py and
 * loaded from model.json, then run entirely in the browser.
 */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const eur = n => "€" + Math.round(n).toLocaleString("en-US");

  // ---------- Input definitions ----------
  const FIELDS = {
    rent_ontime_pct:       { min: 0, max: 100, step: 1,   fmt: v => v + "%", hint: "12/12 months on time = 100%" },
    rent_history_months:   { min: 0, max: 72,  step: 1,   fmt: v => v >= 12 ? (v / 12).toFixed(1).replace(".0", "") + " yrs" : v + " mo", hint: "Living with parents? Leave it low. We still score you on the other habits." },
    bills_ontime_pct:      { min: 0, max: 100, step: 1,   fmt: v => v + "%" },
    savings_rate_pct:      { min: 0, max: 50,  step: 1,   fmt: v => v + "%" },
    savings_months:        { min: 0, max: 12,  step: 1,   fmt: v => v + "/12 mo", label: "Months you saved (last 12)", hint: "Even €10 counts. Consistency matters more than amount" },
    emergency_fund_months: { min: 0, max: 12,  step: 0.5, fmt: v => v + " mo", hint: "Savings ÷ what you spend in a normal month" },
    discretionary_pct:     { min: 5, max: 80,  step: 1,   fmt: v => v + "%", hint: "Share of your spending that isn't rent, food, transport or bills" },
    income_history_months: { min: 0, max: 72,  step: 1,   fmt: v => v >= 12 ? (v / 12).toFixed(1).replace(".0", "") + " yrs" : v + " mo" },
    income_volatility_pct: { min: 0, max: 100, step: 1,   fmt: v => "±" + v + "%", hint: "Fixed salary ≈ ±5%. Freelance often ±30–60%" },
    platform_rating:       { min: 1, max: 5,   step: 0.1, fmt: v => "★ " + Number(v).toFixed(1), hint: "Upwork, Fiverr, Glovo, Uber, or client testimonials" },
    debt_to_income_pct:    { min: 0, max: 90,  step: 1,   fmt: v => v + "%", hint: "Healthy: under 30%" },
    bnpl_active:           { min: 0, max: 10,  step: 1,   fmt: v => String(v) },
    overdrafts_6m:         { min: 0, max: 10,  step: 1,   fmt: v => String(v) },
  };

  const PROFILE_LABEL = { student: "Student", professional: "Young professional", freelancer: "Freelancer / gig worker" };
  const DEFAULTS = {
    student:      { savings_months: 6, rent_ontime_pct: 92, rent_history_months: 10, bills_ontime_pct: 90, income: 700,  savings_rate_pct: 6,  emergency_fund_months: 0.5, discretionary_pct: 38, income_history_months: 8,  income_volatility_pct: 25, platform_rating: 4.5, debt_to_income_pct: 15, bnpl_active: 1, overdrafts_6m: 1 },
    professional: { savings_months: 10, rent_ontime_pct: 97, rent_history_months: 24, bills_ontime_pct: 95, income: 1700, savings_rate_pct: 10, emergency_fund_months: 1.5, discretionary_pct: 30, income_history_months: 20, income_volatility_pct: 8,  platform_rating: 4.5, debt_to_income_pct: 15, bnpl_active: 1, overdrafts_6m: 0 },
    freelancer:   { savings_months: 7, rent_ontime_pct: 94, rent_history_months: 18, bills_ontime_pct: 92, income: 1500, savings_rate_pct: 8,  emergency_fund_months: 1,   discretionary_pct: 30, income_history_months: 14, income_volatility_pct: 45, platform_rating: 4.6, debt_to_income_pct: 15, bnpl_active: 1, overdrafts_6m: 1 },
  };
  const SAMPLES = {
    student:      { name: "Sofia", profile: "student",      v: { savings_months: 2, rent_ontime_pct: 83,  rent_history_months: 8,  bills_ontime_pct: 80, income: 650,  savings_rate_pct: 2,  emergency_fund_months: 0,   discretionary_pct: 48, income_history_months: 6,  income_volatility_pct: 30, platform_rating: 4.5, debt_to_income_pct: 24, bnpl_active: 4, overdrafts_6m: 3 } },
    professional: { name: "Leo",   profile: "professional", v: { savings_months: 12, rent_ontime_pct: 100, rent_history_months: 30, bills_ontime_pct: 98, income: 1900, savings_rate_pct: 15, emergency_fund_months: 3,   discretionary_pct: 25, income_history_months: 28, income_volatility_pct: 5,  platform_rating: 4.5, debt_to_income_pct: 12, bnpl_active: 0, overdrafts_6m: 0 } },
    freelancer:   { name: "Maya",  profile: "freelancer",   v: { savings_months: 8, rent_ontime_pct: 95,  rent_history_months: 18, bills_ontime_pct: 92, income: 1600, savings_rate_pct: 10, emergency_fund_months: 1.5, discretionary_pct: 30, income_history_months: 20, income_volatility_pct: 50, platform_rating: 4.8, debt_to_income_pct: 18, bnpl_active: 1, overdrafts_6m: 1 } },
  };
  const NEUTRAL_RATING = 4.5; // value used for non-freelancers in training data

  let MODEL = null;
  const state = { profile: "student", name: "", income: 1200, values: {}, step: 0, result: null };

  // ---------- Model ----------
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  function predict(values) {
    let z = MODEL.intercept;
    const contrib = {};
    for (const f of MODEL.features) {
      const c = f.coef * ((values[f.key] - f.mean) / f.std);
      contrib[f.key] = c; z += c;
    }
    const pd = sigmoid(z);
    return { pd, score: toScore(pd), contrib };
  }
  // Industry-style scorecard scaling: points are linear in log-odds of default.
  // Base 640 at 50/50 odds; every halving of the odds of default adds ~31 points.
  const toScore = pd => {
    const p = Math.min(Math.max(pd, 1e-6), 1 - 1e-6);
    return Math.round(Math.min(850, Math.max(300, 640 - 45 * Math.log(p / (1 - p)))));
  };
  function band(score) {
    if (score >= 740) return { name: "Excellent", tag: "Loan-ready", color: "#22d3a6", mult: 1.0 };
    if (score >= 670) return { name: "Good", tag: "Nearly ready", color: "#7cd35c", mult: 0.6 };
    if (score >= 580) return { name: "Fair", tag: "Building credit", color: "#ffb547", mult: 0.3 };
    return { name: "High risk", tag: "Not ready yet", color: "#ff5c7a", mult: 0 };
  }
  const modelValues = () => {
    const v = { ...state.values };
    if (state.profile !== "freelancer") v.platform_rating = NEUTRAL_RATING;
    return v;
  };

  // ---------- Navigation ----------
  function go(view) {
    if ((view === "form" || view === "results") && !MODEL) return;
    if (view === "results" && !state.result) view = "profile";
    $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + view));
    if (view === "history") renderHistory();
    window.scrollTo({ top: 0, behavior: "smooth" });
    history.replaceState(null, "", "#" + view);
  }
  document.addEventListener("click", e => {
    const t = e.target.closest("[data-go]");
    if (t) { e.preventDefault(); go(t.dataset.go); }
  });

  // ---------- Form ----------
  function buildRange(container, key, value, onInput, idPrefix) {
    const f = FIELDS[key];
    const id = idPrefix + key;
    const label = container.querySelector("label") || Object.assign(document.createElement("label"), { textContent: (MODEL.features.find(x => x.key === key) || f).label });
    label.htmlFor = id;
    container.innerHTML = "";
    container.appendChild(label);
    const row = document.createElement("div"); row.className = "range-row";
    const input = Object.assign(document.createElement("input"), { type: "range", id, min: f.min, max: f.max, step: f.step, value });
    const out = document.createElement("output"); out.htmlFor = id; out.textContent = f.fmt(value);
    input.addEventListener("input", () => { out.textContent = f.fmt(+input.value); onInput(key, +input.value); });
    row.append(input, out); container.appendChild(row);
    if (f.hint && idPrefix === "f-") { const h = document.createElement("div"); h.className = "hint"; h.textContent = f.hint; container.appendChild(h); }
  }

  function startForm(profile, preset) {
    state.profile = profile;
    const src = preset ? preset.v : DEFAULTS[profile];
    state.values = {};
    Object.keys(FIELDS).forEach(k => state.values[k] = src[k]);
    state.income = src.income;
    state.name = preset ? preset.name : "";
    $("#name").value = state.name;
    $("#income").value = state.income;
    $$("#assess .field.range").forEach(el => buildRange(el, el.dataset.key, state.values[el.dataset.key], (k, v) => state.values[k] = v, "f-"));
    $$(".freelance-only").forEach(el => el.hidden = profile !== "freelancer");
    state.step = 0; showStep();
    go("form");
  }

  function showStep() {
    $$("#assess fieldset").forEach((fs, i) => fs.hidden = i !== state.step);
    $$(".steps .step").forEach((s, i) => s.classList.toggle("on", i <= state.step));
    $("#prev").style.visibility = state.step === 0 ? "hidden" : "visible";
    $("#next").textContent = state.step === 2 ? "See my results →" : "Next →";
  }
  $("#prev").addEventListener("click", () => { if (state.step > 0) { state.step--; showStep(); } });
  $("#next").addEventListener("click", () => {
    if (state.step === 1) {
      const inc = parseFloat($("#income").value);
      if (!(inc >= 0) || inc > 20000) { $("#income").focus(); $("#income").setCustomValidity("Enter an amount between 0 and 20,000"); $("#income").reportValidity(); return; }
      $("#income").setCustomValidity("");
    }
    if (state.step < 2) { state.step++; showStep(); window.scrollTo({ top: 0 }); return; }
    state.name = $("#name").value.trim().slice(0, 30);
    state.income = parseFloat($("#income").value) || 0;
    runAssessment(true);
  });

  $$(".profile-card").forEach(b => b.addEventListener("click", () => startForm(b.dataset.profile)));
  $$("[data-sample]").forEach(b => b.addEventListener("click", () => startForm(b.dataset.sample, SAMPLES[b.dataset.sample])));
  $("#demo-btn").addEventListener("click", () => { const s = SAMPLES.student; startForm(s.profile, s); });

  // ---------- Results ----------
  function runAssessment(save) {
    const r = predict(modelValues());
    state.result = r;
    renderResults(r);
    if (save) saveHistory(r);
    go("results");
  }

  function renderResults(r) {
    const b = band(r.score);
    const v = state.values;
    $("#res-profile").textContent = PROFILE_LABEL[state.profile];
    $("#res-title").textContent = state.name ? `${state.name}, here's your credit readiness` : "Your credit readiness";
    $("#score-num").textContent = r.score;
    const fill = $("#g-fill");
    fill.style.stroke = b.color;
    requestAnimationFrame(() => fill.style.strokeDashoffset = 251.3 * (1 - (r.score - 300) / 550));
    const bandEl = $("#band"); bandEl.textContent = `${b.name} · ${b.tag}`;
    bandEl.style.color = b.color; bandEl.style.background = b.color + "22";

    $("#pd").textContent = (r.pd * 100).toFixed(1) + "%";
    $("#pd").style.color = r.pd > 0.35 ? "var(--bad)" : r.pd > 0.15 ? "var(--warn)" : "var(--good)";
    $("#reliab").textContent = Math.round((1 - r.pd) * 100) + "%";
    $("#readiness").textContent = b.tag;
    const limit = Math.min(3000, Math.floor(state.income * b.mult / 50) * 50);
    $("#limit").textContent = limit > 0 ? `up to ${eur(limit)}` : "Wait. Follow the plan first";
    const verdicts = {
      "Excellent": "Your alternative data shows strong repayment behaviour. You would be a good candidate for a credit-builder card or a small loan. Keep paying in full and on time.",
      "Good": "You're close. Fixing one or two of the issues below would move you into the loan-ready range.",
      "Fair": "You show some good habits, but the risk flags below could lead to missed payments. Start small with a secured credit-builder product.",
      "High risk": "Taking on new credit now could lead to default. Work through the action plan first. Most people can move up a band within 3–6 months.",
    };
    $("#verdict").textContent = verdicts[b.name];

    renderAlerts(v);
    renderFactors(r.contrib);
    renderPlan(r.score);
    renderSim();
    renderApi(state.values);
    $("#auc").textContent = MODEL.metrics.auc;
  }

  function renderAlerts(v) {
    const list = [];
    const debtEur = state.income * v.debt_to_income_pct / 100;
    if (v.overdrafts_6m >= 2) list.push(["bad", `You went overdrawn ${v.overdrafts_6m} times in 6 months. Repeated overdrafts are one of the strongest early signs of default. Turn on low-balance alerts at €50.`]);
    else if (v.overdrafts_6m === 1) list.push(["warn", "You had 1 overdraft recently. Keep a small buffer in your current account."]);
    if (v.bnpl_active >= 3) list.push(["bad", `${v.bnpl_active} Buy-Now-Pay-Later plans at once ("BNPL stacking") makes it easy to lose track of repayments. Don't open new plans until these are paid off.`]);
    else if (v.bnpl_active === 2) list.push(["warn", "2 BNPL plans are active. Put their due dates in your calendar."]);
    if (v.debt_to_income_pct > 40) list.push(["bad", `About ${eur(debtEur)}/month (${v.debt_to_income_pct}% of income) goes to debt. Above 40% is a danger zone for missed payments.`]);
    else if (v.debt_to_income_pct > 30) list.push(["warn", `${v.debt_to_income_pct}% of your income goes to debt (≈${eur(debtEur)}/month). Aim for under 30%.`]);
    if (v.rent_ontime_pct < 90) list.push(["bad", "Late rent is the biggest red flag in alternative credit data. Set up a standing order for the day after payday."]);
    if (v.emergency_fund_months < 1) list.push([v.emergency_fund_months < 0.5 ? "bad" : "warn", "You have less than one month of savings. One surprise expense could push a repayment late."]);
    if (state.profile === "freelancer" && v.income_volatility_pct > 35 && v.emergency_fund_months < 3) list.push(["warn", `Your income swings by ±${v.income_volatility_pct}%, so aim for a 3-month buffer. Put 20% of every invoice into a separate "smoothing" account.`]);
    if (v.discretionary_pct > 45) list.push(["warn", `${v.discretionary_pct}% of your spending goes on non-essentials. Try a weekly "fun money" limit.`]);
    if (v.savings_rate_pct < 5) list.push(["warn", "You save less than 5% of your income. Even €20/week automated adds up to over €1,000 a year."]);
    if (state.profile === "student" && v.debt_to_income_pct > 20) list.push(["warn", "Student tip: know when your student loan repayments start and how much they will be, so it isn't a shock after graduation."]);
    if (!list.length) list.push(["ok", "No default-risk flags detected. Your habits look healthy."]);
    const sev = { bad: 0, warn: 1, ok: 2 };
    list.sort((a, b) => sev[a[0]] - sev[b[0]]);
    $("#alerts").innerHTML = list.map(([c, t]) => `<li class="${c === "bad" ? "" : c}">${t}</li>`).join("");
  }

  function renderFactors(contrib) {
    const feats = MODEL.features.filter(f => state.profile === "freelancer" || f.key !== "platform_rating");
    const items = feats.map(f => ({ ...f, c: contrib[f.key], val: FIELDS[f.key].fmt(state.values[f.key]) }));
    const max = Math.max(...items.map(i => Math.abs(i.c)), 0.01);
    const row = (i, color) => `<li><span>${i.label}</span><b>${i.val}</b><div class="bar"><i style="width:${Math.max(6, Math.abs(i.c) / max * 100)}%;background:${color}"></i></div></li>`;
    const pos = items.filter(i => i.c < -0.02).sort((a, b) => a.c - b.c).slice(0, 4);
    const neg = items.filter(i => i.c > 0.02).sort((a, b) => b.c - a.c).slice(0, 4);
    $("#pos").innerHTML = pos.length ? pos.map(i => row(i, "var(--good)")).join("") : "<li class='muted'>Nothing stands out yet. Keep building habits.</li>";
    $("#neg").innerHTML = neg.length ? neg.map(i => row(i, "var(--bad)")).join("") : "<li class='muted'>Nothing is pulling your score down. 🎉</li>";
  }

  function actions() {
    const v = state.values, inc = state.income;
    const monthlyExp = Math.max(inc * (1 - v.savings_rate_pct / 100), 1);
    const a = [
      { key: "overdrafts_6m", to: 0, title: "Stop overdrafts", desc: "Turn on low-balance notifications and keep a €100 buffer in your current account." },
      { key: "rent_ontime_pct", to: 100, title: "Automate rent", desc: "Set up a standing order for the day after payday so rent is never late." },
      { key: "bills_ontime_pct", to: 100, title: "Put bills on autopay", desc: "Move phone, internet and subscriptions to direct debit, and cancel the ones you don't use." },
      { key: "bnpl_active", to: Math.max(0, v.bnpl_active - 2), title: "Clear BNPL plans", desc: "Pay off your smallest Buy-Now-Pay-Later plans first, and pause new ones." },
      { key: "debt_to_income_pct", to: Math.max(0, v.debt_to_income_pct - 10), title: "Reduce debt load", desc: `Pay down about ${eur(inc * 0.1)}/month of debt (avalanche method: highest interest first).` },
      { key: "savings_rate_pct", to: Math.min(50, v.savings_rate_pct + 5), title: "Save 5% more", desc: `Automate a transfer of ${eur(inc * 0.05)} on payday into a separate savings pot.` },
      { key: "emergency_fund_months", to: Math.min(12, v.emergency_fund_months + 1), title: "Add 1 month to your emergency fund", desc: `Build up about ${eur(monthlyExp)} more in an easy-access account.` },
      { key: "discretionary_pct", to: Math.max(10, v.discretionary_pct - 10), title: "Cut non-essentials by 10 points", desc: "Use a weekly fun-money budget, unsubscribe from shopping emails, and cook twice more a week." },
    ];
    if (state.profile === "freelancer") {
      a.push({ key: "income_volatility_pct", to: Math.max(5, v.income_volatility_pct - 15), title: "Smooth your income", desc: "Look for one retainer client, or pay yourself a fixed monthly 'salary' from a business account." });
      a.push({ key: "platform_rating", to: Math.min(5, +(v.platform_rating + 0.3).toFixed(1)), title: "Strengthen your reputation", desc: "Ask your last 5 clients for reviews and link your platform profiles to show verified income." });
    }
    return a;
  }

  function renderPlan(baseScore) {
    const items = actions().map(a => {
      const sim = { ...modelValues(), [a.key]: a.to };
      return { ...a, gain: predict(sim).score - baseScore };
    }).filter(a => a.gain >= 2).sort((x, y) => y.gain - x.gain).slice(0, 5);
    $("#plan").innerHTML = items.length
      ? items.map(a => `<li><b>${a.title}</b><b class="gain">+${a.gain} pts</b><p>${a.desc}</p></li>`).join("")
      : "<li><b>Maintain your habits</b><p>You're already near the top of the scale. Keep paying on time, and consider a credit-builder card to start a traditional credit history.</p></li>";
  }

  // ---------- What-if simulator ----------
  let simValues = {};
  function renderSim() {
    simValues = { ...state.values };
    const box = $("#sim"); box.innerHTML = "";
    [...MODEL.features, { key: "savings_months" }].forEach(f => {
      if (f.key === "platform_rating" && state.profile !== "freelancer") return;
      const d = document.createElement("div"); d.className = "field";
      box.appendChild(d);
      buildRange(d, f.key, simValues[f.key], (k, val) => { simValues[k] = val; updateSim(); }, "s-");
    });
    updateSim();
  }
  function updateSim() {
    const v = { ...simValues };
    if (state.profile !== "freelancer") v.platform_rating = NEUTRAL_RATING;
    const s = predict(v).score, d = s - state.result.score;
    $("#sim-score").textContent = s;
    const el = $("#sim-delta");
    el.textContent = d === 0 ? "" : (d > 0 ? `(+${d})` : `(${d})`);
    el.className = d > 0 ? "up" : "down";
    clearTimeout(simApiTimer);
    $("#sim-api").textContent = "…";
    const req = ++simApiReq;
    simApiTimer = setTimeout(() => callApi(simValues).then(r => {
      if (req === simApiReq) $("#sim-api").textContent = Math.round(r.probability * 100) + "%";
    }).catch(() => { if (req === simApiReq) $("#sim-api").textContent = "n/a"; }), 500);
  }
  let simApiTimer = null, simApiReq = 0;
  $("#sim-reset").addEventListener("click", renderSim);

  // ---------- External ML API (Flask on PythonAnywhere) ----------
  const ML_API_DIRECT = "https://mirandagelman.pythonanywhere.com/predict";
  const r2 = x => Math.round(Math.min(1, Math.max(0, x)) * 100) / 100;
  // Map the user's answers to the 3 features our Python model was trained on (all 0–1)
  function apiInputs(v) {
    return {
      monthly_savings_consistency: r2(v.savings_months / 12),
      rent_and_bill_payment_history: r2((v.rent_ontime_pct + v.bills_ontime_pct) / 200),
      spending_to_income_ratio: r2(1 - v.savings_rate_pct / 100),
    };
  }
  const qs = inp => new URLSearchParams(Object.entries(inp).map(([k, x]) => [k, x.toFixed(2)])).toString();
  async function callApi(values) {
    const inp = apiInputs(values), q = qs(inp);
    // 1) our Vercel proxy (/api/predict). 2) If that's unavailable (e.g. local preview), call the API directly.
    try {
      const r = await fetch("api/predict?" + q, { cache: "no-store" });
      const j = await r.json();
      if (r.ok && typeof j.probability === "number") return { probability: j.probability, inputs: inp, url: ML_API_DIRECT + "?" + q };
    } catch { /* proxy not available: fall through */ }
    const r = await fetch(ML_API_DIRECT + "?" + q);
    const t = await r.text();
    const m = t.match(/(\d*\.?\d+)\s*$/);
    if (!r.ok || !m) throw new Error("ML API unavailable");
    return { probability: parseFloat(m[1]), inputs: inp, url: ML_API_DIRECT + "?" + q };
  }
  let apiReq = 0;
  async function renderApi(values) {
    const req = ++apiReq, inp = apiInputs(values);
    $("#ai-msc").textContent = inp.monthly_savings_consistency.toFixed(2);
    $("#ai-rbp").textContent = inp.rent_and_bill_payment_history.toFixed(2);
    $("#ai-sir").textContent = inp.spending_to_income_ratio.toFixed(2);
    $("#api-link").href = ML_API_DIRECT + "?" + qs(inp);
    const badge = $("#api-badge"), prob = $("#api-prob"), dec = $("#api-decision");
    badge.textContent = "Connecting…"; badge.className = "api-badge"; prob.textContent = "…"; prob.style.color = ""; dec.textContent = "Asking the model…";
    try {
      const r = await callApi(values);
      if (req !== apiReq) return;
      const p = r.probability;
      badge.textContent = "● Live · PythonAnywhere"; badge.className = "api-badge live";
      prob.textContent = Math.round(p * 100) + "%";
      prob.style.color = p >= 0.75 ? "var(--good)" : p >= 0.5 ? "var(--warn)" : "var(--bad)";
      dec.textContent = p >= 0.75 ? "✅ Likely to repay: a good candidate for a small loan or credit-builder product."
        : p >= 0.5 ? "🟡 Borderline: improve one or two habits before you borrow."
        : "🔴 High chance of missed repayments: borrowing is not recommended yet.";
      state.apiProb = p;
    } catch (e) {
      if (req !== apiReq) return;
      badge.textContent = "Offline"; badge.className = "api-badge err";
      prob.textContent = "–";
      dec.textContent = "The ML API isn't responding right now. The PythonAnywhere app may be asleep or disabled. Try again in a minute.";
    }
  }

  // ---------- History & report ----------
  const KEY = "credx_history_v1";
  const loadHist = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
  function saveHistory(r) {
    try {
      const h = loadHist();
      h.unshift({ at: new Date().toISOString(), name: state.name, profile: state.profile, score: r.score, pd: r.pd, income: state.income, values: { ...state.values } });
      localStorage.setItem(KEY, JSON.stringify(h.slice(0, 20)));
    } catch { /* storage unavailable: ignore */ }
  }
  function renderHistory() {
    const h = loadHist(), box = $("#history-list");
    if (!h.length) { box.innerHTML = `<p class="muted">No checks yet. <button class="btn small" data-go="profile">Run your first check</button></p>`; return; }
    box.innerHTML = h.map((e, i) => {
      const b = band(e.score);
      return `<div class="hist"><div><b style="color:${b.color}">${e.score}</b> <span class="muted">· ${b.name}</span><div class="small muted">${e.name ? e.name + " · " : ""}${PROFILE_LABEL[e.profile]} · ${new Date(e.at).toLocaleString()}</div></div><button class="btn ghost small" data-open="${i}">Open</button></div>`;
    }).join("");
    $$("[data-open]", box).forEach(btn => btn.addEventListener("click", () => {
      const e = h[+btn.dataset.open];
      Object.assign(state, { profile: e.profile, name: e.name, income: e.income, values: { ...e.values } });
      runAssessment(false);
    }));
  }
  $("#download").addEventListener("click", () => window.print());

  // ---------- Boot ----------
  fetch("model.json")
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(m => {
      MODEL = m;
      $("#model-ver").textContent = `Model v${m.version} · AUC ${m.metrics.auc}`;
      const target = location.hash.slice(1);
      if (["profile", "history"].includes(target)) go(target);
    })
    .catch(() => {
      document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="card" role="alert">⚠️ The scoring model couldn't load. Please refresh the page.</div>`);
    });

  // expose for tests
  window.CredX = { predict, band, toScore, get model() { return MODEL; } };
})();
