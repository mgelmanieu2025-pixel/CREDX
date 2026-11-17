/**
 * Vercel serverless function: GET /api/predict
 * Forwards the 3 inputs to our Flask ML model hosted on PythonAnywhere
 * and returns clean JSON to the CredX front-end.
 * (A server-side proxy avoids browser CORS blocks and keeps the API URL in one place.)
 */
const ML_API = "https://mirandagelman.pythonanywhere.com/predict";
const PARAMS = ["monthly_savings_consistency", "rent_and_bill_payment_history", "spending_to_income_ratio"];

module.exports = async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const q = req.query || {};
  const params = new URLSearchParams();
  for (const p of PARAMS) {
    const v = Number(q[p]);
    if (!Number.isFinite(v) || v < 0 || v > 1) {
      return res.status(400).json({ error: `Parameter '${p}' must be a number between 0 and 1` });
    }
    params.set(p, v.toFixed(2));
  }
  const url = `${ML_API}?${params}`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 9000);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    const text = (await r.text()).trim();
    if (!r.ok) throw new Error(`ML API responded ${r.status}`);
    const m = text.match(/(\d*\.?\d+)\s*$/) || text.match(/(\d*\.\d+)/);
    if (!m) throw new Error("Could not read a probability from the ML API response");
    let probability = parseFloat(m[1]);
    if (probability > 1) probability = probability / 100; // in case it is returned as a %
    return res.status(200).json({ probability, inputs: Object.fromEntries(params), source: url, raw: text });
  } catch (e) {
    return res.status(502).json({ error: e.name === "AbortError" ? "ML API timed out" : e.message, source: url });
  } finally {
    clearTimeout(timer);
  }
};
