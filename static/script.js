// ── Freight Cost Prediction ──────────────────────────────────────────────────
async function predictFreight(event) {
  event.preventDefault();

  const btn = document.getElementById("freight-btn");
  const errEl = document.getElementById("freight-err");
  const resultEl = document.getElementById("freight-result");
  const dollars = parseFloat(document.getElementById("freight-dollars").value);

  errEl.style.display = "none";
  resultEl.style.display = "none";

  if (isNaN(dollars) || dollars < 0) {
    showError(errEl, "Please enter a valid invoice value.");
    return;
  }

  setLoading(btn, true);

  try {
    const response = await fetch("/predict/freight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ Dollars: dollars })
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(errEl, data.error || "Server error. Please try again.");
      return;
    }

    const predicted = data.predictions[0].Predicted_Freight;
    showFreightResult(resultEl, predicted);

  } catch (err) {
    showError(errEl, "Could not reach the server. Is Flask running?");
  } finally {
    setLoading(btn, false);
  }
}

function showFreightResult(el, value) {
  el.className = "result neutral";

  const valueEl = el.querySelector("#freight-value");
  if (valueEl) {
    valueEl.textContent = "$" + Number(value).toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  }

  const iconEl = el.querySelector(".result-icon");
  if (iconEl) {
    iconEl.className = "result-icon neutral-icon";
    iconEl.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <path d="M9 3.5v5.25l3 1.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        <circle cx="9" cy="9" r="6.5" stroke="currentColor" stroke-width="1.4"/>
      </svg>`;
  }

  el.style.display = "block";
}


// ── Invoice Anomaly Detection ────────────────────────────────────────────────
async function predictInvoice(event) {
  event.preventDefault();

  const btn = document.getElementById("invoice-btn");
  const errEl = document.getElementById("invoice-err");
  const resultEl = document.getElementById("invoice-result");

  const invoiceQty = parseFloat(document.getElementById("inv-qty").value);
  const invoiceDollars = parseFloat(document.getElementById("inv-dollars").value);
  const freight = parseFloat(document.getElementById("inv-freight").value);
  const totalQty = parseFloat(document.getElementById("inv-total-qty").value);
  const totalDollars = parseFloat(document.getElementById("inv-total-dollars").value);

  errEl.style.display = "none";
  resultEl.style.display = "none";

  const fields = [invoiceQty, invoiceDollars, freight, totalQty, totalDollars];
  if (fields.some(v => isNaN(v) || v < 0)) {
    showError(errEl, "Please fill in all fields with valid non-negative numbers.");
    return;
  }

  setLoading(btn, true);

  try {
    const response = await fetch("/predict/invoice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        invoice_quantity: invoiceQty,
        invoice_dollars: invoiceDollars,
        Freight: freight,
        total_item_quantity: totalQty,
        total_item_dollars: totalDollars
      })
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(errEl, data.error || "Server error. Please try again.");
      return;
    }

    const flag = data.predictions[0].Predicted_Flag;
    showInvoiceResult(resultEl, flag);

  } catch (err) {
    showError(errEl, "Could not reach the server. Is Flask running?");
  } finally {
    setLoading(btn, false);
  }
}

function showInvoiceResult(el, flag) {
  const valueEl = el.querySelector("#invoice-value");
  const noteEl = el.querySelector("#invoice-note");
  const iconEl = el.querySelector("#invoice-icon");

  if (flag === 0) {
    el.className = "result success";

    if (valueEl) valueEl.textContent = "Clear";
    if (noteEl) noteEl.textContent = "No irregularities detected. This invoice is consistent with established spending patterns.";
    if (iconEl) {
      iconEl.className = "result-icon success-icon";
      iconEl.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <path d="M4.5 9.5l3 3L13.5 6" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="9" cy="9" r="6.5" stroke="currentColor" stroke-width="1.4"/>
        </svg>`;
    }
  } else {
    el.className = "result danger";

    if (valueEl) valueEl.textContent = "Flagged";
    if (noteEl) noteEl.textContent = "Anomaly detected. This invoice deviates significantly from expected patterns and requires manual audit review.";
    if (iconEl) {
      iconEl.className = "result-icon danger-icon";
      iconEl.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <path d="M9 6v4M9 12.5v.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
          <path d="M7.5 2.8L1.5 13.2A1.7 1.7 0 003 15.75h12a1.7 1.7 0 001.5-2.55L10.5 2.8a1.7 1.7 0 00-3 0z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>
        </svg>`;
    }
  }

  el.style.display = "block";
}


// ── Helpers ──────────────────────────────────────────────────────────────────
function setLoading(btn, state) {
  if (state) {
    btn.classList.add("loading");
    btn.disabled = true;
  } else {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
}

function showError(el, message) {
  el.textContent = message;
  el.style.display = "block";
}