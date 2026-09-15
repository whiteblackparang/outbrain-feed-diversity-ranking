async function loadJSON(path) {
  const res = await fetch(path);
  return res.json();
}

function renderChart(data) {
  const ctx = document.getElementById("tradeoffChart");

  const gridColor = "rgba(139, 147, 167, 0.15)";
  const textColor = "#8b93a7";

  new Chart(ctx, {
    type: "line",
    data: {
      labels: data.map((d) => `\u03bb=${d.lambda}`),
      datasets: [
        {
          label: "Hit rate",
          data: data.map((d) => d.hit_rate),
          borderColor: "#5fc9bd",
          backgroundColor: "#5fc9bd",
          yAxisID: "y",
          tension: 0.25,
          pointRadius: 4,
        },
        {
          label: "Diversity (ILD)",
          data: data.map((d) => d.diversity),
          borderColor: "#d9a256",
          backgroundColor: "#d9a256",
          yAxisID: "y1",
          tension: 0.25,
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#e9e6dd", font: { family: "IBM Plex Sans KR" } },
        },
      },
      scales: {
        x: {
          ticks: { color: textColor },
          grid: { color: gridColor },
        },
        y: {
          position: "left",
          title: { display: true, text: "hit rate", color: textColor },
          ticks: { color: textColor },
          grid: { color: gridColor },
        },
        y1: {
          position: "right",
          title: { display: true, text: "diversity (ILD)", color: textColor },
          ticks: { color: textColor },
          grid: { display: false },
        },
      },
    },
  });
}

function renderExplanations(samples) {
  const list = document.getElementById("explanationList");

  samples.forEach((s) => {
    const card = document.createElement("div");
    card.className = "explanation-card";

    const meta = document.createElement("p");
    meta.className = "explanation-meta";
    meta.textContent = `세션 #${s.display_id} · 후보 ${s.candidate_count}개 · ILD ${s.before_ild.toFixed(3)} → ${s.after_ild.toFixed(3)}`;

    const text = document.createElement("p");
    text.className = "explanation-text";
    text.textContent = s.explanation;

    card.appendChild(meta);
    card.appendChild(text);
    list.appendChild(card);
  });
}

async function init() {
  const [tradeoff, explanations] = await Promise.all([
    loadJSON("data/diversity_tradeoff.json"),
    loadJSON("data/explanation_samples.json"),
  ]);

  renderChart(tradeoff);
  renderExplanations(explanations);
}

init();