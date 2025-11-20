// ---------------------------
// Satellite Dashboard Enhanced
// ---------------------------

document.addEventListener("DOMContentLoaded", () => {

  // ✅ Random helper functions
  function randFloat(min, max, fixed = 2) {
    return +(Math.random() * (max - min) + min).toFixed(fixed);
  }
  function randChoice(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  // ✅ HTML elements
  const tempEl = document.getElementById("temp");
  const batteryEl = document.getElementById("batteryV");
  const solarEl = document.getElementById("solar");
  const radEl = document.getElementById("radiation");
  const commEl = document.getElementById("comm");
  const powerPctEl = document.getElementById("powerPct");
  const altitudeEl = document.getElementById("altitude");
  const latEl = document.getElementById("latitude");
  const lonEl = document.getElementById("longitude");
  const speedEl = document.getElementById("speed");
  const orientEl = document.getElementById("orient");
  const batteryDegEl = document.getElementById("batteryDeg");
  const impactEl = document.getElementById("impact");
  const anomalyListEl = document.getElementById("anomalyList");
  const debrisEl = document.getElementById("debrisStatus");

  // ✅ Check Chart.js availability
  if (typeof Chart === "undefined") {
    console.error("❌ Chart.js is not loaded! Please include it in your HTML before this script.");
    return;
  }

  // ✅ Telemetry Chart setup
  const ctx = document.getElementById("telemetryChart").getContext("2d");
  const telemetryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: Array(20).fill(""),
      datasets: [
        {
          label: "Battery (V)",
          data: Array(20).fill(3.75),
          borderColor: "#00eaff",
          backgroundColor: "rgba(0,234,255,0.2)",
          tension: 0.3,
        },
        {
          label: "Temperature (°C)",
          data: Array(20).fill(23.36),
          borderColor: "#ffaa00",
          backgroundColor: "rgba(255,170,0,0.2)",
          tension: 0.3,
        },
        {
          label: "Solar Power (W)",
          data: Array(20).fill(109.58),
          borderColor: "#adff2f",
          backgroundColor: "rgba(173,255,47,0.2)",
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 0 },
      scales: {
        x: { display: false },
        y: { grid: { color: "rgba(255,255,255,0.1)" }, ticks: { color: "#ccc" } },
      },
      plugins: {
        legend: { labels: { color: "#fff", font: { size: 12 } } },
      },
    },
  });

  // ✅ Core parameters update (10s cadence)
  function updateCore() {
    const temp = randFloat(18.5, 26.5);
    const batt = randFloat(3.4, 4.05);
    const solar = randFloat(50, 220);
    const rad = Math.random() < 0.08 ? "--" : randFloat(0.01, 12.5);
    const comm = randChoice(["Stable", "Stable", "Fluctuating", "Degraded"]);

    tempEl.textContent = temp;
    batteryEl.textContent = batt;
    solarEl.textContent = solar;
    radEl.textContent = rad;
    commEl.textContent = comm;
    powerPctEl.textContent = `${Math.round(((batt - 3.3) / (4.2 - 3.3)) * 100)}% Battery`;

    // Update chart data
    const datasets = telemetryChart.data.datasets;
    datasets[0].data.push(batt); datasets[0].data.shift();
    datasets[1].data.push(temp); datasets[1].data.shift();
    datasets[2].data.push(solar); datasets[2].data.shift();
    telemetryChart.update();
  }

  updateCore();
  setInterval(updateCore, 10000);

  // ✅ Orbit / motion update (every 10s)
  function updateOrbit() {
    const alt = randFloat(350, 420, 2);
    const lat = randFloat(-85, 85, 5);
    const lon = randFloat(-180, 180, 5);
    const speed = randFloat(7.5, 7.8, 3);
    const orient = randFloat(0, 360, 1);

    altitudeEl.textContent = alt;
    latEl.textContent = lat;
    lonEl.textContent = lon;
    speedEl.textContent = speed;
    orientEl.textContent = orient;
    batteryDegEl.textContent = `${Math.round(randFloat(1200, 3000, 0))}`;
    impactEl.textContent = randChoice(["Nominal", "Minor", "Moderate", "High"]);
  }

  updateOrbit();
  setInterval(updateOrbit, 10000);

  // ✅ Orbit Canvas Visualization
  const orbitCanvas = document.getElementById("orbitCanvas");
  const octx = orbitCanvas.getContext("2d");
  let orbitAngle = 0;

  function resizeOrbit() {
    orbitCanvas.width = orbitCanvas.clientWidth * devicePixelRatio;
    orbitCanvas.height = orbitCanvas.clientHeight * devicePixelRatio;
    octx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
  window.addEventListener("resize", resizeOrbit);
  resizeOrbit();

  function drawOrbit(lat = 0, lon = 0, alt = 400) {
    const w = orbitCanvas.clientWidth;
    const h = orbitCanvas.clientHeight;
    octx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const rx = w * 0.38, ry = h * 0.28;

    // Orbit ellipse
    octx.beginPath();
    octx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
    octx.strokeStyle = "rgba(159,234,255,0.15)";
    octx.lineWidth = 1.8;
    octx.stroke();

    // Earth
    octx.beginPath();
    octx.arc(cx, cy, 12, 0, 2 * Math.PI);
    octx.fillStyle = "rgba(20,100,200,0.95)";
    octx.fill();

    // Satellite
    const satX = cx + rx * Math.cos(orbitAngle);
    const satY = cy + ry * Math.sin(orbitAngle);
    octx.beginPath();
    octx.arc(satX, satY, 6, 0, 2 * Math.PI);
    octx.fillStyle = "#fff";
    octx.fill();

    // Altitude ring
    const altFactor = Math.max(0.35, Math.min(1.0, (420 - Math.min(420, alt)) / 120 + 0.35));
    octx.beginPath();
    octx.arc(satX, satY, 10 + altFactor * 8, 0, 2 * Math.PI);
    octx.strokeStyle = `rgba(255,160,90,${0.28 + 0.4 * (1 - altFactor)})`;
    octx.lineWidth = 2;
    octx.stroke();

    // Info label
    octx.font = "11px monospace";
    octx.fillStyle = "#dbeff6";
    octx.fillText(`Lat:${lat.toFixed(2)} Lon:${lon.toFixed(2)}`, Math.min(w - 160, satX + 12), Math.max(14, satY - 6));
  }

  function animateOrbit() {
    orbitAngle += 0.001;
    if (orbitAngle > 2 * Math.PI) orbitAngle -= 2 * Math.PI;
    const lat = Math.sin(orbitAngle) * 45;
    const lon = Math.cos(orbitAngle) * 180;
    const alt = 380 + Math.sin(orbitAngle * 2) * 20;
    drawOrbit(lat, lon, alt);
    requestAnimationFrame(animateOrbit);
  }
  animateOrbit();

  // ✅ Anomalies & Debris
  function updateAnomalies() {
    anomalyListEl.innerHTML = "";
    const anomalyChance = Math.random();
    if (anomalyChance < 0.12) {
      const anomalies = [
        `Low battery voltage: ${randFloat(3.30, 3.55)} V`,
        `Thermal spike: ${randFloat(55, 95)} °C`,
        `Comm packet loss spike: ${randFloat(2, 12)}%`,
        `Sensor drift detected`,
        `Unexpected attitude oscillation detected`,
      ];
      const pickCount = anomalyChance < 0.04 ? 2 : 1;
      for (let i = 0; i < pickCount; i++) {
        const li = document.createElement("li");
        li.textContent = anomalies[Math.floor(Math.random() * anomalies.length)];
        anomalyListEl.appendChild(li);
      }
      document.querySelector("#anomalies > p")?.remove();
    } else {
      if (!document.querySelector("#anomalies > p")) {
        const p = document.createElement("p");
        p.textContent = "No anomalies detected.";
        document.getElementById("anomalies").prepend(p);
      }
    }

    const debris = Math.random() < 0.08;
    if (debris) {
      debrisEl.textContent = "⚠️ Potential object detected — Closest approach ~ 4.2 km";
      debrisEl.style.color = "#ffd6a5";
    } else {
      debrisEl.textContent = "✅ No debris detected nearby";
      debrisEl.style.color = "#a0ffa0";
    }
  }

  updateAnomalies();
  setInterval(updateAnomalies, 10000);

});
