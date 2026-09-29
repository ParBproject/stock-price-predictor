(function () {
  "use strict";

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const state = {
    payload: null,
    symbol: null,
    forecast: null,
    backtest: null,
  };

  const statusNode = document.getElementById("status");
  const tickerList = document.getElementById("ticker-list");
  const stats = document.getElementById("stats");

  function setStatus(message, isError) {
    statusNode.textContent = message;
    statusNode.dataset.state = isError ? "error" : "ok";
  }

  function money(value, digits) {
    return Number(value).toLocaleString("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function plain(value, digits) {
    return Number(value).toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function percent(fraction) {
    return Number(fraction).toLocaleString("en-US", {
      style: "percent",
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
  }

  function utcStamp(isoDate) {
    return Date.parse(isoDate + "T00:00:00Z") / 1000;
  }

  function formatTick(stamp) {
    const when = new Date(stamp * 1000);
    const month = MONTHS[when.getUTCMonth()];
    const day = when.getUTCDate();
    return month + " " + day + " " + when.getUTCFullYear();
  }

  function axisValues(self, ticks) {
    return ticks.map(formatTick);
  }

  function chartWidth(node) {
    return Math.max(280, node.clientWidth);
  }

  function chartHeight() {
    return window.innerWidth < 720 ? 260 : 380;
  }

  function baseOptions(node, series) {
    const muted = "#94a3b8";
    const grid = "#1e293b";
    return {
      width: chartWidth(node),
      height: chartHeight(),
      cursor: {
        drag: { x: true, y: false, setScale: true },
      },
      scales: { x: { time: true } },
      axes: [
        {
          stroke: muted,
          grid: { stroke: grid },
          ticks: { stroke: grid },
          values: axisValues,
          font: "12px Inter, sans-serif",
        },
        {
          stroke: muted,
          grid: { stroke: grid },
          ticks: { stroke: grid },
          size: 72,
          font: "12px Inter, sans-serif",
          values: (self, ticks) => ticks.map((tick) => plain(tick, tick >= 100 ? 0 : 2)),
        },
      ],
      series: series,
      legend: { live: false },
    };
  }

  function destroyChart(key) {
    if (state[key]) {
      state[key].destroy();
      state[key] = null;
    }
  }

  function mountChart(key, node, data, series) {
    destroyChart(key);
    node.replaceChildren();
    state[key] = new uPlot(baseOptions(node, series), data, node);
  }

  function resetChart(key) {
    const chart = state[key];
    if (!chart) return;
    const xs = chart.data[0];
    chart.setScale("x", { min: xs[0], max: xs[xs.length - 1] });
  }

  function resizeCharts() {
    ["forecast", "backtest"].forEach((key) => {
      const chart = state[key];
      if (!chart) return;
      const node = chart.root.parentElement;
      chart.setSize({ width: chartWidth(node), height: chartHeight() });
    });
  }

  function tickerBySymbol(symbol) {
    return state.payload.tickers.find((item) => item.symbol === symbol);
  }

  function selectedSymbol() {
    const hash = window.location.hash.replace("#", "").toUpperCase();
    if (hash && tickerBySymbol(hash)) return hash;
    return state.payload.tickers[0].symbol;
  }

  function renderTickers() {
    tickerList.replaceChildren();
    state.payload.tickers.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.role = "tab";
      button.dataset.symbol = item.symbol;
      button.setAttribute("aria-selected", item.symbol === state.symbol ? "true" : "false");
      const name = document.createElement("strong");
      name.textContent = item.symbol;
      const ratio = document.createElement("span");
      ratio.className = "ratio";
      ratio.textContent = plain(item.overall.model.mae_ratio, 2) + "× baseline";
      button.append(name, ratio);
      button.addEventListener("click", () => {
        state.symbol = item.symbol;
        history.replaceState(null, "", "#" + item.symbol);
        render();
      });
      tickerList.append(button);
    });
  }

  function renderStats(item) {
    stats.hidden = false;
    document.getElementById("model-mae").textContent = money(item.overall.model.mae, 2);
    document.getElementById("baseline-mae").textContent = money(item.overall.baseline.mae, 2);
    document.getElementById("mae-ratio").textContent = plain(item.overall.model.mae_ratio, 2) + "×";
  }

  function renderForecast(item) {
    const series = item.series;
    const data = [
      series.dates.map(utcStamp),
      series.actual,
      series.forecast,
      series.baseline,
    ];
    mountChart("forecast", document.getElementById("forecast-chart"), data, [
      {},
      { label: "Actual close", stroke: "#e2e8f0", width: 1.6 },
      { label: "Random Forest", stroke: "#10B981", width: 1.6 },
      { label: "Persistence", stroke: "#f59e0b", width: 1.4, dash: [7, 4] },
    ]);
    const chart = document.getElementById("forecast-chart");
    chart.setAttribute(
      "aria-label",
      item.symbol +
        " next-day close. Random Forest MAE " +
        money(item.overall.model.mae, 2) +
        ", persistence MAE " +
        money(item.overall.baseline.mae, 2) +
        "."
    );
  }

  function renderTable(item) {
    const body = document.getElementById("metrics-body");
    body.replaceChildren();
    document.getElementById("table-caption").textContent =
      item.symbol + " per-fold and overall errors, in dollars. Lower is better. The ratio is forest MAE divided by persistence MAE.";

    function addRow(label, windowText, model, baseline, overall) {
      const row = document.createElement("tr");
      if (overall) row.className = "overall";
      const cells = [
        label,
        windowText,
        money(model.mae, 4),
        money(model.rmse, 4),
        money(baseline.mae, 4),
        money(baseline.rmse, 4),
        plain(model.mae_ratio, 2) + "×",
      ];
      cells.forEach((text, index) => {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = text;
        if (index === cells.length - 1 && model.mae_ratio > 1) cell.className = "ratio-cell";
        row.append(cell);
      });
      body.append(row);
    }

    item.folds.forEach((fold) => {
      addRow(
        "Fold " + fold.fold,
        fold.start + " – " + fold.end,
        fold.model,
        fold.baseline,
        false
      );
    });
    addRow(
      "Overall",
      item.series.dates[0] + " – " + item.series.dates[item.series.dates.length - 1],
      item.overall.model,
      item.overall.baseline,
      true
    );
  }

  function renderBacktest(item) {
    const section = document.getElementById("backtest-section");
    const backtest = item.backtest;
    if (!backtest || !backtest.ok) {
      section.hidden = true;
      destroyChart("backtest");
      return;
    }
    section.hidden = false;
    const note = document.getElementById("backtest-note");
    note.textContent =
      "Last fold only (" +
      backtest.start +
      " to " +
      backtest.end +
      "). Forecast strategy ends at " +
      money(backtest.strategy_final, 2) +
      " (" +
      percent(backtest.strategy_return) +
      "). Buy and hold ends at " +
      money(backtest.buy_hold_final, 2) +
      " (" +
      percent(backtest.buy_hold_return) +
      ").";
    const data = [
      backtest.dates.map(utcStamp),
      backtest.strategy,
      backtest.buy_hold,
    ];
    mountChart("backtest", document.getElementById("backtest-chart"), data, [
      {},
      { label: "Forecast strategy", stroke: "#10B981", width: 1.7 },
      { label: "Buy and hold", stroke: "#7dd3fc", width: 1.5, dash: [6, 4] },
    ]);
    document.getElementById("backtest-chart").setAttribute(
      "aria-label",
      item.symbol +
        " simulated equity. Strategy " +
        money(backtest.strategy_final, 2) +
        ", buy and hold " +
        money(backtest.buy_hold_final, 2) +
        "."
    );
  }

  function renderCopy(item) {
    const ratio = item.overall.model.mae_ratio;
    const everyFoldWorse = item.folds.every((fold) => fold.model.mae > fold.baseline.mae);
    let comparison;
    if (ratio > 1) {
      comparison = "The forest does not beat persistence.";
    } else if (ratio === 1) {
      comparison = "The forest matches persistence on MAE.";
    } else {
      comparison = "On overall MAE the forest is below persistence. The table is the comparison; this page still does not claim a trading edge.";
    }
    if (everyFoldWorse) {
      comparison += " Persistence has the lower MAE in every fold.";
    }
    document.getElementById("shows-lead").textContent =
      "On " +
      item.symbol +
      " the Random Forest MAE is " +
      money(item.overall.model.mae, 2) +
      " and the persistence MAE is " +
      money(item.overall.baseline.mae, 2) +
      ", so the forest error is " +
      plain(ratio, 2) +
      "× the baseline. " +
      comparison;

    function meanClose(fold) {
      let sum = 0;
      let count = 0;
      item.series.dates.forEach((day, index) => {
        if (day >= fold.start && day <= fold.end) {
          sum += item.series.actual[index];
          count += 1;
        }
      });
      return count ? sum / count : 0;
    }

    const worstRatio = item.folds.find((fold) => fold.fold === item.worst_fold);
    const worstDollars = item.folds.reduce((best, fold) =>
      fold.model.mae > best.model.mae ? fold : best
    );
    let foldText =
      "The largest dollar error is fold " +
      worstDollars.fold +
      " (" +
      worstDollars.start +
      " to " +
      worstDollars.end +
      "), MAE " +
      money(worstDollars.model.mae, 2) +
      " versus persistence " +
      money(worstDollars.baseline.mae, 2) +
      ". The average close in that window is " +
      money(meanClose(worstDollars), 2) +
      ".";
    if (worstRatio.fold === worstDollars.fold) {
      foldText +=
        " That is also the largest MAE ratio, " +
        plain(worstRatio.model.mae_ratio, 2) +
        "×.";
    } else {
      foldText +=
        " The largest MAE ratio is fold " +
        worstRatio.fold +
        " (" +
        worstRatio.start +
        " to " +
        worstRatio.end +
        "), " +
        plain(worstRatio.model.mae_ratio, 2) +
        "×. The ratio is higher there because the day-to-day move, which is the whole persistence error, is only " +
        money(worstRatio.baseline.mae, 2) +
        ".";
    }
    document.getElementById("shows-fold").textContent = foldText;

    const study = state.payload.study;
    const embargo = item.folds[0].embargo_rows;
    const items = [
      "Adjusted daily bars from " + study.start + " through " + item.last_bar + ", the same window the notebooks request (" + study.end_exclusive + " is exclusive).",
      "Features on date t predict Close on the next session. The split leaves a gap of " +
        embargo +
        (embargo === 1 ? " row" : " rows") +
        " between training labels and the validation window, so a training target cannot land inside the validation features.",
      study.n_splits + " expanding walk-forward folds. The forest is " + study.model + " with " + study.n_estimators + " trees, max depth " + study.max_depth + ", random state " + study.random_state + ", and no grid search.",
      "Persistence uses the close already known at t as the forecast of the next close. MAE and RMSE come from the repository metric function.",
      "The sentiment column is in the feature list and is neutral, because this build does not call a news API.",
      "The LSTM was not trained.",
    ];
    const list = document.getElementById("method-list");
    list.replaceChildren();
    items.forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      list.append(li);
    });
    document.getElementById("as-of").textContent =
      "Data as of " + state.payload.data_as_of + ". Results generated " + state.payload.generated_on + ".";
  }

  function render() {
    const item = tickerBySymbol(state.symbol);
    renderTickers();
    renderStats(item);
    renderForecast(item);
    renderTable(item);
    renderBacktest(item);
    renderCopy(item);
    setStatus(item.symbol + " · " + item.overall.n.toLocaleString("en-US") + " out-of-sample forecasts.", false);
  }

  document.getElementById("reset-forecast").addEventListener("click", () => resetChart("forecast"));
  document.getElementById("reset-backtest").addEventListener("click", () => resetChart("backtest"));
  window.addEventListener("resize", resizeCharts);

  if (typeof uPlot !== "function") {
    setStatus("The chart library did not load from the local vendor folder.", true);
    return;
  }

  fetch("data/demo.json")
    .then((response) => {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    })
    .then((payload) => {
      if (!payload || !Array.isArray(payload.tickers) || payload.tickers.length === 0) {
        throw new Error("demo.json has no tickers");
      }
      state.payload = payload;
      state.symbol = selectedSymbol();
      render();
    })
    .catch((error) => {
      setStatus("Could not read data/demo.json (" + error.message + ").", true);
    });
})();
