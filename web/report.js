/* Offline report UI. The Python renderer embeds the data and these assets. */
(() => {
  "use strict";

  const DAY = 86400000;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = {
    blue: "#275eed",
    paleBlue: "#9cb9fb",
    green: "#258873",
    muted: "#8894a3",
    line: "#e5eaf0",
    selected: "#edf3ff",
  };
  const $ = (id) => document.getElementById(id);
  const METHOD_LABELS = {
    "Linear Regression": "线性回归",
    "Ridge Regression": "岭回归",
    "Lasso Regression": "Lasso 回归",
    "Exponential Smoothing": "指数平滑",
    "Seasonal Pattern": "季节模式",
    "Mean Interval": "均值间隔",
    "Median Interval": "中位数间隔",
    "Recent 3 Mean": "最近三次均值",
    "Adaptive Interval": "自适应间隔",
    "Weighted Interval": "加权间隔",
    "Trend Analysis": "趋势分析",
    "Statistical Ensemble": "统计集成",
  };
  const GROUP_LABELS = {
    "Linear Models": "线性模型",
    "Time Series": "时间序列",
    "Interval Based": "间隔方法",
    Statistical: "统计方法",
  };
  const methodLabel = (name) => METHOD_LABELS[name] || name;
  const groupLabel = (name) => GROUP_LABELS[name] || name || "未分组";
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const number = (value, decimals = 1) =>
    finite(value)
      ? value.toLocaleString("zh-CN", { maximumFractionDigits: decimals })
      : "—";
  const percent = (value) =>
    finite(value) ? `${number(value * 100, 1)}%` : "—";
  const signed = (value) =>
    finite(value) ? `${value > 0 ? "+" : ""}${number(value)}` : "—";
  const dateValue = (value) => {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value))
      return null;
    const timestamp = Date.parse(`${value}T00:00:00Z`);
    return Number.isFinite(timestamp) &&
      new Date(timestamp).toISOString().slice(0, 10) === value
      ? timestamp
      : null;
  };
  const dateLabel = (value) => (dateValue(value) !== null ? value : "—");
  const isoDate = (timestamp) => new Date(timestamp).toISOString().slice(0, 10);
  const shortDate = (timestamp) =>
    isoDate(timestamp).slice(2).replaceAll("-", "/");
  const text = (id, value) => {
    $(id).textContent = value == null ? "—" : String(value);
  };

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined && content !== null)
      node.textContent = String(content);
    return node;
  }

  function svgElement(tag, attributes = {}, content) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([key, value]) =>
      node.setAttribute(key, String(value)),
    );
    if (content !== undefined && content !== null)
      node.textContent = String(content);
    return node;
  }

  function chart(containerId, width, height, label) {
    const node = svgElement("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "group",
      "aria-label": label,
    });
    node.append(svgElement("title", {}, label));
    $(containerId).replaceChildren(node);
    return node;
  }

  function emptyChart(id, message) {
    $(id).replaceChildren(element("p", "no-data", message));
  }

  function mark(node, description) {
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "img");
    node.setAttribute("aria-label", description);
    node.append(svgElement("title", {}, description));
    return node;
  }

  function cell(row, value, className) {
    const node = element("td", className, value);
    row.append(node);
    return node;
  }

  function emptyTable(id, columns, message) {
    const row = element("tr");
    const node = cell(row, message, "empty-message");
    node.colSpan = columns;
    $(id).replaceChildren(row);
  }

  function option(select, value, label) {
    const node = element("option", null, label);
    node.value = value;
    select.append(node);
  }

  function safeURL(value) {
    if (!value) return null;
    try {
      const url = new URL(value);
      return ["https:", "http:"].includes(url.protocol) ? url.href : null;
    } catch (_) {
      return null;
    }
  }

  let report;
  try {
    report = JSON.parse($("report-data").textContent);
    if (
      report.schema_version !== 1 ||
      !report.meta ||
      !Array.isArray(report.releases) ||
      !Array.isArray(report.forecasts)
    ) {
      throw new Error("报告结构无效，或版本不受支持。");
    }
  } catch (error) {
    text("load-error", `无法加载报告：${error.message}`);
    $("load-error").hidden = false;
    $("download-data").disabled = true;
    text("coverage-label", "数据加载失败");
    return;
  }

  const meta = report.meta;
  const summary = report.summary || {};
  const backtest = report.backtest || {};
  const summaries = Array.isArray(backtest.summaries) ? backtest.summaries : [];
  const records = Array.isArray(backtest.records) ? backtest.records : [];
  const asOf = dateValue(meta.as_of);
  const summaryMap = new Map(summaries.map((item) => [item.method, item]));
  const forecasts = report.forecasts.map((item) => ({
    ...item,
    dates: Array.isArray(item.dates) ? item.dates : [],
  }));
  const forecastMap = new Map(forecasts.map((item) => [item.method, item]));
  const methods = [
    ...new Set([
      ...forecasts.map((item) => item.method),
      ...summaries.map((item) => item.method),
    ]),
  ];
  const completeBacktests = summaries.filter(
    (item) => item.eligible && finite(item.mae),
  );
  const ranked = completeBacktests
    .filter((item) => forecastMap.get(item.method)?.status === "ok")
    .sort((a, b) => a.mae - b.mae || a.method.localeCompare(b.method));
  const rankMap = new Map(
    ranked.map((item, index) => [item.method, index + 1]),
  );
  const state = {
    group: "",
    sort: "mae",
    method: methods.includes(summary.best_method)
      ? summary.best_method
      : ranked[0]?.method || methods[0] || "",
  };
  const included = report.releases.filter(
    (item) => item.included && dateValue(item.date) !== null,
  );
  const eventDates = [...new Set(included.map((item) => item.date))].sort();
  const allEventDates = [
    ...new Set(
      report.releases
        .map((item) => item.date)
        .filter((value) => dateValue(value) !== null),
    ),
  ].sort();
  const gapMap = new Map(
    allEventDates.map((date, index) => [
      date,
      index
        ? (dateValue(date) - dateValue(allEventDates[index - 1])) / DAY
        : null,
    ]),
  );
  const steps =
    Number.isInteger(meta.n_predictions) && meta.n_predictions > 0
      ? meta.n_predictions
      : Math.max(1, ...forecasts.map((item) => item.dates.length));

  function timing(value) {
    const date = dateValue(value);
    if (date === null || asOf === null)
      return { text: "无有效日期", className: "" };
    const days = Math.round((date - asOf) / DAY);
    if (days < 0)
      return { text: `已逾期 ${Math.abs(days)} 天`, className: "date-late" };
    if (days === 0) return { text: "数据截点当日", className: "date-future" };
    return { text: `${days} 天后`, className: "date-future" };
  }

  function renderOverview() {
    text("as-of", dateLabel(meta.as_of));
    text("dataset-label", meta.dataset_name || "DeepSeek");
    text(
      "coverage-label",
      `${number(meta.release_count, 0)} 条纳入记录 · ${number(meta.event_count, 0)} 个日期事件`,
    );
    text("methods-label", `${forecasts.length} 种方法 · ${steps} 步预测`);
    text(
      "metric-next",
      dateValue(summary.median_next_date) !== null
        ? summary.median_next_date.replaceAll("-", ".")
        : "暂无预测",
    );
    text(
      "metric-next-note",
      dateValue(summary.median_next_date) !== null
        ? `${timing(summary.median_next_date).text} · 非置信区间`
        : "尚无可用的首步预测",
    );
    text("metric-elapsed", number(meta.elapsed_days, 0));
    text("metric-last-note", `最近发布 ${dateLabel(meta.last_release)}`);
    text("metric-interval", number(summary.median_interval));
    text(
      "metric-interval-note",
      `平均间隔 ${number(summary.mean_interval)} 天`,
    );
    text("metric-mae", number(summary.best_mae));
    text(
      "metric-best-note",
      methodLabel(summary.best_method) || "尚无可排名的完整回测",
    );
    const rangeStart = dateLabel(summary.earliest_next_date);
    const rangeEnd = dateLabel(summary.latest_next_date);
    text(
      "forecast-range",
      rangeStart === "—"
        ? "暂无可用预测"
        : rangeStart === rangeEnd
          ? rangeStart
          : `${rangeStart} — ${rangeEnd}`,
    );
    text(
      "eligible-count",
      `${completeBacktests.length} / ${methods.length} 种`,
    );
    text("fold-count", `${number(backtest.total_folds, 0)} 折`);
    text(
      "releases-count",
      `${number(meta.catalog_count ?? report.releases.length, 0)} 条目录记录`,
    );
    text(
      "backtest-description",
      `从至少 ${number(backtest.min_train_size ?? meta.min_train_size, 0)} 个日期事件起，每折只使用当时已知的历史记录，预测下一次发布。`,
    );
    text(
      "footer-meta",
      `截点 ${dateLabel(meta.as_of)} · 离线报告 · 无外部依赖`,
    );
    const warnings = Array.isArray(report.warnings) ? report.warnings : [];
    if (warnings.length) {
      const details = element("details");
      details.append(
        element("summary", null, `数据与方法说明 · ${warnings.length} 项`),
      );
      const list = element("ul");
      warnings.forEach((warning) => list.append(element("li", null, warning)));
      details.append(list);
      $("warnings").replaceChildren(details);
      $("warnings").hidden = false;
    }
  }

  function renderHistory() {
    if (eventDates.length < 2) {
      emptyChart(
        "history-chart",
        "至少需要两个不同发布日期，才能观察发布间隔。",
      );
      return;
    }
    const intervals = eventDates.slice(1).map((date, index) => ({
      date,
      previous: eventDates[index],
      days: (dateValue(date) - dateValue(eventDates[index])) / DAY,
    }));
    const width = Math.max(300, $("history-chart").clientWidth);
    const height = 210;
    const margin = { left: 34, right: 10, top: 20, bottom: 38 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const maximum = Math.max(1, ...intervals.map((item) => item.days));
    const topValue = Math.max(10, Math.ceil(maximum / 10) * 10);
    const svg = chart(
      "history-chart",
      width,
      height,
      "历史相邻日期事件的发布间隔，单位为天；横轴按事件顺序排列。",
    );
    for (let tick = 0; tick <= 3; tick += 1) {
      const value = (topValue * tick) / 3;
      const y = margin.top + plotHeight * (1 - tick / 3);
      svg.append(
        svgElement("line", {
          x1: margin.left,
          y1: y,
          x2: width - margin.right,
          y2: y,
          stroke: COLORS.line,
          "stroke-dasharray": tick ? "3 4" : "none",
        }),
      );
      svg.append(
        svgElement(
          "text",
          {
            x: margin.left - 9,
            y: y + 3,
            "text-anchor": "end",
            class: "chart-small",
          },
          number(value, 0),
        ),
      );
    }
    svg.append(
      svgElement(
        "text",
        {
          x: margin.left - 10,
          y: 9,
          "text-anchor": "end",
          class: "chart-small",
        },
        "天",
      ),
    );
    const band = plotWidth / intervals.length;
    const barWidth = Math.max(2, Math.min(27, band * 0.61));
    const tickEvery = Math.max(
      1,
      Math.ceil(intervals.length / (width < 450 ? 4 : 6)),
    );
    intervals.forEach((item, index) => {
      const x = margin.left + band * (index + 0.5);
      const barHeight = Math.max(1, (item.days / topValue) * plotHeight);
      const bar = svgElement("rect", {
        x: x - barWidth / 2,
        y: margin.top + plotHeight - barHeight,
        width: barWidth,
        height: barHeight,
        rx: 3,
        fill: index === intervals.length - 1 ? COLORS.blue : COLORS.paleBlue,
      });
      mark(bar, `${item.previous} 至 ${item.date}，间隔 ${item.days} 天`);
      svg.append(bar);
      if (
        index % tickEvery === 0 ||
        (index === intervals.length - 1 && intervals.length % tickEvery > 1)
      ) {
        svg.append(
          svgElement(
            "text",
            {
              x,
              y: height - 15,
              "text-anchor": "middle",
              class: "chart-small",
            },
            shortDate(dateValue(item.date)),
          ),
        );
      }
    });
  }

  function filteredForecasts() {
    return forecasts
      .filter((item) => !state.group || item.group === state.group)
      .sort((a, b) => {
        if (state.sort === "name")
          return methodLabel(a.method).localeCompare(
            methodLabel(b.method),
            "zh-CN",
          );
        if (state.sort === "date") {
          const aDate = a.status === "ok" ? dateValue(a.next_date) : null;
          const bDate = b.status === "ok" ? dateValue(b.next_date) : null;
          return (
            (aDate ?? Infinity) - (bDate ?? Infinity) ||
            a.method.localeCompare(b.method)
          );
        }
        const aSummary = summaryMap.get(a.method);
        const bSummary = summaryMap.get(b.method);
        const aRankable = rankMap.has(a.method);
        const bRankable = rankMap.has(b.method);
        if (aRankable !== bRankable) return aRankable ? -1 : 1;
        return (
          (aRankable ? aSummary.mae - bSummary.mae : 0) ||
          a.method.localeCompare(b.method)
        );
      });
  }

  function selectMethod(method, navigate = false) {
    state.method = method;
    $("backtest-method").value = method;
    renderBacktest();
    renderComparison();
    if (navigate)
      $("backtest").scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
        block: "start",
      });
  }

  function renderForecastChart(items) {
    if (!items.length) {
      emptyChart("forecast-chart", "当前分组没有预测方法。");
      return;
    }
    const width = Math.max(
      300,
      $("forecast-chart").clientWidth -
        (window.innerWidth <= 480 ? 14 : window.innerWidth <= 760 ? 24 : 48),
    );
    const margin = {
      left: width < 500 ? 119 : 164,
      right: width < 500 ? 23 : 46,
      top: 42,
      bottom: 34,
    };
    const rowHeight = 37;
    const height = margin.top + items.length * rowHeight + margin.bottom;
    const plotWidth = width - margin.left - margin.right;
    const validDates = items
      .filter((item) => item.status === "ok")
      .map((item) => dateValue(item.next_date))
      .filter((date) => date !== null);
    const referenceDates = asOf === null ? validDates : [...validDates, asOf];
    if (!referenceDates.length) {
      emptyChart(
        "forecast-chart",
        "当前方法没有有效预测日期；失败原因见下表。",
      );
      return;
    }
    const earliest = Math.min(...referenceDates);
    const latest = Math.max(...referenceDates);
    const padding = Math.max(7 * DAY, (latest - earliest) * 0.13);
    const minDate = earliest - padding;
    const maxDate = latest + padding;
    const xScale = (date) =>
      margin.left + ((date - minDate) / (maxDate - minDate)) * plotWidth;
    const svg = chart(
      "forecast-chart",
      width,
      height,
      "各方法下一次发布日期，按连续时间排列；绿色虚线表示数据截点，左侧预测已逾期。",
    );
    const tickCount = width < 500 ? 2 : 4;
    for (let tick = 0; tick <= tickCount; tick += 1) {
      const date = minDate + ((maxDate - minDate) * tick) / tickCount;
      const x = xScale(date);
      svg.append(
        svgElement("line", {
          x1: x,
          y1: margin.top - 9,
          x2: x,
          y2: height - margin.bottom + 2,
          stroke: COLORS.line,
        }),
      );
      svg.append(
        svgElement(
          "text",
          { x, y: height - 9, "text-anchor": "middle", class: "chart-small" },
          shortDate(date),
        ),
      );
    }
    items.forEach((item, index) => {
      const y = margin.top + rowHeight * index + rowHeight / 2;
      const selected = item.method === state.method;
      if (selected)
        svg.append(
          svgElement("rect", {
            x: 0,
            y: y - 15,
            width,
            height: 30,
            rx: 5,
            fill: COLORS.selected,
            "fill-opacity": 0.7,
          }),
        );
      const group = svgElement("g", {
        class: "interactive-mark",
        role: "button",
        "aria-label": `${methodLabel(item.method)}，${item.status === "ok" ? `预测 ${dateLabel(item.next_date)}，${timing(item.next_date).text}` : "预测失败"}。查看回测细节。`,
        tabindex: 0,
      });
      const maxLength = width < 500 ? 15 : 23;
      const displayName = methodLabel(item.method);
      const label =
        displayName.length > maxLength
          ? `${displayName.slice(0, maxLength - 1)}…`
          : displayName;
      const labelNode = svgElement(
        "text",
        { x: 8, y: y + 3, class: `chart-label${selected ? " selected" : ""}` },
        label,
      );
      labelNode.append(svgElement("title", {}, item.method));
      group.append(labelNode);
      const date = item.status === "ok" ? dateValue(item.next_date) : null;
      if (date !== null) {
        const x = xScale(date);
        const dot = svgElement("circle", {
          cx: x,
          cy: y,
          r: selected ? 6 : 4.5,
          fill: COLORS.blue,
          stroke: "white",
          "stroke-width": 2,
        });
        dot.append(
          svgElement(
            "title",
            {},
            `${item.method}：${item.next_date}，${timing(item.next_date).text}`,
          ),
        );
        group.append(dot);
      } else {
        group.append(
          svgElement(
            "text",
            { x: margin.left + 8, y: y + 3, class: "chart-small" },
            item.status === "error" ? "预测失败 · 见下表" : "暂无有效日期",
          ),
        );
      }
      group.addEventListener("click", () => selectMethod(item.method, true));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectMethod(item.method, true);
        }
      });
      svg.append(group);
    });
    if (asOf !== null) {
      const referenceX = xScale(asOf);
      svg.append(
        svgElement("line", {
          x1: referenceX,
          y1: margin.top - 15,
          x2: referenceX,
          y2: height - margin.bottom + 2,
          stroke: COLORS.green,
          "stroke-width": 1.2,
          "stroke-dasharray": "4 4",
          "pointer-events": "none",
        }),
      );
      svg.append(
        svgElement(
          "text",
          {
            x: referenceX,
            y: 17,
            "text-anchor": "middle",
            class: "chart-small",
            style: `fill:${COLORS.green}`,
          },
          "数据截点",
        ),
      );
    }
  }

  function renderComparison() {
    const items = filteredForecasts();
    text("comparison-count", `${items.length} / ${forecasts.length} 种方法`);
    const headings = [
      "方法 / 分组",
      ...Array.from({ length: steps }, (_, index) => `第 ${index + 1} 次预测`),
      "MAE（天）",
      "±30 天命中率",
      "回测覆盖",
    ];
    const headingRow = element("tr");
    headings.forEach((label) => {
      const th = element("th", null, label);
      th.scope = "col";
      headingRow.append(th);
    });
    $("methods-head").replaceChildren(headingRow);
    $("methods-body").replaceChildren();
    if (!items.length)
      emptyTable("methods-body", headings.length, "当前分组没有预测方法。");
    items.forEach((item) => {
      const methodSummary = summaryMap.get(item.method);
      const row = element(
        "tr",
        item.method === state.method ? "selected-row" : "",
      );
      const methodCell = cell(row);
      const rank = rankMap.get(item.method);
      const methodButton = element("button", "method-name");
      methodButton.type = "button";
      methodButton.setAttribute(
        "aria-label",
        `查看 ${methodLabel(item.method)} 的回测细节`,
      );
      methodButton.title = item.method;
      methodButton.append(
        element("span", `rank${rank === 1 ? " rank-first" : ""}`, rank || "—"),
        document.createTextNode(methodLabel(item.method)),
      );
      methodButton.addEventListener("click", () =>
        selectMethod(item.method, true),
      );
      methodCell.append(
        methodButton,
        element(
          "span",
          "cell-secondary",
          `${groupLabel(item.group)} · ${item.method}`,
        ),
      );
      for (let step = 0; step < steps; step += 1) {
        const date = item.status === "ok" ? item.dates[step] : null;
        const predictionCell = cell(
          row,
          dateLabel(date),
          step === 0 ? "value-emphasis" : "",
        );
        if (step === 0) {
          if (item.status === "error")
            predictionCell.append(
              element(
                "span",
                "cell-secondary date-late",
                `预测失败：${item.error || "未提供错误详情"}`,
              ),
            );
          else if (dateValue(date) !== null) {
            const status = timing(date);
            predictionCell.append(
              element(
                "span",
                `cell-secondary ${status.className}`,
                status.text,
              ),
            );
          }
        }
      }
      const maeCell = cell(row, number(methodSummary?.mae), "value-emphasis");
      if (!rankMap.has(item.method))
        maeCell.append(element("span", "cell-secondary", "未参与排名"));
      cell(row, percent(methodSummary?.hit_rate_30));
      const coverageCell = cell(row);
      if (methodSummary) {
        coverageCell.append(
          element(
            "span",
            `tag ${methodSummary.eligible && item.status === "ok" ? "tag-green" : methodSummary.total_folds > 0 ? "tag-amber" : ""}`,
            methodSummary.total_folds > 0
              ? `${methodSummary.successful_folds} / ${methodSummary.total_folds} 折`
              : "无可用回测",
          ),
        );
        coverageCell.append(
          element(
            "span",
            "cell-secondary",
            methodSummary.eligible
              ? item.status === "ok"
                ? "完整覆盖"
                : "回测完整 · 当前预测失败"
              : methodSummary.total_folds > 0
                ? "存在失败折"
                : "历史数据不足",
          ),
        );
      } else coverageCell.append(element("span", "tag", "暂无回测"));
      $("methods-body").append(row);
    });
    renderForecastChart(items);
  }

  function renderReleases() {
    const query = $("release-search").value.trim().toLocaleLowerCase();
    const year = $("year-filter").value;
    const items = report.releases
      .filter(
        (item) =>
          (!year || String(item.date).startsWith(year)) &&
          (!query ||
            [item.name, item.id, item.notes].some((value) =>
              String(value || "")
                .toLocaleLowerCase()
                .includes(query),
            )),
      )
      .sort(
        (a, b) =>
          String(b.date).localeCompare(String(a.date)) ||
          String(a.name).localeCompare(String(b.name)),
      );
    text(
      "release-results",
      `显示 ${items.length} / ${report.releases.length} 条`,
    );
    $("releases-body").replaceChildren();
    if (!items.length) {
      emptyTable(
        "releases-body",
        5,
        "没有匹配的发布记录，试试其他名称或年份。",
      );
      return;
    }
    items.forEach((item) => {
      const row = element("tr");
      const nameCell = cell(row);
      nameCell.append(
        element("span", "release-name", item.name),
        element("span", "cell-secondary", item.id),
      );
      cell(row, dateLabel(item.date));
      const gap = gapMap.get(item.date);
      cell(row, finite(gap) ? `${number(gap, 0)} 天` : "首个日期事件");
      const statusCell = cell(row);
      const isFuture =
        dateValue(item.date) !== null &&
        asOf !== null &&
        dateValue(item.date) > asOf;
      statusCell.append(
        element(
          "span",
          `tag ${item.included ? "tag-green" : "tag-amber"}`,
          item.included
            ? "已纳入分析"
            : isFuture
              ? "截点之后 · 未纳入"
              : "未纳入分析",
        ),
      );
      const sourceCell = cell(row, null, "source-cell");
      const source = safeURL(item.source_url);
      if (source) {
        const link = element("a", null, "查看来源 ↗");
        link.href = source;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.setAttribute("aria-label", `查看 ${item.name} 的来源（新窗口）`);
        sourceCell.append(link);
      } else sourceCell.append(element("span", null, "来源待核验"));
      if (item.notes)
        sourceCell.append(element("span", "cell-secondary", item.notes));
      $("releases-body").append(row);
    });
  }

  function renderErrorChart(methodRecords) {
    const usable = methodRecords.filter(
      (item) => item.status === "ok" && finite(item.error_days),
    );
    if (!usable.length) {
      emptyChart(
        "error-chart",
        "暂无有效回测偏差。数据不足或失败原因可在逐折明细中查看。",
      );
      return;
    }
    const width = Math.max(
      300,
      $("error-chart").clientWidth - (window.innerWidth <= 760 ? 24 : 48),
    );
    const height = 245;
    const margin = { top: 20, right: 13, bottom: 38, left: 43 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const maximum = Math.max(
      10,
      Math.ceil(
        Math.max(...usable.map((item) => Math.abs(item.error_days))) / 10,
      ) * 10,
    );
    const yScale = (value) =>
      margin.top + (plotHeight * (1 - value / maximum)) / 2;
    const zero = yScale(0);
    const svg = chart(
      "error-chart",
      width,
      height,
      `${state.method} 的逐折预测偏差，单位为天。负值提前，正值延后；失败折不绘制误差。`,
    );
    [-1, -0.5, 0, 0.5, 1].forEach((fraction) => {
      const value = maximum * fraction;
      const y = yScale(value);
      svg.append(
        svgElement("line", {
          x1: margin.left,
          y1: y,
          x2: width - margin.right,
          y2: y,
          stroke: fraction ? COLORS.line : "#b3becc",
          "stroke-dasharray": fraction ? "3 4" : "none",
        }),
      );
      svg.append(
        svgElement(
          "text",
          {
            x: margin.left - 10,
            y: y + 3,
            "text-anchor": "end",
            class: "chart-small",
          },
          signed(value),
        ),
      );
    });
    svg.append(
      svgElement(
        "text",
        {
          x: margin.left - 10,
          y: 9,
          "text-anchor": "end",
          class: "chart-small",
        },
        "天",
      ),
    );
    const band = plotWidth / methodRecords.length;
    const barWidth = Math.max(2, Math.min(25, band * 0.61));
    const tickEvery = Math.max(
      1,
      Math.ceil(methodRecords.length / (width < 500 ? 4 : 8)),
    );
    methodRecords.forEach((item, index) => {
      const x = margin.left + band * (index + 0.5);
      if (item.status === "ok" && finite(item.error_days)) {
        const y = yScale(item.error_days);
        const bar = svgElement("rect", {
          x: x - barWidth / 2,
          y: Math.min(y, zero),
          width: barWidth,
          height: Math.max(2, Math.abs(zero - y)),
          rx: 2,
          fill: item.error_days < 0 ? COLORS.green : COLORS.blue,
        });
        mark(
          bar,
          `第 ${item.fold} 折，${item.actual_name}，实际 ${item.actual_date}，预测 ${dateLabel(item.predicted_date)}，偏差 ${signed(item.error_days)} 天`,
        );
        svg.append(bar);
      } else {
        const missing = svgElement(
          "text",
          { x, y: zero + 3, "text-anchor": "middle", class: "chart-small" },
          "×",
        );
        mark(
          missing,
          `第 ${item.fold} 折预测失败：${item.error || "没有有效预测"}`,
        );
        svg.append(missing);
      }
      if (index % tickEvery === 0)
        svg.append(
          svgElement(
            "text",
            {
              x,
              y: height - 14,
              "text-anchor": "middle",
              class: "chart-small",
            },
            `#${item.fold}`,
          ),
        );
    });
  }

  function renderBacktest() {
    const methodSummary = summaryMap.get(state.method);
    const methodRecords = records
      .filter((item) => item.method === state.method)
      .sort((a, b) => a.fold - b.fold);
    text("selected-method-name", methodLabel(state.method) || "暂无回测方法");
    $("selected-method-name").title = state.method;
    text("selected-mae", number(methodSummary?.mae));
    text("selected-bias", signed(methodSummary?.bias));
    text(
      "selected-coverage",
      methodSummary
        ? `${methodSummary.successful_folds} / ${methodSummary.total_folds}`
        : "—",
    );
    text(
      "selected-method-status",
      methodSummary?.eligible
        ? rankMap.has(state.method)
          ? "完整回测 · 参与排名"
          : "回测完整 · 当前预测失败 · 未排名"
        : methodSummary?.total_folds > 0
          ? "回测不完整 · 未参与排名"
          : "历史数据不足 · 暂无回测",
    );
    $("selected-method-status").className =
      `tag ${rankMap.has(state.method) ? "tag-green" : methodSummary?.total_folds > 0 ? "tag-amber" : ""}`;
    text("fold-detail-count", `${methodRecords.length} 折`);
    $("folds-body").replaceChildren();
    if (!methodRecords.length)
      emptyTable("folds-body", 7, "当前方法暂无逐折回测记录。");
    methodRecords.forEach((item) => {
      const row = element("tr");
      cell(row, String(item.fold).padStart(2, "0"));
      cell(row, dateLabel(item.train_end));
      cell(row, item.actual_name);
      cell(row, dateLabel(item.actual_date));
      cell(row, item.status === "ok" ? dateLabel(item.predicted_date) : "—");
      cell(
        row,
        item.status === "ok" ? signed(item.error_days) : "—",
        finite(item.error_days)
          ? item.error_days < 0
            ? "signed-early"
            : item.error_days > 0
              ? "signed-late"
              : ""
          : "",
      );
      const statusCell = cell(row);
      statusCell.append(
        element(
          "span",
          `tag ${item.status === "ok" ? "tag-green" : "tag-amber"}`,
          item.status === "ok" ? "成功" : "失败",
        ),
      );
      if (item.status !== "ok" && item.error)
        statusCell.append(element("span", "cell-secondary", item.error));
      $("folds-body").append(row);
    });
    renderErrorChart(methodRecords);
  }

  function bindControls() {
    [...new Set(forecasts.map((item) => item.group).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b))
      .forEach((group) => option($("group-filter"), group, groupLabel(group)));
    [...new Set(report.releases.map((item) => String(item.date).slice(0, 4)))]
      .filter((year) => /^\d{4}$/.test(year))
      .sort()
      .reverse()
      .forEach((year) => option($("year-filter"), year, `${year} 年`));
    methods.forEach((method) =>
      option($("backtest-method"), method, methodLabel(method)),
    );
    if (!methods.length) {
      option($("backtest-method"), "", "暂无方法");
      $("backtest-method").disabled = true;
    }
    $("backtest-method").value = state.method;
    $("group-filter").addEventListener("change", (event) => {
      state.group = event.target.value;
      renderComparison();
    });
    $("method-sort").addEventListener("change", (event) => {
      state.sort = event.target.value;
      renderComparison();
    });
    $("release-search").addEventListener("input", renderReleases);
    $("year-filter").addEventListener("change", renderReleases);
    $("backtest-method").addEventListener("change", (event) =>
      selectMethod(event.target.value),
    );
    $("download-data").addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: "application/json;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const link = element("a");
      link.href = url;
      link.download = `deepseek-release-report-${dateValue(meta.as_of) !== null ? meta.as_of : "data"}.json`;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    const links = [...document.querySelectorAll(".nav-link")];
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          entries
            .filter((entry) => entry.isIntersecting)
            .forEach((entry) => {
              links.forEach((link) => {
                const active =
                  link.getAttribute("href") === `#${entry.target.id}`;
                link.classList.toggle("active", active);
                if (active) link.setAttribute("aria-current", "location");
                else link.removeAttribute("aria-current");
              });
            });
        },
        { rootMargin: "-15% 0px -70% 0px", threshold: 0 },
      );
      document
        .querySelectorAll("main > section")
        .forEach((section) => observer.observe(section));
    }
    let resizeTimer;
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        renderHistory();
        renderForecastChart(filteredForecasts());
        renderErrorChart(
          records
            .filter((item) => item.method === state.method)
            .sort((a, b) => a.fold - b.fold),
        );
      }, 100);
    });
  }

  renderOverview();
  bindControls();
  renderHistory();
  renderComparison();
  renderReleases();
  renderBacktest();
})();
