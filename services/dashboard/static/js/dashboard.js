/* ============================================
   Fintech Dashboard - D3.js Chart Engine
   ============================================ */

// --- State ---
let state = {
    currentTicker: null,
    currentPeriod: '1m',
    chartData: [],
    autoRefresh: true,
    refreshInterval: null,
    refreshRate: 10000 // 10 seconds
};

// --- D3 Chart Dimensions ---
const margin = { top: 20, right: 60, bottom: 40, left: 70 };

// --- Date Formatters ---
const formatDate = d3.timeFormat('%d %b %Y');
const formatDateTime = d3.timeFormat('%d %b %Y %H:%M');
const formatPrice = d3.format(',.2f');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initTickers();
    initPeriodButtons();
    initDataToggle();
    initAutoRefresh();
});

// ========================================
// TICKER MANAGEMENT
// ========================================
async function initTickers() {
    const select = document.getElementById('tickerSelect');
    try {
        const response = await fetch('/api/tickers');
        const tickers = await response.json();

        select.innerHTML = '';

        if (tickers.length === 0) {
            select.innerHTML = '<option value="">Veri bekleniyor...</option>';
            return;
        }

        tickers.forEach((ticker, i) => {
            const option = document.createElement('option');
            option.value = ticker;
            option.textContent = ticker;
            select.appendChild(option);
        });

        // Auto-select first ticker
        state.currentTicker = tickers[0];
        select.value = tickers[0];
        loadDashboard();

    } catch (error) {
        console.error('Ticker fetch error:', error);
        select.innerHTML = '<option value="">Bağlantı hatası</option>';
        updateConnectionStatus(false);
    }

    select.addEventListener('change', (e) => {
        state.currentTicker = e.target.value;
        if (state.currentTicker) {
            loadDashboard();
        }
    });
}

// ========================================
// PERIOD BUTTONS
// ========================================
function initPeriodButtons() {
    const buttons = document.querySelectorAll('.period-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentPeriod = btn.dataset.period;
            if (state.currentTicker) {
                loadDashboard();
            }
        });
    });
}

// ========================================
// DATA TABLE TOGGLE
// ========================================
function initDataToggle() {
    const toggle = document.getElementById('dataToggle');
    const container = document.getElementById('dataTableContainer');
    const btn = document.getElementById('expandBtn');

    toggle.addEventListener('click', () => {
        const isShowing = container.style.display !== 'none';
        container.style.display = isShowing ? 'none' : 'block';
        btn.classList.toggle('expanded', !isShowing);
    });
}

// ========================================
// AUTO-REFRESH
// ========================================
function initAutoRefresh() {
    const toggle = document.getElementById('autoRefreshToggle');
    toggle.addEventListener('change', (e) => {
        state.autoRefresh = e.target.checked;
        if (state.autoRefresh) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    });
    startAutoRefresh();
}

function startAutoRefresh() {
    stopAutoRefresh();
    state.refreshInterval = setInterval(() => {
        if (state.currentTicker) {
            loadDashboard(true); // silent refresh
        }
    }, state.refreshRate);
}

function stopAutoRefresh() {
    if (state.refreshInterval) {
        clearInterval(state.refreshInterval);
        state.refreshInterval = null;
    }
}

// ========================================
// MAIN DATA LOADER
// ========================================
async function loadDashboard(silent = false) {
    if (!state.currentTicker) return;

    updateConnectionStatus(true);

    try {
        // Fetch all data in parallel
        const [signalsRes, latestRes, statsRes] = await Promise.all([
            fetch(`/api/signals/${state.currentTicker}?period=${state.currentPeriod}`),
            fetch(`/api/latest/${state.currentTicker}`),
            fetch(`/api/stats/${state.currentTicker}?period=${state.currentPeriod}`)
        ]);

        const signals = await signalsRes.json();
        const latest = await latestRes.json();
        const stats = await statsRes.json();

        // Process data
        state.chartData = signals.map(d => ({
            date: new Date(d.timestamp),
            price: d.price,
            prediction: d.prediction,
            indicators: d.indicators || {}
        }));

        // Update UI
        updateKPIs(latest);
        updateStats(stats);
        renderChart(state.chartData);
        updateDataTable(signals);
        updateChartTitle();

    } catch (error) {
        console.error('Dashboard load error:', error);
        updateConnectionStatus(false);
    }
}

// ========================================
// KPI UPDATES
// ========================================
function updateKPIs(latest) {
    if (latest.error) {
        document.getElementById('kpiPriceValue').textContent = '--';
        document.getElementById('kpiSignalValue').textContent = '--';
        document.getElementById('kpiRSIValue').textContent = '--';
        document.getElementById('kpiSentimentValue').textContent = '--';
        return;
    }

    // Price
    const priceEl = document.getElementById('kpiPriceValue');
    priceEl.textContent = `$${formatPrice(latest.price)}`;
    document.getElementById('kpiPriceSub').textContent = latest.ticker;

    // Signal
    const signalEl = document.getElementById('kpiSignalValue');
    if (latest.prediction === 1) {
        signalEl.textContent = 'AL 🟢';
        signalEl.className = 'kpi-value buy';
        document.getElementById('kpiSignalSub').textContent = 'BUY sinyali aktif';
    } else {
        signalEl.textContent = 'TUT ⚪';
        signalEl.className = 'kpi-value hold';
        document.getElementById('kpiSignalSub').textContent = 'HOLD pozisyonu';
    }

    // RSI
    const indicators = latest.indicators || {};
    const rsi = indicators.RSI || 0;
    const rsiEl = document.getElementById('kpiRSIValue');
    rsiEl.textContent = rsi.toFixed(2);

    if (rsi > 70) {
        rsiEl.className = 'kpi-value overbought';
        document.getElementById('kpiRSISub').textContent = 'Aşırı alım bölgesi';
    } else if (rsi < 30) {
        rsiEl.className = 'kpi-value oversold';
        document.getElementById('kpiRSISub').textContent = 'Aşırı satım bölgesi';
    } else {
        rsiEl.className = 'kpi-value neutral';
        document.getElementById('kpiRSISub').textContent = 'Normal aralıkta';
    }

    // Sentiment
    const sentiment = indicators.Sentiment || 0;
    const sentEl = document.getElementById('kpiSentimentValue');
    sentEl.textContent = sentiment.toFixed(2);
    if (sentiment > 0) {
        sentEl.className = 'kpi-value positive';
        document.getElementById('kpiSentimentSub').textContent = 'Pozitif yönelim';
    } else if (sentiment < 0) {
        sentEl.className = 'kpi-value negative';
        document.getElementById('kpiSentimentSub').textContent = 'Negatif yönelim';
    } else {
        sentEl.className = 'kpi-value';
        document.getElementById('kpiSentimentSub').textContent = 'Nötr';
    }
}

// ========================================
// STATS UPDATES
// ========================================
function updateStats(stats) {
    if (stats.error) return;
    document.getElementById('statTotal').textContent = stats.total_signals || '--';
    document.getElementById('statBuy').textContent = stats.buy_signals || '0';
    document.getElementById('statHold').textContent = stats.hold_signals || '0';
    document.getElementById('statMin').textContent = stats.min_price ? `$${formatPrice(stats.min_price)}` : '--';
    document.getElementById('statMax').textContent = stats.max_price ? `$${formatPrice(stats.max_price)}` : '--';
    document.getElementById('statAvg').textContent = stats.avg_price ? `$${formatPrice(stats.avg_price)}` : '--';
}

// ========================================
// CHART TITLE
// ========================================
function updateChartTitle() {
    const periodLabels = {
        '1w': '1 Hafta',
        '1m': '1 Ay',
        '3m': '3 Ay',
        '1y': '1 Yıl',
        '3y': '3 Yıl'
    };
    const label = periodLabels[state.currentPeriod] || state.currentPeriod;
    document.getElementById('chartTitle').textContent =
        `${state.currentTicker} — ${label} Fiyat Grafiği`;
}

// ========================================
// D3.JS CHART RENDERING
// ========================================
function renderChart(data) {
    const container = document.getElementById('priceChart');
    const placeholder = document.getElementById('chartPlaceholder');

    // Clear existing
    d3.select('#priceChart svg').remove();

    if (!data || data.length === 0) {
        placeholder.style.display = 'flex';
        placeholder.querySelector('p').textContent =
            `${state.currentTicker} için bu dönemde veri bulunamadı`;
        return;
    }

    placeholder.style.display = 'none';

    const width = container.clientWidth;
    const height = container.clientHeight;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select('#priceChart')
        .append('svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // --- Scales ---
    const xScale = d3.scaleTime()
        .domain(d3.extent(data, d => d.date))
        .range([0, innerWidth]);

    const priceExtent = d3.extent(data, d => d.price);
    const pricePadding = (priceExtent[1] - priceExtent[0]) * 0.08 || 1;
    const yScale = d3.scaleLinear()
        .domain([priceExtent[0] - pricePadding, priceExtent[1] + pricePadding])
        .range([innerHeight, 0]);

    // --- Gradient Definition ---
    const defs = svg.append('defs');

    const areaGradient = defs.append('linearGradient')
        .attr('id', 'areaGradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%');

    areaGradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#6366f1')
        .attr('stop-opacity', 0.3);

    areaGradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#6366f1')
        .attr('stop-opacity', 0.02);

    // Line glow filter
    const glowFilter = defs.append('filter')
        .attr('id', 'lineGlow')
        .attr('x', '-20%').attr('y', '-20%')
        .attr('width', '140%').attr('height', '140%');
    glowFilter.append('feGaussianBlur')
        .attr('stdDeviation', '3')
        .attr('result', 'blur');
    glowFilter.append('feMerge')
        .selectAll('feMergeNode')
        .data(['blur', 'SourceGraphic'])
        .enter()
        .append('feMergeNode')
        .attr('in', d => d);

    // --- Grid Lines ---
    // Horizontal grid
    g.append('g')
        .attr('class', 'grid-lines')
        .selectAll('line')
        .data(yScale.ticks(6))
        .enter()
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', d => yScale(d))
        .attr('y2', d => yScale(d));

    // --- Axes ---
    const xAxis = d3.axisBottom(xScale)
        .ticks(getTickCount())
        .tickFormat(getTickFormat())
        .tickSize(0)
        .tickPadding(10);

    const yAxis = d3.axisLeft(yScale)
        .ticks(6)
        .tickFormat(d => `$${formatPrice(d)}`)
        .tickSize(0)
        .tickPadding(10);

    g.append('g')
        .attr('class', 'axis-x')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(xAxis);

    g.append('g')
        .attr('class', 'axis-y')
        .call(yAxis);

    // --- Area Fill ---
    const area = d3.area()
        .x(d => xScale(d.date))
        .y0(innerHeight)
        .y1(d => yScale(d.price))
        .curve(d3.curveMonotoneX);

    g.append('path')
        .datum(data)
        .attr('class', 'price-area')
        .attr('fill', 'url(#areaGradient)')
        .attr('d', area)
        .attr('opacity', 0)
        .transition()
        .duration(800)
        .attr('opacity', 0.6);

    // --- Price Line ---
    const line = d3.line()
        .x(d => xScale(d.date))
        .y(d => yScale(d.price))
        .curve(d3.curveMonotoneX);

    const path = g.append('path')
        .datum(data)
        .attr('class', 'price-line')
        .attr('d', line)
        .attr('filter', 'url(#lineGlow)');

    // Animate line drawing
    const totalLength = path.node().getTotalLength();
    path
        .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
        .attr('stroke-dashoffset', totalLength)
        .transition()
        .duration(1200)
        .ease(d3.easeCubicInOut)
        .attr('stroke-dashoffset', 0);

    // --- BUY Signal Markers ---
    const buySignals = data.filter(d => d.prediction === 1);

    g.selectAll('.buy-marker')
        .data(buySignals)
        .enter()
        .append('circle')
        .attr('class', 'buy-marker')
        .attr('cx', d => xScale(d.date))
        .attr('cy', d => yScale(d.price))
        .attr('r', 0)
        .transition()
        .delay((d, i) => 1200 + i * 30)
        .duration(400)
        .ease(d3.easeBackOut)
        .attr('r', 5);

    // --- Crosshair + Tooltip ---
    setupCrosshair(g, svg, data, xScale, yScale, innerWidth, innerHeight);
}

// ========================================
// CROSSHAIR & TOOLTIP
// ========================================
function setupCrosshair(g, svg, data, xScale, yScale, innerWidth, innerHeight) {
    const tooltip = document.getElementById('chartTooltip');
    const bisectDate = d3.bisector(d => d.date).left;

    // Crosshair elements
    const crosshairGroup = g.append('g')
        .attr('class', 'crosshair-group')
        .style('display', 'none');

    const vLine = crosshairGroup.append('line')
        .attr('class', 'crosshair-line')
        .attr('y1', 0)
        .attr('y2', innerHeight);

    const hLine = crosshairGroup.append('line')
        .attr('class', 'crosshair-line')
        .attr('x1', 0)
        .attr('x2', innerWidth);

    const focusCircle = crosshairGroup.append('circle')
        .attr('class', 'focus-circle')
        .attr('r', 5);

    // Y-axis price label
    const yLabel = crosshairGroup.append('g')
        .attr('class', 'y-label');

    yLabel.append('rect')
        .attr('x', -margin.left)
        .attr('width', margin.left - 5)
        .attr('height', 22)
        .attr('y', -11)
        .attr('rx', 4)
        .attr('fill', '#6366f1');

    const yLabelText = yLabel.append('text')
        .attr('x', -margin.left + 8)
        .attr('dy', '0.35em')
        .attr('fill', 'white')
        .attr('font-size', '11px')
        .attr('font-weight', '500');

    // Overlay for mouse events
    svg.append('rect')
        .attr('class', 'overlay')
        .attr('transform', `translate(${margin.left},${margin.top})`)
        .attr('width', innerWidth)
        .attr('height', innerHeight)
        .attr('fill', 'none')
        .attr('pointer-events', 'all')
        .on('mouseenter', () => {
            crosshairGroup.style('display', null);
            tooltip.style.display = 'block';
        })
        .on('mouseleave', () => {
            crosshairGroup.style('display', 'none');
            tooltip.style.display = 'none';
        })
        .on('mousemove', function (event) {
            const [mx] = d3.pointer(event);
            const x0 = xScale.invert(mx);
            const i = bisectDate(data, x0, 1);
            const d0 = data[i - 1];
            const d1 = data[i];

            if (!d0) return;

            const d = (!d1) ? d0 : (x0 - d0.date > d1.date - x0 ? d1 : d0);

            const cx = xScale(d.date);
            const cy = yScale(d.price);

            // Update crosshair
            vLine.attr('x1', cx).attr('x2', cx);
            hLine.attr('y1', cy).attr('y2', cy);
            focusCircle.attr('cx', cx).attr('cy', cy);
            yLabel.attr('transform', `translate(0,${cy})`);
            yLabelText.text(`$${formatPrice(d.price)}`);

            // Update tooltip
            document.getElementById('tooltipDate').textContent = formatDateTime(d.date);
            document.getElementById('tooltipPrice').textContent = `$${formatPrice(d.price)}`;

            const signalEl = document.getElementById('tooltipSignal');
            if (d.prediction === 1) {
                signalEl.textContent = '🟢 AL Sinyali';
                signalEl.className = 'tooltip-signal buy';
            } else {
                signalEl.textContent = '⚪ TUT';
                signalEl.className = 'tooltip-signal hold';
            }

            // Position tooltip
            const svgRect = svg.node().getBoundingClientRect();
            let tooltipX = svgRect.left + margin.left + cx + 15;
            let tooltipY = svgRect.top + margin.top + cy - 40;

            // Keep tooltip in viewport
            if (tooltipX + 180 > window.innerWidth) {
                tooltipX = svgRect.left + margin.left + cx - 195;
            }

            tooltip.style.left = `${tooltipX}px`;
            tooltip.style.top = `${tooltipY}px`;
        });
}

// ========================================
// DATA TABLE
// ========================================
function updateDataTable(signals) {
    const tbody = document.getElementById('dataTableBody');
    tbody.innerHTML = '';

    // Show last 20 records (newest first)
    const recent = signals.slice(-20).reverse();

    recent.forEach(signal => {
        const tr = document.createElement('tr');
        const date = new Date(signal.timestamp);
        const indicators = signal.indicators || {};

        const signalClass = signal.prediction === 1 ? 'buy' : 'hold';
        const signalText = signal.prediction === 1 ? '🟢 AL' : '⚪ TUT';

        tr.innerHTML = `
            <td>${formatDateTime(date)}</td>
            <td>${signal.ticker}</td>
            <td>$${formatPrice(signal.price)}</td>
            <td><span class="signal-badge ${signalClass}">${signalText}</span></td>
            <td>${(indicators.RSI || 0).toFixed(2)}</td>
            <td>${(indicators.Sentiment || 0).toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ========================================
// HELPERS
// ========================================
function updateConnectionStatus(connected) {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.connection-status span:last-child');
    if (connected) {
        dot.className = 'status-dot connected';
        text.textContent = 'Bağlı';
    } else {
        dot.className = 'status-dot';
        text.textContent = 'Bağlantı Kesildi';
    }
}

function getTickCount() {
    const periodTicks = {
        '1w': 7,
        '1m': 8,
        '3m': 6,
        '1y': 12,
        '3y': 8
    };
    return periodTicks[state.currentPeriod] || 8;
}

function getTickFormat() {
    if (state.currentPeriod === '1w') {
        return d3.timeFormat('%d %b');
    } else if (state.currentPeriod === '1m') {
        return d3.timeFormat('%d %b');
    } else if (state.currentPeriod === '3m') {
        return d3.timeFormat('%b %Y');
    } else {
        return d3.timeFormat('%b %Y');
    }
}

// Handle window resize
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        if (state.chartData.length > 0) {
            renderChart(state.chartData);
        }
    }, 250);
});
