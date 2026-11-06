/**
 * Speedup Chart Component
 * 
 * Usage:
 *   import { createSpeedupChart } from './speedup-chart.js';
 *   createSpeedupChart('#my-container');
 *   // or with custom config:
 *   createSpeedupChart('#my-container', { title: '...', datasets: {...} });
 * 
 * Returns: { chart, destroy }
 */

const DEFAULTS = {
    labels: ['32', '64', '128', '256', '512', '1024'],
    datasets: {
        'Full Attn.':   { data: [0.50, 1.31, 3.97, 13.41, 50.01, 202.39], color: '#ef4444' },
        'FastVGGT':     { data: [0.44, 0.88, 1.96,  4.95, 14.13,  45.49], color: '#22c55e' },
        'Block Sparse': { data: [0.46, 0.85, 1.69,  3.77,  9.64,  29.58], color: '#f59e0b' },
        'Ours':         { data: [0.37, 0.71, 1.44,  3.06,  6.83,  16.38], color: '#3b82f6' },
    },
    speedups: ['1.4×', '1.8×', '2.8×', '4.4×', '7.3×', '12.4×'],
    title: 'Inference Time & Accelerating Speedup',
    subtitle: 'Comparing mean latency (seconds) across sequence lengths.',
    highlightText: 'The speedup gap widens significantly',
    subtitleSuffix: ' as image number grows.',
    yAxisLabel: 'Latency (s)',
    xAxisLabel: '# Images',
    height: 490,
    arrowDuration: 2200,
    labelStartDelay: 350,
    labelGap: 260,
    baselineDataset: 0,   // index of "Full Attn." (first)
    oursDataset: 3,       // index of "Ours" (last)
};

const BADGE_STYLES = [
    { bg: 'rgba(96,165,250,0.16)',  bd: '#60a5fa', clr: '#2563eb' },
    { bg: 'rgba(105,150,250,0.16)', bd: '#6996fa', clr: '#3b5de4' },
    { bg: 'rgba(118,131,244,0.16)', bd: '#7683f4', clr: '#5547d8' },
    { bg: 'rgba(129,115,240,0.16)', bd: '#8173f0', clr: '#6d31cd' },
    { bg: 'rgba(139,92,246,0.16)',  bd: '#8b5cf6', clr: '#7c2dbd' },
    { bg: 'linear-gradient(135deg,#8b5cf6,#a855f7)', bd: 'transparent', clr: '#fff' },
];


export function createSpeedupChart(selector, userConfig = {}) {
    const cfg = { ...DEFAULTS, ...userConfig };
    const root = typeof selector === 'string' ? document.querySelector(selector) : selector;
    if (!root) throw new Error(`SpeedupChart: "${selector}" not found`);

    // ── Build DOM ──
    root.innerHTML = '';
    const container = el('div', 'sc-container');

    const title = el('h1', 'sc-title');
    title.textContent = cfg.title;
    container.appendChild(title);

    const subtitle = el('p', 'sc-subtitle');
    subtitle.innerHTML = `${cfg.subtitle} <span class="sc-highlight">${cfg.highlightText}</span>${cfg.subtitleSuffix}`;
    container.appendChild(subtitle);

    const chartWrapper = el('div', 'sc-chart-wrapper');
    chartWrapper.style.height = cfg.height + 'px';

    const canvas = document.createElement('canvas');
    chartWrapper.appendChild(canvas);

    // SVG overlay
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.classList.add('sc-overlay-svg');
    svg.innerHTML = `
        <defs>
            <linearGradient id="sc-arrowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#60a5fa"/>
                <stop offset="45%" stop-color="#818cf8"/>
                <stop offset="100%" stop-color="#a855f7"/>
            </linearGradient>
            <filter id="sc-blurGlow" x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="8"/>
            </filter>
            <filter id="sc-dotGlow" x="-150%" y="-150%" width="400%" height="400%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <path class="sc-arrow-glow" d=""/>
        <path class="sc-arrow-main" d=""/>
        <polygon fill="#a855f7" opacity="0" style="transition:opacity 0.35s ease"/>
        <circle r="5" fill="white" opacity="0" filter="url(#sc-dotGlow)"/>
    `;
    chartWrapper.appendChild(svg);

    const labelsLayer = el('div', 'sc-labels-layer');
    chartWrapper.appendChild(labelsLayer);

    container.appendChild(chartWrapper);
    root.appendChild(container);

    // ── Refs ──
    const glowPath = svg.querySelector('.sc-arrow-glow');
    const mainPath = svg.querySelector('.sc-arrow-main');
    const arrowHead = svg.querySelector('polygon');
    const dot = svg.querySelector('circle');

    // ── Build Chart.js datasets ──
    const dsEntries = Object.entries(cfg.datasets);
    const chartDatasets = dsEntries.map(([label, d]) => ({
        label,
        data: d.data,
        backgroundColor: d.color,
        borderRadius: 4,
        borderSkipped: 'bottom',
    }));

    let chartInstance, rafId, played = false;

    function initChart() {
        chartInstance = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: { labels: cfg.labels, datasets: chartDatasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart',
                    onComplete() {
                        if (!played) { played = true; setTimeout(playArrow, 400); }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { borderDash: [2, 2], color: '#e2e8f0' },
                        title: { display: true, text: cfg.yAxisLabel, font: { weight: '600' } },
                    },
                    x: {
                        grid: { display: false },
                        title: { display: true, text: cfg.xAxisLabel, font: { weight: '600' } },
                    }
                },
                plugins: {
                    legend: {
                        position: 'top', align: 'start',
                        labels: { usePointStyle: true, pointStyle: 'rectRounded', padding: 14 },
                    },
                    tooltip: { mode: 'index', intersect: false },
                },
                layout: { padding: { top: 45, right: 15 } },
            }
        });
    }

    // ── Arrow animation ──
    function playArrow() {
        const mBase = chartInstance.getDatasetMeta(cfg.baselineDataset);
        const mOurs = chartInstance.getDatasetMeta(cfg.oursDataset);

        const info = cfg.speedups.map((_, i) => {
            const bF = mBase.data[i], bO = mOurs.data[i];
            return { cx: (bF.x + bO.x) / 2, fullY: bF.y, oursY: bO.y };
        });

        const n = info.length;
        const startX = info[0].cx;
        const startY = info[0].fullY - 48;
        const endX   = info[n - 1].cx;
        const endY   = info[n - 1].fullY - 22;

        const c1x = startX + (endX - startX) * 0.33;
        const c1y = startY;
        const c2x = startX + (endX - startX) * 0.66;
        const c2y = startY;

        const pathD = `M${startX},${startY} C${c1x},${c1y} ${c2x},${c2y} ${endX},${endY}`;

        glowPath.setAttribute('d', pathD);
        mainPath.setAttribute('d', pathD);

        const len = mainPath.getTotalLength();
        mainPath.style.strokeDasharray  = len;
        mainPath.style.strokeDashoffset = len;
        mainPath.style.transition       = 'none';
        void mainPath.getBoundingClientRect();
        mainPath.style.transition = `stroke-dashoffset ${cfg.arrowDuration}ms cubic-bezier(0.22,0.61,0.36,1)`;
        mainPath.style.strokeDashoffset = '0';
        glowPath.classList.add('on');

        setTimeout(() => placeArrowhead(mainPath, len), cfg.arrowDuration - 120);

        const t0 = performance.now();
        function tick(now) {
            const p = Math.min((now - t0) / cfg.arrowDuration, 1);
            const e = 1 - Math.pow(1 - p, 2.5);
            const pt = mainPath.getPointAtLength(e * len);
            dot.setAttribute('cx', pt.x);
            dot.setAttribute('cy', pt.y);
            dot.setAttribute('opacity', p < 0.96 ? '0.92' : '0');
            if (p < 1) rafId = requestAnimationFrame(tick);
        }
        dot.setAttribute('opacity', '0.92');
        rafId = requestAnimationFrame(tick);

        // badges
        labelsLayer.innerHTML = '';
        info.forEach((p, i) => {
            const badge = el('div', 'sc-badge');
            badge.style.left = p.cx + 'px';
            badge.style.top  = (p.fullY - 22) + 'px';
            const s = BADGE_STYLES[i] || BADGE_STYLES[BADGE_STYLES.length - 1];

            if (i === n - 1) {
                badge.classList.add('sc-hero');
                badge.style.background = s.bg;
                badge.style.color = s.clr;
                badge.style.border = 'none';
                badge.innerHTML = '🚀 ' + cfg.speedups[i];
            } else {
                badge.style.background = s.bg;
                badge.style.border = '1.5px solid ' + s.bd;
                badge.style.color = s.clr;
                badge.textContent = cfg.speedups[i];
            }
            labelsLayer.appendChild(badge);
            setTimeout(() => badge.classList.add('on'), cfg.labelStartDelay + i * cfg.labelGap);
        });
    }

    function placeArrowhead(pathEl, totalLen) {
        const p  = pathEl.getPointAtLength(totalLen);
        const p2 = pathEl.getPointAtLength(Math.max(0, totalLen - 12));
        const ang = Math.atan2(p.y - p2.y, p.x - p2.x);
        const sz = 15, sp = Math.PI / 5.5;
        arrowHead.setAttribute('points', [
            `${p.x},${p.y}`,
            `${p.x - sz * Math.cos(ang - sp)},${p.y - sz * Math.sin(ang - sp)}`,
            `${p.x - sz * 0.35 * Math.cos(ang)},${p.y - sz * 0.35 * Math.sin(ang)}`,
            `${p.x - sz * Math.cos(ang + sp)},${p.y - sz * Math.sin(ang + sp)}`,
        ].join(' '));
        arrowHead.setAttribute('opacity', '1');
    }

    function resetOverlay() {
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
        glowPath.classList.remove('on');
        arrowHead.setAttribute('opacity', '0');
        dot.setAttribute('opacity', '0');
        labelsLayer.innerHTML = '';
        mainPath.style.transition = 'none';
        mainPath.style.strokeDashoffset = mainPath.style.strokeDasharray || '2000';
    }

    // ── Resize handler ──
    let resizeTimer;
    const onResize = () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            resetOverlay();
            chartInstance.resize();
            setTimeout(playArrow, 300);
        }, 250);
    };
    window.addEventListener('resize', onResize);

    // ── Destroy method ──
    function destroy() {
        window.removeEventListener('resize', onResize);
        if (rafId) cancelAnimationFrame(rafId);
        if (chartInstance) chartInstance.destroy();
        root.innerHTML = '';
    }

    // ── Init ──
    initChart();

    return { chart: chartInstance, destroy };
}


// ── Helper ──
function el(tag, className) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    return e;
}