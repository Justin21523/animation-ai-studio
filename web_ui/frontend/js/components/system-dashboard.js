/**
 * System Dashboard Component
 *
 * Real-time system monitoring dashboard displaying CPU, GPU, RAM, and disk usage.
 * Includes historical charts and live metrics updates via SSE.
 *
 * Features:
 * - Real-time CPU, GPU, RAM usage gauges
 * - GPU temperature and memory display
 * - Disk usage visualization
 * - Historical metrics charts (CPU/GPU over time)
 * - Auto-refresh with SSE streaming
 * - Responsive grid layout
 *
 * Usage:
 * ```javascript
 * const dashboard = new SystemDashboard();
 * await dashboard.mount(container);
 * ```
 */

import BaseComponent from './base-component.js';
import api from '../api/client.js';

export default class SystemDashboard extends BaseComponent {
    constructor() {
        super();

        // Component state
        this.metrics = null;
        this.metricsHistory = [];
        this.loading = false;
        this.error = null;

        // SSE connection
        this.eventSource = null;

        // Chart settings
        this.historyLimit = 60; // Last 60 data points
        this.maxHistoryPoints = 100;
    }

    async onMount() {
        // Subscribe to state changes
        this.subscribeState('system.metrics', (metrics) => {
            this.metrics = metrics;
            this.renderMetrics();
        });

        this.subscribeState('system.history', (history) => {
            this.metricsHistory = history || [];
            this.renderCharts();
        });

        // Initial load
        await this.loadMetrics();
        await this.loadMetricsHistory();

        // Start real-time monitoring
        this.startMonitoring();
    }

    async onUnmount() {
        // Stop monitoring
        this.stopMonitoring();
    }

    /**
     * Load current metrics from API
     */
    async loadMetrics() {
        try {
            const metrics = await api.system.getMetrics();
            this.metrics = metrics;
            this.setState('system.metrics', metrics);

            // Add to history
            this.addToHistory(metrics);

        } catch (error) {
            console.error('Failed to load metrics:', error);
            this.error = error.message;
        }
    }

    /**
     * Load historical metrics
     */
    async loadMetricsHistory() {
        try {
            const history = await api.system.getMetricsHistory(this.historyLimit);
            this.metricsHistory = history;
            this.setState('system.history', history);

        } catch (error) {
            console.error('Failed to load metrics history:', error);
        }
    }

    /**
     * Start real-time monitoring via SSE
     */
    startMonitoring() {
        this.setState('system.isMonitoring', true);

        this.eventSource = api.system.streamMetrics((metrics) => {
            this.metrics = metrics;
            this.setState('system.metrics', metrics);

            // Add to history
            this.addToHistory(metrics);
        });

        // Handle errors
        this.eventSource.addEventListener('error', (event) => {
            console.error('SSE error:', event);
            this.stopMonitoring();

            // Retry after 5 seconds
            setTimeout(() => {
                if (this.isMounted) {
                    this.startMonitoring();
                }
            }, 5000);
        });
    }

    /**
     * Stop real-time monitoring
     */
    stopMonitoring() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        this.setState('system.isMonitoring', false);
    }

    /**
     * Add metrics to history
     */
    addToHistory(metrics) {
        this.metricsHistory.push({
            ...metrics,
            timestamp: Date.now()
        });

        // Limit history size
        if (this.metricsHistory.length > this.maxHistoryPoints) {
            this.metricsHistory.shift();
        }

        this.setState('system.history', this.metricsHistory);
    }

    /**
     * Render dashboard
     */
    renderMetrics() {
        if (!this.metrics) {
            this.showLoading('Loading system metrics...');
            return;
        }

        const html = `
            <div class="system-dashboard-container">
                <!-- Header -->
                <div class="dashboard-header">
                    <h2>System Monitor</h2>
                    <div class="dashboard-controls">
                        <span class="status-indicator ${this.getState('system.isMonitoring') ? 'active' : ''}">
                            ${this.getState('system.isMonitoring') ? '🟢 Live' : '🔴 Offline'}
                        </span>
                        <button class="btn btn-sm" id="refresh-btn">
                            🔄 Refresh
                        </button>
                    </div>
                </div>

                <!-- Metrics Grid -->
                <div class="metrics-grid">
                    ${this.renderCPUGauge()}
                    ${this.renderRAMGauge()}
                    ${this.renderGPUGauge()}
                    ${this.renderDiskUsage()}
                </div>

                <!-- Charts -->
                <div class="charts-container">
                    ${this.renderUsageChart()}
                </div>
            </div>
        `;

        this.render(html);
        this.setupEventListeners();
    }

    /**
     * Render CPU usage gauge
     */
    renderCPUGauge() {
        const cpuPercent = this.metrics.cpu_percent || 0;
        const color = this.getUsageColor(cpuPercent);

        return `
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-icon">🖥️</span>
                    <h3>CPU</h3>
                </div>
                <div class="metric-body">
                    ${this.renderGauge(cpuPercent, color)}
                    <div class="metric-value" style="color: ${color}">
                        ${cpuPercent.toFixed(1)}%
                    </div>
                    <div class="metric-details">
                        ${this.metrics.cpu_count ? `<div>Cores: ${this.metrics.cpu_count}</div>` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render RAM usage gauge
     */
    renderRAMGauge() {
        const ramPercent = this.metrics.ram_percent || 0;
        const ramUsed = this.metrics.ram_used_gb || 0;
        const ramTotal = this.metrics.ram_total_gb || 0;
        const color = this.getUsageColor(ramPercent);

        return `
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-icon">💾</span>
                    <h3>RAM</h3>
                </div>
                <div class="metric-body">
                    ${this.renderGauge(ramPercent, color)}
                    <div class="metric-value" style="color: ${color}">
                        ${ramPercent.toFixed(1)}%
                    </div>
                    <div class="metric-details">
                        <div>${ramUsed.toFixed(1)} / ${ramTotal.toFixed(1)} GB</div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render GPU usage gauge
     */
    renderGPUGauge() {
        if (!this.metrics.gpu_available) {
            return `
                <div class="metric-card metric-card-disabled">
                    <div class="metric-header">
                        <span class="metric-icon">🎮</span>
                        <h3>GPU</h3>
                    </div>
                    <div class="metric-body">
                        <div class="metric-unavailable">No GPU detected</div>
                    </div>
                </div>
            `;
        }

        const gpuPercent = this.metrics.gpu_percent || 0;
        const gpuMemUsed = this.metrics.gpu_memory_used_mb || 0;
        const gpuMemTotal = this.metrics.gpu_memory_total_mb || 0;
        const gpuTemp = this.metrics.gpu_temperature_c || 0;
        const color = this.getUsageColor(gpuPercent);

        return `
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-icon">🎮</span>
                    <h3>GPU</h3>
                </div>
                <div class="metric-body">
                    ${this.renderGauge(gpuPercent, color)}
                    <div class="metric-value" style="color: ${color}">
                        ${gpuPercent.toFixed(1)}%
                    </div>
                    <div class="metric-details">
                        <div>Memory: ${(gpuMemUsed / 1024).toFixed(1)} / ${(gpuMemTotal / 1024).toFixed(1)} GB</div>
                        <div>Temp: ${gpuTemp.toFixed(0)}°C</div>
                        ${this.metrics.gpu_name ? `<div>${this.metrics.gpu_name}</div>` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render disk usage
     */
    renderDiskUsage() {
        const diskPercent = this.metrics.disk_percent || 0;
        const diskUsed = this.metrics.disk_used_gb || 0;
        const diskTotal = this.metrics.disk_total_gb || 0;
        const color = this.getUsageColor(diskPercent);

        return `
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-icon">💽</span>
                    <h3>Disk</h3>
                </div>
                <div class="metric-body">
                    ${this.renderGauge(diskPercent, color)}
                    <div class="metric-value" style="color: ${color}">
                        ${diskPercent.toFixed(1)}%
                    </div>
                    <div class="metric-details">
                        <div>${diskUsed.toFixed(1)} / ${diskTotal.toFixed(1)} GB</div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render circular gauge
     */
    renderGauge(percent, color) {
        const radius = 45;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (percent / 100) * circumference;

        return `
            <svg class="gauge" viewBox="0 0 100 100">
                <circle class="gauge-bg" cx="50" cy="50" r="${radius}"></circle>
                <circle
                    class="gauge-fill"
                    cx="50"
                    cy="50"
                    r="${radius}"
                    stroke="${color}"
                    stroke-dasharray="${circumference}"
                    stroke-dashoffset="${offset}"
                    transform="rotate(-90 50 50)"
                ></circle>
            </svg>
        `;
    }

    /**
     * Get color based on usage percentage
     */
    getUsageColor(percent) {
        if (percent < 50) return '#4caf50';  // Green
        if (percent < 80) return '#ff9800';  // Orange
        return '#f44336';  // Red
    }

    /**
     * Render usage chart
     */
    renderUsageChart() {
        if (this.metricsHistory.length === 0) {
            return '';
        }

        // Prepare chart data
        const cpuData = this.metricsHistory.map(m => m.cpu_percent || 0);
        const ramData = this.metricsHistory.map(m => m.ram_percent || 0);
        const gpuData = this.metricsHistory.map(m => m.gpu_percent || 0);
        const labels = this.metricsHistory.map((m, i) => {
            if (i % 5 === 0) {
                return new Date(m.timestamp).toLocaleTimeString();
            }
            return '';
        });

        return `
            <div class="chart-card">
                <h3>Usage History (Last ${Math.min(this.metricsHistory.length, this.historyLimit)} points)</h3>
                <div class="chart-container">
                    ${this.renderSimpleLineChart(cpuData, ramData, gpuData, labels)}
                </div>
                <div class="chart-legend">
                    <span class="legend-item"><span class="legend-color" style="background: #2196f3"></span> CPU</span>
                    <span class="legend-item"><span class="legend-color" style="background: #4caf50"></span> RAM</span>
                    ${this.metrics.gpu_available ? `<span class="legend-item"><span class="legend-color" style="background: #ff9800"></span> GPU</span>` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Render simple ASCII-style line chart
     */
    renderSimpleLineChart(cpuData, ramData, gpuData, labels) {
        const width = 800;
        const height = 200;
        const padding = 40;

        const maxValue = 100;
        const minValue = 0;

        // Create SVG points
        const createPoints = (data, color) => {
            const points = data.map((value, index) => {
                const x = padding + (index / (data.length - 1 || 1)) * (width - 2 * padding);
                const y = height - padding - ((value - minValue) / (maxValue - minValue)) * (height - 2 * padding);
                return `${x},${y}`;
            }).join(' ');

            return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2" />`;
        };

        return `
            <svg class="line-chart" viewBox="0 0 ${width} ${height}">
                <!-- Grid lines -->
                ${[0, 25, 50, 75, 100].map(val => {
                    const y = height - padding - ((val - minValue) / (maxValue - minValue)) * (height - 2 * padding);
                    return `
                        <line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="#333" stroke-width="1" opacity="0.3"/>
                        <text x="${padding - 10}" y="${y + 5}" fill="#999" font-size="12" text-anchor="end">${val}%</text>
                    `;
                }).join('')}

                <!-- Data lines -->
                ${createPoints(cpuData, '#2196f3')}
                ${createPoints(ramData, '#4caf50')}
                ${this.metrics.gpu_available ? createPoints(gpuData, '#ff9800') : ''}

                <!-- X-axis -->
                <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#999" stroke-width="2"/>

                <!-- Y-axis -->
                <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#999" stroke-width="2"/>
            </svg>
        `;
    }

    /**
     * Render charts (stub for future chart library integration)
     */
    renderCharts() {
        // This is called when history updates
        // Re-render the entire component to update charts
        if (this.metrics) {
            this.renderMetrics();
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Refresh button
        const refreshBtn = this.$('#refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                await this.loadMetrics();
                await this.loadMetricsHistory();
            });
        }
    }
}
