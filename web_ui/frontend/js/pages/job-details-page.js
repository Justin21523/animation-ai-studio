import BaseComponent from '../components/base-component.js';
import api from '../api/client.js';
import router from '../router.js';

export default class JobDetailsPage extends BaseComponent {
    constructor(jobId) {
        super();
        this.jobId = jobId;
        this.job = null;
        this.outputs = [];
        this.logs = null;
        this.sse = null;
        this.refreshTimer = null;
    }

    async onMount() {
        this.render(`<div class="page-container job-details-page">Loading job...</div>`);

        await this.loadAll();
        this.renderPage();
        this.startStreaming();
    }

    async onUnmount() {
        if (this.sse) {
            this.sse.close();
            this.sse = null;
        }
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    async loadAll() {
        this.job = await api.jobs.get(this.jobId);
        try {
            this.outputs = await api.jobs.getOutputs(this.jobId);
        } catch {
            this.outputs = [];
        }
        try {
            this.logs = await api.jobs.getLogs(this.jobId, { tail: 200 });
        } catch {
            this.logs = null;
        }
    }

    renderPage() {
        const job = this.job;
        const logs = this.logs || { stdout: [], stderr: [] };
        const outputs = Array.isArray(this.outputs) ? this.outputs : [];

        const progress = Number(job.progress_percent || 0).toFixed(1);
        const canCancel = job.status === 'running';

        this.render(`
            <div class="page-container job-details-page">
                <div class="page-header">
                    <div>
                        <h2>Job Details</h2>
                        <div class="muted">ID: ${this.escapeHtml(job.job_id)}</div>
                    </div>
                    <div class="page-actions">
                        <button class="btn btn-ghost" id="back-btn">← Back</button>
                        ${canCancel ? `<button class="btn btn-danger" id="cancel-btn">Cancel</button>` : ''}
                        <button class="btn btn-primary" id="refresh-btn">Refresh</button>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h3>Summary</h3>
                    </div>
                    <div class="card-body">
                        <div class="kv-grid">
                            <div><span class="label">Film</span><span class="value">${this.escapeHtml(job.film_name)}</span></div>
                            <div><span class="label">Pipeline</span><span class="value">${this.escapeHtml(job.pipeline_type)}</span></div>
                            <div><span class="label">Status</span><span class="value">${this.escapeHtml(job.status)}</span></div>
                            <div><span class="label">Stage</span><span class="value">${this.escapeHtml(job.current_stage || '-')}</span></div>
                            <div><span class="label">Progress</span><span class="value">${progress}%</span></div>
                            <div><span class="label">Output</span><span class="value">${this.escapeHtml(job.output_base_dir)}</span></div>
                        </div>
                        <div class="progress-bar-container" style="margin-top: 10px;">
                            <div class="progress-bar"><div class="progress-bar-fill" style="width: ${progress}%"></div></div>
                            <span class="progress-text">${progress}%</span>
                        </div>
                        ${job.error_message ? `<div class="job-error" style="margin-top: 10px;">⚠️ ${this.escapeHtml(job.error_message)}</div>` : ''}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h3>Outputs</h3>
                    </div>
                    <div class="card-body">
                        ${outputs.length === 0 ? `<div class="muted">No outputs recorded yet.</div>` : `
                            <div class="outputs-list">
                                ${outputs.map(o => `
                                    <div class="output-row">
                                        <div class="output-type">${this.escapeHtml(o.output_type)}</div>
                                        <div class="output-path">${this.escapeHtml(o.path)}</div>
                                        <div class="output-stage muted">${this.escapeHtml(o.stage)}</div>
                                    </div>
                                `).join('')}
                            </div>
                        `}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h3>Logs</h3>
                        <div class="page-actions">
                            <button class="btn btn-sm" id="refresh-logs-btn">Refresh Logs</button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="log-grid">
                            <div class="log-pane">
                                <div class="log-title">stdout</div>
                                <pre class="log-viewer" id="stdout-log">${this.escapeHtml((logs.stdout || []).join('\n'))}</pre>
                            </div>
                            <div class="log-pane">
                                <div class="log-title">stderr</div>
                                <pre class="log-viewer log-viewer-error" id="stderr-log">${this.escapeHtml((logs.stderr || []).join('\n'))}</pre>
                            </div>
                        </div>
                        ${this.logs && (this.logs.stdout_path || this.logs.stderr_path) ? `
                            <div class="muted" style="margin-top: 8px;">
                                stdout: ${this.escapeHtml(this.logs.stdout_path || '')}<br/>
                                stderr: ${this.escapeHtml(this.logs.stderr_path || '')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `);

        this.$('#back-btn')?.addEventListener('click', () => router.back());
        this.$('#refresh-btn')?.addEventListener('click', async () => {
            await this.loadAll();
            this.renderPage();
        });
        this.$('#refresh-logs-btn')?.addEventListener('click', async () => {
            await this.refreshLogsOnly();
            this.updateLogsView();
        });
        if (canCancel) {
            this.$('#cancel-btn')?.addEventListener('click', async () => {
                try {
                    await api.jobs.cancel(this.jobId);
                } catch (e) {
                    console.error(e);
                }
                await this.loadAll();
                this.renderPage();
            });
        }
    }

    async refreshLogsOnly() {
        try {
            this.logs = await api.jobs.getLogs(this.jobId, { tail: 300 });
        } catch {
            this.logs = null;
        }
    }

    updateLogsView() {
        const logs = this.logs || { stdout: [], stderr: [] };
        const stdoutEl = this.$('#stdout-log');
        const stderrEl = this.$('#stderr-log');
        if (stdoutEl) stdoutEl.textContent = (logs.stdout || []).join('\n');
        if (stderrEl) stderrEl.textContent = (logs.stderr || []).join('\n');
    }

    startStreaming() {
        if (this.sse) {
            this.sse.close();
            this.sse = null;
        }

        this.sse = api.jobs.streamProgress(
            this.jobId,
            async () => {
                try {
                    this.job = await api.jobs.get(this.jobId);
                    this.updateSummaryOnly();
                } catch {
                    // ignore
                }
            },
            async () => {
                await this.loadAll();
                this.renderPage();
            },
            async () => {
                await this.loadAll();
                this.renderPage();
            }
        );

        // Fallback polling: keep the page fresh even if SSE drops.
        this.refreshTimer = setInterval(async () => {
            try {
                const job = await api.jobs.get(this.jobId);
                this.job = job;
                this.updateSummaryOnly();
            } catch {
                // ignore
            }
        }, 5000);
    }

    updateSummaryOnly() {
        const job = this.job;
        if (!job) return;
        // Lightweight: refresh page when job exits running state.
        if (job.status !== 'running') {
            this.loadAll().then(() => this.renderPage());
        }
    }
}
