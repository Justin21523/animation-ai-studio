/**
 * Job List Component
 *
 * Displays a filterable, sortable list of all jobs with their status and progress.
 * Supports pagination, real-time updates, and navigation to job details.
 *
 * Features:
 * - Job list with status badges (pending, running, completed, failed)
 * - Filtering by status (all, running, completed, failed)
 * - Sorting by created_at, film_name, status
 * - Real-time status updates via state subscriptions
 * - Pagination support
 * - Click to navigate to job details
 *
 * Usage:
 * ```javascript
 * const jobList = new JobList();
 * await jobList.mount(container);
 * ```
 */

import BaseComponent from './base-component.js';
import api from '../api/client.js';
import router from '../router.js';

export default class JobList extends BaseComponent {
    constructor() {
        super();

        // Component state
        this.jobs = [];
        this.loading = false;
        this.error = null;

        // Filter/sort settings
        this.currentFilter = 'all';  // 'all', 'running', 'completed', 'failed'
        this.currentSort = 'created_at';
        this.currentSortOrder = 'desc';

        // Pagination
        this.limit = 20;
        this.offset = 0;
        this.total = 0;

        // Auto-refresh
        this.refreshInterval = null;
        this.refreshIntervalMs = 5000; // 5 seconds
    }

    async onMount() {
        // Subscribe to state changes
        this.subscribeState('jobs.list', (jobs) => {
            this.jobs = jobs || [];
            this.renderJobList();
        });

        this.subscribeState('jobs.filter', (filter) => {
            this.currentFilter = filter;
            this.loadJobs();
        });

        // Initial load
        await this.loadJobs();

        // Setup event listeners
        this.setupEventListeners();

        // Start auto-refresh for running jobs
        this.startAutoRefresh();
    }

    async onUnmount() {
        // Stop auto-refresh
        this.stopAutoRefresh();
    }

    /**
     * Load jobs from API
     */
    async loadJobs() {
        this.loading = true;
        this.error = null;
        this.showLoading('Loading jobs...');

        try {
            const params = {
                limit: this.limit,
                offset: this.offset
            };

            // Add status filter if not 'all'
            if (this.currentFilter !== 'all') {
                params.status = this.currentFilter;
            }

            const response = await api.jobs.list(params);
            const jobs = Array.isArray(response) ? response : (response.jobs || []);

            this.jobs = jobs;
            this.total = Array.isArray(response) ? jobs.length : (response.total || jobs.length);
            this.loading = false;

            // Update state
            this.setState('jobs.list', this.jobs);

            // Render
            this.renderJobList();

        } catch (error) {
            console.error('Failed to load jobs:', error);
            this.error = error.message;
            this.loading = false;
            this.showError('Failed to load jobs', error);
        } finally {
            this.loading = false;
        }
    }

    /**
     * Render job list
     */
    renderJobList() {
        if (this.loading) {
            return; // Keep showing loading indicator
        }

        if (this.error) {
            return; // Keep showing error
        }

        if (this.jobs.length === 0) {
            this.showEmpty('No jobs found', '📋');
            return;
        }

        // Sort jobs
        const sortedJobs = this.sortJobs(this.jobs);

        const html = `
            <div class="job-list-container">
                <!-- Header -->
                <div class="job-list-header">
                    <h2>Jobs</h2>
                    <div class="job-list-controls">
                        ${this.renderFilterButtons()}
                        ${this.renderSortDropdown()}
                        <button class="btn btn-primary" id="refresh-btn">
                            <span>🔄</span> Refresh
                        </button>
                    </div>
                </div>

                <!-- Job List -->
                <div class="job-list">
                    ${sortedJobs.map(job => this.renderJobCard(job)).join('')}
                </div>

                <!-- Pagination -->
                ${this.renderPagination()}
            </div>
        `;

        this.render(html);
        this.setupEventListeners();
    }

    /**
     * Render filter buttons
     */
    renderFilterButtons() {
        const filters = [
            { value: 'all', label: 'All', icon: '📋' },
            { value: 'running', label: 'Running', icon: '▶️' },
            { value: 'completed', label: 'Completed', icon: '✅' },
            { value: 'failed', label: 'Failed', icon: '❌' }
        ];

        return `
            <div class="filter-buttons">
                ${filters.map(filter => `
                    <button
                        class="btn btn-filter ${this.currentFilter === filter.value ? 'active' : ''}"
                        data-filter="${filter.value}"
                    >
                        <span>${filter.icon}</span> ${filter.label}
                    </button>
                `).join('')}
            </div>
        `;
    }

    /**
     * Render sort dropdown
     */
    renderSortDropdown() {
        return `
            <select id="sort-select" class="sort-select">
                <option value="created_at" ${this.currentSort === 'created_at' ? 'selected' : ''}>
                    Sort by: Created Date
                </option>
                <option value="film_name" ${this.currentSort === 'film_name' ? 'selected' : ''}>
                    Sort by: Film Name
                </option>
                <option value="status" ${this.currentSort === 'status' ? 'selected' : ''}>
                    Sort by: Status
                </option>
            </select>
            <button id="sort-order-btn" class="btn btn-icon">
                ${this.currentSortOrder === 'desc' ? '⬇️' : '⬆️'}
            </button>
        `;
    }

    /**
     * Render individual job card
     */
    renderJobCard(job) {
        const statusBadge = this.getStatusBadge(job.status);
        const progressBar = job.status === 'running'
            ? this.renderProgressBar(job.progress_percent || 0)
            : '';

        return `
            <div class="job-card" data-job-id="${job.job_id}">
                <div class="job-card-header">
                    <div class="job-info">
                        <h3 class="job-name">${this.escapeHtml(job.film_name)}</h3>
                        <p class="job-id">ID: ${job.job_id.substring(0, 8)}...</p>
                    </div>
                    ${statusBadge}
                </div>

                <div class="job-card-body">
                    <div class="job-details">
                        <div class="job-detail">
                            <span class="label">Pipeline:</span>
                            <span class="value">${this.escapeHtml(job.pipeline_type)}</span>
                        </div>
                        <div class="job-detail">
                            <span class="label">Created:</span>
                            <span class="value">${this.formatDate(job.created_at)}</span>
                        </div>
                        ${job.current_stage ? `
                            <div class="job-detail">
                                <span class="label">Stage:</span>
                                <span class="value">${this.escapeHtml(job.current_stage)}</span>
                            </div>
                        ` : ''}
                        ${job.duration ? `
                            <div class="job-detail">
                                <span class="label">Duration:</span>
                                <span class="value">${this.formatDuration(job.duration)}</span>
                            </div>
                        ` : ''}
                    </div>

                    ${progressBar}

                    ${job.error_message ? `
                        <div class="job-error">
                            <span class="error-icon">⚠️</span>
                            <span class="error-message">${this.escapeHtml(job.error_message)}</span>
                        </div>
                    ` : ''}
                </div>

                <div class="job-card-footer">
                    <button class="btn btn-sm btn-view" data-job-id="${job.job_id}">
                        View Details
                    </button>
                    ${job.status === 'running' ? `
                        <button class="btn btn-sm btn-cancel" data-job-id="${job.job_id}">
                            Cancel
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Get status badge HTML
     */
    getStatusBadge(status) {
        const badges = {
            pending: { icon: '⏳', label: 'Pending', class: 'badge-pending' },
            running: { icon: '▶️', label: 'Running', class: 'badge-running' },
            completed: { icon: '✅', label: 'Completed', class: 'badge-success' },
            failed: { icon: '❌', label: 'Failed', class: 'badge-error' }
        };

        const badge = badges[status] || badges.pending;

        return `
            <span class="status-badge ${badge.class}">
                <span>${badge.icon}</span>
                ${badge.label}
            </span>
        `;
    }

    /**
     * Render progress bar
     */
    renderProgressBar(progress) {
        return `
            <div class="progress-bar-container">
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: ${progress}%"></div>
                </div>
                <span class="progress-text">${progress.toFixed(1)}%</span>
            </div>
        `;
    }

    /**
     * Render pagination controls
     */
    renderPagination() {
        const totalPages = Math.ceil(this.total / this.limit);
        const currentPage = Math.floor(this.offset / this.limit) + 1;

        if (totalPages <= 1) {
            return '';
        }

        return `
            <div class="pagination">
                <button
                    class="btn btn-sm"
                    id="prev-page"
                    ${currentPage === 1 ? 'disabled' : ''}
                >
                    Previous
                </button>
                <span class="page-info">
                    Page ${currentPage} of ${totalPages}
                </span>
                <button
                    class="btn btn-sm"
                    id="next-page"
                    ${currentPage === totalPages ? 'disabled' : ''}
                >
                    Next
                </button>
            </div>
        `;
    }

    /**
     * Sort jobs array
     */
    sortJobs(jobs) {
        const sorted = [...jobs];

        sorted.sort((a, b) => {
            let aValue, bValue;

            switch (this.currentSort) {
                case 'created_at':
                    aValue = Number(a.created_at) || new Date(a.created_at).getTime();
                    bValue = Number(b.created_at) || new Date(b.created_at).getTime();
                    break;
                case 'film_name':
                    aValue = a.film_name.toLowerCase();
                    bValue = b.film_name.toLowerCase();
                    break;
                case 'status':
                    aValue = a.status;
                    bValue = b.status;
                    break;
                default:
                    return 0;
            }

            if (aValue < bValue) return this.currentSortOrder === 'asc' ? -1 : 1;
            if (aValue > bValue) return this.currentSortOrder === 'asc' ? 1 : -1;
            return 0;
        });

        return sorted;
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Filter buttons
        this.delegateEvent('.btn-filter', 'click', (e) => {
            const filter = e.target.closest('.btn-filter').dataset.filter;
            this.currentFilter = filter;
            this.offset = 0; // Reset to first page
            this.setState('jobs.filter', filter);
        });

        // Sort dropdown
        const sortSelect = this.$('#sort-select');
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                this.currentSort = e.target.value;
                this.renderJobList();
            });
        }

        // Sort order button
        const sortOrderBtn = this.$('#sort-order-btn');
        if (sortOrderBtn) {
            sortOrderBtn.addEventListener('click', () => {
                this.currentSortOrder = this.currentSortOrder === 'asc' ? 'desc' : 'asc';
                this.renderJobList();
            });
        }

        // Refresh button
        const refreshBtn = this.$('#refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadJobs();
            });
        }

        // View details buttons
        this.delegateEvent('.btn-view', 'click', (e) => {
            const jobId = e.target.dataset.jobId;
            router.navigate(`/jobs/${jobId}`);
        });

        // Cancel buttons
        this.delegateEvent('.btn-cancel', 'click', async (e) => {
            const jobId = e.target.dataset.jobId;
            await this.cancelJob(jobId);
        });

        // Pagination buttons
        const prevBtn = this.$('#prev-page');
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                this.offset = Math.max(0, this.offset - this.limit);
                this.loadJobs();
            });
        }

        const nextBtn = this.$('#next-page');
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                this.offset += this.limit;
                this.loadJobs();
            });
        }
    }

    /**
     * Cancel a job
     */
    async cancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) {
            return;
        }

        try {
            await api.jobs.cancel(jobId);

            // Refresh job list
            await this.loadJobs();

            alert('Job cancelled successfully');
        } catch (error) {
            console.error('Failed to cancel job:', error);
            alert(`Failed to cancel job: ${error.message}`);
        }
    }

    /**
     * Start auto-refresh for running jobs
     */
    startAutoRefresh() {
        this.refreshInterval = setInterval(() => {
            // Only refresh if there are running jobs
            const hasRunningJobs = this.jobs.some(job => job.status === 'running');
            if (hasRunningJobs) {
                this.loadJobs();
            }
        }, this.refreshIntervalMs);
    }

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
}
