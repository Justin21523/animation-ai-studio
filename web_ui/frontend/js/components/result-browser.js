/**
 * Result Browser Component
 *
 * File browser for exploring job output directories with preview capabilities.
 * Supports navigation, sorting, file downloads, and metadata viewing.
 *
 * Features:
 * - Directory tree navigation with breadcrumbs
 * - Grid and list view modes
 * - File type icons and previews
 * - Sort by name, size, modified date
 * - File download functionality
 * - JSON file preview
 * - Image thumbnail previews
 *
 * Usage:
 * ```javascript
 * const browser = new ResultBrowser();
 * await browser.mount(container);
 * ```
 */

import BaseComponent from './base-component.js';
import api from '../api/client.js';

export default class ResultBrowser extends BaseComponent {
    constructor(initialPath = '') {
        super();

        // Component state
        this.currentPath = initialPath;
        this.files = [];
        this.loading = false;
        this.error = null;

        // View settings
        this.viewMode = 'grid';  // 'grid' or 'list'
        this.sortBy = 'name';    // 'name', 'size', 'modified'
        this.sortOrder = 'asc';  // 'asc' or 'desc'

        // File preview
        this.selectedFile = null;
        this.previewData = null;
    }

    async onMount() {
        // Respect any previously selected path (e.g. deep-link from Action page).
        const existingPath = this.getState('results.currentPath');
        if (existingPath) {
            this.currentPath = existingPath;
        }

        // Subscribe to state changes
        this.subscribeState('results.currentPath', (path) => {
            this.currentPath = path;
            this.loadFiles();
        });

        this.subscribeState('results.viewMode', (mode) => {
            this.viewMode = mode;
            this.renderFiles();
        });

        // Initial load
        await this.loadFiles();
    }

    /**
     * Load files from current directory
     */
    async loadFiles() {
        this.loading = true;
        this.error = null;
        this.showLoading('Loading files...');

        try {
            const response = await api.results.list(this.currentPath);

            this.files = response.files || [];
            this.loading = false;

            // Update state
            this.setState('results.files', this.files);
            this.setState('results.currentPath', this.currentPath);

            // Render
            this.renderFiles();

        } catch (error) {
            console.error('Failed to load files:', error);
            this.error = error.message;
            this.loading = false;
            this.showError('Failed to load files', error);
        } finally {
            this.loading = false;
        }
    }

    /**
     * Render file browser
     */
    renderFiles() {
        if (this.loading) {
            return; // Keep showing loading indicator
        }

        if (this.error) {
            return; // Keep showing error
        }

        if (this.files.length === 0) {
            this.showEmpty('This directory is empty', '📁');
            return;
        }

        // Sort files
        const sortedFiles = this.sortFiles(this.files);

        const html = `
            <div class="result-browser-container">
                <!-- Header -->
                <div class="browser-header">
                    <div class="breadcrumb">
                        ${this.renderBreadcrumb()}
                    </div>
                    <div class="browser-controls">
                        ${this.renderViewModeButtons()}
                        ${this.renderSortDropdown()}
                    </div>
                </div>

                <!-- File Grid/List -->
                <div class="file-container ${this.viewMode}-view">
                    ${sortedFiles.map(file => this.renderFileItem(file)).join('')}
                </div>

                <!-- File Preview Modal (if file selected) -->
                ${this.selectedFile ? this.renderPreviewModal() : ''}
            </div>
        `;

        this.render(html);
        this.setupEventListeners();
    }

    /**
     * Render breadcrumb navigation
     */
    renderBreadcrumb() {
        if (!this.currentPath) {
            return '<span class="breadcrumb-item"><a href="#" data-path="">Results</a></span>';
        }

        const parts = this.currentPath.split('/').filter(p => p);
        const breadcrumbs = [{ name: 'Root', path: '/' }];

        let currentPath = '';
        for (const part of parts) {
            currentPath += `/${part}`;
            breadcrumbs.push({ name: part, path: currentPath });
        }

        return breadcrumbs.map((crumb, index) => `
            <span class="breadcrumb-item">
                ${index > 0 ? '<span class="separator">/</span>' : ''}
                <a href="#" data-path="${crumb.path}">${this.escapeHtml(crumb.name)}</a>
            </span>
        `).join('');
    }

    /**
     * Render view mode buttons
     */
    renderViewModeButtons() {
        return `
            <div class="view-mode-buttons">
                <button
                    class="btn btn-icon ${this.viewMode === 'grid' ? 'active' : ''}"
                    data-view="grid"
                    title="Grid View"
                >
                    ▦
                </button>
                <button
                    class="btn btn-icon ${this.viewMode === 'list' ? 'active' : ''}"
                    data-view="list"
                    title="List View"
                >
                    ☰
                </button>
            </div>
        `;
    }

    /**
     * Render sort dropdown
     */
    renderSortDropdown() {
        return `
            <select id="sort-select" class="sort-select">
                <option value="name" ${this.sortBy === 'name' ? 'selected' : ''}>Sort by: Name</option>
                <option value="size" ${this.sortBy === 'size' ? 'selected' : ''}>Sort by: Size</option>
                <option value="modified" ${this.sortBy === 'modified' ? 'selected' : ''}>Sort by: Modified</option>
            </select>
            <button id="sort-order-btn" class="btn btn-icon">
                ${this.sortOrder === 'desc' ? '⬇️' : '⬆️'}
            </button>
        `;
    }

    /**
     * Render individual file/folder item
     */
    renderFileItem(file) {
        const icon = this.getFileIcon(file);
        const isDirectory = file.type === 'directory';

        if (this.viewMode === 'grid') {
            return `
                <div class="file-item" data-path="${file.path}" data-type="${file.type}">
                    <div class="file-icon">${icon}</div>
                    <div class="file-name" title="${this.escapeHtml(file.name)}">
                        ${this.escapeHtml(file.name)}
                    </div>
                    ${!isDirectory ? `<div class="file-size">${this.formatFileSize(file.size || 0)}</div>` : ''}
                </div>
            `;
        } else {
            return `
                <div class="file-item-list" data-path="${file.path}" data-type="${file.type}">
                    <div class="file-icon">${icon}</div>
                    <div class="file-info">
                        <div class="file-name">${this.escapeHtml(file.name)}</div>
                        <div class="file-meta">
                            ${!isDirectory ? `<span>${this.formatFileSize(file.size || 0)}</span>` : ''}
                            ${file.modified ? `<span>${this.formatDate(file.modified)}</span>` : ''}
                        </div>
                    </div>
                    ${!isDirectory ? `
                        <div class="file-actions">
                            <button class="btn btn-sm btn-download" data-path="${file.path}">
                                Download
                            </button>
                            ${this.isPreviewable(file) ? `
                                <button class="btn btn-sm btn-preview" data-path="${file.path}">
                                    Preview
                                </button>
                            ` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }
    }

    /**
     * Get file type icon
     */
    getFileIcon(file) {
        if (file.type === 'directory') {
            return '📁';
        }

        const ext = file.name.split('.').pop().toLowerCase();

        const iconMap = {
            // Images
            'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'webp': '🖼️',
            // Videos
            'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬', 'webm': '🎬',
            // Audio
            'mp3': '🎵', 'wav': '🎵', 'flac': '🎵', 'aac': '🎵', 'ogg': '🎵',
            // Documents
            'txt': '📄', 'md': '📄', 'pdf': '📕',
            // Data
            'json': '📊', 'csv': '📊', 'xml': '📊', 'yaml': '📊', 'yml': '📊',
            // Code
            'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨', 'sh': '⚙️',
            // Archives
            'zip': '📦', 'tar': '📦', 'gz': '📦', '7z': '📦', 'rar': '📦'
        };

        return iconMap[ext] || '📄';
    }

    /**
     * Check if file is previewable
     */
    isPreviewable(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        const previewable = ['json', 'txt', 'md', 'csv', 'yaml', 'yml', 'log'];
        return previewable.includes(ext);
    }

    /**
     * Render preview modal
     */
    renderPreviewModal() {
        if (!this.selectedFile || !this.previewData) {
            return '';
        }

        return `
            <div class="preview-modal" id="preview-modal">
                <div class="preview-modal-content">
                    <div class="preview-header">
                        <h3>${this.escapeHtml(this.selectedFile.name)}</h3>
                        <button class="btn btn-close" id="close-preview">&times;</button>
                    </div>
                    <div class="preview-body">
                        <pre><code>${this.escapeHtml(this.previewData)}</code></pre>
                    </div>
                    <div class="preview-footer">
                        <button class="btn btn-download" data-path="${this.selectedFile.path}">
                            Download
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Sort files array
     */
    sortFiles(files) {
        const sorted = [...files];

        // Always put directories first
        sorted.sort((a, b) => {
            if (a.type === 'directory' && b.type !== 'directory') return -1;
            if (a.type !== 'directory' && b.type === 'directory') return 1;

            // Then sort by selected criterion
            let aValue, bValue;

            switch (this.sortBy) {
                case 'name':
                    aValue = a.name.toLowerCase();
                    bValue = b.name.toLowerCase();
                    break;
                case 'size':
                    aValue = a.size || 0;
                    bValue = b.size || 0;
                    break;
                case 'modified':
                    aValue = a.modified ? new Date(a.modified).getTime() : 0;
                    bValue = b.modified ? new Date(b.modified).getTime() : 0;
                    break;
                default:
                    return 0;
            }

            if (aValue < bValue) return this.sortOrder === 'asc' ? -1 : 1;
            if (aValue > bValue) return this.sortOrder === 'asc' ? 1 : -1;
            return 0;
        });

        return sorted;
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Breadcrumb navigation
        this.delegateEvent('.breadcrumb-item a', 'click', (e) => {
            e.preventDefault();
            const path = e.target.dataset.path;
            this.navigateTo(path);
        });

        // View mode buttons
        this.delegateEvent('[data-view]', 'click', (e) => {
            const viewMode = e.target.closest('[data-view]').dataset.view;
            this.viewMode = viewMode;
            this.setState('results.viewMode', viewMode);
        });

        // Sort dropdown
        const sortSelect = this.$('#sort-select');
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                this.sortBy = e.target.value;
                this.renderFiles();
            });
        }

        // Sort order button
        const sortOrderBtn = this.$('#sort-order-btn');
        if (sortOrderBtn) {
            sortOrderBtn.addEventListener('click', () => {
                this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
                this.renderFiles();
            });
        }

        // File/folder click
        this.delegateEvent('.file-item, .file-item-list', 'click', (e) => {
            // Ignore if clicking on buttons
            if (e.target.closest('button')) {
                return;
            }

            const item = e.target.closest('[data-path]');
            const path = item.dataset.path;
            const type = item.dataset.type;

            if (type === 'directory') {
                this.navigateTo(path);
            }
        });

        // Download buttons
        this.delegateEvent('.btn-download', 'click', async (e) => {
            e.stopPropagation();
            const path = e.target.dataset.path;
            await this.downloadFile(path);
        });

        // Preview buttons
        this.delegateEvent('.btn-preview', 'click', async (e) => {
            e.stopPropagation();
            const path = e.target.dataset.path;
            await this.previewFile(path);
        });

        // Close preview modal
        const closeBtn = this.$('#close-preview');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.selectedFile = null;
                this.previewData = null;
                this.renderFiles();
            });
        }

        // Close modal on background click
        const modal = this.$('#preview-modal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.selectedFile = null;
                    this.previewData = null;
                    this.renderFiles();
                }
            });
        }
    }

    /**
     * Navigate to directory
     */
    async navigateTo(path) {
        this.currentPath = path;
        this.setState('results.currentPath', path);
    }

    /**
     * Download file
     */
    async downloadFile(path) {
        try {
            const blob = await api.results.download(path);
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = path.split('/').pop();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Failed to download file:', error);
            alert(`Failed to download file: ${error.message}`);
        }
    }

    /**
     * Preview file
     */
    async previewFile(path) {
        try {
            const file = this.files.find(f => f.path === path);
            if (!file) return;

            // Download file as blob
            const blob = await api.results.download(path);
            const text = await blob.text();

            this.selectedFile = file;
            this.previewData = text;

            this.renderFiles();

        } catch (error) {
            console.error('Failed to preview file:', error);
            alert(`Failed to preview file: ${error.message}`);
        }
    }
}
