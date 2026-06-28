/**
 * Base Component Class
 *
 * Abstract base class for all UI components in the Web UI.
 * Provides lifecycle management, state subscription, and common utilities.
 *
 * Features:
 * - Lifecycle methods: mount, unmount, render
 * - Automatic state subscription cleanup
 * - DOM query helpers
 * - Event handling utilities
 *
 * Usage Example:
 * ```javascript
 * class MyComponent extends BaseComponent {
 *     constructor() {
 *         super();
 *         this.title = 'My Component';
 *     }
 *
 *     async onMount() {
 *         // Subscribe to state changes
 *         this.subscribeState('jobs.list', (jobs) => {
 *             this.renderJobList(jobs);
 *         });
 *
 *         // Render initial content
 *         this.render(this.getTemplate());
 *     }
 *
 *     getTemplate() {
 *         return `<div><h1>${this.title}</h1></div>`;
 *     }
 *
 *     onUnmount() {
 *         // Cleanup code (subscriptions auto-cleaned)
 *     }
 * }
 * ```
 */

import state from '../state.js';

export default class BaseComponent {
    constructor() {
        // Container element where component is mounted
        this.container = null;

        // State subscriptions (auto-cleanup on unmount)
        this.subscriptions = [];

        // Event listeners (auto-cleanup on unmount)
        this.eventListeners = [];

        // Component lifecycle state
        this.isMounted = false;
    }

    /**
     * Mount component to DOM container
     * @param {HTMLElement} container - Container element
     */
    async mount(container) {
        if (this.isMounted) {
            console.warn('Component already mounted');
            return;
        }

        this.container = container;
        this.isMounted = true;

        // Call lifecycle hook
        try {
            await this.onMount();
        } catch (error) {
            console.error('Error in onMount:', error);
        }
    }

    /**
     * Unmount component from DOM
     */
    async unmount() {
        if (!this.isMounted) {
            return;
        }

        // Call lifecycle hook
        try {
            await this.onUnmount();
        } catch (error) {
            console.error('Error in onUnmount:', error);
        }

        // Cleanup subscriptions
        this.subscriptions.forEach(subId => state.unsubscribe(subId));
        this.subscriptions = [];

        // Cleanup event listeners
        this.eventListeners.forEach(({ element, event, handler }) => {
            element.removeEventListener(event, handler);
        });
        this.eventListeners = [];

        // Clear container
        if (this.container) {
            this.container.innerHTML = '';
            this.container = null;
        }

        this.isMounted = false;
    }

    /**
     * Render HTML content to container
     * @param {string} html - HTML content
     */
    render(html) {
        if (!this.container) {
            console.error('Cannot render: component not mounted');
            return;
        }

        this.container.innerHTML = html;
    }

    /**
     * Lifecycle hook: called when component is mounted
     * Override in subclasses
     */
    async onMount() {
        // Override in subclass
    }

    /**
     * Lifecycle hook: called when component is unmounted
     * Override in subclasses
     */
    async onUnmount() {
        // Override in subclass
    }

    /**
     * Subscribe to state changes with automatic cleanup
     * @param {string} pattern - State key pattern
     * @param {Function} callback - Callback function
     */
    subscribeState(pattern, callback) {
        const subId = state.subscribe(pattern, callback);
        this.subscriptions.push(subId);
        return subId;
    }

    /**
     * Get state value
     * @param {string} key - State key
     * @returns {*} - State value
     */
    getState(key) {
        return state.get(key);
    }

    /**
     * Set state value
     * @param {string} key - State key
     * @param {*} value - New value
     */
    setState(key, value) {
        state.set(key, value);
    }

    /**
     * Query selector within component container
     * @param {string} selector - CSS selector
     * @returns {HTMLElement|null} - Element or null
     */
    $(selector) {
        if (!this.container) {
            return null;
        }
        return this.container.querySelector(selector);
    }

    /**
     * Query selector all within component container
     * @param {string} selector - CSS selector
     * @returns {NodeList} - List of elements
     */
    $$(selector) {
        if (!this.container) {
            return [];
        }
        return this.container.querySelectorAll(selector);
    }

    /**
     * Add event listener with automatic cleanup
     * @param {HTMLElement|string} elementOrSelector - Element or selector
     * @param {string} event - Event name
     * @param {Function} handler - Event handler
     * @param {Object} options - Event listener options
     */
    addEventListener(elementOrSelector, event, handler, options = {}) {
        let element;

        if (typeof elementOrSelector === 'string') {
            element = this.$(elementOrSelector);
            if (!element) {
                console.warn(`Element not found: ${elementOrSelector}`);
                return;
            }
        } else {
            element = elementOrSelector;
        }

        element.addEventListener(event, handler, options);
        this.eventListeners.push({ element, event, handler });
    }

    /**
     * Delegate event to dynamically created elements
     * @param {string} selector - CSS selector for target elements
     * @param {string} event - Event name
     * @param {Function} handler - Event handler
     */
    delegateEvent(selector, event, handler) {
        const delegateHandler = (e) => {
            const target = e.target.closest(selector);
            if (target && this.container.contains(target)) {
                handler.call(target, e);
            }
        };

        this.addEventListener(this.container, event, delegateHandler, true);
    }

    /**
     * Create element from HTML string
     * @param {string} html - HTML string
     * @returns {HTMLElement} - Created element
     */
    createElement(html) {
        const template = document.createElement('template');
        template.innerHTML = html.trim();
        return template.content.firstChild;
    }

    /**
     * Show loading indicator
     * @param {string} message - Loading message
     */
    showLoading(message = 'Loading...') {
        this.render(`
            <div class="loading-container">
                <div class="spinner"></div>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `);
    }

    /**
     * Show error message
     * @param {string} message - Error message
     * @param {Error} error - Error object (optional)
     */
    showError(message, error = null) {
        const errorDetails = error ? `<pre>${this.escapeHtml(error.stack || error.message)}</pre>` : '';

        this.render(`
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <h2>Error</h2>
                <p>${this.escapeHtml(message)}</p>
                ${errorDetails}
            </div>
        `);
    }

    /**
     * Show empty state message
     * @param {string} message - Empty state message
     * @param {string} icon - Icon (emoji or HTML)
     */
    showEmpty(message, icon = '📭') {
        this.render(`
            <div class="empty-container">
                <div class="empty-icon">${icon}</div>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `);
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} - Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Format date to human-readable string
     * @param {Date|string|number} date - Date object, ISO string, or timestamp
     * @returns {string} - Formatted date string
     */
    formatDate(date) {
        const normalized = typeof date === 'number' && date > 0 && date < 1000000000000
            ? date * 1000
            : date;
        const d = new Date(normalized);
        return d.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    /**
     * Format duration in seconds to human-readable string
     * @param {number} seconds - Duration in seconds
     * @returns {string} - Formatted duration (e.g., "2h 34m 15s")
     */
    formatDuration(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        }

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.round(seconds % 60);

        const parts = [];
        if (hours > 0) parts.push(`${hours}h`);
        if (minutes > 0) parts.push(`${minutes}m`);
        if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

        return parts.join(' ');
    }

    /**
     * Format file size to human-readable string
     * @param {number} bytes - File size in bytes
     * @returns {string} - Formatted size (e.g., "2.5 MB")
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';

        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        const k = 1024;
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return `${(bytes / Math.pow(k, i)).toFixed(2)} ${units[i]}`;
    }

    /**
     * Debounce function calls
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in ms
     * @returns {Function} - Debounced function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Throttle function calls
     * @param {Function} func - Function to throttle
     * @param {number} limit - Limit time in ms
     * @returns {Function} - Throttled function
     */
    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func(...args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}
