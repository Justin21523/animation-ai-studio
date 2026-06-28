/**
 * State Management System - Observer Pattern Implementation
 *
 * Provides centralized state management with reactive updates for the Web UI.
 * Supports nested state paths, wildcard subscriptions, and automatic cleanup.
 *
 * Features:
 * - get(key): Retrieve state values with dot notation support (e.g., 'jobs.activeJob')
 * - set(key, value): Set state values and notify subscribers
 * - subscribe(key, callback): Subscribe to state changes with wildcard support (e.g., 'jobs.*')
 * - unsubscribe(subscriptionId): Remove specific subscription
 * - clear(): Reset all state to initial values
 *
 * Usage Example:
 * ```javascript
 * // Subscribe to job list changes
 * const subId = state.subscribe('jobs.list', (newValue, oldValue) => {
 *     console.log('Job list updated:', newValue);
 * });
 *
 * // Update job list (triggers callback)
 * state.set('jobs.list', [...newJobs]);
 *
 * // Wildcard subscription (receives all jobs.* changes)
 * state.subscribe('jobs.*', (newValue, oldValue, fullKey) => {
 *     console.log(`Job state changed at ${fullKey}`);
 * });
 *
 * // Cleanup
 * state.unsubscribe(subId);
 * ```
 */

class StateManager {
    constructor() {
        // Core state tree - immutable structure principle
        this.state = {
            // Job management state
            jobs: {
                list: [],              // Array of job objects from /api/jobs
                activeJob: null,       // Currently selected/monitored job
                filter: 'all',         // Filter: 'all' | 'running' | 'completed' | 'failed'
                sortBy: 'created_at',  // Sort field
                sortOrder: 'desc'      // 'asc' | 'desc'
            },

            // Result browser state
            results: {
                currentPath: '',             // Empty means backend default results root
                files: [],             // Files in current directory
                selectedFile: null,    // Currently selected file
                viewMode: 'grid',      // 'grid' | 'list'
                sortBy: 'name',        // 'name' | 'size' | 'modified'
                sortOrder: 'asc'
            },

            // System monitoring state
            system: {
                metrics: null,         // Current system metrics (CPU, GPU, RAM)
                history: [],           // Historical metrics for charts
                updateInterval: 5000,  // Metrics update interval (ms)
                isMonitoring: false    // Monitoring active status
            },

            // Configuration editor state
            config: {
                currentFile: null,     // Currently editing config file path
                content: '',           // File content
                isDirty: false,        // Unsaved changes flag
                validationErrors: [],  // Validation error messages
                availableConfigs: []   // List of available config files
            },

            // UI state
            ui: {
                sidebarOpen: true,     // Sidebar visibility
                currentPage: 'jobs',   // Current page/route
                theme: 'dark',         // 'light' | 'dark'
                notifications: []      // Array of notification objects
            },

            // Statistics state
            stats: {
                summary: null,         // Overall statistics
                chartData: null,       // Chart visualization data
                timeRange: '7d'        // '24h' | '7d' | '30d' | 'all'
            }
        };

        // Subscriptions registry
        // Format: { subscriptionId: { pattern, callback, isWildcard } }
        this.subscriptions = new Map();
        this.nextSubscriptionId = 1;

        // State change history for debugging
        this.history = [];
        this.maxHistorySize = 50;
    }

    /**
     * Get value from state using dot notation path
     * @param {string} key - Dot notation path (e.g., 'jobs.list' or 'system.metrics.cpu')
     * @returns {*} - State value at path, or undefined if not found
     */
    get(key) {
        if (!key) {
            return this.state;
        }

        const parts = key.split('.');
        let value = this.state;

        for (const part of parts) {
            if (value === null || value === undefined) {
                return undefined;
            }
            value = value[part];
        }

        return value;
    }

    /**
     * Set value in state and notify subscribers
     * @param {string} key - Dot notation path
     * @param {*} value - New value to set
     * @param {boolean} silent - If true, don't notify subscribers (for internal use)
     */
    set(key, value, silent = false) {
        const oldValue = this.get(key);

        // Skip if value hasn't changed (shallow comparison)
        if (oldValue === value) {
            return;
        }

        // Update state immutably
        const parts = key.split('.');
        const lastPart = parts.pop();

        // Navigate to parent object
        let parent = this.state;
        for (const part of parts) {
            if (!(part in parent)) {
                parent[part] = {};
            }
            parent = parent[part];
        }

        // Set value
        parent[lastPart] = value;

        // Record history
        this._recordHistory(key, oldValue, value);

        // Notify subscribers (unless silent)
        if (!silent) {
            this._notifySubscribers(key, value, oldValue);
        }
    }

    /**
     * Subscribe to state changes with optional wildcard support
     * @param {string} pattern - State key pattern (e.g., 'jobs.list' or 'jobs.*')
     * @param {Function} callback - Callback(newValue, oldValue, fullKey)
     * @returns {number} - Subscription ID for unsubscribing
     */
    subscribe(pattern, callback) {
        if (typeof callback !== 'function') {
            throw new Error('Callback must be a function');
        }

        const subscriptionId = this.nextSubscriptionId++;
        const isWildcard = pattern.includes('*');

        this.subscriptions.set(subscriptionId, {
            pattern,
            callback,
            isWildcard
        });

        return subscriptionId;
    }

    /**
     * Unsubscribe from state changes
     * @param {number} subscriptionId - ID returned from subscribe()
     * @returns {boolean} - True if subscription was found and removed
     */
    unsubscribe(subscriptionId) {
        return this.subscriptions.delete(subscriptionId);
    }

    /**
     * Notify all matching subscribers about a state change
     * @private
     * @param {string} key - Changed state key
     * @param {*} newValue - New value
     * @param {*} oldValue - Old value
     */
    _notifySubscribers(key, newValue, oldValue) {
        this.subscriptions.forEach(({ pattern, callback, isWildcard }) => {
            let shouldNotify = false;

            if (isWildcard) {
                // Wildcard matching: 'jobs.*' matches 'jobs.list', 'jobs.activeJob', etc.
                const patternParts = pattern.split('.');
                const keyParts = key.split('.');

                shouldNotify = this._matchWildcard(patternParts, keyParts);
            } else {
                // Exact match
                shouldNotify = pattern === key;
            }

            if (shouldNotify) {
                try {
                    callback(newValue, oldValue, key);
                } catch (error) {
                    console.error(`Error in state subscriber for key "${key}":`, error);
                }
            }
        });
    }

    /**
     * Match wildcard pattern against key
     * @private
     * @param {string[]} patternParts - Pattern split by dots
     * @param {string[]} keyParts - Key split by dots
     * @returns {boolean} - True if matches
     */
    _matchWildcard(patternParts, keyParts) {
        if (patternParts.length !== keyParts.length) {
            return false;
        }

        for (let i = 0; i < patternParts.length; i++) {
            if (patternParts[i] === '*') {
                continue; // Wildcard matches anything
            }
            if (patternParts[i] !== keyParts[i]) {
                return false;
            }
        }

        return true;
    }

    /**
     * Record state change in history for debugging
     * @private
     */
    _recordHistory(key, oldValue, newValue) {
        this.history.push({
            timestamp: Date.now(),
            key,
            oldValue,
            newValue
        });

        // Limit history size
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        }
    }

    /**
     * Clear all state to initial values
     */
    clear() {
        const oldState = { ...this.state };

        this.state = {
            jobs: { list: [], activeJob: null, filter: 'all', sortBy: 'created_at', sortOrder: 'desc' },
            results: { currentPath: '/', files: [], selectedFile: null, viewMode: 'grid', sortBy: 'name', sortOrder: 'asc' },
            system: { metrics: null, history: [], updateInterval: 5000, isMonitoring: false },
            config: { currentFile: null, content: '', isDirty: false, validationErrors: [], availableConfigs: [] },
            ui: { sidebarOpen: true, currentPage: 'jobs', theme: 'dark', notifications: [] },
            stats: { summary: null, chartData: null, timeRange: '7d' }
        };

        // Notify all subscribers about reset
        this.subscriptions.forEach(({ pattern, callback }) => {
            const newValue = this.get(pattern.replace('.*', ''));
            callback(newValue, oldState, pattern);
        });
    }

    /**
     * Get state change history (for debugging)
     * @param {number} limit - Max number of entries to return
     * @returns {Array} - Recent state changes
     */
    getHistory(limit = 10) {
        return this.history.slice(-limit);
    }

    /**
     * Batch update multiple state keys without triggering individual notifications
     * @param {Object} updates - Object with key-value pairs to update
     *
     * Example:
     * ```javascript
     * state.batchUpdate({
     *     'jobs.list': newJobList,
     *     'jobs.activeJob': activeJob,
     *     'ui.currentPage': 'jobs'
     * });
     * ```
     */
    batchUpdate(updates) {
        const changes = [];

        // Collect all changes
        for (const [key, value] of Object.entries(updates)) {
            const oldValue = this.get(key);
            if (oldValue !== value) {
                this.set(key, value, true); // Silent update
                changes.push({ key, newValue: value, oldValue });
            }
        }

        // Notify subscribers once for all changes
        for (const { key, newValue, oldValue } of changes) {
            this._notifySubscribers(key, newValue, oldValue);
        }
    }

    /**
     * Create a computed property that automatically updates when dependencies change
     * @param {string[]} dependencies - Array of state keys to watch
     * @param {Function} computeFn - Function that computes the value from dependencies
     * @param {string} targetKey - Key to store computed value
     * @returns {number} - Subscription ID for cleanup
     */
    computed(dependencies, computeFn, targetKey) {
        const compute = () => {
            const values = dependencies.map(dep => this.get(dep));
            const computed = computeFn(...values);
            this.set(targetKey, computed);
        };

        // Initial computation
        compute();

        // Subscribe to all dependencies (using wildcard for parent keys)
        const subscriptionIds = dependencies.map(dep => {
            return this.subscribe(dep, compute);
        });

        // Return composite subscription ID
        return subscriptionIds;
    }
}

// Create singleton instance
const state = new StateManager();

// Export for ES6 modules
export default state;

// Also attach to window for non-module usage
if (typeof window !== 'undefined') {
    window.state = state;
}
