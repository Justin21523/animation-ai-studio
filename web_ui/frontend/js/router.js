/**
 * Client-side Router - Hash-based SPA Navigation
 *
 * Provides hash-based routing for Single Page Application (SPA) navigation.
 * Supports parameterized routes, route guards, and lifecycle hooks.
 *
 * Features:
 * - Hash-based routing (#/jobs, #/jobs/123, etc.)
 * - Parameterized routes with pattern matching (/jobs/:id)
 * - Route guards (authentication, authorization)
 * - Lifecycle hooks (beforeEnter, afterEnter, beforeLeave)
 * - Component mounting/unmounting
 * - Navigation history support
 *
 * Usage Example:
 * ```javascript
 * // Register routes
 * router.register('/', () => new HomePage());
 * router.register('/jobs', () => new JobsPage());
 * router.register('/jobs/:id', (params) => new JobMonitorPage(params.id));
 *
 * // Navigate programmatically
 * router.navigate('/jobs/abc-123');
 *
 * // Access current route
 * const currentRoute = router.getCurrentRoute();
 * ```
 */

import state from './state.js';

class Router {
    constructor() {
        // Route registry: pattern -> handler
        this.routes = [];

        // Current route info
        this.currentRoute = null;
        this.currentComponent = null;

        // Navigation history
        this.history = [];
        this.maxHistorySize = 50;

        // Route guards
        this.beforeEachHooks = [];
        this.afterEachHooks = [];

        // Container element for mounting components
        this.containerSelector = '#app-content';
        this.started = false;
    }

    /**
     * Initialize router by listening to hash changes
     * @private
     */
    _init() {
        // Listen to hash changes
        window.addEventListener('hashchange', () => this._handleRouteChange());

        // Handle initial route
        window.addEventListener('DOMContentLoaded', () => this._handleRouteChange());
    }

    /**
     * Start route handling after application routes are registered.
     */
    start() {
        if (this.started) {
            return;
        }
        this.started = true;
        window.addEventListener('hashchange', () => this._handleRouteChange());
        this._handleRouteChange();
    }

    /**
     * Register a route pattern with handler
     * @param {string} pattern - Route pattern (e.g., '/', '/jobs', '/jobs/:id')
     * @param {Function} handler - Function that returns component instance or renders content
     * @param {Object} options - Route options (beforeEnter, afterEnter, meta)
     */
    register(pattern, handler, options = {}) {
        // Normalize pattern
        const normalizedPattern = pattern === '/' ? '/' : pattern.replace(/\/$/, '');

        // Convert pattern to regex
        const regex = this._patternToRegex(normalizedPattern);

        // Extract param names
        const paramNames = this._extractParamNames(normalizedPattern);

        this.routes.push({
            pattern: normalizedPattern,
            regex,
            paramNames,
            handler,
            ...options
        });
    }

    /**
     * Navigate to a route
     * @param {string} path - Route path (e.g., '/jobs/123')
     * @param {Object} options - Navigation options (replace, state)
     */
    navigate(path, options = {}) {
        const { replace = false, state: navState = {} } = options;

        // Normalize path
        const normalizedPath = path.startsWith('/') ? path : `/${path}`;

        // Update hash
        if (replace) {
            window.location.replace(`#${normalizedPath}`);
        } else {
            window.location.hash = normalizedPath;
        }

        // Store navigation state
        if (navState) {
            this._storeNavigationState(normalizedPath, navState);
        }
    }

    /**
     * Go back in history
     */
    back() {
        window.history.back();
    }

    /**
     * Go forward in history
     */
    forward() {
        window.history.forward();
    }

    /**
     * Get current route information
     * @returns {Object} - Route info with path, params, query
     */
    getCurrentRoute() {
        return this.currentRoute;
    }

    /**
     * Register global before-navigation hook
     * @param {Function} hook - Hook function (to, from) => boolean | Promise<boolean>
     */
    beforeEach(hook) {
        this.beforeEachHooks.push(hook);
    }

    /**
     * Register global after-navigation hook
     * @param {Function} hook - Hook function (to, from) => void
     */
    afterEach(hook) {
        this.afterEachHooks.push(hook);
    }

    /**
     * Handle route change (triggered by hashchange event)
     * @private
     */
    async _handleRouteChange() {
        // Parse current hash
        const hash = window.location.hash.slice(1) || '/';
        const [path, queryString] = hash.split('?');

        // Parse query parameters
        const query = this._parseQueryString(queryString);

        // Find matching route
        const match = this._findMatchingRoute(path);

        if (!match) {
            console.warn(`No route found for path: ${path}`);
            this._handle404(path);
            return;
        }

        // Build route object
        const toRoute = {
            path,
            pattern: match.route.pattern,
            params: match.params,
            query,
            meta: match.route.meta || {}
        };

        const fromRoute = this.currentRoute;

        // Run global before hooks
        const canNavigate = await this._runBeforeHooks(toRoute, fromRoute);
        if (!canNavigate) {
            console.log('Navigation cancelled by before hook');
            return;
        }

        // Run route-specific beforeEnter hook
        if (match.route.beforeEnter) {
            const canEnter = await match.route.beforeEnter(toRoute, fromRoute);
            if (!canEnter) {
                console.log('Navigation cancelled by beforeEnter hook');
                return;
            }
        }

        // Unmount current component
        if (this.currentComponent && typeof this.currentComponent.unmount === 'function') {
            await this.currentComponent.unmount();
        }

        // Execute route handler
        let component;
        try {
            component = await match.route.handler(match.params, query);
        } catch (error) {
            console.error('Error executing route handler:', error);
            this._handle500(error);
            return;
        }

        // Mount new component
        if (component) {
            if (typeof component.mount === 'function') {
                const container = document.querySelector(this.containerSelector);
                if (container) {
                    await component.mount(container);
                    this.currentComponent = component;
                } else {
                    console.error(`Container ${this.containerSelector} not found`);
                }
            } else {
                console.warn('Component does not have a mount() method');
            }
        }

        // Update current route
        this.currentRoute = toRoute;

        // Update state
        state.set('ui.currentPage', toRoute.pattern);

        // Add to history
        this._addToHistory(toRoute);

        // Run global after hooks
        await this._runAfterHooks(toRoute, fromRoute);

        // Run route-specific afterEnter hook
        if (match.route.afterEnter) {
            await match.route.afterEnter(toRoute, fromRoute);
        }
    }

    /**
     * Find route that matches the given path
     * @private
     * @param {string} path - Path to match
     * @returns {Object|null} - Match object with route and params, or null
     */
    _findMatchingRoute(path) {
        for (const route of this.routes) {
            const match = path.match(route.regex);
            if (match) {
                // Extract params
                const params = {};
                route.paramNames.forEach((name, index) => {
                    params[name] = match[index + 1];
                });

                return { route, params };
            }
        }

        return null;
    }

    /**
     * Convert route pattern to regex
     * @private
     * @param {string} pattern - Route pattern
     * @returns {RegExp} - Regex for matching
     */
    _patternToRegex(pattern) {
        // Escape special chars except `:` for params
        let regexPattern = pattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&');

        // Replace :param with capture groups
        regexPattern = regexPattern.replace(/:([^/]+)/g, '([^/]+)');

        // Exact match
        return new RegExp(`^${regexPattern}$`);
    }

    /**
     * Extract parameter names from pattern
     * @private
     * @param {string} pattern - Route pattern
     * @returns {string[]} - Array of parameter names
     */
    _extractParamNames(pattern) {
        const matches = pattern.matchAll(/:([^/]+)/g);
        return Array.from(matches, match => match[1]);
    }

    /**
     * Parse query string into object
     * @private
     * @param {string} queryString - Query string (e.g., 'foo=bar&baz=qux')
     * @returns {Object} - Query parameters object
     */
    _parseQueryString(queryString) {
        if (!queryString) {
            return {};
        }

        const params = {};
        const pairs = queryString.split('&');

        for (const pair of pairs) {
            const [key, value] = pair.split('=').map(decodeURIComponent);
            params[key] = value;
        }

        return params;
    }

    /**
     * Run all before-navigation hooks
     * @private
     * @returns {boolean} - True if navigation should proceed
     */
    async _runBeforeHooks(to, from) {
        for (const hook of this.beforeEachHooks) {
            try {
                const result = await hook(to, from);
                if (result === false) {
                    return false;
                }
            } catch (error) {
                console.error('Error in beforeEach hook:', error);
                return false;
            }
        }

        return true;
    }

    /**
     * Run all after-navigation hooks
     * @private
     */
    async _runAfterHooks(to, from) {
        for (const hook of this.afterEachHooks) {
            try {
                await hook(to, from);
            } catch (error) {
                console.error('Error in afterEach hook:', error);
            }
        }
    }

    /**
     * Add route to navigation history
     * @private
     */
    _addToHistory(route) {
        this.history.push({
            ...route,
            timestamp: Date.now()
        });

        // Limit history size
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        }
    }

    /**
     * Store navigation state for current route
     * @private
     */
    _storeNavigationState(path, navState) {
        if (!window.history.state) {
            window.history.replaceState({}, '');
        }

        window.history.state[path] = navState;
    }

    /**
     * Get navigation state for current route
     * @returns {Object} - Navigation state
     */
    getNavigationState() {
        if (!this.currentRoute) {
            return {};
        }

        return window.history.state?.[this.currentRoute.path] || {};
    }

    /**
     * Handle 404 - route not found
     * @private
     */
    _handle404(path) {
        console.error(`404: Route not found for path: ${path}`);

        // Try to find 404 handler
        const notFoundRoute = this.routes.find(r => r.pattern === '/404');
        if (notFoundRoute) {
            notFoundRoute.handler();
        } else {
            // Default 404 handling
            const container = document.querySelector(this.containerSelector);
            if (container) {
                container.innerHTML = `
                    <div class="error-page">
                        <h1>404 - Page Not Found</h1>
                        <p>The page you're looking for doesn't exist.</p>
                        <p><a href="#/">Go to Home</a></p>
                    </div>
                `;
            }
        }
    }

    /**
     * Handle 500 - internal error
     * @private
     */
    _handle500(error) {
        console.error('500: Internal error:', error);

        const container = document.querySelector(this.containerSelector);
        if (container) {
            container.innerHTML = `
                <div class="error-page">
                    <h1>500 - Internal Error</h1>
                    <p>Something went wrong while loading this page.</p>
                    <p>Error: ${error.message}</p>
                    <p><a href="#/">Go to Home</a></p>
                </div>
            `;
        }
    }

    /**
     * Get navigation history
     * @param {number} limit - Max number of entries to return
     * @returns {Array} - Recent navigation history
     */
    getHistory(limit = 10) {
        return this.history.slice(-limit);
    }
}

// Create singleton instance
const router = new Router();

// Export for ES6 modules
export default router;

// Also attach to window for non-module usage
if (typeof window !== 'undefined') {
    window.router = router;
}
