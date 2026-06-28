/**
 * Main Application Entry Point
 *
 * Initializes the Animation AI Studio Web UI application.
 * Sets up routing, loads initial state, and mounts the app.
 */

import state from './state.js';
import router from './router.js';
import api from './api/client.js';

// Import page components
import JobsPage from './pages/jobs-page.js';
import JobDetailsPage from './pages/job-details-page.js';
import ActionPage from './pages/action-page.js';
import CreativePage from './pages/creative-page.js';
import ImagePage from './pages/image-page.js';
import ResultsPage from './pages/results-page.js';
import SystemPage from './pages/system-page.js';

/**
 * Application class
 */
class App {
    constructor() {
        this.initialized = false;
    }

    /**
     * Initialize application
     */
    async init() {
        if (this.initialized) {
            console.warn('App already initialized');
            return;
        }

        console.log('Initializing Animation AI Studio Web UI...');

        try {
            // Check backend health
            await this.checkBackendHealth();

            // Setup router
            this.setupRouter();

            // Setup global error handlers
            this.setupErrorHandlers();

            // Setup navigation
            this.setupNavigation();

            // Start route handling after routes and navigation are ready
            router.start();

            // Load initial data
            await this.loadInitialData();

            // Mark as initialized
            this.initialized = true;

            console.log('App initialized successfully');

        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.showFatalError(error);
        }
    }

    /**
     * Check backend health
     */
    async checkBackendHealth() {
        try {
            const health = await api.health();
            console.log('Backend health check:', health);

            if (health.status !== 'healthy') {
                throw new Error('Backend is not healthy');
            }
        } catch (error) {
            console.error('Backend health check failed:', error);
            const hint = (typeof window !== 'undefined' && window.location && window.location.origin)
                ? window.location.origin
                : 'http://localhost:8100';
            throw new Error(`Cannot connect to backend. Please ensure the backend server is running on ${hint}`);
        }
    }

    /**
     * Setup router with page components
     */
    setupRouter() {
        // Register routes
        router.register('/', () => new JobsPage());
        router.register('/jobs', () => new JobsPage());
        router.register('/jobs/:id', (params) => new JobDetailsPage(params.id));
        router.register('/image', () => new ImagePage());
        router.register('/action', () => new ActionPage());
        router.register('/creative', () => new CreativePage());
        router.register('/results', () => new ResultsPage());
        router.register('/results/:path*', (params) => new ResultsPage(params.path));
        router.register('/system', () => new SystemPage());

        // Set container
        router.containerSelector = '#app-content';

        // Global navigation guards
        router.beforeEach((to, from) => {
            console.log(`Navigating from ${from?.path || 'null'} to ${to.path}`);
            return true;
        });

        router.afterEach((to, from) => {
            // Update active nav item
            this.updateActiveNav(to.path);

            // Scroll to top
            window.scrollTo(0, 0);
        });
    }

    /**
     * Setup global error handlers
     */
    setupErrorHandlers() {
        // Unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            event.preventDefault();
        });

        // Global errors
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
        });
    }

    /**
     * Setup navigation event listeners
     */
    setupNavigation() {
        // Sidebar navigation
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const path = item.dataset.route;
                if (path) {
                    router.navigate(path);
                }
            });
        });

        // Sidebar toggle
        const sidebarToggle = document.getElementById('sidebar-toggle');
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => {
                const sidebar = document.querySelector('.sidebar');
                if (sidebar) {
                    sidebar.classList.toggle('collapsed');
                    state.set('ui.sidebarOpen', !sidebar.classList.contains('collapsed'));
                }
            });
        }
    }

    /**
     * Update active navigation item
     */
    updateActiveNav(path) {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            const route = item.dataset.route;
            if (route === path || (route !== '/' && path.startsWith(route))) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    /**
     * Load initial data
     */
    async loadInitialData() {
        try {
            // Load jobs list
            const jobsResponse = await api.jobs.list({ limit: 10 });
            state.set('jobs.list', jobsResponse.jobs || []);

            // Load system metrics
            const metrics = await api.system.getMetrics();
            state.set('system.metrics', metrics);

        } catch (error) {
            console.warn('Failed to load initial data:', error);
            // Non-fatal, continue anyway
        }
    }

    /**
     * Show fatal error screen
     */
    showFatalError(error) {
        const appContent = document.getElementById('app-content');
        if (appContent) {
            appContent.innerHTML = `
                <div class="fatal-error">
                    <div class="error-icon">⚠️</div>
                    <h1>Failed to Initialize Application</h1>
                    <p>${error.message}</p>
                    <p class="error-details">
                        Please check the console for more details and ensure the backend server is running.
                    </p>
                    <button onclick="location.reload()" class="btn btn-primary">
                        Reload Page
                    </button>
                </div>
            `;
        }
    }
}

// Create app instance
const app = new App();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => app.init());
} else {
    app.init();
}

// Export for debugging
export default app;
window.app = app;
