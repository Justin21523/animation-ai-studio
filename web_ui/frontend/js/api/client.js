/**
 * API Client - Backend Communication Layer
 *
 * Provides a unified interface for communicating with the FastAPI backend.
 * Handles HTTP requests, error handling, and response parsing.
 *
 * Features:
 * - RESTful API methods (GET, POST, DELETE)
 * - Server-Sent Events (SSE) streaming
 * - Automatic error handling
 * - Request/response interceptors
 * - Base URL configuration
 *
 * Usage Example:
 * ```javascript
 * // Get jobs list
 * const jobs = await api.jobs.list({ status: 'running' });
 *
 * // Submit new job
 * const jobId = await api.jobs.create({
 *     film_name: 'test',
 *     pipeline_type: 'cpu_only',
 *     input_video_path: '/path/to/video.mp4'
 * });
 *
 * // Stream job progress
 * api.jobs.streamProgress(jobId, (event) => {
 *     console.log('Progress:', event.data);
 * });
 * ```
 */

class APIClient {
    constructor(baseURL = null) {
        const defaultURL = (typeof window !== 'undefined' && window.location && window.location.origin)
            ? window.location.origin
            : 'http://localhost:8100';
        this.baseURL = baseURL || defaultURL;

        // Request/response interceptors
        this.requestInterceptors = [];
        this.responseInterceptors = [];

        // Active SSE connections
        this.sseConnections = new Map();
    }

    /**
     * Generic HTTP request method
     * @private
     * @param {string} method - HTTP method
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<any>} - Response data
     */
    async _request(method, endpoint, options = {}) {
        const { body, params, headers = {} } = options;

        // Build URL with query params
        const url = new URL(`${this.baseURL}/api${endpoint}`);
        if (params) {
            Object.keys(params).forEach(key => {
                if (params[key] !== null && params[key] !== undefined) {
                    url.searchParams.append(key, params[key]);
                }
            });
        }

        // Build request config
        const config = {
            method,
            headers: {
                ...headers
            }
        };

        const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;
        if (!isFormData) {
            config.headers = {
                'Content-Type': 'application/json',
                ...config.headers
            };
        }

        if (body !== null && body !== undefined) {
            config.body = isFormData ? body : JSON.stringify(body);
        }

        // Let the browser set multipart boundaries for FormData.
        if (isFormData && config.headers && config.headers['Content-Type']) {
            delete config.headers['Content-Type'];
        }

        // Run request interceptors
        for (const interceptor of this.requestInterceptors) {
            await interceptor(config);
        }

        try {
            const response = await fetch(url, config);

            // Run response interceptors
            for (const interceptor of this.responseInterceptors) {
                await interceptor(response);
            }

            // Handle non-2xx responses
            if (!response.ok) {
                const error = await response.json().catch(() => ({
                    detail: response.statusText
                }));
                throw new APIError(error.detail || 'Request failed', response.status, error);
            }

            // Parse response
            const contentType = response.headers.get('Content-Type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            if (error instanceof APIError) {
                throw error;
            }
            throw new APIError(`Network error: ${error.message}`, 0, error);
        }
    }

    /**
     * GET request
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {Promise<any>} - Response data
     */
    async get(endpoint, params = {}) {
        return this._request('GET', endpoint, { params });
    }

    /**
     * POST request
     * @param {string} endpoint - API endpoint
     * @param {Object} body - Request body
     * @returns {Promise<any>} - Response data
     */
    async post(endpoint, body = {}) {
        return this._request('POST', endpoint, { body });
    }

    /**
     * POST multipart/form-data request
     * @param {string} endpoint - API endpoint
     * @param {FormData} formData - FormData payload
     * @returns {Promise<any>} - Response data
     */
    async postForm(endpoint, formData) {
        return this._request('POST', endpoint, { body: formData });
    }

    /**
     * DELETE request
     * @param {string} endpoint - API endpoint
     * @returns {Promise<any>} - Response data
     */
    async delete(endpoint) {
        return this._request('DELETE', endpoint);
    }

    /**
     * Create SSE connection for streaming
     * @param {string} endpoint - SSE endpoint
     * @param {Object} handlers - Event handlers
     * @returns {EventSource} - EventSource instance
     */
    createSSE(endpoint, handlers = {}) {
        const url = `${this.baseURL}/api${endpoint}`;
        const eventSource = new EventSource(url);

        // Setup event handlers
        for (const [event, handler] of Object.entries(handlers)) {
            eventSource.addEventListener(event, handler);
        }

        // Store connection
        this.sseConnections.set(endpoint, eventSource);

        return eventSource;
    }

    /**
     * Close SSE connection
     * @param {string} endpoint - SSE endpoint
     */
    closeSSE(endpoint) {
        const eventSource = this.sseConnections.get(endpoint);
        if (eventSource) {
            eventSource.close();
            this.sseConnections.delete(endpoint);
        }
    }

    /**
     * Close all SSE connections
     */
    closeAllSSE() {
        this.sseConnections.forEach(eventSource => eventSource.close());
        this.sseConnections.clear();
    }

    // ========================================
    // Job API Methods
    // ========================================

    jobs = {
        /**
         * List jobs with optional filtering
         * @param {Object} params - Query parameters (status, limit, offset)
         * @returns {Promise<Object>} - Jobs list response
         */
        list: async (params = {}) => {
            return this.get('/jobs', params);
        },

        /**
         * Get job details
         * @param {string} jobId - Job ID
         * @returns {Promise<Object>} - Job object
         */
        get: async (jobId) => {
            return this.get(`/jobs/${jobId}`);
        },

        /**
         * Create new job
         * @param {Object} submission - Job submission data
         * @returns {Promise<Object>} - Created job response with job_id
         */
        create: async (submission) => {
            return this.post('/jobs', submission);
        },

        /**
         * Cancel job
         * @param {string} jobId - Job ID
         * @returns {Promise<Object>} - Cancellation response
         */
        cancel: async (jobId) => {
            return this.delete(`/jobs/${jobId}`);
        },

        /**
         * Get job outputs
         * @param {string} jobId - Job ID
         * @returns {Promise<Object>} - Job outputs
         */
        getOutputs: async (jobId) => {
            return this.get(`/jobs/${jobId}/outputs`);
        },

        /**
         * Get job logs (stdout/stderr)
         * @param {string} jobId - Job ID
         * @param {Object} params - Query params (tail)
         * @returns {Promise<Object>}
         */
        getLogs: async (jobId, params = {}) => {
            return this.get(`/jobs/${jobId}/logs`, params);
        },

        /**
         * Stream job progress via SSE
         * @param {string} jobId - Job ID
         * @param {Function} onProgress - Progress event handler
         * @param {Function} onComplete - Completion event handler
         * @param {Function} onError - Error event handler
         * @returns {EventSource} - EventSource instance
         */
        streamProgress: (jobId, onProgress, onComplete, onError) => {
            return this.createSSE(`/jobs/${jobId}/progress`, {
                progress: (event) => {
                    const data = JSON.parse(event.data);
                    if (onProgress) onProgress(data);
                },
                stage_complete: (event) => {
                    const data = JSON.parse(event.data);
                    if (onProgress) onProgress(data);
                },
                completed: (event) => {
                    const data = JSON.parse(event.data);
                    if (onComplete) onComplete(data);
                },
                failed: (event) => {
                    const data = JSON.parse(event.data);
                    if (onError) onError(data);
                },
                error: (event) => {
                    if (onError) onError({ error: event.data });
                }
            });
        }
    };

    // ========================================
    // Results API Methods
    // ========================================

    results = {
        /**
         * List files in directory
         * @param {string} path - Directory path
         * @returns {Promise<Object>} - Files list
         */
        list: async (path = '') => {
            return this.get('/results/browse', path ? { path } : {});
        },

        /**
         * Get file metadata
         * @param {string} path - File path
         * @returns {Promise<Object>} - File metadata
         */
        getMetadata: async (path) => {
            return this.get('/results/metadata', { path });
        },

        /**
         * Download file
         * @param {string} path - File path
         * @returns {Promise<Blob>} - File blob
         */
        download: async (path) => {
            const url = `${this.baseURL}/api/results/download?path=${encodeURIComponent(path)}`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new APIError('Download failed', response.status);
            }
            return response.blob();
        }
    };

    // ========================================
    // Action API Methods
    // ========================================

    action = {
        /**
         * Get Action registry metadata (characters, actions, styles)
         * @returns {Promise<Object>}
         */
        getMetadata: async () => {
            return this.get('/action/metadata');
        },

        /**
         * Submit a generate job (multipart with control image)
         * @param {FormData} formData
         * @returns {Promise<Object>}
         */
        generate: async (formData) => {
            return this.postForm('/action/generate', formData);
        },

        /**
         * Submit an animate job (JSON body; uses server-side control_dir path)
         * @param {Object} payload
         * @returns {Promise<Object>}
         */
        animate: async (payload) => {
            return this.post('/action/animate', payload);
        },

        /**
         * Submit a control extraction job (JSON body)
         * @param {Object} payload
         * @returns {Promise<Object>}
         */
        extractControls: async (payload) => {
            return this.post('/action/extract_controls', payload);
        },

        /**
         * Submit an inpaint job (multipart with image+mask)
         * @param {FormData} formData
         * @returns {Promise<Object>}
         */
        inpaint: async (formData) => {
            return this.postForm('/action/inpaint', formData);
        },

        /**
         * Submit a multichar job (multipart with YAML config)
         * @param {FormData} formData
         * @returns {Promise<Object>}
         */
        multichar: async (formData) => {
            return this.postForm('/action/multichar', formData);
        }
    };

    // ========================================
    // Creative Studio API Methods
    // ========================================

    creative = {
        parody: async (payload) => {
            return this.post('/creative/parody', payload);
        },
        analyze: async (payload) => {
            return this.post('/creative/analyze', payload);
        },
        voice: async (payload) => {
            return this.post('/creative/voice', payload);
        },
        workflow: async (payload) => {
            return this.post('/creative/workflow', payload);
        }
    };

    image = {
        getMetadata: async () => {
            return this.get('/image/metadata');
        },
        resolveWorkflow: async (workflow) => {
            return this.get('/image/resolve-workflow', { workflow });
        },
        generate: async (payload) => {
            return this.post('/image/generate', payload);
        }
    };

    // ========================================
    // Config API Methods
    // ========================================

    configs = {
        /**
         * List available config files
         * @returns {Promise<Array>} - Config files list
         */
        list: async () => {
            return this.get('/configs');
        },

        /**
         * Get config file content
         * @param {string} path - Config file path
         * @returns {Promise<Object>} - Config content
         */
        get: async (path) => {
            return this.get('/configs/read', { path });
        },

        /**
         * Save config file
         * @param {string} path - Config file path
         * @param {string} content - Config content (YAML/JSON string)
         * @returns {Promise<Object>} - Save response
         */
        save: async (path, content) => {
            return this.post('/configs/write', { path, content });
        },

        /**
         * Validate config content
         * @param {string} content - Config content
         * @param {string} type - Config type ('yaml' or 'json')
         * @returns {Promise<Object>} - Validation result
         */
        validate: async (content, type) => {
            return this.post('/configs/validate', { content, type });
        }
    };

    // ========================================
    // System API Methods
    // ========================================

    system = {
        /**
         * Get current system metrics
         * @returns {Promise<Object>} - System metrics
         */
        getMetrics: async () => {
            return this.get('/system/metrics');
        },

        /**
         * Get system metrics history
         * @param {number} limit - Number of entries
         * @returns {Promise<Array>} - Metrics history
         */
        getMetricsHistory: async (limit = 100) => {
            return this.get('/system/metrics/history', { limit });
        },

        /**
         * Get disk usage
         * @returns {Promise<Object>} - Disk usage info
         */
        getDiskUsage: async () => {
            return this.get('/system/disk');
        },

        /**
         * Stream system metrics via SSE
         * @param {Function} onMetrics - Metrics event handler
         * @returns {EventSource} - EventSource instance
         */
        streamMetrics: (onMetrics) => {
            return this.createSSE('/system/metrics/stream', {
                metrics: (event) => {
                    const data = JSON.parse(event.data);
                    if (onMetrics) onMetrics(data);
                }
            });
        }
    };

    // ========================================
    // Statistics API Methods
    // ========================================

    stats = {
        /**
         * Get overall statistics
         * @param {string} timeRange - Time range ('24h', '7d', '30d', 'all')
         * @returns {Promise<Object>} - Statistics summary
         */
        getSummary: async (timeRange = '7d') => {
            return this.get('/stats/summary', { time_range: timeRange });
        },

        /**
         * Get chart data
         * @param {string} chartType - Chart type ('jobs', 'success_rate', 'duration')
         * @param {string} timeRange - Time range
         * @returns {Promise<Object>} - Chart data
         */
        getChartData: async (chartType, timeRange = '7d') => {
            return this.get('/stats/charts', { type: chartType, time_range: timeRange });
        }
    };

    // ========================================
    // Health Check
    // ========================================

    /**
     * Check backend health
     * @returns {Promise<Object>} - Health status
     */
    async health() {
        return this.get('/health');
    }
}

/**
 * Custom API Error class
 */
class APIError extends Error {
    constructor(message, statusCode, details = null) {
        super(message);
        this.name = 'APIError';
        this.statusCode = statusCode;
        this.details = details;
    }
}

// Create singleton instance
const api = new APIClient();

// Export for ES6 modules
export default api;
export { APIError };

// Also attach to window for non-module usage
if (typeof window !== 'undefined') {
    window.api = api;
    window.APIError = APIError;
}
