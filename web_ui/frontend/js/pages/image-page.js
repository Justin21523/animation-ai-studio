import BaseComponent from '../components/base-component.js';
import api from '../api/client.js';
import router from '../router.js';

export default class ImagePage extends BaseComponent {
    constructor() {
        super();
        this.metadata = null;
        this.loading = false;
        this.error = null;
    }

    async onMount() {
        this.loading = true;
        this.render(this._renderLoading());
        try {
            this.metadata = await api.image.getMetadata();
            this.loading = false;
            this.render(this._renderPage());
            this._bindEvents();
            this._renderGuiWorkflowStatus();
            this._applyWorkflowState();
            this._applyProviderState();
        } catch (error) {
            this.loading = false;
            this.error = error.message || 'Failed to load image providers';
            this.render(this._renderError());
        }
    }

    _providers() {
        return this.metadata?.providers || [];
    }

    _strategy() {
        return this.metadata?.workflow_strategy || {};
    }

    _workflowList() {
        return this.metadata?.workflow_list || [];
    }

    _workflowDefaults() {
        return this.metadata?.workflow_defaults || {};
    }

    _workflowTaskDefaults() {
        return this.metadata?.workflow_task_defaults || {};
    }

    _guiWorkflows() {
        return this.metadata?.gui_workflows || [];
    }

    _selectedWorkflow() {
        const workflow = this.$('#image-workflow')?.value?.trim();
        if (workflow) return workflow;
        const defaults = this._workflowDefaults();
        return defaults.general_text_to_image || '';
    }

    _selectedProvider() {
        const providerKey = this.$('#image-provider')?.value;
        return this._providers().find((item) => item.provider === providerKey) || null;
    }

    _statusLabel(provider) {
        if (!provider) return 'Unknown';
        if (!provider.enabled) return 'Disabled';
        if (!provider.available) return 'Unavailable';
        return provider.mode === 'cloud' ? 'Cloud Ready' : 'Local Ready';
    }

    _renderLoading() {
        return `
            <div class="page-container">
                <div class="loading-state">
                    <div class="loading-spinner"></div>
                    <p class="loading-message">Loading image providers...</p>
                </div>
            </div>
        `;
    }

    _renderError() {
        return `
            <div class="page-container">
                <div class="error-state">
                    <h2>Image</h2>
                    <p class="error-message">${this.escapeHtml(this.error || 'Unknown error')}</p>
                </div>
            </div>
        `;
    }

    _renderStrategySummary() {
        const strategy = this._strategy();
        const action = strategy.action_controlnet || null;
        const workflowCount = this._workflowList().length;
        if (!action) return '';
        return `
            <div class="image-strategy-banner">
                <div class="image-strategy-banner-title">Workflow Strategy</div>
                <div class="image-strategy-banner-text">
                    Action/ControlNet stays on <strong>${this.escapeHtml(action.provider || 'sdxl')}</strong>.
                    ${this.escapeHtml(action.reason || '')} Loaded ${workflowCount} workflow routes.
                </div>
            </div>
        `;
    }

    _renderPage() {
        const providers = this._providers();
        const workflowList = this._workflowList();
        const defaultWorkflow = this._workflowDefaults().general_text_to_image || workflowList[0] || '';
        const options = providers.map((provider) => {
            const disabled = (!provider.enabled || !provider.available) ? 'disabled' : '';
            const suffix = this._statusLabel(provider);
            return `<option value="${this.escapeHtml(provider.provider)}" ${disabled}>${this.escapeHtml(provider.label)} (${this.escapeHtml(suffix)})</option>`;
        }).join('');
        const workflowOptions = workflowList.map((workflow) => {
            const selected = workflow === defaultWorkflow ? 'selected' : '';
            return `<option value="${this.escapeHtml(workflow)}" ${selected}>${this.escapeHtml(workflow)}</option>`;
        }).join('');

        return `
            <div class="page-container image-page">
                <div class="page-header">
                    <div>
                        <h2>Image</h2>
                        <div class="muted">Provider-aware text-to-image generation.</div>
                    </div>
                </div>

                ${this._renderStrategySummary()}

                <div class="card image-provider-shell">
                    <div class="card-body">
                        <div class="image-provider-grid">
                            <div class="image-provider-form">
                                <form id="image-generate-form">
                                    <div class="form-group">
                                        <label class="form-label">Workflow Intent</label>
                                        <select class="form-control" id="image-workflow">${workflowOptions}</select>
                                        <div class="muted" id="image-workflow-help">Use strategy-driven routing and defaults for this workflow.</div>
                                    </div>

                                    <div class="form-group">
                                        <label class="form-label">Provider</label>
                                        <select class="form-control" id="image-provider">${options}</select>
                                    </div>

                                    <div class="form-group">
                                        <label class="form-label">Prompt</label>
                                        <textarea class="form-control" id="image-prompt" rows="5" placeholder="Describe the image you want to generate"></textarea>
                                    </div>

                                    <div class="form-group" id="image-style-group">
                                        <label class="form-label">Style</label>
                                        <select class="form-control" id="image-style"></select>
                                    </div>

                                    <div class="form-group" id="image-negative-group">
                                        <label class="form-label">Negative Prompt</label>
                                        <input class="form-control" id="image-negative-prompt" placeholder="Optional negative prompt" />
                                    </div>

                                    <div class="image-form-grid">
                                        <div class="form-group">
                                            <label class="form-label">Width</label>
                                            <input class="form-control" id="image-width" type="number" value="1024" min="256" step="8" />
                                        </div>
                                        <div class="form-group">
                                            <label class="form-label">Height</label>
                                            <input class="form-control" id="image-height" type="number" value="1024" min="256" step="8" />
                                        </div>
                                        <div class="form-group" id="image-num-images-group">
                                            <label class="form-label">Num Images</label>
                                            <input class="form-control" id="image-num-images" type="number" value="1" min="1" max="4" />
                                        </div>
                                    </div>

                                    <div class="image-form-grid">
                                        <div class="form-group" id="image-steps-group">
                                            <label class="form-label">Steps</label>
                                            <input class="form-control" id="image-steps" type="number" placeholder="Optional" />
                                        </div>
                                        <div class="form-group" id="image-guidance-group">
                                            <label class="form-label">Guidance Scale</label>
                                            <input class="form-control" id="image-guidance-scale" type="number" step="0.1" placeholder="Optional" />
                                        </div>
                                        <div class="form-group" id="image-seed-group">
                                            <label class="form-label">Seed</label>
                                            <input class="form-control" id="image-seed" type="number" placeholder="Optional" />
                                        </div>
                                    </div>

                                    <div class="image-form-footer">
                                        <div class="image-form-validation" id="image-form-validation"></div>
                                        <button class="btn btn-primary" type="submit" id="image-submit-btn">Submit Image Job</button>
                                    </div>
                                </form>
                            </div>

                            <aside class="image-provider-sidebar">
                                <div class="image-provider-panel">
                                    <div class="image-provider-panel-title">Provider Status</div>
                                    <div class="image-provider-status-badge" id="image-provider-status"></div>
                                    <p class="image-provider-description" id="image-provider-description"></p>
                                    <div class="image-provider-help" id="image-provider-help"></div>
                                </div>

                                <div class="image-provider-panel">
                                    <div class="image-provider-panel-title">Provider Flags</div>
                                    <div class="image-provider-tags" id="image-provider-flags"></div>
                                </div>

                                <div class="image-provider-panel">
                                    <div class="image-provider-panel-title">Recommended For</div>
                                    <div class="image-provider-tags" id="image-provider-tags"></div>
                                </div>

                                <div class="image-provider-panel">
                                    <div class="image-provider-panel-title">Workflow Fit</div>
                                    <div class="image-provider-help" id="image-provider-workflow-fit"></div>
                                    <div class="image-provider-help" id="image-workflow-reason"></div>
                                    <div class="image-provider-tags" id="image-workflow-fallbacks"></div>
                                </div>

                                <div class="image-provider-panel">
                                    <div class="image-provider-panel-title">GUI Workflow Assets</div>
                                    <div class="image-workflow-status-list" id="image-gui-workflows"></div>
                                </div>

                                <div class="image-provider-panel">
                                    <div class="image-provider-panel-title">Usage Notes</div>
                                    <div class="image-provider-notes" id="image-provider-notes"></div>
                                </div>
                            </aside>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    _setText(selector, text) {
        const node = this.$(selector);
        if (node) node.textContent = text || '';
    }

    _renderTags(provider) {
        const node = this.$('#image-provider-tags');
        if (!node) return;
        const tags = provider?.recommended_for || [];
        node.innerHTML = tags.length
            ? tags.map((tag) => `<span class="image-provider-tag">${this.escapeHtml(tag)}</span>`).join('')
            : '<span class="muted">No recommendations configured.</span>';
    }

    _renderFlags(provider) {
        const node = this.$('#image-provider-flags');
        if (!node) return;
        const flags = provider?.ui_flags || [];
        node.innerHTML = flags.length
            ? flags.map((flag) => `<span class="image-provider-tag image-provider-flag">${this.escapeHtml(flag)}</span>`).join('')
            : '<span class="muted">No special flags.</span>';
    }

    _renderNotes(provider) {
        const node = this.$('#image-provider-notes');
        if (!node) return;
        const notes = provider?.usage_notes || [];
        node.innerHTML = notes.length
            ? notes.map((note) => `<div class="image-provider-note">${this.escapeHtml(note)}</div>`).join('')
            : '<div class="muted">No additional usage notes.</div>';
    }

    _renderGuiWorkflowStatus() {
        const node = this.$('#image-gui-workflows');
        if (!node) return;
        const workflows = this._guiWorkflows();
        if (!workflows.length) {
            node.innerHTML = '<div class="muted">No GUI workflows registered.</div>';
            return;
        }

        node.innerHTML = workflows.map((workflow) => {
            const features = workflow.features || [];
            const missing = workflow.missing_count || 0;
            const status = workflow.available ? 'Ready' : `Missing ${missing}`;
            const statusClass = workflow.available ? 'ready' : 'blocked';
            const featureTags = features.map((feature) => (
                `<span class="image-provider-tag image-provider-flag">${this.escapeHtml(feature)}</span>`
            )).join('');
            const missingNames = (workflow.missing_models || []).slice(0, 3).map((item) => item.name).join(', ');
            const missingText = workflow.available
                ? ''
                : `<div class="image-workflow-missing">${this.escapeHtml(missingNames)}${missing > 3 ? '...' : ''}</div>`;
            return `
                <div class="image-workflow-status-item">
                    <div class="image-workflow-status-header">
                        <span>${this.escapeHtml(workflow.label || workflow.id)}</span>
                        <span class="image-provider-status-badge ${statusClass}">${this.escapeHtml(status)}</span>
                    </div>
                    <div class="image-workflow-meta">${this.escapeHtml(workflow.family || 'ComfyUI')} · ${workflow.model_count || 0} models</div>
                    <div class="image-provider-tags">${featureTags}</div>
                    ${missingText}
                </div>
            `;
        }).join('');
    }

    _providerFitText(provider) {
        const strategy = this._strategy();
        const selectedWorkflow = this._selectedWorkflow();
        const selected = selectedWorkflow ? strategy[selectedWorkflow] : null;
        const action = strategy.action_controlnet;

        const lines = [];
        if (selected?.provider === provider.provider) {
            lines.push(`Selected workflow routes to ${provider.provider}.`);
        } else if (selected?.candidates?.includes(provider.provider)) {
            lines.push(`Selected workflow fallback order includes ${provider.provider}.`);
        }
        if (action?.provider === provider.provider) {
            lines.push('This provider is the fixed Action/ControlNet backend.');
        } else if (!provider.action_compatible) {
            lines.push('Not intended for Action/ControlNet workflows.');
        }
        return lines.join(' ') || 'General text-to-image only.';
    }

    _applyWorkflowState() {
        const workflow = this._selectedWorkflow();
        const strategy = this._strategy();
        const providers = this._providers();
        const providerSelect = this.$('#image-provider');
        const workflowHelp = this.$('#image-workflow-help');
        const workflowReason = this.$('#image-workflow-reason');
        const fallbackNode = this.$('#image-workflow-fallbacks');
        const taskDefaults = this._workflowTaskDefaults()[workflow] || {};
        const decision = strategy[workflow] || null;
        const isFixed = String(decision?.mode || '') === 'fixed';

        let selectedProvider = taskDefaults.provider || decision?.provider || this._workflowDefaults()[workflow] || '';
        const candidates = Array.isArray(decision?.candidates) ? decision.candidates : [];
        const isReady = (name) => providers.some((item) => item.provider === name && item.enabled && item.available);
        if (!isReady(selectedProvider)) {
            selectedProvider = candidates.find((name) => isReady(name)) || selectedProvider;
        }
        if (providerSelect && selectedProvider) {
            providerSelect.value = selectedProvider;
        }
        if (providerSelect) {
            providerSelect.disabled = isFixed;
        }

        const widthInput = this.$('#image-width');
        const heightInput = this.$('#image-height');
        const stepsInput = this.$('#image-steps');
        const guidanceInput = this.$('#image-guidance-scale');

        if (widthInput && taskDefaults.width) widthInput.value = taskDefaults.width;
        if (heightInput && taskDefaults.height) heightInput.value = taskDefaults.height;
        if (stepsInput) stepsInput.value = taskDefaults.steps ? String(taskDefaults.steps) : '';
        if (guidanceInput) guidanceInput.value = taskDefaults.guidance_scale ? String(taskDefaults.guidance_scale) : '';

        if (workflowHelp) {
            if (isFixed && selectedProvider) {
                workflowHelp.textContent = `Workflow ${workflow} is fixed to provider ${selectedProvider}.`;
            } else {
                workflowHelp.textContent = selectedProvider
                    ? `Strategy suggests ${selectedProvider} for ${workflow}.`
                    : `No strategy provider resolved for ${workflow}.`;
            }
        }
        if (workflowReason) {
            workflowReason.textContent = decision?.reason || '';
        }
        if (fallbackNode) {
            fallbackNode.innerHTML = candidates.length
                ? candidates.map((name) => `<span class="image-provider-tag">${this.escapeHtml(name)}</span>`).join('')
                : '<span class="muted">No fallback candidates.</span>';
        }
    }

    _applyProviderState() {
        const provider = this._selectedProvider();
        if (!provider) return;

        const styleGroup = this.$('#image-style-group');
        const styleSelect = this.$('#image-style');
        const negativeInput = this.$('#image-negative-prompt');
        const stepsInput = this.$('#image-steps');
        const guidanceInput = this.$('#image-guidance-scale');
        const seedInput = this.$('#image-seed');
        const negativeGroup = this.$('#image-negative-group');
        const stepsGroup = this.$('#image-steps-group');
        const guidanceGroup = this.$('#image-guidance-group');
        const seedGroup = this.$('#image-seed-group');
        const numImagesGroup = this.$('#image-num-images-group');
        const widthInput = this.$('#image-width');
        const heightInput = this.$('#image-height');
        const numImagesInput = this.$('#image-num-images');
        const submitBtn = this.$('#image-submit-btn');
        const validation = this.$('#image-form-validation');
        const statusBadge = this.$('#image-provider-status');

        if (styleGroup && styleSelect) {
            styleGroup.style.display = provider.supports_styles ? '' : 'none';
            styleSelect.innerHTML = (provider.styles || []).map((style) => (
                `<option value="${this.escapeHtml(style)}">${this.escapeHtml(style)}</option>`
            )).join('');
        }

        if (negativeGroup) negativeGroup.style.display = provider.supports_negative_prompt ? '' : 'none';
        if (stepsGroup) stepsGroup.style.display = provider.supports_steps ? '' : 'none';
        if (guidanceGroup) guidanceGroup.style.display = provider.supports_guidance_scale ? '' : 'none';
        if (seedGroup) seedGroup.style.display = provider.supports_seed ? '' : 'none';
        if (numImagesGroup) numImagesGroup.style.display = provider.supports_num_images ? '' : 'none';
        if (negativeInput) {
            negativeInput.disabled = !provider.available || !provider.supports_negative_prompt;
            if (!provider.supports_negative_prompt) negativeInput.value = '';
        }
        if (stepsInput) {
            stepsInput.disabled = !provider.available || !provider.supports_steps;
            if (!provider.supports_steps) stepsInput.value = '';
        }
        if (guidanceInput) {
            guidanceInput.disabled = !provider.available || !provider.supports_guidance_scale;
            if (!provider.supports_guidance_scale) guidanceInput.value = '';
        }
        if (seedInput) {
            seedInput.disabled = !provider.available || !provider.supports_seed;
            if (!provider.supports_seed) seedInput.value = '';
        }

        if (widthInput) {
            const workflow = this._selectedWorkflow();
            const taskDefaults = this._workflowTaskDefaults()[workflow] || {};
            widthInput.value = taskDefaults.width || provider.default_width || 1024;
            widthInput.disabled = !provider.available;
        }
        if (heightInput) {
            const workflow = this._selectedWorkflow();
            const taskDefaults = this._workflowTaskDefaults()[workflow] || {};
            heightInput.value = taskDefaults.height || provider.default_height || 1024;
            heightInput.disabled = !provider.available;
        }
        if (numImagesInput) {
            numImagesInput.max = provider.max_images_per_request || 1;
            numImagesInput.disabled = !provider.available || !provider.supports_num_images;
            if (Number(numImagesInput.value) > Number(numImagesInput.max)) {
                numImagesInput.value = String(numImagesInput.max);
            }
        }

        if (submitBtn) {
            submitBtn.disabled = !provider.enabled || !provider.available;
            submitBtn.textContent = provider.enabled && provider.available ? 'Submit Image Job' : 'Provider Not Ready';
        }

        if (validation) {
            validation.textContent = provider.enabled && provider.available
                ? ''
                : (provider.availability_reason || 'Provider is not ready in this environment.');
        }

        if (statusBadge) {
            statusBadge.textContent = this._statusLabel(provider);
            statusBadge.className = `image-provider-status-badge ${provider.available && provider.enabled ? 'ready' : 'blocked'}`;
        }

        this._setText('#image-provider-description', provider.description || '');
        this._setText('#image-provider-help', provider.availability_reason || 'Provider can accept jobs in the current environment.');
        this._setText('#image-provider-workflow-fit', this._providerFitText(provider));
        this._renderFlags(provider);
        this._renderTags(provider);
        this._renderNotes(provider);
    }

    _bindEvents() {
        this.$('#image-workflow')?.addEventListener('change', () => {
            this._applyWorkflowState();
            this._applyProviderState();
        });
        this.$('#image-provider')?.addEventListener('change', () => this._applyProviderState());
        this.$('#image-generate-form')?.addEventListener('submit', async (event) => {
            event.preventDefault();
            const provider = this._selectedProvider();
            const prompt = this.$('#image-prompt')?.value?.trim();
            const validation = this.$('#image-form-validation');

            if (!provider) {
                if (validation) validation.textContent = 'Select a provider first.';
                return;
            }
            if (!provider.enabled || !provider.available) {
                if (validation) validation.textContent = provider.availability_reason || 'Provider is not available.';
                return;
            }
            if (!prompt) {
                if (validation) validation.textContent = 'Prompt is required.';
                return;
            }

            const numImages = Number(this.$('#image-num-images')?.value || 1);
            if (numImages > Number(provider.max_images_per_request || 1)) {
                if (validation) validation.textContent = `Provider allows at most ${provider.max_images_per_request} images per request.`;
                return;
            }

            if (validation) validation.textContent = '';

            const payload = {
                provider: provider.provider,
                workflow: this._selectedWorkflow() || null,
                prompt,
                negative_prompt: provider.supports_negative_prompt ? (this.$('#image-negative-prompt')?.value?.trim() || null) : null,
                style: provider.supports_styles ? (this.$('#image-style')?.value || null) : null,
                width: Number(this.$('#image-width')?.value || provider.default_width || 1024),
                height: Number(this.$('#image-height')?.value || provider.default_height || 1024),
                steps: provider.supports_steps && this.$('#image-steps')?.value ? Number(this.$('#image-steps')?.value) : null,
                guidance_scale: provider.supports_guidance_scale && this.$('#image-guidance-scale')?.value ? Number(this.$('#image-guidance-scale')?.value) : null,
                seed: provider.supports_seed && this.$('#image-seed')?.value ? Number(this.$('#image-seed')?.value) : null,
                num_images: provider.supports_num_images ? numImages : 1,
            };

            const res = await api.image.generate(payload);
            router.navigate(`/jobs/${res.job_id}`);
        });
    }
}
