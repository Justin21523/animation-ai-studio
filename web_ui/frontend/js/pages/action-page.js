import BaseComponent from '../components/base-component.js';
import api from '../api/client.js';
import router from '../router.js';

export default class ActionPage extends BaseComponent {
    constructor() {
        super();
        this.metadata = null;
        this.loading = false;
        this.error = null;

        this.activeJob = null;
        this.jobSseEndpoint = null;
        this.outputFiles = [];
        this.gallerySourcePath = null;
    }

    async onMount() {
        this.loading = true;
        this.render(this._renderLoading());

        try {
            this.metadata = await api.action.getMetadata();
            this.loading = false;
            this.render(this._renderPage());
            this._setupEventListeners();
        } catch (error) {
            console.error('Failed to load action metadata:', error);
            this.loading = false;
            this.error = error.message || 'Failed to load metadata';
            this.render(this._renderError());
        }
    }

    async onUnmount() {
        if (this.jobSseEndpoint) {
            api.closeSSE(this.jobSseEndpoint);
            this.jobSseEndpoint = null;
        }
    }

    _renderLoading() {
        return `
            <div class="page-container">
                <div class="loading-state">
                    <div class="loading-spinner"></div>
                    <p class="loading-message">Loading Action metadata...</p>
                </div>
            </div>
        `;
    }

    _renderError() {
        return `
            <div class="page-container">
                <div class="error-state">
                    <h2>Action</h2>
                    <p class="error-message">${this.escapeHtml(this.error || 'Unknown error')}</p>
                </div>
            </div>
        `;
    }

    _renderPage() {
        const chars = (this.metadata?.characters || []).map(c => String(c));
        const actions = this.metadata?.actions || {};
        const actionKeys = Object.keys(actions);
        const controlTypes = this.metadata?.control_types || ['auto', 'pose', 'canny', 'softedge', 'lineart', 'tile'];
        const styles = this.metadata?.styles || ['pixar_3d'];
        const negKeys = this.metadata?.negative_prompt_keys || ['character', 'default'];
        const defaults = this.metadata?.defaults || {};

        return `
            <div class="page-container action-page">
                <div class="job-list-header">
                    <h2>Action</h2>
                    <div class="job-list-controls">
                        <button class="btn btn-ghost" id="open-results-btn" title="Open Results Browser">
                            📁 Results
                        </button>
                    </div>
                </div>

                <div class="card" style="padding: 12px; margin-bottom: 16px;">
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <button class="btn btn-sm btn-ghost" type="button" data-scroll="generate">Generate</button>
                        <button class="btn btn-sm btn-ghost" type="button" data-scroll="extract">Extract Controls</button>
                        <button class="btn btn-sm btn-ghost" type="button" data-scroll="animate">Animate</button>
                        <button class="btn btn-sm btn-ghost" type="button" data-scroll="inpaint">Inpaint</button>
                        <button class="btn btn-sm btn-ghost" type="button" data-scroll="multichar">Multi-Char</button>
                    </div>
                </div>

                <div class="card" id="action-section-generate" style="padding: 16px; margin-bottom: 16px;">
                    <h3 style="margin-top: 0;">Generate</h3>

                    <form id="action-generate-form">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <label>
                                Character
                                <select name="character" required class="sort-select" style="width: 100%;">
                                    ${chars.map(c => `<option value="${this.escapeHtml(c)}">${this.escapeHtml(c)}</option>`).join('')}
                                </select>
                            </label>

                            <label>
                                Control Type
                                <select name="control_type" class="sort-select" style="width: 100%;">
                                    ${controlTypes.map(t => `<option value="${this.escapeHtml(t)}">${this.escapeHtml(t)}</option>`).join('')}
                                </select>
                            </label>

                            <label>
                                Action (preset key or free text)
                                <input name="action" list="action-presets" class="form-control" style="width: 100%;" placeholder="run / jump / fight ..." />
                                <datalist id="action-presets">
                                    ${actionKeys.map(k => `<option value="${this.escapeHtml(k)}"></option>`).join('')}
                                </datalist>
                            </label>

                            <label>
                                Style
                                <select name="style" class="sort-select" style="width: 100%;">
                                    ${styles.map(s => `<option value="${this.escapeHtml(s)}" ${s === defaults.style ? 'selected' : ''}>${this.escapeHtml(s)}</option>`).join('')}
                                </select>
                            </label>

                            <label>
                                Scene
                                <input name="scene" class="form-control" style="width: 100%;" placeholder="optional scene" />
                            </label>

                            <label>
                                Extra
                                <input name="extra" class="form-control" style="width: 100%;" placeholder="extra prompt text" />
                            </label>

                            <label style="grid-column: 1 / span 2;">
                                Prompt Override (optional)
                                <input name="prompt" class="form-control" style="width: 100%;" placeholder="If set, overrides character/action/scene/style assembly" />
                            </label>

                            <label>
                                Negative Prompt Key
                                <select name="negative_prompt_key" class="sort-select" style="width: 100%;">
                                    ${negKeys.map(k => `<option value="${this.escapeHtml(k)}" ${k === defaults.negative_prompt_key ? 'selected' : ''}>${this.escapeHtml(k)}</option>`).join('')}
                                </select>
                            </label>

                            <label>
                                Negative Prompt Override (optional)
                                <input name="negative_prompt" class="form-control" style="width: 100%;" placeholder="optional negative prompt override" />
                            </label>

                            <label>
                                Control Image
                                <input name="control_image" type="file" accept="image/*" required />
                            </label>

                            <label>
                                Reference Image (optional)
                                <input name="reference_image" type="file" accept="image/*" />
                            </label>

                            <label>
                                Width
                                <input name="width" type="number" value="${defaults.width || 1024}" min="256" step="8" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Height
                                <input name="height" type="number" value="${defaults.height || 1024}" min="256" step="8" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Steps
                                <input name="steps" type="number" value="${defaults.steps || 30}" min="1" max="200" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Guidance Scale
                                <input name="guidance_scale" type="number" value="${defaults.guidance_scale || 7.5}" step="0.1" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                ControlNet Scale (optional)
                                <input name="controlnet_scale" type="number" step="0.05" class="form-control" style="width: 100%;" placeholder="leave blank for default" />
                            </label>

                            <label>
                                Seed (optional)
                                <input name="seed" type="number" class="form-control" style="width: 100%;" placeholder="leave blank for random" />
                            </label>

                            <label>
                                Num Images
                                <input name="num_images" type="number" value="1" min="1" max="8" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Consistency Threshold
                                <input name="consistency_threshold" type="number" value="0.65" step="0.01" min="0" max="1" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Max Retries
                                <input name="max_retries" type="number" value="0" min="0" max="20" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Consistency Device
                                <select name="consistency_device" class="sort-select" style="width: 100%;">
                                    <option value="cpu" selected>cpu</option>
                                    <option value="cuda">cuda</option>
                                </select>
                            </label>

                            <label>
                                Timeout (seconds, optional)
                                <input name="timeout" type="number" class="form-control" style="width: 100%;" placeholder="e.g. 1800" />
                            </label>
                        </div>

                        <div style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                            <label style="display: flex; gap: 8px; align-items: center;">
                                <input name="no_preprocess" type="checkbox" />
                                No preprocess (treat control image as precomputed)
                            </label>
                            <button class="btn btn-primary" type="submit" id="submit-generate-btn">Generate</button>
                        </div>
                    </form>
                </div>

                <div class="card" id="action-section-extract" style="padding: 16px; margin-bottom: 16px;">
                    <h3 style="margin-top: 0;">Extract Controls</h3>
                    <p style="margin-top: 0; color: var(--color-text-secondary); font-size: 12px;">
                        Uses a server-side path (video file or frames directory).
                    </p>

                    <form id="action-extract-form">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <label>
                                Mode
                                <select name="mode" class="sort-select" style="width: 100%;">
                                    <option value="video" selected>video</option>
                                    <option value="frames">frames</option>
                                </select>
                            </label>

                            <label>
                                Types (comma separated)
                                <input name="types" class="form-control" style="width: 100%;" placeholder="pose,canny,softedge,lineart,tile" />
                            </label>

                            <label style="grid-column: 1 / -1;">
                                Input Path (video path or frames dir)
                                <input name="input_path" class="form-control" style="width: 100%;" placeholder="/mnt/data/.../video.mp4 or /mnt/data/.../frames" required />
                            </label>

                            <label>
                                Pattern (frames mode)
                                <input name="pattern" class="form-control" value="*.png" style="width: 100%;" />
                            </label>

                            <label>
                                FPS (video mode, optional)
                                <input name="fps" type="number" class="form-control" style="width: 100%;" placeholder="e.g. 12" />
                            </label>

                            <label>
                                Detect Resolution (optional)
                                <input name="detect_resolution" type="number" class="form-control" style="width: 100%;" placeholder="e.g. 512" />
                            </label>

                            <label>
                                Image Resolution (optional)
                                <input name="image_resolution" type="number" class="form-control" style="width: 100%;" placeholder="e.g. 1024" />
                            </label>

                            <label>
                                Timeout (seconds, optional)
                                <input name="timeout" type="number" class="form-control" style="width: 100%;" placeholder="e.g. 3600" />
                            </label>
                        </div>

                        <div style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                            <label style="display: flex; gap: 8px; align-items: center;">
                                <input name="overwrite" type="checkbox" />
                                Overwrite
                            </label>
                            <button class="btn btn-primary" type="submit" id="submit-extract-btn">Extract</button>
                        </div>
                    </form>
                </div>

                <div class="card" id="action-section-animate" style="padding: 16px; margin-bottom: 16px;">
                    <h3 style="margin-top: 0;">Animate</h3>
                    <p style="margin-top: 0; color: var(--color-text-secondary); font-size: 12px;">
                        Uses a server-side control frames directory (e.g. pose frames).
                    </p>

                    <form id="action-animate-form">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <label>
                                Character
                                <select name="character" required class="sort-select" style="width: 100%;">
                                    ${chars.map(c => `<option value="${this.escapeHtml(c)}">${this.escapeHtml(c)}</option>`).join('')}
                                </select>
                            </label>

                            <label>
                                Control Type
                                <select name="control_type" class="sort-select" style="width: 100%;">
                                    ${controlTypes.map(t => `<option value="${this.escapeHtml(t)}">${this.escapeHtml(t)}</option>`).join('')}
                                </select>
                            </label>

                            <label style="grid-column: 1 / -1;">
                                Control Dir
                                <input name="control_dir" class="form-control" style="width: 100%;" placeholder="/mnt/data/.../controls/pose" required />
                            </label>

                            <label>
                                Pattern
                                <input name="pattern" class="form-control" value="*.png" style="width: 100%;" />
                            </label>

                            <label>
                                Every
                                <input name="every" type="number" value="1" min="1" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Limit (0 = all)
                                <input name="limit" type="number" value="0" min="0" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Action (optional)
                                <input name="action" list="action-presets" class="form-control" style="width: 100%;" placeholder="run / jump / fight ..." />
                            </label>

                            <label>
                                Style
                                <select name="style" class="sort-select" style="width: 100%;">
                                    ${styles.map(s => `<option value="${this.escapeHtml(s)}" ${s === defaults.style ? 'selected' : ''}>${this.escapeHtml(s)}</option>`).join('')}
                                </select>
                            </label>

                            <label style="grid-column: 1 / -1;">
                                Prompt (optional)
                                <input name="prompt" class="form-control" style="width: 100%;" placeholder="full prompt (optional)" />
                            </label>

                            <label style="grid-column: 1 / -1;">
                                Extra (optional)
                                <input name="extra" class="form-control" style="width: 100%;" placeholder="extra prompt text" />
                            </label>

                            <label>
                                Width
                                <input name="width" type="number" value="${defaults.width || 1024}" min="256" step="8" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Height
                                <input name="height" type="number" value="${defaults.height || 1024}" min="256" step="8" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Steps
                                <input name="steps" type="number" value="${defaults.steps || 30}" min="1" max="200" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Guidance Scale
                                <input name="guidance_scale" type="number" value="${defaults.guidance_scale || 7.5}" step="0.1" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                ControlNet Scale (optional)
                                <input name="controlnet_scale" type="number" step="0.05" class="form-control" style="width: 100%;" placeholder="leave blank for default" />
                            </label>

                            <label>
                                Seed (optional)
                                <input name="seed" type="number" class="form-control" style="width: 100%;" placeholder="leave blank for random" />
                            </label>

                            <label>
                                Refine Sequence (optional, comma separated)
                                <input name="refine_sequence" class="form-control" style="width: 100%;" placeholder="tile,lineart,softedge" />
                            </label>

                            <label>
                                Write Video
                                <select name="write_video" class="sort-select" style="width: 100%;">
                                    <option value="false" selected>false</option>
                                    <option value="true">true</option>
                                </select>
                            </label>

                            <label>
                                FPS
                                <input name="fps" type="number" value="12" min="1" max="60" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Timeout (seconds, optional)
                                <input name="timeout" type="number" class="form-control" style="width: 100%;" placeholder="e.g. 3600" />
                            </label>
                        </div>

                        <div style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                            <label style="display: flex; gap: 8px; align-items: center;">
                                <input name="no_preprocess" type="checkbox" />
                                No preprocess
                            </label>
                            <button class="btn btn-primary" type="submit" id="submit-animate-btn">Animate</button>
                        </div>
                    </form>
                </div>

                <div class="card" id="action-section-inpaint" style="padding: 16px; margin-bottom: 16px;">
                    <h3 style="margin-top: 0;">Inpaint</h3>
                    <form id="action-inpaint-form">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <label>
                                Character
                                <select name="character" required class="sort-select" style="width: 100%;">
                                    ${chars.map(c => `<option value="${this.escapeHtml(c)}">${this.escapeHtml(c)}</option>`).join('')}
                                </select>
                            </label>

                            <label>
                                Control Type
                                <select name="control_type" class="sort-select" style="width: 100%;">
                                    ${controlTypes.map(t => `<option value="${this.escapeHtml(t)}">${this.escapeHtml(t)}</option>`).join('')}
                                </select>
                            </label>

                            <label>
                                Image
                                <input name="image" type="file" required accept="image/*" />
                            </label>

                            <label>
                                Mask
                                <input name="mask" type="file" required accept="image/*" />
                            </label>

                            <label>
                                Control Image (optional)
                                <input name="control_image" type="file" accept="image/*" />
                            </label>

                            <label>
                                Strength
                                <input name="strength" type="number" value="0.55" step="0.01" min="0" max="1" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Steps
                                <input name="steps" type="number" value="${defaults.steps || 30}" min="1" max="200" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Guidance Scale
                                <input name="guidance_scale" type="number" value="${defaults.guidance_scale || 7.5}" step="0.1" class="form-control" style="width: 100%;" />
                            </label>

                            <label>
                                Seed (optional)
                                <input name="seed" type="number" class="form-control" style="width: 100%;" placeholder="leave blank for random" />
                            </label>
                        </div>

                        <div style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                            <label style="display: flex; gap: 8px; align-items: center;">
                                <input name="no_preprocess" type="checkbox" />
                                No preprocess
                            </label>
                            <button class="btn btn-primary" type="submit" id="submit-inpaint-btn">Inpaint</button>
                        </div>
                    </form>
                </div>

                <div class="card" id="action-section-multichar" style="padding: 16px; margin-bottom: 16px;">
                    <h3 style="margin-top: 0;">Multi-Char</h3>
                    <p style="margin-top: 0; color: var(--color-text-secondary); font-size: 12px;">
                        Upload a YAML config for layered generation (see scripts/generation/action/multichar.py --help).
                    </p>
                    <form id="action-multichar-form">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <label style="grid-column: 1 / -1;">
                                Config YAML
                                <input name="config" type="file" required accept=".yaml,.yml,text/yaml" />
                            </label>

                            <label>
                                Timeout (seconds, optional)
                                <input name="timeout" type="number" class="form-control" style="width: 100%;" placeholder="e.g. 3600" />
                            </label>
                        </div>
                        <div style="margin-top: 12px;">
                            <button class="btn btn-primary" type="submit" id="submit-multichar-btn">Run</button>
                        </div>
                    </form>
                </div>

                <div id="action-job-status"></div>
                <div id="action-output-gallery"></div>
            </div>
        `;
    }

    _setupEventListeners() {
        const scrollButtons = this.$$('[data-scroll]');
        scrollButtons.forEach((btn) => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const key = btn.dataset.scroll;
                const target = this.$(`#action-section-${key}`);
                if (target && target.scrollIntoView) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });

        const form = this.$('#action-generate-form');
        if (form) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this._submitGenerate(form);
            });
        }

        const extractForm = this.$('#action-extract-form');
        if (extractForm) {
            extractForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this._submitExtractControls(extractForm);
            });
        }

        const animateForm = this.$('#action-animate-form');
        if (animateForm) {
            animateForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this._submitAnimate(animateForm);
            });
        }

        const inpaintForm = this.$('#action-inpaint-form');
        if (inpaintForm) {
            inpaintForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this._submitInpaint(inpaintForm);
            });
        }

        const multicharForm = this.$('#action-multichar-form');
        if (multicharForm) {
            multicharForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this._submitMultichar(multicharForm);
            });
        }

        const openResultsBtn = this.$('#open-results-btn');
        if (openResultsBtn) {
            openResultsBtn.addEventListener('click', (e) => {
                e.preventDefault();
                router.navigate('/results');
            });
        }
    }

    async _submitGenerate(form) {
        const submitBtn = this.$('#submit-generate-btn');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
        }

        this._setJobStatus({ message: 'Submitting job...', status: 'pending' });
        this._setGallery([]);

        try {
            const fd = new FormData();
            const formData = new FormData(form);

            // Copy known fields (ensures checkbox/value normalization)
            const fields = [
                'character', 'control_type', 'action', 'scene', 'prompt', 'extra', 'style',
                'negative_prompt', 'negative_prompt_key',
                'width', 'height', 'steps', 'guidance_scale',
                'controlnet_scale', 'seed', 'num_images',
                'consistency_threshold', 'max_retries', 'consistency_device', 'timeout'
            ];
            fields.forEach((k) => {
                if (formData.has(k)) {
                    const v = formData.get(k);
                    if (v !== null && v !== undefined && String(v).length > 0) {
                        fd.append(k, v);
                    }
                }
            });

            // Checkbox
            fd.append('no_preprocess', formData.get('no_preprocess') ? 'true' : 'false');

            // Files
            const controlFile = formData.get('control_image');
            if (!(controlFile instanceof File) || controlFile.size === 0) {
                throw new Error('Control image is required');
            }
            fd.append('control_image', controlFile, controlFile.name);

            const refFile = formData.get('reference_image');
            if (refFile instanceof File && refFile.size > 0) {
                fd.append('reference_image', refFile, refFile.name);
            }

            const resp = await api.action.generate(fd);
            const jobId = resp.job_id;
            const outputDir = resp.output_dir;

            this.activeJob = { jobId, outputDir, status: 'pending' };
            this._setJobStatus({
                message: `Job submitted: ${jobId}`,
                status: 'pending',
                outputDir
            });

            await this._monitorJob(jobId, outputDir);
        } catch (error) {
            console.error('Failed to submit generate job:', error);
            this._setJobStatus({ message: error.message || 'Submit failed', status: 'failed' });
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Generate';
            }
        }
    }

    async _submitExtractControls(form) {
        const submitBtn = this.$('#submit-extract-btn');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
        }

        this._setJobStatus({ message: 'Submitting job...', status: 'pending' });
        this._setGallery([]);

        try {
            const formData = new FormData(form);
            const mode = String(formData.get('mode') || 'video');
            const inputPath = String(formData.get('input_path') || '').trim();
            if (!inputPath) {
                throw new Error('Input path is required');
            }

            const payload = {
                mode,
                pattern: String(formData.get('pattern') || '*.png'),
                overwrite: Boolean(formData.get('overwrite')),
            };
            if (mode === 'frames') {
                payload.frames_dir = inputPath;
            } else {
                payload.video = inputPath;
            }

            const types = String(formData.get('types') || '').trim();
            if (types) {
                payload.types = types.split(',').map(s => s.trim()).filter(Boolean);
            }
            const fps = String(formData.get('fps') || '').trim();
            if (fps) payload.fps = Number(fps);
            const det = String(formData.get('detect_resolution') || '').trim();
            if (det) payload.detect_resolution = Number(det);
            const imgRes = String(formData.get('image_resolution') || '').trim();
            if (imgRes) payload.image_resolution = Number(imgRes);
            const timeout = String(formData.get('timeout') || '').trim();
            if (timeout) payload.timeout = Number(timeout);

            const resp = await api.action.extractControls(payload);
            const jobId = resp.job_id;
            const outputDir = resp.output_dir;

            this.activeJob = { jobId, outputDir, status: 'pending' };
            this._setJobStatus({ message: `Job submitted: ${jobId}`, status: 'pending', outputDir });
            await this._monitorJob(jobId, outputDir);
        } catch (error) {
            console.error('Failed to submit extract job:', error);
            this._setJobStatus({ message: error.message || 'Submit failed', status: 'failed' });
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Extract';
            }
        }
    }

    async _submitAnimate(form) {
        const submitBtn = this.$('#submit-animate-btn');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
        }

        this._setJobStatus({ message: 'Submitting job...', status: 'pending' });
        this._setGallery([]);

        try {
            const formData = new FormData(form);
            const payload = {
                character: String(formData.get('character') || ''),
                control_type: String(formData.get('control_type') || 'pose'),
                control_dir: String(formData.get('control_dir') || '').trim(),
                pattern: String(formData.get('pattern') || '*.png'),
                every: Number(formData.get('every') || 1),
                limit: Number(formData.get('limit') || 0),
                action: String(formData.get('action') || '').trim() || null,
                prompt: String(formData.get('prompt') || '').trim() || null,
                extra: String(formData.get('extra') || '').trim(),
                style: String(formData.get('style') || 'pixar_3d'),
                width: Number(formData.get('width') || 1024),
                height: Number(formData.get('height') || 1024),
                steps: Number(formData.get('steps') || 30),
                guidance_scale: Number(formData.get('guidance_scale') || 7.5),
                seed: String(formData.get('seed') || '').trim() ? Number(formData.get('seed')) : null,
                seed_mode: 'fixed',
                no_preprocess: Boolean(formData.get('no_preprocess')),
                write_video: String(formData.get('write_video') || 'false') === 'true',
                fps: Number(formData.get('fps') || 12),
            };
            if (!payload.character) throw new Error('Character is required');
            if (!payload.control_dir) throw new Error('Control dir is required');

            const scale = String(formData.get('controlnet_scale') || '').trim();
            if (scale) payload.controlnet_scale = Number(scale);

            const refine = String(formData.get('refine_sequence') || '').trim();
            if (refine) {
                payload.refine_sequence = refine.split(',').map(s => s.trim()).filter(Boolean);
            }

            const timeout = String(formData.get('timeout') || '').trim();
            if (timeout) payload.timeout = Number(timeout);

            const resp = await api.action.animate(payload);
            const jobId = resp.job_id;
            const outputDir = resp.output_dir;

            this.activeJob = { jobId, outputDir, status: 'pending' };
            this._setJobStatus({ message: `Job submitted: ${jobId}`, status: 'pending', outputDir });
            await this._monitorJob(jobId, outputDir);
        } catch (error) {
            console.error('Failed to submit animate job:', error);
            this._setJobStatus({ message: error.message || 'Submit failed', status: 'failed' });
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Animate';
            }
        }
    }

    async _submitInpaint(form) {
        const submitBtn = this.$('#submit-inpaint-btn');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
        }

        this._setJobStatus({ message: 'Submitting job...', status: 'pending' });
        this._setGallery([]);

        try {
            const formData = new FormData(form);
            const imageFile = formData.get('image');
            const maskFile = formData.get('mask');
            if (!(imageFile instanceof File) || imageFile.size === 0) throw new Error('Image is required');
            if (!(maskFile instanceof File) || maskFile.size === 0) throw new Error('Mask is required');

            const fd = new FormData();
            ['character', 'control_type', 'steps', 'guidance_scale', 'strength', 'seed'].forEach((k) => {
                const v = formData.get(k);
                if (v !== null && v !== undefined && String(v).length > 0) {
                    fd.append(k, v);
                }
            });
            fd.append('no_preprocess', formData.get('no_preprocess') ? 'true' : 'false');
            fd.append('image', imageFile, imageFile.name);
            fd.append('mask', maskFile, maskFile.name);

            const controlFile = formData.get('control_image');
            if (controlFile instanceof File && controlFile.size > 0) {
                fd.append('control_image', controlFile, controlFile.name);
            }

            const resp = await api.action.inpaint(fd);
            const jobId = resp.job_id;
            const outputDir = resp.output_dir;

            this.activeJob = { jobId, outputDir, status: 'pending' };
            this._setJobStatus({ message: `Job submitted: ${jobId}`, status: 'pending', outputDir });
            await this._monitorJob(jobId, outputDir);
        } catch (error) {
            console.error('Failed to submit inpaint job:', error);
            this._setJobStatus({ message: error.message || 'Submit failed', status: 'failed' });
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Inpaint';
            }
        }
    }

    async _submitMultichar(form) {
        const submitBtn = this.$('#submit-multichar-btn');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
        }

        this._setJobStatus({ message: 'Submitting job...', status: 'pending' });
        this._setGallery([]);

        try {
            const formData = new FormData(form);
            const cfg = formData.get('config');
            if (!(cfg instanceof File) || cfg.size === 0) throw new Error('Config YAML is required');

            const fd = new FormData();
            fd.append('config', cfg, cfg.name);
            const timeout = String(formData.get('timeout') || '').trim();
            if (timeout) fd.append('timeout', timeout);

            const resp = await api.action.multichar(fd);
            const jobId = resp.job_id;
            const outputDir = resp.output_dir;

            this.activeJob = { jobId, outputDir, status: 'pending' };
            this._setJobStatus({ message: `Job submitted: ${jobId}`, status: 'pending', outputDir });
            await this._monitorJob(jobId, outputDir);
        } catch (error) {
            console.error('Failed to submit multichar job:', error);
            this._setJobStatus({ message: error.message || 'Submit failed', status: 'failed' });
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Run';
            }
        }
    }

    async _monitorJob(jobId, outputDir) {
        const endpoint = `/jobs/${jobId}/progress`;
        this.jobSseEndpoint = endpoint;

        api.jobs.streamProgress(
            jobId,
            (evt) => {
                if (!evt) return;
                const msg = evt.message || evt.stage || 'Running...';
                this._setJobStatus({ message: msg, status: 'running', outputDir });
            },
            async () => {
                if (this.jobSseEndpoint) {
                    api.closeSSE(this.jobSseEndpoint);
                    this.jobSseEndpoint = null;
                }
                this._setJobStatus({ message: 'Completed', status: 'completed', outputDir });
                await this._loadOutputs(outputDir);
            },
            (err) => {
                if (this.jobSseEndpoint) {
                    api.closeSSE(this.jobSseEndpoint);
                    this.jobSseEndpoint = null;
                }
                const msg = err?.error || err?.message || 'Failed';
                this._setJobStatus({ message: msg, status: 'failed', outputDir });
            }
        );
    }

    async _loadOutputs(outputDir) {
        const listImages = async (path) => {
            const listing = await api.results.list(path);
            const files = listing.files || [];
            const images = files.filter(f => f.type === 'file' && this._isImageFile(f.name));
            const dirs = files.filter(f => f.type === 'directory');
            return { listing, files, images, dirs };
        };

        try {
            let chosenPath = outputDir;
            let res = await listImages(outputDir);

            // Most tasks produce images in the root output dir.
            if (res.images.length === 0) {
                // Common nested outputs:
                // - animate: frames/
                // - extract_controls: controls/<type>/ and frames/
                const dirByName = new Map(res.dirs.map(d => [String(d.name), d]));

                const candidates = [];
                if (dirByName.has('frames')) candidates.push(dirByName.get('frames').path);

                if (dirByName.has('controls')) {
                    const controlsRoot = dirByName.get('controls').path;
                    try {
                        const controlsRes = await listImages(controlsRoot);
                        if (controlsRes.images.length > 0) {
                            candidates.push(controlsRoot);
                        } else if (controlsRes.dirs.length > 0) {
                            const prefer = ['pose', 'canny', 'softedge', 'lineart', 'tile'];
                            const pick =
                                prefer.map((n) => controlsRes.dirs.find(d => String(d.name) === n)).find(Boolean) ||
                                controlsRes.dirs[0];
                            if (pick) candidates.push(pick.path);
                        }
                    } catch (e) {
                        console.warn('Failed to browse controls output:', e);
                    }
                }

                if (dirByName.has('previews')) candidates.push(dirByName.get('previews').path);
                if (dirByName.has('passes')) candidates.push(dirByName.get('passes').path);

                for (const candidate of candidates) {
                    try {
                        const r = await listImages(candidate);
                        if (r.images.length > 0) {
                            res = r;
                            chosenPath = candidate;
                            break;
                        }
                    } catch (e) {
                        console.warn('Failed to browse candidate output path:', candidate, e);
                    }
                }
            }

            this.outputFiles = res.images;
            this.gallerySourcePath = chosenPath;
            this._setGallery(res.images, chosenPath);
        } catch (error) {
            console.error('Failed to load output files:', error);
            this._setJobStatus({ message: `Completed, but failed to browse outputs: ${error.message}`, status: 'completed', outputDir });
        }
    }

    _isImageFile(name) {
        const ext = String(name).split('.').pop().toLowerCase();
        return ['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(ext);
    }

    _setJobStatus({ message, status, outputDir }) {
        const container = this.$('#action-job-status');
        if (!container) return;

        const statusClass = {
            pending: 'badge-pending',
            running: 'badge-running',
            completed: 'badge-success',
            failed: 'badge-error'
        }[status] || 'badge-pending';

        const outputLink = outputDir
            ? `<a href="#/results" class="btn btn-sm btn-view" id="open-output-btn" data-output="${this.escapeHtml(outputDir)}">Browse Output</a>`
            : '';

        container.innerHTML = `
            <div class="card" style="padding: 16px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px;">
                    <div>
                        <div class="status-badge ${statusClass}">
                            <span>${this.escapeHtml(status || '')}</span>
                        </div>
                        <div style="margin-top: 8px; color: var(--color-text-secondary);">
                            ${this.escapeHtml(message || '')}
                        </div>
                        ${outputDir ? `<div style="margin-top: 6px; font-size: 12px;">${this.escapeHtml(outputDir)}</div>` : ''}
                    </div>
                    <div style="display: flex; gap: 8px;">
                        ${outputLink}
                    </div>
                </div>
            </div>
        `;

        const openOutputBtn = this.$('#open-output-btn');
        if (openOutputBtn) {
            openOutputBtn.addEventListener('click', (e) => {
                e.preventDefault();
                const out = openOutputBtn.dataset.output;
                if (out) {
                    this.setState('results.currentPath', out);
                    router.navigate('/results');
                }
            });
        }
    }

    _setGallery(files, sourcePath = null) {
        const container = this.$('#action-output-gallery');
        if (!container) return;

        this.gallerySourcePath = sourcePath;

        if (!files || files.length === 0) {
            container.innerHTML = '';
            return;
        }

        const items = files.map((f) => {
            const name = this.escapeHtml(f.name);
            const url = `${api.baseURL}/api/results/download?path=${encodeURIComponent(f.path)}`;
            const rawPath = String(f.path);
            return `
                <div class="card" style="padding: 8px;">
                    <a href="${url}" target="_blank" rel="noreferrer">
                        <img src="${url}" alt="${name}" style="width: 100%; height: auto; display: block;" />
                    </a>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                        <div style="font-size: 12px; color: var(--color-text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                            ${name}
                        </div>
                        <a class="btn btn-sm btn-download" href="${api.baseURL}/api/results/download?path=${encodeURIComponent(rawPath)}" download>
                            Download
                        </a>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="card" style="padding: 16px;">
                <h3 style="margin-top: 0;">Outputs</h3>
                ${sourcePath ? `<div style="margin-top: -8px; margin-bottom: 12px; font-size: 12px; color: var(--color-text-secondary);">${this.escapeHtml(sourcePath)}</div>` : ''}
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px;">
                    ${items}
                </div>
            </div>
        `;
    }
}
