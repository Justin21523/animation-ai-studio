/**
 * Creative Studio Page
 *
 * Submit Creative Studio jobs (parody/analyze/voice) to the shared job queue.
 */

import BaseComponent from '../components/base-component.js';
import api from '../api/client.js';
import router from '../router.js';

export default class CreativePage extends BaseComponent {
    async onMount() {
        this.render(this.template());
        this.bindEvents();
    }

    template() {
        return `
            <div class="page-container creative-page">
                <div class="page-header">
                    <div>
                        <h2>Creative Studio</h2>
                        <div class="muted">Run parody, analysis, and voice as queued jobs.</div>
                    </div>
                </div>

                <div class="card" style="margin-bottom: 16px;">
                    <div class="card-header"><h3>Parody</h3></div>
                    <div class="card-body">
                        <div class="form-group">
                            <label class="form-label">Input Video Path</label>
                            <input class="form-control" id="parody-input" placeholder="/path/to/video.mp4" />
                        </div>
                        <div class="form-group">
                            <label class="form-label">Style</label>
                            <select class="form-control" id="parody-style">
                                <option value="dramatic">dramatic</option>
                                <option value="chaotic">chaotic</option>
                                <option value="wholesome">wholesome</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Duration (seconds, optional)</label>
                            <input class="form-control" id="parody-duration" placeholder="30" />
                        </div>
                        <div class="form-group">
                            <label class="form-label">Effects (comma-separated, optional)</label>
                            <input class="form-control" id="parody-effects" placeholder="zoom_punch,speed_ramp" />
                        </div>
                        <button class="btn btn-primary" id="parody-submit">Submit Parody Job</button>
                    </div>
                </div>

                <div class="card" style="margin-bottom: 16px;">
                    <div class="card-header"><h3>Analyze</h3></div>
                    <div class="card-body">
                        <div class="form-group">
                            <label class="form-label">Input Video Path</label>
                            <input class="form-control" id="analyze-input" placeholder="/path/to/video.mp4" />
                        </div>
                        <div class="form-group">
                            <label class="form-label">Options</label>
                            <div style="display:flex; gap:12px; flex-wrap:wrap;">
                                <label><input type="checkbox" id="analyze-visual" checked /> visual</label>
                                <label><input type="checkbox" id="analyze-audio" /> audio</label>
                                <label><input type="checkbox" id="analyze-context" /> context</label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Sample Rate</label>
                            <input class="form-control" id="analyze-sample-rate" value="30" />
                        </div>
                        <button class="btn btn-primary" id="analyze-submit">Submit Analyze Job</button>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header"><h3>Voice (TTS)</h3></div>
                    <div class="card-body">
                        <div class="form-group">
                            <label class="form-label">Character</label>
                            <input class="form-control" id="voice-character" placeholder="luca" />
                        </div>
                        <div class="form-group">
                            <label class="form-label">Text</label>
                            <input class="form-control" id="voice-text" placeholder="Silenzio, Bruno!" />
                        </div>
                        <div class="form-group">
                            <label class="form-label">Emotion</label>
                            <input class="form-control" id="voice-emotion" value="neutral" />
                        </div>
                        <div class="form-group">
                            <label class="form-label">Intensity</label>
                            <input class="form-control" id="voice-intensity" value="0.8" />
                        </div>
                        <button class="btn btn-primary" id="voice-submit">Submit Voice Job</button>
                    </div>
                </div>
            </div>
        `;
    }

    bindEvents() {
        this.$('#parody-submit')?.addEventListener('click', async () => {
            const input = this.$('#parody-input')?.value?.trim();
            const style = this.$('#parody-style')?.value;
            const durationRaw = this.$('#parody-duration')?.value?.trim();
            const effectsRaw = this.$('#parody-effects')?.value?.trim();
            if (!input) return alert('Missing input video path');

            const payload = {
                input_video_path: input,
                style,
                duration: durationRaw ? Number(durationRaw) : null,
                effects: effectsRaw ? effectsRaw.split(',').map(x => x.trim()).filter(Boolean) : null,
            };
            const res = await api.creative.parody(payload);
            router.navigate(`/jobs/${res.job_id}`);
        });

        this.$('#analyze-submit')?.addEventListener('click', async () => {
            const input = this.$('#analyze-input')?.value?.trim();
            if (!input) return alert('Missing input video path');
            const payload = {
                input_video_path: input,
                visual: !!this.$('#analyze-visual')?.checked,
                audio: !!this.$('#analyze-audio')?.checked,
                context: !!this.$('#analyze-context')?.checked,
                sample_rate: Number(this.$('#analyze-sample-rate')?.value || 30),
            };
            const res = await api.creative.analyze(payload);
            router.navigate(`/jobs/${res.job_id}`);
        });

        this.$('#voice-submit')?.addEventListener('click', async () => {
            const character = this.$('#voice-character')?.value?.trim();
            const text = this.$('#voice-text')?.value?.trim();
            if (!character || !text) return alert('Missing character/text');
            const payload = {
                character,
                text,
                emotion: this.$('#voice-emotion')?.value?.trim() || 'neutral',
                intensity: Number(this.$('#voice-intensity')?.value || 0.8),
            };
            const res = await api.creative.voice(payload);
            router.navigate(`/jobs/${res.job_id}`);
        });
    }
}
