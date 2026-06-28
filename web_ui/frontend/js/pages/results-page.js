import BaseComponent from '../components/base-component.js';
import ResultBrowser from '../components/result-browser.js';

export default class ResultsPage extends BaseComponent {
    constructor(path = '') {
        super();
        this.path = path;
        this.browser = null;
    }

    async onMount() {
        this.render(`
            <div class="page-container results-page">
                <div id="result-browser-container"></div>
            </div>
        `);

        const container = this.$('#result-browser-container');
        if (container) {
            this.browser = new ResultBrowser(this.path);
            await this.browser.mount(container);
        }
    }

    async onUnmount() {
        if (this.browser) {
            await this.browser.unmount();
            this.browser = null;
        }
    }
}
