import BaseComponent from '../components/base-component.js';
import SystemDashboard from '../components/system-dashboard.js';

export default class SystemPage extends BaseComponent {
    constructor() {
        super();
        this.dashboard = null;
    }

    async onMount() {
        this.render(`
            <div class="page-container system-page">
                <div id="system-dashboard-container"></div>
            </div>
        `);

        const container = this.$('#system-dashboard-container');
        if (container) {
            this.dashboard = new SystemDashboard();
            await this.dashboard.mount(container);
        }
    }

    async onUnmount() {
        if (this.dashboard) {
            await this.dashboard.unmount();
            this.dashboard = null;
        }
    }
}
