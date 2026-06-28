/**
 * Jobs Page
 * Main page displaying the job list
 */

import BaseComponent from '../components/base-component.js';
import JobList from '../components/job-list.js';

export default class JobsPage extends BaseComponent {
    constructor() {
        super();
        this.jobList = null;
    }

    async onMount() {
        // Render page structure
        this.render(`
            <div class="page-container jobs-page">
                <div id="job-list-container"></div>
            </div>
        `);

        // Mount job list component
        const listContainer = this.$('#job-list-container');
        if (listContainer) {
            this.jobList = new JobList();
            await this.jobList.mount(listContainer);
        }
    }

    async onUnmount() {
        // Unmount job list
        if (this.jobList) {
            await this.jobList.unmount();
            this.jobList = null;
        }
    }
}
