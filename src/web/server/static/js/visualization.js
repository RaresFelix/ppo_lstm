class AgentVisualization {
    constructor(taskType) {
        this.taskType = taskType; // 'maze' or 'memory'
        this.currentRun = 0;
        this.currentFrame = 0;
        this.runs = [];
        this.maxFrames = 0;
        this.imageCache = new Map();
        this.preloadRadius = 5;

        // Get DOM elements with task-specific IDs
        this.mainFrame = document.getElementById(`mainFrame_${taskType}`);
        this.memoryHeatmap = document.getElementById(`memoryHeatmap_${taskType}`);
        this.frameSlider = document.getElementById(`frameSlider_${taskType}`);
        this.frameNumber = document.getElementById(`frameNumber_${taskType}`);
        this.maxFrame = document.getElementById(`maxFrame_${taskType}`);
        this.runInfo = document.getElementById(`runInfo_${taskType}`);

        this.setupEventListeners();
        this.loadRuns();
    }

    async loadRuns() {
        const response = await fetch(`/api/runs/${this.taskType}`);
        this.runs = await response.json();
        if (this.runs.length > 0) {
            this.updateVisualization();
        }
    }

    setupEventListeners() {
        document.getElementById(`prevRun_${this.taskType}`).addEventListener('click', () => this.changeRun(-1));
        document.getElementById(`nextRun_${this.taskType}`).addEventListener('click', () => this.changeRun(1));
        this.frameSlider.addEventListener('input', (e) => {
            this.currentFrame = parseInt(e.target.value);
            // Use a temporary preview while sliding
            this.updateVisualizationThrottled();
            // Preload images around new position
            this.preloadImages(this.currentFrame);
        });

        this.frameSlider.addEventListener('change', () => {
            // Final update when sliding stops
            this.updateVisualization();
        });
    }

    changeRun(delta) {
        this.currentRun = (this.currentRun + delta + this.runs.length) % this.runs.length;
        this.currentFrame = 0;
        this.updateVisualization();
    }

    updateVisualizationThrottled() {
        if (this.updateTimeout) clearTimeout(this.updateTimeout);
        this.updateTimeout = setTimeout(() => this.updateVisualization(), 50);
    }

    async preloadImages(centerFrame) {
        if (this.runs.length === 0) return;
        
        const run = this.runs[this.currentRun];
        
        for (let offset = -this.preloadRadius; offset <= this.preloadRadius; offset++) {
            const frame = Math.max(0, Math.min(centerFrame + offset, this.maxFrames));
            const paddedFrame = String(frame).padStart(4, '0');
            
            const mainUrl = `/api/frame/${this.taskType}/${run.id}/env/${paddedFrame}`;
            const memoryUrl = `/api/frame/${this.taskType}/${run.id}/memory/${paddedFrame}`;
            
            // Only preload if not already in cache
            for (const url of [mainUrl, memoryUrl]) {
                if (!this.imageCache.has(url)) {
                    const img = new Image();
                    img.src = url;
                    this.imageCache.set(url, img);
                }
            }
        }
    }

    updateVisualization() {
        if (this.runs.length === 0) return;

        const run = this.runs[this.currentRun];
        this.maxFrames = run.frames - 1;
        this.frameSlider.max = this.maxFrames;
        this.frameSlider.value = this.currentFrame;
        
        const paddedFrame = String(this.currentFrame).padStart(4, '0');
        
        this.mainFrame.src = `/api/frame/${this.taskType}/${run.id}/env/${paddedFrame}`;
        this.memoryHeatmap.src = `/api/frame/${this.taskType}/${run.id}/memory/${paddedFrame}`;
        
        this.frameNumber.textContent = this.currentFrame;
        this.maxFrame.textContent = this.maxFrames;
        this.runInfo.textContent = `Run ${this.currentRun + 1} of ${this.runs.length}`;
    }
}
