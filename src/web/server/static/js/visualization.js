class AgentVisualization {
    constructor(taskType) {
        this.taskType = taskType; // 'maze' or 'memory'
        this.currentRun = 0;
        this.currentFrame = 0;
        this.runs = [];
        this.maxFrames = 0;
        this.imageCache = new Map();
        this.preloadRadius = 20;
        this.loadingPromise = null;  // Add this line
        this.loadedRuns = new Set();  // Add this line
        this.isLoadingRuns = false;   // Add this line

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
            // Load current run first
            await this.preloadAllImages();
            this.updateVisualization();
            // Start loading other runs in the background
            this.loadRemainingRuns();
        }
    }

    async loadRemainingRuns() {
        if (this.isLoadingRuns) return;
        this.isLoadingRuns = true;

        try {
            for (let i = 0; i < this.runs.length; i++) {
                if (i !== this.currentRun && !this.loadedRuns.has(i)) {
                    const prevRun = this.currentRun;
                    this.currentRun = i;
                    await this.preloadAllImages();
                    this.loadedRuns.add(i);
                    this.currentRun = prevRun;
                }
            }
        } finally {
            this.isLoadingRuns = false;
        }
    }

    async preloadAllImages() {
        if (this.runs.length === 0) return;
        if (this.loadedRuns.has(this.currentRun)) return;
        
        const run = this.runs[this.currentRun];
        const totalFrames = run.frames;
        
        const loadPromises = [];
        
        for (let frame = 0; frame <= totalFrames - 1; frame++) {
            const paddedFrame = String(frame).padStart(4, '0');
            const mainUrl = `/api/frame/${this.taskType}/${run.id}/env/${paddedFrame}`;
            const memoryUrl = `/api/frame/${this.taskType}/${run.id}/memory/${paddedFrame}`;
            
            for (const url of [mainUrl, memoryUrl]) {
                if (!this.imageCache.has(url)) {
                    const promise = new Promise((resolve, reject) => {
                        const img = new Image();
                        img.onload = () => resolve();
                        img.onerror = () => reject();
                        img.src = url;
                        this.imageCache.set(url, img);
                    });
                    loadPromises.push(promise);
                }
            }
        }
        
        this.loadingPromise = Promise.all(loadPromises);
        await this.loadingPromise;

        this.loadedRuns.add(this.currentRun);
    }

    setupEventListeners() {
        document.getElementById(`prevRun_${this.taskType}`).addEventListener('click', () => this.changeRun(-1));
        document.getElementById(`nextRun_${this.taskType}`).addEventListener('click', () => this.changeRun(1));
        this.frameSlider.addEventListener('input', (e) => {
            this.currentFrame = parseInt(e.target.value);
            this.updateVisualization();  // Remove throttling, update immediately
        });

        // Remove the 'change' event listener since we update immediately on 'input'
    }

    async changeRun(delta) {
        const newRun = (this.currentRun + delta + this.runs.length) % this.runs.length;
        
        // If the new run isn't loaded yet, load it
        if (!this.loadedRuns.has(newRun)) {
            const prevRun = this.currentRun;
            this.currentRun = newRun;
            await this.preloadAllImages();
            // Continue loading remaining runs in the background
            this.loadRemainingRuns();
        }
        
        this.currentRun = newRun;
        this.currentFrame = 0;
        this.updateVisualization();
    }

    updateVisualization() {
        if (this.runs.length === 0) return;

        const run = this.runs[this.currentRun];
        this.maxFrames = run.frames - 1;
        this.frameSlider.max = this.maxFrames;
        this.frameSlider.value = this.currentFrame;
        
        const paddedFrame = String(this.currentFrame).padStart(4, '0');
        
        const mainUrl = `/api/frame/${this.taskType}/${run.id}/env/${paddedFrame}`;
        const memoryUrl = `/api/frame/${this.taskType}/${run.id}/memory/${paddedFrame}`;
        
        // Use cached images
        if (this.imageCache.has(mainUrl)) {
            this.mainFrame.src = this.imageCache.get(mainUrl).src;
        }
        if (this.imageCache.has(memoryUrl)) {
            this.memoryHeatmap.src = this.imageCache.get(memoryUrl).src;
        }
        
        this.frameNumber.textContent = this.currentFrame;
        this.maxFrame.textContent = this.maxFrames;
        this.runInfo.textContent = `Run ${this.currentRun + 1} of ${this.runs.length}`;
    }
}
