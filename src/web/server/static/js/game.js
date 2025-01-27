const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d', { alpha: false });
let lastActionTime = 0;
const DEBOUNCE_TIME = 100; // ms

function displayGame(gameData) {
    if (!gameData) return;
    
    try {
        const width = gameData[0].length;
        const height = gameData.length;
        
        // Update canvas size to match the container size
        const container = canvas.parentElement;
        const containerSize = Math.min(container.clientWidth, 400);
        canvas.width = containerSize;
        canvas.height = containerSize;
        
        // Clear the entire canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Create the image data
        const buffer = new Uint8ClampedArray(width * height * 4);
        const data = new ImageData(buffer, width, height);
        
        let i = 0;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixel = gameData[y][x];
                buffer[i] = pixel[0];     // R
                buffer[i + 1] = pixel[1]; // G
                buffer[i + 2] = pixel[2]; // B
                buffer[i + 3] = 255;      // A
                i += 4;
            }
        }
        
        // Create a temporary canvas to hold the image data
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(data, 0, 0);
        
        // Calculate scaling and position to center the image
        const scale = Math.min(canvas.width / width, canvas.height / height);
        const scaledWidth = width * scale;
        const scaledHeight = height * scale;
        const x = (canvas.width - scaledWidth) / 2;
        const y = (canvas.height - scaledHeight) / 2;
        
        // Draw the scaled image centered on the main canvas
        ctx.imageSmoothingEnabled = false; // Keep pixel art sharp
        ctx.drawImage(tempCanvas, x, y, scaledWidth, scaledHeight);
        
    } catch (error) {
        console.error('Error displaying game:', error);
    }
}

async function performAction(action) {
    const now = Date.now();
    if (now - lastActionTime < DEBOUNCE_TIME) return;
    lastActionTime = now;

    try {
        const response = await fetch('/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action })
        });
        
        if (response.ok) {
            const gameData = await response.json();
            displayGame(gameData);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Add keyboard controls
document.addEventListener('keydown', (e) => {
    switch(e.key.toLowerCase()) {
        case 'a': performAction(0); break;
        case 'w': performAction(2); break;
        case 'd': performAction(1); break;
    }
});

// Add click listeners to all buttons
document.querySelectorAll('.action-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        performAction(parseInt(btn.dataset.action));
    });
});

// Add view toggle handler
document.getElementById('viewToggle').addEventListener('change', async (e) => {
    try {
        const response = await fetch('/toggle_view', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_view: e.target.checked })
        });
        
        if (response.ok) {
            const gameData = await response.json();
            displayGame(gameData);
        }
    } catch (error) {
        console.error('Error:', error);
    }
});

// Add new visualization controls
const frameSlider = document.getElementById('frameSlider');
const frameNumber = document.getElementById('frameNumber');
const mazeFrame = document.getElementById('mazeFrame');
const memoryHeatmap = document.getElementById('memoryHeatmap');

// Run management
let currentRunIndex = 0;
let runs = [];

// Add image caching
const imageCache = {
    maze: {},
    memory: {}
};

async function preloadImages(runId, frameCount) {
    const promises = [];
    for (let i = 0; i < frameCount; i++) {
        const frameStr = i.toString().padStart(4, '0');
        
        // Preload maze image
        if (!imageCache.maze[`${runId}_${i}`]) {
            const mazePromise = new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    imageCache.maze[`${runId}_${i}`] = img;
                    resolve();
                };
                img.onerror = reject;
                img.src = `/static/runs/${runId}/images/env_${frameStr}.png`;
            });
            promises.push(mazePromise);
        }
        
        // Preload memory image
        if (!imageCache.memory[`${runId}_${i}`]) {
            const memoryPromise = new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    imageCache.memory[`${runId}_${i}`] = img;
                    resolve();
                };
                img.onerror = reject;
                img.src = `/static/runs/${runId}/images/memory_${frameStr}.png`;
            });
            promises.push(memoryPromise);
        }
    }
    
    try {
        await Promise.all(promises);
        console.log(`Preloaded all images for run ${runId}`);
    } catch (error) {
        console.error('Error preloading images:', error);
    }
}

async function preloadInitialFrames() {
    const promises = [];
    for (const run of runs) {
        const frameStr = '0000';
        
        // Only preload first frame of each run if not already cached
        if (!imageCache.maze[`${run.id}_0`]) {
            promises.push(new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    imageCache.maze[`${run.id}_0`] = img;
                    resolve();
                };
                img.onerror = reject;
                img.src = `/static/runs/${run.id}/images/env_${frameStr}.png`;
            }));
        }
        
        if (!imageCache.memory[`${run.id}_0`]) {
            promises.push(new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    imageCache.memory[`${run.id}_0`] = img;
                    resolve();
                };
                img.onerror = reject;
                img.src = `/static/runs/${run.id}/images/memory_${frameStr}.png`;
            }));
        }
    }
    
    try {
        await Promise.all(promises);
        console.log('Preloaded initial frames for all runs');
    } catch (error) {
        console.error('Error preloading initial frames:', error);
    }
}

function updateImages(frame) {
    const run = runs[currentRunIndex];
    const mazeImg = imageCache.maze[`${run.id}_${frame}`];
    const memoryImg = imageCache.memory[`${run.id}_${frame}`];
    
    if (mazeImg) mazeFrame.src = mazeImg.src;
    if (memoryImg) memoryHeatmap.src = memoryImg.src;
}

async function loadRuns() {
    const response = await fetch('/get_runs');
    runs = await response.json();
    if (runs.length > 0) {
        // Preload initial frames first
        await preloadInitialFrames();
        updateRunDisplay();
        updateFrameSlider();
    }
}

async function updateRunDisplay() {
    const run = runs[currentRunIndex];
    document.getElementById('runInfo').textContent = run.id;
    
    // Start preloading images immediately
    await preloadImages(run.id, run.frame_count);
    updateFrameSlider();
}

function updateFrameSlider() {
    const run = runs[currentRunIndex];
    const frameSlider = document.getElementById('frameSlider');
    frameSlider.max = run.frame_count - 1;
    document.getElementById('maxFrame').textContent = run.frame_count - 1;
    
    // Reset to first frame
    frameSlider.value = 0;
    frameNumber.textContent = 0;
    updateImages(0);
}

document.getElementById('prevRun').addEventListener('click', () => {
    currentRunIndex = (currentRunIndex - 1 + runs.length) % runs.length;
    updateRunDisplay();
});

document.getElementById('nextRun').addEventListener('click', () => {
    currentRunIndex = (currentRunIndex + 1) % runs.length;
    updateRunDisplay();
});

frameSlider.addEventListener('input', function(e) {
    const frame = parseInt(e.target.value);
    frameNumber.textContent = frame;
    updateImages(frame);
});

loadRuns();