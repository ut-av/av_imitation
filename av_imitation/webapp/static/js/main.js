const { createApp, ref, computed, onMounted, watch, onBeforeUpdate, nextTick } = Vue;

// IndexedDB Helper
const FrameDB = {
    dbName: 'AVImitationDB',
    storeName: 'frames',
    version: 1,
    db: null,

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = (event) => {
                console.error("IndexedDB error:", event.target.error);
                reject(event.target.error);
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    const store = db.createObjectStore(this.storeName, { keyPath: 'id' });
                    store.createIndex('bag', 'bag', { unique: false });
                }
            };
        });
    },

    async saveFrame(bag, timestamp, blob) {
        if (!this.db) await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction([this.storeName], 'readwrite');
            const store = tx.objectStore(this.storeName);
            const id = `${bag}_${timestamp.toFixed(3)}`;
            const request = store.put({ id, bag, timestamp, blob });

            request.onsuccess = () => resolve();
            request.onerror = (e) => reject(e.target.error);
        });
    },

    async getFrame(bag, timestamp) {
        if (!this.db) await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction([this.storeName], 'readonly');
            const store = tx.objectStore(this.storeName);
            const id = `${bag}_${timestamp.toFixed(3)}`;
            const request = store.get(id);

            request.onsuccess = () => resolve(request.result ? request.result.blob : null);
            request.onerror = (e) => reject(e.target.error);
        });
    },

    async getFramesForBag(bag) {
        if (!this.db) await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction([this.storeName], 'readonly');
            const store = tx.objectStore(this.storeName);
            const index = store.index('bag');
            const request = index.getAll(IDBKeyRange.only(bag));

            request.onsuccess = () => resolve(request.result);
            request.onerror = (e) => reject(e.target.error);
        });
    },

    async clear() {
        if (!this.db) await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction([this.storeName], 'readwrite');
            const store = tx.objectStore(this.storeName);
            const request = store.clear();

            request.onsuccess = () => resolve();
            request.onerror = (e) => reject(e.target.error);
        });
    },

    async clearBag(bag) {
        if (!this.db) await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction([this.storeName], 'readwrite');
            const store = tx.objectStore(this.storeName);
            const index = store.index('bag');
            const request = index.openKeyCursor(IDBKeyRange.only(bag));

            request.onsuccess = () => {
                const cursor = request.result;
                if (cursor) {
                    store.delete(cursor.primaryKey);
                    cursor.continue();
                } else {
                    resolve();
                }
            };
            request.onerror = (e) => reject(e.target.error);
        });
    },

    async getSize() {
        if (!this.db) await this.init();
        return new Promise((resolve, reject) => {
            const tx = this.db.transaction([this.storeName], 'readonly');
            const store = tx.objectStore(this.storeName);
            const request = store.getAll(); // This might be heavy for large DBs, but simple for now

            request.onsuccess = () => {
                let size = 0;
                if (request.result) {
                    request.result.forEach(item => {
                        if (item.blob) size += item.blob.size;
                    });
                }
                resolve(size);
            };
            request.onerror = (e) => reject(e.target.error);
        });
    }
};

createApp({
    compilerOptions: {
        delimiters: ['[[', ']]']
    },
    setup() {
        const bags = ref([]);
        const currentBag = ref(null);
        const loadingBags = ref(false);
        const duration = ref(0);
        const startTime = ref(0);
        const currentTime = ref(0);
        const isPlaying = ref(false);
        const description = ref("");
        const cuts = ref([]); // Array of {start, end}
        const activeCutStart = ref(null);
        const tags = ref([]);
        const allTags = ref([]);

        // Robust Image Loading
        const sampleImageRefs = ref([]);
        const collectImageRef = (el) => {
            if (el) sampleImageRefs.value.push(el);
        };
        onBeforeUpdate(() => {
            sampleImageRefs.value = [];
        });

        // Navigation State
        const currentStep = ref(1);

        // Processing State
        const processingBag = ref("");
        const defaultOptions = (() => {
            const defaults = {
                resize: false,
                width: 320,
                height: 240,
                channels: 'rgb',
                canny: false,

                depth: false,
            };

            const saved = localStorage.getItem('processingOptions');
            if (saved) {
                try {
                    const parsed = JSON.parse(saved);
                    // Merge saved options
                    return { ...defaults, ...parsed };
                } catch (e) {
                    console.error("Failed to parse saved options", e);
                    return defaults;
                }
            }
            return defaults;
        })();

        const options = ref(defaultOptions);

        // Persist options
        watch(options, (newVal) => {
            localStorage.setItem('processingOptions', JSON.stringify(newVal));
        }, { deep: true });
        const isProcessing = ref(false);
        const processingStatus = ref("");

        // Dataset Builder State
        const processedBags = ref([]);
        const loadingProcessedBags = ref(false);
        const selectedProcessedBags = ref([]);
        const datasetName = ref("dataset_v1");
        const datasetOptions = ref({
            historyRate: 5.0,
            historyDuration: 5.0,
            futureRate: 5.0,
            futureDuration: 3.0
        });
        const filterResolution = ref("");
        const filterChannels = ref("");
        const filterCanny = ref(false);
        const filterDepth = ref(false);
        const filterTags = ref("");

        const isGenerating = ref(false);
        const generationStatus = ref("");

        // Visualization State
        const datasets = ref([]);
        const loadingDatasets = ref(false);
        const selectedDataset = ref(null);
        const datasetData = ref(null);
        const currentVizBag = ref(null);
        const currentSampleIndex = ref(0);

        // Analysis State
        const selectedAnalysisDataset = ref(null);
        const velocityChart = ref(null);
        const curvatureChart = ref(null);

        const timeline = ref(null);
        const timelineScrollContainer = ref(null);
        const timelineContent = ref(null);
        let playInterval = null;
        let previewInterval = null;
        const previewTime = ref(0);

        // Timeline State
        const zoomLevel = ref(10); // pixels per second
        const selectedCutIndex = ref(-1);
        const selectedHandle = ref(null); // 'start' or 'end'

        // Drag state - only active when drag initiated from a handle
        const isDraggingHandle = ref(false);
        const draggedCutIndex = ref(-1);
        const draggedHandleType = ref(null); // 'start' or 'end'

        // Context menu state
        const showContextMenu = ref(false);
        const contextMenuX = ref(0);
        const contextMenuY = ref(0);
        const contextMenuCutIndex = ref(-1);

        // History & Reset
        const history = ref([]);
        const historyIndex = ref(-1);
        const originalCuts = ref([]);

        // Settings State
        const showSettings = ref(false);
        // Default to true if not set (null !== 'false' is true)
        const persistCache = ref(localStorage.getItem('persistCache') !== 'false');
        const cacheSize = ref(0);

        const cacheSizeMB = computed(() => (cacheSize.value / (1024 * 1024)).toFixed(2));

        watch(persistCache, (newVal) => {
            localStorage.setItem('persistCache', newVal);
        });

        const updateCacheSize = async () => {
            try {
                cacheSize.value = await FrameDB.getSize();
            } catch (e) {
                console.error("Failed to get cache size", e);
            }
        };

        const clearCache = async () => {
            if (confirm("Are you sure you want to clear the cache?")) {
                await FrameDB.clear();
                await updateCacheSize();
                // Also clear memory cache if we are clearing everything
                frameCache.value.clear();
            }
        };

        // Fetch bags on mount
        onMounted(async () => {
            loadingBags.value = true;
            try {
                await FrameDB.init();
                await updateCacheSize();
                await fetchTags();
                const res = await fetch('/api/bags');
                bags.value = await res.json();
            } catch (e) {
                console.error("Failed to load bags", e);
            } finally {
                loadingBags.value = false;
            }
        });

        const fetchTags = async () => {
            try {
                const res = await fetch('/api/tags');
                allTags.value = await res.json();
            } catch (e) {
                console.error("Failed to load tags", e);
            }
        };

        const addTag = async (tagName) => {
            if (!tagName || !currentBag.value) return;

            // Add to current bag if not exists
            if (!tags.value.includes(tagName)) {
                tags.value.push(tagName);
                saveMetadata();
            }

            // Add to global tags if not exists
            if (!allTags.value.includes(tagName)) {
                try {
                    const res = await fetch('/api/tags', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ tags: [tagName] })
                    });
                    if (res.ok) {
                        allTags.value = await res.json();
                    }
                } catch (e) {
                    console.error("Failed to add tag globally", e);
                }
            }
        };

        const removeTag = (tagName) => {
            if (!currentBag.value) return;
            const idx = tags.value.indexOf(tagName);
            if (idx !== -1) {
                tags.value.splice(idx, 1);
                saveMetadata();
            }
        };

        const frameCache = ref(new Map()); // Key: timestamp.toFixed(3), Value: blobUrl
        const frameTimestamps = ref([]); // Sorted array of timestamps
        const preloadController = ref(null);
        const bufferingProgress = ref(0);

        const currentFrameUrl = computed(() => {
            if (!currentBag.value) return null;

            // Determine which time to show
            let timeToShow = currentTime.value;
            if (bufferingProgress.value < 100 && !isPlaying.value) {
                // Show preview loop if buffering and not playing
                timeToShow = previewTime.value;
            }

            // Find nearest timestamp in cache
            // Since frameTimestamps is sorted, we can use binary search or just find closest
            // For now, simple iteration or find is okay if array isn't huge, but binary search is better.
            // Let's just find the closest one.

            if (frameTimestamps.value.length === 0) return null;

            // Simple linear search for nearest (optimization: binary search)
            // Or just use the one that matches toFixed(1) if we want to be loose?
            // No, we want precise lookup.

            // Optimization: Binary search
            let l = 0, r = frameTimestamps.value.length - 1;
            let closest = frameTimestamps.value[0];
            let minDiff = Math.abs(timeToShow - closest);

            while (l <= r) {
                const m = Math.floor((l + r) / 2);
                const val = frameTimestamps.value[m];
                const diff = Math.abs(timeToShow - val);

                if (diff < minDiff) {
                    minDiff = diff;
                    closest = val;
                }

                if (val < timeToShow) {
                    l = m + 1;
                } else {
                    r = m - 1;
                }
            }

            // If the closest frame is too far away (e.g. > 0.2s), maybe don't show it?
            // But for "nearest neighbor" playback, we usually want to show it.

            const key = closest.toFixed(3);
            if (frameCache.value.has(key)) {
                return frameCache.value.get(key);
            }

            // Fallback to direct URL (will be slow/cancelled if not cached)
            return `/api/bag/${currentBag.value}/frame/${timeToShow.toFixed(2)}?_=${Date.now()}`;
        });

        const formattedCurrentTime = computed(() => {
            if (!startTime.value) return "00:00:00";
            const date = new Date((startTime.value + currentTime.value) * 1000);
            return date.toLocaleString();
        });

        const preloadBag = async (bagName, durationSec, expectedFrames) => {
            if (preloadController.value) {
                preloadController.value.abort();
            }
            const controller = new AbortController();
            preloadController.value = controller;

            frameCache.value.clear();
            frameTimestamps.value = [];
            bufferingProgress.value = 0;

            // 1. If persistence is ON, try to load from DB first
            if (persistCache.value) {
                try {
                    const cachedFrames = await FrameDB.getFramesForBag(bagName);
                    cachedFrames.forEach(frame => {
                        const key = frame.timestamp.toFixed(3);
                        const url = URL.createObjectURL(frame.blob);
                        frameCache.value.set(key, url);
                        frameTimestamps.value.push(frame.timestamp);
                    });

                    frameTimestamps.value.sort((a, b) => a - b);

                    // Check if we have enough frames to skip streaming
                    // Use exact expectedFrames if provided
                    if (expectedFrames && frameCache.value.size >= expectedFrames * 0.95) {
                        console.log(`Loaded ${frameCache.value.size} frames from cache (expected ~${expectedFrames}). Skipping stream.`);
                        bufferingProgress.value = 100;
                        loadingBags.value = false; // Just in case
                        return;
                    }

                } catch (e) {
                    console.error("Error loading from DB", e);
                }
            }

            // Start preview loop (same as before)
            if (previewInterval) clearInterval(previewInterval);
            let previewIndex = 0;
            previewInterval = setInterval(() => {
                if (bufferingProgress.value >= 100) {
                    clearInterval(previewInterval);
                    return;
                }
                previewTime.value = (previewIndex % (Math.floor(durationSec) + 1));
                previewIndex++;
            }, 1000);

            try {
                const response = await fetch(`/api/bag/${bagName}/stream_frames`, {
                    signal: controller.signal
                });

                if (!response.ok) {
                    throw new Error(`Stream failed: ${response.statusText}`);
                }

                const reader = response.body.getReader();
                let receivedLength = 0;
                let chunks = []; // Array of Uint8Array

                // We need a way to process chunks as they come.
                // We'll maintain a "buffer" of unprocessed bytes.
                // Since concatenating large arrays is expensive, we can just keep a list of chunks
                // and a pointer, or use a small buffer for the header.

                // Let's use a simpler approach: Append to a buffer. 
                // Warning: This might grow large if we don't slice it.
                // Better: Keep a "working buffer" that we slice from.

                let buffer = new Uint8Array(0);

                const appendBuffer = (newChunk) => {
                    const tmp = new Uint8Array(buffer.length + newChunk.length);
                    tmp.set(buffer, 0);
                    tmp.set(newChunk, buffer.length);
                    buffer = tmp;
                };

                // Estimate total size? We don't know it.
                // We can use duration to estimate progress?
                // Or just count frames?
                // Let's assume 10Hz * duration = total frames.
                // const estimatedFrames = durationSec * 10;
                let framesLoaded = 0;
                let totalFrames = 0;
                let readHeader = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    appendBuffer(value);

                    // Read Header (Total Frames)
                    if (!readHeader) {
                        if (buffer.length < 4) continue; // Wait for more data
                        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                        totalFrames = view.getUint32(0, true);
                        buffer = buffer.slice(4);
                        readHeader = true;
                        // console.log("Total frames:", totalFrames);
                    }

                    // Process buffer
                    while (true) {
                        // Need at least 12 bytes for header (8 timestamp + 4 size)
                        if (buffer.length < 12) break;

                        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                        const timestamp = view.getFloat64(0, true); // little-endian
                        const size = view.getUint32(8, true);

                        // Check if we have the full image
                        if (buffer.length < 12 + size) break;

                        // Extract image
                        const imageData = buffer.slice(12, 12 + size);

                        // Create Blob
                        const blob = new Blob([imageData], { type: 'image/jpeg' });
                        const url = URL.createObjectURL(blob);
                        const key = timestamp.toFixed(3);

                        frameCache.value.set(key, url);
                        frameTimestamps.value.push(timestamp);
                        // Keep sorted? Or sort at end? 
                        // Sorting every frame is slow. 
                        // Since stream is sequential, we can just push.

                        if (persistCache.value) {
                            FrameDB.saveFrame(bagName, timestamp, blob).catch(e => console.warn("Failed to save frame", e));
                        }

                        // Remove processed part from buffer
                        buffer = buffer.slice(12 + size);

                        framesLoaded++;
                        // Update progress
                        if (totalFrames > 0) {
                            const progress = Math.min(100, Math.round((framesLoaded / totalFrames) * 100));
                            bufferingProgress.value = progress;
                        }
                    }
                }

                // Sort timestamps after streaming
                frameTimestamps.value.sort((a, b) => a - b);

                bufferingProgress.value = 100;
                if (persistCache.value) {
                    updateCacheSize();
                }

            } catch (e) {
                if (e.name === 'AbortError') {
                    console.log("Stream aborted");
                } else {
                    console.error("Stream error", e);
                }
            }
        };

        const telemetry = ref([]);

        const currentTelemetry = computed(() => {
            if (!telemetry.value || telemetry.value.length === 0) {
                return { steer: 0, throttle: 0, l1: false, l2: 0, velocity: 0, curvature: 0 };
            }

            // Find closest telemetry point
            // Since telemetry is sorted by time, we can use binary search or just findIndex
            // For simplicity and since array might be large, let's try a simple approach first
            // or binary search if performance is needed.
            // Given 100Hz data for 10 mins = 60000 points. Array.find might be slow if called every frame.
            // Let's use a simple index tracking since playback is sequential usually.

            // Optimization: Cache last index?
            // For now, let's just find the first point > currentTime and take the one before it.

            // Binary search implementation for performance
            let low = 0;
            let high = telemetry.value.length - 1;
            let idx = 0;

            while (low <= high) {
                const mid = Math.floor((low + high) / 2);
                if (telemetry.value[mid].time < currentTime.value) {
                    idx = mid;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }

            return telemetry.value[idx] || { steer: 0, throttle: 0, l1: false, l2: 0, velocity: 0, curvature: 0 };
        });

        const getBarStyle = (val) => {
            // val is -1 to 1
            // Center is 50%
            const pct = Math.abs(val) * 50; // 0 to 50%
            const left = val < 0 ? 50 - pct : 50;
            return {
                left: `${left}%`,
                width: `${pct}%`
            };
        };

        // Timeline Helpers
        const timelineWidth = computed(() => duration.value * zoomLevel.value);

        const ticks = computed(() => {
            if (!duration.value) return [];
            const t = [];
            // Determine interval
            // Target ~100px per major tick
            const targetPx = 100;
            const rawInterval = targetPx / zoomLevel.value;

            // Snap to nice numbers: 1, 2, 5, 10, 30, 60
            const niceIntervals = [1, 2, 5, 10, 30, 60];
            let interval = niceIntervals[0];
            for (const nice of niceIntervals) {
                if (nice >= rawInterval) {
                    interval = nice;
                    break;
                }
            }
            if (rawInterval > 60) interval = 60;

            for (let i = 0; i <= duration.value; i += interval) {
                t.push({
                    time: i,
                    left: i * zoomLevel.value,
                    major: true,
                    label: i % 60 === 0 ? `${i / 60}m` : `${i}s`
                });
            }
            return t;
        });

        const setZoom = (level) => {
            // Try to keep current time centered or at least visible
            const container = timelineScrollContainer.value;
            const centerTime = currentTime.value;

            zoomLevel.value = level;

            // Wait for DOM update then scroll
            setTimeout(() => {
                if (container) {
                    const scrollPos = (centerTime * zoomLevel.value) - (container.clientWidth / 2);
                    container.scrollLeft = Math.max(0, scrollPos);
                }
            }, 0);
        };

        const handleScroll = (e) => {
            // Optional: sync things if needed
        };

        const handleWheel = (e) => {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                let newZoom = zoomLevel.value * delta;
                newZoom = Math.max(10, Math.min(200, newZoom));

                // Snap to presets if close? Nah, just set it.
                // Or just step through presets: 10, 20, 50, 100
                const presets = [10, 20, 50, 100, 200];
                if (e.deltaY < 0) { // Zoom in
                    const next = presets.find(p => p > zoomLevel.value);
                    if (next) zoomLevel.value = next;
                } else { // Zoom out
                    const prev = [...presets].reverse().find(p => p < zoomLevel.value);
                    if (prev) zoomLevel.value = prev;
                }
            }
        };

        const selectCut = (index, handle) => {
            selectedCutIndex.value = index;
            selectedHandle.value = handle;
        };

        const handleHandleMouseDown = (index, handleType, event) => {
            event.stopPropagation();
            event.preventDefault();

            // Initiate drag mode
            isDraggingHandle.value = true;
            draggedCutIndex.value = index;
            draggedHandleType.value = handleType;

            // Select this cut
            selectedCutIndex.value = index;
            selectedHandle.value = handleType;
        };

        const showContextMenuFor = (index, event) => {
            event.preventDefault();
            contextMenuX.value = event.clientX;
            contextMenuY.value = event.clientY;
            contextMenuCutIndex.value = index;
            showContextMenu.value = true;
        };

        const deleteSelectedCut = () => {
            if (contextMenuCutIndex.value !== -1) {
                pushHistory();
                cuts.value.splice(contextMenuCutIndex.value, 1);
                selectedCutIndex.value = -1;
            }
            showContextMenu.value = false;
        };

        const handleRightClick = (index, event) => {
            showContextMenuFor(index, event);
        };

        // History
        const pushHistory = () => {
            // Remove future history if we are in middle
            if (historyIndex.value < history.value.length - 1) {
                history.value = history.value.slice(0, historyIndex.value + 1);
            }
            // Deep copy cuts
            history.value.push(JSON.parse(JSON.stringify(cuts.value)));
            if (history.value.length > 1000) history.value.shift();
            historyIndex.value = history.value.length - 1;
        };

        const undo = () => {
            if (historyIndex.value >= 0) {
                // Wait, if we are at the end, we need to save current state first?
                // Standard undo: 
                // 1. State A. Push A.
                // 2. Change to B. Push B.
                // 3. Undo -> Restore A.

                // My pushHistory implementation is "push before change".
                // So history contains [State0, State1, State2].
                // Current state is State3 (not in history yet?).

                // Let's simplify: History stores snapshots.
                // When we undo, we go back one snapshot.

                // Actually, let's just save the *previous* state before modification.

                const prev = history.value[historyIndex.value];
                if (prev) {
                    cuts.value = JSON.parse(JSON.stringify(prev));
                    historyIndex.value--;
                }
            }
        };

        // We need to push history *before* making changes.
        // Helper to wrap changes? Or just call pushHistory() manually.

        const resetCuts = () => {
            if (confirm("Reset cuts to original values?")) {
                pushHistory();
                cuts.value = JSON.parse(JSON.stringify(originalCuts.value));
            }
        };

        const previewImageUrl = ref(null);
        const previewStatus = ref('');

        const startPreview = async () => {
            if (!currentBag.value) return;

            if (previewImageUrl.value) {
                URL.revokeObjectURL(previewImageUrl.value);
            }

            previewStatus.value = "Generating preview...";
            previewImageUrl.value = null;

            const payload = {
                bag_name: currentBag.value,
                timestamp: currentTime.value,
                options: {
                    channels: options.value.channels,
                    canny: options.value.canny,

                    depth: options.value.depth
                }
            };

            if (options.value.resize) {
                payload.options.resolution = [options.value.width, options.value.height];
            }

            try {
                const res = await fetch('/api/preview_processing', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (res.ok) {
                    const blob = await res.blob();
                    previewImageUrl.value = URL.createObjectURL(blob);
                    previewStatus.value = "";
                } else {
                    const data = await res.json();
                    previewStatus.value = "Error: " + (data.error || "Unknown error");
                }
            } catch (e) {
                console.error("Preview error", e);
                previewStatus.value = "Error generating preview.";
            }
        };

        const selectBag = async (bag) => {
            // bag is now an object
            currentBag.value = bag.name;
            currentTime.value = 0;

            if (previewImageUrl.value) {
                startPreview();
            } else {
                previewStatus.value = '';
            }
            isPlaying.value = false;
            if (playInterval) clearInterval(playInterval); // Reset play state

            // Reset processing state
            processingStatus.value = '';
            processingProgress.value = 0;
            isProcessing.value = false;
            if (processingPollInterval) clearInterval(processingPollInterval);

            // Check for existing processing
            checkExistingProcessing();

            // Load metadata
            description.value = bag.description || '';
            cuts.value = [];
            activeCutStart.value = null;
            telemetry.value = [];
            selectedCutIndex.value = -1;
            tags.value = bag.tags || [];
            history.value = [];
            historyIndex.value = -1;

            // Stop any existing preload
            if (preloadController.value) {
                preloadController.value.abort();
            }
            frameCache.value.clear();
            bufferingProgress.value = 0;

            try {
                const res = await fetch(`/api/bag/${bag.name}/info`);
                const data = await res.json();
                duration.value = data.info.duration;
                startTime.value = data.info.start_time;

                if (data.user_meta) {
                    description.value = data.user_meta.description || "";
                    cuts.value = data.user_meta.cuts || [];
                    tags.value = data.user_meta.tags || [];
                } else {
                    // If no user meta, maybe we have default cuts from backend?
                    // The backend sends `cuts` in bag list, but maybe not in info?
                    // Let's assume `data.user_meta.cuts` is the source of truth.
                    // If it's empty, we might want to use the "L1" cuts if they exist?
                    // The backend `get_bag_info` returns `user_meta`.
                    // If we want the "original" L1 cuts, we might need to fetch them or calculate them.
                    // For now, let's assume the backend provides them or we save them initially.
                    // Wait, `originalCuts` should be the L1 cuts.
                    // If `user_meta` exists, `cuts` are user edits.
                    // Where do we get L1 cuts?
                    // In `bag-list`, we use `bag.cuts`. Those are generated by `generate_default_cuts`.
                    // We should probably grab those.
                    const bagInList = bags.value.find(b => b.name === bag.name);
                    if (bagInList && bagInList.cuts) {
                        originalCuts.value = JSON.parse(JSON.stringify(bagInList.cuts));
                        if (!data.user_meta || !data.user_meta.cuts) {
                            cuts.value = JSON.parse(JSON.stringify(bagInList.cuts));
                        }
                    } else {
                        originalCuts.value = [];
                    }
                }

                if (data.telemetry) {
                    telemetry.value = data.telemetry;
                }

                // Start preloading
                // Pass image_count if available, otherwise fallback to duration estimation
                const imageCount = data.info.image_count || Math.ceil(duration.value * 10);
                preloadBag(bag.name, duration.value, imageCount);

            } catch (e) {
                console.error("Failed to load bag info", e);
            }
        };

        const deleteBag = async (bag) => {
            if (confirm(`Are you sure you want to delete ${bag.name}? This cannot be undone.`)) {
                try {
                    // Clear cache first
                    await FrameDB.clearBag(bag.name); // Need to implement clearBag in FrameDB or just clear all? 
                    // Actually FrameDB doesn't have clearBag yet. Let's add it or just ignore for now as it will be orphaned.
                    // Better to clean up.

                    const res = await fetch(`/api/bag/${bag.name}`, { method: 'DELETE' });
                    if (res.ok) {
                        // Remove from list
                        bags.value = bags.value.filter(b => b.name !== bag.name);
                        if (currentBag.value === bag.name) {
                            currentBag.value = null;
                        }
                    } else {
                        alert("Failed to delete bag");
                    }
                } catch (e) {
                    console.error("Error deleting bag", e);
                    alert("Error deleting bag");
                }
            }
        };

        const getSparklinePoints = (bag) => {
            if (!bag.duration || !bag.cuts || bag.cuts.length === 0) return "";

            // Generate points for SVG polyline
            // We want a line that is high when "recording" (L1 pressed) and low when "cut" (L1 not pressed)
            // Wait, the requirement says "cuts from the joystick L1 button".
            // Usually "cuts" means the parts we KEEP or the parts we REMOVE?
            // In the code: "If L1 is pressed, we are 'recording'. If we were in a cut, close it."
            // So `cuts` array seems to store the intervals where L1 was NOT pressed (or maybe the other way around?)
            // Let's re-read `generate_default_cuts` in app.py.
            // "If L1 is pressed, we are 'recording'... if current_cut_start is not None: cuts.append..."
            // So `cuts` are the intervals where L1 was PRESSED? No.
            // "L1 not pressed. We should be in a cut. if current_cut_start is None: current_cut_start = t_sec"
            // So `cuts` are the intervals where L1 is NOT pressed.
            // So we want the line to be LOW during `cuts` and HIGH otherwise.

            const width = 100;
            const height = 20;
            const points = [];

            // Start high (recording) by default? Or low?
            // "If the bag starts and L1 is NOT held, we start a cut at 0" -> Low
            // Let's assume High (recording) is the baseline state we want to visualize as "active".

            let currentX = 0;

            // Sort cuts just in case
            const sortedCuts = [...bag.cuts].sort((a, b) => a.start - b.start);

            points.push(`0,0`); // Start top-left (High)

            sortedCuts.forEach(cut => {
                const startX = (cut.start / bag.duration) * width;
                const endX = (cut.end / bag.duration) * width;

                // Line stays high until start of cut
                points.push(`${startX},0`);
                // Drop to low
                points.push(`${startX},${height}`);
                // Stay low until end of cut
                points.push(`${endX},${height}`);
                // Go back high
                points.push(`${endX},0`);
            });

            // Finish at end
            points.push(`${width},0`);

            return points.join(" ");
        };

        const togglePlay = () => {
            if (bufferingProgress.value < 100) return;
            isPlaying.value = !isPlaying.value;
            if (isPlaying.value) {
                playInterval = setInterval(() => {
                    currentTime.value += 0.1; // 10Hz playback approx
                    if (currentTime.value >= duration.value) {
                        currentTime.value = duration.value;
                        togglePlay();
                    }
                }, 100);
            } else {
                clearInterval(playInterval);
            }
        };

        const seek = (event) => {
            if (!duration.value) return;
            // event.target might be the cursor or segment, so use timelineContent ref
            const rect = timelineContent.value.getBoundingClientRect();
            const x = event.clientX - rect.left;
            // x is pixels from start of timeline
            const time = x / zoomLevel.value;
            currentTime.value = Math.max(0, Math.min(duration.value, time));
        };

        const handleMouseMove = (event) => {
            if (event.buttons === 1) { // Dragging
                if (isDraggingHandle.value && draggedCutIndex.value !== -1) {
                    // Dragging a handle - resize the cut
                    if (!duration.value) return;

                    const rect = timelineContent.value.getBoundingClientRect();
                    const x = event.clientX - rect.left;
                    const time = x / zoomLevel.value;
                    const clampedTime = Math.max(0, Math.min(duration.value, time));

                    const cut = cuts.value[draggedCutIndex.value];

                    if (draggedHandleType.value === 'start') {
                        // Move start endpoint
                        cut.start = Math.min(clampedTime, cut.end);
                    } else if (draggedHandleType.value === 'end') {
                        // Move end endpoint
                        cut.end = Math.max(clampedTime, cut.start);
                    }
                } else {
                    // Normal timeline seek
                    seek(event);
                }
            }
        };

        const markStart = () => {
            activeCutStart.value = currentTime.value;
        };

        const markEnd = () => {
            if (activeCutStart.value !== null) {
                pushHistory();
                let start = activeCutStart.value;
                let end = currentTime.value;
                if (start > end) [start, end] = [end, start];

                cuts.value.push({ start, end });
                activeCutStart.value = null;
            }
        };

        const clearCuts = () => {
            if (confirm("Clear all cuts?")) {
                pushHistory();
                cuts.value = [];
                activeCutStart.value = null;
            }
        };

        const getSegmentStyle = (segment) => {
            if (!duration.value) return {};
            const startPx = segment.start * zoomLevel.value;
            const endPx = segment.end * zoomLevel.value;
            return {
                left: `${startPx}px`,
                width: `${endPx - startPx}px`
            };
        };

        const saveMetadata = async (showMessage = false) => {
            if (!currentBag.value) return;

            const metadata = {
                bag_name: currentBag.value,
                description: description.value,
                cuts: cuts.value,
                tags: tags.value,
                duration: duration.value, // Save duration too for list view
                start_time: startTime.value // Save start_time for list view
            };

            try {
                await fetch(`/api/bag/${currentBag.value}/metadata`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(metadata)
                });
                if (showMessage) {
                    alert('Metadata saved!');
                }

                // Update local bag list item
                const bagIndex = bags.value.findIndex(b => b.name === currentBag.value);
                if (bagIndex !== -1) {
                    bags.value[bagIndex].description = description.value;
                    bags.value[bagIndex].cuts = cuts.value;
                    bags.value[bagIndex].tags = tags.value;
                    bags.value[bagIndex].duration = duration.value;
                    bags.value[bagIndex].start_time = startTime.value;
                }

            } catch (e) {
                console.error("Failed to save metadata", e);
                if (showMessage) {
                    alert('Failed to save metadata');
                }
            }
        };

        // Keyboard shortcuts
        window.addEventListener('keydown', (e) => {
            if (!currentBag.value) return;

            if (e.code === 'Space') {
                e.preventDefault(); // Prevent scrolling
                togglePlay();
            } else if (e.code === 'BracketLeft') {
                markStart();
            } else if (e.code === 'BracketRight') {
                markEnd();
            } else if (e.code === 'ArrowLeft') {
                if (selectedCutIndex.value !== -1 && selectedHandle.value) {
                    pushHistory();
                    const cut = cuts.value[selectedCutIndex.value];
                    if (selectedHandle.value === 'start') {
                        cut.start = Math.max(0, cut.start - 0.1);
                        if (cut.start > cut.end) cut.start = cut.end;
                    } else {
                        cut.end = Math.max(cut.start, cut.end - 0.1);
                    }
                } else {
                    currentTime.value = Math.max(0, currentTime.value - 0.1);
                }
            } else if (e.code === 'ArrowRight') {
                if (selectedCutIndex.value !== -1 && selectedHandle.value) {
                    pushHistory();
                    const cut = cuts.value[selectedCutIndex.value];
                    if (selectedHandle.value === 'start') {
                        cut.start = Math.min(cut.end, cut.start + 0.1);
                    } else {
                        cut.end = Math.min(duration.value, cut.end + 0.1);
                    }
                } else {
                    currentTime.value = Math.min(duration.value, currentTime.value + 0.1);
                }
            } else if (e.code === 'KeyZ' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                undo();
            }
        });

        // Global mouseup to end handle dragging
        window.addEventListener('mouseup', () => {
            if (isDraggingHandle.value) {
                // Save history at the end of drag
                pushHistory();
                isDraggingHandle.value = false;
                draggedCutIndex.value = -1;
                draggedHandleType.value = null;
            }
        });

        // Close context menu on click
        window.addEventListener('click', () => {
            showContextMenu.value = false;
        });

        const processingProgress = ref(0);
        const processingTotal = ref(0);
        const processingCurrent = ref(0);
        const existingProcessing = ref(null); // { timestamp: number, path: string }
        let processingPollInterval = null;

        const checkExistingProcessing = async () => {
            if (!currentBag.value) return;

            const payload = {
                bag_name: currentBag.value,
                options: {
                    channels: options.value.channels,
                    canny: options.value.canny,

                    depth: options.value.depth
                }
            };

            if (options.value.resize) {
                payload.options.resolution = [options.value.width, options.value.height];
            }

            try {
                const res = await fetch('/api/check_processing', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.exists) {
                    existingProcessing.value = data;
                } else {
                    existingProcessing.value = null;
                }
            } catch (e) {
                console.error("Error checking existing processing", e);
            }
        };

        // Watch options to check for existing processing
        watch(options, () => {
            checkExistingProcessing();
        }, { deep: true });

        // Auto-save cuts when they change
        let saveTimeout = null;
        watch(cuts, () => {
            // Debounce saves to avoid excessive API calls during drag operations
            if (!currentBag.value) return;

            if (saveTimeout) clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                saveMetadata();
            }, 1000); // Save 1 second after last change
        }, { deep: true });

        const pollProcessingStatus = async () => {
            if (!processingBag.value) return;

            try {
                const res = await fetch(`/api/processing_status/${processingBag.value}`);
                if (res.ok) {
                    const data = await res.json();
                    processingProgress.value = data.progress;
                    processingTotal.value = data.total;
                    processingCurrent.value = data.current;

                    if (data.status === 'done') {
                        processingStatus.value = "Processing complete!";
                        isProcessing.value = false;
                        clearInterval(processingPollInterval);
                        checkExistingProcessing(); // Refresh existing status
                        fetchProcessedBags(); // Refresh list
                    } else if (data.status === 'error') {
                        processingStatus.value = "Error: " + data.error;
                        isProcessing.value = false;
                        clearInterval(processingPollInterval);
                    } else if (data.status === 'cancelled') {
                        processingStatus.value = "Processing cancelled.";
                        isProcessing.value = false;
                        clearInterval(processingPollInterval);
                    } else {
                        processingStatus.value = `Processing... ${data.current}/${data.total} (${data.progress.toFixed(1)}%)`;
                    }
                }
            } catch (e) {
                console.error("Error polling status", e);
            }
        };

        const fetchProcessedBags = async () => {
            loadingProcessedBags.value = true;
            try {
                const res = await fetch('/api/processed_bags');
                processedBags.value = await res.json();
            } catch (e) {
                console.error("Failed to load processed bags", e);
            } finally {
                loadingProcessedBags.value = false;
            }
        };

        const startProcessing = async (targetBagOverride = null, skipConfirm = false) => {
            const targetBag = targetBagOverride || currentBag.value;
            if (!targetBag) return;

            if (!skipConfirm && existingProcessing.value && targetBag === currentBag.value) {
                if (!confirm("Processing output already exists for these options. Overwrite?")) {
                    return;
                }
            }

            isProcessing.value = true;
            processingStatus.value = `Starting processing for ${targetBag}...`;
            processingProgress.value = 0;
            processingBag.value = targetBag;

            const payload = {
                bag_name: targetBag,
                options: {
                    channels: options.value.channels,
                    canny: options.value.canny,
                    depth: options.value.depth
                }
            };

            if (options.value.resize) {
                payload.options.resolution = [options.value.width, options.value.height];
            }

            try {
                const res = await fetch('/api/process_bag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                if (data.error) {
                    processingStatus.value = "Error: " + data.error;
                    isProcessing.value = false;
                    return false; // Return failure
                } else {
                    processingStatus.value = `Processing started...`;
                    // Start polling
                    if (processingPollInterval) clearInterval(processingPollInterval);

                    // We need a promise to wait for completion if we are batch processing
                    // But the poll function is purely side-effect based.
                    // Let's rely on the poll status or return a promise?
                    // For batch processing, we can just wait until isProcessing becomes false?
                    // But pollProcessingStatus clears interval.
                    processingPollInterval = setInterval(pollProcessingStatus, 1000);
                    return true;
                }
            } catch (e) {
                console.error("Processing error", e);
                processingStatus.value = "Error starting processing.";
                isProcessing.value = false;
                return false;
            }
        };



        const isBatchProcessing = ref(false);
        const batchProgress = ref(0);
        const batchTotal = ref(0);
        const batchCurrent = ref(0);

        const reprocessAll = async () => {
            if (isProcessing.value) return;

            await fetchProcessedBags();

            // 1. Get unique bags that have been processed at least once
            // processedBags.value contains objects like { bag_name: "foo", ... }
            // We want the original bag names.
            const uniqueBags = new Set(processedBags.value.map(pb => pb.bag_name));

            // Filter to ensure the bag actually exists in the current bags list
            const availableBagNames = new Set(bags.value.map(b => b.name));
            const bagsList = Array.from(uniqueBags).filter(name => availableBagNames.has(name));

            console.log("Found processed bags for reprocessing (filtered by availability):", bagsList);

            if (bagsList.length === 0) {
                alert("No available processed bags found to reprocess.");
                return;
            }

            if (!confirm(`This will reprocess ${bagsList.length} bags that have been previously processed with the CURRENT options selected on the left. This may take a long time. Continue?`)) return;

            console.log("Batch Reprocessing starting for:", bagsList);

            console.log("Batch Reprocessing:", bagsList);

            isBatchProcessing.value = true;
            batchTotal.value = bagsList.length;
            batchCurrent.value = 0;
            batchProgress.value = 0;

            let processedCount = 0;
            for (const bagName of bagsList) {
                batchCurrent.value = processedCount + 1;
                batchProgress.value = (processedCount / batchTotal.value) * 100;

                // Update UI to show what's happening (though startProcessing does this too)
                processingStatus.value = `Batch: Processing ${bagName} (${processedCount + 1}/${bagsList.length})...`;

                const started = await startProcessing(bagName, true);

                if (started) {
                    // Wait for it to finish
                    // We can poll isProcessing
                    while (isProcessing.value) {
                        await new Promise(r => setTimeout(r, 1000));
                    }
                }
                processedCount++;
            }


            batchProgress.value = 100;
            isBatchProcessing.value = false;
            alert(`Batch processing complete. Processed ${processedCount} bags.`);
        };



        const cancelProcessing = async () => {
            if (!processingBag.value) return;
            try {
                await fetch(`/api/cancel_processing/${processingBag.value}`, { method: 'POST' });
                processingStatus.value = "Cancelling...";
            } catch (e) {
                console.error("Error cancelling", e);
            }
        };



        // Watch step change to load processed bags
        watch(currentStep, (newStep) => {
            if (newStep === 2 || newStep === 1) {
                fetchProcessedBags();
            }
        });

        const generationProgress = ref(0);
        const generationPollInterval = ref(null);

        const currentBagProcessedList = computed(() => {
            if (!currentBag.value || !processedBags.value) return [];
            return processedBags.value.filter(b => b.bag_name === currentBag.value);
        });

        // Enrich processed bags with original bag metadata
        const enrichedProcessedBags = computed(() => {
            return processedBags.value.map(pBag => {
                const originalBag = bags.value.find(b => b.name === pBag.bag_name);
                return {
                    ...pBag,
                    duration: originalBag?.duration || 0,
                    start_time: originalBag?.start_time || 0,
                    image_count: originalBag ? originalBag.image_count : 0,
                    tags: originalBag?.tags || []
                };
            }).sort((a, b) => {
                // Sort by start_time descending (newest first)
                return b.start_time - a.start_time;
            });
        });

        const duplicateDataset = computed(() => {
            if (selectedProcessedBags.value.length === 0 || !datasets.value) return null;

            // Create a set of selected bag names
            const selectedNames = new Set(selectedProcessedBags.value.map(b => b.bag_name));

            // Check each existing dataset
            for (const ds of datasets.value) {
                // ds is now an object { dataset_name, source_bags, samples_count, parameters }
                if (!ds.source_bags) continue;

                const sourceNames = new Set(ds.source_bags);

                // Check for exact match of bags
                if (selectedNames.size === sourceNames.size &&
                    [...selectedNames].every(name => sourceNames.has(name))) {

                    // Check parameters if they exist
                    if (ds.parameters) {
                        const params = ds.parameters;
                        const opts = datasetOptions.value;

                        // Use a small epsilon for float comparison or just direct comparison if they are numbers
                        // Inputs are v-model.number, so they should be numbers.
                        if (params.history_rate === opts.historyRate &&
                            params.history_duration === opts.historyDuration &&
                            params.future_rate === opts.futureRate &&
                            params.future_duration === opts.futureDuration) {
                            return ds.dataset_name;
                        }
                    } else {
                        // If dataset has no parameters (legacy?), maybe just warn based on bags?
                        // Or assume it's a duplicate if we can't verify parameters?
                        // Let's assume strict matching: if we can't verify params, we don't warn, 
                        // OR we warn but say "potential duplicate". 
                        // But user asked to incorporate params. So if params don't match, it's NOT a duplicate.
                        // If ds has no params, we can't match params, so it's not a duplicate in this strict sense.
                        // However, legacy datasets might be considered duplicates?
                        // Let's stick to strict parameter matching.
                    }
                }
            }
            return null;
        });

        const pollGenerationStatus = async (jobId) => {
            try {
                const res = await fetch(`/api/generation_status/${jobId}`);
                const data = await res.json();

                if (data.error) {
                    generationStatus.value = "Error: " + data.error;
                    isGenerating.value = false;
                    clearInterval(generationPollInterval.value);
                } else {
                    generationProgress.value = data.progress;

                    if (data.status === 'done') {
                        generationStatus.value = `Success! Generated ${data.count} samples in ${data.file}`;
                        isGenerating.value = false;
                        clearInterval(generationPollInterval.value);
                        fetchDatasets(); // Refresh datasets list
                    } else if (data.status === 'error') {
                        generationStatus.value = "Error: " + data.error;
                        isGenerating.value = false;
                        clearInterval(generationPollInterval.value);
                    } else {
                        generationStatus.value = `Generating... ${data.current}/${data.total} (${data.progress.toFixed(1)}%)`;
                    }
                }
            } catch (e) {
                console.error("Error polling generation status", e);
            }
        };

        const availableResolutions = computed(() => {
            const resolutions = new Set();
            processedBags.value.forEach(pBag => {
                if (pBag.options.resize) {
                    resolutions.add(`${pBag.options.resolution[0]}x${pBag.options.resolution[1]}`);
                } else {
                    resolutions.add("Original");
                }
            });
            return Array.from(resolutions).sort();
        });

        const filteredProcessedBags = computed(() => {
            return enrichedProcessedBags.value.filter(pBag => {
                // Resolution
                if (filterResolution.value) {
                    let res = "Original";
                    if (pBag.options.resize) {
                        res = `${pBag.options.resolution[0]}x${pBag.options.resolution[1]}`;
                    }
                    if (res !== filterResolution.value) return false;
                }

                // Channels
                if (filterChannels.value && pBag.options.channels !== filterChannels.value) return false;

                // Canny - strict match for consistency or loose? 
                // Let's do strict match: if filter checked, must have it. If not checked, don't care?
                // Actually requirement says "only bags with the same properties are shown".
                // So if I check Canny, I want Canny bags. If I uncheck, do I want "Not Canny" or "Any"?
                // "filter above the list ... so that only bags with the same properties are shown" implies narrowing.
                // If I select "Canny", show only Canny bags.
                if (filterCanny.value && !pBag.options.canny) return false;

                // Depth
                if (filterDepth.value && !pBag.options.depth) return false;

                // Tags
                if (filterTags.value && !pBag.tags.includes(filterTags.value)) return false;

                return true;
            });
        });

        const clearFilters = () => {
            filterResolution.value = "";
            filterChannels.value = "";
            filterCanny.value = false;
            filterDepth.value = false;
            filterTags.value = "";
        };

        const selectAllVisible = () => {
            filteredProcessedBags.value.forEach(pBag => {
                // Avoid duplicates using bag_name + options checking or just checking if object is in array
                // Since pBag objects might be transiently created in enrichedProcessedBags, we should check by unique ID or path.
                // pBag.path is unique.
                if (!selectedProcessedBags.value.some(b => b.path === pBag.path)) {
                    selectedProcessedBags.value.push(pBag);
                }
            });
        };

        const deselectAllVisible = () => {
            const visiblePaths = new Set(filteredProcessedBags.value.map(b => b.path));
            selectedProcessedBags.value = selectedProcessedBags.value.filter(b => !visiblePaths.has(b.path));
        };

        const generateDataset = async () => {
            if (selectedProcessedBags.value.length === 0) return;

            isGenerating.value = true;
            generationStatus.value = "Starting generation...";
            generationProgress.value = 0;

            // Extract options from the first selected bag to save as dataset parameters
            // We assume user has filtered correctly so all selected bags have compatible options.
            const representativeBag = selectedProcessedBags.value[0];
            const isResized = !!representativeBag.options.resolution;

            const datasetProcessingOptions = {
                resize: isResized,
                width: isResized ? representativeBag.options.resolution[0] : null,
                height: isResized ? representativeBag.options.resolution[1] : null,
                channels: representativeBag.options.channels,
                canny: representativeBag.options.canny,
                depth: representativeBag.options.depth
            };

            const payload = {
                selected_bags: selectedProcessedBags.value,
                dataset_name: datasetName.value,
                history_rate: datasetOptions.value.historyRate,
                history_duration: datasetOptions.value.historyDuration,
                future_rate: datasetOptions.value.futureRate,
                future_duration: datasetOptions.value.futureDuration,
                processing_options: datasetProcessingOptions
            };

            try {
                const res = await fetch('/api/generate_dataset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (res.ok) {
                    const data = await res.json();
                    const jobId = data.job_id;
                    generationStatus.value = "Job started. ID: " + jobId;

                    // Poll for status
                    generationPollInterval.value = setInterval(() => pollGenerationStatus(jobId), 1000);
                } else {
                    const data = await res.json();
                    generationStatus.value = "Error: " + data.error;
                    isGenerating.value = false;
                }
            } catch (e) {
                console.error("Failed to generate dataset", e);
                generationStatus.value = "Error: " + e.message;
                isGenerating.value = false;
            }
        };

        const currentLoadedDataset = ref(null);

        const resetDatasetForm = () => {
            currentLoadedDataset.value = null;
            datasetName.value = "dataset";
            datasetOptions.value = {
                historyRate: 5,
                historyDuration: 5.0,
                futureRate: 5,
                futureDuration: 3.0
            };
            selectedProcessedBags.value = [];
            clearFilters();
        };

        const loadDatasetConfig = (ds) => {
            console.log("Loading dataset config:", ds);
            currentLoadedDataset.value = ds.dataset_name;
            datasetName.value = ds.dataset_name;

            if (ds.parameters) {
                datasetOptions.value = {
                    historyRate: ds.parameters.history_rate || 5,
                    historyDuration: ds.parameters.history_duration || 5.0,
                    futureRate: ds.parameters.future_rate || 5,
                    futureDuration: ds.parameters.future_duration || 3.0
                };
            }

            // Select corresponding processed bags
            selectedProcessedBags.value = [];
            if (ds.source_bags && processedBags.value) {
                const sourceBagNames = new Set(ds.source_bags);

                // We need to match not just bag name but also processing options if possible
                // The dataset parameters contain 'options' which are the processing options
                // Also legacy datasets might have parameters directly on parameters object
                const params = ds.parameters || {};
                const dsOptions = params.options || {};

                // Fallback to top-level parameters if not in options
                const dsChannels = dsOptions.channels || params.channels;
                const dsCanny = dsOptions.canny !== undefined ? dsOptions.canny : params.canny;
                const dsDepth = dsOptions.depth !== undefined ? dsOptions.depth : params.depth;

                // Resolution check logic
                // Check resize/resolution
                // Logic: 
                // 1. If options.resize is explicitly defined, use it.
                // 2. If options.width/height defined, imply resize.
                // 3. Fallback to top level params width/height.

                let dsResize = dsOptions.resize;
                let dsWidth = dsOptions.width || params.width;
                let dsHeight = dsOptions.height || params.height;

                console.log("Dataset Options (Merged):", {
                    channels: dsChannels,
                    canny: dsCanny,
                    depth: dsDepth,
                    resize: dsResize,
                    width: dsWidth,
                    height: dsHeight
                });

                // Use enrichedProcessedBags to ensure object reference equality for v-model
                const bagsPool = enrichedProcessedBags.value || processedBags.value;

                bagsPool.forEach(pBag => {
                    if (sourceBagNames.has(pBag.bag_name)) {
                        const currOpts = pBag.options;
                        const currIsResized = !!currOpts.resolution;

                        let resolutionMatch = false;

                        if (dsResize) {
                            resolutionMatch = currIsResized &&
                                dsWidth === currOpts.resolution[0] &&
                                dsHeight === currOpts.resolution[1];
                        } else if (dsWidth && dsHeight) {
                            // Fallback: if width/height are set, imply resize check
                            resolutionMatch = currIsResized &&
                                dsWidth === currOpts.resolution[0] &&
                                dsHeight === currOpts.resolution[1];
                        } else if (currIsResized) {
                            // Dataset says nothing about resize (or resize=false/null), but bag IS resized.
                            // Fail match to avoid selecting wrong processing version.
                            resolutionMatch = false;
                        } else {
                            // Neither has resize info -> Match.
                            resolutionMatch = true;
                        }

                        // Simple check: channels, canny, depth
                        // Normalize channels: undefined/null treated as 'rgb' if we assume default, but safer to check strict first.
                        // If dsChannels is undefined, maybe we shouldn't fail? Assuming legacy "rgb"?
                        // Let's assume strict if defined, loose if undefined?
                        // If undefined, it matches anything? Or matches default 'rgb'?
                        // Looking at logs, bag is 'gray', ds is undefined. Match failed.
                        // If the dataset was created without channels info, it likely used whatever was default or available.
                        // But if we want to be strict, we might fail.
                        // However, if the goal is "it's not working", we probably want to be permissive if data is missing.

                        const channelsMatch = !dsChannels || (dsChannels === currOpts.channels);
                        // For booleans, undefined usually means false/off in this context?
                        const cannyMatch = (!!dsCanny === !!currOpts.canny);
                        const depthMatch = (!!dsDepth === !!currOpts.depth);

                        const match = resolutionMatch && channelsMatch && cannyMatch && depthMatch;

                        if (match) {
                            selectedProcessedBags.value.push(pBag);
                        } else {
                            if (!channelsMatch) {
                                console.log(`Channels mismatch for ${pBag.bag_name}: DS='${dsChannels}' (${typeof dsChannels}) vs Bag='${currOpts.channels}' (${typeof currOpts.channels})`);
                            }
                            console.log(`Bag ${pBag.bag_name} matching failed. Res: ${resolutionMatch}, Ch: ${channelsMatch}, Can: ${cannyMatch}, Dep: ${depthMatch}`);
                        }
                    }
                });
            }
        };

        // Visualization Logic
        const fetchDatasets = async () => {
            loadingDatasets.value = true;
            try {
                const res = await fetch('/api/datasets');
                datasets.value = await res.json();
            } catch (e) {
                console.error("Failed to load datasets", e);
            } finally {
                loadingDatasets.value = false;
            }
        };

        watch(currentStep, (newStep) => {
            if (newStep === 3 || newStep === 2 || newStep === 4) { // Fetch datasets for step 2, 3, and 4
                fetchDatasets();
            }
        });

        const loadingDataset = ref(false);

        const selectDataset = async (ds) => {
            const name = ds.dataset_name || ds; // Handle both object and string (legacy)
            loadingDataset.value = true;
            try {
                const res = await fetch(`/api/dataset/${name}`);
                datasetData.value = await res.json();
                selectedDataset.value = ds;
                currentVizBag.value = null;
                currentSampleIndex.value = 0;
            } catch (e) {
                console.error("Failed to load dataset", e);
            } finally {
                loadingDataset.value = false;
            }
        };

        const deleteDataset = async (ds) => {
            const name = ds.dataset_name || ds;
            if (!confirm(`Are you sure you want to delete dataset "${name}"? This cannot be undone.`)) {
                return;
            }

            try {
                const res = await fetch(`/api/dataset/${name}`, { method: 'DELETE' });
                const data = await res.json();

                if (data.error) {
                    alert("Error deleting dataset: " + data.error);
                } else {
                    // Refresh list
                    fetchDatasets();
                    // If deleted dataset was selected, clear selection
                    if (selectedDataset.value && (selectedDataset.value.dataset_name === name || selectedDataset.value === name)) {
                        selectedDataset.value = null;
                        datasetData.value = null;
                        currentVizBag.value = null;
                        currentSampleIndex.value = 0;
                    }
                }
            } catch (e) {
                console.error("Delete error", e);
                alert("Failed to delete dataset");
            }
        };

        const datasetBags = computed(() => {
            if (!datasetData.value) return [];
            const bags = new Set();
            datasetData.value.samples.forEach(s => bags.add(s.bag));
            return Array.from(bags).sort();
        });

        const datasetParams = computed(() => {
            return datasetData.value ? datasetData.value.parameters : {};
        });

        const vizSamples = computed(() => {
            if (!datasetData.value || !currentVizBag.value) return [];
            return datasetData.value.samples.filter(s => s.bag === currentVizBag.value);
        });

        const currentSample = computed(() => {
            if (vizSamples.value.length === 0) return null;
            return vizSamples.value[currentSampleIndex.value];
        });

        const selectDatasetBag = (bagName) => {
            currentVizBag.value = bagName;
            currentSampleIndex.value = 0;
        };

        const selectSample = (idx) => {
            currentSampleIndex.value = idx;
        };

        const nextSample = () => {
            if (currentSampleIndex.value < vizSamples.value.length - 1) {
                currentSampleIndex.value++;
            }
        };

        const prevSample = () => {
            if (currentSampleIndex.value > 0) {
                currentSampleIndex.value--;
            }
        };

        const loadingSample = ref(false);

        const checkLoadingStatus = () => {
            if (!loadingSample.value) return;
            // If we have refs, check if they are all complete
            if (sampleImageRefs.value.length > 0) {
                const allLoaded = sampleImageRefs.value.every(img => img.complete);
                if (allLoaded) {
                    loadingSample.value = false;
                }
            }
        };

        watch(currentSample, async () => {
            if (currentSample.value) {
                loadingSample.value = true;
                await nextTick();
                checkLoadingStatus(); // Check immediately for cached images
            }
        });

        const getSteerStyle = (val) => {
            // val is steer, -1 to 1. Center 50%.
            const pct = Math.abs(val) * 50;
            const left = val < 0 ? 50 - pct : 50;
            return {
                left: `${left}%`,
                width: `${pct}%`
            };
        };


        const datasetStats = computed(() => {
            let totalDuration = 0;
            let totalExamples = 0;

            selectedProcessedBags.value.forEach(pBag => {
                const originalBag = bags.value.find(b => b.name === pBag.bag_name);
                if (originalBag) {
                    totalDuration += originalBag.duration || 0;

                    const duration = originalBag.duration || 0;
                    const imageCount = originalBag.image_count || 0;

                    if (duration > 0 && imageCount > 0) {
                        const frameRate = imageCount / duration;
                        const historyDuration = datasetOptions.value.historyDuration || 0;
                        const futureDuration = datasetOptions.value.futureDuration || 0;

                        const validDuration = Math.max(0, duration - historyDuration - futureDuration);
                        const examples = Math.floor(validDuration * frameRate);

                        totalExamples += examples;
                    }
                }
            });

            return {
                duration: totalDuration,
                examples: totalExamples,
                formattedDuration: new Date(totalDuration * 1000).toISOString().substr(11, 8) // HH:MM:SS
            };
        });

        // Analysis Logic
        const selectAnalysisDataset = async (ds) => {
            const name = ds.dataset_name || ds;
            selectedAnalysisDataset.value = name;

            try {
                const res = await fetch(`/api/dataset/${name}/stats`);
                const stats = await res.json();

                if (stats.error) {
                    console.error("Error fetching stats:", stats.error);
                    return;
                }

                nextTick(() => {
                    renderCharts(stats);
                });

            } catch (e) {
                console.error("Failed to load dataset stats", e);
            }
        };

        const renderCharts = (stats) => {
            if (velocityChart.value) velocityChart.value.destroy();
            if (curvatureChart.value) curvatureChart.value.destroy();

            const createHistogramData = (data, bins = 50, fixedMin = null, fixedMax = null) => {
                let min = fixedMin !== null ? fixedMin : Infinity;
                let max = fixedMax !== null ? fixedMax : -Infinity;

                if (fixedMin === null || fixedMax === null) {
                    // Avoid spread syntax for large arrays to prevent stack overflow
                    for (let i = 0; i < data.length; i++) {
                        if (data[i] < min) min = data[i];
                        if (data[i] > max) max = data[i];
                    }
                }

                if (min === Infinity) { min = 0; max = 1; }
                if (min === max) { max = min + 1; }

                const step = (max - min) / bins;
                const histogram = new Array(bins).fill(0);
                const labels = new Array(bins).fill(0).map((_, i) => (min + i * step).toFixed(2));

                data.forEach(val => {
                    const bin = Math.min(Math.floor((val - min) / step), bins - 1);
                    if (bin >= 0 && bin < bins) {
                        histogram[bin]++;
                    }
                });

                return { labels, histogram };
            };

            const velData = createHistogramData(stats.velocities, 50, -0.1, 2.1);
            const curData = createHistogramData(stats.curvatures, 50, -1.1, 1.1);

            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: '#333' } },
                    x: { grid: { color: '#333' } }
                }
            };

            const ctxVel = document.getElementById('velocityChart').getContext('2d');
            velocityChart.value = new Chart(ctxVel, {
                type: 'bar',
                data: {
                    labels: velData.labels,
                    datasets: [{
                        label: 'Velocity',
                        data: velData.histogram,
                        backgroundColor: '#bb86fc',
                        borderColor: '#bb86fc',
                        borderWidth: 1
                    }]
                },
                options: commonOptions
            });

            const ctxCur = document.getElementById('curvatureChart').getContext('2d');
            curvatureChart.value = new Chart(ctxCur, {
                type: 'bar',
                data: {
                    labels: curData.labels,
                    datasets: [{
                        label: 'Curvature',
                        data: curData.histogram,
                        backgroundColor: '#03dac6',
                        borderColor: '#03dac6',
                        borderWidth: 1
                    }]
                },
                options: commonOptions
            });
        };

        // Visualization Logic (Video)
        const vizCanvas = ref(null);
        const isVizPlaying = ref(false);
        const autoPlayViz = ref(false);
        const vizStatus = ref("Idle");
        const currentVizVelocity = ref(null);
        const currentVizCurvature = ref(null);
        let animationFrameId = null;

        // Helper to draw curvature overlay
        const drawCurvature = (ctx, width, height, curvature, velocity) => {
            const cx = width / 2;
            const cy = height - 20;

            // Draw center line
            ctx.strokeStyle = "white";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx, cy - 40);
            ctx.stroke();

            const wheelbase = 0.324; // Standard wheelbase

            ctx.strokeStyle = "#00ff00"; // Green
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(cx, cy);

            // Draw Spline (Path Integration) matching vesc_driver.cpp kinematics
            ctx.beginPath();
            ctx.moveTo(cx, cy);

            // Simulation parameters
            const dt = 0.05; // Time step
            const horizon = 1.0; // Prediction horizon in seconds
            const scale = 100.0; // Pixels per meter (heuristic)

            let sim_x = 0;
            let sim_y = 0;
            let sim_theta = Math.PI / 2; // Start facing UP

            // v is passed in. If v is 0 (or null), assume a default speed for visualization shape
            let sim_v = velocity;
            if (!sim_v || Math.abs(sim_v) < 0.1) sim_v = 1.0;

            for (let t = 0; t < horizon; t += dt) {
                // Kinematics from vesc_driver.cpp updateOdometry
                // del_x = v * dt * cos(theta)
                // del_y = v * dt * sin(theta)
                // del_theta = v * k * dt

                const dist = sim_v * dt;
                const del_x = dist * Math.cos(sim_theta);
                const del_y = dist * Math.sin(sim_theta);
                const del_theta = dist * curvature;

                sim_x += del_x;
                sim_y += del_y;
                sim_theta += del_theta;

                // Project to Canvas (Top-down approximation)
                // Canvas X = cx - sim_x * scale (Note: Inverted X for Left Turn = Left Draw)
                // Wait, simulation: theta=PI/2 (Up).
                // Left Turn (Positive Curvature) -> theta increases (+) -> turns Left (towards PI).
                // cos(PI) = -1. So x becomes negative.
                // We want Left Turn to be drawn Left (Canvas X < cx).
                // So Canvas X = cx + sim_x * scale.
                // Let's trace:
                // Start: x=0, y=0, th=PI/2.
                // Step 1: k>0. th increases to PI/2 + e. (Quadrant 2).
                // cos(Q2) is Negative. sin(Q2) is Positive.
                // dx is Neg, dy is Pos.
                // x becomes Neg. y becomes Pos.
                // Canvas X = cx + x (Neg) -> Left. Correct.
                // Canvas Y = cy - y (Pos) -> Up. Correct.

                ctx.lineTo(cx + sim_x * scale, cy - sim_y * scale);
            }
            ctx.stroke();
        };

        const loadImage = (src) => {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = reject;
                img.src = src;
            });
        };

        const playSequentialVideo = async () => {
            if (!vizCanvas.value) return;
            const canvas = document.getElementById('vizCanvas');
            const ctx = canvas.getContext('2d');

            isVizPlaying.value = true;
            vizStatus.value = "Playing...";

            // Loop until end or stopped
            while (isVizPlaying.value && currentSampleIndex.value < vizSamples.value.length) {
                const s = vizSamples.value[currentSampleIndex.value];
                const src = `/api/processed_file/${s.current_image}`;

                try {
                    // Load current image
                    const img = await loadImage(src);

                    if (!isVizPlaying.value) break;

                    // Draw
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                    // Overlay Velocity/Curvature
                    if (s.future_actions && s.future_actions.length > 0) {
                        const [c, v] = s.future_actions[0];
                        drawCurvature(ctx, canvas.width, canvas.height, c, v);
                        currentVizVelocity.value = v;
                        currentVizCurvature.value = c;
                    } else {
                        currentVizVelocity.value = null;
                        currentVizCurvature.value = null;
                    }

                    // Preload next image to smooth playback
                    if (currentSampleIndex.value + 1 < vizSamples.value.length) {
                        const nextSrc = `/api/processed_file/${vizSamples.value[currentSampleIndex.value + 1].current_image}`;
                        const nextImg = new Image();
                        nextImg.src = nextSrc;
                    }

                    // Advance
                    if (currentSampleIndex.value < vizSamples.value.length - 1) {
                        currentSampleIndex.value++;
                    } else {
                        // End of bag
                        isVizPlaying.value = false;
                        vizStatus.value = "End";
                    }

                    // Wait for frame rate (approx 15fps = 66ms, 20fps = 50ms)
                    await new Promise(r => setTimeout(r, 66));

                } catch (e) {
                    console.error("Playback error", e);
                    isVizPlaying.value = false;
                    break;
                }
            }

            if (!isVizPlaying.value && vizStatus.value !== "End") {
                vizStatus.value = "Stopped";
            }
        };

        const toggleVizPlay = () => {
            if (isVizPlaying.value) {
                isVizPlaying.value = false;
                vizStatus.value = "Stopped";
            } else {
                playSequentialVideo();
            }
        };

        // Watchers
        watch(currentSample, () => {
            if (currentSample.value && !isVizPlaying.value) {
                nextTick(async () => {
                    if (!vizCanvas.value && document.getElementById('vizCanvas')) {
                        vizCanvas.value = document.getElementById('vizCanvas');
                    }

                    // Draw static preview (Current Frame)
                    if (!currentSample.value) return;

                    const canvas = document.getElementById('vizCanvas');
                    if (!canvas) return;
                    const ctx = canvas.getContext('2d');

                    // Clear canvas first
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    const src = `/api/processed_file/${currentSample.value.current_image}`;
                    try {
                        const img = await loadImage(src);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                        // Also draw overlay for the static frame so we can see what it looks like
                        if (currentSample.value.future_actions && currentSample.value.future_actions.length > 0) {
                            const [c, v] = currentSample.value.future_actions[0];
                            drawCurvature(ctx, canvas.width, canvas.height, c, v);
                            currentVizVelocity.value = v;
                            currentVizCurvature.value = c;
                        }
                    } catch (e) { console.error(e); }
                });
            }
        });

        // Ensure canvas ref is populated when tab changes
        watch(currentStep, (val) => {
            if (val === 3) {
                nextTick(() => {
                    vizCanvas.value = document.getElementById('vizCanvas');
                    // Trigger a redraw if we have a sample
                    if (currentSample.value && !isVizPlaying.value) {
                        // Force update to draw static frame
                        const temp = currentSample.value;
                        // Just relying on the currentSample watcher might not work if it doesn't change
                        // So let's manually trigger the watcher logic or copied logic.
                        // The active watcher on currentSample should fire if we just accessed it? No.
                        // Let's just manually draw.
                        // Actually, the above watcher logic is fine if we can invoke it.
                        // But for simplicity, the user will probably click a bag or sample soon.
                    }
                });
            } else {
                isVizPlaying.value = false; // Stop if leaving tab
            }
        });

        return {
            bags,
            currentBag,
            loadingBags,
            duration,
            currentTime,
            isPlaying,
            currentFrameUrl,
            description,
            cuts,
            activeCutStart,
            timelineScrollContainer,
            timelineContent,
            bufferingProgress,
            selectBag,
            togglePlay,
            seek,
            handleMouseMove,
            markStart,
            markEnd,
            clearCuts,
            getSegmentStyle,
            saveMetadata,
            // New Settings
            showSettings,
            persistCache,
            cacheSizeMB,
            clearCache,
            clearCache,
            formattedCurrentTime,
            tags,
            allTags,
            addTag,
            removeTag,
            // New Features
            deleteBag,
            getSparklinePoints,
            // Timeline
            zoomLevel,
            setZoom,
            timelineWidth,
            ticks,
            handleScroll,
            handleWheel,
            selectCut,
            handleHandleMouseDown,
            selectedCutIndex,
            selectedHandle,
            handleRightClick,
            showContextMenu,
            contextMenuX,
            contextMenuY,
            deleteSelectedCut,
            resetCuts,
            undo,
            formatBagDate: (bag) => {
                const options = {
                    year: 'numeric',
                    month: 'numeric',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: false
                };

                // If bag is an object with start_time, use it
                if (bag.start_time) {
                    const date = new Date(bag.start_time * 1000);
                    if (!isNaN(date.getTime())) {
                        return date.toLocaleString(undefined, options);
                    }
                }

                // Fallback to parsing filename
                const bagName = bag.name || bag.bag_name || bag;

                if (typeof bagName !== 'string') {
                    return '';
                }

                // Expects roboracer_YYYY_MM_DD_HH_MM_SS
                const parts = bagName.split('_');
                if (parts.length >= 7 && parts[0] === 'roboracer') {
                    const year = parts[1];
                    const month = parts[2];
                    const day = parts[3];
                    const hour = parts[4];
                    const minute = parts[5];
                    const second = parts[6];
                    const date = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`);
                    if (!isNaN(date.getTime())) {
                        return date.toLocaleString(undefined, options);
                    }
                }
                return bagName;
            },
            // Telemetry
            currentTelemetry,
            getBarStyle,
            // Navigation
            currentStep,

            // Processing
            processingBag,
            options,
            isProcessing,
            processingStatus,
            processingProgress,
            processingTotal,
            processingCurrent,
            existingProcessing,
            startProcessing,
            reprocessAll,
            cancelProcessing,
            currentBagProcessedList,
            startPreview,
            previewImageUrl,
            previewStatus,
            isBatchProcessing,
            batchProgress,
            batchTotal,
            batchCurrent,

            // Dataset Builder
            processedBags,
            loadingProcessedBags,
            selectedProcessedBags,
            datasetName,
            datasetOptions,
            isGenerating,
            generationStatus,
            generateDataset,

            // Visualization
            datasets,
            loadingDatasets,
            selectedDataset,
            currentDataset: selectedDataset,
            selectDataset,
            datasetBags,
            currentVizBag,
            currentDatasetBag: currentVizBag,
            vizSamples,
            datasetSamples: vizSamples,
            currentSampleIndex,
            currentSample,
            datasetParams,
            getSteerStyle,
            selectDatasetBag,
            selectSample,
            nextSample,
            prevSample,
            datasetStats,
            enrichedProcessedBags,
            filteredProcessedBags,
            filterResolution,
            filterChannels,
            filterCanny,
            filterDepth,
            filterTags,
            availableResolutions,
            clearFilters,
            selectAllVisible,
            deselectAllVisible,
            duplicateDataset,
            generationProgress,
            loadingDataset,
            datasetStats,
            enrichedProcessedBags,
            filteredProcessedBags,
            filterResolution,
            filterChannels,
            filterCanny,
            filterDepth,
            filterTags,
            availableResolutions,
            clearFilters,
            selectAllVisible,
            deselectAllVisible,
            duplicateDataset,
            generationProgress,
            loadingDataset,
            loadingSample,
            checkLoadingStatus,
            collectImageRef,
            deleteDataset,
            currentLoadedDataset,
            resetDatasetForm,
            loadDatasetConfig,
            // Analysis
            selectedAnalysisDataset,
            selectAnalysisDataset,
            // Video Viz
            vizCanvas,
            isVizPlaying,
            autoPlayViz,
            vizStatus,
            currentVizVelocity,
            currentVizCurvature,
            toggleVizPlay
        };
    }
}).mount('#app');
