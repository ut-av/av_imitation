const { createApp, ref, computed, onMounted, watch } = Vue;

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
                const res = await fetch('/api/bags');
                bags.value = await res.json();
            } catch (e) {
                console.error("Failed to load bags", e);
            } finally {
                loadingBags.value = false;
            }
        });

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

        const handleRightClick = (index) => {
            pushHistory();
            cuts.value.splice(index, 1);
            selectedCutIndex.value = -1;
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

        const selectBag = async (bag) => {
            // bag is now an object
            currentBag.value = bag.name;
            currentTime.value = 0;
            isPlaying.value = false;
            if (playInterval) clearInterval(playInterval); // Reset play state
            cuts.value = [];
            description.value = "";
            activeCutStart.value = null;
            telemetry.value = [];
            selectedCutIndex.value = -1;
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
                seek(event);
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

        const saveMetadata = async () => {
            if (!currentBag.value) return;

            const metadata = {
                bag_name: currentBag.value,
                description: description.value,
                cuts: cuts.value,
                duration: duration.value, // Save duration too for list view
                start_time: startTime.value // Save start_time for list view
            };

            try {
                await fetch(`/api/bag/${currentBag.value}/metadata`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(metadata)
                });
                alert('Metadata saved!');

                // Update local bag list item
                const bagIndex = bags.value.findIndex(b => b.name === currentBag.value);
                if (bagIndex !== -1) {
                    bags.value[bagIndex].description = description.value;
                    bags.value[bagIndex].cuts = cuts.value;
                    bags.value[bagIndex].duration = duration.value;
                    bags.value[bagIndex].start_time = startTime.value;
                }

            } catch (e) {
                console.error("Failed to save metadata", e);
                alert('Failed to save metadata');
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
            formattedCurrentTime,
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
            selectedCutIndex,
            selectedHandle,
            handleRightClick,
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
                const bagName = bag.name || bag;
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
            getBarStyle
        };
    }
}).mount('#app');
