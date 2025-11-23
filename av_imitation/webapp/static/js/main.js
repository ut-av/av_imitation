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
            const id = `${bag}_${timestamp.toFixed(2)}`;
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
            const id = `${bag}_${timestamp.toFixed(2)}`;
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
        let playInterval = null;
        let previewInterval = null;
        const previewTime = ref(0);

        // Settings State
        const showSettings = ref(false);
        const persistCache = ref(localStorage.getItem('persistCache') === 'true');
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

        const frameCache = ref(new Map()); // Key: timestamp.toFixed(1), Value: blobUrl
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

            // Try to get from cache first
            const key = timeToShow.toFixed(1);
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

        const preloadBag = async (bagName, durationSec) => {
            if (preloadController.value) {
                preloadController.value.abort();
            }
            const controller = new AbortController();
            preloadController.value = controller;

            frameCache.value.clear();
            bufferingProgress.value = 0;

            const step = 0.1;
            const totalSteps = Math.ceil(durationSec / step);
            let loaded = 0;

            // 1. If persistence is ON, try to load from DB first
            if (persistCache.value) {
                try {
                    const cachedFrames = await FrameDB.getFramesForBag(bagName);
                    cachedFrames.forEach(frame => {
                        const key = frame.timestamp.toFixed(1);
                        const url = URL.createObjectURL(frame.blob);
                        frameCache.value.set(key, url);
                    });
                } catch (e) {
                    console.error("Error loading from DB", e);
                }
            }

            // 2. We'll fetch integers first for preview, then the rest
            const integers = [];
            const fractions = [];
            for (let t = 0; t <= durationSec; t += step) {
                if (Math.abs(t - Math.round(t)) < 0.01) {
                    integers.push(t);
                } else {
                    fractions.push(t);
                }
            }

            // Combine: integers first, then fractions
            const allTimes = [...integers, ...fractions];

            // Start preview loop
            if (previewInterval) clearInterval(previewInterval);
            let previewIndex = 0;
            previewInterval = setInterval(() => {
                if (bufferingProgress.value >= 100) {
                    clearInterval(previewInterval);
                    return;
                }
                // Cycle through integers that are loaded
                // Actually, just cycle through 0, 1, 2... up to duration
                // The image viewer will show what's available or try to fetch
                // But to be smooth, we should probably only show what's loaded or just cycle integers

                previewTime.value = (previewIndex % (Math.floor(durationSec) + 1));
                previewIndex++;
            }, 1000); // 1Hz

            for (const t of allTimes) {
                if (controller.signal.aborted) break;

                const key = t.toFixed(1);
                // Skip if already cached
                if (frameCache.value.has(key)) {
                    loaded++;
                    bufferingProgress.value = Math.round((loaded / totalSteps) * 100);
                    continue;
                }

                try {
                    const res = await fetch(`/api/bag/${bagName}/frame/${t.toFixed(2)}`);
                    if (res.ok) {
                        const blob = await res.blob();
                        const url = URL.createObjectURL(blob);
                        frameCache.value.set(key, url);

                        // Save to DB if persistence is ON
                        if (persistCache.value) {
                            FrameDB.saveFrame(bagName, t, blob).catch(e => console.warn("Failed to save frame", e));
                        }
                    }
                } catch (e) {
                    console.warn(`Failed to preload frame at ${t}`, e);
                }

                loaded++;
                bufferingProgress.value = Math.round((loaded / totalSteps) * 100);

                // Yield to main thread occasionally if needed (await fetch does this mostly)
            }

            // Update size after preload
            if (persistCache.value) {
                updateCacheSize();
            }
        };

        const selectBag = async (bag) => {
            currentBag.value = bag;
            currentTime.value = 0;
            isPlaying.value = false;
            cuts.value = [];
            description.value = "";
            activeCutStart.value = null;

            // Stop any existing preload
            if (preloadController.value) {
                preloadController.value.abort();
            }
            frameCache.value.clear();
            bufferingProgress.value = 0;

            try {
                const res = await fetch(`/api/bag/${bag}/info`);
                const data = await res.json();
                duration.value = data.info.duration;
                startTime.value = data.info.start_time;

                if (data.user_meta) {
                    description.value = data.user_meta.description || "";
                    cuts.value = data.user_meta.cuts || [];
                }

                // Start preloading
                preloadBag(bag, duration.value);

            } catch (e) {
                console.error("Failed to load bag info", e);
            }
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
            const rect = timeline.value.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const percentage = Math.max(0, Math.min(1, x / rect.width));
            currentTime.value = percentage * duration.value;
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
                let start = activeCutStart.value;
                let end = currentTime.value;
                if (start > end) [start, end] = [end, start];

                cuts.value.push({ start, end });
                activeCutStart.value = null;
            }
        };

        const clearCuts = () => {
            cuts.value = [];
            activeCutStart.value = null;
        };

        const getSegmentStyle = (segment) => {
            if (!duration.value) return {};
            const startPct = (segment.start / duration.value) * 100;
            const endPct = (segment.end / duration.value) * 100;
            return {
                left: `${startPct}%`,
                width: `${endPct - startPct}%`
            };
        };

        const saveMetadata = async () => {
            if (!currentBag.value) return;

            const metadata = {
                bag_name: currentBag.value,
                description: description.value,
                cuts: cuts.value
            };

            try {
                await fetch(`/api/bag/${currentBag.value}/metadata`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(metadata)
                });
                alert('Metadata saved!');
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
                currentTime.value = Math.max(0, currentTime.value - 0.1);
            } else if (e.code === 'ArrowRight') {
                currentTime.value = Math.min(duration.value, currentTime.value + 0.1);
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
            timeline,
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
            formattedCurrentTime
        };
    }
}).mount('#app');
