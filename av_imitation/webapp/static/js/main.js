const { createApp, ref, computed, onMounted, watch } = Vue;

createApp({
    compilerOptions: {
        delimiters: ['[[', ']]']
    },
    setup() {
        const bags = ref([]);
        const currentBag = ref(null);
        const loadingBags = ref(false);
        const duration = ref(0);
        const currentTime = ref(0);
        const isPlaying = ref(false);
        const description = ref("");
        const cuts = ref([]); // Array of {start, end}
        const activeCutStart = ref(null);

        const timeline = ref(null);
        let playInterval = null;

        // Fetch bags on mount
        onMounted(async () => {
            loadingBags.value = true;
            try {
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

            // Try to get from cache first
            const key = currentTime.value.toFixed(1);
            if (frameCache.value.has(key)) {
                return frameCache.value.get(key);
            }

            // Fallback to direct URL (will be slow/cancelled if not cached)
            return `/api/bag/${currentBag.value}/frame/${currentTime.value.toFixed(2)}?_=${Date.now()}`;
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

            // We'll fetch sequentially to avoid overwhelming the browser/server
            for (let t = 0; t <= durationSec; t += step) {
                if (controller.signal.aborted) break;

                const key = t.toFixed(1);
                // Skip if already cached (though we cleared it)
                if (frameCache.value.has(key)) continue;

                try {
                    const res = await fetch(`/api/bag/${bagName}/frame/${t.toFixed(2)}`);
                    if (res.ok) {
                        const blob = await res.blob();
                        const url = URL.createObjectURL(blob);
                        frameCache.value.set(key, url);
                    }
                } catch (e) {
                    console.warn(`Failed to preload frame at ${t}`, e);
                }

                loaded++;
                bufferingProgress.value = Math.round((loaded / totalSteps) * 100);

                // Yield to main thread occasionally if needed (await fetch does this mostly)
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
            saveMetadata
        };
    }
}).mount('#app');


