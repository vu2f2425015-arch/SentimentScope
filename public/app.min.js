        // Utility: High-performance debounce function to eliminate jank & CPU waste
        function debounce(fn, wait) {
            let timeout;
            return function(...args) {
                clearTimeout(timeout);
                timeout = setTimeout(() => fn.apply(this, args), wait);
            };
        }

        // Global State for Clipboard & Copy Operations
        let latestSingleResult = null;
        let latestQuickResult = null;

        // Global Chart Instances
        let dashboardDoughnutChart = null;
        let dashClassMetricsChart = null;
        let liveConfidenceBarChart = null;
        let perfBenchmarkChart = null;
        let liveTimelineLineChart = null;

        // Timeline dataset buffer
        const timelineData = {
            timestamps: [],
            posPct: [],
            neuPct: [],
            negPct: []
        };

        // Safe DOM Element Property Setters (Prevents uncaught null errors from breaking JS execution)
        function safeSetText(id, text) {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
        }
        function safeSetWidth(id, widthVal) {
            const el = document.getElementById(id);
            if (el) el.style.width = widthVal;
        }

        // Theme State Management
        function isDarkMode() {
            return document.documentElement.classList.contains('dark');
        }
        function getChartTextColor() {
            return isDarkMode() ? '#f8fafc' : '#0f172a';
        }
        function getChartMutedColor() {
            return isDarkMode() ? '#94a3b8' : '#64748b';
        }

        function applyTheme(isDark) {
            if (isDark) {
                document.documentElement.classList.add('dark');
                document.documentElement.classList.remove('light');
                localStorage.setItem('theme', 'dark');
            } else {
                document.documentElement.classList.remove('dark');
                document.documentElement.classList.add('light');
                localStorage.setItem('theme', 'light');
            }
            updateThemeIcons(isDark);
            updateChartThemes();
            if (window.updateDotFieldTheme) {
                window.updateDotFieldTheme(isDark);
            }
        }

        function toggleDarkMode() {
            applyTheme(!isDarkMode());
        }

        function updateThemeIcons(isDark) {
            const iconDesktop = document.getElementById('theme-icon-desktop');
            const textDesktop = document.getElementById('theme-text-desktop');
            const iconMobile = document.getElementById('theme-icon-mobile');
            const iconDrawer = document.getElementById('theme-icon-mobile-drawer');
            const textDrawer = document.getElementById('theme-text-mobile-drawer');

            if (isDark) {
                if (iconDesktop) iconDesktop.innerText = 'light_mode';
                if (textDesktop) textDesktop.innerText = 'Light Theme';
                if (iconMobile) iconMobile.innerText = 'light_mode';
                if (iconDrawer) iconDrawer.innerText = 'light_mode';
                if (textDrawer) textDrawer.innerText = 'Light Theme';
            } else {
                if (iconDesktop) iconDesktop.innerText = 'dark_mode';
                if (textDesktop) textDesktop.innerText = 'Dark Theme';
                if (iconMobile) iconMobile.innerText = 'dark_mode';
                if (iconDrawer) iconDrawer.innerText = 'dark_mode';
                if (textDrawer) textDrawer.innerText = 'Dark Theme';
            }
        }

        (function initTheme() {
            const saved = localStorage.getItem('theme');
            if (saved === 'light') {
                applyTheme(false);
            } else {
                applyTheme(true); // Default to sleek ultra-dark theme
            }
        })();

        // -------------------------------------------------------------
        // DotField Interactive Background Engine (React Bits & Dynamic Mode System)
        // -------------------------------------------------------------
        (function initDotFieldEngine() {
            const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            if (prefersReducedMotion) return;

            const canvas = document.getElementById('shader-canvas');
            if (!canvas) return;

            const ctx = canvas.getContext('2d', { alpha: true });
            if (!ctx) return;

            const TWO_PI = Math.PI * 2;
            let dots = [];
            let animationFrameId = null;
            let lastFrameTime = 0;

            const mouse = { x: -9999, y: -9999, prevX: -9999, prevY: -9999, speed: 0 };
            const size = { w: 0, h: 0 };
            let engagement = 0;
            let glowOpacity = 0;
            let frameCount = 0;

            // Palette Definitions for Theme Modes (RGB arrays for smooth lerp)
            const TAB_THEMES = {
                dark: {
                    dashboard:   { from: [168, 85, 247],  to: [99, 102, 241],   glow: [168, 85, 247] }, // Purple/Indigo
                    single:      { from: [6, 182, 212],   to: [99, 102, 241],   glow: [6, 182, 212] },   // Cyan/Indigo
                    batch:       { from: [236, 72, 153],  to: [168, 85, 247],   glow: [236, 72, 153] },  // Pink/Magenta
                    performance: { from: [16, 185, 129],  to: [6, 182, 212],    glow: [16, 185, 129] },  // Emerald/Teal
                    reports:     { from: [245, 158, 11],  to: [168, 85, 247],   glow: [245, 158, 11] }   // Amber/Violet
                },
                light: {
                    dashboard:   { from: [99, 102, 241],  to: [124, 58, 237],   glow: [99, 102, 241] },
                    single:      { from: [14, 165, 233],  to: [99, 102, 241],   glow: [14, 165, 233] },
                    batch:       { from: [219, 39, 119],  to: [124, 58, 237],   glow: [219, 39, 119] },
                    performance: { from: [16, 185, 129],  to: [14, 165, 233],   glow: [16, 185, 129] },
                    reports:     { from: [217, 119, 6],   to: [124, 58, 237],   glow: [217, 119, 6] }
                }
            };

            const PULSE_THEMES = {
                positive: { from: [16, 185, 129],  to: [52, 211, 153],  glow: [16, 185, 129] },
                negative: { from: [239, 68, 68],   to: [248, 113, 113], glow: [239, 68, 68] },
                neutral:  { from: [245, 158, 11],  to: [251, 191, 36],  glow: [245, 158, 11] }
            };

            let activeTab = 'dashboard';
            let activeThemeMode = isDarkMode() ? 'dark' : 'light';
            let pulseMode = null;
            let pulseTimer = null;

            // Current interpolated RGB values
            const currFrom = [...TAB_THEMES[activeThemeMode][activeTab].from];
            const currTo   = [...TAB_THEMES[activeThemeMode][activeTab].to];
            const currGlow = [...TAB_THEMES[activeThemeMode][activeTab].glow];

            function getTargetPalette() {
                if (pulseMode && PULSE_THEMES[pulseMode]) {
                    return PULSE_THEMES[pulseMode];
                }
                const themeObj = TAB_THEMES[activeThemeMode] || TAB_THEMES.dark;
                return themeObj[activeTab] || themeObj.dashboard;
            }

            function lerp(start, end, amt) {
                return start + (end - start) * amt;
            }

            function updateColors() {
                const target = getTargetPalette();
                for (let i = 0; i < 3; i++) {
                    currFrom[i] = lerp(currFrom[i], target.from[i], 0.05);
                    currTo[i]   = lerp(currTo[i], target.to[i], 0.05);
                    currGlow[i] = lerp(currGlow[i], target.glow[i], 0.05);
                }
            }

            // Global Hook API for Mode Reactivity
            window.setDotFieldTabMode = function(tabName) {
                if (TAB_THEMES.dark[tabName]) {
                    activeTab = tabName;
                }
            };

            window.updateDotFieldTheme = function(isDark) {
                activeThemeMode = isDark ? 'dark' : 'light';
            };

            window.triggerDotFieldPulse = function(sentimentStr) {
                const s = (sentimentStr || '').toLowerCase();
                if (PULSE_THEMES[s]) {
                    pulseMode = s;
                    clearTimeout(pulseTimer);
                    pulseTimer = setTimeout(() => {
                        pulseMode = null;
                    }, 3200);
                }
            };

            function resize() {
                const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
                const w = window.innerWidth;
                const h = window.innerHeight;

                canvas.width = Math.floor(w * dpr);
                canvas.height = Math.floor(h * dpr);
                canvas.style.width = `${w}px`;
                canvas.style.height = `${h}px`;
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

                size.w = w;
                size.h = h;
                buildDots(w, h);
            }

            function buildDots(w, h) {
                const dotSpacing = 28;
                const cols = Math.floor(w / dotSpacing);
                const rows = Math.floor(h / dotSpacing);
                const padX = (w % dotSpacing) / 2;
                const padY = (h % dotSpacing) / 2;
                dots = new Array(rows * cols);
                let idx = 0;

                for (let row = 0; row < rows; row++) {
                    for (let col = 0; col < cols; col++) {
                        const ax = padX + col * dotSpacing + dotSpacing / 2;
                        const ay = padY + row * dotSpacing + dotSpacing / 2;
                        dots[idx++] = { ax, ay, sx: ax, sy: ay };
                    }
                }
            }

            function onPointerMove(e) {
                let clientX, clientY;
                if (e.touches && e.touches.length > 0) {
                    clientX = e.touches[0].clientX;
                    clientY = e.touches[0].clientY;
                } else {
                    clientX = e.clientX;
                    clientY = e.clientY;
                }
                if (clientX !== undefined && clientY !== undefined) {
                    const dx = clientX - (mouse.prevX > -100 ? mouse.prevX : clientX);
                    const dy = clientY - (mouse.prevY > -100 ? mouse.prevY : clientY);
                    const dist = Math.hypot(dx, dy);
                    mouse.speed += (dist - mouse.speed) * 0.4;
                    mouse.prevX = mouse.x;
                    mouse.prevY = mouse.y;
                    mouse.x = clientX;
                    mouse.y = clientY;
                }
            }

            function onPointerLeave() {
                mouse.x = -9999;
                mouse.y = -9999;
                mouse.prevX = -9999;
                mouse.prevY = -9999;
                mouse.speed = 0;
            }

            window.addEventListener('mousemove', onPointerMove, { passive: true });
            window.addEventListener('touchmove', onPointerMove, { passive: true });
            window.addEventListener('mouseleave', onPointerLeave, { passive: true });
            window.addEventListener('resize', debounce(resize, 150), { passive: true });

            document.addEventListener('visibilitychange', () => {
                if (!document.hidden && !animationFrameId) {
                    lastFrameTime = performance.now();
                    animationFrameId = requestAnimationFrame(tick);
                }
            });

            resize();

            function tick(currentTime) {
                if (document.hidden) {
                    animationFrameId = null;
                    return;
                }

                // Smoothly throttle to ~60fps maximum to eliminate CPU waste on 120/144Hz screens
                if (currentTime - lastFrameTime < 16) {
                    animationFrameId = requestAnimationFrame(tick);
                    return;
                }
                lastFrameTime = currentTime;

                frameCount++;
                const { w, h } = size;
                const len = dots.length;
                const t = frameCount * 0.02;

                updateColors();

                const isDark = activeThemeMode === 'dark';
                const fromAlpha = isDark ? 0.85 : 0.65;
                const toAlpha   = isDark ? 0.60 : 0.45;
                const glowAlpha = isDark ? 0.55 : 0.35;

                const rFrom = Math.round(currFrom[0]), gFrom = Math.round(currFrom[1]), bFrom = Math.round(currFrom[2]);
                const rTo   = Math.round(currTo[0]),   gTo   = Math.round(currTo[1]),   bTo   = Math.round(currTo[2]);
                const rGlow = Math.round(currGlow[0]), gGlow = Math.round(currGlow[1]), bGlow = Math.round(currGlow[2]);

                // Natural speed decay
                mouse.speed *= 0.94;
                if (mouse.speed < 0.005) mouse.speed = 0;

                const targetEngagement = Math.min(mouse.speed / 5, 1);
                engagement += (targetEngagement - engagement) * 0.06;
                if (engagement < 0.001) engagement = 0;
                const eng = engagement;

                glowOpacity += (eng - glowOpacity) * 0.08;

                ctx.clearRect(0, 0, w, h);

                // Ambient Radial Glow Spotlight centered on cursor
                if (glowOpacity > 0.01 && mouse.x > 0 && mouse.y > 0) {
                    const glowGrad = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 220);
                    glowGrad.addColorStop(0, `rgba(${rGlow}, ${gGlow}, ${bGlow}, ${glowAlpha})`);
                    glowGrad.addColorStop(0.5, `rgba(${rGlow}, ${gGlow}, ${bGlow}, ${glowAlpha * 0.4})`);
                    glowGrad.addColorStop(1, 'transparent');
                    ctx.fillStyle = glowGrad;
                    ctx.beginPath();
                    ctx.arc(mouse.x, mouse.y, 220, 0, TWO_PI);
                    ctx.fill();
                }

                // Dot Matrix Grid
                const grad = ctx.createLinearGradient(0, 0, w, h);
                grad.addColorStop(0, `rgba(${rFrom}, ${gFrom}, ${bFrom}, ${fromAlpha})`);
                grad.addColorStop(1, `rgba(${rTo}, ${gTo}, ${bTo}, ${toAlpha})`);
                ctx.fillStyle = grad;

                const cr = 360;
                const crSq = cr * cr;
                const rad = 1.35;
                const bulgeStrength = 65;
                const mouseActive = eng > 0.01 && mouse.x > -50 && mouse.x < w + 50 && mouse.y > -50 && mouse.y < h + 50;

                ctx.beginPath();

                for (let i = 0; i < len; i++) {
                    const d = dots[i];
                    
                    if (mouseActive) {
                        const dx = mouse.x - d.ax;
                        const dy = mouse.y - d.ay;
                        const distSq = dx * dx + dy * dy;

                        if (distSq < crSq) {
                            const dist = Math.sqrt(distSq);
                            const tNorm = 1 - dist / cr;
                            const push = tNorm * tNorm * bulgeStrength * eng;
                            const angle = Math.atan2(dy, dx);
                            d.sx += (d.ax - Math.cos(angle) * push - d.sx) * 0.15;
                            d.sy += (d.ay - Math.sin(angle) * push - d.sy) * 0.15;
                        } else {
                            d.sx += (d.ax - d.sx) * 0.1;
                            d.sy += (d.ay - d.sy) * 0.1;
                        }
                    } else if (Math.abs(d.sx - d.ax) > 0.01 || Math.abs(d.sy - d.ay) > 0.01) {
                        d.sx += (d.ax - d.sx) * 0.1;
                        d.sy += (d.ay - d.sy) * 0.1;
                    }

                    let drawX = d.sx;
                    let drawY = d.sy;
                    drawY += Math.sin(d.ax * 0.03 + t) * 0.35;
                    drawX += Math.cos(d.ay * 0.03 + t * 0.7) * 0.2;

                    // Sparkle effect
                    const hash = ((i * 2654435761) ^ (frameCount >> 3)) >>> 0;
                    if ((hash % 100) < 3) {
                        ctx.moveTo(drawX + rad * 1.6, drawY);
                        ctx.arc(drawX, drawY, rad * 1.6, 0, TWO_PI);
                    } else {
                        ctx.moveTo(drawX + rad, drawY);
                        ctx.arc(drawX, drawY, rad, 0, TWO_PI);
                    }
                }

                ctx.fill();

                animationFrameId = requestAnimationFrame(tick);
            }

            animationFrameId = requestAnimationFrame(tick);
        })();

        // -------------------------------------------------------------
        // Single Page Navigation & Tab Switching
        // -------------------------------------------------------------
        const tabs = ['dashboard', 'single', 'batch', 'performance', 'reports'];

        function switchTab(targetTab) {
            tabs.forEach(tab => {
                const viewEl = document.getElementById(`view-${tab}`);
                const navEl = document.getElementById(`nav-${tab}`);
                
                if (tab === targetTab) {
                    viewEl.classList.remove('hidden');
                    if (navEl) {
                        navEl.setAttribute('aria-selected', 'true');
                        navEl.className = "w-full min-h-[44px] flex items-center gap-3.5 px-4 py-3 rounded-xl text-primary bg-indigo-500/10 font-bold transition-all border border-indigo-500/20 shadow-sm focus-visible:ring-2 focus-visible:ring-primary";
                    }
                } else {
                    viewEl.classList.add('hidden');
                    if (navEl) {
                        navEl.setAttribute('aria-selected', 'false');
                        navEl.className = "w-full min-h-[44px] flex items-center gap-3.5 px-4 py-3 rounded-xl text-on-surface-variant hover:text-primary hover:bg-indigo-500/5 font-medium transition-all focus-visible:ring-2 focus-visible:ring-primary";
                    }
                }
            });

            if (window.setDotFieldTabMode) {
                window.setDotFieldTabMode(targetTab);
            }

            if (targetTab === 'reports' || targetTab === 'performance') {
                loadMetrics();
            }
        }

        // Global Modal Pop-Up Lightbox & Latest Proportions State
        let modalChartInstance = null;
        let latestSentimentProportions = { pos: 0, neu: 0, neg: 0 };

        function openChartModal(chartType) {
            if (typeof Chart === 'undefined') {
                showToast("Chart.js is still loading or unavailable.", "warning");
                return;
            }
            const modal = document.getElementById('chart-modal');
            const titleEl = document.getElementById('modal-chart-title');
            const subtitleEl = document.getElementById('modal-chart-subtitle');
            const statsGrid = document.getElementById('modal-stats-grid');
            const ctxModal = document.getElementById('modal-chart-canvas');

            if (!modal || !ctxModal) return;

            if (modalChartInstance) {
                modalChartInstance.destroy();
                modalChartInstance = null;
            }

            const textColor = getChartTextColor();
            const mutedColor = getChartMutedColor();

            if (chartType === 'sentiment') {
                titleEl.innerHTML = `<span class="material-symbols-outlined text-indigo-400 text-2xl">donut_large</span> Sentiment Class Distribution (Expanded Modal View)`;
                subtitleEl.innerText = "High-resolution distribution breakdown for the current active dataset.";

                const p = latestSentimentProportions.pos || 0;
                const n = latestSentimentProportions.neu || 0;
                const neg = latestSentimentProportions.neg || 0;

                modalChartInstance = new Chart(ctxModal, {
                    type: 'doughnut',
                    data: {
                        labels: ['Positive', 'Neutral', 'Negative'],
                        datasets: [{
                            data: [p, n, neg],
                            backgroundColor: [
                                'rgba(16, 185, 129, 0.85)',
                                'rgba(245, 158, 11, 0.85)',
                                'rgba(239, 68, 68, 0.85)'
                            ],
                            hoverBackgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                            borderColor: isDarkMode() ? 'rgba(23, 20, 46, 0.6)' : 'rgba(255, 255, 255, 0.8)',
                            borderWidth: 3,
                            borderRadius: 6,
                            spacing: 4,
                            hoverOffset: 18
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '62%',
                        animation: { animateRotate: true, animateScale: true, duration: 800, easing: 'easeOutQuart' },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'bottom',
                                labels: {
                                    font: { family: 'Plus Jakarta Sans', size: 13, weight: '700' },
                                    color: textColor,
                                    usePointStyle: true,
                                    pointStyleWidth: 14,
                                    padding: 20
                                }
                            },
                            tooltip: {
                                enabled: true,
                                backgroundColor: isDarkMode() ? 'rgba(23, 20, 46, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                                titleColor: textColor,
                                bodyColor: textColor,
                                padding: 14,
                                callbacks: {
                                    label: (ctx) => ` ${ctx.label}: ${ctx.raw}%`
                                }
                            }
                        }
                    }
                });

                statsGrid.innerHTML = `
                    <div class="glass-inner p-4 rounded-2xl">
                        <span class="text-[10px] text-emerald-400 font-bold uppercase tracking-wider block">Positive Sentiment</span>
                        <span class="text-xl font-extrabold text-emerald-400 mt-1 block">${p}%</span>
                        <span class="text-[10px] text-on-surface-variant">Active Class Share</span>
                    </div>
                    <div class="glass-inner p-4 rounded-2xl">
                        <span class="text-[10px] text-amber-400 font-bold uppercase tracking-wider block">Neutral Sentiment</span>
                        <span class="text-xl font-extrabold text-amber-400 mt-1 block">${n}%</span>
                        <span class="text-[10px] text-on-surface-variant">Active Class Share</span>
                    </div>
                    <div class="glass-inner p-4 rounded-2xl">
                        <span class="text-[10px] text-rose-400 font-bold uppercase tracking-wider block">Negative Sentiment</span>
                        <span class="text-xl font-extrabold text-rose-400 mt-1 block">${neg}%</span>
                        <span class="text-[10px] text-on-surface-variant">Active Class Share</span>
                    </div>
                `;
            } else if (chartType === 'timeline') {
                titleEl.innerHTML = `<span class="material-symbols-outlined text-purple-400 text-2xl">show_chart</span> Live Sentiment Trajectory (Expanded Modal View)`;
                subtitleEl.innerText = "Real-time rolling sentiment trajectory across continuous post ingestions.";

                const posGrad = ctxModal.getContext('2d').createLinearGradient(0, 0, 0, 300);
                posGrad.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
                posGrad.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

                const neuGrad = ctxModal.getContext('2d').createLinearGradient(0, 0, 0, 300);
                neuGrad.addColorStop(0, 'rgba(245, 158, 11, 0.4)');
                neuGrad.addColorStop(1, 'rgba(245, 158, 11, 0.0)');

                const negGrad = ctxModal.getContext('2d').createLinearGradient(0, 0, 0, 300);
                negGrad.addColorStop(0, 'rgba(239, 68, 68, 0.4)');
                negGrad.addColorStop(1, 'rgba(239, 68, 68, 0.0)');

                modalChartInstance = new Chart(ctxModal, {
                    type: 'line',
                    data: {
                        labels: timelineData.timestamps.length ? timelineData.timestamps : ['Now'],
                        datasets: [
                            {
                                label: 'Positive %',
                                data: timelineData.posPct.length ? timelineData.posPct : [latestSentimentProportions.pos],
                                borderColor: '#10b981',
                                backgroundColor: posGrad,
                                borderWidth: 3,
                                fill: true,
                                tension: 0.4,
                                pointRadius: 5,
                                pointHoverRadius: 9,
                                pointBackgroundColor: isDarkMode() ? '#17142e' : '#ffffff',
                                pointBorderColor: '#10b981',
                                pointBorderWidth: 2
                            },
                            {
                                label: 'Neutral %',
                                data: timelineData.neuPct.length ? timelineData.neuPct : [latestSentimentProportions.neu],
                                borderColor: '#f59e0b',
                                backgroundColor: neuGrad,
                                borderWidth: 3,
                                fill: true,
                                tension: 0.4,
                                pointRadius: 5,
                                pointHoverRadius: 9,
                                pointBackgroundColor: isDarkMode() ? '#17142e' : '#ffffff',
                                pointBorderColor: '#f59e0b',
                                pointBorderWidth: 2
                            },
                            {
                                label: 'Negative %',
                                data: timelineData.negPct.length ? timelineData.negPct : [latestSentimentProportions.neg],
                                borderColor: '#ef4444',
                                backgroundColor: negGrad,
                                borderWidth: 3,
                                fill: true,
                                tension: 0.4,
                                pointRadius: 5,
                                pointHoverRadius: 9,
                                pointBackgroundColor: isDarkMode() ? '#17142e' : '#ffffff',
                                pointBorderColor: '#ef4444',
                                pointBorderWidth: 2
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: { duration: 600, easing: 'easeOutQuart' },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: { font: { family: 'Plus Jakarta Sans', size: 12, weight: '700' }, color: textColor, usePointStyle: true }
                            },
                            tooltip: {
                                enabled: true,
                                backgroundColor: isDarkMode() ? 'rgba(23, 20, 46, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                                titleColor: textColor,
                                bodyColor: textColor,
                                padding: 14
                            }
                        },
                        scales: {
                            x: {
                                grid: { display: false },
                                ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: mutedColor }
                            },
                            y: {
                                min: 0,
                                max: 100,
                                grid: { color: isDarkMode() ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)' },
                                ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: mutedColor, callback: (v) => `${v}%` }
                            }
                        }
                    }
                });

                statsGrid.innerHTML = `
                    <div class="glass-inner p-4 rounded-2xl">
                        <span class="text-[10px] text-purple-400 font-bold uppercase tracking-wider block">Time Buffers</span>
                        <span class="text-xl font-extrabold text-on-background mt-1 block">${timelineData.timestamps.length || 1} Points</span>
                        <span class="text-[10px] text-on-surface-variant">Rolling Sequence</span>
                    </div>
                    <div class="glass-inner p-4 rounded-2xl">
                        <span class="text-[10px] text-indigo-400 font-bold uppercase tracking-wider block">Ingestion Source</span>
                        <span class="text-xl font-extrabold text-indigo-400 mt-1 block">${window.currentActiveDataset ? 'Batch Upload' : 'Live Ingestion'}</span>
                        <span class="text-[10px] text-on-surface-variant">Active Mode</span>
                    </div>
                    <div class="glass-inner p-4 rounded-2xl">
                        <span class="text-[10px] text-emerald-400 font-bold uppercase tracking-wider block">Telemetry Status</span>
                        <span class="text-xl font-extrabold text-emerald-400 mt-1 block">Live Active</span>
                        <span class="text-[10px] text-on-surface-variant">Real-Time Sync</span>
                    </div>
                `;
            }

            modal.classList.remove('opacity-0', 'pointer-events-none');
            modal.classList.add('opacity-100', 'pointer-events-auto');
            const content = document.getElementById('chart-modal-content');
            if (content) {
                content.classList.remove('scale-95');
                content.classList.add('scale-100');
            }
        }

        function closeChartModal() {
            const modal = document.getElementById('chart-modal');
            const content = document.getElementById('chart-modal-content');
            if (content) {
                content.classList.remove('scale-100');
                content.classList.add('scale-95');
            }
            if (modal) {
                modal.classList.remove('opacity-100', 'pointer-events-auto');
                modal.classList.add('opacity-0', 'pointer-events-none');
            }
        }

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeChartModal();
        });

        function initDashboardCharts() {
            if (typeof Chart === 'undefined') {
                console.warn('Chart.js library is not available');
                return;
            }
            const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            const textColor = getChartTextColor();
            const mutedColor = getChartMutedColor();

            // 1. Sentiment Distribution Donut Chart
            const ctxDonutSentiment = document.getElementById('dash-sentiment-chart');
            if (ctxDonutSentiment) {
                dashboardDoughnutChart = new Chart(ctxDonutSentiment, {
                    type: 'doughnut',
                    data: {
                        labels: ['Positive', 'Neutral', 'Negative'],
                        datasets: [{
                            data: [0, 0, 0],
                            backgroundColor: [
                                'rgba(16, 185, 129, 0.85)',
                                'rgba(245, 158, 11, 0.85)',
                                'rgba(239, 68, 68, 0.85)'
                            ],
                            hoverBackgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                            borderColor: isDarkMode() ? 'rgba(23, 20, 46, 0.6)' : 'rgba(255, 255, 255, 0.8)',
                            borderWidth: 3,
                            borderRadius: 6,
                            spacing: 4,
                            hoverOffset: 12
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '68%',
                        animation: prefersReduced ? false : { animateRotate: true, animateScale: true, duration: 800, easing: 'easeOutQuart' },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                enabled: true,
                                backgroundColor: isDarkMode() ? 'rgba(23, 20, 46, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                                titleColor: textColor,
                                bodyColor: textColor,
                                borderColor: isDarkMode() ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)',
                                borderWidth: 1,
                                padding: 12,
                                callbacks: {
                                    label: function(context) {
                                        return ` ${context.label}: ${context.raw}%`;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            // 1B. Class Performance & Metrics Chart (Precision, Recall, F1)
            const ctxClassChart = document.getElementById('dash-class-chart');
            if (ctxClassChart) {
                dashClassMetricsChart = new Chart(ctxClassChart, {
                    type: 'bar',
                    data: {
                        labels: ['Positive', 'Neutral', 'Negative'],
                        datasets: [
                            {
                                label: 'Precision %',
                                data: [66.3, 67.6, 66.3],
                                backgroundColor: isDarkMode() ? '#818cf8' : '#4f46e5',
                                borderRadius: 6
                            },
                            {
                                label: 'Recall %',
                                data: [66.1, 67.6, 74.2],
                                backgroundColor: isDarkMode() ? '#34d399' : '#10b981',
                                borderRadius: 6
                            },
                            {
                                label: 'F1 Score %',
                                data: [66.4, 67.6, 67.0],
                                backgroundColor: isDarkMode() ? '#c084fc' : '#a855f7',
                                borderRadius: 6
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: prefersReduced ? false : { duration: 800, easing: 'easeOutQuart' },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: {
                                    font: { family: 'Plus Jakarta Sans', size: 10, weight: '700' },
                                    color: textColor,
                                    boxWidth: 10,
                                    padding: 10
                                }
                            },
                            tooltip: {
                                enabled: true,
                                backgroundColor: isDarkMode() ? 'rgba(23, 20, 46, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                                titleColor: textColor,
                                bodyColor: textColor,
                                borderColor: isDarkMode() ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)',
                                borderWidth: 1,
                                padding: 10,
                                callbacks: {
                                    label: function(context) {
                                        return ` ${context.dataset.label}: ${context.raw}%`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                grid: { display: false },
                                ticks: { font: { family: 'Plus Jakarta Sans', size: 10, weight: '700' }, color: textColor }
                            },
                            y: {
                                min: 50,
                                max: 100,
                                grid: { color: isDarkMode() ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.05)' },
                                ticks: {
                                    font: { family: 'JetBrains Mono', size: 9 },
                                    color: mutedColor,
                                    callback: function(val) { return val + '%'; }
                                }
                            }
                        }
                    }
                });
            }

            // 2. Live Item Confidence Levels Bar Chart
            const ctxBar = document.getElementById('live-confidence-chart');
            if (ctxBar) {
                liveConfidenceBarChart = new Chart(ctxBar, {
                    type: 'bar',
                    data: {
                        labels: ['<50%', '50-70%', '70-90%', '>90%'],
                        datasets: [{
                            label: 'Ingested Items',
                            data: [0, 0, 0, 0],
                            backgroundColor: isDarkMode() ? '#818cf8' : '#4f46e5',
                            borderRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: prefersReduced ? false : { duration: 600 },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                enabled: true,
                                backgroundColor: isDarkMode() ? '#111827' : '#ffffff',
                                titleColor: textColor,
                                bodyColor: textColor
                            }
                        },
                        scales: {
                            x: { grid: { display: false }, ticks: { font: { family: 'Plus Jakarta Sans', size: 9, weight: '700' }, color: mutedColor } },
                            y: { display: false }
                        }
                    }
                });
            }

            // 3. Dynamic Multi-Model Benchmark Comparison Chart
            const ctxBenchmark = document.getElementById('perf-benchmark-chart');
            if (ctxBenchmark) {
                perfBenchmarkChart = new Chart(ctxBenchmark, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'Macro F1',
                                data: [],
                                backgroundColor: isDarkMode() ? '#818cf8' : '#4f46e5',
                                borderRadius: 8
                            },
                            {
                                label: 'Test Accuracy',
                                data: [],
                                backgroundColor: isDarkMode() ? '#34d399' : '#10b981',
                                borderRadius: 8
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: prefersReduced ? false : { duration: 800, easing: 'easeOutQuart' },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: { font: { family: 'Plus Jakarta Sans', size: 11, weight: '700' }, color: textColor, boxWidth: 12 }
                            },
                            tooltip: {
                                enabled: true,
                                backgroundColor: isDarkMode() ? '#111827' : '#ffffff',
                                titleColor: textColor,
                                bodyColor: textColor,
                                borderColor: isDarkMode() ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                                borderWidth: 1,
                                padding: 12
                            }
                        },
                        scales: {
                            x: { grid: { display: false }, ticks: { font: { family: 'Plus Jakarta Sans', size: 10, weight: '700' }, color: textColor } },
                            y: { min: 0.0, max: 1.0, ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: mutedColor } }
                        }
                    }
                });
            }

            // 4. Real-time Sentiment Trend Timeline Line Chart (Attractive Curved Gradient Area Styling & Empty Initial State)
            const ctxTimeline = document.getElementById('live-timeline-chart');
            if (ctxTimeline) {
                const posGrad = ctxTimeline.getContext('2d').createLinearGradient(0, 0, 0, 200);
                posGrad.addColorStop(0, 'rgba(16, 185, 129, 0.35)');
                posGrad.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

                const neuGrad = ctxTimeline.getContext('2d').createLinearGradient(0, 0, 0, 200);
                neuGrad.addColorStop(0, 'rgba(245, 158, 11, 0.35)');
                neuGrad.addColorStop(1, 'rgba(245, 158, 11, 0.0)');

                const negGrad = ctxTimeline.getContext('2d').createLinearGradient(0, 0, 0, 200);
                negGrad.addColorStop(0, 'rgba(239, 68, 68, 0.35)');
                negGrad.addColorStop(1, 'rgba(239, 68, 68, 0.0)');

                liveTimelineLineChart = new Chart(ctxTimeline, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'Positive %',
                                data: [],
                                borderColor: '#10b981',
                                backgroundColor: posGrad,
                                borderWidth: 2.5,
                                fill: true,
                                tension: 0.4,
                                pointRadius: 4,
                                pointHoverRadius: 7,
                                pointBackgroundColor: isDarkMode() ? '#17142e' : '#ffffff',
                                pointBorderColor: '#10b981',
                                pointBorderWidth: 2
                            },
                            {
                                label: 'Neutral %',
                                data: [],
                                borderColor: '#f59e0b',
                                backgroundColor: neuGrad,
                                borderWidth: 2.5,
                                fill: true,
                                tension: 0.4,
                                pointRadius: 4,
                                pointHoverRadius: 7,
                                pointBackgroundColor: isDarkMode() ? '#17142e' : '#ffffff',
                                pointBorderColor: '#f59e0b',
                                pointBorderWidth: 2
                            },
                            {
                                label: 'Negative %',
                                data: [],
                                borderColor: '#ef4444',
                                backgroundColor: negGrad,
                                borderWidth: 2.5,
                                fill: true,
                                tension: 0.4,
                                pointRadius: 4,
                                pointHoverRadius: 7,
                                pointBackgroundColor: isDarkMode() ? '#17142e' : '#ffffff',
                                pointBorderColor: '#ef4444',
                                pointBorderWidth: 2
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: prefersReduced ? false : { duration: 650, easing: 'easeOutCubic' },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: {
                                    font: { family: 'Plus Jakarta Sans', size: 11, weight: '700' },
                                    usePointStyle: true,
                                    pointStyle: 'circle',
                                    color: textColor,
                                    boxWidth: 10
                                }
                            },
                            tooltip: {
                                enabled: true,
                                backgroundColor: isDarkMode() ? 'rgba(23, 20, 46, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                                titleColor: textColor,
                                bodyColor: textColor,
                                borderColor: isDarkMode() ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)',
                                borderWidth: 1,
                                padding: 12,
                                boxPadding: 6
                            }
                        },
                        scales: {
                            x: {
                                grid: { display: false },
                                ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: mutedColor }
                            },
                            y: {
                                min: 0,
                                max: 100,
                                grid: { color: isDarkMode() ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.05)' },
                                ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: mutedColor }
                            }
                        }
                    }
                });
            }
        }

        function updateChartThemes() {
            const textColor = getChartTextColor();
            const mutedColor = getChartMutedColor();

            if (dashboardDoughnutChart) {
                dashboardDoughnutChart.data.datasets[0].borderColor = isDarkMode() ? 'rgba(23, 20, 46, 0.6)' : 'rgba(255, 255, 255, 0.8)';
                dashboardDoughnutChart.update();
            }
            if (dashClassMetricsChart) {
                dashClassMetricsChart.options.plugins.legend.labels.color = textColor;
                dashClassMetricsChart.options.scales.x.ticks.color = textColor;
                dashClassMetricsChart.options.scales.y.ticks.color = mutedColor;
                dashClassMetricsChart.options.scales.y.grid.color = isDarkMode() ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.05)';
                dashClassMetricsChart.data.datasets[0].backgroundColor = isDarkMode() ? '#818cf8' : '#4f46e5';
                dashClassMetricsChart.data.datasets[1].backgroundColor = isDarkMode() ? '#34d399' : '#10b981';
                dashClassMetricsChart.data.datasets[2].backgroundColor = isDarkMode() ? '#c084fc' : '#a855f7';
                dashClassMetricsChart.update();
            }
            if (liveConfidenceBarChart) {
                liveConfidenceBarChart.options.scales.x.ticks.color = mutedColor;
                liveConfidenceBarChart.data.datasets[0].backgroundColor = isDarkMode() ? '#818cf8' : '#4f46e5';
                liveConfidenceBarChart.update();
            }
            if (perfBenchmarkChart) {
                perfBenchmarkChart.options.plugins.legend.labels.color = textColor;
                perfBenchmarkChart.options.scales.x.ticks.color = textColor;
                perfBenchmarkChart.options.scales.y.ticks.color = mutedColor;
                perfBenchmarkChart.data.datasets[0].backgroundColor = isDarkMode() ? '#818cf8' : '#4f46e5';
                perfBenchmarkChart.data.datasets[1].backgroundColor = isDarkMode() ? '#34d399' : '#10b981';
                perfBenchmarkChart.update();
            }
            if (liveTimelineLineChart) {
                liveTimelineLineChart.options.plugins.legend.labels.color = textColor;
                liveTimelineLineChart.options.scales.x.ticks.color = mutedColor;
                liveTimelineLineChart.options.scales.y.ticks.color = mutedColor;
                liveTimelineLineChart.options.scales.y.grid.color = isDarkMode() ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.05)';
                liveTimelineLineChart.data.datasets.forEach(ds => {
                    ds.pointBackgroundColor = isDarkMode() ? '#17142e' : '#ffffff';
                });
                liveTimelineLineChart.update();
            }
        }

        function updateDonutCenterText(posPct, neuPct, negPct) {
            const centerValue = document.getElementById('donut-center-value');
            const centerSentiment = document.getElementById('donut-center-sentiment');
            const centerLabel = document.getElementById('donut-center-label');

            if (!centerValue || !centerSentiment) return;

            const total = posPct + neuPct + negPct;
            if (total === 0) {
                centerValue.innerText = '--';
                centerSentiment.innerText = 'No Data';
                centerSentiment.className = 'text-xs font-bold text-on-surface-variant uppercase';
                if (centerLabel) centerLabel.innerText = 'Awaiting';
                return;
            }

            // Find the dominant sentiment
            if (posPct >= neuPct && posPct >= negPct) {
                centerValue.innerText = `${posPct}%`;
                centerSentiment.innerText = 'Positive';
                centerSentiment.className = 'text-xs font-bold text-emerald-400 uppercase';
            } else if (neuPct >= posPct && neuPct >= negPct) {
                centerValue.innerText = `${neuPct}%`;
                centerSentiment.innerText = 'Neutral';
                centerSentiment.className = 'text-xs font-bold text-amber-400 uppercase';
            } else {
                centerValue.innerText = `${negPct}%`;
                centerSentiment.innerText = 'Negative';
                centerSentiment.className = 'text-xs font-bold text-rose-400 uppercase';
            }
            if (centerLabel) centerLabel.innerText = 'Dominant';
        }

        function updateDashboardChart(posPct, neuPct, negPct) {
            latestSentimentProportions = { pos: posPct, neu: neuPct, neg: negPct };

            document.getElementById('chart-stat-pos').innerText = `${posPct}%`;
            document.getElementById('chart-stat-neu').innerText = `${neuPct}%`;
            document.getElementById('chart-stat-neg').innerText = `${negPct}%`;

            const srPos = document.getElementById('sr-dash-pos');
            if (srPos) srPos.innerText = `${posPct}%`;
            const srNeu = document.getElementById('sr-dash-neu');
            if (srNeu) srNeu.innerText = `${neuPct}%`;
            const srNeg = document.getElementById('sr-dash-neg');
            if (srNeg) srNeg.innerText = `${negPct}%`;

            // Update donut center text
            updateDonutCenterText(posPct, neuPct, negPct);

            if (dashboardDoughnutChart) {
                dashboardDoughnutChart.data.datasets[0].data = [posPct, neuPct, negPct];
                dashboardDoughnutChart.update('active');
            }

            // Push to timeline buffer
            if (liveTimelineLineChart) {
                const nowStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                timelineData.timestamps.push(nowStr);
                timelineData.posPct.push(posPct);
                timelineData.neuPct.push(neuPct);
                timelineData.negPct.push(negPct);

                if (timelineData.timestamps.length > 7) {
                    timelineData.timestamps.shift();
                    timelineData.posPct.shift();
                    timelineData.neuPct.shift();
                    timelineData.negPct.shift();
                }

                liveTimelineLineChart.data.labels = timelineData.timestamps;
                liveTimelineLineChart.data.datasets[0].data = timelineData.posPct;
                liveTimelineLineChart.data.datasets[1].data = timelineData.neuPct;
                liveTimelineLineChart.data.datasets[2].data = timelineData.negPct;
                liveTimelineLineChart.update();
            }
        }

        // -------------------------------------------------------------
        // Resilient Network Client & Dynamic API Base Configuration
        // -------------------------------------------------------------
        function getApiBase() {
            const custom = localStorage.getItem('SENTIMENTSCOPE_API_URL');
            // If stored URL was set to Vercel origin fallback or old non-existent Render URL, clear it
            if (custom && custom.trim()) {
                if ((custom.trim() === window.location.origin && window.location.protocol !== 'file:') || custom.includes('sentimentscope-api.onrender.com')) {
                    localStorage.removeItem('SENTIMENTSCOPE_API_URL');
                } else {
                    return custom.trim().replace(/\/+$/, '');
                }
            }
            if (window.ENV_API_URL && window.ENV_API_URL.trim()) return window.ENV_API_URL.trim().replace(/\/+$/, '');
            const metaUrl = document.querySelector('meta[name="api-base-url"]')?.getAttribute('content');
            if (metaUrl && metaUrl.trim() && !metaUrl.includes('__RENDER_URL__') && !metaUrl.includes('__FLY_URL__') && !metaUrl.includes('__RAILWAY_URL__')) return metaUrl.trim().replace(/\/+$/, '');
            if (window.location.protocol === 'file:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                return 'http://127.0.0.1:8000';
            }
            // Default to live Render backend service URL when hosted on Vercel or cloud CDN
            return 'https://sentimentscope-api-nj7l.onrender.com';
        }

        function setApiBase(url) {
            if (!url || !url.trim() || url === window.location.origin) {
                localStorage.removeItem('SENTIMENTSCOPE_API_URL');
            } else {
                localStorage.setItem('SENTIMENTSCOPE_API_URL', url.trim().replace(/\/+$/, ''));
            }
        }

        async function fetchWithTimeout(url, options = {}, timeoutMs = 25000) {
            const controller = new AbortController();
            const id = setTimeout(() => controller.abort(), timeoutMs);
            try {
                let apiBase = getApiBase().replace(/\/+$/, '');
                // Sanitize any redundant or leftover /api prefix
                let cleanUrl = url;
                if (cleanUrl.startsWith('/api/')) {
                    cleanUrl = cleanUrl.replace(/^\/api/, '');
                }
                const finalUrl = cleanUrl.startsWith('http') ? cleanUrl : `${apiBase}${cleanUrl}`;
                const response = await fetch(finalUrl, {
                    ...options,
                    signal: controller.signal
                });
                clearTimeout(id);
                return response;
            } catch (err) {
                clearTimeout(id);
                if (err.name === 'AbortError') {
                    throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
                }
                throw err;
            }
        }

        async function parseErrorDetail(res, defaultMsg = 'Request failed') {
            let msg = `HTTP ${res.status}: ${defaultMsg}`;
            try {
                const text = await res.text();
                if (text && text.trim()) {
                    try {
                        const parsed = JSON.parse(text);
                        msg = parsed.detail || parsed.message || msg;
                    } catch (_) {
                        if (text.length < 200) msg = text.trim();
                    }
                }
            } catch (_) {}
            return msg;
        }

        async function checkHealth() {
            const ind = document.getElementById('health-indicator');
            const banner = document.getElementById('backend-offline-banner');
            const targetEl = document.getElementById('offline-target-url');
            let activeUrl = getApiBase();
            if (targetEl) targetEl.innerText = activeUrl;

            const startPing = performance.now();
            try {
                let res;
                try {
                    res = await fetchWithTimeout('/health', {}, 5000);
                } catch (firstErr) {
                    // Fallback attempt to window.location.origin if custom URL failed
                    if (activeUrl !== window.location.origin && window.location.protocol !== 'file:') {
                        console.warn("Primary API base failed, attempting same-origin fallback:", window.location.origin);
                        const fallbackUrl = `${window.location.origin}/health`;
                        res = await fetch(fallbackUrl);
                        if (res.ok) {
                            setApiBase(window.location.origin);
                            activeUrl = window.location.origin;
                            if (targetEl) targetEl.innerText = activeUrl;
                        } else {
                            throw firstErr;
                        }
                    } else {
                        throw firstErr;
                    }
                }

                const latencyMs = Math.round(performance.now() - startPing);
                if (!res.ok) throw new Error("Status: " + res.status);
                const data = await res.json();
                if (data.status === 'healthy') {
                    if (ind) ind.className = "w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse-glow";
                    safeSetText('health-text', `API Online (${latencyMs}ms)`);
                    if (banner) banner.classList.add('hidden');
                    if (data.model_name) {
                        safeSetText('dash-model-name', data.model_name);
                        safeSetText('perf-best-model', data.model_name);
                    }
                    return true;
                } else {
                    if (ind) ind.className = "w-2.5 h-2.5 rounded-full bg-amber-400";
                    safeSetText('health-text', "Degraded");
                    if (banner) banner.classList.add('hidden');
                    return false;
                }
            } catch (err) {
                console.warn("API health check failed:", err);
                if (ind) ind.className = "w-2.5 h-2.5 rounded-full bg-rose-500";
                safeSetText('health-text', "Offline");
                safeSetText('dash-model-name', "Server Offline");
                if (banner) banner.classList.remove('hidden');
                return false;
            }
        }

        // Toast Helper
        function showToast(msg, icon = 'info') {
            const toast = document.getElementById('toast');
            document.getElementById('toast-msg').innerText = msg;
            document.getElementById('toast-icon').innerText = icon;
            
            toast.classList.remove('translate-y-20', 'opacity-0');
            setTimeout(() => {
                toast.classList.add('translate-y-20', 'opacity-0');
            }, 3500);
        }

        // -------------------------------------------------------------
        // Quick & Single Sentiment Prediction
        // -------------------------------------------------------------
        function setSampleText(txt) {
            document.getElementById('dash-quick-input').value = txt;
        }

        const updateCharCount = debounce(function() {
            const el = document.getElementById('single-input');
            if (el) {
                safeSetText('single-char-count', `${el.value.length} characters`);
            }
        }, 100);

        function copySingleResult() {
            if (!latestSingleResult) {
                showToast("Run an analysis first to copy the outcome.", "info");
                return;
            }
            const formatted = JSON.stringify(latestSingleResult, null, 2);
            navigator.clipboard.writeText(formatted).then(() => {
                showToast("Full analysis outcome copied to clipboard!", "content_copy");
            }).catch(() => {
                showToast("Failed to copy outcome to clipboard.", "error");
            });
        }

        function copyQuickResult() {
            if (!latestQuickResult) {
                showToast("Run quick prediction first to copy the outcome.", "info");
                return;
            }
            const snippet = `"${latestQuickResult.text}" -> ${latestQuickResult.sentiment.toUpperCase()} (Confidence: ${(latestQuickResult.confidence * 100).toFixed(1)}%)`;
            navigator.clipboard.writeText(snippet).then(() => {
                showToast("Quick prediction copied to clipboard!", "content_copy");
            }).catch(() => {
                showToast("Failed to copy quick prediction.", "error");
            });
        }

        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        async function runQuickPredict() {
            const text = document.getElementById('dash-quick-input').value.trim();
            if (!text) {
                showToast("Please enter text before running test.", "warning");
                return;
            }

            const btn = document.getElementById('dash-quick-btn');
            btn.disabled = true;
            btn.innerHTML = `<span class="material-symbols-outlined text-base animate-spin-fast">sync</span> Processing...`;

            try {
                const res = await fetchWithTimeout('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });

                if (!res.ok) {
                    const errMsg = await parseErrorDetail(res, 'Quick prediction failed');
                    throw new Error(errMsg);
                }

                const data = await res.json();
                latestQuickResult = { text: text, ...data };
                document.getElementById('dash-quick-result').classList.remove('hidden');
                
                const badge = document.getElementById('dash-res-badge');
                badge.innerText = data.sentiment;
                if (data.sentiment === 'positive') {
                    badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
                } else if (data.sentiment === 'negative') {
                    badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide bg-rose-500/10 text-rose-400 border border-rose-500/30";
                } else {
                    badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide bg-amber-500/10 text-amber-400 border border-amber-500/30";
                }

                document.getElementById('dash-prob-pos').innerText = `${(data.probabilities.positive * 100).toFixed(1)}%`;
                document.getElementById('dash-prob-neu').innerText = `${(data.probabilities.neutral * 100).toFixed(1)}%`;
                document.getElementById('dash-prob-neg').innerText = `${(data.probabilities.negative * 100).toFixed(1)}%`;

                if (window.triggerDotFieldPulse) {
                    window.triggerDotFieldPulse(data.sentiment);
                }

                showToast(`Analysis complete: ${data.sentiment.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`, "check_circle");
            } catch (err) {
                showToast(err.message, "error");
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<span class="material-symbols-outlined text-base">send</span> Test Sentiment`;
            }
        }

        async function runSinglePredict() {
            const text = document.getElementById('single-input').value.trim();
            if (!text) {
                showToast("Text input cannot be empty.", "warning");
                return;
            }

            const btn = document.getElementById('single-analyze-btn');
            const skel = document.getElementById('single-result-skeleton');
            btn.disabled = true;
            btn.innerHTML = `<span class="material-symbols-outlined text-lg animate-spin-fast">sync</span> Analyzing...`;
            if (skel) skel.classList.remove('hidden');

            try {
                const res = await fetchWithTimeout('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });

                if (!res.ok) {
                    const errMsg = await parseErrorDetail(res, 'Single prediction failed');
                    throw new Error(errMsg);
                }

                const data = await res.json();
                latestSingleResult = { text: text, ...data };
                
                document.getElementById('single-sentiment-label').innerText = data.sentiment;
                const badge = document.getElementById('single-sentiment-badge');
                if (data.sentiment === 'positive') {
                    badge.className = "glass-inner px-4 py-1.5 rounded-full flex items-center gap-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
                } else if (data.sentiment === 'negative') {
                    badge.className = "glass-inner px-4 py-1.5 rounded-full flex items-center gap-2 bg-rose-500/10 text-rose-400 border border-rose-500/30";
                } else {
                    badge.className = "glass-inner px-4 py-1.5 rounded-full flex items-center gap-2 bg-amber-500/10 text-amber-400 border border-amber-500/30";
                }

                if (window.triggerDotFieldPulse) {
                    window.triggerDotFieldPulse(data.sentiment);
                }

                document.getElementById('single-cleaned-text').innerText = `"${data.cleaned_text || '(no remaining stop words)'}"`;
                document.getElementById('single-confidence').innerText = `${(data.confidence * 100).toFixed(1)}%`;
                document.getElementById('single-confidence-bar').style.width = `${(data.confidence * 100).toFixed(1)}%`;

                const posPct = (data.probabilities.positive * 100).toFixed(1);
                const neuPct = (data.probabilities.neutral * 100).toFixed(1);
                const negPct = (data.probabilities.negative * 100).toFixed(1);

                document.getElementById('prob-val-pos').innerText = `${posPct}%`;
                document.getElementById('prob-bar-pos').style.width = `${posPct}%`;

                document.getElementById('prob-val-neu').innerText = `${neuPct}%`;
                document.getElementById('prob-bar-neu').style.width = `${neuPct}%`;

                document.getElementById('prob-val-neg').innerText = `${negPct}%`;
                document.getElementById('prob-bar-neg').style.width = `${negPct}%`;

                document.getElementById('single-latency').innerText = `${data.latency_ms} ms`;
                if (data.pipeline_model) {
                    const smEl = document.getElementById('single-model-meta');
                    if (smEl) smEl.innerText = data.pipeline_model;
                }
                if (data.tokenizer_type) {
                    const stEl = document.getElementById('single-tokenizer-meta');
                    if (stEl) stEl.innerText = data.tokenizer_type;
                }

                showToast(`Single prediction executed in ${data.latency_ms}ms`, "check_circle");
            } catch (err) {
                showToast(err.message, "error");
            } finally {
                if (skel) skel.classList.add('hidden');
                btn.disabled = false;
                btn.innerHTML = `<span class="material-symbols-outlined text-lg">auto_awesome</span> Analyze Sentiment`;
            }
        }

        // -------------------------------------------------------------
        // Batch File Upload Processing
        // -------------------------------------------------------------
        function downloadSampleCSV() {
            const csvContent = "data:text/csv;charset=utf-8," + 
                "text\n" +
                "\"The customer support team was incredibly helpful and quick to resolve my issue.\"\n" +
                "\"Package arrived on Thursday as scheduled in plain packaging.\"\n" +
                "\"Extremely disappointed with the poor build quality, broke immediately.\"\n" +
                "\"Fantastic experience overall! Highly recommended to everyone.\"\n" +
                "\"Item matches description provided on the page.\"\n";
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "sample_sentiment_batch.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function handleFileSelect(evt) {
            const files = evt.target.files;
            if (files.length > 0) {
                processCSV(files[0]);
            }
        }

        const dropZone = document.getElementById('drop-zone');
        if (dropZone) {
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
                dropZone.addEventListener(evt, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                }, false);
            });

            ['dragenter', 'dragover'].forEach(evt => {
                dropZone.addEventListener(evt, () => dropZone.classList.add('border-indigo-500', 'bg-indigo-500/5'), false);
            });

            ['dragleave', 'drop'].forEach(evt => {
                dropZone.addEventListener(evt, () => dropZone.classList.remove('border-indigo-500', 'bg-indigo-500/5'), false);
            });

            dropZone.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) {
                    processCSV(files[0]);
                }
            });
        }

        function escapeHtml(str) {
            if (!str) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        let batchState = {
            allPredictions: [],
            filteredPredictions: [],
            currentPage: 1,
            pageSize: 25
        };

        function initBatchPaginationControls() {
            const searchInput = document.getElementById('batch-search-input');
            const pageSizeSelect = document.getElementById('batch-page-size');
            const prevBtn = document.getElementById('batch-prev-page');
            const nextBtn = document.getElementById('batch-next-page');

            if (searchInput) {
                searchInput.addEventListener('input', () => {
                    const q = searchInput.value.trim().toLowerCase();
                    if (!q) {
                        batchState.filteredPredictions = batchState.allPredictions;
                    } else {
                        batchState.filteredPredictions = batchState.allPredictions.filter(item =>
                            (item.text && item.text.toLowerCase().includes(q)) ||
                            (item.sentiment && item.sentiment.toLowerCase().includes(q))
                        );
                    }
                    batchState.currentPage = 1;
                    renderBatchTablePage();
                });
            }

            if (pageSizeSelect) {
                pageSizeSelect.addEventListener('change', () => {
                    batchState.pageSize = parseInt(pageSizeSelect.value, 10) || 25;
                    batchState.currentPage = 1;
                    renderBatchTablePage();
                });
            }

            if (prevBtn) {
                prevBtn.addEventListener('click', () => {
                    if (batchState.currentPage > 1) {
                        batchState.currentPage--;
                        renderBatchTablePage();
                    }
                });
            }

            if (nextBtn) {
                nextBtn.addEventListener('click', () => {
                    const totalPages = Math.ceil(batchState.filteredPredictions.length / batchState.pageSize) || 1;
                    if (batchState.currentPage < totalPages) {
                        batchState.currentPage++;
                        renderBatchTablePage();
                    }
                });
            }
        }

        function renderBatchTablePage() {
            const tableBody = document.getElementById('batch-table-body');
            const paginationInfo = document.getElementById('batch-pagination-info');
            const pageIndicator = document.getElementById('batch-page-indicator');
            const prevBtn = document.getElementById('batch-prev-page');
            const nextBtn = document.getElementById('batch-next-page');

            if (!tableBody) return;

            const total = batchState.filteredPredictions.length;
            const pageSize = batchState.pageSize;
            const totalPages = Math.ceil(total / pageSize) || 1;

            if (batchState.currentPage > totalPages) batchState.currentPage = totalPages;
            if (batchState.currentPage < 1) batchState.currentPage = 1;

            const startIdx = (batchState.currentPage - 1) * pageSize;
            const endIdx = Math.min(startIdx + pageSize, total);
            const pageItems = batchState.filteredPredictions.slice(startIdx, endIdx);

            tableBody.innerHTML = '';
            if (pageItems.length === 0) {
                tableBody.innerHTML = `<tr><td colspan="5" class="p-6 text-center text-on-surface-variant font-medium">No matching predictions found in preview sample.</td></tr>`;
            } else {
                pageItems.forEach((item, idx) => {
                    const tr = document.createElement('tr');
                    let badgeClass = "bg-amber-500/10 text-amber-400 border border-amber-500/30";
                    if (item.sentiment === 'positive') badgeClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
                    if (item.sentiment === 'negative') badgeClass = "bg-rose-500/10 text-rose-400 border border-rose-500/30";

                    tr.innerHTML = `
                        <td class="p-3 font-mono text-on-surface-variant">${startIdx + idx + 1}</td>
                        <td class="p-3 font-body text-on-background max-w-md truncate" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</td>
                        <td class="p-3"><span class="px-2.5 py-0.5 rounded-full font-bold uppercase text-[10px] ${badgeClass}">${item.sentiment}</span></td>
                        <td class="p-3 font-semibold font-mono text-indigo-400">${(item.confidence * 100).toFixed(1)}%</td>
                        <td class="p-3 font-mono text-on-surface-variant">${item.latency_ms}ms</td>
                    `;
                    tableBody.appendChild(tr);
                });
            }

            if (paginationInfo) {
                if (total === 0) {
                    paginationInfo.innerText = "Showing 0 to 0 of 0 preview rows";
                } else {
                    paginationInfo.innerText = `Showing ${(startIdx + 1).toLocaleString()} to ${endIdx.toLocaleString()} of ${total.toLocaleString()} preview rows`;
                }
            }

            if (pageIndicator) {
                pageIndicator.innerText = `Page ${batchState.currentPage} of ${totalPages}`;
            }

            if (prevBtn) prevBtn.disabled = (batchState.currentPage <= 1);
            if (nextBtn) nextBtn.disabled = (batchState.currentPage >= totalPages);
        }

        async function processCSV(file) {
            if (!file.name.endsWith('.csv')) {
                showToast("Please select a valid .csv file.", "warning");
                return;
            }

            // Early warning for massive files (1M rows is typically 50MB - 150MB+)
            if (file.size > 35 * 1024 * 1024) {
                showToast(`File size is ${(file.size / (1024 * 1024)).toFixed(1)}MB. Synchronous web upload is capped at 35MB / 100,000 rows to prevent browser timeouts. For 1 Million rows, use the offline streaming CLI: python src/batch_inference.py`, "warning", 10000);
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            const progContainer = document.getElementById('batch-progress-container');
            const summaryCards = document.getElementById('batch-summary-cards');
            const tableContainer = document.getElementById('batch-results-table-container');

            progContainer.classList.remove('hidden');
            document.getElementById('batch-file-name').innerText = file.name;
            document.getElementById('batch-progress-status').innerText = "Uploading & processing batch...";

            try {
                // Extended timeout to 300s (5 min) to allow server to process up to 100,000 rows comfortably
                const res = await fetchWithTimeout('/predict/batch', {
                    method: 'POST',
                    body: formData
                }, 300000);

                if (!res.ok) {
                    const errMsg = await parseErrorDetail(res, 'Batch processing failed');
                    throw new Error(errMsg);
                }

                const data = await res.json();

                document.getElementById('batch-total-rows').innerText = Number(data.total_rows).toLocaleString();
                document.getElementById('batch-pos-pct').innerText = `${data.positive_pct}%`;
                document.getElementById('batch-neu-pct').innerText = `${data.neutral_pct}%`;
                document.getElementById('batch-neg-pct').innerText = `${data.negative_pct}%`;
                summaryCards.classList.remove('hidden');

                // Update preview callout notice banner
                const noticeBanner = document.getElementById('batch-preview-notice');
                const noticeTotal = document.getElementById('batch-notice-total');
                const noticePreview = document.getElementById('batch-notice-preview');
                const tableSubtitle = document.getElementById('batch-table-subtitle');
                const previewRowsCount = data.preview_count || (data.predictions ? data.predictions.length : 0);

                if (noticeTotal) noticeTotal.innerText = Number(data.total_rows).toLocaleString();
                if (noticePreview) noticePreview.innerText = Number(previewRowsCount).toLocaleString();
                if (tableSubtitle) tableSubtitle.innerText = `Showing preview of first ${Number(previewRowsCount).toLocaleString()} rows out of ${Number(data.total_rows).toLocaleString()} total analyzed rows`;
                if (noticeBanner) noticeBanner.classList.remove('hidden');

                // Populate pagination state & render first page
                batchState.allPredictions = data.predictions || [];
                batchState.filteredPredictions = data.predictions || [];
                batchState.currentPage = 1;
                const searchInput = document.getElementById('batch-search-input');
                if (searchInput) searchInput.value = '';

                renderBatchTablePage();
                tableContainer.classList.remove('hidden');

                // Store active dataset globally so dashboard graphs persist uploaded data
                window.currentActiveDataset = {
                    source: 'batch_upload',
                    total_rows: data.total_rows,
                    positive_pct: data.positive_pct,
                    neutral_pct: data.neutral_pct,
                    negative_pct: data.negative_pct
                };

                // Immediately update dashboard telemetry & charts with uploaded batch dataset!
                updateDashboardChart(data.positive_pct, data.neutral_pct, data.negative_pct);

                // Also refresh the live feed table to switch to batch mode display
                loadLiveFeed();

                if (data.note) {
                    showToast(data.note, "info", 7000);
                }
                showToast(`Batch of ${Number(data.total_rows).toLocaleString()} rows processed & dashboard updated!`, "check_circle");
            } catch (err) {
                showToast(err.message, "error", 9000);
            } finally {
                progContainer.classList.add('hidden');
            }
        }

        // -------------------------------------------------------------
        // Reports / Model Metrics Fetching & Dynamic Benchmark Chart
        // -------------------------------------------------------------
        async function loadMetrics() {
            const benchSkeleton = document.getElementById('perf-benchmark-skeleton');
            const benchError = document.getElementById('perf-benchmark-error');
            if (benchSkeleton) benchSkeleton.classList.remove('hidden');
            if (benchError) benchError.classList.add('hidden');

            try {
                const res = await fetchWithTimeout('/model/metrics', {}, 25000);
                if (!res.ok) throw new Error("Metrics endpoint unavailable");
                const data = await res.json();

                const bestName = data.best_model || data.best_model_name || "Logistic Regression";
                safeSetText('report-best-model-name', bestName);
                safeSetText('dash-model-name', bestName);
                safeSetText('perf-best-model', bestName);

                // Priority: Update primary dashboard accuracy card immediately
                let activeMetrics = data.metrics;
                if (!activeMetrics && data.models && data.models[bestName]) {
                    activeMetrics = data.models[bestName];
                }
                if (activeMetrics) {
                    const primaryAcc = activeMetrics.accuracy !== undefined ? activeMetrics.accuracy : 0.6592;
                    safeSetText('dash-accuracy', `${(primaryAcc * 100).toFixed(1)}%`);
                    safeSetText('perf-accuracy', `${(primaryAcc * 100).toFixed(1)}%`);
                }

                // 1. Dynamic Benchmark Comparison Chart Update
                if (data.models && perfBenchmarkChart) {
                    const modelNames = Object.keys(data.models);
                    const cvF1Scores = [];
                    const accuracyScores = [];
                    const srTbody = document.getElementById('sr-benchmark-tbody');
                    if (srTbody) srTbody.innerHTML = '';

                    modelNames.forEach(mName => {
                        const mData = data.models[mName];
                        const cvF1 = mData.cv_f1_mean !== undefined ? mData.cv_f1_mean : (mData.macro_f1 || 0);
                        const acc = mData.accuracy || 0;
                        cvF1Scores.push(cvF1);
                        accuracyScores.push(acc);

                        if (srTbody) {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `<td>${mName}</td><td>${(cvF1*100).toFixed(1)}%</td><td>${(acc*100).toFixed(1)}%</td>`;
                            srTbody.appendChild(tr);
                        }
                    });

                    perfBenchmarkChart.data.labels = modelNames;
                    perfBenchmarkChart.data.datasets[0].data = cvF1Scores;
                    perfBenchmarkChart.data.datasets[1].data = accuracyScores;
                    perfBenchmarkChart.update();
                }

                // 2. Extra Grid Search / Calibration Metadata
                if (data.winning_tfidf_params) {
                    const tfParams = data.winning_tfidf_params;
                    safeSetText('perf-tfidf-params', `max_features=${tfParams.max_features || 20000}, ngrams=${JSON.stringify(tfParams.ngram_range || [1,2])}`);
                }
                if (data.winning_lr_params) {
                    safeSetText('perf-lr-params', `C = ${data.winning_lr_params.C || 1.0} (lbfgs solver)`);
                }
                if (data.calibration_eval) {
                    const cData = data.calibration_eval;
                    safeSetText('perf-calib-type', `${cData.winning_calibration || 'sigmoid'} (Brier: ${cData.sigmoid_brier || 0.4581} vs ${cData.uncalibrated_brier || 0.4688})`);
                }

                // 3. Primary Metrics & Confusion Matrix Update
                let metrics = data.metrics;
                if (!metrics && data.models && data.models[bestName]) {
                    metrics = data.models[bestName];
                }

                if (metrics) {
                    const acc = metrics.accuracy !== undefined ? metrics.accuracy : 0.6592;
                    const f1 = metrics.macro_f1 !== undefined ? metrics.macro_f1 : 0.6528;
                    const prec = metrics.precision_macro !== undefined ? metrics.precision_macro : 0.6454;
                    const rec = metrics.recall_macro !== undefined ? metrics.recall_macro : 0.6657;
                    const brier = metrics.brier_score !== undefined ? metrics.brier_score : 0.4499;

                    safeSetText('dash-accuracy', `${(acc * 100).toFixed(1)}%`);
                    safeSetText('perf-accuracy', `${(acc * 100).toFixed(1)}%`);
                    safeSetText('perf-macro-f1', f1.toFixed(3));
                    safeSetText('perf-brier', brier.toFixed(4));

                    safeSetText('perf-prec', `${(prec * 100).toFixed(1)}%`);
                    safeSetText('perf-rec', `${(rec * 100).toFixed(1)}%`);
                    safeSetText('perf-f1', `${(f1 * 100).toFixed(1)}%`);

                    safeSetWidth('perf-bar-prec', `${(prec * 100).toFixed(1)}%`);
                    safeSetWidth('perf-bar-rec', `${(rec * 100).toFixed(1)}%`);
                    safeSetWidth('perf-bar-f1', `${(f1 * 100).toFixed(1)}%`);

                    safeSetText('report-f1', f1.toFixed(3));
                    safeSetText('report-acc', acc.toFixed(3));

                    if (metrics.confusion_matrix) {
                        const cm = metrics.confusion_matrix;
                        safeSetText('cm-00', cm[0][0]);
                        safeSetText('cm-01', cm[0][1]);
                        safeSetText('cm-02', cm[0][2]);

                        safeSetText('cm-10', cm[1][0]);
                        safeSetText('cm-11', cm[1][1]);
                        safeSetText('cm-12', cm[1][2]);

                        safeSetText('cm-20', cm[2][0]);
                        safeSetText('cm-21', cm[2][1]);
                        safeSetText('cm-22', cm[2][2]);
                    }

                    if (metrics.class_metrics) {
                        const cm = metrics.class_metrics;
                        if (cm.negative) {
                            safeSetText('class-neg-p', cm.negative.precision.toFixed(2));
                            safeSetText('class-neg-r', cm.negative.recall.toFixed(2));
                            safeSetText('class-neg-f1', cm.negative.f1.toFixed(2));
                        }
                        if (cm.neutral) {
                            safeSetText('class-neu-p', cm.neutral.precision.toFixed(2));
                            safeSetText('class-neu-r', cm.neutral.recall.toFixed(2));
                            safeSetText('class-neu-f1', cm.neutral.f1.toFixed(2));
                        }
                        if (cm.positive) {
                            safeSetText('class-pos-p', cm.positive.precision.toFixed(2));
                            safeSetText('class-pos-r', cm.positive.recall.toFixed(2));
                            safeSetText('class-pos-f1', cm.positive.f1.toFixed(2));
                        }
                    }
                }
            } catch (err) {
                console.warn("Could not fetch metrics:", err);
                if (benchError) benchError.classList.remove('hidden');
            } finally {
                if (benchSkeleton) benchSkeleton.classList.add('hidden');
            }
        }

        // -------------------------------------------------------------
        // Live Ingestion & Telemetry Integration
        // -------------------------------------------------------------
        async function loadLiveStats() {
            // If user has uploaded a batch dataset, keep displaying its telemetry on the dashboard
            if (window.currentActiveDataset) {
                updateDashboardChart(
                    window.currentActiveDataset.positive_pct,
                    window.currentActiveDataset.neutral_pct,
                    window.currentActiveDataset.negative_pct
                );
                return;
            }

            const chartSkeleton = document.getElementById('dash-chart-skeleton');
            const chartError = document.getElementById('dash-chart-error');
            if (chartSkeleton) chartSkeleton.classList.remove('hidden');
            if (chartError) chartError.classList.add('hidden');

            try {
                const res = await fetchWithTimeout('/live/stats?source=live', {}, 25000);
                if (!res.ok) throw new Error("Stats endpoint unavailable");
                const data = await res.json();

                const totalCnt = data.total_count || 0;
                const posPct = totalCnt > 0 ? (data.positive_pct || 0.0) : 0.0;
                const neuPct = totalCnt > 0 ? (data.neutral_pct || 0.0) : 0.0;
                const negPct = totalCnt > 0 ? (data.negative_pct || 0.0) : 0.0;

                safeSetText('live-total-cnt', totalCnt);
                safeSetText('live-pos-pct', `${posPct}%`);
                safeSetText('live-neu-pct', `${neuPct}%`);
                safeSetText('live-neg-pct', `${negPct}%`);
                
                const prunedCnt = data.rows_pruned_last_cycle !== undefined ? data.rows_pruned_last_cycle : 0;
                safeSetText('live-pruned-info', `Pruned: ${prunedCnt} rows`);

                updateDashboardChart(posPct, neuPct, negPct);
            } catch (err) {
                console.warn("Could not fetch live stats:", err);
                if (chartError) chartError.classList.remove('hidden');
            } finally {
                if (chartSkeleton) chartSkeleton.classList.add('hidden');
            }
        }

        // Feed pagination & diffing state
        let feedPage = 1;
        const feedLimit = 15;
        let lastRenderedFeedSignature = "";
        let isLiveFeedLoading = false;

        function changeFeedPage(delta) {
            const newPage = feedPage + delta;
            if (newPage < 1) return;
            loadLiveFeed(newPage);
        }

        async function loadLiveFeed(page) {
            if (typeof page === 'number') {
                feedPage = Math.max(1, page);
            }
            if (isLiveFeedLoading) return;
            isLiveFeedLoading = true;

            const tbody = document.getElementById('live-feed-table-body');
            const skeleton = document.getElementById('live-feed-skeleton');
            const prevBtn = document.getElementById('feed-prev-btn');
            const nextBtn = document.getElementById('feed-next-btn');
            const pageIndicator = document.getElementById('feed-page-indicator');

            if (skeleton && (!tbody || tbody.children.length === 0 || typeof page === 'number')) {
                skeleton.classList.remove('hidden');
            }

            try {
                const isBatchActive = !!window.currentActiveDataset;
                const titleTextEl = document.getElementById('telemetry-title-text');
                const badgeEl = document.getElementById('telemetry-source-badge');
                const iconEl = document.getElementById('telemetry-icon');

                if (isBatchActive) {
                    if (titleTextEl) titleTextEl.innerText = "Uploaded Batch Predictions Telemetry";
                    if (badgeEl) {
                        badgeEl.innerText = "BATCH UPLOAD ACTIVE";
                        badgeEl.className = "text-[10px] glass-inner px-3 py-1 rounded-full text-indigo-400 font-bold uppercase border border-indigo-500/20 ml-2";
                    }
                    if (iconEl) iconEl.innerText = "cloud_done";
                } else {
                    if (titleTextEl) titleTextEl.innerText = "Recent Live Feed Telemetry";
                    if (badgeEl) {
                        badgeEl.innerText = "LIVE STREAM ACTIVE";
                        badgeEl.className = "text-[10px] glass-inner px-3 py-1 rounded-full text-purple-400 font-bold uppercase border border-purple-500/20 ml-2";
                    }
                    if (iconEl) iconEl.innerText = "rss_feed";
                }

                const sourceParam = isBatchActive ? 'batch_upload' : 'live';
                const offset = (feedPage - 1) * feedLimit;
                const res = await fetchWithTimeout(`/live/feed?limit=${feedLimit}&offset=${offset}&source=${sourceParam}`, {}, 8000);
                if (!res.ok) return;
                const items = await res.json();

                if (pageIndicator) {
                    pageIndicator.innerText = `Page ${feedPage}`;
                }
                if (prevBtn) {
                    prevBtn.disabled = (feedPage <= 1);
                }
                if (nextBtn) {
                    nextBtn.disabled = (!items || items.length < feedLimit);
                }

                if (!items || items.length === 0) {
                    if (feedPage === 1) {
                        tbody.innerHTML = `<tr><td colspan="5" class="p-5 text-center text-on-surface-variant italic">No telemetry stored yet. Upload a CSV batch or click "Fetch Live Posts".</td></tr>`;
                        lastRenderedFeedSignature = "empty";
                    } else {
                        tbody.innerHTML = `<tr><td colspan="5" class="p-5 text-center text-on-surface-variant italic">No more items on this page.</td></tr>`;
                    }
                    return;
                }

                // Row diffing: compare signature of received items
                const signature = items.map(it => `${it.id || ''}:${it.sentiment}:${it.confidence}:${it.created_at || ''}`).join('|');
                if (signature === lastRenderedFeedSignature && tbody.children.length === items.length) {
                    // Data is identical — skip DOM destruction to eliminate thrashing
                    return;
                }
                lastRenderedFeedSignature = signature;

                tbody.innerHTML = '';
                const confBins = [0, 0, 0, 0]; // <50%, 50-70%, 70-90%, >90%

                items.forEach(item => {
                    const tr = document.createElement('tr');
                    
                    let badgeClass = "bg-amber-500/10 text-amber-400 border border-amber-500/30";
                    if (item.sentiment === 'positive') badgeClass = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
                    if (item.sentiment === 'negative') badgeClass = "bg-rose-500/10 text-rose-400 border border-rose-500/30";

                    const conf = item.confidence || 0;
                    if (conf < 0.5) confBins[0]++;
                    else if (conf < 0.7) confBins[1]++;
                    else if (conf < 0.9) confBins[2]++;
                    else confBins[3]++;

                    const timeStr = item.created_at ? new Date(item.created_at).toLocaleTimeString() : 'Just now';

                    tr.innerHTML = `
                        <td class="p-3.5"><span class="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">${item.source || 'batch'}</span></td>
                        <td class="p-3.5 font-body text-on-background max-w-md truncate" title="${item.text}">${item.text}</td>
                        <td class="p-3.5"><span class="px-2.5 py-0.5 rounded-full font-bold uppercase text-[10px] ${badgeClass}">${item.sentiment}</span></td>
                        <td class="p-3.5 font-mono font-bold text-indigo-400">${(conf * 100).toFixed(1)}%</td>
                        <td class="p-3.5 font-mono text-on-surface-variant text-[11px]">${timeStr}</td>
                    `;
                    tbody.appendChild(tr);
                });

                if (liveConfidenceBarChart) {
                    liveConfidenceBarChart.data.datasets[0].data = confBins;
                    liveConfidenceBarChart.update();
                }
            } catch (err) {
                console.warn("Could not fetch live feed:", err);
            } finally {
                isLiveFeedLoading = false;
                if (skeleton) skeleton.classList.add('hidden');
            }
        }

        async function triggerLiveFetch() {
            const input = document.getElementById('live-kw-input');
            const keyword = input.value.trim();
            if (!keyword) {
                showToast("Please enter a valid search keyword.", "warning");
                return;
            }

            const btn = document.getElementById('live-trigger-btn');
            btn.disabled = true;
            btn.innerHTML = `<span class="material-symbols-outlined text-sm animate-spin-fast">sync</span> Ingesting...`;

            try {
                window.currentActiveDataset = null; // Clear batch dataset lock so live ingested posts take precedence
                const res = await fetchWithTimeout(`/live/trigger?keyword=${encodeURIComponent(keyword)}`, {
                    method: 'POST'
                }, 20000);

                if (res.status === 429) {
                    const errMsg = await parseErrorDetail(res, 'Keyword is in cooldown');
                    showToast(errMsg, "warning");
                    return;
                }

                if (!res.ok) {
                    const errMsg = await parseErrorDetail(res, 'Live fetch failed');
                    throw new Error(errMsg);
                }

                const data = await res.json();
                showToast(`Live fetch for '${data.keyword}': Ingested ${data.ingested_count} posts.`, "check_circle");
                
                loadLiveStats();
                loadLiveFeed();
            } catch (err) {
                showToast(err.message, "error");
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<span class="material-symbols-outlined text-sm">sync</span> Fetch Live Posts`;
            }
        }

        // Initialize on load
        async function initApp() {
            initDashboardCharts();
            initBatchPaginationControls();
            const isOnline = await checkHealth();
            if (isOnline) {
                await Promise.allSettled([
                    loadMetrics(),
                    loadLiveStats(),
                    loadLiveFeed()
                ]);
            }
        }

        // -------------------------------------------------------------
        // Backend API Settings Modal Handlers
        // -------------------------------------------------------------
        function openApiConfigModal() {
            const modal = document.getElementById('api-config-modal');
            const input = document.getElementById('api-url-input');
            const results = document.getElementById('api-test-results');
            if (results) results.classList.add('hidden');
            if (input) input.value = getApiBase();
            if (modal) modal.classList.remove('hidden');
        }

        function closeApiConfigModal() {
            const modal = document.getElementById('api-config-modal');
            if (modal) modal.classList.add('hidden');
        }

        function setApiInput(val) {
            const input = document.getElementById('api-url-input');
            if (input) input.value = val;
        }

        async function testApiUrl() {
            const input = document.getElementById('api-url-input');
            const testUrl = (input.value || '').trim().replace(/\/+$/, '');
            const btn = document.getElementById('btn-test-api');
            const results = document.getElementById('api-test-results');
            const dot = document.getElementById('api-test-dot');
            const msg = document.getElementById('api-test-msg');
            const latEl = document.getElementById('api-test-latency');

            if (!testUrl) {
                showToast("Please enter a valid URL.", "warning");
                return;
            }

            if (btn) {
                btn.disabled = true;
                btn.innerHTML = `<span class="material-symbols-outlined text-xs animate-spin-fast">sync</span> Pinging...`;
            }

            const startTime = performance.now();
            try {
                const res = await fetch(`${testUrl}/health`, { method: 'GET' });
                const ping = Math.round(performance.now() - startTime);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                
                results.className = "p-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 text-[11px] flex items-center justify-between";
                dot.className = "w-2 h-2 rounded-full bg-emerald-400";
                msg.innerText = `Connected: ${data.model_name || 'API Ready'}`;
                latEl.innerText = `${ping}ms`;
                results.classList.remove('hidden');
            } catch (err) {
                const ping = Math.round(performance.now() - startTime);
                results.className = "p-3 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-300 text-[11px] flex items-center justify-between";
                dot.className = "w-2 h-2 rounded-full bg-rose-400";
                msg.innerText = `Connection Failed: ${err.message || 'Offline'}`;
                latEl.innerText = `${ping}ms`;
                results.classList.remove('hidden');
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = `<span class="material-symbols-outlined text-xs">network_check</span> Test Ping`;
                }
            }
        }

        function saveApiConfig() {
            const input = document.getElementById('api-url-input');
            const val = (input.value || '').trim();
            setApiBase(val);
            closeApiConfigModal();
            showToast(`Backend target set to: ${getApiBase()}`, "check_circle");
            initApp();
        }

        function resetApiConfig() {
            localStorage.removeItem('SENTIMENTSCOPE_API_URL');
            const input = document.getElementById('api-url-input');
            if (input) input.value = getApiBase();
            closeApiConfigModal();
            showToast("Reset to default API URL", "info");
            initApp();
        }

        // Back to Top button scroll monitor (debounced to eliminate scroll jank)
        window.addEventListener('scroll', debounce(() => {
            const btn = document.getElementById('back-to-top-btn');
            if (!btn) return;
            if (window.scrollY > 300) {
                btn.classList.remove('translate-y-16', 'opacity-0', 'pointer-events-none');
                btn.classList.add('translate-y-0', 'opacity-100');
            } else {
                btn.classList.add('translate-y-16', 'opacity-0', 'pointer-events-none');
                btn.classList.remove('translate-y-0', 'opacity-100');
            }
        }, 50), { passive: true });

        
        // -------------------------------------------------------------
        // Mobile Drawer & Keyboard Shortcuts Handlers
        // -------------------------------------------------------------
        function toggleMobileDrawer(open) {
            const drawer = document.getElementById('mobile-drawer');
            const backdrop = document.getElementById('mobile-drawer-backdrop');
            if (!drawer || !backdrop) return;
            if (open) {
                drawer.classList.remove('closed');
                drawer.classList.add('open');
                backdrop.classList.remove('closed');
                backdrop.classList.add('open');
                document.body.style.overflow = 'hidden';
            } else {
                drawer.classList.add('closed');
                drawer.classList.remove('open');
                backdrop.classList.add('closed');
                backdrop.classList.remove('open');
                document.body.style.overflow = '';
            }
        }

        // Global Keyboard Shortcuts (Ctrl+Enter, /, Escape)
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const drawer = document.getElementById('mobile-drawer');
                if (drawer && drawer.classList.contains('open')) {
                    toggleMobileDrawer(false);
                    return;
                }
                const apiModal = document.getElementById('api-config-modal');
                if (apiModal && !apiModal.classList.contains('hidden')) {
                    closeApiConfigModal();
                    return;
                }
                const chartModal = document.getElementById('chart-modal');
                if (chartModal && chartModal.classList.contains('opacity-100')) {
                    closeChartModal();
                    return;
                }
            }

            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const singleView = document.getElementById('view-single');
                const isSingleActive = singleView && !singleView.classList.contains('hidden');
                const singleInput = document.getElementById('single-input');
                if (isSingleActive || (singleInput && document.activeElement === singleInput)) {
                    e.preventDefault();
                    runSinglePredict();
                    return;
                }
            }

            if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
                e.preventDefault();
                switchTab('single');
                const singleInput = document.getElementById('single-input');
                if (singleInput) {
                    setTimeout(() => {
                        singleInput.focus();
                        singleInput.select();
                    }, 50);
                }
            }
        });

        window.addEventListener('DOMContentLoaded', initApp);
