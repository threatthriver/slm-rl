/**
 * Cognition Studio Frontend Controller
 * Handles interactive tabs, real-time inference, CSAO step attribution,
 * Prover-Skeptic duel simulation, curriculum inspection, dynamic SVG telemetry,
 * and verifiable reward simulation.
 */

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initHealthCheck();
    initReasoningPlayground();
    initCSAOInspector();
    initDuelArena();
    initCurriculumPyramid();
    initTelemetryAndReward();
});

// ============================================================================
// 1. Navigation & Routing
// ============================================================================

const TAB_METADATA = {
    playground: {
        title: "Deliberative Reasoning Studio",
        subtitle: "Test-time reasoning inspection, step attribution, and self-correction verification",
    },
    csao: {
        title: "CSAO Step-Level Pivot Inspector",
        subtitle: "Counterfactual credit assignment isolating calculation slips from valid deductions",
    },
    duel: {
        title: "Prover-Skeptic Autonomous Self-Play Arena",
        subtitle: "Zero-teacher adversarial verification game matrix with dual SLM roles",
    },
    curriculum: {
        title: "Dynamic 5-Tier Reasoning Curriculum",
        subtitle: "Auto-adaptive difficulty progression based on rolling pass rate thresholds",
    },
    telemetry: {
        title: "Telemetry & Verifiable Reward Studio",
        subtitle: "Live Apple MPS training curves and composite verifiable reward simulation",
    },
    compare: {
        title: "Architectural Comparator: OpenAI vs. Frontier SLM-RL",
        subtitle: "Direct empirical benchmark across algorithms, hardware, and parameter memory",
    },
};

function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");

            navItems.forEach(n => n.classList.remove("active"));
            tabContents.forEach(t => t.classList.remove("active"));

            item.classList.add("active");
            const activeContent = document.getElementById(`tab-${targetTab}`);
            if (activeContent) activeContent.classList.add("active");

            if (TAB_METADATA[targetTab]) {
                pageTitle.textContent = TAB_METADATA[targetTab].title;
                pageSubtitle.textContent = TAB_METADATA[targetTab].subtitle;
            }

            // Tab-specific refreshes
            if (targetTab === "telemetry") renderTelemetryChart();
            if (targetTab === "curriculum") refreshCurriculum();
        });
    });
}

// ============================================================================
// 2. Health & Hardware Status
// ============================================================================

async function initHealthCheck() {
    try {
        const res = await fetch("/api/health");
        if (!res.ok) return;
        const data = await res.json();

        const statusEl = document.getElementById("backend-status");
        if (statusEl) {
            if (data.device === "mps") {
                statusEl.textContent = `Apple MPS Engine Active (${data.memory_allocated_mb} MB)`;
            } else if (data.device === "cuda") {
                statusEl.textContent = `CUDA Engine Active (${data.memory_allocated_mb} MB)`;
            } else {
                statusEl.textContent = `CPU Engine Active (Mock Fallback)`;
            }
        }

        const formatEl = document.getElementById("header-format");
        if (formatEl) formatEl.textContent = `${(data.format_compliance * 100).toFixed(1)}%`;

        const accEl = document.getElementById("header-acc");
        if (accEl) accEl.textContent = `${(data.gsm8k_pass_rate * 100).toFixed(1)}%`;

        const memEl = document.getElementById("header-mem");
        if (memEl) memEl.textContent = `${(data.lora_parameters / 1000).toFixed(0)}k LoRA`;
    } catch (e) {
        console.warn("Backend health check warning:", e);
    }
}

// ============================================================================
// 3. Reasoning Playground
// ============================================================================

function initReasoningPlayground() {
    const promptInput = document.getElementById("prompt-input");
    const tempSlider = document.getElementById("temp-slider");
    const tempVal = document.getElementById("temp-val");
    const tokensSlider = document.getElementById("tokens-slider");
    const tokensVal = document.getElementById("tokens-val");
    const btnGenerate = document.getElementById("btn-generate");
    const viewer = document.getElementById("reasoning-viewer");
    const latencyTag = document.getElementById("latency-tag");

    // Slider sync
    if (tempSlider && tempVal) {
        tempSlider.addEventListener("input", () => tempVal.textContent = tempSlider.value);
    }
    if (tokensSlider && tokensVal) {
        tokensSlider.addEventListener("input", () => tokensVal.textContent = tokensSlider.value);
    }

    // Benchmark preset pills
    const pills = document.querySelectorAll(".preset-pills .pill");
    pills.forEach(pill => {
        pill.addEventListener("click", () => {
            const prob = pill.getAttribute("data-problem");
            if (promptInput) promptInput.value = prob;
        });
    });

    // Run inference
    if (btnGenerate) {
        btnGenerate.addEventListener("click", async () => {
            const text = promptInput ? promptInput.value.trim() : "";
            if (!text) {
                alert("Please enter a question or select a benchmark preset.");
                return;
            }

            btnGenerate.disabled = true;
            btnGenerate.innerHTML = `<span class="spinner"></span> Thinking...`;
            if (latencyTag) latencyTag.textContent = "Deliberating on MPS...";

            viewer.innerHTML = `
                <div class="loading-state">
                    <div class="deliberation-pulse"></div>
                    <p>Generating step-by-step reasoning tokens...</p>
                    <span class="subtext">Phase-Gated Deliberation Loop Active</span>
                </div>
            `;

            try {
                const res = await fetch("/api/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        prompt: text,
                        temperature: parseFloat(tempSlider ? tempSlider.value : 0.2),
                        max_tokens: parseInt(tokensSlider ? tokensSlider.value : 128),
                    }),
                });

                const data = await res.json();
                renderDeliberationOutput(data, viewer, latencyTag);
            } catch (err) {
                viewer.innerHTML = `<div class="error-box">Inference error: ${err.message}</div>`;
                if (latencyTag) latencyTag.textContent = "Error";
            } finally {
                btnGenerate.disabled = false;
                btnGenerate.innerHTML = `Run Deliberation`;
            }
        });
    }
}

function renderDeliberationOutput(data, container, latencyTag) {
    if (latencyTag) {
        latencyTag.textContent = `${data.latency_ms} ms (${data.tokens_per_sec} tok/s)`;
    }

    const ahaTag = data.has_aha
        ? `<span class="badge purple">Aha! Self-Correction</span>`
        : ``;

    const formatTag = data.has_format
        ? `<span class="badge emerald">Format: 100% Compliant</span>`
        : `<span class="badge amber">Partial Format</span>`;

    container.innerHTML = `
        <div class="output-wrapper">
            <div class="output-header-bar">
                <div class="badges-row">
                    ${formatTag}
                    ${ahaTag}
                    <span class="badge blue">${data.device.toUpperCase()}</span>
                </div>
                <div class="stats-strip">
                    <span><b>${data.think_word_count}</b> deliberation words</span>
                    <span>•</span>
                    <span><b>${data.word_count}</b> total words</span>
                </div>
            </div>

            <!-- Deliberation Think Box -->
            <div class="trace-box think-box">
                <div class="trace-box-title">
                    <span>Deliberation Process (&lt;think&gt;)</span>
                </div>
                <div class="trace-content">${escapeHtml(data.think || "(Direct response)")}</div>
            </div>

            <!-- Verified Answer Box -->
            <div class="trace-box answer-box">
                <div class="trace-box-title">
                    <span>Verified Solution (&lt;answer&gt;)</span>
                </div>
                <div class="answer-content-large">${escapeHtml(data.answer || "N/A")}</div>
            </div>
        </div>
    `;
}

// ============================================================================
// 4. CSAO Pivot Inspector
// ============================================================================

function initCSAOInspector() {
    const csaoText = document.getElementById("csao-input-text");
    const csaoGamma = document.getElementById("csao-gamma");
    const csaoGammaVal = document.getElementById("csao-gamma-val");
    const btnAnalyze = document.getElementById("btn-analyze-csao");
    const resultsContainer = document.getElementById("csao-step-results");

    if (csaoGamma && csaoGammaVal) {
        csaoGamma.addEventListener("input", () => csaoGammaVal.textContent = csaoGamma.value);
    }

    if (btnAnalyze) {
        btnAnalyze.addEventListener("click", async () => {
            const chain = csaoText ? csaoText.value.trim() : "";
            if (!chain) return;

            btnAnalyze.disabled = true;
            btnAnalyze.textContent = "Analyzing Deduction...";

            try {
                const res = await fetch("/api/analyze-steps", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        chain: chain,
                        gamma: parseFloat(csaoGamma ? csaoGamma.value : 0.9),
                    }),
                });

                const data = await res.json();
                renderCSAOResults(data, resultsContainer);
            } catch (err) {
                resultsContainer.innerHTML = `<div class="error-box">Analysis error: ${err.message}</div>`;
            } finally {
                btnAnalyze.disabled = false;
                btnAnalyze.textContent = "Analyze Step Credit";
            }
        });

        // Trigger on initial page load
        btnAnalyze.click();
    }
}

function renderCSAOResults(data, container) {
    if (!container) return;

    let html = `
        <div class="csao-summary-card ${data.pivot_localized ? 'pivot-found' : 'all-sound'}">
            <div class="summary-icon">${data.pivot_localized ? '[PIVOT]' : '[VALID]'}</div>
            <div class="summary-text">${data.summary}</div>
        </div>
    `;

    data.steps.forEach(step => {
        let cardClass = "step-card";
        let badge = "";

        if (step.status === "valid") {
            cardClass += " step-valid";
            badge = `<span class="badge emerald">+${step.advantage.toFixed(2)} Advantage (Protected)</span>`;
        } else if (step.status === "pivot_error") {
            cardClass += " step-pivot";
            badge = `<span class="badge rose">${step.advantage.toFixed(2)} Fatal Pivot Error</span>`;
        } else {
            cardClass += " step-downstream";
            badge = `<span class="badge amber">${step.advantage.toFixed(2)} Downstream Fallout</span>`;
        }

        html += `
            <div class="${cardClass}">
                <div class="step-card-header">
                    <span class="step-badge-num">Step ${step.step_number}</span>
                    ${badge}
                </div>
                <div class="step-card-body">${escapeHtml(step.text)}</div>
                <div class="step-card-footer">${escapeHtml(step.explanation)}</div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// ============================================================================
// 5. Prover-Skeptic Duel Arena
// ============================================================================

let duelRound = 1;

function initDuelArena() {
    const btnDuel = document.getElementById("btn-trigger-duel");
    const proverBubble = document.getElementById("prover-bubble");
    const skepticBubble = document.getElementById("skeptic-bubble");
    const proverPayoff = document.getElementById("prover-payoff");
    const skepticPayoff = document.getElementById("skeptic-payoff");
    const outcomeTag = document.getElementById("duel-outcome-tag");

    if (btnDuel) {
        btnDuel.addEventListener("click", async () => {
            btnDuel.disabled = true;
            btnDuel.textContent = "Duel in Progress...";
            if (outcomeTag) outcomeTag.textContent = "Prover formulating proof...";

            try {
                const res = await fetch("/api/duel", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ round_idx: duelRound }),
                });

                const data = await res.json();
                duelRound++;

                // Animate Prover bubble
                if (proverBubble) proverBubble.innerHTML = `<pre>${escapeHtml(data.prover_output)}</pre>`;
                if (outcomeTag) outcomeTag.textContent = "Skeptic auditing logic...";

                setTimeout(() => {
                    // Animate Skeptic bubble
                    if (skepticBubble) skepticBubble.innerHTML = `<pre>${escapeHtml(data.skeptic_output)}</pre>`;

                    // Update Payoffs
                    if (proverPayoff) {
                        proverPayoff.textContent = `${data.prover_payoff > 0 ? '+' : ''}${data.prover_payoff.toFixed(1)}`;
                        proverPayoff.style.color = data.prover_payoff > 0 ? '#34d399' : '#f43f5e';
                    }
                    if (skepticPayoff) {
                        skepticPayoff.textContent = `${data.skeptic_payoff > 0 ? '+' : ''}${data.skeptic_payoff.toFixed(1)}`;
                        skepticPayoff.style.color = data.skeptic_payoff > 0 ? '#34d399' : '#f43f5e';
                    }

                    if (outcomeTag) {
                        outcomeTag.textContent = `Verdict: ${data.winner}`;
                        outcomeTag.style.color = data.winner.includes("Skeptic") ? "#fbbf24" : "#34d399";
                    }

                    btnDuel.disabled = false;
                    btnDuel.textContent = "Start Next Round";
                }, 600);

            } catch (err) {
                if (outcomeTag) outcomeTag.textContent = "Duel Error";
                btnDuel.disabled = false;
                btnDuel.textContent = "Retry Duel";
            }
        });
    }
}

// ============================================================================
// 6. Curriculum Mastery Pyramid
// ============================================================================

async function refreshCurriculum() {
    try {
        const res = await fetch("/api/curriculum");
        if (!res.ok) return;
        const data = await res.json();

        // Wire interactive click for each tier
        const tiers = document.querySelectorAll(".tier-level");
        tiers.forEach(tierEl => {
            const tierNum = parseInt(tierEl.getAttribute("data-tier"));
            tierEl.style.cursor = "pointer";
            tierEl.title = "Click to run SLM verification test on this tier challenge!";

            tierEl.onclick = async () => {
                tiers.forEach(t => t.classList.remove("active-tier"));
                tierEl.classList.add("active-tier");
                await runTierChallenge(tierNum);
            };
        });
    } catch (e) {
        console.warn("Curriculum fetch error:", e);
    }
}

async function runTierChallenge(tierNum) {
    const activeTier = document.querySelector(`.tier-level[data-tier="${tierNum}"]`);
    if (!activeTier) return;

    const originalText = activeTier.querySelector(".tier-info").innerHTML;
    activeTier.querySelector(".tier-info").innerHTML = `
        <h4>Running Tier ${tierNum} Challenge...</h4>
        <span>Generating SLM Deliberation trace...</span>
    `;

    try {
        const res = await fetch("/api/test-tier", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tier: tierNum }),
        });
        const data = await res.json();

        const statusPill = activeTier.querySelector(".tier-status");
        if (data.is_correct) {
            statusPill.className = "tier-status mastered";
            statusPill.textContent = "Verified";
        } else {
            statusPill.className = "tier-status in-progress";
            statusPill.textContent = "Target Slip";
        }

        activeTier.querySelector(".tier-info").innerHTML = `
            <h4>Tier ${tierNum}: ${escapeHtml(data.problem.substring(0, 45))}...</h4>
            <span>Answer: <b>${escapeHtml(data.output.answer || "N/A")}</b> (Target: ${data.expected_target}) • ${data.is_correct ? 'Correct' : 'Needs tuning'}</span>
        `;
    } catch (e) {
        activeTier.querySelector(".tier-info").innerHTML = originalText;
    }
}

function initCurriculumPyramid() {
    refreshCurriculum();
}

// ============================================================================
// 7. Telemetry & Reward Studio
// ============================================================================

function initTelemetryAndReward() {
    // Reward Simulator sliders
    const slMath = document.getElementById("sl-math");
    const slFmt = document.getElementById("sl-fmt");
    const slAha = document.getElementById("sl-aha");
    const slComp = document.getElementById("sl-comp");

    const rMathW = document.getElementById("r-math-w");
    const rFmtW = document.getElementById("r-fmt-w");
    const rAhaW = document.getElementById("r-aha-w");
    const rCompW = document.getElementById("r-comp-w");

    const simScoreDisplay = document.getElementById("sim-score-display");

    function updateSimScore() {
        if (!slMath || !slFmt || !slAha || !slComp) return;
        const wMath = parseFloat(slMath.value);
        const wFmt = parseFloat(slFmt.value);
        const wAha = parseFloat(slAha.value);
        const wComp = parseFloat(slComp.value);

        if (rMathW) rMathW.textContent = wMath.toFixed(2);
        if (rFmtW) rFmtW.textContent = wFmt.toFixed(2);
        if (rAhaW) rAhaW.textContent = wAha.toFixed(2);
        if (rCompW) rCompW.textContent = wComp.toFixed(2);

        // Assume standard high-quality deliberation: correct=true, formatted=true, aha=true, optimal tokens
        const score = (wMath * 1.0) + (wFmt * 1.0) + (wAha * 1.0) + (wComp * 1.0);
        if (simScoreDisplay) {
            simScoreDisplay.textContent = `+${score.toFixed(3)}`;
            simScoreDisplay.style.color = score > 0.7 ? '#10b981' : (score > 0.4 ? '#f59e0b' : '#f43f5e');
        }
    }

    [slMath, slFmt, slAha, slComp].forEach(slider => {
        if (slider) slider.addEventListener("input", updateSimScore);
    });

    updateSimScore();
}

async function renderTelemetryChart() {
    const box = document.getElementById("svg-chart-box");
    if (!box) return;

    try {
        const res = await fetch("/api/telemetry");
        if (!res.ok) return;
        const data = await res.json();
        const history = data.history || [];

        if (history.length === 0) return;

        const width = 540;
        const height = 240;
        const pad = 35;

        const maxReward = 1.0;
        const minReward = 0.0;
        const steps = history.length;

        // Construct SVG polyline for rewards
        const rewardPoints = history.map((pt, idx) => {
            const x = pad + (idx / Math.max(1, steps - 1)) * (width - 2 * pad);
            const y = height - pad - (pt.mean_reward / maxReward) * (height - 2 * pad);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(" ");

        // Construct SVG polyline for entropy (scaled down)
        const entropyPoints = history.map((pt, idx) => {
            const x = pad + (idx / Math.max(1, steps - 1)) * (width - 2 * pad);
            const normEnt = (pt.entropy - 6.4) / 0.3; // normalize around 6.5
            const clamped = Math.max(0, Math.min(1, normEnt));
            const y = height - pad - clamped * (height - 2 * pad);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(" ");

        box.innerHTML = `
            <svg width="100%" height="100%" viewBox="0 0 ${width} ${height}" class="telemetry-svg">
                <defs>
                    <linearGradient id="rewardGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="#10b981" stop-opacity="0.4"/>
                        <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>

                <!-- Grid lines -->
                <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
                <line x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
                <line x1="${pad}" y1="${pad}" x2="${width - pad}" y2="${pad}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>

                <!-- Y Axis Labels -->
                <text x="${pad - 8}" y="${pad + 4}" fill="#64748b" font-size="10" text-anchor="end">1.0 R</text>
                <text x="${pad - 8}" y="${height / 2 + 4}" fill="#64748b" font-size="10" text-anchor="end">0.5 R</text>
                <text x="${pad - 8}" y="${height - pad + 4}" fill="#64748b" font-size="10" text-anchor="end">0.0 R</text>

                <!-- Reward Line -->
                <polyline fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${rewardPoints}" />

                <!-- Entropy Line -->
                <polyline fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="3" stroke-linecap="round" points="${entropyPoints}" />

                <!-- Legends -->
                <circle cx="${width - 160}" cy="18" r="4" fill="#10b981"/>
                <text x="${width - 150}" y="21" fill="#e2e8f0" font-size="11">Mean Reward (Pass@1)</text>

                <circle cx="${width - 160}" cy="34" r="4" fill="#3b82f6"/>
                <text x="${width - 150}" y="37" fill="#94a3b8" font-size="11">Deliberation Entropy</text>
            </svg>
        `;
    } catch (e) {
        console.warn("Chart render error:", e);
    }
}

// Helper utilities
function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
