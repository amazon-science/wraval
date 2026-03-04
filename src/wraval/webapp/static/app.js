/* WRAVAL Webapp — frontend logic */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
});

/* ── Tab switching ── */

function initTabs() {
    const tabs = document.querySelectorAll("[role='tab']");
    tabs.forEach((tab) => {
        tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    });
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll("[role='tab']").forEach((btn) => {
        const isActive = btn.dataset.tab === tabName;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", isActive);
    });

    // Update panels
    document.querySelectorAll("[role='tabpanel']").forEach((panel) => {
        const isActive = panel.id === `panel-${tabName}`;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
    });

    // Load tab content
    const loaders = {
        prompts: loadPromptsTab,
        inference: loadInferenceTab,
        judge: loadJudgeTab,
        data: loadDataTab,
    };
    if (loaders[tabName]) {
        loaders[tabName]();
    }
}

/* ── Tab content loaders (placeholders for task 8) ── */

function loadPromptsTab() {
    // Will be implemented in task 8.1
}

function loadInferenceTab() {
    // Will be implemented in task 8.2
}

function loadJudgeTab() {
    // Will be implemented in task 8.3
}

function loadDataTab() {
    // Will be implemented in task 8.4
}
