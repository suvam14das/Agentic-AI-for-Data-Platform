let sessionId = null;
const messagesDiv = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');

async function newSession() {
    console.log('Starting newSession');
    const response = await fetch('http://localhost:8000/new_session', { method: 'POST' });
    const data = await response.json();
    sessionId = data.session_id;
    console.log('New sessionId:', sessionId);
    updateUrlWithSession();
    console.log('Called updateUrlWithSession');
    messagesDiv.innerHTML = '';
    addMessage('ComputeGuru', 'Welcome to Delta AI Chat! How can I help you today?');
}

async function sendMessage() {
    if (!sessionId) {
        alert('Please start a new session first.');
        return;
    }
    const message = messageInput.value.trim();
    if (!message) return;
    addMessage('User', message);
    messageInput.value = '';

    showLoading();

    const response = await fetch(`http://localhost:8000/chat/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    const data = await response.json();
    removeLoading();
    addMessage('ComputeGuru', data.response);
}


function extractTaggedPayload(text, tagName) {
    const re = new RegExp(`<${tagName}>([\\s\\S]*?)<\\/${tagName}>`, 'm');
    const match = text.match(re);
    if (!match) return { payload: null, stripped: text };
    return {
        payload: match[1],
        stripped: text.replace(match[0], "").trim(),
    };
}

function csvParse(text) {
    // Minimal RFC4180-ish CSV parser: handles quotes, commas, newlines.
    const rows = [];
    let row = [];
    let cur = "";
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        const next = text[i + 1];

        if (inQuotes) {
            if (ch === '"' && next === '"') {
                cur += '"';
                i++;
            } else if (ch === '"') {
                inQuotes = false;
            } else {
                cur += ch;
            }
        } else {
            if (ch === '"') {
                inQuotes = true;
            } else if (ch === ",") {
                row.push(cur);
                cur = "";
            } else if (ch === "\n") {
                row.push(cur);
                cur = "";
                // ignore empty trailing row caused by final newline
                if (!(row.length === 1 && row[0] === "")) rows.push(row);
                row = [];
            } else if (ch === "\r") {
                // ignore
            } else {
                cur += ch;
            }
        }
    }
    // last cell
    row.push(cur);
    if (!(row.length === 1 && row[0] === "")) rows.push(row);
    return rows;
}

function renderTable(container, headers, dataRows) {
    container.innerHTML = "";

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");

    let sortState = { col: null, dir: 1 };

    const tbody = document.createElement("tbody");

    function paintBody() {
        tbody.innerHTML = "";
        dataRows.forEach((r) => {
            const tr = document.createElement("tr");
            headers.forEach((_, idx) => {
                const td = document.createElement("td");
                td.textContent = r[idx] ?? "";
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    headers.forEach((h, idx) => {
        const th = document.createElement("th");
        th.textContent = h;

        th.onclick = () => {
            const dir = sortState.col === idx ? -sortState.dir : 1;
            sortState = { col: idx, dir };

            dataRows.sort((a, b) => {
                const av = (a[idx] ?? "").toString();
                const bv = (b[idx] ?? "").toString();
                // numeric sort if both look numeric
                const an = Number(av);
                const bn = Number(bv);
                const aIsNum = av !== "" && !Number.isNaN(an);
                const bIsNum = bv !== "" && !Number.isNaN(bn);

                if (aIsNum && bIsNum) return (an - bn) * dir;
                return av.localeCompare(bv) * dir;
            });
            paintBody();
        };

        trh.appendChild(th);
    });

    thead.appendChild(trh);
    table.appendChild(thead);
    table.appendChild(tbody);

    paintBody();
    container.appendChild(table);
}

async function openArtifactPanel(artifact) {
    const overlay = document.getElementById("artifact-overlay");
    const title = document.getElementById("artifact-title");
    const meta = document.getElementById("artifact-meta");
    const tableContainer = document.getElementById("artifact-table-container");
    const download = document.getElementById("artifact-download");

    const filename = artifact.csv_filename || "artifact.csv";
    title.textContent = filename;

    const downloadUrl = artifact.download_url || (artifact.csv_filename ? `/artifacts/${artifact.csv_filename}` : "#");
    download.href = downloadUrl;

    meta.textContent = `Rows: ${artifact.row_count ?? "?"} | Columns: ${(artifact.columns || []).length} | Path: ${artifact.csv_path || ""}`;

    tableContainer.innerHTML = "Loading CSV...";
    overlay.classList.remove("hidden");

    try {
        const res = await fetch(downloadUrl);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const csvText = await res.text();

        const parsed = csvParse(csvText);
        const headers = parsed[0] || [];
        const dataRows = parsed.slice(1);

        renderTable(tableContainer, headers, dataRows);
    } catch (e) {
        console.error("Failed to load/render CSV:", e);
        tableContainer.innerHTML = `Failed to load CSV (${e}).`;
    }
}

function closeArtifactPanel(evt) {
    // called by overlay click / close button. If event exists, ignore clicks inside panel.
    const overlay = document.getElementById("artifact-overlay");
    overlay.classList.add("hidden");
}

function addMessage(sender, text) {
    const div = document.createElement("div");
    div.classList.add("message", sender.toLowerCase());

    let content = text;
    let chartConfig = null;
    let artifact = null;

    if (sender === "ComputeGuru") {
        // Extract artifact JSON (table/csv) if present
        const artifactExtract = extractTaggedPayload(text, "artifact");
        if (artifactExtract.payload) {
            try {
                artifact = JSON.parse(artifactExtract.payload);
                content = artifactExtract.stripped || "Query executed. Results available as CSV artifact.";
            } catch (e) {
                console.error("Error parsing artifact JSON:", e);
                content = text;
            }
        }

        // Extract chart config if present
        const chartExtract = extractTaggedPayload(content, "chart-data");
        if (chartExtract.payload) {
            try {
                chartConfig = JSON.parse(chartExtract.payload);
                content = chartExtract.stripped;
            } catch (e) {
                console.error("Error parsing chart data:", e);
            }
        }

        if (content.includes("<table") || content.includes("<style") || content.includes("<img")) {
            // legacy HTML rendering (kept for backward compatibility)
        } else {
            content = marked.parse(content);
        }
    }

    div.innerHTML = `<strong>${sender}:</strong> ${content}`;
    messagesDiv.appendChild(div);

    // Render artifact actions inline (Open / Download)
    if (artifact && artifact.status === "ok" && artifact.type === "table" && artifact.format === "csv") {
        const actions = document.createElement("div");
        actions.style.marginTop = "8px";

        const openBtn = document.createElement("button");
        openBtn.textContent = "Open table";
        openBtn.onclick = () => openArtifactPanel({
            ...artifact,
            download_url: artifact.download_url || `/artifacts/${artifact.csv_filename}`
        });

        const downloadLink = document.createElement("a");
        downloadLink.textContent = "Download CSV";
        downloadLink.href = artifact.download_url || `/artifacts/${artifact.csv_filename}`;
        downloadLink.target = "_blank";
        downloadLink.rel = "noopener";
        downloadLink.style.marginLeft = "10px";

        actions.appendChild(openBtn);
        actions.appendChild(downloadLink);
        div.appendChild(actions);
    }

    if (chartConfig) {
        const canvasWrapper = document.createElement("div");
        canvasWrapper.style.width = "100%";
        canvasWrapper.style.height = "400px";
        const canvas = document.createElement("canvas");
        canvasWrapper.appendChild(canvas);
        div.appendChild(canvasWrapper);

        new Chart(canvas, chartConfig);
    }

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading';
    loadingDiv.classList.add('message', 'ai');
    loadingDiv.innerHTML = `
        <strong>ComputeGuru:</strong>
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    messagesDiv.appendChild(loadingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Function to get query parameter
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

// Function to update URL with session ID without reloading
function updateUrlWithSession() {
    const newUrl = `${window.location.pathname}?session=${sessionId}`;
    console.log('Updating URL to:', newUrl);
    history.pushState({ sessionId }, '', newUrl);
    console.log('pushState called');
}

// Function to fetch and render history
async function fetchHistory() {
    try {
        const response = await fetch(`http://localhost:8000/history/${sessionId}`);
        if (!response.ok) {
            throw new Error('Session not found');
        }
        const data = await response.json();
        messagesDiv.innerHTML = '';
        data.history.forEach(msg => {
            addMessage('User', msg.user);
            addMessage('ComputeGuru', msg.ai);
        });
    } catch (error) {
        console.error('Error fetching history:', error);
        // If session not found, start new session
        await newSession();
    }
}

function removeLoading() {
    const loadingDiv = document.getElementById('loading');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Initialize on page load
async function init() {
    console.log('Starting init');
    const urlSessionId = getQueryParam('session');
    console.log('urlSessionId:', urlSessionId);
    if (urlSessionId) {
        sessionId = urlSessionId;
        await fetchHistory();
    } else {
        await newSession();
    }
    console.log('Init completed, current URL:', window.location.href);
}

init();
