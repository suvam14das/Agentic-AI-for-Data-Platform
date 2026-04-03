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


function addMessage(sender, text) {
    const div = document.createElement('div');
    div.classList.add('message', sender.toLowerCase());
    
    let content = text;
    let chartConfig = null;
    
    if (sender === 'ComputeGuru') {
        const chartMatch = text.match(/<chart-data>(.*?)<\/chart-data>/s);
        if (chartMatch) {
            try {
                chartConfig = JSON.parse(chartMatch[1]);
                content = text.replace(chartMatch[0], ''); // Remove chart data from text
            } catch (e) {
                console.error('Error parsing chart data:', e);
            }
            if (content.trim()) {
                content = marked.parse(content);
            } else {
                content = '';
            }
        } else if (text.includes("<table") || text.includes("<style") || text.includes("<img")) {
            content = text;
        } else {
            content = marked.parse(text);
        }
    }
    
    div.innerHTML = `<strong>${sender}:</strong> ${content}`;
    messagesDiv.appendChild(div);
    
    if (chartConfig) {
        const canvasWrapper = document.createElement('div');
        canvasWrapper.style.width = '100%';
        canvasWrapper.style.height = '400px'; // Adjustable height
        const canvas = document.createElement('canvas');
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
