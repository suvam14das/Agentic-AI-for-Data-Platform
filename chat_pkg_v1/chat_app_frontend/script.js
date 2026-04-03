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

async function runSQL() {
    if (!sessionId) {
        alert('Please start a new session first.');
        return;
    }

    showLoading();

    const response = await fetch(`http://localhost:8000/chat/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: "run sql" })
    });
    const data = await response.json();
    
    removeLoading();
    
    if (data.response === "No SQL query available.") {
        showError('No current SQL query found.');
    } else {
        addMessage('ComputeGuru', data.response);
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.textContent = message;
    errorDiv.style.position = 'fixed';
    errorDiv.style.top = '20px';
    errorDiv.style.left = '50%';
    errorDiv.style.transform = 'translateX(-50%)';
    errorDiv.style.backgroundColor = 'rgba(255, 0, 0, 0.7)';
    errorDiv.style.color = 'white';
    errorDiv.style.padding = '10px 20px';
    errorDiv.style.borderRadius = '5px';
    errorDiv.style.zIndex = '1000';
    document.body.appendChild(errorDiv);

    setTimeout(() => {
        errorDiv.style.opacity = '0';
        errorDiv.style.transition = 'opacity 0.5s';
        setTimeout(() => errorDiv.remove(), 500);
    }, 3000);
}

function addMessage(sender, text) {
    const div = document.createElement('div');
    div.classList.add('message', sender.toLowerCase());
    
    let content = text;
    // Modify to add more types of content in the future if needed
    if (sender === 'ComputeGuru') {
        if (text.includes("<table") || text.includes("<style")) {
        content = text;
        } else {
            content = marked.parse(text);
        }
    }
    
    div.innerHTML = `<strong>${sender}:</strong> ${content}`;
    messagesDiv.appendChild(div);
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
