let userId = localStorage.getItem('userId') || generateUserId();
let consentGiven = localStorage.getItem('consentGiven') === 'true';
let currentQuestion = '';

function generateUserId() {
    const id = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('userId', id);
    return id;
}

if (consentGiven) {
    document.getElementById('consent-banner').style.display = 'none';
}

async function handleConsent(consent) {
    consentGiven = consent;
    localStorage.setItem('consentGiven', consent);
    document.getElementById('consent-banner').style.display = 'none';
    
    try {
        await fetch('/api/consent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, consent_given: consent })
        });
    } catch (error) {
        console.error('Failed to update consent:', error);
    }
}

function addMessage(content, isUser = false, data = {}) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    let messageHTML = `<div class="message-content">`;
    
    if (typeof content === 'string') {
        const paragraphs = content.split('\n').filter(p => p.trim());
        paragraphs.forEach(p => {
            messageHTML += `<p>${escapeHtml(p)}</p>`;
        });
    }
    
    if (!isUser && data.confidence !== undefined) {
        const confidencePercent = Math.round(data.confidence * 100);
        messageHTML += `<div class="confidence-badge">Confidence: ${confidencePercent}%</div>`;
    }
    
    if (!isUser && data.sources && data.sources.length > 0) {
        messageHTML += `<div class="sources">Sources: `;
        messageHTML += data.sources.map(s => s.collection.replace('Platform', ' Platform')).join(', ');
        messageHTML += `</div>`;
    }
    
    if (!isUser && data.offer_callback) {
        messageHTML += `
            <div class="callback-offer">
                <p><strong>Would you like Brandon to call you back?</strong></p>
                <button onclick="openCallbackModal()">Yes, request a callback</button>
            </div>
        `;
    }
    
    messageHTML += `</div>`;
    messageDiv.innerHTML = messageHTML;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function sendQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    
    if (!query) return;
    
    currentQuestion = query;
    addMessage(query, true);
    input.value = '';
    
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;
    sendBtn.textContent = 'Thinking...';
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                user_id: userId,
                consent_given: consentGiven
            })
        });
        
        const data = await response.json();
        addMessage(data.response, false, data);
    } catch (error) {
        addMessage('Sorry, I encountered an error. Please try again.', false);
        console.error('Error:', error);
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
    }
}

document.getElementById('query-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendQuery();
    }
});

function openCallbackModal() {
    document.getElementById('callback-question').value = currentQuestion;
    document.getElementById('callback-modal').classList.add('active');
}

function closeCallbackModal() {
    document.getElementById('callback-modal').classList.remove('active');
}

document.getElementById('callback-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        user_id: userId,
        name: document.getElementById('callback-name').value,
        phone: document.getElementById('callback-phone').value,
        email: document.getElementById('callback-email').value,
        question: document.getElementById('callback-question').value
    };
    
    try {
        const response = await fetch('/api/callback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        closeCallbackModal();
        addMessage(data.message, false);
        
        document.getElementById('callback-form').reset();
    } catch (error) {
        alert('Failed to submit callback request. Please try again.');
        console.error('Error:', error);
    }
});
