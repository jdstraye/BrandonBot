let userId = localStorage.getItem('userId') || generateUserId();
let consentGiven = localStorage.getItem('consentGiven') === 'true';
let disclosureAcknowledged = localStorage.getItem('disclosureAcknowledged') === 'true';
let currentQuestion = '';
let conversationHistory = [];

function generateUserId() {
    const id = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('userId', id);
    return id;
}

if (disclosureAcknowledged) {
    document.getElementById('ai-disclosure').style.display = 'none';
    if (!consentGiven) {
        document.getElementById('consent-banner').style.display = 'block';
    }
}

if (consentGiven) {
    document.getElementById('consent-banner').style.display = 'none';
}

function dismissDisclosure() {
    disclosureAcknowledged = true;
    localStorage.setItem('disclosureAcknowledged', 'true');
    document.getElementById('ai-disclosure').style.display = 'none';
    document.getElementById('consent-banner').style.display = 'block';
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
    
    if (!isUser && data.escalation_level === 'high') {
        messageHTML += `
            <div class="escalation-offer">
                <p><strong>I sense this is important to you.</strong> Would you like someone from Brandon's team to give you a call directly?</p>
                <button onclick="openCallbackModal()">Yes, please call me</button>
            </div>
        `;
    } else if (!isUser && data.escalation_level === 'medium') {
        messageHTML += `
            <div class="callback-offer">
                <p><strong>Would you like to speak with someone from the team?</strong></p>
                <button onclick="openCallbackModal()">Request a callback</button>
            </div>
        `;
    } else if (!isUser && data.offer_callback) {
        messageHTML += `
            <div class="callback-offer">
                <p><strong>Would you like someone from the team to call you back?</strong></p>
                <button onclick="openCallbackModal()">Yes, request a callback</button>
            </div>
        `;
    }
    
    if (!isUser) {
        messageHTML += `<span class="ai-generated-tag">AI-generated response</span>`;
    }
    
    messageHTML += `</div>`;
    messageDiv.innerHTML = messageHTML;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    conversationHistory.push({
        role: isUser ? 'user' : 'assistant',
        content: content
    });
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
                consent_given: consentGiven,
                conversation_history: conversationHistory.slice(-10)
            })
        });
        
        const data = await response.json();
        addMessage(data.response, false, {
            ...data,
            escalation_level: data.escalation_level,
            offer_callback: data.offer_callback
        });
    } catch (error) {
        addMessage('Sorry, I encountered an error. Please try again or request a callback if you need immediate assistance.', false);
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

function openVolunteerModal() {
    document.getElementById('volunteer-modal').classList.add('active');
}

function closeVolunteerModal() {
    document.getElementById('volunteer-modal').classList.remove('active');
}

function openDonateModal() {
    document.getElementById('donate-modal').classList.add('active');
}

function closeDonateModal() {
    document.getElementById('donate-modal').classList.remove('active');
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

document.getElementById('volunteer-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const interests = [];
    document.querySelectorAll('input[name="interests"]:checked').forEach(cb => {
        interests.push(cb.value);
    });
    
    const formData = {
        name: document.getElementById('volunteer-name').value,
        email: document.getElementById('volunteer-email').value,
        phone: document.getElementById('volunteer-phone').value || '',
        zip_code: document.getElementById('volunteer-zip').value || '',
        interests: interests,
        availability: document.getElementById('volunteer-availability').value || 'flexible'
    };
    
    try {
        const response = await fetch('/api/volunteer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        closeVolunteerModal();
        addMessage(data.message || 'Thank you for volunteering! Someone from the team will be in touch soon.', false);
        
        document.getElementById('volunteer-form').reset();
    } catch (error) {
        alert('Failed to submit volunteer registration. Please try again.');
        console.error('Error:', error);
    }
});

document.getElementById('donate-interest-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        name: document.getElementById('donate-name').value,
        email: document.getElementById('donate-email').value,
        phone: document.getElementById('donate-phone').value || '',
        message: document.getElementById('donate-message').value || ''
    };
    
    try {
        const response = await fetch('/api/donate-interest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        closeDonateModal();
        addMessage(data.message || 'Thank you for your interest in supporting the campaign! Someone from the team will reach out with secure donation options.', false);
        
        document.getElementById('donate-interest-form').reset();
    } catch (error) {
        alert('Failed to submit donation interest. Please try again.');
        console.error('Error:', error);
    }
});

document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
});
