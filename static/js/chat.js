// Chat module
import ApiService from './api.js';
import AppState from './state.js';
import UIComponents from './ui-components.js';

const ChatModule = (function() {
    let socket = null;
    
    // Initialize the chat module
    function init(socketInstance) {
        socket = socketInstance;
        setupEventListeners();
    }
    
    // Setup chat-specific event listeners
    function setupEventListeners() {
        const chatInput = document.getElementById('chat-input');
        const sendButton = document.getElementById('send-button');
        const voiceRecordButton = document.getElementById('voice-record-button');
        
        if (sendButton) {
            sendButton.addEventListener('click', sendChatMessage);
        }
        
        if (chatInput) {
            chatInput.addEventListener('keypress', e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendChatMessage();
                }
            });
        }
        
        if (voiceRecordButton) {
            voiceRecordButton.addEventListener('click', toggleVoiceRecording);
        }
    }
    
    // Socket event handlers
    function handleTypingIndicator(data) {
        data.status === 'started' ? showTypingIndicator() : hideTypingIndicator();
    }

    function handleAIResponse(data) {
        hideTypingIndicator();
        appendMessage(data.response, 'ai');
    }
    
    // Core chat functions
    function sendChatMessage() {
        const chatInput = document.getElementById('chat-input');
        const message = chatInput.value.trim();
        if (!message) return;
        
        appendMessage(message, 'user');
        chatInput.value = '';
        showTypingIndicator();
        
        ApiService.sendMessage(message)
            .then(data => {
                hideTypingIndicator();
                if (data.error) {
                    UIComponents.showToast(data.message);
                    return;
                }
                
                if (data.response) {
                    appendMessage(data.response, 'ai');
                    if (data.emotion === 'sadness' && data.emotion_score > 0.6) {
                        checkLoneliness(message);
                    }
                }
            });
    }
    
    function appendMessage(content, sender) {
        const chatMessages = document.getElementById('chat-messages');
        const msgElement = document.createElement('div');
        msgElement.className = `message ${sender}-message`;
        msgElement.innerHTML = `<div class="message-content"><p>${content}</p></div>`;
        chatMessages.appendChild(msgElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function showTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }
    
    function checkLoneliness(message) {
        // Example function to check if user needs additional support
        // This would be implemented based on your specific requirements
        console.log('Checking loneliness indicators in message:', message);
    }
    
    function toggleVoiceRecording() {
        // Voice recording functionality
        console.log('Voice recording toggled');
        // Implement speech-to-text functionality
    }
    
    // Register socket event handlers
    function registerSocketHandlers(socket) {
        socket.on('typing_indicator', handleTypingIndicator);
        socket.on('ai_response', handleAIResponse);
    }
    
    // Public API
    return {
        init,
        appendMessage,
        registerSocketHandlers
    };
})();

// Export the chat module
export default ChatModule;