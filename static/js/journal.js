// Journal module
import ApiService from './api.js';
import AppState from './state.js';
import UIComponents from './ui-components.js';

const JournalModule = (function() {
    // Initialize the journal module
    function init() {
        setupEventListeners();
        loadJournalEntries();
    }
    
    // Setup journal-specific event listeners
    function setupEventListeners() {
        const journalInput = document.getElementById('journal-input');
        const journalSubmit = document.getElementById('journal-submit');
        
        if (journalSubmit) {
            journalSubmit.addEventListener('click', submitJournalEntry);
        }
        
        if (journalInput) {
            journalInput.addEventListener('keypress', e => {
                if (e.key === 'Enter' && e.ctrlKey) {
                    e.preventDefault();
                    submitJournalEntry();
                }
            });
        }
    }
    
    // Core journal functions
    async function submitJournalEntry() {
        const journalInput = document.getElementById('journal-input');
        const entry = journalInput.value.trim();
        if (!entry) return;
        
        console.log('Submitting journal entry:', entry);
        const result = await ApiService.submitJournal(entry);
        console.log('Journal submission result:', result);
        
        if (result.error) {
            UIComponents.showToast(result.message);
            return;
        }
        
        journalInput.value = '';
        UIComponents.showToast('Journal entry saved successfully');
        
        // Refresh the journal entries
        loadJournalEntries();
    }
    
    async function loadJournalEntries() {
        console.log('Loading journal entries...');
        const journalEntries = document.getElementById('journal-entries');
        if (!journalEntries) {
            console.error('Journal entries container not found');
            return;
        }
        
        // Show loading state
        journalEntries.innerHTML = '<div class="loading">Loading journal entries...</div>';
        
        try {
            const result = await ApiService.getJournalEntries();
            console.log('Journal entries loaded:', result);
            
            if (result.error) {
                console.error('Error loading journal entries:', result.message);
                journalEntries.innerHTML = '<div class="error">Failed to load journal entries</div>';
                return;
            }
            
            if (!result.entries || result.entries.length === 0) {
                journalEntries.innerHTML = '<div class="empty-state">No journal entries yet. Start writing!</div>';
                return;
            }
            
            // Render journal entries
            journalEntries.innerHTML = '';
            result.entries.forEach(entry => {
                const entryElement = document.createElement('div');
                entryElement.className = 'journal-entry';
                entryElement.innerHTML = `
                    <div class="entry-date">${formatDate(entry.timestamp)}</div>
                    <div class="entry-content">${entry.content}</div>
                    <div class="entry-emotions">
                        ${renderEmotions(entry.emotions)}
                    </div>
                `;
                journalEntries.appendChild(entryElement);
            });
        } catch (error) {
            console.error('Exception in loadJournalEntries:', error);
            journalEntries.innerHTML = '<div class="error">An error occurred while loading journal entries</div>';
        }
    }
    
    // Helper functions
    function formatDate(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    function renderEmotions(emotions) {
        if (!emotions || Object.keys(emotions).length === 0) {
            return '';
        }
        
        return Object.entries(emotions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([emotion, score]) => {
                return `<span class="emotion-tag" data-emotion="${emotion}">${emotion} (${Math.round(score * 100)}%)</span>`;
            })
            .join('');
    }
    
    // Public API
    return {
        init,
        loadJournalEntries
    };
})();

// Export the journal module
export default JournalModule;