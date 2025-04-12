// API service module
const ApiService = (function() {
    // Helper functions
    function handleResponse(response) {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    }

    function handleError(context) {
        return error => {
            console.error(`Error in ${context}:`, error);
            return { error: true, message: `Sorry, something went wrong with ${context}. Please try again.` };
        };
    }

    // API methods
    async function getUserInfo() {
        try {
            const response = await fetch('/api/user');
            return handleResponse(response);
        } catch (error) {
            return handleError('user info')(error);
        }
    }

    async function sendMessage(message, context = { type: 'chat' }) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, context })
            });
            return handleResponse(response);
        } catch (error) {
            return handleError('chat')(error);
        }
    }

    async function submitJournal(entry) {
        try {
            const response = await fetch('/api/journal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: entry })
            });
            return handleResponse(response);
        } catch (error) {
            return handleError('journal')(error);
        }
    }

    async function getJournalEntries() {
        try {
            const response = await fetch('/api/journal');
            return handleResponse(response);
        } catch (error) {
            return handleError('journal entries')(error);
        }
    }

    async function getInsights() {
        try {
            const response = await fetch('/api/insights');
            return handleResponse(response);
        } catch (error) {
            return handleError('insights')(error);
        }
    }

    async function getMemories() {
        try {
            const response = await fetch('/api/memory-vault');
            return handleResponse(response);
        } catch (error) {
            return handleError('memories')(error);
        }
    }

    async function addMemory(memoryType, content) {
        try {
            const response = await fetch('/api/memory-vault', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ memory_type: memoryType, content })
            });
            return handleResponse(response);
        } catch (error) {
            return handleError('adding memory')(error);
        }
    }

    async function deleteMemory(memoryId) {
        try {
            const response = await fetch(`/api/memory-vault/${memoryId}`, {
                method: 'DELETE'
            });
            return handleResponse(response);
        } catch (error) {
            return handleError('deleting memory')(error);
        }
    }

    async function searchMemories(query) {
        try {
            const response = await fetch(`/api/memory-vault/search?q=${encodeURIComponent(query)}`);
            return handleResponse(response);
        } catch (error) {
            return handleError('searching memories')(error);
        }
    }

    async function updatePreferences(preferences) {
        try {
            const response = await fetch('/api/preferences', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(preferences)
            });
            return handleResponse(response);
        } catch (error) {
            return handleError('preferences')(error);
        }
    }

    async function logout() {
        try {
            const response = await fetch('/api/logout', { method: 'POST' });
            return handleResponse(response);
        } catch (error) {
            return handleError('logout')(error);
        }
    }

    // Public API
    return {
        getUserInfo,
        sendMessage,
        submitJournal,
        getJournalEntries,
        getInsights,
        getMemories,
        addMemory,
        deleteMemory,
        searchMemories,
        updatePreferences,
        logout
    };
})();

// Export the API service
export default ApiService;