// Main application module
import AppState from './state.js';
import ApiService from './api.js';
import ChatModule from './chat.js';
import JournalModule from './journal.js';
import UIComponents from './ui-components.js';

document.addEventListener('DOMContentLoaded', function() {
    // Initialize socket connection
    let socket = null;
    
    // Cache DOM elements
    const elements = {
        appInterface: document.getElementById('app-interface'),
        userName: document.getElementById('user-name'),
        navItems: document.querySelectorAll('.nav-item'),
        contentSections: document.querySelectorAll('.content-section'),
        logoutButton: document.getElementById('logout-button'),
        nightModeToggle: document.getElementById('night-mode-toggle'),
        refreshInsights: document.getElementById('refresh-insights'),
        savePreferences: document.getElementById('save-preferences')
    };

    // Initialize the application
    async function init() {
        await getUserInfo();
        initSocket();
        setupEventListeners();
        
        // Initialize modules
        ChatModule.init(socket);
        JournalModule.init();
        
        // Apply initial state
        applyNightMode(AppState.get('nightMode'));
    }

    // Initialize socket connection
    function initSocket() {
        socket = io();
        socket.on('connect', () => {
            const currentUser = AppState.get('currentUser');
            if (currentUser) socket.emit('join', { user_id: currentUser });
        });
        
        // Register socket event handlers
        ChatModule.registerSocketHandlers(socket);
        
        // Training events
        socket.on('training_started', handleTrainingStarted);
        socket.on('training_completed', handleTrainingCompleted);
    }

    // Socket event handlers
    function handleTrainingStarted(data) {
        const currentUser = AppState.get('currentUser');
        if (data.user_id === currentUser) UIComponents.showTrainingModal();
    }

    function handleTrainingCompleted(data) {
        const currentUser = AppState.get('currentUser');
        if (data.user_id === currentUser) {
            // Close training modal and show success message
            document.querySelector('.modal-container')?.remove();
            UIComponents.showToast('Training completed successfully!');
        }
    }

    // Get current user information
    async function getUserInfo() {
        const data = await ApiService.getUserInfo();
        if (data.error) {
            window.location.href = '/login';
            return;
        }
        
        if (data.user) {
            AppState.set('currentUser', data.user.user_id);
            if (elements.userName) {
                elements.userName.textContent = data.user.username;
            }
        } else {
            window.location.href = '/login';
        }
    }

    // Setup application-wide event listeners
    function setupEventListeners() {
        console.log("Setting up event listeners");
        console.log("Nav items found:", elements.navItems.length);
        
        // Navigation
        elements.navItems.forEach(item => {
            item.addEventListener('click', () => {
                console.log("Clicked on nav item:", item.dataset.section);
                switchSection(item.dataset.section);
            });
        });
        
        // Night mode toggle
        if (elements.nightModeToggle) {
            console.log("Night mode toggle found");
            elements.nightModeToggle.addEventListener('click', toggleNightMode);
        } else {
            console.error("Night mode toggle element not found");
        }
        
        // Logout button
        if (elements.logoutButton) {
            console.log("Logout button found");
            elements.logoutButton.addEventListener('click', handleLogout);
        } else {
            console.error("Logout button element not found");
        }
        
        // Insights refresh
        if (elements.refreshInsights) {
            console.log("Refresh insights button found");
            elements.refreshInsights.addEventListener('click', loadInsights);
        } else {
            console.error("Refresh insights button element not found");
        }
        
        // Save preferences
        if (elements.savePreferences) {
            console.log("Save preferences button found");
            elements.savePreferences.addEventListener('click', saveUserPreferences);
        } else {
            console.error("Save preferences button element not found");
        }
        
        // Add topic button
        const addTopicButton = document.querySelector('.topic-tag.add-topic');
        if (addTopicButton) {
            console.log("Add topic button found");
            addTopicButton.addEventListener('click', addNewTopic);
        } else {
            console.error("Add topic button element not found");
        }
        
        // Set up event delegation for topic tag removal (clicking on existing topics)
        const topicsContainer = document.querySelector('.topics-container');
        if (topicsContainer) {
            console.log("Topics container found");
            topicsContainer.addEventListener('click', handleTopicClick);
        } else {
            console.error("Topics container element not found");
        }
        
        // Add direct click handler for preferences nav item for debugging
        const preferencesNavItem = document.querySelector('.nav-item[data-section="preferences"]');
        if (preferencesNavItem) {
            console.log("Preferences nav item found");
            preferencesNavItem.addEventListener('click', function() {
                console.log("Direct click on preferences nav item");
                const preferencesSection = document.getElementById('preferences-section');
                
                // Ensure all sections are properly hidden
                elements.contentSections.forEach(section => {
                    section.classList.add('hidden');
                });
                
                // Show preferences section
                if (preferencesSection) {
                    console.log("Showing preferences section directly");
                    preferencesSection.classList.remove('hidden');
                    loadUserPreferences();
                } else {
                    console.error("Preferences section element not found");
                }
            });
        } else {
            console.error("Preferences nav item not found");
        }
    }

    // Switch between different sections of the app
    function switchSection(sectionId) {
        console.log(`Switching to section: ${sectionId}`);
        AppState.set('sections', { active: sectionId });
        
        // Update active nav item
        elements.navItems.forEach(item => {
            console.log(`Nav item: ${item.dataset.section} - Active: ${item.dataset.section === sectionId}`);
            item.classList.toggle('active', item.dataset.section === sectionId);
        });
        
        // Show/hide sections
        elements.contentSections.forEach(section => {
            console.log(`Section: ${section.id} - Visible: ${section.id === `${sectionId}-section`}`);
            section.classList.toggle('hidden', section.id !== `${sectionId}-section`);
        });
        
        // Load section-specific data
        const sectionLoaders = {
            'insights': loadInsights,
            'memory-vault': loadMemories,
            'journal': () => JournalModule.loadJournalEntries(),
            'preferences': loadUserPreferences // Add preferences loader
        };
        
        if (sectionLoaders[sectionId]) {
            console.log(`Loading data for section: ${sectionId}`);
            sectionLoaders[sectionId]();
        }
    }

    // Toggle night mode
    function toggleNightMode() {
        const nightMode = !AppState.get('nightMode');
        AppState.set('nightMode', nightMode);
        applyNightMode(nightMode);
    }
    
    // Apply night mode
    function applyNightMode(enabled) {
        document.body.classList.toggle('night-mode', enabled);
    }

    // Load insights data
    async function loadInsights() {
        const insightsSection = document.getElementById('insights-section');
        if (!insightsSection) return;
        
        const loadingIndicator = UIComponents.showLoading(insightsSection, 'Loading insights...');
        
        const data = await ApiService.getInsights();
        UIComponents.hideLoading(loadingIndicator);
        
        if (data.error) {
            insightsSection.innerHTML = '<div class="error-state">Failed to load insights</div>';
            return;
        }
        
        // Render insights (to be implemented based on your specific data structure)
        renderInsights(data.insights);
    }
    
    // Render insights
    function renderInsights(insights) {
        // Implement based on your specific insights data structure
        const insightsSection = document.getElementById('insights-section');
        insightsSection.innerHTML = '<h2>Your Insights</h2>';
        
        // Example implementation
        if (!insights || insights.length === 0) {
            insightsSection.innerHTML += '<div class="empty-state">No insights available yet. Continue chatting or journaling to generate insights.</div>';
            return;
        }
        
        const insightsList = document.createElement('div');
        insightsList.className = 'insights-list';
        
        insights.forEach(insight => {
            const insightElement = document.createElement('div');
            insightElement.className = 'insight-card';
            insightElement.innerHTML = `
                <h3>${insight.title}</h3>
                <p>${insight.description}</p>
                <div class="insight-metadata">
                    <span class="date">${new Date(insight.created_at || Date.now()).toLocaleDateString()}</span>
                    <span class="type">${insight.type || 'general'}</span>
                </div>
            `;
            insightsList.appendChild(insightElement);
        });
        
        insightsSection.appendChild(insightsList);
    }

    // Load memories data
    async function loadMemories() {
        const memorySection = document.getElementById('memory-vault-section');
        if (!memorySection) return;
        
        const loadingIndicator = UIComponents.showLoading(memorySection, 'Loading memories...');
        
        const data = await ApiService.getMemories();
        UIComponents.hideLoading(loadingIndicator);
        
        if (data.error) {
            memorySection.innerHTML = '<div class="error-state">Failed to load memories</div>';
            return;
        }
        
        // Render memories (to be implemented based on your specific data structure)
        renderMemories(data.memories);
    }
    
    // Render memories
    function renderMemories(memories) {
        // Implement based on your specific memories data structure
        const memorySection = document.getElementById('memory-vault-section');
        memorySection.innerHTML = '<h2>Memory Vault</h2>';
        
        // Example implementation
        if (!memories || memories.length === 0) {
            memorySection.innerHTML += '<div class="empty-state">No memories stored yet. Your AI companion will remember important information from your conversations.</div>';
            return;
        }
        
        const memoriesList = document.createElement('div');
        memoriesList.className = 'memories-list';
        
        memories.forEach(memory => {
            const memoryElement = document.createElement('div');
            memoryElement.className = 'memory-card';
            memoryElement.innerHTML = `
                <h3>${memory.topic}</h3>
                <p>${memory.content}</p>
                <div class="memory-metadata">
                    <span class="date">${new Date(memory.created).toLocaleDateString()}</span>
                    <span class="importance">${getImportanceLabel(memory.importance)}</span>
                </div>
            `;
            memoriesList.appendChild(memoryElement);
        });
        
        memorySection.appendChild(memoriesList);
    }
    
    // Helper to get importance label
    function getImportanceLabel(importance) {
        const labels = {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Critical'
        };
        return labels[importance] || 'Unknown';
    }
    
    // Save user preferences
    async function saveUserPreferences() {
        // Collect preferences from UI elements
        const preferences = {
            nightMode: AppState.get('nightMode'),
            communication_style: document.getElementById('communication-style').value,
            response_length: document.getElementById('response-length').value,
            support_preferences: {
                offer_advice: document.getElementById('offer-advice').checked,
                emotional_support: document.getElementById('emotional-support').checked,
                challenge_thoughts: document.getElementById('challenge-thoughts').checked
            },
            interests: [] // Will be populated with topics
        };
        
        // Collect topic tags (excluding the "add" button)
        const topicTags = document.querySelectorAll('.topic-tag:not(.add-topic)');
        topicTags.forEach(tag => {
            preferences.interests.push(tag.textContent.trim());
        });
        
        const loadingIndicator = UIComponents.showLoading(null, 'Saving preferences...');
        
        const result = await ApiService.updatePreferences(preferences);
        UIComponents.hideLoading(loadingIndicator);
        
        if (result.error) {
            UIComponents.showToast(result.message || 'Failed to save preferences');
            return;
        }
        
        UIComponents.showToast('Preferences saved successfully');
    }
    
    // Load user preferences
    async function loadUserPreferences() {
        console.log("Loading user preferences");
        const preferencesSection = document.getElementById('preferences-section');
        if (!preferencesSection) {
            console.error("Preferences section not found");
            return;
        }
        
        // Make sure the preferences section is visible
        preferencesSection.classList.remove('hidden');
        
        // Create loading overlay instead of replacing content
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.style.position = 'absolute';
        loadingOverlay.style.top = '0';
        loadingOverlay.style.left = '0';
        loadingOverlay.style.width = '100%';
        loadingOverlay.style.height = '100%';
        loadingOverlay.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        loadingOverlay.style.display = 'flex';
        loadingOverlay.style.justifyContent = 'center';
        loadingOverlay.style.alignItems = 'center';
        loadingOverlay.style.zIndex = '1000';
        
        const loadingSpinner = document.createElement('div');
        loadingSpinner.className = 'spinner';
        loadingSpinner.style.width = '50px';
        loadingSpinner.style.height = '50px';
        loadingSpinner.style.border = '5px solid #f3f3f3';
        loadingSpinner.style.borderTop = '5px solid #3498db';
        loadingSpinner.style.borderRadius = '50%';
        loadingSpinner.style.animation = 'spin 1s linear infinite';
        
        // Add animation style
        const style = document.createElement('style');
        style.textContent = '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }';
        document.head.appendChild(style);
        
        loadingOverlay.appendChild(loadingSpinner);
        
        // Set relative positioning on the preferences section for proper overlay positioning
        preferencesSection.style.position = 'relative';
        preferencesSection.appendChild(loadingOverlay);
        
        try {
            // Get preferences from server
            const response = await fetch('/api/preferences', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to load preferences');
            }
            
            const data = await response.json();
            
            // Remove the loading overlay
            preferencesSection.removeChild(loadingOverlay);
            
            if (data.error) {
                UIComponents.showToast(data.message || 'Failed to load preferences');
                return;
            }
            
            const preferences = data.preferences || {};
            console.log("Loaded preferences:", preferences);
            
            // Set communication style
            const commStyleSelect = document.getElementById('communication-style');
            if (commStyleSelect && preferences.communication_style) {
                console.log("Setting communication style to:", preferences.communication_style);
                commStyleSelect.value = preferences.communication_style;
            }
            
            // Set response length
            const responseLengthSelect = document.getElementById('response-length');
            if (responseLengthSelect && preferences.response_length) {
                console.log("Setting response length to:", preferences.response_length);
                responseLengthSelect.value = preferences.response_length;
            }
            
            // Set support preferences checkboxes
            if (preferences.support_preferences) {
                const offerAdvice = document.getElementById('offer-advice');
                if (offerAdvice && preferences.support_preferences.offer_advice !== undefined) {
                    offerAdvice.checked = preferences.support_preferences.offer_advice;
                }
                
                const emotionalSupport = document.getElementById('emotional-support');
                if (emotionalSupport && preferences.support_preferences.emotional_support !== undefined) {
                    emotionalSupport.checked = preferences.support_preferences.emotional_support;
                }
                
                const challengeThoughts = document.getElementById('challenge-thoughts');
                if (challengeThoughts && preferences.support_preferences.challenge_thoughts !== undefined) {
                    challengeThoughts.checked = preferences.support_preferences.challenge_thoughts;
                }
            }
            
            // Set night mode
            if (preferences.nightMode !== undefined) {
                AppState.set('nightMode', preferences.nightMode);
                applyNightMode(preferences.nightMode);
            }
            
            // Handle interests/topics
            if (preferences.interests && preferences.interests.length > 0) {
                const topicsContainer = preferencesSection.querySelector('.topics-container');
                if (topicsContainer) {
                    // Clear existing topics except the add button
                    const existingTopics = topicsContainer.querySelectorAll('.topic-tag:not(.add-topic)');
                    existingTopics.forEach(topic => topic.remove());
                    
                    // Add the add-topic button back if it was removed
                    const addTopicButton = topicsContainer.querySelector('.add-topic');
                    
                    // Add saved topics
                    preferences.interests.forEach(interest => {
                        const topicTag = document.createElement('div');
                        topicTag.className = 'topic-tag';
                        topicTag.textContent = interest;
                        if (addTopicButton) {
                            topicsContainer.insertBefore(topicTag, addTopicButton);
                        } else {
                            topicsContainer.appendChild(topicTag);
                        }
                    });
                }
            }
        } catch (error) {
            // Remove the loading overlay
            if (loadingOverlay.parentNode === preferencesSection) {
                preferencesSection.removeChild(loadingOverlay);
            }
            UIComponents.showToast('Error loading preferences');
            console.error('Error loading preferences:', error);
        }
    }
    
    // Handle logout
    async function handleLogout() {
        UIComponents.confirm(
            'Are you sure you want to log out?',
            async () => {
                const result = await ApiService.logout();
                if (result.error) {
                    UIComponents.showToast(result.message);
                    return;
                }
                
                window.location.href = '/login';
            },
            () => {} // Cancel callback
        );
    }

    // Handle topic click (for removal)
    function handleTopicClick(event) {
        const target = event.target;
        // Only handle clicks on non-add topic tags
        if (target.classList.contains('topic-tag') && !target.classList.contains('add-topic')) {
            // If in edit mode (showing X) or user holds Ctrl/Cmd, remove the topic
            if (event.ctrlKey || event.metaKey || target.dataset.editMode === 'true') {
                target.remove();
            } else {
                // Toggle edit mode by adding an X or visual indicator
                target.dataset.editMode = 'true';
                target.style.position = 'relative';
                
                // Add remove indicator
                const removeIndicator = document.createElement('span');
                removeIndicator.className = 'remove-indicator';
                removeIndicator.textContent = '×';
                removeIndicator.style.position = 'absolute';
                removeIndicator.style.top = '-5px';
                removeIndicator.style.right = '-5px';
                removeIndicator.style.backgroundColor = '#ff5555';
                removeIndicator.style.color = 'white';
                removeIndicator.style.borderRadius = '50%';
                removeIndicator.style.width = '18px';
                removeIndicator.style.height = '18px';
                removeIndicator.style.textAlign = 'center';
                removeIndicator.style.lineHeight = '16px';
                removeIndicator.style.fontSize = '14px';
                removeIndicator.style.cursor = 'pointer';
                
                target.appendChild(removeIndicator);
                
                // Auto-remove edit mode after a delay
                setTimeout(() => {
                    if (target.dataset.editMode === 'true') {
                        target.dataset.editMode = 'false';
                        const indicator = target.querySelector('.remove-indicator');
                        if (indicator) indicator.remove();
                    }
                }, 3000); // Remove after 3 seconds
            }
        }
    }

    // Add a new topic
    function addNewTopic() {
        // Prompt user for new topic
        UIComponents.prompt(
            'Add a new topic of interest',
            'Enter a topic name:',
            (topic) => {
                if (!topic || topic.trim().length === 0) return;
                
                // Create new topic tag
                const topicsContainer = document.querySelector('.topics-container');
                const addButton = document.querySelector('.topic-tag.add-topic');
                
                if (topicsContainer && addButton) {
                    const topicTag = document.createElement('div');
                    topicTag.className = 'topic-tag';
                    topicTag.textContent = topic.trim();
                    
                    // Insert before the add button
                    topicsContainer.insertBefore(topicTag, addButton);
                }
            }
        );
    }

    // Start the application
    init();
});