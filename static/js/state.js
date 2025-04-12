// State management module
const AppState = (function() {
    // Private state
    const state = {
        currentUser: null,
        nightMode: localStorage.getItem('nightMode') === 'true',
        sections: {
            active: 'chat'
        }
    };

    // Observers pattern for state changes
    const observers = {};

    // Subscribe to state changes
    function subscribe(key, callback) {
        if (!observers[key]) {
            observers[key] = [];
        }
        observers[key].push(callback);
        return () => unsubscribe(key, callback);
    }

    // Unsubscribe from state changes
    function unsubscribe(key, callback) {
        if (observers[key]) {
            observers[key] = observers[key].filter(cb => cb !== callback);
        }
    }

    // Notify observers of state changes
    function notify(key) {
        if (observers[key]) {
            observers[key].forEach(callback => callback(state[key]));
        }
    }

    // Get state
    function get(key) {
        return key ? state[key] : {...state};
    }

    // Set state
    function set(key, value) {
        if (typeof key === 'object') {
            // Handle batch update
            Object.entries(key).forEach(([k, v]) => {
                state[k] = v;
                notify(k);
            });
        } else {
            state[key] = value;
            notify(key);
            
            // Special handling for certain state changes
            if (key === 'nightMode') {
                localStorage.setItem('nightMode', value);
            }
        }
    }

    // Public API
    return {
        get,
        set,
        subscribe
    };
})();

// Export the state module
export default AppState;