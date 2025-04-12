// UI Components module
const UIComponents = (function() {
    // Toast message component
    function showToast(message, duration = 3000) {
        const toast = document.createElement('div');
        toast.className = 'toast-message visible';
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('fadeout');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
    
    // Modal component
    function showModal(title, content, actions = []) {
        // Create modal container
        const modalContainer = document.createElement('div');
        modalContainer.className = 'modal-container';
        
        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.className = 'modal-content';
        
        // Add title
        const titleElement = document.createElement('h2');
        titleElement.className = 'modal-title';
        titleElement.textContent = title;
        modalContent.appendChild(titleElement);
        
        // Add content
        const contentElement = document.createElement('div');
        contentElement.className = 'modal-body';
        if (typeof content === 'string') {
            contentElement.innerHTML = content;
        } else {
            contentElement.appendChild(content);
        }
        modalContent.appendChild(contentElement);
        
        // Add actions
        if (actions.length > 0) {
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'modal-actions';
            
            actions.forEach(action => {
                const button = document.createElement('button');
                button.textContent = action.label;
                button.className = `modal-button ${action.type || 'secondary'}`;
                button.addEventListener('click', () => {
                    if (action.callback) action.callback();
                    closeModal(modalContainer);
                });
                actionsContainer.appendChild(button);
            });
            
            modalContent.appendChild(actionsContainer);
        }
        
        // Add close button
        const closeButton = document.createElement('button');
        closeButton.className = 'modal-close';
        closeButton.innerHTML = '&times;';
        closeButton.addEventListener('click', () => closeModal(modalContainer));
        modalContent.appendChild(closeButton);
        
        // Add to DOM
        modalContainer.appendChild(modalContent);
        document.body.appendChild(modalContainer);
        
        // Add animation
        setTimeout(() => modalContainer.classList.add('visible'), 10);
        
        return modalContainer;
    }
    
    function closeModal(modalContainer) {
        modalContainer.classList.remove('visible');
        setTimeout(() => modalContainer.remove(), 300);
    }
    
    // Loading indicator
    function showLoading(container, message = 'Loading...') {
        const loadingElement = document.createElement('div');
        loadingElement.className = 'loading-indicator';
        loadingElement.innerHTML = `
            <div class="spinner"></div>
            <p>${message}</p>
        `;
        
        if (typeof container === 'string') {
            container = document.querySelector(container);
        }
        
        if (container) {
            container.innerHTML = '';
            container.appendChild(loadingElement);
        } else {
            loadingElement.classList.add('fullscreen');
            document.body.appendChild(loadingElement);
        }
        
        return loadingElement;
    }
    
    function hideLoading(loadingElement) {
        if (loadingElement && loadingElement.parentNode) {
            loadingElement.parentNode.removeChild(loadingElement);
        }
    }
    
    // Training progress modal
    function showTrainingModal() {
        return showModal(
            'Training in Progress',
            `<div class="training-progress">
                <div class="spinner"></div>
                <p>Your AI companion is learning from your data.</p>
                <p>This may take a few minutes...</p>
            </div>`,
            []
        );
    }
    
    // Confirmation dialog
    function confirm(message, onConfirm, onCancel) {
        return showModal(
            'Confirmation',
            `<p>${message}</p>`,
            [
                {
                    label: 'Cancel',
                    type: 'secondary',
                    callback: onCancel
                },
                {
                    label: 'Confirm',
                    type: 'primary',
                    callback: onConfirm
                }
            ]
        );
    }
    
    // Prompt dialog
    function prompt(title, message, onSubmit) {
        const inputContainer = document.createElement('div');
        inputContainer.className = 'prompt-container';
        
        const messageElement = document.createElement('p');
        messageElement.textContent = message;
        inputContainer.appendChild(messageElement);
        
        const inputElement = document.createElement('input');
        inputElement.className = 'prompt-input';
        inputElement.type = 'text';
        inputContainer.appendChild(inputElement);
        
        const modalContainer = showModal(
            title,
            inputContainer,
            [
                {
                    label: 'Cancel',
                    type: 'secondary',
                    callback: () => {}
                },
                {
                    label: 'Submit',
                    type: 'primary',
                    callback: () => {
                        if (onSubmit) onSubmit(inputElement.value);
                    }
                }
            ]
        );
        
        // Focus input
        setTimeout(() => inputElement.focus(), 100);
        
        // Add enter key handling
        inputElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                if (onSubmit) onSubmit(inputElement.value);
                closeModal(modalContainer);
            }
        });
        
        return modalContainer;
    }
    
    // Public API
    return {
        showToast,
        showModal,
        closeModal,
        showLoading,
        hideLoading,
        showTrainingModal,
        confirm,
        prompt
    };
})();

// Export the UI components module
export default UIComponents;