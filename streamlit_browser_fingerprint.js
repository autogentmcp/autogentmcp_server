/**
 * Comprehensive Browser Fingerprinting for Streamlit
 * This script gathers detailed browser and system information for session management
 * Usage in Streamlit: st.components.v1.html(fingerprint_js, height=0)
 */

function generateBrowserFingerprint() {
    const fingerprint = {};
    
    try {
        // Basic browser information
        fingerprint.user_agent = navigator.userAgent || null;
        fingerprint.language = navigator.language || navigator.userLanguage || null;
        fingerprint.platform = navigator.platform || null;
        fingerprint.cookie_enabled = navigator.cookieEnabled || false;
        fingerprint.do_not_track = navigator.doNotTrack || navigator.msDoNotTrack || null;
        
        // Screen and display information
        if (screen) {
            fingerprint.screen_resolution = `${screen.width}x${screen.height}`;
            fingerprint.color_depth = screen.colorDepth || null;
            fingerprint.pixel_depth = screen.pixelDepth || null;
        }
        
        // Viewport information
        fingerprint.viewport_size = `${window.innerWidth}x${window.innerHeight}`;
        
        // Timezone
        try {
            fingerprint.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            fingerprint.timezone_offset = new Date().getTimezoneOffset();
        } catch (e) {
            fingerprint.timezone = null;
            fingerprint.timezone_offset = null;
        }
        
        // Hardware information
        fingerprint.hardware_concurrency = navigator.hardwareConcurrency || null;
        fingerprint.device_memory = navigator.deviceMemory || null;
        
        // Connection information
        if (navigator.connection) {
            fingerprint.connection_type = navigator.connection.effectiveType || null;
            fingerprint.connection_downlink = navigator.connection.downlink || null;
        }
        
        // WebGL fingerprint
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            if (gl) {
                const renderer = gl.getParameter(gl.RENDERER);
                const vendor = gl.getParameter(gl.VENDOR);
                fingerprint.webgl_fingerprint = `${vendor} - ${renderer}`;
                fingerprint.webgl_version = gl.getParameter(gl.VERSION);
            }
        } catch (e) {
            fingerprint.webgl_fingerprint = null;
        }
        
        // Canvas fingerprint
        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            ctx.textBaseline = 'top';
            ctx.font = '14px Arial';
            ctx.fillText('Browser fingerprint canvas test 🚀', 2, 2);
            ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
            ctx.fillRect(100, 5, 80, 20);
            fingerprint.canvas_fingerprint = btoa(canvas.toDataURL()).slice(-50); // Last 50 chars
        } catch (e) {
            fingerprint.canvas_fingerprint = null;
        }
        
        // Audio fingerprint
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const analyser = audioContext.createAnalyser();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(analyser);
            oscillator.frequency.value = 1000;
            oscillator.start();
            
            // Simple audio fingerprint
            fingerprint.audio_fingerprint = `${audioContext.sampleRate}-${analyser.frequencyBinCount}`;
            
            setTimeout(() => {
                try {
                    oscillator.stop();
                    audioContext.close();
                } catch (e) {}
            }, 100);
        } catch (e) {
            fingerprint.audio_fingerprint = null;
        }
        
        // Installed plugins (limited in modern browsers)
        try {
            const plugins = Array.from(navigator.plugins || []).map(p => p.name).sort();
            fingerprint.plugins_hash = btoa(plugins.join(',')).slice(-20);
        } catch (e) {
            fingerprint.plugins_hash = null;
        }
        
        // Local storage test
        try {
            const testKey = '_fingerprint_test';
            localStorage.setItem(testKey, 'test');
            fingerprint.local_storage_enabled = localStorage.getItem(testKey) === 'test';
            localStorage.removeItem(testKey);
        } catch (e) {
            fingerprint.local_storage_enabled = false;
        }
        
        // Session storage test
        try {
            const testKey = '_fingerprint_test';
            sessionStorage.setItem(testKey, 'test');
            fingerprint.session_storage_enabled = sessionStorage.getItem(testKey) === 'test';
            sessionStorage.removeItem(testKey);
        } catch (e) {
            fingerprint.session_storage_enabled = false;
        }
        
        // Battery API (deprecated but might still work)
        if (navigator.getBattery) {
            navigator.getBattery().then(battery => {
                fingerprint.battery_charging = battery.charging;
                fingerprint.battery_level = Math.round(battery.level * 100);
            }).catch(() => {});
        }
        
        // Media devices
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            navigator.mediaDevices.enumerateDevices().then(devices => {
                const deviceTypes = devices.map(d => d.kind).sort();
                fingerprint.media_devices_hash = btoa(deviceTypes.join(',')).slice(-20);
            }).catch(() => {});
        }
        
        // Generate a quality score
        let quality_score = 0;
        const important_fields = [
            'user_agent', 'screen_resolution', 'timezone', 'canvas_fingerprint', 
            'webgl_fingerprint', 'hardware_concurrency', 'device_memory'
        ];
        
        important_fields.forEach(field => {
            if (fingerprint[field] !== null && fingerprint[field] !== undefined) {
                quality_score += 1;
            }
        });
        
        fingerprint.quality_score = quality_score / important_fields.length;
        fingerprint.timestamp = Date.now();
        
    } catch (error) {
        fingerprint.error = error.message;
    }
    
    return fingerprint;
}

// Function to send fingerprint to backend
async function sendFingerprintToBackend(fingerprint, baseUrl = 'http://localhost:8001') {
    try {
        const response = await fetch(`${baseUrl}/enhanced/session/create`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                browser_fingerprint: fingerprint
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Store session info in local storage for persistence
            try {
                localStorage.setItem('mcp_session_id', data.session_id);
                localStorage.setItem('mcp_user_id', data.user_id);
                localStorage.setItem('mcp_fingerprint_quality', data.fingerprint_quality);
            } catch (e) {
                // Could not store session in localStorage
            }
            
            return data;
        } else {
            return null;
        }
    } catch (error) {
        return null;
    }
}

// Function to get existing session or create new one
async function getOrCreateSession(baseUrl = 'http://localhost:8001') {
    try {
        // Check if we have an existing session
        const existingSessionId = localStorage.getItem('mcp_session_id');
        if (existingSessionId) {
            return {
                session_id: existingSessionId,
                user_id: localStorage.getItem('mcp_user_id'),
                is_existing: true
            };
        }
        
        // Generate new fingerprint and create session
        const fingerprint = generateBrowserFingerprint();
        
        const sessionData = await sendFingerprintToBackend(fingerprint, baseUrl);
        if (sessionData) {
            return {
                ...sessionData,
                is_existing: false
            };
        }
        
        // Fallback: generate simple session ID
        const fallbackSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('mcp_session_id', fallbackSessionId);
        
        return {
            session_id: fallbackSessionId,
            user_id: 'anonymous',
            is_existing: false,
            is_fallback: true
        };
        
    } catch (error) {
        // Return a fallback session
        const fallbackSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        return {
            session_id: fallbackSessionId,
            user_id: 'anonymous',
            is_existing: false,
            is_fallback: true,
            error: error.message
        };
    }
}

// Function to create new conversation
async function createNewConversation(sessionId, baseUrl = 'http://localhost:8001') {
    try {
        const response = await fetch(`${baseUrl}/enhanced/session/${sessionId}/new_conversation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            return data;
        } else {
            return null;
        }
    } catch (error) {
        return null;
    }
}

// Export functions for use in Streamlit
window.generateBrowserFingerprint = generateBrowserFingerprint;
window.sendFingerprintToBackend = sendFingerprintToBackend;
window.getOrCreateSession = getOrCreateSession;
window.createNewConversation = createNewConversation;

// Auto-initialize session when script loads
window.addEventListener('load', async () => {
    // Auto-create session if not exists
    const sessionData = await getOrCreateSession();
    window.currentSession = sessionData;
    
    // Dispatch custom event with session data for Streamlit to catch
    window.dispatchEvent(new CustomEvent('sessionInitialized', {
        detail: sessionData
    }));
});

console.log('Browser fingerprinting script loaded');
