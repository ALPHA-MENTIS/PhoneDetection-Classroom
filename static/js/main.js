// Get all DOM elements
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const videoFeed = document.getElementById('video-feed');
const placeholder = document.querySelector('.placeholder');
const alertBanner = document.getElementById('alert-banner');
const alertSound = document.getElementById('alert-sound');
const phoneCountElement = document.getElementById('phone-count');
const statusElement = document.getElementById('status');
const videoFileInput = document.getElementById('video-file-input');
const videoPathInput = document.getElementById('video-path');

// Tab switching
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(button => {
    button.addEventListener('click', function() {
        // Remove active class from all tabs
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked tab
        this.classList.add('active');
        const tabName = this.getAttribute('data-tab');
        document.getElementById(tabName + '-tab').classList.add('active');
    });
});

// File input handler
videoFileInput.addEventListener('change', function() {
    if (this.files.length > 0) {
        videoPathInput.value = this.files[0].name;
        // Store the full path (in real app, you'd upload this to server)
        videoPathInput.dataset.fullPath = this.files[0].path || this.files[0].name;
    }
});

// Variables to track state
let isDetecting = false;
let checkInterval = null;
let lastPhoneCount = 0;

// Get current active source
function getCurrentSource() {
    const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
    
    if (activeTab === 'camera') {
        return {
            type: 'camera',
            path: document.getElementById('camera-id').value
        };
    } else {
        const videoPath = videoPathInput.dataset.fullPath || videoPathInput.value;
        if (!videoPath) {
            return null;
        }
        return {
            type: 'video',
            path: videoPath
        };
    }
}

// START button handler
startBtn.addEventListener('click', async function() {
    const source = getCurrentSource();
    
    if (!source) {
        alert('Please select a video file');
        return;
    }
    
    try {
        const response = await fetch('/start_detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source_type: source.type,
                source_path: source.path
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            isDetecting = true;
            
            // Update UI
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusElement.textContent = 'DETECTING';
            statusElement.style.color = '#2ecc71';
            
            // Update status icon
            document.querySelector('.status-stat .stat-icon').textContent = '🔴';
            
            // Hide placeholder and show video feed
            placeholder.style.display = 'none';
            videoFeed.style.display = 'block';
            videoFeed.src = '/video_feed?' + new Date().getTime();
            
            // Start checking for phones
            startPhoneCountCheck();
            
        } else {
            alert('Error starting detection: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
});

// STOP button handler
stopBtn.addEventListener('click', async function() {
    try {
        const response = await fetch('/stop_detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            isDetecting = false;
            
            // Update UI
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusElement.textContent = 'IDLE';
            statusElement.style.color = '#95a5a6';
            
            // Update status icon
            document.querySelector('.status-stat .stat-icon').textContent = '🟢';
            
            // Show placeholder and hide video feed
            placeholder.style.display = 'block';
            videoFeed.style.display = 'none';
            videoFeed.src = '';
            
            // Stop checking for phones
            stopPhoneCountCheck();
            
            // Hide alert and reset count
            alertBanner.classList.remove('active');
            phoneCountElement.textContent = '0';
            lastPhoneCount = 0;
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
});

// Function to start checking phone count periodically
function startPhoneCountCheck() {
    checkInterval = setInterval(async () => {
        try {
            const response = await fetch('/get_phone_count');
            const data = await response.json();
            
            if (data.detecting) {
                const currentCount = data.count;
                
                // Update the phone count display
                phoneCountElement.textContent = currentCount;
                
                // If phones are detected, show alert and play sound
                if (currentCount > 0) {
                    alertBanner.classList.add('active');
                    
                    // Update alert text
                    if (currentCount === 1) {
                        document.getElementById('alert-text').textContent = 'PHONE DETECTED!';
                    } else {
                        document.getElementById('alert-text').textContent = currentCount + ' PHONES DETECTED!';
                    }
                    
                    // Play beep sound
                    if (alertSound.paused) {
                        alertSound.play().catch(e => console.log('Audio play failed:', e));
                    }
                } else {
                    // No phones detected
                    alertBanner.classList.remove('active');
                    alertSound.pause();
                    alertSound.currentTime = 0;
                }
                
                lastPhoneCount = currentCount;
            }
        } catch (error) {
            console.error('Error checking phone count:', error);
        }
    }, 500);
}

// Function to stop checking phone count
function stopPhoneCountCheck() {
    if (checkInterval) {
        clearInterval(checkInterval);
        checkInterval = null;
    }
    
    alertSound.pause();
    alertSound.currentTime = 0;
}

// Output path selector (placeholder function)
function selectOutputPath() {
    const path = prompt('Enter output video path (optional):');
    if (path) {
        document.getElementById('output-path').value = path;
    }
}