let streamInterval = null;
let socket = null;
let currentStream = null;

async function startStream(socketUrl) {
    console.log("Initializing stream...");
    
    // 1. Get Webcam Access
    if (!currentStream) {
        try {
            // FIX: Use { video: true } to accept ANY available camera
            currentStream = await navigator.mediaDevices.getUserMedia({ 
                video: true 
            });
        } catch (err) {
            console.error("Camera Error:", err);
            alert("Could not start camera. Check permissions or close other apps.");
            return;
        }
    }

    // 2. Attach to UI
    attachStreamToVideo();

    // 3. Open WebSocket
    if (!socket || socket.readyState === WebSocket.CLOSED) {
        socket = new WebSocket(socketUrl);
        setupSocket();
    }
}

function attachStreamToVideo() {
    const videoElement = document.getElementById("webcam-video");
    if (videoElement && currentStream) {
        if (videoElement.srcObject !== currentStream) {
            videoElement.srcObject = currentStream;
            videoElement.play().catch(e => console.log("Play handled:", e));
        }
    } else {
        // Retry if React hasn't rendered the element yet
        setTimeout(attachStreamToVideo, 100);
    }
}

function setupSocket() {
    const canvasElement = document.createElement("canvas");
    
    socket.onopen = () => {
        console.log("WS Open");
        if (streamInterval) clearInterval(streamInterval);

        streamInterval = setInterval(() => {
            const videoElement = document.getElementById("webcam-video");
            
            // STRICT CHECK: Ensure video is actually playing and has dimensions
            if (socket.readyState === WebSocket.OPEN && 
                videoElement && 
                videoElement.readyState === 4 && 
                videoElement.videoWidth > 10) {
                
                 canvasElement.width = videoElement.videoWidth;
                 canvasElement.height = videoElement.videoHeight;
                 
                 const ctx = canvasElement.getContext("2d");
                 ctx.drawImage(videoElement, 0, 0);
                 
                 // Compress to 0.5 quality to save bandwidth
                 const dataURL = canvasElement.toDataURL("image/jpeg", 0.5);
                 socket.send(JSON.stringify({ "frame": dataURL }));
            } else {
                // Keep trying to attach if loose
                attachStreamToVideo();
            }
        }, 200); // 5 FPS
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateVisuals(data);
    };
    
    socket.onerror = (e) => console.error("WS Error:", e);
}

function stopStream() {
    if (streamInterval) clearInterval(streamInterval);
    if (socket) socket.close();
    
    // Stop hardware
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    
    const v = document.getElementById("video-container");
    if(v) v.style.borderColor = "gray";
}

function updateVisuals(data) {
    // 1. Update CSS Border (Instant feedback)
    const vid = document.getElementById("video-container");
    if (vid) {
        let color = "gray";
        if(data.status === "safe") color = "#38A169"; 
        if(data.status === "warning") color = "#DD6B20"; 
        if(data.status === "critical") color = "#E53E3E"; 
        vid.style.borderColor = color;
    }

    // 2. Send to Reflex State (Hidden Bridge)
    const bridge = document.getElementById("js-data-bridge");
    if (bridge) {
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
        nativeSetter.call(bridge, JSON.stringify(data));
        bridge.dispatchEvent(new Event('input', { bubbles: true }));
    }
}