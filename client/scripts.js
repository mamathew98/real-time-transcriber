
let websocket;
let context;
let processor;
let globalStream;

const websocket_uri = 'ws://localhost:8765';
const bufferSize = 4096;
let isRecording = false;

function initWebSocket() {
    const websocketAddress = document.getElementById('websocketAddress').value;
    const selectedLanguage = document.getElementById('languageSelect').value;
    const outputLanguage = document.getElementById('outputLanguageSelect').value;
    const translate = document.getElementById('translateCheckbox').checked;
    const speechTranslation = document.getElementById('speechTranslationCheckbox').checked;
    language = selectedLanguage !== 'multilingual' ? selectedLanguage : null;
    
    if (!websocketAddress) {
        console.log("WebSocket address is required.");
        return;
    }

    websocket = new WebSocket(websocketAddress);
    websocket.onopen = () => {
        console.log("WebSocket connection established");
        document.getElementById("webSocketStatus").textContent = 'Connected';
        document.getElementById('startButton').disabled = false;

        // Send initial config to the server
        sendAudioConfig(outputLanguage, translate, speechTranslation);
    };
    websocket.onclose = event => {
        console.log("WebSocket connection closed", event);
        document.getElementById("webSocketStatus").textContent = 'Not Connected';
        document.getElementById('startButton').disabled = true;
        document.getElementById('stopButton').disabled = true;
    };
    websocket.onmessage = event => {
        console.log("Message from server:", event.data);
        const transcript_data = JSON.parse(event.data);
        if (transcript_data['type'] && transcript_data['type'] == 'summary'){
            updateSummary(transcript_data)
        } else if (transcript_data['type'] && transcript_data['type'] == 'audio'){
            playAudio(transcript_data['data'])
        } else {
            updateTranscription(transcript_data);

        }
    };
}

function playAudio(audioData) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    audioContext.decodeAudioData(audioData, buffer => {
        if (source) {
            source.stop();
        }
        source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);
        source.start(0);
    }, error => console.error('Error decoding audio data', error));
}

function updateTranscription(transcript_data) {
    const transcriptionDiv = document.getElementById('transcription');

    transcriptionDiv.textContent += transcript_data['text'] + '\n';

}

function updateSummary(transcript_data) {
    const summaryDiv = document.getElementById('summary');

    summaryDiv.textContent = transcript_data['text'] ;
}


function startRecording() {
    if (isRecording) return;
    isRecording = true;

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    context = new AudioContext();

    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        globalStream = stream;
        const input = context.createMediaStreamSource(stream);
        processor = context.createScriptProcessor(bufferSize, 1, 1);
        processor.onaudioprocess = e => processAudio(e);
        input.connect(processor);
        processor.connect(context.destination);

        sendAudioConfig(document.getElementById('outputLanguageSelect').value, document.getElementById('translateCheckbox').checked);

    }).catch(error => console.error('Error accessing microphone', error));

    // Disable start button and enable stop button
    document.getElementById('startButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    if (globalStream) {
        globalStream.getTracks().forEach(track => track.stop());
    }
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (context) {
        context.close().then(() => context = null);
    }
    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
}

function sendAudioConfig() {


    const audioConfig = {
        type: 'config',
        data: {
            sampleRate: context.sampleRate,
            bufferSize: bufferSize,
            channels: 1, // Assuming mono channel
            language: language,
            outputLanguage: document.getElementById('outputLanguageSelect').value,
            translate: document.getElementById('translateCheckbox').checked,
            speechTranslation: document.getElementById('speechTranslationCheckbox').checked
        }
    };

    websocket.send(JSON.stringify(audioConfig));
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
    if (inputSampleRate === outputSampleRate) {
        return buffer;
    }
    var sampleRateRatio = inputSampleRate / outputSampleRate;
    var newLength = Math.round(buffer.length / sampleRateRatio);
    var result = new Float32Array(newLength);
    var offsetResult = 0;
    var offsetBuffer = 0;
    while (offsetResult < result.length) {
        var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        var accum = 0, count = 0;
        for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

function processAudio(e) {
    const inputSampleRate = context.sampleRate;
    const outputSampleRate = 16000; // Target sample rate

    const left = e.inputBuffer.getChannelData(0);
    const downsampledBuffer = downsampleBuffer(left, inputSampleRate, outputSampleRate);
    const audioData = convertFloat32ToInt16(downsampledBuffer);
    
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(audioData);
    }
}

function summarizeText() {
    const transcriptionDiv = document.getElementById('transcription');
    const transcriptionText = transcriptionDiv.textContent;

    const summaryRequest = {
        type: 'summarize',
        data: {
            text: transcriptionText
        }
    };

    websocket.send(JSON.stringify(summaryRequest));
}

function convertFloat32ToInt16(buffer) {
    let l = buffer.length;
    const buf = new Int16Array(l);
    while (l--) {
        buf[l] = Math.min(1, buffer[l]) * 0x7FFF;
    }
    return buf.buffer;
}

// Initialize WebSocket on page load
//  window.onload = initWebSocket;


