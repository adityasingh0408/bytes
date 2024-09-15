// Function to handle file upload
function handleFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    // Check if a file is selected
    if (!file) {
        alert('Please select a file.');
        return;
    }
    
    // Display a loading message
    const loadingMessage = document.getElementById('loadingMessage');
    loadingMessage.innerText = 'Uploading file...';
    
    // Create a FormData object to send the file to the server
    const formData = new FormData();
    formData.append('file', file);
    
    // Send a POST request to the server
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide the loading message
        loadingMessage.innerText = '';

        // Display the detection result
        const resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '';
        
        const resultText = document.createElement('p');
        resultText.innerText = `Detection result: ${data.result}`;
        resultContainer.appendChild(resultText);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while uploading the file.');
        loadingMessage.innerText = '';
    });
}

// Function to open the laptop camera
function openCamera() {
    // Access the user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            const videoElement = document.createElement('video');
            videoElement.srcObject = stream;
            videoElement.autoplay = true;
            videoElement.setAttribute('playsinline', ''); // For Safari
            videoElement.style.position = 'fixed';
            videoElement.style.top = '40%';
            videoElement.style.left = '70%';
            videoElement.style.transform = 'translate(-50%, -50%)';
            document.body.appendChild(videoElement);

            // Hide the main container
            const mainContainer = document.getElementById('mainContainer');
            mainContainer.style.display = 'none';
        })
        .catch(error => {
            console.error('Error accessing camera:', error);
            // alert('An error occurred while accessing the camera.');
        });
}

function previewImage() {
    const preview = document.getElementById('preview');
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            preview.classList.add('front-image');
        }
        reader.readAsDataURL(file);
    } else {
        preview.style.display = null;
    }
}

function deleteImage() {
    const preview = document.getElementById('preview');
    preview.src = '#';
    preview.style.display = 'none';
    preview.classList.remove('front-image');
    const fileInput = document.getElementById('fileInput');
    fileInput.value = ''; // Clear the file input
}