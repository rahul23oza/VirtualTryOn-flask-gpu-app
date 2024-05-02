// script.js
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const feedback = document.getElementById('upload-feedback');
    feedback.textContent = 'Uploading image...';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const uploadContainer = document.getElementById('upload-container');
        const selectObjectContainer = document.getElementById('select-object-container');

        uploadContainer.style.display = 'none';
        selectObjectContainer.style.display = 'block';

        const uploadedImage = doc.querySelector('img');
        const canvas = document.getElementById('select-object-canvas');
        const context = canvas.getContext('2d');
        canvas.width = uploadedImage.width;
        canvas.height = uploadedImage.height;
        context.drawImage(uploadedImage, 0, 0);

        const selectObjectButton = document.getElementById('select-object-button');
        const selectObjectFeedback = document.getElementById('select-object-feedback');

        let inputPoints = [];
        canvas.addEventListener('click', function(event) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            inputPoints.push([x, y]);
            context.beginPath();
            context.arc(x, y, 5, 0, 2 * Math.PI);
            context.fillStyle = 'red';
            context.fill();
        });

        selectObjectButton.addEventListener('click', function() {
            selectObjectFeedback.textContent = 'Selecting object...';
            fetch('/select_object', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ input_points: inputPoints })
            })
            .then(response => response.text())
            .then(html => {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                const maskImagesContainer = document.getElementById('mask-images-container');
                maskImagesContainer.innerHTML = '';
                const maskImages = doc.querySelectorAll('#mask-images-container img');
                maskImages.forEach(maskImage => {
                    maskImagesContainer.appendChild(maskImage);
                });
                maskImagesContainer.style.display = 'block';
                selectObjectFeedback.textContent = '';
            })
            .catch(error => {
                selectObjectFeedback.textContent = `Error: ${error}`;
            });
        });
    })
    .catch(error => {
        feedback.textContent = `Error: ${error}`;
    });
});

function showSelectObjectContainer() {
    const uploadContainer = document.getElementById('upload-container');
    const selectObjectContainer = document.getElementById('select-object-container');
    uploadContainer.style.display = 'none';
    selectObjectContainer.style.display = 'block';
}