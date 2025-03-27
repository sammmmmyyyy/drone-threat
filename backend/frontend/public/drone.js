function uploadImage() {
    let input = document.getElementById("imageUpload");
    if (input.files.length === 0) {
        alert("Please select an image!");
        return;
    }

    let formData = new FormData();
    formData.append("image", input.files[0]);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.result_image) {
            document.getElementById("resultImage").src = "http://127.0.0.1:5000" + data.result_image;
        } else {
            alert("Detection failed!");
        }
    })
    .catch(error => console.error("Error:", error));
}
