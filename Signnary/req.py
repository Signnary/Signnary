import requests

# Define the URL of your Flask API endpoint
url = "http://127.0.0.1:5000/predict"

# Open the image file
with open(r"D:\Bangkit\Caps\Signnary\testJ1.jpeg", "rb") as file:
    # Define the files to be sent in a dictionary
    files = {"file": file}

    # Send the POST request with the image file
    response = requests.post(url, files=files)

# Print the response
print(response.json())
