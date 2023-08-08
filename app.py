import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from torch import nn

# Initialize the Flask app
app = Flask(__name__)

# Define the image transformations to be applied
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the input image to 224 x 224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Load the pre-trained VGG16 model
trained_vgg_model = models.vgg11()

# Modify the model's final layer to have the correct number of output classes
num_classes = 10
trained_vgg_model.classifier[-1] = nn.Linear(trained_vgg_model.classifier[-1].in_features, num_classes)

# Set the model to evaluation mode and load the trained weights
trained_vgg_model.load_state_dict(torch.load("data/trained_vgg_model_quantized.pt"))
trained_vgg_model.eval()

# Define the class labels
CLASSES = [
    'Abstract_Expressionism',
    'Art_Nouveau_Modern',
    'Baroque',
    'Expressionism',
    'Impressionism',
    'Northern_Renaissance',
    'Post_Impressionism',
    'Realism',
    'Romanticism',
    'Symbolism'
]


# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if request.files.get('image'):  # Check if an image file was uploaded
        # Preprocess the image and pass it through the model
        image = Image.open(request.files['image'].stream).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = trained_vgg_model(input_tensor)

        # Get the probabilities for all classes
        probabilities = torch.softmax(output, dim=1)[0].tolist()
        class_probabilities = {CLASSES[class_id]: prob for class_id, prob in enumerate(probabilities)}
        sorted_probabilities = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)

        return jsonify(sorted_probabilities)

    return jsonify({'error': 'No image provided'})  # Return an error message if no image file was uploaded


if __name__ == '__main__':
    app.run()  # Run the Flask app if this module is executed directly (not imported)
