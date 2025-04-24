import torch
import timm
from torchvision import transforms
from PIL import Image

# Load model
def load_model(model_path, num_classes=5):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Kelas
class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Prediksi bentuk wajah dari gambar
def predict_face_shape(model, image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Tambah batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return class_names[pred]
