import torch,json,io
import torch.nn as nn
from torchvision import transforms,models
from PIL import Image

class Inference:

    def __init__(self,model_file,image_file):
        self.results = None
        self.image_tensor = None
        self.cnn_model = models.resnet152(pretrained=True)
        self.training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fully_connected_layer = nn.Sequential(
                                       nn.Linear(2048,800),
                           nn.ReLU(),
                           nn.Dropout(p=0.35),
                           nn.Linear(800,120),
                           nn.ReLU(),
                           nn.LogSoftmax(dim=1))
        self.set_image_tensor(image_file)
        self.load_cnn_model(model_file)

    def load_cnn_model(self,model_file):
        self.cnn_model.fc = self.fully_connected_layer
        self.cnn_model.load_state_dict(torch.load(model_file))

    def transform_image(self,image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def set_image_tensor(self,image_file):
        with open(image_file, 'rb') as file:
            image_bytes = file.read()
        transformed_image = self.transform_image(image_bytes=image_bytes)
        self.image_tensor = transformed_image

    def get_image_prediction(self):
        self.cnn_model.eval()
        outputs = self.cnn_model.forward(self.image_tensor)
        _, label = outputs.max(1)
        inference_result = {
            "prediction":label.item(),
            "results":outputs.tolist()
        }
        return json.dumps(inference_result)

inf = Inference(image_file='dog-breed-identification/organized-data/train/chihuahua/209d8e99fc9bbdefe264bd4c71200b4b.jpg',model_file='trained-cnn-epoch-18.pth')
print(inf.get_image_prediction())



