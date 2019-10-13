import torch,json
from torch import optim
from torchvision import datasets, transforms,models
import torch.nn as nn
import matplotlib.pyplot as plt

class ModelTrainer:

    def __init__(self,learning_rate):
        self.normalization_transform = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalization_transform,])

        self.test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalization_transform,])

        self.train_dataset = datasets.ImageFolder("dog-breed-identification/organized-data/train", transform=self.train_transforms)
        self.test_dataset = datasets.ImageFolder("dog-breed-identification/organized-data/test", transform=self.test_transforms)
        self.cnn_model = models.resnet152(pretrained=True)
        self.freeze_model_parameters()
        self.fully_connected_layer = nn.Sequential(
                                       nn.Linear(2048,800),
                           nn.ReLU(),
                           nn.Dropout(p=0.35),
                           nn.Linear(800,120),
                           nn.ReLU(),
                           nn.LogSoftmax(dim=1))
        self.loss_function = nn.NLLLoss()
        self.cnn_model.fc = self.fully_connected_layer
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.cnn_model.fc.parameters(), lr=self.learning_rate)
        self.training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=True)
        self.cnn_model.to(self.training_device)
        self.results = {}

    def freeze_model_parameters(self):
        for parameter in self.cnn_model.parameters():
            parameter.requires_grad = False

    def train_model(self,epochs):
            for e in range(epochs):
                self.cnn_model.train()
                total_training_loss = 0
                accuracy = 0
                total_test_loss = 0
                for images,labels in self.train_loader:
                    images,labels = images.to(self.training_device),labels.to(self.training_device)
                    output = self.cnn_model.forward(images)
                    training_loss = self.loss_function(output,labels)
                    training_loss.backward()
                    training_loss_value = training_loss.item()
                    self.optimizer.step()
                    total_training_loss += training_loss.item()
                    print("Loss this iteration", training_loss_value, "current epoch ",e)

                with torch.no_grad():
                    self.cnn_model.eval()
                    for images,labels in self.test_loader:
                        images,labels = images.to(self.training_device),labels.to(self.training_device)
                        answers = self.cnn_model(images)
                        probabilities = torch.exp(answers)
                        test_loss = self.loss_function(answers,labels)
                        test_loss_value = test_loss.item()
                        total_test_loss += test_loss_value
                        highest_probability,predicted_class = probabilities.topk(1,dim=1)
                        equals = predicted_class == labels.view(*predicted_class.shape)
                        accuracy_tensor = torch.mean(equals.type(torch.FloatTensor))/len(self.test_loader)
                        accuracy += accuracy_tensor.item()
                        print('validation loss: %s' %(str(test_loss_value)))
                    print("accuracy %s for epoch: %s" %(accuracy,e))
                self.save_results(epoch=e,
                                  test_loss=total_test_loss/len(self.test_loader),
                                  validation_loss=total_training_loss/len(self.train_loader),
                                  accuracy=accuracy)
                self.save_model(e)


    def save_results(self,epoch,test_loss,validation_loss,accuracy):
        self.results[epoch] = {
            "validationLoss":validation_loss,
            "testLoss":test_loss,
            "accuracy":accuracy
        }
        with open('results/resnet-152-800-002.json', 'w') as outfile:
            json.dump(self.results, outfile)

    def save_model(self,epoch):
        print("Saving model")
        model_file_name = "trained-cnn-epoch-%s.pth" %(epoch)
        torch.save(self.cnn_model.state_dict(),model_file_name)

    def plot_results(self):
        epochs = []
        training_losses = []
        validation_losses = []
        accuracies = []
        with open('results/resnet-152-800-002.json', 'rb') as results_file:
            results = json.loads(results_file.read())
            for epoch in results:
                epochs.append(epoch)
                training_losses.append(results[epoch]["testLoss"])
                validation_losses.append(results[epoch]["validationLoss"])
                accuracies.append(results[epoch]["accuracy"])
        plt.plot(epochs,training_losses,label='train loss')
        plt.plot(epochs,validation_losses,label='test loss')
        #plt.plot(epochs,accuracies,label='accuracy')
        plt.xlabel('epochs')
        plt.ylabel('value')
        plt.legend()
        plt.show()

dog_breed_classifier = ModelTrainer(learning_rate=0.0015)
#dog_breed_classifier.train_model(epochs=30)
dog_breed_classifier.plot_results()
