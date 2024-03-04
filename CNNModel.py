import torch
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb
from PIL import Image
import matplotlib.pyplot as plt

class CNNModel:
    def __init__(self,
                 dataset_path,
                 train_size=0.8,
                 test_size=0.1,
                 val_size=0.1,
                 imgsz=(224,224),
                 batch_size = 32,
                 shuffle = True,
                 norm_mean_rgb=(0.485, 0.456, 0.406), 
                 norm_std_rgb=(0.229, 0.224, 0.225),
                 custom_model=None,
                 pretrained_model=models.vgg16,
                 pretrained=True,
                 device="0",
                 use_wandb=True,
                 wandb_project="CNNModels",
                 experiment_name="Test"):

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.imgsz = imgsz
        self.norm_mean_rgb = norm_mean_rgb
        self.norm_std_rgb = norm_std_rgb
        self.device = torch.device("cpu" if device=="cpu" else f"cuda:{device}")
        if not torch.cuda.is_available() and self.device != "cpu":
            raise(f"torch.cuda.is_available(): {torch.cuda.is_available()}, check CUDA or set param device='cpu'")
        if self.use_wandb:
            wandb.init(project=self.wandb_project, name=self.experiment_name)


        self.dataset = self.get_dataset(self.dataset_path)
        self.model = self.create_architecture(custom_model, pretrained_model, pretrained)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        self.num_imgs = len(self.dataset.imgs)

    
    def create_architecture(self, custom_model, pretrained_model_arch, pretrained):
        if custom_model is None and pretrained_model_arch is None:
            raise("Error with model. Please put pretrained_model or custom_model")
        
        if custom_model:
            model = custom_model.to(self.device)
            return model
        
        self.num_classes = len(self.dataset.classes)
        
        # Загрузка предварительно обученной модели
        model = pretrained_model_arch(pretrained=pretrained)
        try:
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, self.num_classes)
        except AttributeError:  # ResNet-50
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        
        # Замораживаем веса всех слоев, кроме последнего
        for param in model.parameters():
            param.requires_grad = False
        try:
            model.classifier[-1].weight.requires_grad = True
            model.classifier[-1].bias.requires_grad = True
        except AttributeError:  # ResNet-50
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True

        # Перемещение модели на GPU
        model = model.to(self.device)

        return model


    
    def train(self, num_epochs=5, batch_size=None, loss=torch.nn.CrossEntropyLoss(), optimizer=torch.optim.SGD, lr=0.001, finish_train=True):
        if batch_size is None:
            batch_size = self.batch_size
            
        train_loader, test_loader, valid_loader = self.get_DataLoader(batch_size)
        optimizer = optimizer(self.model.parameters(), lr=lr)
        criterion = loss
        device = self.device

        for epoch in range(num_epochs):
            self.model.train()  # Установим модель в режим обучения
            current_img = 0
            for images, labels in train_loader:
                current_img += batch_size
                print(f"Training {current_img}/{int(self.num_imgs*self.train_size)} imgs of epoch {epoch+1}")
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss = loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        
            # Оценка на тестовой выборке
            self.model.eval()  # Установим модель в режим оценки
            all_preds, all_labels = [], []
            test_loss = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())
        
            # Вычисление метрик
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')
            accuracy = accuracy_score(all_labels, all_preds)
        
            print(f'Test Metrics - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}')
            if self.use_wandb:
                # Логирование потерь и метрик для тестового набора данных
                wandb.log({"Train Loss": train_loss, 
                           "Test Loss": test_loss / len(test_loader), 
                           "Test Precision": precision, 
                           "Test Recall": recall, 
                           "Test F1": f1, 
                           "Test Accuracy": accuracy})
        if finish_train:
            if self.use_wandb:
                wandb.finish()

    def predict(self, file_path):
        image = Image.open(file_path).convert('RGB')
        transform = self.get_transform()
        input_image = transform(image).unsqueeze(0)
        # Перемещение входного изображения на устройство (GPU или CPU)
        input_image = input_image.to(self.device)
        # Установка модели в режим оценки
        self.model.eval()
        # Предсказание класса
        with torch.no_grad():
            output = self.model(input_image)
            _, predicted_class = torch.max(output, 1)
        # Получение названия предсказанного класса
        class_index = predicted_class.item()
        class_name = self.dataset.classes[class_index]

        return class_name

    def get_DataLoader(self, batch_size, num_workers=4, pin_memory=True):
        train_dataset, test_dataset, valid_dataset = self.split_train_test_val()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=self.shuffle, pin_memory=pin_memory, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        
        return train_loader, test_loader, valid_loader
    
    
    def split_train_test_val(self):
        train_dataset, test_dataset, valid_dataset = random_split(self.dataset, [self.train_size, self.test_size, self.val_size])
        return train_dataset, test_dataset, valid_dataset
        
    
    def get_dataset(self, dataset_path):
        dataset = ImageFolder(root=dataset_path, transform=self.get_transform())
        return dataset

    # функция возвращает трансформер для изображений
    def get_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.imgsz),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean_rgb, self.norm_std_rgb)
        ])
        return transform