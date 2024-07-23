from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torch.nn import TripletMarginLoss

class SiameseNetwork(nn.Module):
    """
    Defines the Siamese Neural Network Class
    """
    def __init__(self, num_classes, embedding_dim=128):
        super().__init__()

        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(2048, embedding_dim)  # ResNet101 has 2048 features in the last layer
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_once(self, x):
        # x shape: (batch_size, 1024, 2)
        # Reshape input to (batch_size, channels, height, width)
        x = x.permute(0, 2, 1).unsqueeze(2)  # Shape: (batch_size, 2, 1, 1024)

        # Pass through ResNet
        x = self.resnet(x)
        x = x.view(x.size(0), -1)

        # Get embedding
        x = self.fc(x)

        # Normalize embedding
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)
        anchor_class = self.classifier(anchor_out)
        return anchor_out, positive_out, negative_out, anchor_class

    def train_epoch(self, args, model, device, train_loader, optimizer, epoch):
        model.train()
        triplet_criterion = TripletMarginLoss(margin=1, p=2)
        classification_criterion = nn.CrossEntropyLoss()

        for batch_idx, (anchor, positive, negative, labels) in enumerate(train_loader):
            anchor, positive, negative, labels = (anchor.to(device), positive.to(device),
                                                  negative.to(device), labels.to(device))
            optimizer.zero_grad()
            anchor_out, positive_out, negative_out, anchor_class = model(anchor, positive, negative)
            triplet_loss = triplet_criterion(anchor_out, positive_out, negative_out)
            classification_loss = classification_criterion(anchor_class, labels)
            loss = triplet_loss + classification_loss  # Use both losses
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(anchor)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                print(f'Triplet Loss: {triplet_loss.item():.6f}, '
                      f'Classification Loss: {classification_loss.item():.6f}')
                if args.dry_run:
                    break

    def test(self, model: nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader):
        model.eval()
        test_loss = 0
        triplet_correct = 0
        classification_correct = 0
        total_samples = 0
        triplet_criterion = TripletMarginLoss(margin=1, p=2)
        classification_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for anchor, positive, negative, labels in test_loader:
                anchor, positive, negative, labels = (anchor.to(device), positive.to(device),
                                                      negative.to(device), labels.to(device))
                anchor_out, positive_out, negative_out, anchor_class = model(anchor, positive, negative)

                triplet_loss = triplet_criterion(anchor_out, positive_out, negative_out)
                classification_loss = classification_criterion(anchor_class, labels)
                test_loss += triplet_loss.item() + classification_loss.item()

                dist_positive = torch.nn.functional.pairwise_distance(anchor_out, positive_out)
                dist_negative = torch.nn.functional.pairwise_distance(anchor_out, negative_out)
                triplet_correct += (dist_positive < dist_negative).sum().item()
                classification_correct += (anchor_class.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

        test_loss /= len(test_loader.dataset)
        triplet_accuracy = 100. * triplet_correct / total_samples
        classification_accuracy = 100. * classification_correct / total_samples

        print(f'\nTest set: Average loss: {test_loss:.4f}')
        print(f'Triplet Accuracy: {triplet_correct}/{total_samples} ({triplet_accuracy:.2f}%)')
        print(f'Classification Accuracy: {classification_correct}/{total_samples} '
              f'({classification_accuracy:.2f}%)\n')