import os
import json
import sys

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
# from model import resnet34


def main():
    # Set Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} for this project'.format(device))

    # Image_Loader and Label
    # path_root = '/Users/maojietang/Downloads/Image_Segmentation-main/deep-learning-for-image-processing-master'
    path_root = '/content/drive/MyDrive'
    image_path = os.path.join(path_root, 'data_set', 'flower_data')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Get train_set
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform['train'])
    train_num = len(train_dataset)

    # Get label
    category_to_label = train_dataset.class_to_idx
    label_to_category = dict((val, key) for key, val in category_to_label.items())

    # Write to json
    json_label = json.dumps(label_to_category, indent=4)
    with open('class_label.json', 'w') as json_file:
        json_file.write(json_label)

    # Load train_loader / val_loader
    batch_size = 32
    nw = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    batch_num = len(train_loader)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                       transform=data_transform['val'])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    # Model Initial
    model = resnet34(num_classes=5, include_top=True)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Train
    epochs = 3
    best_acc = 0.0
    save_path = './ResNet34.pth'
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        total_loss = 0.0
        for step, data in enumerate(train_bar):
            optimizer.zero_grad()
            img, label = data
            predict = model(img.to(device))
            loss = loss_function(predict, label.to(device))
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print('Epoch[{}/{}] train_loss: %.3f'.format(epoch + 1,
                                                     epochs,
                                                     total_loss))

        model.eval()
        val_bar = tqdm(val_loader, file=sys.stdout)
        acc = 0.0
        with torch.no_grad():
            for data in val_bar:
                val_img, val_label = data
                predict = model(val_img.to(device))
                output = torch.max(predict, dim=1)[1]
                acc += torch.eq(output, val_label.to(device)).sum().item()
        accuracy = acc / val_num
        val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                   epochs)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, total_loss / batch_num, accuracy))

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
    print('Finish Training!')


if __name__ == "__main__":
    main()
