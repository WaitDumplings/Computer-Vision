import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm

from model import vgg


def main():
    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {} for this project'.format(device))

    # Set ImageLoader
    data_root = '/Users/maojietang/Downloads/Image_Segmentation-main/deep-learning-for-image-processing-master'
    image_path = os.path.join(data_root, 'data_set', 'flower_data')

    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    #     "val": transforms.Compose([transforms.Resize((224, 224)),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    train_set = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transform['train'])
    train_num = len(train_set)

    class_ = train_set.class_to_idx
    class_dict = dict((val, key) for key, val in class_.items())

    json_label = json.dumps(class_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_label)

    # val_set
    val_set = datasets.ImageFolder(os.path.join(image_path, 'val'), transform=data_transform['val'])
    val_num = len(val_set)

    batch_size =32
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=0)
    model = vgg(model_name='vgg11', num_classes=5, init_weights=True)
    model.to(device)
    #
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    #
    epochs = 3
    train_bar_count = len(train_loader)
    best_acc = 0.0
    save_path = './VGG.pth'
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            img, label = data
            optimizer.zero_grad()
            outputs = model(img.to(device))
            loss = loss_function(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('train epoch[{}/{}] loss:{:.3f}'.format(epoch + 1, epochs, running_loss))

        model.eval()
        acc = 0.0
        with torch.no_grad():
            for data in val_loader:
                val_img, val_label = data
                val_outputs = model(val_img.to(device))
                predict_y = torch.max(val_outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_label.to(device)).sum().item()
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_bar_count, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
    print('Finish Training')


if __name__ == '__main__':
    main()
