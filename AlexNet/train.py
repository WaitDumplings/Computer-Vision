import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    # Step one:  Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))

    # Step two: construct dataholder (need to get data first)
    # From images to tensors (Not in training process)
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = '/Users/maojietang/Downloads/Image_Segmentation-main/deep-learning-for-image-processing-master'
    image_path = os.path.join(data_root, 'data_set', 'flower_data')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # ImageFolder can extract all images from a file
    # Format: Dataset
    # ---------------Category1: image1, image2, ...
    # ---------------Category2: image1, image2, ...
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # train_dataset can get all images in Dataset.
    # For each category, we can get in class_to_idx

    train_num = len(train_dataset)

    # flower_list is a dict: {category_number: category_name}
    flower_list = train_dataset.class_to_idx
    # exchange key and val, and get the dict: {category_name: category_number}
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # set a json_file to store key and val.
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # Step Three: load data for taining
    # set back and number of workers
    batch_size = 32
    nw = 0

    # load train/val data to dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4,
                                                  shuffle=False,
                                                  num_workers=nw)

    print("Using {} images for training, {} images for validation".format(train_num, val_num))

    # import model/net
    net = AlexNet(num_class=5, init_weight=True)

    net.to(device)

    # set loss function
    loss_function = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # set initial parameter
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    # the number of batches
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistcs
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # val
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finish Training")


if __name__ == '__main__':
    main()
