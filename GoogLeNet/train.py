import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

# from model import GoogLeNet

def main():
    # Step one: set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('Using {} for this projec'.format(device))

    # Step two: set data/label
    # transform
    transform = {'train': transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                 'val': transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # path_root = '/content/drive/MyDrive'
    path_root = '/Users/maojietang/Downloads/Image_Segmentation-main/deep-learning-for-image-processing-master'
    image_path = os.path.join(path_root, 'data_set', 'flower_data')

    # trainset/valset
    train_set = datasets.ImageFolder(os.path.join(image_path, 'train'), transform=transform['train'])
    val_set = datasets.ImageFolder(os.path.join(image_path, 'val'), transform=transform['val'])
    train_num = len(train_set)
    val_num = len(val_set)

    # label to json
    category_label = train_set.class_to_idx
    label_category = dict((val, key) for key, val in category_label.items())

    json_content = json.dumps(label_category, indent=4)
    with open('class_category.json', 'w') as json_file:
        json_file.write(json_content)

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               shuffle=True,
                                               batch_size = 32,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             shuffle=False,
                                             batch_size = 10,
                                             num_workers=0)

    # Initial Model
    model = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    model.to(device)
    # Loss/Optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Model_Train
    epochs = 10
    best_acc = 0.0
    train_batch_num = len(train_loader)
    save_path = './GoogLeNet.pth'
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        for step, data in enumerate(train_bar):
            img, label = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = model(img.to(device))
            loss0 = loss_function(logits, label.to(device))
            loss1 = loss_function(aux_logits1, label.to(device))
            loss2 = loss_function(aux_logits2, label.to(device))
            loss = loss0 + 0.3*loss1 + 0.3*loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print train loss
            print('train epoch:[{}/{}] train loss:{:3f}'.format(epoch + 1, epochs, loss))

        # eval
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                val_img, val_label = data
                output = model(val_img.to(device))
                result = torch.max(output, dim=1)[1]
                acc += torch.eq(result, val_label.to(device)).sum().item()
        accuracy = acc/val_num

        # Print
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_batch_num, accuracy))

        if best_acc<accuracy:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)

print('Finish Training')

if __name__ == '__main__':
    main()
