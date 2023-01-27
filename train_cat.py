from data.catdataset import CatPresentSingle
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from torchvision.models import resnet50

def main():
    data_path = '/home/nadav.nissim/data/xray/cat1data'
    dataset = CatPresentSingle(data_path, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        for i, (img, label) in enumerate(dataloader):
            img = img.cuda()
            label = label[:, [0]].cuda()
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            num_correct = (out.argmax(dim=1) == label).sum().item()
            print(f"Epoch {epoch}, batch {i}, loss {loss.item()}, accuracy {num_correct / len(label)}")

    # test the model


if __name__ == '__main__':
    main()