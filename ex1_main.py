import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
import os

from my_models_skeleton import ViT, CrossViT   # rename the skeleton file for your implementation / comment before testing for ResNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='r18', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)/len(output)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


def run(args):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    # Download and load the training data
    transform = transforms.Compose([transforms.ToTensor(),
                                    # ImageNet mean/std values should also fit okayish for CIFAR
									transforms.Normalize(mean=mean, std=std )
                                    ])

    # TODO: adjust folder
    augmentations = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        ])
    #augmentations = transforms.Compose([])
    dataset = datasets.CIFAR10('cifar10', download=True, train=True, transform=transforms.Compose([augmentations, transform]))
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # Download and load the test data
    # TODO: adjust folder
    testset = datasets.CIFAR10('cifar10', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(weights=None, num_classes=10) # resnet18 is designed for 224x224 inputs, cifar is 32x32
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # use 3x kernel with stride 1 instead of 7x kernel with stride 2. keep #feature maps the same
        model.maxpool = nn.Identity() # dont downscale using max pool
    elif args.model == "vit":
        model = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1,
                    emb_dropout = 0.1) 
    elif args.model == "cvit":
        model = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, 
                         lg_dim = 128, sm_patch_size = 8, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 128, 
                         sm_enc_dim_head = 64, lg_patch_size = 16, 
                         lg_enc_depth = 2, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 128, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

    # Define the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")

    def get_device():
        if torch.cuda.is_available():
            print("using cuda")
            return torch.device("cuda")
        try:
            import torch_directml
            if torch_directml.is_available():
                print("using directml")
                return torch_directml.device()
        except ImportError:
            pass
        print("using cpu")
        return torch.device("cpu")
    
    device = get_device()    
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.model == "r18":
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=args.epochs)

    best_val_los = None
    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        val_loss = test(model, device, valloader, criterion, set="Validation")

        if args.model == "r18":
            scheduler.step()

        if best_val_los is None or val_loss < best_val_los:
            best_val_los = val_loss
            print(f"new model with best val loss: {val_loss}")
            if args.save_model:
                print("saving new best model weights")
                WEIGHTS_SAVE_DIR="weights"

                if not os.path.isdir(WEIGHTS_SAVE_DIR):
                    os.mkdir(WEIGHTS_SAVE_DIR)

                torch.save(model.state_dict(), f"{WEIGHTS_SAVE_DIR}/{args.model}_ep{args.epochs}.pt")


    test(model, device, testloader, criterion)

if __name__ == '__main__':
    args = parse_args()
    run(args)
