import torchvision

resnet = torchvision.models.resnet18(pretrained=True)
print(resnet)
