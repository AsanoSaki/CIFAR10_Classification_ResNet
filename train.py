import numpy as np
import yaml
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from data.dataset import CifarDataset
from models.ResNet34 import PretrainedResNet34, ResNet34
from util.trainer import trainer

if __name__ == '__main__':
    with open('configs/CIFAR10.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(cfg)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    train_trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((96, 96)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

    dataset = CifarDataset(csv_file=cfg['DATASET']['TRAIN_LABELS_PATH'], root_dir=cfg['DATASET']['TRAIN_DATASET_PATH'], transform=trans)

    indexes = list(range(len(dataset)))
    train_indexes, val_indexes = train_test_split(indexes, train_size=cfg['DATASET']['TRAIN_RATIO'], random_state=1)
    train_dataset = Subset(dataset, train_indexes)
    val_dataset = Subset(dataset, val_indexes)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['DATALOADER']['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg['DATALOADER']['BATCH_SIZE'], shuffle=False)

    net = PretrainedResNet34(cfg['DATASET']['CATEGORY_NUM'])
    # print(net)

    trainer(net, train_loader, val_loader, cfg['TRAINER']['EPOCHS'], cfg['TRAINER']['LR'],
            cfg['TRAINER']['DEVICE'], cfg['TRAINER']['WRITE_PATH'], cfg['TRAINER']['SAVE_PATH'])