import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def trainer(net, train_iter, test_iter, num_epochs, lr, device, writer_path=None, save_path=None):
    print(f'---------- Training on {device} ----------')
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    if writer_path is not None:
        writer = SummaryWriter(writer_path)

    best_acc = 0.0
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc = [], []
        for x, y in tqdm(train_iter):
            x, y = x.to(device), y.to(device)
            y_hat = net(x)

            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_hat = y_hat.argmax(axis=1)
            acc = (y_hat.type(y.dtype) == y).float().mean()
            train_loss.append(loss.item())
            train_acc.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)
        print(f'[ Train | epoch: {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}')

        net.eval()
        valid_loss, valid_acc = [], []
        with torch.no_grad():
            for x, y in tqdm(test_iter):
                x, y = x.to(device), y.to(device)
                y_hat = net(x)
                loss = loss_function(y_hat, y)
                y_hat = y_hat.argmax(axis=1)
                acc = (y_hat.type(y.dtype) == y).float().mean()
                valid_loss.append(loss.item())
                valid_acc.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_acc) / len(valid_acc)
        print(f"[ Valid | epoch: {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if writer_path is not None:
            writer.add_scalars('loss', {'train': train_loss,
                                        'valid': valid_loss}, epoch + 1)
            writer.add_scalars('acc', {'train': train_acc,
                                       'valid': valid_acc}, epoch + 1)

        if save_path is not None and valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(net.state_dict(), save_path)
            print('Saving model with acc {:.3f}'.format(best_acc))

    if writer_path is not None:
        writer.close()
