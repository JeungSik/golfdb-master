from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os


if __name__ == '__main__':

    # training configuration --------------------------------------------
    split = 1
    iterations = 10000
    it_save = 100   # save model every 100 iterations
    n_cpu = 6
    seq_length = 4
    batch_size = 24         # batch size
    k = 10                  # frozen layers
    rnn_layers = 1
    rnn_hidden = 256

    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    # create model -----------------------------------------------------
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=rnn_layers,
                          lstm_hidden=rnn_hidden,
                          bidirectional=False,
                          dropout=True,
                          device=device)
    freeze_layers(k, model)
    model.train()
    model.to(device)
    # model.cuda()

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    # weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    weights = torch.FloatTensor([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    # load dataset -----------------------------------------------------
    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([
                         ToTensor(),
                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ]),
                     train=True)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for sample in data_loader:
            # images, labels = sample['images'].cuda(), sample['labels'].cuda()
            images, labels = sample['images'].to(device), sample['labels'].to(device)
            # labels = labels.view(batch_size*seq_length)
            # seq_length의 마지막 label만 적용
            # labels = labels[:, seq_length-1:]
            labels = labels.view(batch_size)

            logits = model(images).to(device)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLabel: {}\t{loss.val:.4f} ({loss.avg:.4f})'
                      .format(i, labels, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break
