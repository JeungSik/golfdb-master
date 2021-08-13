from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds


def eval(model, split, seq_length, n_cpu, disp, device):
    dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    np.set_printoptions(linewidth=np.inf)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        labels = labels[:, seq_length - 1:]

        batch = 0

        # while batch * seq_length < images.shape[1]:
        while batch + seq_length <= images.shape[1]:
            # if (batch + 1) * seq_length > images.shape[1]:
            #    image_batch = images[:, batch * seq_length:, :, :, :]
            # else:
            #    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            image_batch = images[:, batch : batch + seq_length, :, :, :]
            image_batch = image_batch.squeeze()

            firstlast_img = [image_batch[0, :, :, :].tolist(), image_batch[-1, :, :, :].tolist()]
            image_tensor = torch.tensor(firstlast_img)
            image_tensor = image_tensor.unsqueeze(0)


            # logits = model(image_batch.cuda())
            #logits = model(image_batch).to(device)

            logits = model(image_tensor).to(device)

            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        l, p, c = correct_preds(probs, labels.squeeze())
        if disp:
            print('Sample {} ..............................................................................'.format(i))
            print('Labels:\t{}'.format(str(l).replace(' ', '')))
            print('Probs:\t{}'.format(str(p).replace(' ', '')))
            print('Result:\t{}'.format(str(c).replace(' ', '')))
        correct.append(np.mean(c))
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    split = 1
    seq_length = 4
    n_cpu = 6

    rnn_layers = 1
    rnn_hidden = 256

    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=rnn_layers,
                          lstm_hidden=rnn_hidden,
                          bidirectional=False,
                          dropout=False,
                          device=device)

    save_dict = torch.load('models/swingnet_10000.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    # model.cuda()
    model.to(device)

    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True, device)
    print('Average PCE: {}'.format(PCE))


