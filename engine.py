from utils.tools import AverageMeter, ProgressMeter, cal_acc
import torch.nn.functional as F
import torch
from model.loss import get_retrieval_loss


def train_model(args, train_loader, model, optimizer, epoch, device):
    re_losses = AverageMeter('Retrieval Loss', ':.4')
    qtt = AverageMeter('Quantity Loss', ':.4')
    progress = ProgressMeter(len(train_loader),
                             [re_losses, qtt],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    for i, (data, label, image_root) in enumerate(train_loader):
        images = data.to(device, dtype=torch.float32, non_blocking=True)
        labels = label.to(device, dtype=torch.int64, non_blocking=True)
        # print(labels)
        # print(root)

        pred, feature = model(images)
        retri_loss, quantity_loss = get_retrieval_loss(pred, labels, args.num_classes, device)
        total_loss = retri_loss + args.bias * quantity_loss

        re_losses.update(retri_loss)
        qtt.update(quantity_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)


# @torch.no_grad()
# def evaluation(args, val_loader, model, device):
#     model.eval()
#     entropy_losses = 0
#     L = len(val_loader)
#     preds_record = []
#     labels_record = []
#     for i, batch in enumerate(val_loader):
#         images = batch['images'].to(device, dtype=torch.float32, non_blocking=True)
#         labels = batch['labels'].to(device, dtype=torch.int64, non_blocking=True)
#         # b = labels.shape[0]
#         # labels = labels.unsqueeze(-1).expand(b, p["num_slot"])
#
#         pred = model(images)
#         entropy_losses += F.nll_loss(F.log_softmax(pred, dim=1), labels).item()
#         preds_record.append(pred)
#         labels_record.append(labels)
#
#     pred = torch.cat(preds_record, dim=0)
#     labels = torch.cat(labels_record, dim=0)
#     acc = cal_acc(pred, labels)
#     return acc, round(entropy_losses/L, 4), labels, pred