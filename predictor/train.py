import time
import torch
import torch.distributed as dist
import sys
import pandas as pd

best_loss = 10000000


def train(cfg, model, datamodule, optimizer, scheduler, hydra_dir, best_loss_old=None):
    global best_loss
    train_loss_epoch = []
    val_loss_epoch = []
    train_index_epoch = []
    val_index_epoch = []
    best_change = 0

    if best_loss_old != None:
        best_loss = best_loss_old

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()
    test_loaders = datamodule.test_dataloader()

    for epoch in range(cfg.train.start_epochs, cfg.train.max_epochs):
        best_change += 1
        sys.stdout.flush()

        train_loss, train_index = train_step(cfg, model, train_loader, optimizer, epoch)
        val_losses, val_indexes = val_step(cfg, model, val_loaders, optimizer, epoch)
        scheduler.step(metrics=val_losses[0].avg)

        # save model in the training
        if cfg.accelerator == 'DDP':
            loss = torch.tensor(val_losses[0].avg).cuda()
            torch.cuda.synchronize()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            if torch.distributed.get_rank() == 0:
                world_size = torch.cuda.device_count()
                loss_avg = loss / world_size
                if (loss_avg > best_loss):
                    filename = 'model_' + cfg.expname + '_notbest' + '.pth'
                    save_model(hydra_dir, filename, model.module.state_dict(), epoch, loss_avg)
                else:
                    best_loss = loss_avg
                    best_change = 0
                    dist.broadcast(best_change, src=0)
                    filename = 'model_' + cfg.expname + '.pth'
                    save_model(hydra_dir, filename, model.module.state_dict(), epoch, loss_avg)
                    print('save model with loss = ', loss_avg, file=sys.stdout)

                    loss_dict = {
                        'train_loss_epoch': train_loss_epoch,
                        'val_loss_epoch': val_loss_epoch,
                        'train_index_epoch': train_index_epoch,
                        'val_index_epoch': val_index_epoch,
                    }
                    loss_df = pd.DataFrame(loss_dict)
                    excel_file = 'loss_file_' + cfg.expname + '.xlsx'
                    excel_file = hydra_dir / excel_file
                    loss_df.to_excel(excel_file, index=False)

        else: #do not use DDP
            if (val_losses[0].avg < best_loss):
                best_loss = val_losses[0].avg
                best_change = 0
                filename = 'model_' + cfg.expname + '.pth'
                save_model(hydra_dir, filename, model.state_dict(), epoch, val_losses[0].avg)
                print('save model with loss = ', best_loss, file=sys.stdout)

                loss_dict = {
                    'train_loss_epoch': train_loss_epoch,
                    'val_loss_epoch': val_loss_epoch,
                    'train_index_epoch': train_index_epoch,
                    'val_index_epoch': val_index_epoch,
                }
                loss_df = pd.DataFrame(loss_dict)
                excel_file = 'loss_file_' + cfg.expname + '.xlsx'
                excel_file = hydra_dir / excel_file
                loss_df.to_excel(excel_file, index=False)

            else:
                filename = 'model_' + cfg.expname + '_notbest' + '.pth'
                save_model(hydra_dir, filename, model.state_dict(), epoch, val_losses[0].avg)


        train_loss_epoch.append(train_loss.avg.cpu().detach().numpy())
        val_loss_epoch.append(val_losses[0].avg.cpu().detach().numpy())
        train_index_epoch.append(train_index.avg.cpu().detach().numpy())
        val_index_epoch.append(val_indexes[0].avg.cpu().detach().numpy())

        if best_change > cfg.train.most_patience:
            break

    test_losses = val_step(cfg, model, test_loaders, optimizer, epoch, prefix='test')
    if cfg.accelerator == 'DDP':
        if torch.distributed.get_rank() == 0:
            loss_dict = {
                'train_loss_epoch': train_loss_epoch,
                'val_loss_epoch': val_loss_epoch,
                'train_index_epoch': train_index_epoch,
                'val_index_epoch': val_index_epoch,
            }
            loss_df = pd.DataFrame(loss_dict)
            excel_file = 'loss_file_' + cfg.expname + '.xlsx'
            excel_file = hydra_dir / excel_file
            loss_df.to_excel(excel_file, index=False)
    else:
        loss_dict = {
            'train_loss_epoch': train_loss_epoch,
            'val_loss_epoch': val_loss_epoch,
            'train_index_epoch': train_index_epoch,
            'val_index_epoch': val_index_epoch,
        }
        loss_df = pd.DataFrame(loss_dict)
        excel_file = 'loss_file_' + cfg.expname + '.xlsx'
        excel_file = hydra_dir / excel_file
        loss_df.to_excel(excel_file, index=False)

    return 0





def train_step(cfg, model, train_loader, optimizer, epoch):
    train_loss = AverageMeter()
    index = AverageMeter()
    batch_time = AverageMeter()

    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):

        if cfg.accelerator != 'cpu':
            batch = batch.cuda()

        outputs = model(batch)
        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss.data.cpu()
        train_loss.update(loss, batch.num_nodes)
        index.update(outputs['index'].data.cpu(), batch.num_nodes)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.train.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time}\t'
                  'Loss {train_loss}\t'
                  'Index {index}'.format(epoch, i, len(train_loader),
                                             batch_time=batch_time,
                                             train_loss=train_loss,
                                             index=index),
                                             file=sys.stdout)
            sys.stdout.flush()
        sys.stdout.flush()

    return train_loss, index


def val_step(cfg, model, val_loaders, optimizer, epoch, prefix='val'):
    # switch to evaluate mode
    model.eval()

    val_losses = []
    indexes = []

    for val_loader in val_loaders:
        val_loss = AverageMeter()
        index = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        for i, batch in enumerate(val_loader):
            if cfg.accelerator != 'cpu':
                batch = batch.cuda()

            outputs = model(batch)
            loss = outputs['loss']

            loss.data.cpu()
            val_loss.update(loss.data.cpu(), batch.num_nodes)
            index.update(outputs['index'].data.cpu(), batch.num_nodes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.train.print_freq == 0:
                print('{3}: [{0}][{1}/{2}]\t'
                      'Time {batch_time}\t'
                      'Loss {val_loss}\t'
                      'Index {index}'.format(epoch, i, len(val_loader), prefix,
                                             batch_time=batch_time,
                                             val_loss=val_loss,
                                             index=index), file=sys.stdout)

        if prefix == 'test':
            print('-----------------Test Result------------------')
            print(f'{prefix}_loss', val_loss)
            print(f'{prefix}_index', index)

        val_losses.append(val_loss)
        indexes.append(index)

    if prefix == 'test':
        return val_losses, indexes
    else:
        return val_losses, indexes

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return ('%.3f(%.3f)' % (self.val, self.avg))

def save_model(hydra_dir, filename, model, epoch, loss_avg):
    path = hydra_dir / filename
    data = {'model': model,
            'epoch': epoch + 1,
            'val_loss': loss_avg}
    torch.save(data, path)