import os
import torch.optim
import configs
from data.datamgr import SimpleDataManager, SetDataManager
from io_utils import parse_args, get_resume_file
from model import *

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def do_learning(net, epoch, optimizer, base_loader,inner_step=5):

    net.train()
    print_freq = 10
    avg_loss = 0

    for i , (x,_) in enumerate(base_loader):
        iter_loss = 0
        for _ in range(inner_step):
            optimizer.zero_grad()
            loss = net.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            iter_loss = iter_loss + loss.item()
        iter_loss = iter_loss / float(inner_step)
        avg_loss = avg_loss + iter_loss

        if i % print_freq == 0 :
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(base_loader), avg_loss / float(i + 1)))
    return avg_loss


def do_evaluation(meta_net, test_loader, n_spt, n_qry, inner_step=5):
    acc_all = []
    iter_num = len(test_loader)
    net_clone = meta_net.clone()
    optimizer_clone = get_optimizer(net_clone, state)
    for i , (x,_) in enumerate(test_loader):
        x = x.split(n_spt+n_qry, dim=1)
        x_spt = x[0]
        x_qry = x[1]
        net = net_clone
        optimizer = optimizer_clone
        for _ in range(inner_step):
            optimizer.zero_grad()
            loss = net.set_forward_loss(x_spt)
            loss.backward()
            optimizer.step()


        correct_this, count_this = net.correct(x_qry)
        acc_all.append(correct_this / count_this * 100)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    a = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num))

    return acc_mean, a




def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer

if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)

    meta_lr = 1.
    image_size = 84


    if params.stop_epoch == -1:
        if params.n_shot == 1:
            params.stop_epoch = 200
        elif params.n_shot == 5:
            params.stop_epoch = 100
        else:
            params.stop_epoch = 100  # default

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file   = configs.data_dir[params.dataset] + 'novel.json'


    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, mode='test')
    val_datamgr = SetDataManager(image_size, n_eposide=600, n_query=n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    meta_net = model(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)
    meta_net = meta_net.cuda()
    meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=meta_lr)

    params.checkpoint_dir = '%s/checkpoints/%s/%s' %(configs.save_dir, params.dataset, params.model)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            meta_net.load_state_dict(tmp['state'])

    max_acc = 0
    state = None

    for epoch in range(start_epoch,stop_epoch):
        meta_lr = meta_lr * (1. - epoch / float(stop_epoch-start_epoch))
        set_learning_rate(meta_optimizer, meta_lr)

        net = meta_net.clone()
        optimizer = get_optimizer(net, state)

        loss = do_learning(net,epoch,optimizer,base_loader)
        state = optimizer.state_dict()  # save optimizer state

        meta_net.point_grad_to(net)
        meta_optimizer.step()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)



        acc ,a = do_evaluation(meta_net, val_loader, params.n_shot ,n_query)

        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, f'{a}_best_model.pth')
            torch.save({'epoch': epoch, 'state': meta_net.state_dict(),'meta_optimizer': meta_optimizer.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state': meta_net.state_dict(),
                'meta_optimizer': meta_optimizer.state_dict()
                }, outfile)






