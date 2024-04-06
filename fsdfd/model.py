import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

def one_hot(y, num_class):
    y = y.long()
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

class MamlModel(nn.Module):

    def __init__(self, n_way, n_support, n_query, loss_type = 'mse'):
        super(MamlModel,self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.loss_type = loss_type
        self.n_query = n_query  # (change depends on input)
        self.feat_dim = [64,19,19]

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class model(MamlModel):

    def __init__(self, n_way, n_support, n_query):
        super(model,self).__init__(n_way, n_support, n_query)

        self.conv = nn.Sequential(
            # 28 x 28 - 1
            nn.Conv2d(3, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 14 x 14 - 64
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 7 x 7 - 64
            nn.Conv2d(64, 64, 3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4 x 4 - 64
            nn.Conv2d(64, 64, 3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 2 x 2 - 64
        )

        self.classifier_part1 = nn.Sequential(

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(2),

        )

        self.linear_classifier = nn.Sequential(
            nn.Linear(1024,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )

    def auto_classifier(self, x_support_flat, x_query_flat):
        out = torch.abs(x_support_flat - x_query_flat)
        out = self.linear_classifier(out)
        out = F.sigmoid(out)
        return out


    def forward(self, x):

        x = Variable(x.cuda())
        extend_final_feat_dim = self.feat_dim.copy()
        x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.conv(x)
        z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        # spt 每一类的原型
        z_support = z_support.contiguous().view(self.n_way, self.n_support, *self.feat_dim).mean(1)  # [5, 64, 19, 19]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)  # [80, 64, 19, 19]

        z_support_ext = z_support.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)  # [80, 5, 64, 19, 19]
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)  # [5, 80, 64, 19, 19]
        z_query_ext = torch.transpose(z_query_ext, 0, 1)  # [80, 5, 64, 19, 19]
        z_support_ext = z_support_ext.view(-1, *extend_final_feat_dim)  # [400, 64, 19, 19]
        z_query_ext = z_query_ext.contiguous().view(-1, *extend_final_feat_dim)  # [400, 64, 19, 19]

        x_support = self.classifier_part1(z_support_ext)  # [400, 64, 19, 19] ==> [400, 64, 4, 4]
        x_query = self.classifier_part1(z_query_ext)  # [400, 64, 19, 19] ==> [400, 64, 4, 4]

        x_support_flat = x_support.view(self.n_way * self.n_way * self.n_query, -1)  # [400, 1024]
        x_query_flat = x_query.view(self.n_way * self.n_way * self.n_query, -1)

        # cosine similarity
        # sorce = F.cosine_similarity(x_support_flat, x_query_flat, dim=1).view(self.n_way * self.n_way * self.n_query)

        # pairwise distance similarity
        # distance = F.pairwise_distance(x_support_flat, x_query_flat, p=2, keepdim=True).view(self.n_way * self.n_way * self.n_query)
        # sorce = 1 / (1 + distance)

        # auto similarity
        sorce = self.auto_classifier(x_support_flat, x_query_flat)

        return sorce.view(-1, self.n_way)

    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

        scores = self.forward(x)
        if self.loss_type == 'mse':
            y_oh = one_hot(y, self.n_way)
            y_oh = Variable(y_oh.cuda())

            return self.loss_fn(scores, y_oh )
        else:
            y = Variable(y.cuda())
            return self.loss_fn(scores, y )

    def correct(self, x):
        scores = self.forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def clone(self):
        clone = model(self.n_way, self.n_support, self.n_query)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone




if __name__ == '__main__':


    model = model(5,5,16)

    print(model.conv)
