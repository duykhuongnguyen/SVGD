import torch
from torch import Tensor
from torch.autograd import Variable, grad
from torch.nn import Module
from torch.nn.functional import log_softmax, cross_entropy, kl_div


def RBF(X: Tensor, Y: Tensor, sigma: float):
    '''
    Get the RBF kernel of Y with respect to X. The formula is
        exp(-||x-y||^2/h),
    where h = 2 * sigma ^2.
    '''

    dnorm2 = torch.linalg.norm(X, dim=1, ord=2).unsqueeze(1) ** 2 \
           + torch.linalg.norm(Y, dim=1, ord=2).unsqueeze(0) ** 2 \
           - 2 * X @ Y.T

    # because of numerical error, the diagonal is only close to 0
    dnorm2 -= dnorm2.diag().diag()

    # TODO: default value for h -- median^2/log n
    h = 1e-8 + 2 * sigma ** 2

    K_XY = torch.exp(-dnorm2 / h)

    return K_XY


def svgd(model, X, y, random_init=True, n=4, epsilon=8/255, num_steps=20, loss_type='ce', step_size=1/255,
         x_min=-1, x_max=1, sigma=1, output_all=False, postprocess=(lambda x:x)):
    '''
    Currently only works with RBF kernel, and L_inf norm.

    @param model        : the base model
    @param X            : the clean input
    @param Y            : the clean output
    @param random_init  : whether the adversarial examples starts at the clean example or somewhere near
    @param n            : number of adversarial examples generated per clean one
    @param epsilon      : the maximum perturbation budget
    @param num_steps    : number of adversarial example optimization step
    @param loss_type    : the loss used as unnormalized probability for adversarial examples
                            can be 'ce' for cross-entropy or 'kl' for KL-divergence
    @param step_size    : the step size used at each adversarial optimization step
    @param x_min        : the minimum valid value of a pixel, used for clipping
    @param x_max        : the maximum valid value of a pixel, used for clipping
    @param sigma        : the width of the RBF kernel
    @param output_all   : whether to output all intermediate adversarial examples
    '''

    model_status = model.training
    model.eval()

    # batch_size, c, h, w = X.size()

    X_particles = X.repeat(n, 1)
    X_adv = X.repeat(n, 1)

    if random_init:
        X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)

    if output_all:
        all_advs = []

    for _ in range(num_steps):

        # reset leaf variable
        X_adv = Variable(X_adv.data, requires_grad=True)
        if loss_type == 'ce':
            loss = cross_entropy(model(postprocess(X_adv.clone())), y.repeat(n), reduction='none')
        elif loss_type == 'kl':
            # currently the code doesn't work correctly, since matrix calculus is complicated
            raise NotImplementedError

            loss = kl_div(log_softmax(model(X_adv), dim=1), log_softmax(model(X_particles), dim=1),
                          log_target=True, reduction='none')

        # since each loss term is independent of other examples, nabla_xi log(loss(x_j)) should be 0
        grad_log = grad(loss.sum(), X_adv)[0]

        # reset leaf variable again
        X_adv = Variable(X_adv.data, requires_grad=True)
        # X_adv = X_adv.detach().view(batch_size * n, -1)
        K_XX = RBF(X_adv.detach(), X_adv, sigma)
        grad_K = grad(K_XX.sum(), X_adv)[0] # / batch_size

        # print(K_XX.shape, grad_log.shape, grad_K.shape)
        phi = (K_XX @ grad_log + grad_K) / n
        # PGD-like, remove the .sign() if needed!
        X_adv = (X_adv + step_size * phi.sign())#.reshape(batch_size * n, c, h, w)

        # clamping adversary into valid image
        X_adv = torch.clamp(X_adv, X_particles - epsilon, X_particles + epsilon)
        X_adv = torch.clamp(X_adv, x_min, x_max)

        # X_adv = postprocess(X_adv)

        if output_all:
            all_advs.append(X_adv.detach())

    model.train(model_status)

    if output_all:
        return all_advs
    else:
        # reset everything so that X_adv can be used as input later on
        X_adv = X_adv.detach()
        return X_adv


class SVGD:
    '''
    Currently only works with RBF kernel, and L_inf norm.

    @param model        : the base model
    @param X            : the clean input
    @param Y            : the clean output
    @param random_init  : whether the adversarial examples starts at the clean example or somewhere near
    @param n            : number of adversarial examples generated per clean one
    @param epsilon      : the maximum perturbation budget
    @param num_steps    : number of adversarial example optimization step
    @param loss_type    : the loss used as unnormalized probability for adversarial examples
                            can be 'ce' for cross-entropy or 'kl' for KL-divergence
    @param step_size    : the step size used at each adversarial optimization step
    @param x_min        : the minimum valid value of a pixel, used for clipping
    @param x_max        : the maximum valid value of a pixel, used for clipping
    @param sigma        : the width of the RBF kernel
    @param output_all   : whether to output all intermediate adversarial examples
    '''

    def __init__(self, model, random_init=True, epsilon=8/255, num_steps=20, loss_type='ce', step_size=1/255,
                 clip_min=0, clip_max=1, sigma=1) -> None:

        self.model = model
        self.random_init = random_init
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.loss_type = loss_type
        self.step_size = step_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sigma = sigma


    def __call__(self, data, n=4, output_all=False, ensemble=True, postprocess=(lambda x: x)):
        '''
        If output_all is True, return the bundled n adversaries for each iteration.
        If ensemble is True, return only the one of the successful attacks if exists.
        '''

        label = self.model(data).argmax(dim=1)
        batch_size = data.size(0)

        X_adv = svgd(
            self.model, data, label, random_init=self.random_init, n=n, epsilon=self.epsilon, num_steps=self.num_steps,
            loss_type=self.loss_type, step_size=self.step_size, x_min=-1, x_max=1, sigma=1, output_all=output_all,
            postprocess=postprocess
        )

        if output_all or not ensemble:
            return X_adv

        y_adv = self.model(X_adv).argmax(dim=1)
        y_clean = label.repeat(n)
        success = y_adv != y_clean

        X_adv_ = torch.empty_like(data)
        for i in range(batch_size):
            for j in range(n):
                if success[i + j * batch_size]:
                    X_adv_[i] = X_adv[i + j * batch_size]
                    break
            else:
                X_adv_[i] = X_adv[i]

        return X_adv_
