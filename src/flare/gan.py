# -*- coding: utf-8 -*-

import torch as th
from torch.autograd import Variable

import flare.learner as fl


class GANLearner(fl.BaseLearner):
    def __init__(self, modelg, predictg, optimizerg, modeld, predictd, optimizerd, lossg, lossd, extrag, extrad, batch=1, picker=None):
        super(GANLearner, self).__init__(batch=batch, picker=picker)
        self.modelg = modelg
        self.predictg = predictg
        self.optimizerg = optimizerg
        self.modeld = modeld
        self.predictd = predictd
        self.optimizerd = optimizerd
        self.lossg = lossg
        self.lossd = lossd
        self.extrag = extrag
        self.extrad = extrad

        self.ones = Variable(th.ones(batch), requires_grad=False)
        self.zeros = Variable(th.zeros(batch), requires_grad=False)
        if th.cuda.is_available():
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()

    def batch_size_changed(self, new_val, orig_val):
        if 'batch_size_changed' in dir(self.modelg):
            self.modelg.batch_size_changed(new_val, orig_val)
        if 'batch_size_changed' in dir(self.modeld):
            self.modeld.batch_size_changed(new_val, orig_val)

        self.ones = Variable(th.ones(self.batch), requires_grad=False)
        self.zeros = Variable(th.zeros(self.batch), requires_grad=False)
        if th.cuda.is_available():
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()

    def before_test(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.modelg = self.modelg.cuda()
            self.modeld = self.modeld.cuda()
            self.gpu_flag = True
        self.modelg.eval()
        self.modeld.eval()

    def before_learn(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.modelg = self.modelg.cuda()
            self.modeld = self.modeld.cuda()
            self.gpu_flag = True
        self.modelg.train()
        self.modeld.train()

    def test_step(self, xs, ys):
        output = self.predictg(xs)
        return output, self.lossg(output, ys)

    def reduce_mean_to_batch(self, array):
        sz = array.size()
        lng = len(sz)
        size = 1
        for i in range(lng - 1, 0, -1):
            size = size * sz[i]
            array = th.sum(array, dim=i)
        return array / size

    def learn_step(self, xs, ys, retain_graph=False):

        #train d on real ys
        self.optimizerd.zero_grad()
        outputd = self.predictd(self.extrad(ys, xs))
        l = self.lossd(self.reduce_mean_to_batch(outputd), self.ones)
        l.backward(retain_graph=retain_graph)
        self.optimizerd.step()

        #train d on fake ys
        self.optimizerd.zero_grad()
        outputg = self.predictg(self.extrag(xs, ys))
        outputd = self.predictd(self.extrad(outputg, xs))
        l = self.lossd(self.reduce_mean_to_batch(outputd), self.zeros)
        l.backward(retain_graph=retain_graph)
        self.optimizerd.step()

        #train g on fake ys
        self.optimizerg.zero_grad()
        outputg = self.predictg(self.extrag(xs, ys))
        outputd = self.predictd(self.extrad(outputg, xs))
        l = self.lossd(self.reduce_mean_to_batch(outputd), self.ones) + self.lossg(outputg, ys)
        l.backward(retain_graph=retain_graph)
        self.optimizerg.step()


class SpecialTemporalGANLearner(fl.BaseLearner):
    def __init__(self, modelg, predictg, optimizerg, modeld, predictd, optimizerd, lossg, lossd, extra, batch=1, picker=None, features=0, timescale=30):
        super(SpecialTemporalGANLearner, self).__init__(batch=batch, picker=picker)
        self.modelg = modelg
        self.predictg = predictg
        self.optimizerg = optimizerg
        self.modeld = modeld
        self.predictd = predictd
        self.optimizerd = optimizerd
        self.lossg = lossg
        self.lossd = lossd
        self.extra = extra
        self.timescale = timescale
        self.features = features

        self.ones = Variable(th.ones(batch), requires_grad=False)
        self.zeros = Variable(th.zeros(batch), requires_grad=False)
        if th.cuda.is_available():
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()

    def batch_size_changed(self, new_val, orig_val):
        if 'batch_size_changed' in dir(self.modelg):
            self.modelg.batch_size_changed(new_val, orig_val)
        if 'batch_size_changed' in dir(self.modeld):
            self.modeld.batch_size_changed(new_val, orig_val)

        self.ones = Variable(th.ones(self.batch), requires_grad=False)
        self.zeros = Variable(th.zeros(self.batch), requires_grad=False)
        if th.cuda.is_available():
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()

    def before_test(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.modelg = self.modelg.cuda()
            self.modeld = self.modeld.cuda()
            self.gpu_flag = True
        self.modelg.eval()
        self.modeld.eval()

    def before_learn(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.modelg = self.modelg.cuda()
            self.modeld = self.modeld.cuda()
            self.gpu_flag = True
        self.modelg.train()
        self.modeld.train()

    def test_step(self, xs, ys):
        chs = xs.size()[1]
        bfc = chs - self.features
        if chs == bfc:
            output = self.predictg(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc])
        else:
            output = self.predictg(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc], sxfs=xs[:, bfc:chs], syfs=ys[:, bfc:chs])
        return output, self.lossg(output, xs[:, 1:chs], ys[:, 1:chs], model=self.modelg)

    def reduce_mean_to_batch(self, array):
        sz = array.size()
        lng = len(sz)
        size = 1
        for i in range(lng - 1, 0, -1):
            size = size * sz[i]
            array = th.sum(array, dim=i)
        return array / size

    def learn_step(self, xs, ys, retain_graph=False):
        chs = xs.size()[1]
        bfc = chs - self.features

        if 0 < th.max(xs[:, 0, :, 0, 0].data) < self.timescale:
            #train d on real ys
            self.optimizerd.zero_grad()
            outputd = self.predictd(self.extra(ys[:, 1:bfc], xs))
            l = self.lossd(self.reduce_mean_to_batch(outputd), self.ones)
            l.backward(retain_graph=retain_graph)
            self.optimizerd.step()

            #train d on fake ys
            self.optimizerd.zero_grad()
            if chs == bfc:
                outputg = self.predictg(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc])
            else:
                outputg = self.predictg(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc], sxfs=xs[:, bfc:chs],
                                       syfs=ys[:, bfc:chs])
            outputd = self.predictd(self.extra(outputg, xs))
            l = self.lossd(self.reduce_mean_to_batch(outputd), self.zeros)
            l.backward(retain_graph=retain_graph)
            self.optimizerd.step()

            #train g on fake ys
            self.optimizerg.zero_grad()
            if chs == bfc:
                outputg = self.predictg(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc])
            else:
                outputg = self.predictg(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc], sxfs=xs[:, bfc:chs],
                                       syfs=ys[:, bfc:chs])
            outputd = self.predictd(self.extra(outputg, xs))
            l = self.lossd(self.reduce_mean_to_batch(outputd), self.ones) + self.lossg(outputg, xs[:, 1:bfc], ys[:, 1:bfc], model=self.modelg)
            l.backward(retain_graph=retain_graph)
            self.optimizerg.step()
