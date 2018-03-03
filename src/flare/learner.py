# -*- coding: utf-8 -*-

import types
import numpy as np
import itertools

import torch as th
from torch.autograd import Variable

import flare.pipe as fp
import flare.util.iter as fui

def cast(element):
    element = np.array(element, dtype=np.float32)
    if th.cuda.is_available():
        return th.FloatTensor(element).cuda()
    else:
        return th.FloatTensor(element)


class BaseLearner():
    def __init__(self, batch=1, picker=None):
        super(BaseLearner, self).__init__()
        self.gpu_flag = False

        self.batch = batch
        self.old_batch = None
        self.status = 'construction'

        self.picker = picker
        if self.picker == None:
            self.picker = fp.pick_all

    def input_channel_size(self, channel_element):
        element_size = list(fui.deep_size(channel_element))
        batched_channel_size = [1, 1] + element_size
        return tuple(batched_channel_size)

    def before_test(self):
        pass

    def before_learn(self):
        pass

    def before_test_step(self, xs, ys):
        if 'model' in dir(self) and 'before_test_step' in dir(self.model):
            self.model.before_test_step(xs, ys)

    def before_learn_step(self, xs, ys):
        if 'model' in dir(self) and 'before_learn_step' in dir(self.model):
            self.model.before_learn_step(xs, ys)

    def test_step(self, xs, ys):
        raise NotImplemented()

    def learn_step(self, xs, ys, retain_graph=False):
        raise NotImplemented()

    def after_test_step(self, xs, ys):
        pass

    def after_learn_step(self, xs, ys):
        pass

    def after_test(self, vxs, vys, vrslt, lss):
        pass

    def after_learn(self):
        pass

    def batch_size_changed(self, new_val, orig_val):
        pass

    def set_batch_size(self, bsize):
        self.old_batch = self.batch
        self.batch = bsize
        self.batch_size_changed(bsize, self.old_batch)

    def restore_batch_size(self):
        bsize = self.batch
        self.batch = self.old_batch
        self.batch_size_changed(self.old_batch, bsize)

    def batches(self, data, batch=None):
        if batch is None:
            batch = self.batch
        data_type = type(data)
        if data_type == list or data_type == tuple:
            return fp.batches(data, pick=fp.pick_value, batch_size=batch)
        elif data_type == dict:
            return fp.batches(data, pick=self.picker, batch_size=batch)
        elif data_type == types.GeneratorType:
            return fp.batches(data, pick=fp.pick_value, batch_size=batch)

    def test(self, dataset):
        statusold = self.status
        self.status = 'test'
        self.set_batch_size(1)

        self.before_test()

        test_loss = 0
        count = 0
        for data in dataset:
            for x, y in data:
                xs = cast(x)
                ys = cast(y)
                vxs = Variable(xs, requires_grad=False)
                vys = Variable(ys, requires_grad=False)
                self.before_test_step(vxs, vys)
                vrslt, lss = self.test_step(vxs, vys)
                self.after_test_step(vxs, vys)
                test_loss += lss
                count += 1

        if count > 0:
            avgloss = test_loss / count
            print('###################################################')
            print('loss:', avgloss.data[0])
            print('###################################################')
            self.after_test(vxs.data, vys.data, vrslt.data, lss.data)

            self.status = statusold

            result = avgloss.data[0]
        else:
            self.status = statusold
            result = 1.0

        if self.old_batch is not None:
            self.restore_batch_size()

        return result

    def learn(self, dataset, validset, retain_graph=False, steps=-1, valid_size=-1):
        self.status = 'learn'

        self.before_learn()

        idx = 0
        for batch in self.batches(dataset, batch=self.batch):
            xs, ys = [], []
            for data in batch:
                for x, y in data:
                    xs.append(cast(x))
                    ys.append(cast(y))

            xs = th.cat(xs, dim=0)
            ys = th.cat(ys, dim=0)

            vxs = Variable(xs, requires_grad=False)
            vys = Variable(ys, requires_grad=False)

            self.before_learn_step(vxs, vys)
            self.learn_step(vxs, vys, retain_graph=retain_graph)
            self.after_learn_step(vxs, vys)

            if steps != -1 and valid_size != -1 and idx % steps == 0:
                self.test(itertools.islice(validset, valid_size))

        self.after_learn()

        return self.test(validset)


class StandardLearner(BaseLearner):
    def __init__(self, model, predict, loss, optimizer, batch=1, picker=None):
        super(StandardLearner, self).__init__(batch=batch, picker=picker)
        self.model = model
        self.predict = predict
        self.loss = loss
        self.optimizer = optimizer

    def batch_size_changed(self, new_val, orig_val):
        if 'batch_size_changed' in dir(self.model):
            self.model.batch_size_changed(new_val, orig_val)

    def before_test(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.model = self.model.cuda()
            self.gpu_flag = True

        self.model.eval()

    def before_learn(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.model = self.model.cuda()
            self.gpu_flag = True

        self.model.train()

    def test_step(self, xs, ys):
        output = self.predict(xs)
        szy = ys.size()[1]
        return output, self.loss(output[:, 0:szy], ys)

    def learn_step(self, xs, ys, retain_graph=False):
        self.optimizer.zero_grad()
        output = self.predict(xs)
        l = self.loss(output, ys)
        l.backward(retain_graph=retain_graph)
        self.optimizer.step()


class TemporalLearner(BaseLearner):
    def __init__(self, model, predict, loss, optimizer, batch=1, picker=None, timescale=30):
        super(TemporalLearner, self).__init__(batch=batch, picker=picker)
        self.model = model
        self.predict = predict
        self.loss = loss
        self.optimizer = optimizer
        self.timescale = timescale

    def batch_size_changed(self, new_val, orig_val):
        if 'batch_size_changed' in dir(self.model):
            self.model.batch_size_changed(new_val, orig_val)

    def before_test(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.model = self.model.cuda()
            self.gpu_flag = True

        self.model.eval()

    def before_learn(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.model = self.model.cuda()
            self.gpu_flag = True

        self.model.train()

    def test_step(self, xs, ys):
        chs = xs.size()[1]
        output = self.predict(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:chs])
        return output, self.loss(output, ys[:, 1:chs])

    def learn_step(self, xs, ys, retain_graph=False):
        if 0 < th.max(xs[:, 0, :, 0, 0].data) < self.timescale:
            self.optimizer.zero_grad()

            chs = xs.size()[1]
            output = self.predict(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:chs])
            l = self.loss(output, ys[:, 1:chs])
            l.backward(retain_graph=retain_graph)
            self.optimizer.step()


class SpecialTemporalLearner(BaseLearner):
    def __init__(self, model, predict, loss, optimizer, features=0, batch=1, picker=None, timescale=30):
        super(SpecialTemporalLearner, self).__init__(batch=batch, picker=picker)
        self.model = model
        self.predict = predict
        self.loss = loss
        self.optimizer = optimizer
        self.timescale = timescale
        self.features = features

    def batch_size_changed(self, new_val, orig_val):
        if 'batch_size_changed' in dir(self.model):
            self.model.batch_size_changed(new_val, orig_val)

    def before_test(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.model = self.model.cuda()
            self.gpu_flag = True

        self.model.eval()

    def before_learn(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.model = self.model.cuda()
            self.gpu_flag = True

        self.model.train()

    def test_step(self, xs, ys):
        chs = xs.size()[1]
        bfc = chs - self.features
        if chs == bfc:
            output = self.predict(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc])
        else:
            output = self.predict(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc], sxfs=xs[:, bfc:chs], syfs=ys[:, bfc:chs])
        return output, self.loss(output, xs[:, 1:chs], ys[:, 1:chs], model=self.model)

    def learn_step(self, xs, ys, retain_graph=False):
        if 0 < th.max(xs[:, 0, :, 0, 0].data) < self.timescale:
            self.optimizer.zero_grad()

            chs = xs.size()[1]
            bfc = chs - self.features
            if chs == bfc:
                output = self.predict(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc])
            else:
                output = self.predict(xs[:, 0::chs], ys[:, 0::chs], xs[:, 1:bfc], sxfs=xs[:, bfc:chs], syfs=ys[:, bfc:chs])
            if 'model' in dir(self):
                l = self.loss(output, xs[:, 1:chs], ys[:, 1:chs], model=self.model)
            else:
                l = self.loss(output, xs[:, 1:chs], ys[:, 1:chs])
            l.backward(retain_graph=retain_graph)
            self.optimizer.step()


class PQLearner(BaseLearner):
    def __init__(self, modelp, predictp, optimizerp, modelq, predictq, optimizerq, loss, batch=1, picker=None):
        super(PQLearner, self).__init__(batch=batch, picker=picker)
        self.modelp = modelp
        self.predictp = predictp
        self.optimizerp = optimizerp
        self.modelq = modelq
        self.predictq = predictq
        self.optimizerq = optimizerq
        self.loss = loss

    def batch_size_changed(self, new_val, orig_val):
        if 'batch_size_changed' in dir(self.modelp):
            self.modelp.batch_size_changed(new_val, orig_val)
        if 'batch_size_changed' in dir(self.modelq):
            self.modelq.batch_size_changed(new_val, orig_val)

    def before_test(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.modelp = self.modelp.cuda()
            self.modelq = self.modelq.cuda()
            self.gpu_flag = True
        self.modelp.eval()
        self.modelq.eval()

    def before_learn(self):
        if th.cuda.is_available() and not self.gpu_flag:
            self.modelp = self.modelp.cuda()
            self.modelq = self.modelq.cuda()
            self.gpu_flag = True
        self.modelp.train()
        self.modelq.train()

    def test_step(self, xs, ys):
        output = self.predictq(xs)
        return output, self.loss(output, ys)

    def learn_step(self, xs, ys, retain_graph=False):
        self.optimizerp.zero_grad()
        outputp = self.predictp(xs)
        l = self.loss(outputp, ys)
        l.backward(retain_graph=retain_graph)
        self.optimizerp.step()

        self.optimizerq.zero_grad()
        outputq = self.predictq(xs)
        l = self.loss(outputq, ys)
        l.backward(retain_graph=retain_graph)
        self.optimizerq.step()
