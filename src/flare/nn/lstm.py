# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from flare.nn.sequential import Sequential3
from torch.autograd import Variable


class ConvLSTM(th.nn.Module):
    def __init__(self, d_in, d_out, ksize, padding=0, bsize=1, width=511, height=511):
        super(ConvLSTM, self).__init__()
        ch_out = d_out
        ch_in = 2 * ch_out + d_in

        self.batch = bsize
        self.relu = nn.ReLU()

        self.f = nn.Conv2d(ch_in, ch_out, ksize, padding=padding)
        self.i = nn.Conv2d(ch_in, ch_out, ksize, padding=padding)
        self.o = nn.Conv2d(ch_in, ch_out, ksize, padding=padding)
        self.C = nn.Conv2d(ch_in, ch_out, ksize, padding=padding)

        self.state = Variable(th.rand(bsize, ch_out, width, height))
        self.history = Variable(th.rand(bsize, ch_out, width, height))
        self.old_state = None
        self.old_history = None

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

    def reset(self):
        sizes = list(self.state.data.size())
        self.state = Variable(th.rand(*sizes))
        self.history = Variable(th.rand(*sizes))
        self.old_state = None
        self.old_history = None

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

    def batch_size_changed(self, new_val, orig_val, force=False):
        sizes = list(self.state.data.size())
        sizes[0] = new_val
        if force or new_val != self.batch:
            self.old_state = self.state
            self.old_history = self.history
            self.state = Variable(th.zeros(sizes))
            self.history = Variable(th.zeros(sizes))
        elif self.old_state is not None:
            self.state = self.old_state
            self.history = self.old_history

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

        self.batch = new_val

    def forward(self, x):
        self.state = Variable(self.state.data)
        self.history = Variable(self.history.data)

        x = th.cat([self.state, self.history, x], dim=1)

        gate_forget = th.sigmoid(self.f(x))
        gate_output = th.sigmoid(self.o(x))
        gate_input = th.sigmoid(self.i(x))

        update = th.tanh(self.C(x))

        state = gate_forget * self.state + gate_input * update
        output = gate_output * th.tanh(state)

        print('forget:', th.max(gate_forget.data), th.min(gate_forget.data))
        print('output:', th.max(gate_output.data), th.min(gate_output.data))
        print('input:', th.max(gate_input.data), th.min(gate_input.data))
        print('update:', th.max(update.data), th.min(update.data))
        print('state:', th.max(state.data), th.min(state.data))

        self.state = state
        self.history = output

        return output


class StackedConvLSTM(th.nn.Module):
    def __init__(self, stacked_layers, d_in, d_hidden, d_out, ksize, padding=0, bsize=1, width=511, height=511):
        super(StackedConvLSTM, self).__init__()

        self._array_ = [ConvLSTM(d_in, d_hidden, ksize, padding=padding, bsize=bsize, width=width, height=height)]
        for i in range(stacked_layers - 1):
            self._array_.append(ConvLSTM(d_hidden, d_hidden, ksize, padding=padding, bsize=bsize, width=width, height=height))
        self._array_.append(ConvLSTM(d_hidden, d_out, ksize, padding=padding, bsize=bsize, width=width, height=height))

        self.lstm = nn.Sequential(*self._array_)

    def reset(self):
        for lstm in self._array_:
            lstm.reset()

    def batch_size_changed(self, new_val, orig_val, force=False):
        for l in self.lstm:
            l.batch_size_changed(new_val, orig_val, force=force)

    def forward(self, x):
        result = self.lstm(x)
        return result


class Conv3DLSTM(th.nn.Module):
    def __init__(self, seq_size, d_in, d_out, ksize, padding=0, bsize=1, width=511, height=511):
        super(Conv3DLSTM, self).__init__()

        ch_out = d_out
        ch_in = 2 * ch_out + d_in

        self.batch = bsize
        self.seq = seq_size

        self.f = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)
        self.i = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)
        self.o = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)
        self.C = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)

        self.state = Variable(th.zeros([bsize, ch_out, seq_size, width, height]))
        self.history = Variable(th.zeros([bsize, ch_out, seq_size, width, height]))
        self.old_state = None
        self.old_history = None

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

    def batch_size_changed(self, new_val, orig_val):
        sizes = list(self.state.data.size())
        sizes[0] = new_val
        if new_val != self.batch:
            self.old_state = self.state
            self.old_history = self.history
            self.state = Variable(th.zeros(sizes))
            self.history = Variable(th.zeros(sizes))
        elif self.old_state is not None:
            self.state = self.old_state
            self.history = self.old_history

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

    def forward(self, xs):
        self.state = Variable(self.state.data)
        self.history = Variable(self.history.data)

        xs = th.cat([self.state, self.history, xs], dim=1)

        gate_forget = th.sigmoid(self.f(xs))
        gate_output = th.sigmoid(self.o(xs))
        gate_input = th.sigmoid(self.i(xs))

        update = th.tanh(self.C(xs))

        state = gate_forget * self.state + gate_input * update
        output = gate_output * th.tanh(state)

        self.state = state
        self.history = output

        return output


class StackedConv3DLSTM(th.nn.Module):
    def __init__(self, stacked_layers, seq_size, d_in, d_hidden, d_out, ksize, padding=0, bsize=1, width=511, height=511):
        super(StackedConv3DLSTM, self).__init__()

        lstm = [Conv3DLSTM(seq_size, d_in, d_hidden, ksize, padding=padding, bsize=bsize, width=width, height=height)]
        for i in range(stacked_layers - 1):
            lstm.append(Conv3DLSTM(seq_size, d_hidden, d_hidden, ksize, padding=padding, bsize=bsize, width=width, height=height))

        self.lstm = nn.Sequential(*lstm)
        self.out = nn.Conv3d(d_hidden, d_out, ksize, padding=padding)

    def batch_size_changed(self, new_val, orig_val):
        for l in self.lstm:
            l.batch_size_changed(new_val, orig_val)

    def forward(self, x):
        result = self.lstm(x)
        output = th.tanh(self.out(result))
        return output


class SquentialConvLSTM(th.nn.Module):
    def __init__(self, seq_size, dim, wsize, ksize, padding=0, bsize=1, width=511, height=511):
        super(SquentialConvLSTM, self).__init__()

        self.batch = bsize
        self.seq = seq_size
        self.wnd = wsize

        self.lstm = Conv3DLSTM(self.wnd, dim, dim, ksize, padding=padding, bsize=bsize, width=width, height=height)

    def batch_size_changed(self, new_val, orig_val):
        self.lstm.batch_size_changed(new_val, orig_val)

    def forward(self, xs):
        sz = xs.size()

        estm = None
        output = []

        for i in range(self.seq - self.wnd + 1):
            estm = self.lstm(xs[:, :, i:(i + self.wnd), :, :])

        for i in range(self.seq):
            output.append(estm[:, :, 0::self.wnd])
            estm = self.lstm(estm)

            result = th.cat(output, dim=2)

        return result


class StackedSquentialConvLSTM(th.nn.Module):
    def __init__(self, stacked_layers, seq_size, dim, wsize, ksize, padding=0, bsize=1, width=511, height=511):
        super(StackedSquentialConvLSTM, self).__init__()

        lstm = [SquentialConvLSTM(seq_size, dim, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height)]
        for i in range(stacked_layers - 1):
            lstm.append(SquentialConvLSTM(seq_size, dim, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height))

        self.lstm = nn.Sequential(*lstm)

    def batch_size_changed(self, new_val, orig_val):
        for l in self.lstm:
            l.batch_size_changed(new_val, orig_val)

    def forward(self, x):
        output = self.lstm(x)
        return output


class TemporalConvLSTM(th.nn.Module):
    def __init__(self, seq_size, d_in, d_out, wsize, ksize, padding=0, bsize=1, width=511, height=511, temporal_scale=1.0):
        super(TemporalConvLSTM, self).__init__()

        ch_out = d_out
        ch_in = 2 * ch_out + d_in

        self.batch = bsize
        self.seq = seq_size
        self.wnd = wsize

        self.width = width
        self.height = height

        self.scale = temporal_scale

        self.w = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)
        self.f = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)
        self.i = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)
        self.o = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)
        self.C = nn.Conv3d(ch_in, ch_out, ksize, padding=padding)

        self.state = Variable(th.zeros([bsize, ch_out, self.wnd, width, height]))
        self.history = Variable(th.zeros([bsize, ch_out, self.wnd, width, height]))
        self.old_state = None
        self.old_history = None

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

    def batch_size_changed(self, new_val, orig_val):
        sizes = list(self.state.data.size())
        sizes[0] = new_val
        if new_val != self.batch:
            self.old_state = self.state
            self.old_history = self.history
            self.state = Variable(th.zeros(sizes))
            self.history = Variable(th.zeros(sizes))
        elif self.old_state is not None:
            self.state = self.old_state
            self.history = self.old_history

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

    def reset(self):
        sizes = list(self.state.data.size())
        self.state = Variable(th.zeros(sizes))
        self.history = Variable(th.zeros(sizes))
        self.old_state = None
        self.old_history = None

        if th.cuda.is_available():
            self.state = self.state.cuda()
            self.history = self.history.cuda()

    def predict(self, d, x):
        self.state = Variable(self.state.data)
        self.history = Variable(self.history.data)

        x = th.cat([self.state, self.history, x], dim=1)
        window = (1 / (th.sigmoid(self.scale / d * self.w(x)) + 0.1) - 0.9)

        gate_forget = th.exp(-th.sigmoid(self.f(x)) * window)
        gate_output = 1 - th.exp(-th.sigmoid(self.o(x)) * window)
        gate_input = th.exp(-th.sigmoid(self.i(x)) * window)

        update = th.tanh(self.C(x))

        state = gate_forget * self.state + gate_input * update
        output = gate_output * th.tanh(state)

        self.state = state
        self.history = output

        return output

    def forward(self, ts, ds, xs):
        szt = ts.size()
        szd = ds.size()

        estm = None
        for i in range(szt[2] - self.wnd + 1):
            es = xs[:, :, i:(i + self.wnd)]
            dt = ts[:, :, i:(i + self.wnd)]
            estm = self.predict(dt, es)

        output = []
        for i in range(szd[2]):
            output.append(estm[:, :, 0::self.wnd])
            dt = ds[:, :, i:(i + self.wnd)]
            if dt.size()[2] == estm.size()[2]:
                estm = self.predict(dt, estm)

        result = th.cat(output, dim=2)

        return result


class StackedTemporalConvLSTM(th.nn.Module):
    def __init__(self, stacked_layers, seq_size, d_in, d_hidden, d_out, wsize, ksize, padding=0, bsize=1, width=511, height=511, temporal_scale=1.0):
        super(StackedTemporalConvLSTM, self).__init__()

        lstm = [TemporalConvLSTM(seq_size, d_in, d_hidden, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale)]
        for i in range(stacked_layers - 1):
            lstm.append(TemporalConvLSTM(seq_size, d_hidden, d_hidden, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale))
        lstm.append(TemporalConvLSTM(seq_size, d_hidden, d_out, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale))

        self.lstm = Sequential3(*lstm)

    def batch_size_changed(self, new_val, orig_val):
        for i in range(len(self.lstm)):
            self.lstm[i].batch_size_changed(new_val, orig_val)

    def forward(self, xs, ts, ds):
        result = self.lstm.forward(xs, ts, ds)
        return result


class ResTemporalConvLSTM(th.nn.Module):
    def __init__(self, seq_size, d_in, d_hidden, d_out, wsize, ksize, padding=0, bsize=1, width=511, height=511, temporal_scale=1.0):
        super(ResTemporalConvLSTM, self).__init__()

        lstm = [TemporalConvLSTM(seq_size, d_in, d_hidden, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale)]
        lstm.append(TemporalConvLSTM(seq_size, d_hidden, d_hidden, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale))
        lstm.append(TemporalConvLSTM(seq_size, d_hidden, d_out, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale))

        self.lstm = Sequential3(*lstm)

    def batch_size_changed(self, new_val, orig_val):
        for i in range(len(self.lstm)):
            self.lstm[i].batch_size_changed(new_val, orig_val)

    def forward(self, xs, ts, ds):
        result = xs + self.lstm.forward(xs, ts, ds)
        return result


class StackedResTemporalConvLSTM(th.nn.Module):
    def __init__(self, stacked_layers, seq_size, d_in, d_hidden, d_out, wsize, ksize, padding=0, bsize=1, width=511, height=511, temporal_scale=1.0):
        super(StackedResTemporalConvLSTM, self).__init__()

        lstm = [ResTemporalConvLSTM(seq_size, d_in, d_hidden, d_hidden, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale)]
        for i in range(stacked_layers - 1):
            lstm.append(ResTemporalConvLSTM(seq_size, d_hidden, d_hidden, d_hidden, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale))
        lstm.append(TemporalConvLSTM(seq_size, d_hidden, d_out, wsize, ksize, padding=padding, bsize=bsize, width=width, height=height, temporal_scale=temporal_scale))

        self.lstm = Sequential3(*lstm)

    def batch_size_changed(self, new_val, orig_val):
        for i in range(len(self.lstm)):
            self.lstm[i].batch_size_changed(new_val, orig_val)

    def forward(self, xs, ts, ds):
        result = self.lstm.forward(xs, ts, ds)
        return result
