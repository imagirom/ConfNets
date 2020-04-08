import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """modified convgru implementation of https://github.com/jacobkimmel/pytorch_convgru"""
    def __init__(self, input_size, hidden_size, kernel_size, conv_type, 
            invert_update_gate=False,
            out_gate_activation=torch.tanh):
        """
        Generate a convolutional GRU cell

        :param invert_update_gate:
        :param out_gate_activation:
        """
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.invert_update_gate = invert_update_gate
        padding = kernel_size // 2
        hs = hidden_size
        self.reset_gate = conv_type(input_size + hs, hs, kernel_size, padding=padding)
        self.update_gate = conv_type(input_size + hs, hs, kernel_size, padding=padding)
        self.out_gate = conv_type(input_size + hs, hs, kernel_size, padding=padding)

        assert callable(out_gate_activation),\
            "Expected out_gate_activation to be callable, but got {} instead"\
            .format(type(out_gate_activation))
        self.out_gate_activation = out_gate_activation

        # Initial hidden state
        self.hidden_state = None

        # init.orthogonal(self.reset_gate.weight)
        # init.orthogonal(self.update_gate.weight)
        # init.orthogonal(self.out_gate.weight)
        # init.constant(self.reset_gate.bias, 0.)
        # init.constant(self.update_gate.bias, 0.)
        # init.constant(self.out_gate.bias, 0.)

    def forward(self, input_):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate hidden state of zeros, if currently None
        if self.hidden_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            self.hidden_state = input_.new(torch.zeros(state_size))

        # if batch size changes, reset hidden state to 0-s.
        if self.hidden_state.shape[0] != batch_size:
            self.hidden_state = torch.zeros((batch_size, self.hidden_size, *spatial_size))

        # data size is [batch, channel, ...]
        stacked_inputs = torch.cat([input_, self.hidden_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = self.out_gate_activation(self.out_gate(torch.cat([input_, self.hidden_state * reset], dim=1)))

        # Sometimes an architecture has these the other way around
        if not self.invert_update_gate:
            self.hidden_state = self.hidden_state * (1-update) + out_inputs * update
        else:
            self.hidden_state = self.hidden_state * update + out_inputs * (1-update)

        return self.hidden_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_sizes, n_layers, conv_type):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_size : integer. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if isinstance(kernel_sizes, (list, tuple)):
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes]*n_layers

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            cell = ConvGRUCell(self.input_size, self.hidden_size, self.kernel_sizes[i], conv_type)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length

    def forward(self, input_, hidden=None):
        """
        Parameters
        ----------
        x : 5D/6D input tensor. (batch, channels, sequence, height, width, *depth)).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        """

        # warmstart Gru hidden state with one pass through layer 0

        sl = self.sequence_length
        upd_hidden = []
        for batch in range(input_.shape[0] // self.sequence_length):
            time_index = batch * sl

            for idx in range(self.n_layers):
                upd_cell_hidden = self.cells[idx](input_[time_index:time_index + 1]).detach()

            for s in range(self.sequence_length):
                x = input_[time_index + s:time_index + s + 1]
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    # pass through layer
                    upd_cell_hidden = cell(x)

                upd_hidden.append(upd_cell_hidden)

        # retain tensors in list to allow different hidden sizes
        return torch.cat(upd_hidden, dim=0)
