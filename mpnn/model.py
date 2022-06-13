import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


# fully connected feed forward neural network util

def feed_forward(
    input_dim, num_hidden_layers, hidden_dim, output_dim, activation=nn.ReLU
):
    dims = [input_dim] + [hidden_dim] * num_hidden_layers
    layers = []
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# Edge network message function

class EdgeNetwork(nn.Module):
    def __init__(
        self,
        num_edge_class,
        node_dim=50,
        edge_num_layers=4,
        edge_hidden_dim=50,
        activation=nn.ReLU,
    ):
        super().__init__()

        # "To allow vector valued edge features, we propose the message
        # function M(h_v, h_w, e_{vw}) = A(e_{vw})h_w where A(e_{vw}) is a neural
        # network which maps the edge vector e_vw to a d x d matrix."
        self.edge_network = nn.Sequential(
            feed_forward(
                num_edge_class + 1,
                edge_num_layers,
                edge_hidden_dim,
                node_dim ** 2,
                activation,
            ),
            Rearrange("b n1 n2 (d1 d2) -> b n1 n2 d1 d2", d1=node_dim),
        )

        self.message_bias = torch.nn.Parameter(torch.zeros(2 * node_dim))
        self.node_dim = node_dim

    def forward(self, node_states, adjacency_in, distance, reuse_graph=False):
        if not reuse_graph:
            distance = distance.unsqueeze(-1)
            adjacency_in = F.one_hot(adjacency_in)
            adjacency_in = torch.concat([adjacency_in, distance], dim=-1)

            # adjacency out is tranpose of adjacency in
            adjacency_out = adjacency_in.permute(0, 2, 1, 3)

            self._adjacency_in = self.edge_network(adjacency_in)
            self._adjacency_out = self.edge_network(adjacency_out)

        # multiply each node state (dimension v in node_states) by d x d edge
        # network matrix (dimensions ij in adjacency)
        # m_v = \sum_w A(e_{vw}) h_w
        # m_{vi} = \sum_w \sum_j A(e_{vw})_{ij} h_{wj}
        m_in = torch.einsum("bvwij,bwj->bvi", self._adjacency_in, node_states)
        m_out = torch.einsum("bvwij,bwj->bvi", self._adjacency_out, node_states)

        messages = torch.concat([m_in, m_out], dim=2)
        messages = messages + self.message_bias

        return messages


# Wrapper for nn.GRUCell that can handle additional node dimension


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """"""
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, messages, node_states):
        b = messages.shape[0]
        node_states = rearrange(node_states, "b n d -> (b n) d")
        messages = rearrange(messages, "b n d -> (b n) d")
        node_states = self.cell(messages, node_states)
        return rearrange(node_states, "(b n) d -> b n d", b=b)


# Graph readout function from GG-NN https://arxiv.org/abs/1511.05493

class GraphLevelOutput(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_output_hidden_layers,
        output_dim,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.i = feed_forward(
            input_dim=input_dim,
            num_hidden_layers=num_output_hidden_layers,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
        )
        self.j = feed_forward(
            input_dim=input_dim,
            num_hidden_layers=num_output_hidden_layers,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(self, node_states, mask):
        i_out = self.i(node_states)
        j_out = self.j(node_states)
        # tanh not used on j_out for regression
        gated = torch.sigmoid(i_out) + j_out

        gated = gated * mask.unsqueeze(-1)
        # sum over nodes
        return gated.sum(dim=1)


class MPNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_edge_class,
        node_dim=50,
        num_propagation_steps=6,
        num_output_hidden_layers=1,
        edge_num_layers=4,
        edge_hidden_dim=50,
        hidden_dim=200,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.message_function = EdgeNetwork(
            num_edge_class=num_edge_class,
            node_dim=node_dim,
            edge_num_layers=edge_num_layers,
            edge_hidden_dim=edge_hidden_dim,
            activation=activation,
        )

        self.update_function = GRU(input_dim=node_dim * 2, hidden_dim=node_dim)

        self.readout = GraphLevelOutput(
            input_dim=node_dim + input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_output_hidden_layers=num_output_hidden_layers,
        )

        self.node_dim = node_dim
        self.num_propagation_steps = num_propagation_steps

    def forward(self, node_input, adjacency_in, distance, mask):

        # pad node input to match internal node dimension
        pad_input = F.pad(node_input, (0, self.node_dim - node_input.shape[-1]))

        # node representations h_t over time t
        h = [pad_input]

        for t in range(self.num_propagation_steps):
            messages = self.message_function(
                h[-1], adjacency_in, distance, reuse_graph=(t != 0)
            )
            h.append(self.update_function(messages, h[-1]))

        # combine final node state with original features
        h_final = torch.concat([h[-1], node_input], dim=-1)

        out = self.readout(h_final, mask)
        return out
