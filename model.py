import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


def feed_forward(
    input_dim, num_hidden_layers, hidden_dim, output_dim, activation=nn.ReLU
):
    """
    Fully connected feed forward neural network

    Args:
        input_dim: last dimension of input
        num_hidden_layers: number of hidden layers
        output_dim: last dimension of output
        activation: activation function
    """
    dims = [input_dim] + [hidden_dim] * num_hidden_layers
    layers = []
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class EdgeNetwork(nn.Module):
    def __init__(
        self,
        edge_dim,
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
                edge_dim, edge_num_layers, edge_hidden_dim, node_dim ** 2, activation
            ),
            Rearrange("b n1 n2 (d1 d2) -> b n1 n2 d1 d2", d1=node_dim),
        )

        self.message_bias = torch.nn.Parameter(torch.zeros(2 * node_dim))
        self.node_dim = node_dim

    def forward(self, node_states, adjacency_in, distance, reuse_graph=False):
        """
        Forward pass

        Args:
            node_states: [batch_size, num_nodes, node_dim] (h_{t-1})
            adjacency_in: [batch_size, num_nodes, num_nodes] integer
            distance: [batch_size, num_nodes, num_nodes]
        """
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


en = EdgeNetwork(4)

num_bonds = 3
num_nodes = 30
batch_size = 20
node_dim = 50

node_states = torch.randn(batch_size, num_nodes, node_dim)
adjacency_in = torch.randint(0, num_bonds, (batch_size, num_nodes, num_nodes))
distance = torch.randn(batch_size, num_nodes, num_nodes)

messages = en(node_states, adjacency_in, distance)

print(messages.shape)
