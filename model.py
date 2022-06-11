import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


def feed_forward_nn(
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
        self.edge_network = feed_forward_nn(
            edge_dim, edge_num_layers, edge_hidden_dim, node_dim**2, activation
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
            adjacency_out = adjacency_in.permute(0, 2, 1)
            distance = distance.unsqueeze(-1)

            adjacency_in = F.one_hot(adjacency_in)
            adjacency_in = torch.concat([adjacency_in, distance], dim=-1)
            adjacency_in = self.edge_network(adjacency_in)
            adjacency_in = rearrange(
                adjacency_in, "b n1 n2 (d1 d2) -> b (n1 d1) (n2 d2)", d1=self.node_dim
            )

            adjacency_out = F.one_hot(adjacency_out)
            adjacency_out = torch.concat([adjacency_out, distance], dim=-1)
            adjacency_out = self.edge_network(adjacency_out)
            adjacency_out = rearrange(
                adjacency_out, "b n1 n2 (d1 d2) -> b (n1 d1) (n2 d2)", d1=self.node_dim
            )

            self._adjacency_in = adjacency_in
            self._adjacency_out = adjacency_out


        print(self._adjacency_in.shape)
        print(self._adjacency_out.shape)

        node_states = rearrange(node_states, "b n d -> b (n d)").unsqueeze(-1)

        a_in_mul = rearrange(
            (self._adjacency_in @ node_states).squeeze(2),
            "b (n d) -> b n d",
            d=self.node_dim,
        )
        a_out_mul = rearrange(
            (self._adjacency_in @ node_states).squeeze(2),
            "b (n d) -> b n d",
            d=self.node_dim,
        )
        print(a_in_mul.shape)
        print(a_out_mul.shape)

        messages = torch.concat([a_in_mul, a_out_mul], dim=2)
        messages = messages + self.message_bias

        print(messages.shape)
        return messages


en = EdgeNetwork(4)

num_bonds = 3
num_nodes = 30
batch_size = 20
node_dim = 50

node_states = torch.randn(batch_size, num_nodes, node_dim)
adjacency_in = torch.randint(0, num_bonds, (batch_size, num_nodes, num_nodes))
distance = torch.randn(batch_size, num_nodes, num_nodes)

messages1 = en(node_states, adjacency_in, distance)

idx = torch.randperm(num_nodes)

distance = distance[:, idx][:, :, idx]
adjacency_in = adjacency_in[:, idx][:, :, idx]
node_states = node_states[:, idx]

messages2 = en(node_states, adjacency_in, distance)

print(torch.allclose(messages2, messages1[:, idx]))



