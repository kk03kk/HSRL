from __future__ import annotations

import torch
from torch import nn

HiddenState = tuple[torch.Tensor, torch.Tensor]


class ProjectedLSTMCell(nn.Module):
    """LSTM cell with a linear projection from cell state to hidden output."""

    def __init__(
        self,
        input_size: int,
        cell_size: int,
        projection_size: int,
        forget_bias: float = 1.0,
    ):
        super().__init__()
        self.cell_size = cell_size
        self.projection_size = projection_size
        self.forget_bias = forget_bias
        self.gates = nn.Linear(input_size + projection_size, cell_size * 4)
        self.projection = nn.Linear(cell_size, projection_size, bias=False)

    def forward(self, inputs: torch.Tensor, state: HiddenState) -> HiddenState:
        previous_projection, previous_cell = state
        gates = self.gates(torch.cat((inputs, previous_projection), dim=-1))
        input_gate, candidate, forget_gate, output_gate = gates.chunk(4, dim=-1)
        cell = (
            torch.sigmoid(forget_gate + self.forget_bias) * previous_cell
            + torch.sigmoid(input_gate) * torch.tanh(candidate)
        )
        raw_output = torch.sigmoid(output_gate) * torch.tanh(cell)
        return self.projection(raw_output), cell
