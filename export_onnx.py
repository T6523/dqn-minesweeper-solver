"""
The exported model takes boards of shape (batch, 7, 6, 6) and returns (batch, 36) values, one per cell.
It averages the network's answer over the 8 rotations / mirrors of the board, so the client needs no
symmetry code and the bot plays a little stronger than the raw network.
"""
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from dqn import DQN, Symmetric, environment, GRID_SIZE, INPUT_CHANNEL

CHECKPOINT = 'weight/best_ddqn.pt'
OUTPUT = 'docs/ddqn.onnx'
SYMMETRY_COUNT = 8


class Wrapper(nn.Module):
    """model + symmetry averaging, packaged as one module so it exports as one ONNX graph.

    For each of the 8 symmetries s:
      1. view : reorder the board's cells with table[s]           (the board as seen rotated/mirrored)
      2. ask  : run the model on that view                        (values, in the view's cell numbering)
      3. back : reorder the values with reverse_table[s]          (values, in the original cell numbering)
    then average the 8 answers.  table[s] and reverse_table[s] undo each other, so step 3 cancels step 1.
    """

    def __init__(self, model, grid_size=GRID_SIZE):
        super().__init__()
        self.model = model
        self.grid_size = grid_size
        symmetric = Symmetric(grid_size)
        # buffers are tensors that belong to the module: they get saved inside the ONNX file as constants
        self.register_buffer('table', symmetric.table)                  # (8, n*n): new position -> old position
        self.register_buffer('reverse_table', symmetric.reverse_table)  # (8, n*n): old position -> new position

    def forward(self, x):                                    # x: (batch, channel, n, n)
        batch, channel = x.shape[0], x.shape[1]
        flat = torch.flatten(x, start_dim=2)                 # (batch, channel, n*n): one row of cells per channel

        # 1. the 8 views. Indexing the last dimension with a table row reorders the cells.
        views = [flat[:, :, self.table[s]] for s in range(SYMMETRY_COUNT)]   # 8 x (batch, channel, n*n)
        views = torch.stack(views, dim=1)                                    # (batch, 8, channel, n*n)
        views = views.reshape(batch * SYMMETRY_COUNT, channel, self.grid_size, self.grid_size)

        # 2. ask the model about all 8 views of every board in one go
        q = self.model(views)                                                # (batch*8, n*n)
        q = q.reshape(batch, SYMMETRY_COUNT, -1)                             # (batch, 8, n*n)

        # 3. translate each answer back to the original cell numbering, then average
        q_back = [q[:, s][:, self.reverse_table[s]] for s in range(SYMMETRY_COUNT)]   # 8 x (batch, n*n)
        return torch.stack(q_back, dim=1).mean(dim=1)                        # (batch, n*n)


def load_model(checkpoint):
    model = DQN()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return model.eval()


def check_rotation(wrapper):
    """Because the wrapper averages over all 8 symmetries, rotating its input must rotate its output the same way.
    This catches a wrong table: using the same table for both the view and the translate-back step fails here."""
    symmetric = Symmetric(GRID_SIZE)
    x = torch.rand(4, INPUT_CHANNEL, GRID_SIZE, GRID_SIZE)
    with torch.no_grad():
        base = wrapper(x)
        for s in range(SYMMETRY_COUNT):
            ids = torch.full((4,), s)
            rotated_input = wrapper(symmetric.reorder(x, ids))             # rotate the boards, then ask
            rotated_output = torch.gather(base, 1, symmetric.table[ids])   # ask, then rotate the answer
            assert torch.allclose(rotated_input, rotated_output, atol=1e-5), f'symmetry {s}: wrapper is not symmetric'
    print('rotation check passed: rotating the input rotates the output, for all 8 symmetries')


def export(wrapper, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    dummy = torch.zeros(1, INPUT_CHANNEL, GRID_SIZE, GRID_SIZE)            # only fixes the shape; values don't matter
    torch.onnx.export(
        wrapper, dummy, path,
        input_names=['observation'], output_names=['q_values'],
        dynamic_axes={'observation': {0: 'batch'}, 'q_values': {0: 'batch'}},   # any number of boards per call
        opset_version=17,
        dynamo=False,                                                      # the classic exporter: simplest, most compatible
    )
    print(f'exported {path} ({os.path.getsize(path) / 1024:.0f} KB)')


def real_observations(count, seed=0):
    """Observations from random games: boards at every stage, like the ones the bot will see."""
    random.seed(seed)
    env, out = environment(), []
    observe, mask = env.reset()
    while len(out) < count:
        out.append(observe)
        observe, mask, _, done = env.step(int(random.choice(np.flatnonzero(mask))))
        if done:
            observe, mask = env.reset()
    return np.stack(out)


def check_onnx(wrapper, path):
    """The saved file must give the same numbers as the PyTorch wrapper, for any batch size."""
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    observations = real_observations(300)
    with torch.no_grad():
        expected = wrapper(torch.from_numpy(observations)).numpy()
    got = session.run(None, {'observation': observations})[0]
    worst = float(np.abs(got - expected).max())
    for size in (1, 5):                                                    # the batch dimension is dynamic
        sliced = session.run(None, {'observation': observations[:size]})[0]
        assert sliced.shape == (size, GRID_SIZE ** 2), sliced.shape
        assert np.allclose(sliced, got[:size], atol=1e-5)
    assert worst < 1e-4, f'ONNX and PyTorch disagree by {worst}'
    print(f'onnx check passed: max difference to PyTorch {worst:.1e} on {len(observations)} real boards, '
          f'batch sizes 1 and 5 work')


if __name__ == '__main__':
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    output = sys.argv[2] if len(sys.argv) > 2 else OUTPUT
    wrapper = Wrapper(load_model(checkpoint)).eval()
    check_rotation(wrapper)
    export(wrapper, output)
    check_onnx(wrapper, output)
