import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NoisyLinear import NoisyLinear
from model.video_cnn import VideoCNN


class Network(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # # set advantage layer

        # # set value layer

        self.video_cnn = VideoCNN(se=False)
        self.gru = nn.GRU(512, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        # self.v_cls = nn.Linear(1024 * 2, self.args["model"]["numclasses"])
        self.v_cls = NoisyLinear(1024 * 2, int(out_dim) * atom_size)
        self.v_cls2 = NoisyLinear(1024 * 2, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        return dist

    def adv(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        self.gru.flatten_parameters()

        with torch.cuda.amp.autocast():
            f_x = self.video_cnn(x)
            f_x = self.dropout(f_x)
        f_x = f_x.float()

        h, _ = self.gru(f_x)

        f_x = self.v_cls(self.dropout(h)).mean(1)

        return f_x

    def val(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        self.gru.flatten_parameters()

        with torch.cuda.amp.autocast():
            f_x = self.video_cnn(x)
            f_x = self.dropout(f_x)
        f_x = f_x.float()

        h, _ = self.gru(f_x)

        f_x = self.v_cls2(self.dropout(h)).mean(1)

        return f_x

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""

        state_value = self.val(x)
        action_value = self.adv(x)
        # 根据不同公式将两个输出合并
        action_value = action_value - action_value.mean(dim=1, keepdim=True)  # 按行求平均值，保持维度便于计算

        Q = state_value + action_value
        Q = F.softmax(Q, dim=-1)
        # Q = Q.clamp(min=1e-3)
        return Q

        # self.gru.flatten_parameters()
        #
        # with torch.cuda.amp.autocast():
        #     f_x = self.video_cnn(x)
        #     f_x = self.dropout(f_x)
        # f_x = f_x.float()
        #
        # h, _ = self.gru(f_x)
        #
        # f_x = self.v_cls(self.dropout(h)).mean(1)
        #
        # return f_x

    def reset_noise(self):
        """Reset all noisy layers."""
        # self.advantage_hidden_layer.reset_noise()
        # self.advantage_layer.reset_noise()
        # self.value_hidden_layer.reset_noise()
        # self.value_layer.reset_noise()
        self.v_cls.reset_noise()
        self.v_cls2.reset_noise()
