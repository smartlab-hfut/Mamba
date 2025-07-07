import torch
import torch.nn as nn
import torch.nn.functional as F

            # 频带个数可调


class SimpMamba(nn.Module):
    def __init__(self, d_model, d_state=6, d_conv=4, expand=2, dt_scale=1.0, device=None, dtype=None):
        super(SimpMamba, self).__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model

        factory_kwargs = {"device": device, "dtype": dtype}

        # Projection layers
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, **factory_kwargs)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False, **factory_kwargs)

        # Time step projection
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.d_inner ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # SSM Parameters (A matrix to learn)
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D1 = nn.Parameter(torch.ones(self.d_inner, device=device))

        # Error projection layer
        self.error_proj = nn.Linear(self.d_model, self.d_inner)
        self.pred_proj = nn.Linear(self.d_state, 3)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, **factory_kwargs)
        self.activation = nn.SiLU()

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(self.d_inner, self.d_inner, batch_first=True)

        # Observer parameters (L is the observer gain matrix)
        self.L = nn.Parameter(torch.ones(328, self.d_inner, self.d_state, device=device))  # Observer gain matrix
        self.C = nn.Parameter(torch.ones(100, self.d_state, self.d_state, device=device))  # Output matrix C




    def forward(self, hidden_states, state):
        batch_size, seq_len, _ = hidden_states.shape

        # Initialize errors in the first training iteration


        # Input projection (to process voltage data)
        xz = self.in_proj(hidden_states)

        x, z = xz.chunk(2, dim=-1)

        # Convolution (extract features)
        x = x.transpose(1, 2)  # Rearranging for conv1d [batch, features, length]
        x = self.conv1d(x)[:, :, :seq_len]  # Convolution and trimming
        x = x.transpose(1, 2)  # Revert to [batch, length, features]
        x = self.activation(x)

        # Generate SSM parameters
        x_proj = self.x_proj(x)
        B, C = x_proj.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x))

        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # A matrix for system dynamics
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)  # B matrix for input dynamics

        # Initialize state (observed state estimation)
        ssm_state_t = state

        # First branch: actual system dynamics (using A, B, C)
        actual_state = torch.einsum('blnd,blnd->blnd', ssm_state_t, dA) + dB

        # print("actual_state", actual_state.size())
        # print("C",C.size())

        # Final output prediction using the corrected state
        y_pred = torch.einsum("bldn,ldn->bln", actual_state, self.C) + self.D1 * x
        zy = self.activation(z)
        y_pred = y_pred * zy
        out = self.out_proj(y_pred)

        return out, actual_state  # Return errors and updated y_pred







