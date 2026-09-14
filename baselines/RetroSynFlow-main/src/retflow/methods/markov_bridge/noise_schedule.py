import numpy as np
import torch

from retflow.utils.wrappers import GraphWrapper


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif noise_schedule == "custom":
            raise NotImplementedError()
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2  # (timesteps + 1, )

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, device):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == "polynomial":
            betas = polynomial_beta_schedule_discrete(timesteps)
        elif noise_schedule == "linear":
            betas = linear_beta_schedule_discrete(timesteps)
        elif noise_schedule == "custom":
            betas = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer("betas", torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)  # type: ignore

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        self.alphas_bar = self.alphas_bar.to(device)
        self.betas = self.betas.to(device)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]  # type: ignore

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar[t_int.long()]


class AbsorbingStateTransition:
    def __init__(self, abs_state: int, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes

        self.u_x = torch.zeros(1, self.X_classes, self.X_classes)
        self.u_x[:, :, abs_state] = 1

        self.u_e = torch.zeros(1, self.E_classes, self.E_classes)
        self.u_e[:, :, abs_state] = 1

        self.u_y = torch.zeros(1, self.y_classes, self.y_classes)
        self.u_e[:, :, abs_state] = 1

    def get_Qt(self, beta_t):
        """Returns two transition matrix for X and E"""
        beta_t = beta_t.unsqueeze(1)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes).unsqueeze(0)
        return q_x, q_e, q_y

    def get_Qt_bar(self, alpha_bar_t):
        """beta_t: (bs)
        Returns transition matrices for X and E"""

        alpha_bar_t = alpha_bar_t.unsqueeze(1)

        q_x = (
            alpha_bar_t * torch.eye(self.X_classes).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_x
        )  # (bs, dx_in, dx_out)
        q_e = (
            alpha_bar_t * torch.eye(self.E_classes).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_e
        )  # (bs, de_in, de_out)
        q_y = (
            alpha_bar_t * torch.eye(self.y_classes).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_y
        )

        return q_x, q_e, q_y


class InterpolationTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes

    def get_Qt(self, beta_t, X_T, E_T, y_T, node_mask, device):
        """X_T (bs, n, dx), E_T (bs, n, n, de)"""
        """ Returns two transition matrix for X and E"""

        beta_t = beta_t.unsqueeze(1)  # (bs, 1, 1)
        beta_t = beta_t.to(device)

        q_x_1 = (1 - beta_t) * torch.eye(self.X_classes, device=device)  # (bs, dx, dx)
        q_x_2 = (
            beta_t.unsqueeze(-1)
            * torch.ones_like(X_T).unsqueeze(-1)
            * X_T.unsqueeze(-2)
        )  # (bs, n, dx, dx)
        q_x = q_x_1.unsqueeze(1) + q_x_2
        q_x[~node_mask] = torch.eye(q_x.shape[-1], device=device)

        q_e_1 = (1 - beta_t) * torch.eye(self.E_classes, device=device)  # (bs, de, de)
        q_e_2 = (
            beta_t.unsqueeze(-1).unsqueeze(-1)
            * torch.ones_like(E_T).unsqueeze(-1)
            * E_T.unsqueeze(-2)
        )  # (bs, n, n, de, de)
        q_e = q_e_1.unsqueeze(1).unsqueeze(1) + q_e_2

        diag = (
            torch.eye(E_T.shape[1], dtype=torch.bool)
            .unsqueeze(0)
            .expand(E_T.shape[0], -1, -1)
        )
        q_e[diag] = torch.eye(q_e.shape[-1], device=device)

        edge_mask = node_mask[:, None, :] & node_mask[:, :, None]
        q_e[~edge_mask] = torch.eye(q_e.shape[-1], device=device)

        return GraphWrapper(X=q_x, E=q_e, y=y_T)

    def get_Qt_bar(self, alpha_bar_t, X_T, E_T, y_T, node_mask, device):
        """
        alpha_bar_t: (bs, 1)
        X_T: (bs, n, dx)
        E_T: (bs, n, n, de)
        y_T: (bs, dy)

        Returns transition matrices for X, E, and y
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)  # (bs, 1, 1)
        alpha_bar_t = alpha_bar_t.to(device)

        q_x_1 = alpha_bar_t * torch.eye(self.X_classes, device=device)  # (bs, dx, dx)
        q_x_2 = (
            (1 - alpha_bar_t).unsqueeze(-1)
            * torch.ones_like(X_T).unsqueeze(-1)
            * X_T.unsqueeze(-2)
        )  # (bs, n, dx, dx)
        q_x = q_x_1.unsqueeze(1) + q_x_2
        q_x[~node_mask] = torch.eye(q_x.shape[-1], device=device)

        q_e_1 = alpha_bar_t * torch.eye(self.E_classes, device=device)  # (bs, de, de)
        q_e_2 = (
            (1 - alpha_bar_t).unsqueeze(-1).unsqueeze(-1)
            * torch.ones_like(E_T).unsqueeze(-1)
            * E_T.unsqueeze(-2)
        )  # (bs, n, n, de, de)
        q_e = q_e_1.unsqueeze(1).unsqueeze(1) + q_e_2

        diag = (
            torch.eye(E_T.shape[1], dtype=torch.bool)
            .unsqueeze(0)
            .expand(E_T.shape[0], -1, -1)
        )
        q_e[diag] = torch.eye(q_e.shape[-1], device=device)

        edge_mask = node_mask[:, None, :] & node_mask[:, :, None]
        q_e[~edge_mask] = torch.eye(q_e.shape[-1], device=device)

        return GraphWrapper(X=q_x, E=q_e, y=y_T)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def polynomial_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas = (1 - 2 * s) * (1 - (x / steps) ** 2)[1:]
    betas = 1 - alphas
    return betas.squeeze()


def linear_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas = (1 - 2 * s) * (1 - (x / steps))[1:]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)
