import torch
from torch import nn
import numpy as np

class DilatedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, radius=10, device='cuda'):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size).to(device)
        self.index = torch.arange(0, radius * hidden_size, radius)
        self.dilation = 0
        self.device = device

    def forward(self, state, hidden):
        """At each time step only the corresponding part of the state is updated
        and the output is pooled across the previous c out- puts."""
        d_idx = self.dilation_idx.to(self.device)
        hx, cx = hidden

        hx[:, d_idx], cx[:, d_idx] = self.rnn(state, (hx[:, d_idx], cx[:, d_idx]))
        detached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detached_hx = detached_hx.view(detached_hx.shape[0], self.hidden_size, self.radius-1)
        detached_hx = detached_hx.sum(-1)

        y = (hx[:, d_idx] + detached_hx) / self.radius
        return y, (hx, cx)

    def masked_idx(self, dilated_idx):
        """Because we do not want to have gradients flowing through all
        parameters but only at the dilation index, this function creates a
        'negated' version of dilated_index, everything EXCEPT these indices."""
        masked_idx = torch.arange(1, self.radius * self.hidden_size + 1)
        masked_idx[dilated_idx] = 0
        masked_idx = masked_idx.nonzero()
        masked_idx = masked_idx - 1
        return masked_idx

    @property
    def dilation_idx(self):
        """Keep track at which dilation we currently we are."""
        dilation_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.radius
        return dilation_idx


def Normalizer(latent_representation):
    minimum = latent_representation.detach().min()
    maximum = latent_representation.detach().max()
    latent_representation_normalized = (latent_representation - minimum) / (maximum - minimum + 1e-9)
    return latent_representation_normalized


class Policy_Network(nn.Module):
    def __init__(self, d, time_horizon, num_workers, device):
        super().__init__()
        self.device = device
        self.Mrnn = DilatedLSTM(799, 3, time_horizon, device=device).to(device)
        self.num_workers = num_workers
		
    def forward(self, z, goal_5_norm, goal_4_norm, goal_3_norm, hidden, mask):
        goal_x_info = torch.cat(([goal_5_norm.detach(), goal_4_norm.detach(), goal_3_norm.detach(), z]), dim=1).to(self.device)
        hidden = (mask * hidden[0], mask * hidden[1])
        policy_network_result, hidden = self.Mrnn(goal_x_info, hidden)
        policy_network_result = (policy_network_result - policy_network_result.detach().min(1, keepdim=True)[0]) / \
                                (policy_network_result.detach().max(1, keepdim=True)[0] - policy_network_result.detach().min(1, keepdim=True)[0])

        return policy_network_result.type(torch.int), hidden

def init_hidden(n_workers, h_dim, device, grad=False):
    return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
            torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))

class Goal_Normalizer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
    def forward(self, goal_5, goal_4, goal_3, goal_2):
        minimum = min(goal_5.detach().min(), goal_4.detach().min(), goal_3.detach().min(), goal_2.detach().min())
        maximum = max(goal_5.detach().max(), goal_4.detach().max(), goal_3.detach().max(), goal_2.detach().max())
        goal_5_norm = (goal_5 - minimum) / (maximum - minimum)
        goal_4_norm = (goal_4 - minimum) / (maximum - minimum)
        goal_3_norm = (goal_3 - minimum) / (maximum - minimum)
        goal_2_norm = (goal_2 - minimum) / (maximum - minimum)
        return goal_5_norm, goal_4_norm, goal_3_norm, goal_2_norm




""" Actor """

class GaussianPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
		super(GaussianPolicy, self).__init__()
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
		self.logstd_linear = nn.Linear(hidden_dims[-1], action_dim)

		self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2

	def forward(self, state, goal):
		x = self.fc(torch.cat([state, goal], -1))
		mean = self.mean_linear(x)
		log_std = self.logstd_linear(x)
		std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
		normal = torch.distributions.Normal(mean, std)
		return normal

	def sample(self, state, goal):
		normal = self.forward(state, goal)
		x_t = normal.rsample()
		action = torch.tanh(x_t)
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
		log_prob = log_prob.sum(-1, keepdim=True)
		mean = torch.tanh(normal.mean)
		return action, log_prob, mean



""" Critic """

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
		super(Critic, self).__init__()
		fc = [nn.Linear(2*state_dim + action_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
		fc += [nn.Linear(hidden_dims[-1], 1)]
		self.fc = nn.Sequential(*fc)

	def forward(self, state, action, goal):
		x = torch.cat([state, action, goal], -1)
		return self.fc(x)


class EnsembleCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], n_Q=2):
		super(EnsembleCritic, self).__init__()
		ensemble_Q = [Critic(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims) for _ in range(n_Q)]			
		self.ensemble_Q = nn.ModuleList(ensemble_Q)
		self.n_Q = n_Q

	def forward(self, state, action, goal):
		Q = [self.ensemble_Q[i](state, action, goal) for i in range(self.n_Q)]
		Q = torch.cat(Q, dim=-1)
		return Q

""" High-level policy """

class hierarchy5(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(hierarchy5, self).__init__()	
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], state_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward_forth(self, state, goal):
		h = self.fc( torch.cat([state, goal], -1) )	

		return h

	def forward_back(self, h, hierarchies_selected):
		h = hierarchies_selected[:, 0].unsqueeze(dim=1) * h
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.laplace.Laplace(mean, scale)

		return distribution
	
class hierarchy4(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(hierarchy4, self).__init__()	
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], state_dim)	
		self.hierarchy_forup = nn.Linear(hidden_dims[-1], hidden_dims[-1])	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward_forth(self, state, goal):
		h = self.fc( torch.cat([state, goal], -1) )	

		return h
	
	def forward_back(self, h, h_up, hierarchies_selected):
		h_up = self.hierarchy_forup(h_up.detach())
		h = hierarchies_selected[:, 1].unsqueeze(dim=1) * h
		h = (h + h_up)
		h = Normalizer(h)
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.laplace.Laplace(mean, scale)

		return distribution

class hierarchy3(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(hierarchy3, self).__init__()	
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], state_dim)	
		self.hierarchy_forup = nn.Linear(hidden_dims[-1], hidden_dims[-1])	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward_forth(self, state, goal):
		h = self.fc( torch.cat([state, goal], -1) )	

		return h
	
	def forward_back(self, h, h_up, hierarchies_selected):
		h_up = self.hierarchy_forup(h_up.detach())
		h = hierarchies_selected[:, 2].unsqueeze(dim=1) * h
		h = (h + h_up)
		h = Normalizer(h)
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.laplace.Laplace(mean, scale)

		return distribution
	
	
class hierarchy2(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(hierarchy2, self).__init__()	
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], state_dim)	
		self.hierarchy_forup = nn.Linear(hidden_dims[-1], hidden_dims[-1])	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward_forth(self, state, goal):
		h = self.fc( torch.cat([state, goal], -1) )	

		return h
	
	def forward_back(self, h, h_up):
		h_up = self.hierarchy_forup(h_up.detach())
		h = (h + h_up)
		h = Normalizer(h)
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.laplace.Laplace(mean, scale)

		return distribution

class LaplacePolicy(nn.Module):	
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(LaplacePolicy, self).__init__()
		
		self.device=torch.device("cuda")

		self.policy_network = Policy_Network(state_dim, 1, 2048, self.device)

		self.hierarchy5 = hierarchy5(state_dim)
		self.hierarchy4 = hierarchy4(state_dim)
		self.hierarchy3 = hierarchy3(state_dim)
		self.hierarchy2 = hierarchy2(state_dim)

		self.goal_normalizer = Goal_Normalizer(state_dim)

		self.hidden_policy_network = init_hidden(2048, 300 * 4 * 31, device=self.device, grad=True)
		self.masks = [torch.ones(2048, 1).to(self.device) for _ in range(2 * 1 + 1)]
		self.hierarchies_selected = torch.ones_like(torch.empty(2048, 3))

	def forward(self, state, goal):	
		
		
		h5 = self.hierarchy5.forward_forth(state, goal)
		h4 = self.hierarchy4.forward_forth(state, goal)
		h3 = self.hierarchy3.forward_forth(state, goal)
		h2 = self.hierarchy2.forward_forth(state, goal)

		h5, h4, h3, h2 = self.goal_normalizer(h5, h4, h3, h2)
		self.hierarchies_selected, self.hidden_policy_network = self.policy_network(state, h5, h4, h3, self.hidden_policy_network, self.masks[-1])
		
		distribution5 = self.hierarchy5.forward_back(h5, self.hierarchies_selected)
		distribution4 = self.hierarchy4.forward_back(h4, h5, self.hierarchies_selected)
		distribution3 = self.hierarchy3.forward_back(h3, h4, self.hierarchies_selected)
		distribution2 = self.hierarchy2.forward_back(h2, h3)
		#distribution = distribution2
		#distribution = torch.distributions.laplace.Laplace((mean2+mean3)/2, (scale2+scale3)/2)
		#(distribution2.loc + distribution3.loc) / (np.linalg.norm(distribution2.loc.detach().numpy()) + np.linalg.norm(distribution3.loc.detach().numpy()))

		return distribution2, distribution3, distribution4, distribution5


""" Encoder """
def weights_init_encoder(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Encoder(nn.Module):
	def __init__(self, n_channels=3, state_dim=16):
		super(Encoder, self).__init__()
		self.encoder_conv = nn.Sequential(
			nn.Conv2d(n_channels, 32, 3, 2), nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2), nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2), nn.ReLU(),
			nn.Conv2d(32, 32, 3, 1), nn.ReLU()
		)
		self.fc = nn.Linear(32*7*7, state_dim)
		self.apply(weights_init_encoder)

	def forward(self, x):
		h = self.encoder_conv(x).view(x.size(0), -1)
		state = self.fc(h)
		return state

