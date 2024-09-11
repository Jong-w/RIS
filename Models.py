import torch
from torch import nn
import numpy as np

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

class hierarchy3_forth(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(hierarchy3_forth, self).__init__()	
		fc3 = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc3 += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc3 = nn.Sequential(*fc3)

		self.mean3 = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale3 = nn.Linear(hidden_dims[-1], state_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward(self, state, goal):
		h3 = self.fc3( torch.cat([state, goal], -1) )	
		mean3 = self.mean3(h3)
		scale3 = self.log_scale3(h3).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution3 = torch.distributions.laplace.Laplace(mean3, scale3)

		return mean3, scale3, distribution3


class hierarchy2_forth(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(hierarchy2_forth, self).__init__()	
		fc2 = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc2 += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc2 = nn.Sequential(*fc2)

		self.mean2 = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale2 = nn.Linear(hidden_dims[-1], state_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward(self, state, goal):
		h2 = self.fc2( torch.cat([state, goal], -1) )	
		mean2 = self.mean2(h2)
		scale2 = self.log_scale2(h2).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution2 = torch.distributions.laplace.Laplace(mean2, scale2)

		return mean2, scale2, distribution2

class LaplacePolicy(nn.Module):	
	def __init__(self, state_dim, hidden_dims=[256, 256]):	
		super(LaplacePolicy, self).__init__()	
		fc1 = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc1 += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc1 = nn.Sequential(*fc1)
		fc2 = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc2 += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc2 = nn.Sequential(*fc2)
		fc3 = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc3 += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc3 = nn.Sequential(*fc3)
		fc4 = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc4 += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc4 = nn.Sequential(*fc4)
		fc5 = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc5 += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc5 = nn.Sequential(*fc5)


		self.mean1 = nn.Linear(hidden_dims[-1], state_dim)	
		self.mean2 = nn.Linear(hidden_dims[-1], state_dim)	
		self.mean3 = nn.Linear(hidden_dims[-1], state_dim)	
		self.mean4 = nn.Linear(hidden_dims[-1], state_dim)	
		self.mean5 = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale1 = nn.Linear(hidden_dims[-1], state_dim)
		self.log_scale2 = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale3 = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale4 = nn.Linear(hidden_dims[-1], state_dim)	
		self.log_scale5 = nn.Linear(hidden_dims[-1], state_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

		self.hierarchy2_forth = hierarchy2_forth(state_dim)
		self.hierarchy3_forth = hierarchy3_forth(state_dim)

	def hierarchy3_forth(self, state, goal):
		h3 = self.fc3( torch.cat([state, goal], -1) )	
		mean3 = self.mean3(h3)
		scale3 = self.log_scale3(h3).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution3 = torch.distributions.laplace.Laplace(mean3, scale3)

		return mean3, scale3, distribution3

	def hierarchy2_forth(self, state, goal):
		h2 = self.fc2( torch.cat([state, goal], -1) )	
		mean2 = self.mean2(h2)
		scale2 = self.log_scale2(h2).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution2 = torch.distributions.laplace.Laplace(mean2, scale2)

		return mean2, scale2, distribution2

	def forward(self, state, goal):	
		'''
		h5 = self.fc5( torch.cat([state, goal], -1) )	
		mean5 = self.mean5(h5)
		scale5 = self.log_scale1(h5).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution5 = torch.distributions.laplace.Laplace(mean5, scale5)

		h4 = self.fc4( torch.cat([state, goal], -1) )	
		mean4 = self.mean4(h4)
		scale4 = self.log_scale4(h4).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution4 = torch.distributions.laplace.Laplace(mean4, scale4)'''

		mean3, scale3, distribution3 = self.hierarchy3_forth(state, goal)
		mean2, scale2, distribution2 = self.hierarchy2_forth(state, goal)

		#distribution = distribution2
		distribution = torch.distributions.laplace.Laplace((mean2+mean3)/2, (scale2+scale3)/2)
		#(distribution2.loc + distribution3.loc) / (np.linalg.norm(distribution2.loc.detach().numpy()) + np.linalg.norm(distribution3.loc.detach().numpy()))

		return distribution, distribution2, distribution3 #, distribution4, distribution5


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

