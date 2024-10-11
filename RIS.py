import numpy as np
import torch
import torch.nn.functional as F

from Models_sampling import GaussianPolicy, EnsembleCritic, LaplacePolicy, Encoder#, Policy_Network   #####

from utils.data_aug import random_translate


import torch.nn as nn


class RIS(object):
	def __init__(self, state_dim, action_dim, alpha=0.1, Lambda=0.1, image_env=False, n_ensemble=10, gamma=0.99, tau=0.005, target_update_interval=1, h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, logger=None, device=torch.device("cuda")):		
		# Actor
		self.actor = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
		self.actor_target = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		# Critic
		self.critic 		= EnsembleCritic(state_dim, action_dim).to(device)
		self.critic_target 	= EnsembleCritic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

		# Subgoal policy 
		self.subgoal_net = LaplacePolicy(state_dim).to(device)
		
		self.subgoal_optimizer_hierarchy5 = torch.optim.Adam(self.subgoal_net.hierarchy5.parameters(), lr=h_lr)
		self.subgoal_optimizer_hierarchy4 = torch.optim.Adam(self.subgoal_net.hierarchy4.parameters(), lr=h_lr)
		self.subgoal_optimizer_hierarchy3 = torch.optim.Adam(self.subgoal_net.hierarchy3.parameters(), lr=h_lr)
		self.subgoal_optimizer_hierarchy2 = torch.optim.Adam(self.subgoal_net.hierarchy2.parameters(), lr=h_lr)

		#policy network
		self.policynetwork_optimizer = torch.optim.Adam(self.subgoal_net.policy_network.parameters(), lr=h_lr)

		# Encoder (for vision-based envs)
		self.image_env = image_env
		if self.image_env:
			self.encoder = Encoder(state_dim=state_dim).to(device)
			self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=enc_lr)

		# Actor-Critic Hyperparameters
		self.tau = tau
		self.target_update_interval = target_update_interval
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# High-level policy hyperparameters
		self.Lambda = Lambda
		self.n_ensemble = n_ensemble

		# Utils
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.logger = logger
		self.total_it = 0

	def save(self, folder, save_optims=False):
		torch.save(self.actor.state_dict(),		 folder + "actor.pth")
		torch.save(self.critic.state_dict(),		folder + "critic.pth")
		torch.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
		if self.image_env:
			torch.save(self.encoder.state_dict(), folder + "encoder.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
			torch.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
			if self.image_env:
				torch.save(self.encoder_optimizer.state_dict(), folder + "encoder_opti")

	def load(self, folder):
		self.actor.load_state_dict(torch.load(folder+"actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder+"critic.pth", map_location=self.device))
		self.subgoal_net.load_state_dict(torch.load(folder+"subgoal_net.pth", map_location=self.device))
		if self.image_env:
			self.encoder.load_state_dict(torch.load(folder+"encoder.pth", map_location=self.device))

	def select_action(self, state, goal):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			if self.image_env:
				state = state.view(1, 3, 84, 84)
				goal = goal.view(1, 3, 84, 84)
				state = self.encoder(state)
				goal = self.encoder(goal)
			action, _, _ = self.actor.sample(state, goal)
		return action.cpu().data.numpy().flatten()
		
	def value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
		return V

	def sample_subgoal(self, state, goal):
		subgoal_distribution, distribution2, distribution3, distribution4, distribution5 = self.subgoal_net(state, goal)   #same with original subgoal
		#subgoal = subgoal_distribution.rsample((self.n_ensemble,)) 
		#subgoal = torch.cat([distribution2.rsample((3,)), distribution3.rsample((3,)), distribution4.rsample((3,)), distribution5.rsample((3,))])
		#subgoal = torch.transpose(subgoal, 0, 1)
		#return subgoal
		return subgoal_distribution
	
	def sample_action_and_KL(self, state, goal):
		batch_size = state.size(0)
		# Sample action, subgoals and KL-divergence
		action_dist = self.actor(state, goal)
		action = action_dist.rsample()

		with torch.no_grad():
			subgoal = self.sample_subgoal(state, goal)
		
		prior_action_dist = self.actor_target(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), subgoal)
		prior_prob = prior_action_dist.log_prob(action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.action_dim)).sum(-1, keepdim=True).exp()
		prior_log_prob = torch.log(prior_prob.mean(1) + self.epsilon)
		D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob

		action = torch.tanh(action)
		return action, D_KL

	def train_highlevel_policy(self, state, goal, subgoal):
		# Compute subgoal distribution 
		#subgoal_distribution2, subgoal_distribution3, subgoal_distribution4, subgoal_distribution5 = self.subgoal_net(state, goal)#####
		subgoal_distribution_final, subgoal_distribution2, subgoal_distribution3, subgoal_distribution4, subgoal_distribution5  = self.subgoal_net(state, goal)

		with torch.no_grad():
			# Compute target value
			new_subgoal2 = subgoal_distribution2.loc
			new_subgoal3 = subgoal_distribution3.loc
			new_subgoal4 = subgoal_distribution4.loc
			new_subgoal5 = subgoal_distribution5.loc

			weight_hierarchy = [ 1.6, 1.2, 0.8, 0.4 ]

			policy_v_1_hierarchy2 = weight_hierarchy[0] * self.value(state, new_subgoal2) 
			policy_v_2_hierarchy2 = (2-weight_hierarchy[0]) * self.value(new_subgoal2, goal)  
			policy_v_hierarchy2 = torch.cat([policy_v_1_hierarchy2, policy_v_2_hierarchy2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]

			policy_v_1_hierarchy3 = weight_hierarchy[1] * self.value(state, new_subgoal3)
			policy_v_2_hierarchy3 = (2-weight_hierarchy[1]) * self.value(new_subgoal3, goal)
			policy_v_hierarchy3 = torch.cat([policy_v_1_hierarchy3, policy_v_2_hierarchy3], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
			
			policy_v_1_hierarchy4 = weight_hierarchy[2] * self.value(state, new_subgoal4)
			policy_v_2_hierarchy4 = (2-weight_hierarchy[2]) * self.value(new_subgoal4, goal)
			policy_v_hierarchy4 = torch.cat([policy_v_1_hierarchy4, policy_v_2_hierarchy4], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]

			policy_v_1_hierarchy5 = weight_hierarchy[3] * self.value(state, new_subgoal5)
			policy_v_2_hierarchy5 = (2-weight_hierarchy[3]) * self.value(new_subgoal5, goal)
			policy_v_hierarchy5 = torch.cat([policy_v_1_hierarchy5, policy_v_2_hierarchy5], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
			# Compute subgoal distance loss
			v_1 = self.value(state, subgoal)
			v_2 = self.value(subgoal, goal)
			v = torch.cat([v_1, v_2], -1).clamp(min=-100.0, max=0.0).abs().max(-1)[0]
			adv2 = - (v - policy_v_hierarchy2)
			weight2 = F.softmax(adv2/self.Lambda, dim=0)
			adv3 = - (v - policy_v_hierarchy3)
			weight3 = F.softmax(adv3/self.Lambda, dim=0)
			adv4 = - (v - policy_v_hierarchy4)
			weight4 = F.softmax(adv4/self.Lambda, dim=0)
			adv5 = - (v - policy_v_hierarchy5)
			weight5 = F.softmax(adv5/self.Lambda, dim=0)

		log_prob2 = subgoal_distribution2.log_prob(subgoal).sum(-1)
		subgoal_loss2 = - (log_prob2 * weight2).mean()

		log_prob3 = subgoal_distribution3.log_prob(subgoal).sum(-1)
		subgoal_loss3 = - (log_prob3 * weight3).mean()

		log_prob4 = subgoal_distribution4.log_prob(subgoal).sum(-1)
		subgoal_loss4 = - (log_prob4 * weight4).mean()

		log_prob5 = subgoal_distribution5.log_prob(subgoal).sum(-1)
		subgoal_loss5 = - (log_prob5 * weight5).mean()

		# Update network
		#self.subgoal_optimizer.zero_grad()
		#subgoal_loss.backward()
		#self.subgoal_optimizer.step()
		
		self.subgoal_optimizer_hierarchy5.zero_grad()
		subgoal_loss5.backward()
		self.subgoal_optimizer_hierarchy5.step()
		
		self.subgoal_optimizer_hierarchy4.zero_grad()
		subgoal_loss4.backward()
		self.subgoal_optimizer_hierarchy4.step()

		self.subgoal_optimizer_hierarchy3.zero_grad()
		subgoal_loss3.backward()
		self.subgoal_optimizer_hierarchy3.step()

		self.subgoal_optimizer_hierarchy2.zero_grad()
		subgoal_loss2.backward()
		self.subgoal_optimizer_hierarchy2.step()

		# Log variables   #we have to fix this to get better log data
		if self.logger is not None:
			self.logger.store(
				adv5 = adv5.mean().item(),
				ratio_adv5 = adv5.ge(0.0).float().mean().item(),

				adv4 = adv4.mean().item(),
				ratio_adv4 = adv4.ge(0.0).float().mean().item(),

				adv3 = adv3.mean().item(),
				ratio_adv3 = adv3.ge(0.0).float().mean().item(),

				adv2 = adv2.mean().item(),
				ratio_adv2 = adv2.ge(0.0).float().mean().item(),
			)

	def train(self, state, action, reward, next_state, done, goal, subgoal):
		""" Encode images (if vision-based environment), use data augmentation """
		if self.image_env:
			state = state.view(-1, 3, 84, 84)
			next_state = next_state.view(-1, 3, 84, 84)
			goal = goal.view(-1, 3, 84, 84)
			subgoal = subgoal.view(-1, 3, 84, 84)

			# Data augmentation
			state = random_translate(state, pad=8)
			next_state = random_translate(next_state, pad=8)
			goal = random_translate(goal, pad=8)
			subgoal = random_translate(subgoal, pad=8)

			# Stop gradient for subgoal goal and next state
			state = self.encoder(state)
			with torch.no_grad():
				goal = self.encoder(goal)
				next_state = self.encoder(next_state)
				subgoal = self.encoder(subgoal)

		""" Critic """
		# Compute target Q
		with torch.no_grad():
			next_action, _, _ = self.actor.sample(next_state, goal)
			target_Q = self.critic_target(next_state, next_action, goal)
			target_Q = torch.min(target_Q, -1, keepdim=True)[0]
			target_Q = reward + (1.0-done) * self.gamma*target_Q

		# Compute critic loss
		Q = self.critic(state, action, goal)
		critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()

		# Optimize the critic
		if self.image_env: self.encoder_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		if self.image_env: self.encoder_optimizer.step()
		self.critic_optimizer.step()

		# Stop backpropagation to encoder
		if self.image_env:
			state = state.detach()
			goal = goal.detach()
			subgoal = subgoal.detach()

		""" High-level policy learning """
		self.train_highlevel_policy(state, goal, subgoal)

		""" Actor """
		# Sample action
		action, D_KL = self.sample_action_and_KL(state, goal)

		# Compute actor loss
		Q = self.critic(state, action, goal)
		Q = torch.min(Q, -1, keepdim=True)[0]
		actor_loss = (self.alpha*D_KL - Q).mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		self.policynetwork_optimizer.zero_grad()
		actor_loss.backward(retain_graph=True)
		self.actor_optimizer.step()
		self.policynetwork_optimizer.step()

		"""policy network learning"""
		#actor_loss = (self.alpha*D_KL - Q).mean()
		#self.policynetwork_optimizer.zero_grad()
		#actor_loss.backward()
		#self.policynetwork_optimizer.step()

		# Update target networks
		self.total_it += 1
		if self.total_it % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


		# Log variables
		if self.logger is not None:
			self.logger.store(
				actor_loss   = actor_loss.item(),
				critic_loss  = critic_loss.item(),
				D_KL		 = D_KL.mean().item()				
			)
