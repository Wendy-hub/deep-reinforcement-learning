# import random
# from collections import OrderedDict
#
# import matplotlib.pyplot as plt
#
# import torch
# from torch import nn
# from torchvision.transforms import Resize
# from sklearn.manifold import TSNE
#
# import os
# import numpy as np
# import itertools
# from baselines_wrappers import DummyVecEnv
# from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
# import time
#
# import msgpack
# from msgpack_numpy import patch as msgpack_numpy_patch
#
# msgpack_numpy_patch()
#
#
# def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
#     n_input_channels = observation_space.shape[0]
#
#     cnn = nn.Sequential(
#         nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
#         nn.ReLU(),
#         nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
#         nn.ReLU(),
#         nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
#         nn.ReLU(),
#         nn.Flatten())
#
#     # Compute shape by doing one forward pass
#     with torch.no_grad():
#         n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
#
#     out = nn.Sequential(OrderedDict([
#         ('cnn', cnn),
#         ('ll', nn.Linear(n_flatten, final_layer)),
#         ('relu', nn.ReLU())
#     ]))
#
#     return out
#
#
# class Network(nn.Module):
#     def __init__(self, env, device):
#         super().__init__()
#
#         self.num_actions = env.action_space.n
#         self.device = device
#
#         conv_net = nature_cnn(env.observation_space)
#
#         self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))
#
#     def forward(self, x):
#         return self.net(x)
#
#     def act(self, obses, epsilon):
#         obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
#         q_values = self(obses_t)
#
#         max_q_indices = torch.argmax(q_values, dim=1)
#         actions = max_q_indices.detach().tolist()
#
#         for i in range(len(actions)):
#             rnd_sample = random.random()
#             if rnd_sample <= epsilon:
#                 actions[i] = random.randint(0, self.num_actions - 1)
#
#         return actions
#
#     def load(self, load_path):
#         if not os.path.exists(load_path):
#             raise FileNotFoundError(load_path)
#
#         with open(load_path, 'rb') as f:
#             params_numpy = msgpack.loads(f.read())
#
#         params = {k: torch.as_tensor(v, device=self.device) for k, v in params_numpy.items()}
#
#         self.load_state_dict(params)
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('device:', device)
#
# make_env = lambda: make_atari_deepmind('BreakoutNoFrameskip-v4', scale_values=True)
#
# vec_env = DummyVecEnv([make_env for _ in range(1)])
#
# env = BatchedPytorchFrameStack(vec_env, k=4)  # Return PytorchLazyFrames
#
# net = Network(env, device)
# net = net.to(device)
#
# net.load('./atari_model_SCALED_lr5e-05.pack')
#
# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
#
#
# model = MyModel()
# model.fc2.register_forward_hook(get_activation('ll'))
# x = torch.randn(1, 25)
# output = model(x)
# print(activation['ll'])
#
#
# obs = env.reset()
# beginning_episode = True
# for t in range(10):
#     if isinstance(obs[0], PytorchLazyFrames):
#         act_obs = np.stack([o.get_frames() for o in obs])
#         action = net.act(act_obs, 0.0)
#     else:
#         action = net.act(obs, 0.0)
#     # fire!
#     if beginning_episode:
#         action = [1]
#         beginning_episode = False
#
#     obs, rew, done, _ = env.step(action)
#
#     tmp = obs[0]
#     if isinstance(tmp, PytorchLazyFrames):
#         # 处理PytorchLazyFrames对象
#         tmp = tmp.get_frames()
#         tmp = np.expand_dims(tmp, 0)
#     else:
#         # 处理普通数组
#         if len(tmp.shape) == 1:
#             tmp = np.expand_dims(tmp, 0)
#     states.append(tmp)
#     env.render()
#     time.sleep(0.02)
#
#     if done[0]:
#         obs = env.reset()
#         beginning_episode = True
#
#
# # 拟合t-SNE模型
# model = TSNE(n_components=2, random_state=0)
# tsne_data = model.fit_transform(state_representations)
#
# # 绘图
# plt.figure(figsize=(10,10))
# plt.scatter(tsne_data[:,0], tsne_data[:,1], alpha=0.5)
# plt.show()