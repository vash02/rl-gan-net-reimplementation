#!/usr/bin/env python
# coding: utf-8

# #### Mounting drive location

# In[ ]:


# Reading the input data
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')


# #### Loading pretrained models

# In[ ]:


import torch
model_dir = '/content/drive/MyDrive/RLFinalProjectFiles/models/'

encoder_state_dict = torch.load(model_dir + "ae_encoder.pth")  # Load encoder weights
# encoder.eval()  # Set encoder to evaluation mode

decoder_state_dict = torch.load(model_dir + "ae_decoder.pth")  # Load encoder weights
# decoder.eval()  # Set encoder to evaluation mode

gen_state_dict = torch.load(model_dir + "generator1.pth")  # Load generator weights
# gen.eval()  # Set generator to evaluation mode

disc_state_dict = torch.load(model_dir + "discriminator1.pth")  # Load decoder weights
# disc.eval()  # Set decoder to evaluation mode

actor_state_dict = torch.load(model_dir + "ddpg_RLGAN_actor1.pth")  # Load actor weights
# actor.eval()  # Set actor to evaluation mode

critic_state_dict = torch.load(model_dir + "ddpg_RLGAN_critic1.pth")  # Load critic weights
# critic.eval()  # Set critic to evaluation mode


# #### AE

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, num_points):
        super(Encoder, self).__init__()
        self.num_points = num_points

        self.encoder_conv1 = nn.Conv1d(3, 64, 1)
        self.encoder_conv2 = nn.Conv1d(64, 128, 1)
        self.encoder_fc1 = nn.Linear(128 * num_points, 1024)
        self.encoder_fc2 = nn.Linear(1024, 256)

        #  # Cast weight tensor and bias tensor to Double data type
        # self.encoder_conv1.weight.data = self.encoder_conv1.weight.data.double()
        # self.encoder_conv1.bias.data = self.encoder_conv1.bias.data.double()
        # self.encoder_conv2.weight.data = self.encoder_conv2.weight.data.double()
        # self.encoder_conv2.bias.data = self.encoder_conv2.bias.data.double()

    def forward(self, x):
        x = x.to(self.encoder_conv1.bias.dtype)
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x = x.view(-1, 128 * self.num_points)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, num_points):
        super(Decoder, self).__init__()
        self.num_points = num_points

        self.decoder_fc1 = nn.Linear(256, 1024)
        self.decoder_fc2 = nn.Linear(1024, 128 * num_points)
        self.decoder_conv1 = nn.Conv1d(128, 64, 1)
        self.decoder_conv2 = nn.Conv1d(64, 3, 1)

    def forward(self, x):
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        x = x.view(-1, 128, self.num_points)
        x = F.relu(self.decoder_conv1(x))
        x = self.decoder_conv2(x)
        return x


# #### VAE

# In[ ]:


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PointCloudVariationalAutoencoder(nn.Module):
#     def __init__(self, num_points, latent_dim):
#         super(PointCloudVariationalAutoencoder, self).__init__()
#         self.num_points = num_points
#         self.latent_dim = latent_dim

#         # Encoder
#         self.encoder_conv1 = nn.Conv1d(3, 64, 1)
#         self.encoder_conv2 = nn.Conv1d(64, 128, 1)
#         self.encoder_fc1 = nn.Linear(128 * num_points, 2048)
#         self.encoder_fc2_mean = nn.Linear(2048, latent_dim)
#         self.encoder_fc2_logvar = nn.Linear(2048, latent_dim)

#         # Decoder
#         self.decoder_fc1 = nn.Linear(latent_dim, 2048)
#         self.decoder_fc2 = nn.Linear(2048, 128 * num_points)
#         self.decoder_conv1 = nn.Conv1d(128, 64, 1)
#         self.decoder_conv2 = nn.Conv1d(64, 3, 1)

#         # Upsampling layer for encoder
#         self.upsample = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)

#     def forward(self, x):
#         # Encoder
#         x = F.relu(self.encoder_conv1(x))
#         x = F.relu(self.encoder_conv2(x))
#         if x.size(-1) == 1024:  # Check if input size is 1024
#             x = self.upsample(x)
#         x = x.view(-1, 128 * self.num_points)
#         x = F.relu(self.encoder_fc1(x))
#         z_mean = self.encoder_fc2_mean(x)
#         z_logvar = self.encoder_fc2_logvar(x)

#         # Reparameterization trick
#         epsilon = torch.randn_like(z_logvar)
#         z = z_mean + torch.exp(0.5 * z_logvar) * epsilon

#         # Decoder
#         x = F.relu(self.decoder_fc1(z))
#         x = F.relu(self.decoder_fc2(x))
#         x = x.view(-1, 128, self.num_points)
#         x = F.relu(self.decoder_conv1(x))
#         x = self.decoder_conv2(x)

#         return x, z

#     def get_encoder(self):
#         return Encoder(self.num_points, self.latent_dim)

#     def get_decoder(self):
#         return Decoder()

# class Encoder(nn.Module):
#     def __init__(self, num_points, latent_dim):
#         super(Encoder, self).__init__()
#         self.num_points = num_points
#         self.latent_dim = latent_dim

#         self.encoder_conv1 = nn.Conv1d(3, 64, 1)
#         self.encoder_conv2 = nn.Conv1d(64, 128, 1)
#         self.encoder_fc1 = nn.Linear(128 * num_points, 2048)
#         self.encoder_fc2_mean = nn.Linear(2048, latent_dim)
#         self.encoder_fc2_logvar = nn.Linear(2048, latent_dim)

#         # Upsampling layer for encoder
#         self.upsample = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)

#     def forward(self, x):
#         x = F.relu(self.encoder_conv1(x))
#         x = F.relu(self.encoder_conv2(x))
#         if x.size(-1) == 1024:  # Check if input size is 1024
#             x = self.upsample(x)
#         x = x.view(-1, 128 * self.num_points)
#         x = F.relu(self.encoder_fc1(x))
#         z_mean = self.encoder_fc2_mean(x)
#         z_logvar = self.encoder_fc2_logvar(x)
#         return z_mean, z_logvar

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.feature_num = 256
#         self.output_point_number = 2048

#         self.linear1 = nn.Linear(self.feature_num, self.output_point_number*2)
#         self.linear2 = nn.Linear(self.output_point_number*2, self.output_point_number*3)
#         self.linear3 = nn.Linear(self.output_point_number*3, self.output_point_number*4)
#         self.linear_out = nn.Linear(self.output_point_number*4, self.output_point_number*3)

#         # Special initialization for linear_out to get a uniform distribution over the space
#         self.linear_out.bias.data.uniform_(-1, 1)

#     def forward(self, x):
#         # Reshape from feature vector NxC to NxC
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear3(x))
#         x = self.linear_out(x)

#         return x.view(-1, 3, self.output_point_number)


# #### GAN

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim=1, gfvs_dim=256, conv_dim=64):
        super(Generator, self).__init__()
        self.gfvs_dim = gfvs_dim

        # Calculate the initial image width based on the GFVs shape
        self.initial_image_width = int(gfvs_dim / (conv_dim * 4))

        # Adjust the input dimension to match the latent vector size
        self.fc = nn.Linear(z_dim, conv_dim * 8 * self.initial_image_width)

        # Define the convolutional layers
        self.conv1 = nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1)
        self.conv4 = nn.ConvTranspose2d(conv_dim, 1, 4, 2, 1)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 64 * 8, self.initial_image_width, 1)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = torch.tanh(self.conv4(out))  # Apply tanh activation for image generation
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, gfvs_dim=256, conv_dim=64):
        super(Discriminator, self).__init__()
        self.gfvs_dim = gfvs_dim

        # Calculate the final image size based on the GFVs shape
        self.final_image_size = int(np.sqrt(gfvs_dim/ conv_dim / 4))

        # Adjust the input dimension to match the GFVs size
        self.conv1 = nn.Conv2d(1, conv_dim, 4)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(conv_dim * 4, 1, 1)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = self.conv4(out)
        return out.squeeze()  # Return a 1D output


# #### Actor-Critic (DDPG)

# In[ ]:


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)#400
		self.l2 = nn.Linear(400, 400)
		self.l2_additional = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l2_additional(x))
		x = self.max_action * torch.tanh(self.l3(x))
		return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3_additional = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(x))
        # print("x size: ",x.size())
        # print("u size: "u.size())
        # u = u.view(u.size(0), -1)
        # x = x.squeeze(1)
        # u = u.squeeze(1)
        x = F.relu(self.l2(torch.cat([x, u], 2)))
        x = self.l3_additional(x)
        x = self.l3(x)
        return x


# In[ ]:


get_ipython().system('pip install faiss-cpu')


# In[ ]:


import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLL(nn.Module):
    def __init__(self):
        super(NLL,self).__init__()
    def forward(self,x):
        neglog = - F.log_softmax(x,dim=0)
        # greater the value greater the chance of being real
        probe = torch.mean(-F.log_softmax(x,dim=0))#F.softmax(x,dim=0)

      #  print(x.cpu().data.n/umpy())
       # print(-torch.log(x).cpu().data.numpy())
        return probe

class MSE(nn.Module):
    def __init__(self,reduction = 'elementwise_mean'):
        super(MSE,self).__init__()
        self.reduction = reduction
    def forward(self,x,y):
        mse = F.mse_loss(x,y,reduction =self.reduction)
        return mse


class Norm(nn.Module):
    def __init__(self,dims):
        super(Norm,self).__init__()
        self.dims =dims

    def forward(self,x):
        z2 = torch.norm(x,p=2)
        out = (z2-self.dims)
        out = out*out
        return out

import torch
import numpy as np
import faiss

def robust_norm(var):
    '''
    :param var: Variable of BxCxHxW
    :return: p-norm of BxCxW
    '''
    result = ((var**2).sum(dim=2) + 1e-8).sqrt()
    # result = (var ** 2).sum(dim=2)

    # try to make the points less dense, caused by the backward loss
    # result = result.clamp(min=7e-3, max=None)
    return result

class ChamferLoss(nn.Module):
    def __init__(self, opt):
        super(ChamferLoss, self).__init__()
        self.opt = opt
        self.dimension = 3
        self.k = 1

        # place holder
        self.forward_loss = torch.FloatTensor([0])
        self.backward_loss = torch.FloatTensor([0])

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in CPU
                 I: Variable of Nxk, type LongTensor, in CPU
        '''
        D, I = index.search(query, k)

        D_var = torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))

        return D_var, I_var

    def forward(self, predict_pc, gt_pc):
        '''
        :param predict_pc: Bx3xM Variable in CPU
        :param gt_pc: Bx3xN Variable in CPU
        :return:
        '''

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(torch.transpose(predict_pc.data.clone(), 1, 2).numpy())  # BxMx3
        gt_pc_np = np.ascontiguousarray(torch.transpose(gt_pc.data.clone(), 1, 2).numpy())  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2])
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2])

        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_var = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i,k,...] = gt_pc[i].index_select(1, I_var[:,k])

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_var = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i,k,...] = predict_pc[i].index_select(1, I_var[:,k])

        # compute loss ===================================================
        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        forward_loss_element = robust_norm(selected_gt_by_predict-predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict))
        self.forward_loss = forward_loss_element.mean()
        self.forward_loss_array = forward_loss_element.mean(dim=1).mean(dim=1)

        # selected_predict(Bxkx3xN) vs gt_pc(Bx3xN)
        backward_loss_element = robust_norm(selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt))  # BxkxN
        self.backward_loss = backward_loss_element.mean()
        self.backward_loss_array = backward_loss_element.mean(dim=1).mean(dim=1)

        # self.loss_array = self.forward_loss_array + self.backward_loss_array
        return self.forward_loss + self.backward_loss # + self.sparsity_loss

    def __call__(self, predict_pc, gt_pc):
        # start_time = time.time()
        loss = self.forward(predict_pc, gt_pc)
        # print(time.time()-start_time)
        return loss


# #### Testing the trained agent on test dataset

# In[ ]:


get_ipython().system('pip install plyfile')


# In[ ]:


import torch
import numpy as np
from torch.utils.data import Dataset
from plyfile import PlyData

def load_ply(path, with_faces=False, with_color=False):
    # print(path)
    ply_data = PlyData.read(path)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val

def write_ply(points, output_path, faces=None, colors=None):
    with open(output_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')

        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')

        if faces is not None:
            f.write('element face {}\n'.format(len(faces)))
            f.write('property list uchar int vertex_index\n')
            f.write('end_header\n')

            for point in points:
                f.write('{} {} {}\n'.format(point[0], point[1], point[2]))

            for face in faces:
                f.write('3 {} {} {}\n'.format(face[0], face[1], face[2]))
        else:
            f.write('end_header\n')

            for point in points:
                f.write('{} {} {}\n'.format(point[0], point[1], point[2]))


    # print("PLY file saved to:", output_path)



class PointCloudDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        file_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for dir in dirs:
                subdir_path = os.path.join(root, dir)
                # print(subdir_path)
                for file in os.listdir(subdir_path):
                    if file.endswith(".ply"):
                        # print(os.path.join(subdir_path, file))
                        file_paths.append(os.path.join(subdir_path, file))
        return file_paths
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        ply_data = load_ply(self.file_paths[idx])
        points = torch.tensor(ply_data, dtype=torch.float32)
        return points


# #### Environment class

# In[ ]:


class envs(nn.Module):
    def __init__(self,model_G,model_D,model_encoder,model_decoder,epoch_size):
        super(envs,self).__init__()

        self.nll = NLL()
        self.mse = MSE(reduction='elementwise_mean')
        z_dim = 1
        self.norm = Norm(dims=z_dim)
        self.chamfer = ChamferLoss({})
        self.epoch = 0
        self.epoch_size =epoch_size

        self.model_G = model_G
        self.model_D = model_D
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.j = 1
        self.figures = 3
        self.attempts = 5
        self.end = time.time()
        self.state_dim = 128
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
    def reset(self,epoch_size, figures =3):
        self.j = 1;
        self.i = 0;
        self.figures = figures;
        self.epoch_size= epoch_size
    def agent_input(self,input):
        with torch.no_grad():
          #Commented part required only for VAE
            input = input.transpose(1,2)#.float()
            # input = F.interpolate(input.float(), scale_factor=2, mode='nearest-exact')
            # encoder_out_mean, encoder_out_logvar = self.model_encoder(input)
            # epsilon = torch.randn_like(encoder_out_logvar)
            # encoder_out = encoder_out_mean + torch.exp(0.5 * encoder_out_logvar) * epsilon
            encoder_out = self.model_encoder(input)
            out = encoder_out.detach()
            # out = out.view(-1, 256)
        return out
    def forward(self,input,action,render=False, disp=False):
        state_dim = 256
        with torch.no_grad():
#Commented code only required for VAE
            # Encoder  output
            input = input.transpose(1,2)#.float()
            # input = F.interpolate(input.float(), scale_factor=2, mode='nearest-exact')
            # encoder_out_mean, encoder_out_logvar = self.model_encoder(input)
            # epsilon = torch.randn_like(encoder_out_logvar)
            # encoder_out = encoder_out_mean + torch.exp(0.5 * encoder_out_logvar) * epsilon
            encoder_out = self.model_encoder(input)
            encoder_out = encoder_out.view(-1, state_dim)

            # D Decoder Output
            pc_1 = self.model_decoder(encoder_out)
            # Generator Input
            z = action
            z = z.view(-1,1)

            # Generator Output
            out_GD = self.model_G(z)
            out_G = torch.squeeze(out_GD, dim=1)
            num_samples = encoder_out.shape[0]
            out_G = out_G[:num_samples]
            out_G = out_G.contiguous().view(-1, state_dim)

            # Discriminator Output
            out_D = self.model_D(out_GD) # TODO Alert major mistake

            # H Decoder Output
            pc_1_G = self.model_decoder(out_G)


            # Preprocesing of Input PC and Predicted PC for Visdom
            trans_input = torch.squeeze(input, dim=1)
            trans_input = torch.transpose(trans_input, 1, 2)
            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :] # D Decoder PC
            pc_1_G_temp = pc_1_G[0, :, :] # H Decoder PC


        # Discriminator Loss
        loss_D = self.nll(out_D)

        # Loss Between Noisy GFV and Clean GFV
        loss_GFV = self.mse(out_G, encoder_out)

        # Norm Loss
        loss_norm = self.norm(z)

        # Chamfer loss
        loss_chamfer = self.chamfer(pc_1_G, pc_1)  # #self.chamfer(pc_1_G, trans_input) instantaneous loss of batch items

        # States Formulation
        state_curr = np.array([loss_D.data.numpy(), loss_GFV.data.numpy()
                                  , loss_chamfer.data.numpy(), loss_norm.data.numpy()])
      #  state_prev = self.state_prev

        reward_D = state_curr[0]#state_curr[0] - self.state_prev[0]
        reward_GFV =-state_curr[1]# -state_curr[1] + self.state_prev[1]
        reward_chamfer = -state_curr[2]#-state_curr[2] + self.state_prev[2]
        reward_norm =-state_curr[3] # - state_curr[3] + self.state_prev[3]
        # Reward Formulation
        reward = ( reward_D *0.01 + reward_GFV * 10.0 + reward_chamfer *100.0 + reward_norm*1/10)

        # measured elapsed time
        # print("elapsed time: ",time.time() - self.end)
        self.end = time.time()

        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1

        done = True
        state = out_G.detach().data.numpy().squeeze()
        return state, _, reward, done


# ####Initialising saved models

# In[ ]:


# model_encoder = Encoder(num_points=2048, latent_dim=256)
model_encoder = Encoder(1024)
model_encoder.load_state_dict(encoder_state_dict)
# out_enc = encoder(train_data[0].transpose(0,1))

model_G = Generator()
model_G.load_state_dict(gen_state_dict)
# out_gen = generator(fixed_z)

model_D = Discriminator()
model_D.load_state_dict(disc_state_dict)
# out_disc = discriminator(out_gen)

model_decoder = Decoder(1024)
model_decoder.load_state_dict(decoder_state_dict)
# out_gen = out_gen.reshape(batch_size, 256)
# out_dec = decoder(out_enc)

model_actor = Actor(256, 1, 10)
model_actor.load_state_dict(actor_state_dict)

model_critic = Critic(256, 1)
model_critic.load_state_dict(critic_state_dict)


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# In[ ]:


import os
def testRL(model_encoder,model_decoder, model_G,model_D,model_actor,model_critic):

    model_encoder.eval()
    model_decoder.eval()
    model_G.eval()
    model_D.eval()
    model_actor.eval()
    model_critic.eval()

    drive_root = '/content/drive/MyDrive/RLFinalProjectFiles'
    org_root = os.path.join(drive_root, 'shape_net_core_uniform_samples_2048')

    del_ratio = 50

    test_data = PointCloudDataset(os.path.join(org_root + '_pointsremoved', 'test', str(del_ratio)))

    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=1,
                                               shuffle=False,
                                               pin_memory=True)

    epoch_size = len(test_loader)




    env = envs(model_G, model_D, model_encoder, model_decoder, epoch_size)

    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize variables
    reward_list = []
    sample_count = 0
    average_rewards = []

    # Loop over test data
    for i, input in enumerate(test_loader):
        obs = env.agent_input(input)
        action = model_actor(obs)
        action = torch.tensor(action).unsqueeze(dim=0)
        out_GD = model_G(action)
        out_G = torch.squeeze(out_GD, dim=1)
        out_G = out_G.contiguous().view(-1, 256)
        out_dec = model_decoder(out_G)

        new_state, _, reward, done = env(input, action)
        # print("Action:", action)
        # print("Reward:", reward)
        print("Resuts for RL Model 1")
        visualize_point_cloud(input.transpose(1,2).detach().numpy())
        print("Reward:", reward)
        visualize_point_cloud(out_dec.detach().numpy())

        # Accumulate rewards
        reward_list.append(reward)
        sample_count += 1

        if sample_count == 5:
          break

        # Calculate average reward every 20 samples
        if sample_count % 20 == 0:
            average_reward = np.mean(reward_list)
            average_rewards.append(average_reward)
            reward_list = []  # Reset reward list

    # Plot average rewards
    plt.plot(np.arange(20, len(average_rewards) * 20 + 20, 20), average_rewards)
    plt.xlabel('Sample Index')
    plt.ylabel('Average Reward')
    plt.title('Average Reward for Every 20 Samples')
    plt.grid(True)
    plt.show()


# In[ ]:


testRL(model_encoder,model_decoder, model_G,model_D,model_actor,model_critic)


# ### Visualise GroundTruth images

# In[ ]:


list_file_paths = ['/content/drive/MyDrive/RLFinalProjectFiles/shape_net_core_uniform_samples_2048/04090263/98f15a80dc5ea719d0a6af9bfb470a20.ply']

data = load_ply(list_file_paths[0])
# data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, pin_memory=True)

# Print dataset length to verify if data is loaded
print(f"Dataset length: {len(data)}")

visualize_point_cloud(data)  # Visualize point cloud


# In[ ]:




