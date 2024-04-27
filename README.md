# rl-gan-net-reimplementation
RL-GAN-Net-Reimplementation

# References
### The projct and re-implementation references are the following research paper and code repository
https://arxiv.org/pdf/1904.12304 <br />
https://github.com/iSarmad/RL-GAN-Net <br />
https://proceedings.mlr.press/v32/silver14.pdf

### Instructions for executing the pipline
1. Get input data from the official ShapeNet repository for point cloud data or from here https://github.com/optas/latent_3d_points <br />
2. Run TrainingAE.py script, to train Autoencoder on the complete point cloud data and tetsing on partial point cloud data, and then saving GFVs generated to be used for training GANs.<br />
3. Run TrainingGAN.py scrpt to train GAN on GFVs.
4. Run TrainingRLAgent.py to create environment for the RL agent using AE and GAN outputs and then training the agent using DDPG algorithm.
5. After running the pipeline, you can use VisualizingRLAgentREsults.py script to visualise results using the saved models.
