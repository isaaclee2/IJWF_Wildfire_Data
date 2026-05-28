import torch, torch.nn as nn, numpy as np
from tqdm import tqdm,trange
from scipy import integrate
import os
from ema_pytorch import EMA
from optimal_transport import OTPlanSampler

def get_activation(activation):
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'SiLU':
        return nn.SiLU()
    elif activation == 'Tanh':
        return nn.Tanh()
    elif activation == 'Sigmoid':
        return nn.Sigmoid()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

class modelNN(nn.Module):
    def __init__(self, n_dim_x=1, n_dim_y=1, width=256, depth=4, activation='ReLU'):
        super(modelNN, self).__init__()
        self.activation = nn.ReLU() if activation == 'ReLU' else nn.SiLU()
        self.n_dim_x = n_dim_x
        self.n_dim_y = n_dim_y
        self.n_dim = n_dim_x + n_dim_y
        self.width = width
        layers = [nn.Linear(self.n_dim + 1, width), self.activation]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        layers.append(nn.Linear(width, n_dim_x))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x, t):
        x_inp = torch.cat([x, t], dim=-1)
        return self.fc(x_inp) 

class modelNN2(nn.Module):
    def __init__(self, n_dim_x=1, n_dim_y=1, width=256, depth=4, activation='ReLU'):
        super(modelNN2, self).__init__()
        self.activation = nn.ReLU() if activation == 'ReLU' else nn.SiLU()
        self.n_dim_x = n_dim_x
        self.n_dim_y = n_dim_y
        self.n_dim = n_dim_x + n_dim_y
        self.width = width
        layers = [nn.Linear(self.n_dim + 4, width), self.activation]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        layers.append(nn.Linear(width, n_dim_x))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x, t):
        t = t.squeeze()
        embed = [t - 0.5, torch.cos(2*np.pi*t), torch.sin(2*np.pi*t), -torch.cos(4*np.pi*t)]
        embed = torch.stack(embed, dim=-1)
        x_inp = torch.cat([x, embed], dim=-1)
        return self.fc(x_inp)        

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.lin1 = nn.Linear(time_dim, time_dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(time_dim, time_dim)

        freqs = torch.exp(
            -torch.arange(0, time_dim, 2) * (np.log(10000) / time_dim)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        # t: (B,)
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = self.lin2(self.act(self.lin1(emb)))
        return emb

class modelNN3(nn.Module):
    def __init__(
        self,
        n_dim_x=256,
        n_dim_y=1152,
        width=512,
        depth=4,
        time_dim=128,
        activation="SiLU",
    ):
        super().__init__()

        self.time_embed = TimeEmbedding(time_dim)
        self.act = nn.SiLU() if activation == "SiLU" else nn.ReLU()

        input_dim = n_dim_x + n_dim_y + time_dim

        layers = [nn.Linear(input_dim, width), self.act]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), self.act]

        layers.append(nn.Linear(width, n_dim_x))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t = t.squeeze()
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)
        

def return_model(model_type):
    if model_type == 'modelNN':
        return modelNN
    elif model_type == 'modelNN2':
        return modelNN2
    elif model_type == 'modelNN3':
        return modelNN3
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class Flow_bridge(nn.Module):
    def __init__(self, n_dim_x=1, n_dim_y=1, width=256, depth=4, activation='ReLU', model_type='modelNN', device='cpu', prior_to_posterior=False, eta=0, OT=None, ot_method="exact"):
        super().__init__()
        self.net = return_model(model_type)(n_dim_x=n_dim_x, n_dim_y=n_dim_y, width=width, depth=depth, activation=activation).to(device)
        self.loss_func = nn.MSELoss(reduction='mean')
        self.device = device
        self.prior_to_posterior = prior_to_posterior
        #self.antithetic = True if prior_to_posterior else False
        self.eta = eta # 0: eta0, 1: eta1 for loss
               
    def forward(self, x, y, t):
        inp = torch.cat([x, y], dim=-1)
        return self.net(inp, t)
    
    def x_t(self, x_0, x_1, z, t):
        if self.prior_to_posterior:
            return t* (x_1 - x_0) + x_0 + z*self.gamma(t)
        else:
            return t* (x_1 - x_0) + x_0
        
    def x_t_m(self, x_0, x_1, z, t):
        if self.prior_to_posterior:
            return t* (x_1 - x_0) + x_0 - z*self.gamma(t)
        else:
            return t* (x_1 - x_0) + x_0
    
    
    def dot_x_t(self, x_0, x_1, z, t):
        if self.prior_to_posterior:
            #return x_1 - x_0 - z*torch.sqrt(2*t)/torch.sqrt(1-t) # u(t,x) equation 3.8 in stochastich interpolant paper
            return x_1 - x_0 + z*self.gamma_dot(t)
        else:
            return x_1 - x_0
        
    def dot_x_t_m(self, x_0, x_1, z, t):
        if self.prior_to_posterior:
            #return x_1 - x_0 - z*torch.sqrt(2*t)/torch.sqrt(1-t) # u(t,x) equation 3.8 in stochastich interpolant paper
            return x_1 - x_0 - z*self.gamma_dot(t)
        else:
            return x_1 - x_0
    
    def step(self, x_t, y, t_start, t_end):
        t_diff = t_end - t_start
        t_mid = t_start + t_diff / 2
        x_mid = x_t + self(x_t, y, t_start) * t_diff / 2
        return x_t + (t_end - t_start) * self(x_mid, y, t_mid)
    
    def step_sde(self, x_t, y, t_start, t_end):
        """
        One Euler–Maruyama step for SDE:
        dX = u(t, X, y) dt + sqrt(2) dW
        """
        t_diff = t_end - t_start
        drift = self(x_t, y, t_start)  # u(t, X, y)

        # Brownian noise: Normal(0, sqrt(dt))
        noise = torch.randn_like(x_t) * torch.sqrt(t_diff)

        return x_t + drift * t_diff + (2.0**0.5) * noise
    
    def loss_eta0(self, x_0, x_1, y, z, t):
        x_t = self.x_t(x_0, x_1, z, t)
        dot_x_t = x_0 # self.dot_x_t(x_0, x_1, z, t)
        out = self(x_t, y, t)
        #custom loss: (out^2 - 2*out*dot_x_t), averaged over batch
        loss_val = (torch.sum(out * out, dim=1) - 2 * torch.sum(out * dot_x_t, dim=1)).mean()#.abs()
        return loss_val #self.loss_func(out, dot_x_t) # 
    
    def gamma(self, t):
        return torch.sqrt(2*t*(1-t))
    
    def gamma_dot(self, t):
        return (1-2*t)/torch.sqrt(2*t*(1-t))
    
    def loss_eta1(self, x_0, x_1, y, z, t):
        x_t = self.x_t(x_0, x_1, z, t)
        dot_x_t = x_1 # self.dot_x_t(x_0, x_1, z, t)
        out = self(x_t, y, t)
        #custom loss: (out^2 - 2*out*dot_x_t), averaged over batch
        loss_val = (torch.sum(out * out, dim=1) - 2 * torch.sum(out * dot_x_t, dim=1)).mean()#.abs()
        return loss_val #self.loss_func(out, dot_x_t) # 
    

    def loss_antithetic(self, x_0, x_1, y, z, t):
        x_t = self.x_t(x_0, x_1, z, t)
        x_t_m = self.x_t_m(x_0, x_1, z, t)
        dot_x_t = self.dot_x_t(x_0, x_1, z, t)
        dot_x_t_m = self.dot_x_t_m(x_0, x_1, z, t)
        out = self(x_t, y, t)
        out_m = self(x_t_m, y, t)
        #custom loss: (out^2 - 2*out*dot_x_t), averaged over batch
        #loss_val = (torch.sum(out * out, dim=1) - 2 * torch.sum(out * dot_x_t, dim=1)).mean()#.abs()
        #loss_val += (torch.sum(out_m * out_m, dim=1) - 2 * torch.sum(out_m * dot_x_t_m, dim=1)).mean()
        loss_val = self.loss_func(out, dot_x_t) + self.loss_func(out_m, dot_x_t_m)
        return 0.5*loss_val #self.loss_func(out, dot_x_t) #

    def train(self, X, Y, n_iters, batch_size, lr, save_path=None):
        lr  = float(lr)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        velocity_ema = EMA(self.net, beta=0.9999)
        loss_history = []
        save_freq = 5000 #100 if n_iters < 100 else 5000
        update_freq = 100 #if n_iters < 100 else 1000
        if not os.path.exists(save_path + f"/checkpoints{self.eta}"):
            os.makedirs(save_path + f"/checkpoints{self.eta}")

        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        if self.prior_to_posterior:
            assert batch_size <= int(0.5*X.shape[0]), "Batch size must be less than or equal to half the number of samples in X"
        
        self.net.train()
        #pbar = tqdm(range(n_iters), desc="Loss: ", ncols=100, colour='green')
        for i in range(n_iters):
            self.net.train(mode=True)
            # randomly sample indices for the batch
            # idx = torch.randint(0, X.shape[0], (2*batch_size,)) if self.prior_to_posterior else torch.randint(0, X.shape[0], (batch_size,))
            # idx_1 = idx[:batch_size] 
            idx_1 = torch.randint(0, X.shape[0], (batch_size,))
            
            x_1 = X[idx_1].to(self.device)
            y = Y[idx_1].to(self.device)
            
            if self.prior_to_posterior:
                # idx_0 = idx[batch_size:]
                pos_ = torch.randperm(idx_1.shape[0])
                idx_0 = idx_1[pos_]
                x_0 = X[idx_0].to(self.device)
            else:
                x_0 = torch.randn_like(x_1).to(self.device) 

            z = torch.randn_like(x_1).to(self.device) 

            eps = 1e-4  # small positive number
            t = (torch.rand(batch_size, 1) * (1 - eps) + eps).to(self.device) #samples in [0, 1) will shifts it to [eps, 1) to accout for the singularity at zero in velovity function in schrodinger bridge case for prior to posterior training 
            if self.eta==0:
                loss = self.loss_eta0(x_0, x_1, y, z, t)
            else:
                loss = self.loss_eta1(x_0, x_1, y, z, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            velocity_ema.update()
            
            if (i+1) % update_freq == 0:
                print(f"Iteration {i+1}/{n_iters}, Loss: {loss.item():.4f}")
            #     pbar.set_description(f"Loss: {loss.item():.4f}")
                
            loss_history.append(loss.item())
            
            if (i + 1) % save_freq == 0 and save_path is not None:
                save_chkpt = save_path + f"/checkpoints{self.eta}/model_{i+1}.pt"
                torch.save(self.net.state_dict(), save_chkpt)
                save_opt = save_path + f"/checkpoints{self.eta}/optimizer_{i+1}.pt"
                torch.save(optimizer.state_dict(), save_opt)
                save_ema = save_path + f"/checkpoints{self.eta}/ema_{i+1}.pt"
                torch.save(velocity_ema.state_dict(), save_ema)
                
        # delete everything occupying GPU memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        # del optimizer, x_0, x_1, y, t, loss, idx_0, idx_1
        return loss_history
    
    def sample(self, x_start, y_cond, n_steps, return_path=False):
        x_path = []
        x_start = x_start.to(self.device)
        y_cond = y_cond.view(1, -1).expand(x_start.shape[0], -1).to(self.device)
        t = torch.linspace(0, 1, n_steps)
        x_t = x_start
        for i in trange(1, t.shape[0]):
            t_prev = t[i-1].view(1, 1).expand(x_t.shape[0], 1).to(self.device)
            t_curr = t[i].view(1, 1).expand(x_t.shape[0], 1).to(self.device)
            if self.prior_to_posterior:
                x_t = self.step_sde(x_t, y_cond, t_prev, t_curr)
            else:
                x_t = self.step(x_t, y_cond, t_prev, t_curr)
            if return_path:
                x_path.append(x_t.detach().cpu())

        if return_path:
            return torch.stack(x_path, dim=0)  # (n_steps-1, batch, dim)
        else:
            return x_t.detach().cpu()

    
    def odeint_sampler(self, x_start, y_cond, n_steps, return_path=False):
        self.net.eval()
        x_start = x_start.to(self.device)
        if y_cond.dim() == 1:
            y_cond = y_cond.view(1, -1).expand(x_start.shape[0], -1).to(self.device)
        else:
            y_cond = y_cond.to(self.device)
        t_eval = torch.linspace(0, 1, n_steps)
        
        def vel_wrapper(sample, t):
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(x_start.shape)
            with torch.no_grad():    
                velocity = self(sample, y_cond, t)
            return velocity.detach().cpu()
        
        def ode_func(t, x):        
            batch_time = torch.ones(x_start.shape[0], 1) * t
            rhs = vel_wrapper(x, batch_time.to(self.device))
            return rhs.numpy().reshape((-1,)).astype(np.float64)
        
        err_tol = 1e-5
        t_eval = np.linspace(0.0, 1.0, n_steps)
        res = integrate.solve_ivp(ode_func, (0.0, 1.0), x_start.reshape(-1).cpu().numpy(), rtol=err_tol, atol=err_tol, method='RK45', dense_output=True, t_eval=t_eval)  
        
        lat_shape = [x_start.shape[0], x_start.shape[1], len(res.t)]
        res_loc = torch.tensor(res.y, device=self.device, dtype=torch.float32).reshape(lat_shape)
        
        final_samples = res_loc[:,:,-1].detach().cpu() #.numpy()
        if return_path:
            res_loc = res_loc.permute(2, 0, 1)
            return res_loc.detach().cpu() #.numpy()
        else:        
            return final_samples
    
    def odeint_sampler_eta(self, x_start, y_cond, n_steps, eta1_flow, return_path=False):
        self.net.eval()
        x_start = x_start.to(self.device)
        y_cond = y_cond.view(1, -1).expand(x_start.shape[0], -1).to(self.device)
        t_eval = torch.linspace(0, 1, n_steps)
        t_eval[0] = 0.0 + 1e-4 # to avoid singularity at t=0 in gamma(t) in schrodinger bridge case
        t_eval[-1] = 1.0 - 1e-4
        
        def vel_wrapper(sample, t):
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(x_start.shape)
            with torch.no_grad():
                eta0 = self(sample, y_cond, t)
                eta1 = eta1_flow(sample, y_cond, t)
                eta_z = (sample - (1-t)*eta0 - t*eta1)/self.gamma(t)   
                velocity = -eta0 + eta1 + self.gamma_dot(t)*eta_z
            return velocity.detach().cpu()
        
        def ode_func(t, x):        
            batch_time = torch.ones(x_start.shape[0], 1) * t
            rhs = vel_wrapper(x, batch_time.to(self.device))
            return rhs.numpy().reshape((-1,)).astype(np.float64)
        
        err_tol = 1e-5
        t_eval = t_eval.numpy()
        res = integrate.solve_ivp(ode_func, (0.0+1e-4, 1.0-1e-4), x_start.reshape(-1).cpu().numpy(), rtol=err_tol, atol=err_tol, method='RK45', dense_output=True, t_eval=t_eval)  
        lat_shape = [x_start.shape[0], x_start.shape[1], len(res.t)]
        res_loc = torch.tensor(res.y, device=self.device, dtype=torch.float32).reshape(lat_shape)
        
        final_samples = res_loc[:,:,-1].detach().cpu() #.numpy()
        if return_path:
            res_loc = res_loc.permute(2, 0, 1)
            return res_loc.detach().cpu() #.numpy()
        else:        
            return final_samples

class Flow(nn.Module):
    def __init__(self, n_dim_x=1, n_dim_y=1, width=256, depth=4, activation='ReLU', model_type='modelNN', device='cpu', prior_to_posterior=False, OT=False, ot_method="exact"):
        super().__init__()
        self.net = return_model(model_type)(n_dim_x=n_dim_x, n_dim_y=n_dim_y, width=width, depth=depth, activation=activation).to(device)
        self.loss_func = nn.MSELoss(reduction='mean')
        self.device = device
        self.prior_to_posterior = prior_to_posterior
        self.OT = OT
        self.ot_method = ot_method
        self.vel_ema = EMA(self.net, beta=0.99) # added from diffusion code

    def forward(self, x, y, t, ema=False):
        inp = torch.cat([x, y], dim=-1)
        return self.net(inp, t) if not ema else self.vel_ema.ema_model(inp, t)

    def x_t(self, x_0, x_1, z, t):
        return t* (x_1 - x_0) + x_0

    def dot_x_t(self, x_0, x_1, z, t):
        return x_1 - x_0
    
    def step(self, x_t, y, t_start, t_end):
        t_diff = t_end - t_start
        t_mid = t_start + t_diff / 2
        x_mid = x_t + self(x_t, y, t_start) * t_diff / 2
        return x_t + (t_end - t_start) * self(x_mid, y, t_mid)

    def loss(self, x_0, x_1, y, z, t, ema=False):
        x_t = self.x_t(x_0, x_1, z, t)
        dot_x_t = self.dot_x_t(x_0, x_1, z, t)
        out = self(x_t, y, t, ema=ema)
        return self.loss_func(out, dot_x_t)

    def train(self, X, Y, X_test, Y_test, n_iters, batch_size, lr, save_path=None, n_test_freq=200, save_freq=2000):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(lr), betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        loss_history = []
        test_loss_history = []

        test_loss_moving_avg_list = []
        window_size = 1000
        patience = [20, 30, 50, 70]
        min_delta = 0.1

        #save_freq = 2000 #100 if n_iters < 100 else 5000
        update_freq = 100 #if n_iters < 100 else 1000
        if not os.path.exists(save_path + "/checkpoints"):
            os.makedirs(save_path + "/checkpoints")

        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        assert X_test.shape[0] == Y_test.shape[0], "X_test and Y_test must have the same number of samples"

        if self.prior_to_posterior:
            assert batch_size <= int(0.5*X.shape[0]), "Batch size must be less than or equal to half the number of samples in X"

        if self.OT:
            ot_sampler = OTPlanSampler(method=self.ot_method, reg=0.1, normalize_cost=True)
            print("Using mini-batch OT Plan sampler for training.")
        
        self.net.train()
        #pbar = tqdm(range(n_iters), desc="Loss: ", ncols=100, colour='green')
        X_t = X_test.to(self.device)
        Y_t = Y_test.to(self.device)
        flag1 = [0]*len(patience)
        flag2 = [0]*len(patience)
        condition_triggered1 = [False]*len(patience)
        condition_triggered2 = [False]*len(patience)
        for i in range(n_iters):
            self.net.train(mode=True)
            # randomly sample indices for the batch
            # idx = torch.randint(0, X.shape[0], (2*batch_size,)) if self.prior_to_posterior else torch.randint(0, X.shape[0], (batch_size,))
            # idx_1 = idx[:batch_size] 
            idx_1 = torch.randint(0, X.shape[0], (batch_size,))
            
            x_1 = X[idx_1].to(self.device)
            y = Y[idx_1].to(self.device)
            
            if self.prior_to_posterior:
                # idx_0 = idx[batch_size:]
                pos_ = torch.randperm(idx_1.shape[0])
                idx_0 = idx_1[pos_]
                x_0 = X[idx_0].to(self.device)
                y_0 = Y[idx_0].to(self.device)
                if (i+1) % n_test_freq == 0:
                    #idx_t = torch.randint(0, X.shape[0], (X_t.shape[0],))
                    idx_t = torch.randperm(X_t.size(0), device=self.device)
                    x_0t = X_t[idx_t] #.to(self.device)
                    y_0t = Y_t[idx_t] #.to(self.device)
            else:
                x_0 = torch.randn_like(x_1).to(self.device) 
                if (i+1) % n_test_freq == 0:
                    x_0t = torch.randn_like(X_t).to(self.device)

            z = torch.randn_like(x_1).to(self.device) 

            if self.OT:
                with torch.no_grad():
                    x_0, x_1, _, y = ot_sampler.sample_plan_with_labels(x_0, x_1, y0=y_0 if self.prior_to_posterior else None, y1=y, replace=False)
                    if (i+1) % n_test_freq == 0:
                        x_0t, X_t, _, Y_t = ot_sampler.sample_plan_with_labels(x_0t, X_t, y0=y_0t if self.prior_to_posterior else None, y1=Y_t, replace=False)
            
            eps = 1e-8  # small positive number
            t = (torch.rand(batch_size, 1) * (1 - eps) + eps).to(self.device) #samples in [0, 1) will shifts it to [eps, 1) to accout for the singularity at zero in velovity function in schrodinger bridge case for prior to posterior training 
            loss = self.loss(x_0, x_1, y, z, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.vel_ema.update()
            if (i+1) % n_test_freq == 0:
                z_test = torch.randn_like(X_t).to(self.device)
                t_test = (torch.rand(X_t.shape[0], 1) * (1 - eps) + eps).to(self.device)
                test_loss = self.loss(x_0t, X_t, Y_t, z_test, t_test)
                test_loss_history.append(test_loss.item())
            
            if (i+1) % update_freq == 0:
                print(f"Iteration {i+1}/{n_iters}, Loss: {loss.item():.4f}")
            #     pbar.set_description(f"Loss: {loss.item():.4f}")
                
            loss_history.append(loss.item())
            
            with torch.no_grad():
                self.net.eval()

                
                # compute moving average of test loss by averaging the test losses over the last 'window_size' iterations
                if len(test_loss_history) >= window_size:
                    moving_avg_test_loss = np.mean(test_loss_history[-window_size:])
                    test_loss_moving_avg_list.append(moving_avg_test_loss)
                    
                    # stop training if the moving average of test loss does not improve for 'patience' number of times
                    for k in range(len(patience)):    
                        recent_moving_avgs = test_loss_moving_avg_list[-patience[k]:]
                        if not condition_triggered1[k] and all(moving_avg_test_loss > 1.001*prev_avg for prev_avg in recent_moving_avgs[:-1]):    
                            print(f"Early stopping (patience={patience[k]} and 0.1% dif) at iteration {i+1} with moving average test loss {moving_avg_test_loss:.4f}")
                            condition_triggered1[k] = True
                            flag1[k] = i+1
                        if not condition_triggered2[k] and all(moving_avg_test_loss > prev_avg for prev_avg in recent_moving_avgs[:-1]):    
                            print(f"Early stopping (patience={patience[k]} and 0% dif) at iteration {i+1} with moving average test loss {moving_avg_test_loss:.4f}")
                            condition_triggered2[k] = True
                            flag2[k] = i+1
                    #     save_chkpt = save_path + f"/checkpoints/model_{i+1}.pt"
                    #     torch.save(self.net.state_dict(), save_chkpt)
                    #     save_opt = save_path + f"/checkpoints/optimizer_{i+1}.pt"
                    #     torch.save(optimizer.state_dict(), save_opt)
                    #     save_ema = save_path + f"/checkpoints/ema_{i+1}.pt" ## added for diffusion consistancy
                    #     torch.save(self.vel_ema.ema_model.state_dict(), save_ema)
                    #     break   
                else:
                    moving_avg_test_loss = np.mean(test_loss_history)
                    test_loss_moving_avg_list.append(moving_avg_test_loss)
                
            
            if (i + 1) % save_freq == 0 and save_path is not None:
                save_chkpt = save_path + f"/checkpoints/model_{i+1}.pt"
                torch.save(self.net.state_dict(), save_chkpt)
                save_opt = save_path + f"/checkpoints/optimizer_{i+1}.pt"
                torch.save(optimizer.state_dict(), save_opt)
                save_ema = save_path + f"/checkpoints/ema_{i+1}.pt"
                torch.save(self.vel_ema.ema_model.state_dict(), save_ema)
        # save final model
        save_chkpt = save_path + f"/checkpoints/model_0.pt"
        torch.save(self.net.state_dict(), save_chkpt)
        save_opt = save_path + f"/checkpoints/optimizer_0.pt"
        torch.save(optimizer.state_dict(), save_opt)
        save_ema = save_path + f"/checkpoints/ema_0.pt"
        torch.save(self.vel_ema.ema_model.state_dict(), save_ema)    

        # delete everything occupying GPU memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        # del optimizer, x_0, x_1, y, t, loss, idx_0, idx_1
        return loss_history, test_loss_history, test_loss_moving_avg_list, flag1, flag2, patience
    
    def sample(self, x_start, y_cond, n_steps):
        x_path = []
        x_start = x_start.to(self.device)
        y_cond = y_cond.view(1, -1).expand(x_start.shape[0], -1).to(self.device)
        t = torch.linspace(0, 1, n_steps)
        x_t = x_start
        for i in trange(1, t.shape[0]):
            t_prev = t[i-1].view(1, 1).expand(x_t.shape[0], 1).to(self.device)
            t_curr = t[i].view(1, 1).expand(x_t.shape[0], 1).to(self.device)
            x_t = self.step(x_t, y_cond, t_prev, t_curr)
            x_path.append(x_t)
            
            
        # reshape the output to have shape (n_steps, n_samples, n_dim_x)
        x_path = torch.stack(x_path, dim=0).detach().cpu()
        return x_path

    def odeint_sampler(self, x_start, y_cond, n_steps, return_path=False):
        self.net.eval()
        self.vel_ema.ema_model.eval()

        x_start = x_start.to(self.device)
        if y_cond.dim() == 1:
            y_cond = y_cond.view(1, -1).expand(x_start.shape[0], -1).to(self.device)
        else:
            y_cond = y_cond.to(self.device)
        
        t_eval = torch.linspace(0, 1, n_steps)
        
        def vel_wrapper(sample, t):
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(x_start.shape)
            with torch.no_grad():    
                velocity = self(sample, y_cond, t, ema=True)
            return velocity.detach().cpu()
        
        def ode_func(t, x):        
            batch_time = torch.ones(x_start.shape[0], 1) * t
            rhs = vel_wrapper(x, batch_time.to(self.device))
            return rhs.numpy().reshape((-1,)).astype(np.float64)
        
        err_tol = 1e-5
        t_eval = np.linspace(0.0, 1.0, n_steps)
        res = integrate.solve_ivp(ode_func, (0.0, 1.0), x_start.reshape(-1).cpu().numpy(), rtol=err_tol, atol=err_tol, method='RK45', dense_output=True, t_eval=t_eval)  
        
        lat_shape = [x_start.shape[0], x_start.shape[1], len(res.t)]
        res_loc = torch.tensor(res.y, device=self.device, dtype=torch.float32).reshape(lat_shape)
        
        final_samples = res_loc[:,:,-1].detach().cpu() #.numpy()
        if return_path:
            res_loc = res_loc.permute(2, 0, 1)
            return res_loc.detach().cpu() #.numpy()
        else:        
            return final_samples

if __name__ == "__main__":
    print("Flow model codes.")
           
            
            
            
            
            
            
            
            
            
        
        