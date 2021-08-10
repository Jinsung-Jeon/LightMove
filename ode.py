import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torchdiffeq import odeint, odeint_adjoint
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

class Attension(nn.Module):
    def __init__(self, data_dim, activation):
        super(Attension, self).__init__()
        self.data_dim = data_dim
        #self.output_dim = output_dim
        self.activation =activation
        '''
        self.attn = nn.Sequential(
            nn.Conv1d(self.data_dim, self.data_dim, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv1d(self.data_dim, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        '''
        self.attn = nn.Sequential(
            nn.Linear(self.data_dim, (self.data_dim//2)),
            nn.Tanh(),
            nn.Linear((self.data_dim//2), 1),
            nn.Tanh(),
        )
        if self.activation == 'softmax':
            self.act = nn.Softmax(dim=1)
        else:
            self.act = nn.Sigmoid()
        #self.max_pool = nn.MaxPool1d(1)
    def forward(self, x):
        # input  -> [256, 5, 1024]
        # self.attn(x) -> [256, 1, 1024]
        # repeat -> [256, 5, 1024]
        out_attn = self.attn(x).repeat(1,510)
        
        # einsum(dotproduct) -> [256, 5, 1]
        out = torch.einsum('ij,ij->ij', x, out_attn)
        if self.activation == 'softmax':
            out = self.act(out)
            # [256, 5, 1024]
            out = x * out
            # [256, 32*32]
            out = out.sum(dim=1)
            out = out.unsqueeze(1)
        else:
            out = self.act(out)
            out = x * out

        return out

class FullGRUODECell_Autonomous(torch.nn.Module):
    
    def __init__(self, hidden_size, model_method, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.model_method = model_method

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """

        '''
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        '''
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))
        dh = (1 - z) * (u - h)

        return dh

class FullGRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        """
        Executes one step with GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh
# orignal
class ODEBlock_3(nn.Module):

    def __init__(self, odefunc, hidden):
        super(ODEBlock_3, self).__init__()
        #self.integration_time = torch.tensor([0, 1]).float()
        self.all_layers = nn.ModuleList()
        self.odefunc = odefunc
        self.hidden_size = hidden
        
        #self.p_model = torch.nn.Sequential(
        #    torch.nn.Linear(self.hidden_size, int(self.hidden_size/2), bias=True),
        #    torch.nn.ReLU(),
        #    torch.nn.Dropout(p=0.2),
        #    torch.nn.Linear(int(self.hidden_size/2), self.hidden_size, bias=True),
        #    )
        
        self.p_model = nn.GRU(self.hidden_size, self.hidden_size, 1, dropout=0.3)        
        

    def ode_step(self, h, delta_t, current_time):
        """Executes a single ODE step."""
        #eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        #eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
        solution = odeint(self.odefunc,h,torch.tensor([current_time,current_time+delta_t]),method='euler' )

        h0 = torch.zeros(1, 1, self.hidden_size).cuda()
        h = solution[1].unsqueeze(1)
        p, _ = self.p_model(h, h0)
        p = p.squeeze(1)
        current_time += delta_t
        
        return p, current_time

    def forward(self, h):
        current_time = 0
        obs_time = 1
        #step = 1/(self.jump_time)
        while current_time < 1:
            h, current_time = self.ode_step(h, 0.2, current_time)
        return h

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
# 2ti jump

class ODEBlock_2(nn.Module):

    def __init__(self, odefunc, hidden):
        super(ODEBlock_2, self).__init__()
        #self.integration_time = torch.tensor([0, 1]).float()
        self.all_layers = nn.ModuleList()
        self.odefunc = odefunc
        self.hidden_size = hidden
        
        #self.p_model = torch.nn.Sequential(
        #    torch.nn.Linear(self.hidden_size, int(self.hidden_size/2), bias=True),
        #    torch.nn.ReLU(),
        #    torch.nn.Dropout(p=0.2),
        #    torch.nn.Linear(int(self.hidden_size/2), self.hidden_size, bias=True),
        #    )
        
        self.p_model = nn.GRU(self.hidden_size, self.hidden_size, 1, dropout=0.3)        
        

    def ode_step(self, h, delta_t, current_time):
        """Executes a single ODE step."""
        #eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        #eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)

        solution = odeint(self.odefunc,h,torch.tensor([current_time,current_time+delta_t]),method='rk4' )

        h0 = torch.zeros(1, 1, self.hidden_size).cuda()
        h = solution[1].unsqueeze(1)
        p, _ = self.p_model(h, h0)
        p = p.squeeze(1)
        current_time += delta_t
        
        return p, current_time

    def forward(self, h):
        current_time = 0
        obs_time = 1
        #step = 1/(self.jump_time)
        while current_time < 1:
            h, current_time = self.ode_step(h, 0.5, current_time)
        return h

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
'''
# 2times linear jump
'''
class ODEBlock_1(nn.Module):

    def __init__(self, odefunc, hidden):
        super(ODEBlock_1, self).__init__()
        #self.integration_time = torch.tensor([0, 1]).float()
        self.all_layers = nn.ModuleList()
        self.odefunc = odefunc
        self.hidden_size = hidden
        
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, int(self.hidden_size/2), bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(int(self.hidden_size/2), self.hidden_size, bias=True),
            )
        
        #self.p_model = nn.GRU(self.hidden_size, self.hidden_size, 1, dropout=0.3)        
        

    def ode_step(self, h, delta_t, current_time):
        """Executes a single ODE step."""
        #eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        #eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
        solution = odeint(self.odefunc,h,torch.tensor([current_time,current_time+delta_t]),method='euler' )
        p = self.p_model(solution[1].unsqueeze(1))
        #h0 = torch.zeros(1, 1, self.hidden_size).cuda()
        #h = solution[1].unsqueeze(1)
        #p, _ = self.p_model(h, h0)
        p = p.squeeze(1)
        current_time += delta_t
        
        return p, current_time

    def forward(self, h):
        current_time = 0
        obs_time = 1
        #step = 1/(self.jump_time)
        while current_time < 1:
            h, current_time = self.ode_step(h, 0.5, current_time)
        return h

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
'''
'''
# no jump
class ODEBlock_0(nn.Module):

    def __init__(self, odefunc, hidden):
        super(ODEBlock_0, self).__init__()
        #self.integration_time = torch.tensor([0, 1]).float()
        self.all_layers = nn.ModuleList()
        self.odefunc = odefunc
        self.hidden_size = hidden
        
        #self.p_model = torch.nn.Sequential(
        #    torch.nn.Linear(self.hidden_size, int(self.hidden_size/2), bias=True),
        #    torch.nn.ReLU(),
        #    torch.nn.Dropout(p=0.2),
        #    torch.nn.Linear(int(self.hidden_size/2), self.hidden_size, bias=True),
        #    )
        
        #self.p_model = nn.GRU(self.hidden_size, self.hidden_size, 1, dropout=0.3)        
        

    #def ode_step(self, h, delta_t, current_time):
    #    """Executes a single ODE step."""
    #    #eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
    #    #eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
    #    solution = odeint(self.odefunc,h,torch.tensor([current_time,current_time#+delta_t]),method='euler' )
    #    p = self.p_model(solution[1].unsqueeze(1))
    #    #h0 = torch.zeros(1, 1, self.hidden_size).cuda()
    #    #h = solution[1].unsqueeze(1)
    #    #p, _ = self.p_model(h, h0)
    #    p = p.squeeze(1)
    #    current_time += delta_t
    #    
    #    return p, current_time

    def forward(self, h):
        current_time = torch.tensor(0,device = h.device, dtype = torch.float64)
        obs_time = torch.tensor(1,device = h.device, dtype = torch.float32)
        #step = 1/(self.jump_time)
        solution = odeint(self.odefunc,h,torch.tensor([current_time,obs_time]),method='euler' )
        h = solution[1].unsqueeze(1)
        return h

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

'''
class FullGRUODECell_Autonomous_fine(torch.nn.Module):
    
    def __init__(self, hidden_size, model_method, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.model_method = model_method
        self.hidden_size = hidden_size
        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hh_fine = torch.nn.Linear(hidden_size, hidden_size*hidden_size, bias=False)
        self.lin_hz_fine = torch.nn.Linear(hidden_size, hidden_size*hidden_size, bias=False)
        self.lin_hr_fine = torch.nn.Linear(hidden_size, hidden_size*hidden_size, bias=False)
        

    def forward(self, t, h):

        x = torch.zeros_like(h)



        #short_term = h[-10:] 
        #long_term = h[:-10]
        #x = torch.zeros_like(short_term)

        #import pdb
        #pdb.set_trace()
        # self.lin_hr_fine(short_term) => 510 * 510
        
        
        hr_weight = torch.reshape(self.lin_hr_fine(h), (h.size(0),self.hidden_size, self.hidden_size))
        hz_weight = torch.reshape(self.lin_hz_fine(h), (h.size(0),self.hidden_size, self.hidden_size))
        hh_weight = torch.reshape(self.lin_hh_fine(h), (h.size(0),self.hidden_size, self.hidden_size))
        
        hr_weight_small = F.linear(h[0], hr_weight[0]).unsqueeze(0)
        hz_weight_small = F.linear(h[0], hz_weight[0]).unsqueeze(0)

        for i, hr, hz  in zip(h[1:], hr_weight[1:], hz_weight[1:]):
            hr_weight_small = torch.cat((hr_weight_small, F.linear(i, hr).unsqueeze(0)))
            hz_weight_small = torch.cat((hz_weight_small, F.linear(i, hz).unsqueeze(0)))

        r = torch.sigmoid(x + (self.lin_hr(h) + hr_weight_small))
        z = torch.sigmoid(x + (self.lin_hz(h) + hz_weight_small))
        
        hh_weight_small = F.linear(r[0]*h[0], hh_weight[0]).unsqueeze(0)
        for i, hh,rr  in zip(h[1:], hh_weight[1:], r[1:]):
            hh_weight_small = torch.cat((hh_weight_small, F.linear(rr*i, hh).unsqueeze(0)))

        u = torch.tanh(x + (self.lin_hh(r * h) + hh_weight_small))
        dh_new = (1 - z) * (u - h)
        #dh = torch.cat((dh_new, long_term), dim=0)

        return dh_new
'''
class FullGRUODECell_Autonomous_fine(torch.nn.Module):
    
    def __init__(self, hidden_size, model_method, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.model_method = model_method
        self.hidden_size = hidden_size
        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        #self.lin_hh_fine = torch.nn.Linear(hidden_size, hidden_size*hidden_size, bias=False)
        self.lin_hz_fine = torch.nn.Linear(hidden_size, hidden_size*hidden_size, bias=False)
        #self.lin_hr_fine = torch.nn.Linear(hidden_size, hidden_size*hidden_size, bias=False)

    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """

        '''
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        '''
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        
        hz_weight = torch.reshape(self.lin_hz_fine(h), (h.size(0),self.hidden_size, self.hidden_size))
        hz_weight_small = F.linear(h[0], hz_weight[0]).unsqueeze(0)
        for i, hz  in zip(h[1:], hz_weight[1:]):
            hz_weight_small = torch.cat((hz_weight_small, F.linear(i, hz).unsqueeze(0)))
        z = torch.sigmoid(x + (self.lin_hz(h) + hz_weight_small))

        
        u = torch.tanh(x + self.lin_hh(r * h))
        dh = (1 - z) * (u - h)

        return dh

