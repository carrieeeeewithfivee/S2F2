#_backwardsmallpastmaskfutureflowloss_noid_featuretahn
#final
import torch
import torch.nn as nn

class StateFlowEncoder(nn.Module):
    def __init__(self, statedim=128, outdim=62):
        super(StateFlowEncoder, self).__init__()
        self.convc1 = nn.Conv2d(statedim, 64, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(96, outdim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, state, flow):
        state = self.relu(self.convc1(state))
        flo = self.relu(self.convf1(flow))
        flo = self.relu(self.convf2(flo))
        state_flo = torch.cat([state, flo], dim=1)
        out = self.relu(self.conv(state_flo))
        return torch.cat([out, flow], dim=1)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=512):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
        
def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class Decoder(nn.Module):
    def __init__(self, future_len):
        super(Decoder, self).__init__()
        self.fut = future_len
        #encoder process
        self.encode = StateFlowEncoder(statedim=64, outdim=62)

        ##decoder
        setattr(self, 'D_gru', ConvGRU(hidden_dim=64, input_dim=64))
        #self.convD = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        setattr(self, 'D_flow', FlowHead(input_dim=64, hidden_dim=128))

    #def forward(self, state, encoded, p_delta_flow, index): #index of t-1 centers
    def forward(self, feature): #index of t-1 centers
        #def forward(self, state, encoded): #index of t-1 centers
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        b,c,h,w = feature.shape
        _coords1 = coords_grid(b, h, w).to('cuda').detach()
        coords0 = coords_grid(b, h, w).to('cuda').detach()
        f_flows = []       

        #split the feature into input and state
        state, inp = torch.split(feature, [64, 64], dim=1)
        state = torch.tanh(state)
        inp = torch.relu(inp)

        #decode, future
        for i in range(0, self.fut):
            _coords1 = _coords1.detach()
            flow = _coords1-coords0
            #encode process
            cat_encoded = self.encode(inp,flow) #raft has a small encoder here (corr, flow, inp)
            h = getattr(self, 'D_gru')(state, cat_encoded)
            state = h

            #f_delta_flow = getattr(self, 'D_flow')(self.convD(h))
            f_delta_flow = getattr(self, 'D_flow')(h) #larger flow
            f_flow = _coords1 + f_delta_flow
            _coords1 = f_flow #0~1 0~2 0~3
            f_flows.append(f_flow)

        #prediction_dict = {'p_flows': p_flow, 'f_flows':f_flows} #coords instead of flow
        prediction_dict = {'f_flows':f_flows}
        return prediction_dict

def get_gru_decoder(future_len):
    model = Decoder(future_len)
    return model