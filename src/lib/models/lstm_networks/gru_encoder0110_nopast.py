#_backwardsmallpastmaskfutureflowloss_noid_featuretahn
#(final)
import torch
import torch.nn as nn

class Hidden_state_conv(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(Hidden_state_conv, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=True)
    def forward(self, x):
        return torch.tanh(self.conv1(x))
'''
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
'''
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=64):
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
'''
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.hidden_state_conv = Hidden_state_conv(input_dim=64, hidden_dim=128)
        #encoder
        setattr(self, 'E_gru', ConvGRU(hidden_dim=128, input_dim=64))
        #self.convE = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        #setattr(self, 'ED_flow', FlowHead(input_dim=128, hidden_dim=256))

        #id
        #self.heads = {'hm': 1,
        #        'wh': 4,
        #        'id': 128,
        #        'reg': 2}
        '''
        self.heads = {'id': 128, }
        head_conv = 256
        final_kernel = 1
        input_dim = 64
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(input_dim, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(input_dim, classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
        '''
    def forward(self, state, features, flag):
        #if torch.equal(flag,torch.cuda.FloatTensor(1).fill_(1)): #is one, first feature doesnt have hidden state
        #    state = self.hidden_state_conv(features)
        if torch.equal(flag,torch.ones(1)): #is one, first feature doesnt have hidden state
            state = self.hidden_state_conv(features)

        state = getattr(self, 'E_gru')(state, features)
        #encoded = self.convE(state) #64
        #det
        '''
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(encoded)
        '''
        #decoder, predict t-1~t
        #p_delta_flow = getattr(self, 'ED_flow')(encoded)
        #p_delta_flow = getattr(self, 'ED_flow')(state) #large past flow
        return state #, p_delta_flow #, z
        #return state, encoded, z

def get_gru_encoder():
    model = Encoder()
    return model