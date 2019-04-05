import torch
import numpy as np
import pdb


class LSTM(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(dim_embeddings, 256)
        self.lstm2 = torch.nn.LSTM(dim_embeddings, 256)
        #self.bn = torch.nn.BatchNorm1d(256)
        #self.bn2 = torch.nn.BatchNorm1d(256)
        self.m = torch.nn.Bilinear(256,256,1)
        
        self.emb=dim_embeddings
    def forward(self, context, context_lens, options, option_lens):
        out2,self.hidden2 = self.lstm2(context.permute([1,0,2]))
        pool2 = torch.nn.MaxPool1d(kernel_size=out2.shape[0])
        encoding2 = out2#[-1]
        #encoding2 = self.bn2(encoding2)
        encoding2 = pool2(encoding2.permute([2,1,0]))
        encoding2 = encoding2.permute([2,1,0])[0]
        logits = []
        
        for i, option in enumerate(options.transpose(1, 0)):
            
            out,self.hidden  = self.lstm(option.permute([1,0,2]))
            pool = torch.nn.MaxPool1d(kernel_size=out.shape[0])
            encoding = out#[-1]
            #encoding = self.bn(encoding)
            encoding = pool(encoding.permute([2,1,0]))
            encoding = encoding.permute([2,1,0])[0]
            
            logit = self.m(encoding,encoding2)
            
            
            logits.append(logit)
            
        
        logits = torch.stack(logits, 1)
        logits = logits.permute([2,0,1])[0]       
        #pdb.set_trace()     
             
        return logits
