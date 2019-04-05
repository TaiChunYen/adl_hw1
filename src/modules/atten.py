import torch
import numpy as np
import pdb

class Attension(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(Attension, self).__init__()
        self.lstm = torch.nn.LSTM(dim_embeddings, 128,bidirectional=True)
        self.lstm2 = torch.nn.LSTM(dim_embeddings, 128,bidirectional=True)
        
        #self.wam = torch.nn.Linear(256, 256, bias=False)
        #self.wqm = torch.nn.Linear(256, 256, bias=False)
        #self.wms = torch.nn.Linear(256, 1, bias=False)
        #self.att = torch.nn.Linear(256, 256, bias=False)
        self.sm = torch.nn.Softmax(dim=-1)
        self.lstm3 = torch.nn.LSTM(256*4, 128,bidirectional=True,num_layers=2,dropout=0.2)
        self.m = torch.nn.Bilinear(256,256,1)
        self.drop = torch.nn.Dropout(0.2)
        #self.coss = torch.nn.CosineSimilarity(dim=1)
        

    def forward(self, context, context_lens, options, option_lens):
        #context=self.drop(context)
        out2,self.hidden2 = self.lstm2(context.permute([1,0,2]))
        #bn2 = torch.nn.BatchNorm1d(context.shape[0],momentum=0.5).cuda()
        #out2 = bn2(out2)
        out2 = self.drop(out2)
        
        pool2 = torch.nn.MaxPool1d(kernel_size=out2.shape[0])        
        encoding2 = pool2(out2.permute([2,1,0]))
        encoding2 = encoding2.permute([2,1,0])[0]
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            #option=self.drop(option)
            out,self.hidden  = self.lstm(option.permute([1,0,2]))
            #bn = torch.nn.BatchNorm1d(256,momentum=0.5).cuda()#
            #out = bn(out)
            out = self.drop(out)
            
            pool = torch.nn.MaxPool1d(kernel_size=out.shape[0])
            #ec = pool(out.permute([2,1,0]))
            #ec = ec.permute([2,1,0])[0]
            out_=out.permute([1,0,2]).contiguous()
            out2_=out2.permute([1,0,2]).contiguous()

            batch_size, output_len, dimensions = out_.size()
            query_len = out2_.size(1)            

            '''rq = encoding2.view([batch_size,1,dimensions])
            rq = rq.expand(batch_size,output_len,dimensions).contiguous()            
            rq = rq.view(batch_size*output_len,dimensions)
            ra = out_.view(-1,dimensions)
            atten = torch.tanh(torch.add(self.wqm(rq),self.wam(ra)))
            atten = self.wms(atten)
            s0 = atten.view(batch_size,output_len)
            s0 = self.sm(s0)
            sd = torch.zeros(batch_size,dimensions,dimensions).cuda()
            sd.as_strided(s0.size(),[sd.stride(0), sd.size(2) + 1]).copy_(s0)
            mix = torch.bmm(out_,sd)'''
            
            '''out_ = out_.view(batch_size * output_len, dimensions)
            out_ = self.att(out_)
            out_ = self.drop(out_)'''
                       
            out_ = out_.view(batch_size, output_len, dimensions)
            attention_scores = torch.bmm(out_, out2_.transpose(1, 2).contiguous())
            attention_scores = attention_scores.view(batch_size *output_len , query_len)
            attention_weights = self.sm(attention_scores)
            attention_weights = attention_weights.view(batch_size, output_len, query_len)
            mix = torch.bmm(attention_weights, out2_)                    
            

            l2=torch.cat((out_,mix,out_*mix,out_-mix),2)
            #l2=self.drop(l2)
            l3,self.hidden3=self.lstm3(l2.permute([1,0,2]))           
            l3 = self.drop(l3)
            encoding = pool(l3.permute([2,1,0]))
            encoding = encoding.permute([2,1,0])[0]
            logit = self.m(encoding,encoding2)
            #logit = self.coss(encoding,encoding2)
            logits.append(logit)

        logits = torch.stack(logits, 1)
        logits = logits.permute([2,0,1])[0]
        

        return logits
