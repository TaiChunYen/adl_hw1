import torch
import pdb

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        # TODO
        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.	
        self.n += predicts.size()[0]
        
        if(predicts.size()[1]<self.at+1):
             self.n_corrects += predicts.size()[0]
        else:
             for i in range(predicts.size()[0]):
                    m=sorted(predicts[i])[-(self.at+1)]
                    for j in range(predicts.size()[1]):
                           if(predicts[i][j]>m):
                                  predicts[i][j]=1
                           else:
                                  predicts[i][j]=0
                    predicts[i]=predicts[i].long()                 
                    if(batch['labels'][i][0]*predicts[i][0]==1):
                           self.n_corrects += 1
                    #pdb.set_trace()
                                                                                                                                                                 
                                                                  
                            

    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
