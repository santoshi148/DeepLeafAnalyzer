import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def classification_metrics(y_true, y_pred):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    wp,wr,wf,_=precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0)
    return {'accuracy':accuracy_score(y_true,y_pred),'macro_precision':p,'macro_recall':r,'macro_f1':f,'weighted_f1':wf}
