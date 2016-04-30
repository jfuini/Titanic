# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 11:15:46 2016

@author: John Fuini
"""

def f1(pred, ans):
    tp = 0
    fp = 0
    fn = 0
    for i in range(0,len(pred)):
        if pred[i] == 1 and ans[i] == 1:
            tp = tp + 1
        elif pred[i] == 1 and ans[i] == 0:
            fp = fp + 1
        elif pred[i] == 0 and ans[i] == 1:
            fn = fn + 1
    
    prec = float(tp)/(tp + fp)
    rec = float(tp)/(tp + fn)             
    
    f1 = 2.0*prec*rec/(prec + rec)
    return f1