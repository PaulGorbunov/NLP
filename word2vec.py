import time
import re
import math
import numpy as np
from random import randint as ri
from functools import *
import sys
import pickle
import warnings


warnings.filterwarnings('error')

def get_text(name):
    with open(name,"r") as f:
        te = f.read()
    return te

def prep_text(t):
    for j in range(len(t)):
        t[j] = re.findall("(?=[a-zA-ZÇâêîôûàèùëïüé]{1})[a-zA-ZÇâêîôûàèùëïüé-]*[a-zA-ZÇâêîôûàèùëïüé]+['’]?[a-zA-Z]{0,4}",t[j])
        for k in range(len(t[j])):
            t[j][k] = t[j][k].lower()

features = 15
win = 1
end = 500
theta = 0.001
books = ["alp.txt"]
texts = [get_text(e) for e in books]
prep_text(texts)
lxcn = []
one_hot = {}
matrix_w = [] #lxcn x features
matrix_u = [] #features x lxcn
        
def cr_lexcn(lex,cn):
    for lan in lex:
        for word in lan:
            if not word in cn:
                cn.append(word)
    cn.sort()

def cr_1_hot(l,t):
    for k in range(len(l)):
        t[l[k]] = [[0]*len(l)]
        t[l[k]][0][k] = 1 

def cr_matrix():
    matrix_w.clear()
    matrix_u.clear()
    (lambda:[matrix_w.append([round(0.001*ri(1,9),3) for f in range(features)])for u in range(len(lxcn))])() #lxcn x features
    (lambda:[matrix_u.append([round(0.001*ri(1,9),3) for f in range(len(lxcn))])for u in range(features)])() #features x lxcn
    
def check_us(ind,te):
    bou =  lambda x: True if x >= 0 and x < len(te) else False
    return [te[i] for i in range(ind-1,ind-win-1,-1) if  bou(i)] + [te[i] for i in range(ind+1,ind+win+1,1) if  bou(i)] 
    
def softmax_vec(vec):
    dev = sum(list(map(lambda x: math.e**x,vec)))
    return [math.e**(v)/dev for v in vec]
    

def gradient(sm_vec,u_position,w_vec,fl):
    if fl == 0:
        one_hot_u = [[0] for u in range(len(lxcn))]
        one_hot_u[u_position] = [1]
        u0 = np.transpose(np.dot(matrix_u, one_hot_u))[0] #ndarray
        expect_u = sum([sm_vec[i]*np.transpose(matrix_u)[i] for i in range(len(lxcn))])
        grad_v  = u0 - expect_u
        gen = [(d,grad_v[d]) for d in range(len(grad_v))] #grad_v[d]
    else:
        exp_v = sm_vec[u_position] * w_vec
        grad_u  = w_vec - exp_v
        gen = [(d,grad_u[d]) for d in range(len(grad_u))] #grad_u[d]
    return gen

    
    
def step(vc,text):
    ind_l = [e for e in range(len(text)) if text[e] == vc]
    near_w = list(reduce(lambda x,y:x+y,[check_us(i,text) for i in ind_l]))
    
    w_vec = np.dot(one_hot[vc],matrix_w)[0] # ndarray
    step_m = np.dot(w_vec,matrix_u) # ndarray
    try:
        dev = sum(list(map(lambda x: math.e**(x),step_m)))
        sm_vec = [math.e**(t)/dev for t in step_m]
    except Warning:
        print ("Warning was raised")
        sys.exit(0)
    for s in range(len(near_w)):
        u_pos = lxcn.index(near_w[s])
        u0 = np.transpose(matrix_u)[u_pos] # ndarray
        exp_u = sum([sm_vec[i] * np.transpose(matrix_u)[i] for i in range(len(lxcn))]) #ndarray
        grad_v = u0 - exp_u #ndarray
        grad_u = w_vec - sm_vec[u_pos]*w_vec #ndarray
        
        num = lxcn.index(vc)
        for u in range(len(matrix_w[num])):
            matrix_w[num][u] += theta * grad_v[u]
        
        for h in range(features):
            matrix_u[h][u_pos] += theta*grad_u[h]

def start():
    cou = 0
    while True:
        for text in texts :
            gen = (word for word in lxcn)
            for c_wrd in gen:
                step(c_wrd,text)
                    
                         
        cou += 1
        if cou % (end*0.05) == 0:
            with open("tmp","wb") as f:
                pickle.dump(("o_h,lex,w,u,feat,win",one_hot,lxcn,matrix_w,matrix_u,features,win),f)
            print((cou/end)*100, "%")
        if cou == end:
            break
    
def check(w):
    vc = np.dot(one_hot[w],matrix_w)
    s = softmax_vec(np.dot(vc,matrix_u)[0])
    so = softmax_vec(np.dot(vc,matrix_u)[0])
    so.sort()
    so = list(reversed(so))
    o = 0
    for u in so:
        print(lxcn[s.index(u)])
        o += 1
        if o >= win*2:
            break

def auto_comp(word):
    vc = np.dot(one_hot[word],matrix_w)
    ans = softmax_vec(np.dot(vc,matrix_u)[0])
    ave = sum(ans)/len(ans)
    m = max(ans)
    adv_words = [lxcn[i] for i in range(len(ans)) if m - ans[i] <  ave]
    print(adv_words)
    
def test(text):
    m = len(lxcn)
    t = 0
    for u in lxcn:
        vc = np.dot(one_hot[u],matrix_w)
        s = softmax_vec(np.dot(vc,matrix_u)[0])
        pred_w = lxcn[s.index(max(s))]
        ind_l = [e for e in range(len(text)) if text[e] == pred_w]
        near_w = list(reduce(lambda x,y:x+y,[check_us(i,text) for i in ind_l]))     
        if (u in near_w):
            t += 1
    print ((t/m)*100,"%")
        
def load(name):
    with open(name,"rb") as f:
        q = pickle.load(f)
    for u in q[1].keys():
        one_hot[u] = q[1][u]
    lxcn.clear()
    matrix_w.clear()
    matrix_u.clear()
    (lambda: [lxcn.append(u) for u in q[2]])()
    (lambda: [matrix_w.append(u) for u in q[3]])()
    (lambda: [matrix_u.append(u) for u in q[4]])()

def more(name):
    start_time = time.time()
    load(name)
    start()
    print("\n--- %s seconds ---" % (time.time() - start_time))

def main():
    start_time = time.time()
    cr_lexcn(texts,lxcn)
    cr_1_hot(lxcn,one_hot)
    cr_matrix()
    start()
    print("\n--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()

'''
def step_(matrix_w,matrix_u,win,vc,o_h,text,lxcn):
    ind_l = [e for e in range(len(text)) if text[e] == vc]
    near_w = list(reduce(lambda x,y:x+y,[check_us(i,text,win) for i in ind_l]))
    ans = []
    w_vec = np.dot(o_h,matrix_w)[0] # ndarray
    step_m = np.dot(w_vec,matrix_u) # ndarray
    dev = sum(list(map(lambda x: np.e**(x),step_m)))
    sm_vec = [np.e**(t)/dev for t in step_m]
    for s in range(len(near_w)):
        u_pos = lxcn.index(near_w[s])
        u0 = np.transpose(matrix_u)[u_pos] # ndarray
        exp_u = sum([sm_vec[i] * np.transpose(matrix_u)[i] for i in range(len(lxcn))]) #ndarray
        grad_v = u0 - exp_u #ndarray
        grad_u = w_vec - sm_vec[u_pos]*w_vec #ndarray
        ans.append([o_h,u_pos,grad_v,grad_u])
    return ans

'''
