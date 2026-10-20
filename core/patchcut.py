"""Independent metric alpha-expansion using SciPy's integer max-flow solver.

No upstream reconstruction code is copied. Pairwise seam costs are a metric
(sum of RGB L1 distances at the two ends + Potts), enabling submodular moves.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from scipy.sparse.linalg import cg,LinearOperator

def binary_cut(d0,d1,i,j,e00,e01,e10,e11,scale=4096):
    n=len(d0);a=np.asarray(d0,np.float64).copy();b=np.asarray(d1,np.float64).copy()
    c=(e01+e10-e00-e11)*.5
    if len(c) and c.min() < -1e-5:raise ValueError('Non-submodular move')
    c=np.maximum(c,0)
    np.add.at(b,i,e10-e00-c);np.add.at(b,j,e01-e00-c)
    minimum=np.minimum(a,b);a-=minimum;b-=minimum
    # Label 0 = source-reachable; source->node cuts incur d1.
    nodes=np.arange(n);src=n;sink=n+1
    rows=np.concatenate([np.full(n,src),nodes,i,j]);cols=np.concatenate([nodes,np.full(n,sink),j,i])
    values=np.rint(np.concatenate([b,a,c,c])*scale).clip(0,2**30-1).astype(np.int64)
    graph=coo_matrix((values,(rows,cols)),shape=(n+2,n+2)).tocsr();graph.eliminate_zeros()
    flow=maximum_flow(graph,src,sink,method='dinic')
    residual=graph-flow.flow;residual.data=(residual.data>0).astype(np.int8);residual.eliminate_zeros()
    reachable=breadth_first_order(residual,src,directed=True,return_predecessors=False)
    select=np.ones(n+2,bool);select[reachable]=False
    return select[:n]

def solve(C,Q,i,j,smoothness=4.,floor=.03,epochs=2,edge_weight=None,log=None):
    start=time.perf_counter();K,N=Q.shape;ix=np.arange(N)
    # Hidden-camera RGB is not an observation. Give it the best available
    # local colour for seam-cost evaluation, while forbidding that label with
    # its unary cost. This preserves the metric and avoids phantom black seams.
    C=np.array(C,copy=True);reference=C[Q.argmax(0),ix].copy()
    for k in range(K):
        absent=Q[k]<.06;C[k,absent]=reference[absent]
    valid=(Q>=.06)&(Q>=Q.max(0)*.2)
    D=-np.log(np.maximum(Q,1e-8)/np.maximum(Q.max(0),1e-8)[None])
    D=np.where(valid,D,10000).astype(np.float32)
    labels=Q.argmax(0).astype(np.int32)
    ew=np.ones(len(i),np.float32) if edge_weight is None else np.asarray(edge_weight,np.float32)
    def pair(a,b):
        v=(np.abs(C[a,i]-C[b,i]).sum(-1)+np.abs(C[a,j]-C[b,j]).sum(-1))*.5
        return smoothness*ew*(v+floor)*(a!=b)
    def energy(l):return float(D[l,ix].sum(dtype=np.float64)+pair(l[i],l[j]).sum(dtype=np.float64))
    initial=energy(labels);history=[initial];cuts=0
    for epoch in range(epochs):
        changed=False
        for alpha in range(K):
            if not valid[alpha].any():continue
            li=labels[i];lj=labels[j]
            e00=pair(li,lj);e01=pair(li,alpha);e10=pair(alpha,lj);e11=np.zeros_like(e00)
            move=binary_cut(D[labels,ix],D[alpha],i,j,e00,e01,e10,e11)
            candidate=labels.copy();candidate[move]=alpha;new=energy(candidate);cuts+=1
            if new<history[-1]-1e-5:
                labels=candidate;history.append(new);changed=True
            if log:log(dict(epoch=epoch,alpha=alpha,energy=history[-1],elapsed=time.perf_counter()-start))
        if not changed:break
    return labels,dict(seconds=time.perf_counter()-start,nodes=N,edges=len(i),cuts=cuts,initial_energy=initial,
                       final_energy=history[-1],accepted_moves=len(history)-1,invalid_choices=int((~valid[labels,ix]).sum()),
                       changed_labels=float((labels!=Q.argmax(0)).mean()))

def level_seams(C,Q,labels,i,j,screen=.01,max_gain=1.3):
    """Screened log-colour Poisson correction, preserving black and fine gradients.

    Only cross-camera seams produce a forcing term; ordinary source gradients
    are retained. Cross-view gradients require observations at both endpoints.
    """
    start=time.perf_counter();N=len(labels);ix=np.arange(N);chosen=C[labels,ix]
    log=np.log(C.clip(0)+.003);Y=log[labels,ix]
    different=labels[i]!=labels[j];ii=i[different];jj=j[different];li=labels[ii];lj=labels[jj]
    supported=(Q[li,ii]>=.06)&(Q[li,jj]>=.06)&(Q[lj,ii]>=.06)&(Q[lj,jj]>=.06)
    ii=ii[supported];jj=jj[supported];li=labels[ii];lj=labels[jj]
    grad=.5*((log[li,ii]-log[li,jj])+(log[lj,ii]-log[lj,jj]))
    residual=grad-(Y[ii]-Y[jj]);b=np.zeros((N,3),np.float32)
    np.add.at(b,ii,residual);np.add.at(b,jj,-residual)
    degree=np.bincount(np.concatenate([i,j]),minlength=N).astype(np.float32)
    diag=degree+screen
    A=coo_matrix((np.concatenate([diag,-np.ones(2*len(i),np.float32)]),
                 (np.concatenate([ix,i,j]),np.concatenate([ix,j,i]))),shape=(N,N)).tocsr()
    pre=LinearOperator((N,N),matvec=lambda x:x/diag,dtype=np.float32)
    correction=np.zeros((N,3),np.float32);statuses=[]
    for channel in range(3):
        correction[:,channel],info=cg(A,b[:,channel],M=pre,rtol=1e-5,atol=1e-7,maxiter=300);statuses.append(int(info))
    correction=np.clip(correction,-np.log(max_gain),np.log(max_gain))
    corrected=chosen*np.exp(correction)
    return corrected,correction,dict(seconds=time.perf_counter()-start,supported_seam_edges=len(ii),
        solver_status=statuses,max_gain=max_gain,max_log_correction=float(abs(correction).max()))
