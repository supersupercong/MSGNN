
import torch

index_neighbours_cache = {}
def index_neighbours(xe_patch, ye_patch, window_size, scale):
    r"""
    This function generates the indexing tensors that define neighborhoods for each query patch in (father) features
    It selects a neighborhood of window_size x window_size patches around each patch in xe (son) features
    Index tensors get cached in order to speed up execution time
    """
    if cfg.NETWORK.WITH_WINDOW == False:
        return None
        # dev = xe_patch.get_device()
        # key = "{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,dev)
        # if not key in index_neighbours_cache:
        #     I = torch.tensor(range(n), device=dev, dtype=torch.int64).view(1,1,n)
        #     I = I.repeat(b, m, 1)
        #     index_neighbours_cache[key] = I

        # I = index_neighbours_cache[key]
        # return Variable(I, requires_grad=False)

    b,_,_,_,n1,n2 = xe_patch.shape
    s = window_size
    
    if s>=n1 and s>=n2: 
        cfg.NETWORK.WITH_WINDOW = False
        return None

    s = min(min(s, n1), n2)
    o = s**2
    b,_,_,_,m1,m2 = ye_patch.shape

    dev = xe_patch.get_device()
    key = "{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,dev)
    if not key in index_neighbours_cache:
        I = torch.empty(1, m1 * m2, o, device=dev, dtype=torch.int64)

        ih = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,s,1)
        iw = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,1,s)*n2

        i = torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m1,1,1,1)
        j = torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1,m2,1,1)

        i_s = (torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m1,1,1,1)//2.0).long()
        j_s = (torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1,m2,1,1)//2.0).long()

        ch = (i_s-s//scale).clamp(0,n1-s)
        cw = (j_s-s//scale).clamp(0,n2-s)

        cidx = ch*n2+cw
        mI = cidx + ih + iw
        mI = mI.view(m1*m2,-1)
        I[0,:,:] = mI

        index_neighbours_cache[key] = I

    I = index_neighbours_cache[key]
    I = I.repeat(b,1,1)

    return Variable(I, requires_grad=False)

if __name__ == '__main__':
    index_neighbours(xe_patch=torch.ones(3,256,256,64,64), ye_patch=torch.ones(3,256,256,64,64), window_size=30, scale=2)   