import numpy as np
from matplotlib import pyplot as plt



def getGuessCost(m):
    """

    """
    
    return guess_cost_cst*(m**guess_cost_potens)

def T(m):
    """

    """
    return cost_cst*(m**cost_potens)

def getCost(m,n):
    """
    m: Size of original matrix
    n: Size of matrix we can guess

    """
    s = np.log2(m)

    cost = 0
    for i in range(int(s)+1):
        r = 2**i
        print('m=%g,n=%g,cost=%g \t r = %g' %(m,n,cost,r))
        if n==m/r:
            if n!=1:
                cost += r*getGuessCost(n)
            break
        else:
            cost += r*T(m/r)
        print('m=%g,n=%g,cost=%g \t r = %g' %(m,n,cost,r))            
    
    # print('m=%g,n=%g,cost=%g \t r = %g' %(m,n,cost,r))
    return cost

guess_cost_cst = 10
guess_cost_potens = 0
cost_cst = 10
cost_potens = 2
fig = plt.figure()
ax = fig.gca()

m = 2**6
nmax = int(np.log2(m))

for cost_potens in [1,1.2,1.4,1.6,1.8,2]:
    costlist = [getCost(m,2**n)/(getCost(m,1)) for n in range(nmax+1)]
    ax.plot(range(nmax+1),costlist)


# costlist = [getCost(m,2**n)/getCost(m,0) for n in range(nmax)]
# ax.plot(np.power(2,range(nmax)),costlist)

# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_ylim([0,1.2])


fig.savefig('cost_analysis.png')
