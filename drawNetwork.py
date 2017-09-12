import numpy as np

width = 12
height= 10/1.5
box = [0,width,0,height]
nlayer = 3
nnodes = [2,3,2]

cf = 0.1
cfw = 0.0
hsep = height*(1-cf)/(max(nnodes)-1)

pos = np.zeros(shape=(nlayer,max(nnodes)),dtype=object)

for i in range(nlayer):
    nnodes_i = nnodes[i]
    bottom = height/2 - hsep*(nnodes_i-1)/2
    for j in range(nnodes_i):
        pos[i][j] = [(i)*width*(1-cfw)/(nlayer-1)+width*cfw/2,
                     bottom + j*hsep]

# pos = [[width/(nlayer+1),height/(nnodes+1)],
#        [2*width/(nlayer+1),height/(nnodes+1)],
#        [3*width/(nlayer+1),height/(nnodes+1)]]
radius = 1.2


print('\\begin{tikzpicture}')
# print('[cir/.style={circle,draw,black,minimum size=%gcm}]' %(radius))
print('\\centering')

print('\\node at (%g,%g) {  };' %(0,0))
print('\\node at (%g,%g) {  };' %(0,height))
print('\\node at (%g,%g) {  };' %(width,height))
print('\\node at (%g,%g) {  };' %(width,0))

for i in range(nlayer):
    nnodes_i = nnodes[i] 
    for j in range(nnodes_i):
        position = pos[i][j]
        # print('\\node[cir,name=n%i%i] at (%g,%g) {};' %(i,j,position[0],position[1]))
        if i ==1 and j == nnodes_i-1:
            print('\\node[draw,circle,minimum size=%gcm,name=n%i%i,align=center] at (%g,%g) {$z_{1}^{(1)} =$\\\\$ \\text{tanh}([\\vec{x}^TW^{(1)}]_1+ b_1^{(1)})$};' %(radius,i,j,position[0],position[1]))
        elif i ==2 and j == nnodes_i-1:
            print('\\node[draw,circle,minimum size=%gcm,name=n%i%i,align=center] at (%g,%g) {$\hat{y}_{1} =$\\\\$ \\text{softmax}([z^{T(1)}W^{(2)}]_1 + b_1^{(2)})$};' %(radius,i,j,position[0],position[1]))
        elif i ==0 and j == nnodes_i-1:
            print('\\node[draw,circle,minimum size=%gcm,name=n%i%i,align=center] at (%g,%g) {$x_1$};' %(radius,i,j,position[0],position[1]))
        elif i ==0 and j == 0:
            print('\\node[draw,circle,minimum size=%gcm,name=n%i%i,align=center] at (%g,%g) {$x_2$};' %(radius,i,j,position[0],position[1]))                        
        else:
            print('\\node[draw,circle,minimum size=%gcm,name=n%i%i] at (%g,%g) {};' %(radius,i,j,position[0],position[1]))



for j in range(nlayer):
    nnodes_tmp = nnodes[j]    
    for k in range(nnodes_tmp):
        if j != nlayer-1:
            nnodes_tmp2 = nnodes[j+1]
            for i in range(nnodes_tmp2):
                if j==0 and k==1 and i == nnodes_tmp2-1:
                    print('\\draw[->,name=a%i%i%i%i] (n%i%i) -- node[sloped,anchor=center,above] {$W_{11}^{(1)}$} (n%i%i);' %(j,k,j+1,i,j,k,j+1,i))
                elif j==1 and k==nnodes_tmp-1 and i == nnodes_tmp2-1:
                    print('\\draw[->,name=a%i%i%i%i] (n%i%i) -- node[sloped,anchor=center,above] {$W_{11}^{(2)}$} (n%i%i);' %(j,k,j+1,i,j,k,j+1,i))
                else:
                    print('\\draw[->,name=a%i%i%i%i] (n%i%i) -- (n%i%i);' %(j,k,j+1,i,j,k,j+1,i))                    

            

    



print('\\end{tikzpicture}')

