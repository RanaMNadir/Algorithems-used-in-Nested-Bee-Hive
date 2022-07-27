import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# plot 3d space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
df = pd.read_csv ('125000_data.csv')
df = pd.DataFrame(df)


a=df.z.tolist()
b=df.y.tolist()
c=df.x.tolist()
ax.scatter(a,b,c, c='r', marker='o', alpha=0.2)

#directional_trvers
def trevers(sx,sy,sz,nodex,nodey,nodez):
    weight=[]
    tim=0
    x=sx
    y=sy
    z=sz
    j=1
    a=[]
    b=[]
    c=[]
    a.append(sx)
    b.append(sy)
    c.append(sz)
   # print(nodex,nodey,nodez)
    #print(x, y,z)

    while (j):
        ndf=df[(df['x']== x) & ( df['y']== y) & (df['z']== z) ]


        #print(ndf)


        zpath_up= ndf.z_up.values
        zpath_down=ndf.z_down.values
        ypath_up=ndf.y_up.values
        ypath_down=ndf.y_down.values
        xpath_up=ndf.X_up.values
        xpath_down=ndf.X_down.values
        #print(xpath_up)
        s = time.time()

        if (x==nodex and y==nodey and z==nodez):
            print('welcome to destination:',x,',',y,',',z)
            j=0
        else:

            if(x<nodex):
                    x=x+1
                    weight.append(xpath_up)


            elif(x>nodex):
                    x=x-1
                    weight.append(xpath_down)

            elif(y<nodey):
                    y=y+1
                    weight.append(ypath_up)

            elif(y>nodey):
                    y=y-1
                    weight.append(ypath_down)

            elif (z < nodez):
                    z = z + 1
                    weight.append(zpath_up)

            elif(z>nodez):
                    z = z - 1
                    weight.append(zpath_down)
            a.append(x)
            b.append(y)
            c.append(z)
            e = time.time()
            tim=tim+(e-s)


    return a,b,c,weight,tim

columns_names=["Max_nodes","From","To","Time","Cost","Hopes"]
nf=pd.DataFrame(columns=columns_names)
for h in range(1, 50):

    # start state
    sx=0
    sy=0
    sz=0

    # Destination state

    dx=49
    dy=49
    dz = h
    ddf = df[(df['x']== dx) & ( df['y']== dy) & (df['z']== dz) ]
    print(ddf)
    dest=ddf.vert.values
    print(dest[0])
#====================================================lambda 1=================---xyz----=======================================
    f=trevers(sx,sy,sz,dx,dy,dz)

    x=f[0]
    y=f[1]
    z=f[2]
    we=f[3]
    tim=f[4]


    w_whole = sum(we)
    nf = nf.append({"Max_nodes": dest[0]+1, "From": 0.0, "To": dest[0], "Time": tim, "Cost": w_whole[0],
                    "Hopes": len(z) - 1}, ignore_index=True)
    print(nf)

nf.to_csv("Directional_trevers_results.csv")
