import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plot 3d space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
df = pd.read_csv('125000_data.csv')
df = pd.DataFrame(df)

a = df.z.tolist()
b = df.y.tolist()
c = df.x.tolist()
ax.scatter(a, b, c, c='r', marker='o', alpha=0.1)


# greedy
def trevers(sx, sy, sz, nodex, nodey, nodez):
    x = sx
    y = sy
    z = sz
    # print(nodex,nodey,nodez)
    # print(x, y,z)
    j = 1
    tim = 0
    a = []
    b = []
    c = []
    while (j):

        ndf = df[(df['x'] == x) & (df['y'] == y) & (df['z'] == z)]

        # print("Time Data Frame: ", e - s)
        # print(ndf)

        zpath_up = ndf.z_up.values
        zpath_down = ndf.z_down.values
        ypath_up = ndf.y_up.values
        ypath_down = ndf.y_down.values
        xpath_up = ndf.X_up.values
        xpath_down = ndf.X_down.values
        # dest = ndf.vert.values
        # print(xpath_up)

        s = time.time()

        if (x == nodex and y == nodey and z == nodez):
            print('welcome to destination:', x, ',', y, ',', z)
            j = 0

        else:
            # in case of - - -
            if (nodex < x and nodey < y and nodez < z):
                # print("nodex < x and nodey < y and nodez < z ")
                if (xpath_down == min(xpath_down, ypath_down, zpath_down)):
                    x = x - 1
                    # print("current state:",x,',',y,',',z)
                elif (ypath_down == min(xpath_down, ypath_down, zpath_down)):
                    y = y - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(xpath_down, ypath_down, zpath_down)):
                    z = z - 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of + - -
            elif (nodex > x and nodey < y and nodez < z):
                # print("nodex > x and nodey < y and nodez < z")
                if (xpath_up == min(xpath_up, ypath_down, zpath_down)):
                    x = x + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_down == min(xpath_up, ypath_down, zpath_down)):
                    y = y - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(xpath_up, ypath_down, zpath_down)):
                    z = z - 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of - + -
            elif (nodex < x and nodey > y and nodez < z):
                # print("nodex < x and nodey > y and nodez < z")
                if (xpath_down == min(xpath_down, ypath_up, zpath_down)):
                    x = x - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_up == min(xpath_down, ypath_up, zpath_down)):
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(xpath_down, ypath_up, zpath_down)):
                    z = z - 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of - - +
            elif (nodex < x and nodey < y and nodez > z):
                # print("nodex < x and nodey < y and nodez > z")
                if (xpath_down == min(xpath_down, ypath_down, zpath_up)):
                    x = x - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_down == min(xpath_down, ypath_down, zpath_up)):
                    y = y - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_up == min(xpath_down, ypath_down, zpath_up)):
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of + + +
            elif (nodex > x and nodey > y and nodez > z):
                # print("nodex > x and nodey > y and nodez > z")
                if (xpath_up == min(xpath_up, ypath_up, zpath_up)):
                    x = x + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_up == min(xpath_up, ypath_up, zpath_up)):
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_up == min(xpath_up, ypath_up, zpath_up)):
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of - + +
            elif (nodex < x and nodey > y and nodez > z):
                # print("nodex < x and nodey > y and nodez > z")
                if (xpath_down == min(xpath_down, ypath_up, zpath_up) and nodex < 0):
                    x = x - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_up == min(xpath_down, ypath_up, zpath_up) and nodex < 0):
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_up == min(xpath_down, ypath_up, zpath_up) and nodex < 0):
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of + - +
            elif (nodex > x and nodey < y and nodez > z):
                # print("nodex > x and nodey < y and nodez > z")
                if (xpath_up == min(xpath_up, ypath_down, zpath_up)):
                    x = x + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_down == min(xpath_up, ypath_down, zpath_up) and nodey < 0):
                    y = y - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_up == min(xpath_up, ypath_down, zpath_up)):
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of + + -
            elif (nodex > x and nodey > y and nodez < z):
                # print("nodex > x and nodey > y and nodez < z")
                if (xpath_up == min(xpath_up, ypath_up, zpath_down)):
                    x = x + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_up == min(xpath_up, ypath_up, zpath_down)):
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(xpath_up, ypath_up, zpath_down)):
                    z = z - 1
                    # print("current state:", x, ',', y, ',', z)

            # =============================== equal Section ============================================
            # in case of = + -
            elif (nodex == x and nodey > y and nodez < z):
                # print("nodex = x and nodey > y and nodez < z")

                if (ypath_up == min(ypath_up, zpath_down)):
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(ypath_up, zpath_down) and nodez < 0):
                    z = z - 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of = - +
            elif (nodex == x and nodey < y and nodez > z):
                # print("nodex = x and nodey < y and nodez > z")

                if (ypath_down == min(ypath_down, zpath_up)):
                    y = y - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(ypath_up, zpath_up)):
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of = + +
            elif (nodex == x and nodey > y and nodez > z):
                # print("nodex = x and nodey > y and nodez > z")

                if (ypath_up == min(ypath_up, zpath_up)):
                    # print
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)
                # elif (zpath_up == min(ypath_up, zpath_up)):
                else:
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of + = +
            elif (nodex > x and nodey == y and nodez > z):
                # print("nodex > x and nodey == y and nodez > z")

                if (xpath_up == min(xpath_up, zpath_up)):
                    x = x + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_up == min(xpath_up, zpath_up)):
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of - = -
            elif (nodex < x and nodey == y and nodez < z):
                # print("nodex < x and nodey == y and nodez < z")

                if (xpath_down == min(xpath_down, zpath_down)):
                    x = x - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(xpath_down, zpath_down)):
                    z = z - 1
                    # print("current state:", x, ',', y, ',', z)


            # in case of + = -
            elif (nodex > x and nodey == y and nodez < z):
                # print("nodex > x and nodey == y and nodez < z")

                if (xpath_up == min(xpath_up, zpath_down)):
                    x = x + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_down == min(xpath_up, zpath_down)):
                    z = z - 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of - = +
            elif (nodex < x and nodey == y and nodez > z):
                # print("nodex < x and nodey == y and nodez > z")

                if (xpath_down == min(xpath_down, zpath_up)):
                    x = x - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (zpath_up == min(xpath_down, zpath_up)):
                    z = z + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of - + =
            elif (nodex < x and nodey > y and nodez == z):
                # print("nodex < x and nodey > y and nodez = z")

                if (xpath_down == min(xpath_down, ypath_up)):
                    x = x - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_up == min(xpath_down, ypath_up)):
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)

            # in case of - - =
            elif (nodex < x and nodey < y and nodez == z):
                # print("nodex < x and nodey < y and nodez = z")

                if (xpath_down == min(xpath_down, ypath_down)):
                    x = x - 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_down == min(xpath_down, ypath_down)):
                    y = y - 1
                    # print("current state:", x, ',', y, ',', z)


            # in case of + + =
            elif (nodex > x and nodey > y and nodez == z):
                # print("nodex > x and nodey > y and nodez = z")

                if (xpath_up == min(xpath_up, ypath_up)):
                    x = x + 1
                    # print("current state:", x, ',', y, ',', z)
                elif (ypath_up == min(xpath_up, ypath_up)):
                    y = y + 1
                    # print("current state:", x, ',', y, ',', z)


            # in case of + = =
            elif (nodex > x and nodey == y and nodez == z):
                # print("nodex > x and nodey = y and nodez = z")
                x = x + 1
                # print("current state:", x, ',', y, ',', z)

            # in case of - = =
            elif (nodex < x and nodey == y and nodez == z):
                # print("nodex > x and nodey = y and nodez = z")
                x = x - 1
                # print("current state:", x, ',', y, ',', z)

            # in case of = = +
            elif (nodex == x and nodey == y and nodez > z):
                # print("nodex = x and nodey = y and nodez > z")
                z = z + 1
                # print("current state:", x, ',', y, ',', z)

            # in case of = = -
            elif (nodex == x and nodey == y and nodez < z):
                # print("nodex = x and nodey = y and nodez < z")
                z = z - 1
                # print("current state:", x, ',', y, ',', z)

            # in case of = + =
            elif (nodex == x and nodey > y and nodez == z):
                # print("nodex = x and nodey > y and nodez = z")
                y = y + 1
                # print("current state:", x, ',', y, ',', z)

            # in case of = - =
            elif (nodex == x and nodey < y and nodez == z):
                # print("nodex = x and nodey > y and nodez = z")
                y = y - 1

        a.append(x)
        b.append(y)
        c.append(z)
        e = time.time()
        tim = tim + (e - s)
        # print("current state:", x, ',', y, ',', z)
        # print(x,y,z)
    return a, b, c, tim


columns_names = ["Max_nodes", "From", "To", "Time", "Cost", "Hopes"]
nf = pd.DataFrame(columns=columns_names)
for h in range(1, 50):

    # start state
    sx = 0
    sy = 0
    sz = 0

    # Destination state

    dx = 49
    dy = 49
    dz = h
    ddf = df[(df['x'] == dx) & (df['y'] == dy) & (df['z'] == dz)]
    print(ddf)
    dest = ddf.vert.values
    print(dest[0])

    j = 1
    hop = 0
    i = 0

    f = trevers(sx, sy, sz, dx, dy, dz)
    # print("hola :",f[2][1])
    x = f[0]
    y = f[1]
    z = f[2]
    tim = f[3]
    # for x in (len(x)):
    #  print(x[x])
    print("hola :", x)
    print("Time taken:", tim)

    weight = []
    w_whole = 0
    print(dest)
    print("X-len :", len(x))
    for i in range(len(x) - 1):
        # jdf = df[(df['x'] ==x[i])&(df['y'] ==y[i])&(df['z'] ==z[i])]
        jdf = df[(df['x'] == x[i]) & (df['y'] == y[i]) & (df['z'] == z[i])]
        print("X,y,z :", x[i], ",", y[i], ",", z[i])
        print("jdf :", jdf)
        print("i :", i)
        print(jdf.X_up.values[0])
        xu = jdf.X_up.values[0]
        xd = jdf.X_down.values[0]
        yu = jdf.y_up.values[0]
        yd = jdf.y_down.values[0]
        zu = jdf.z_up.values[0]
        zd = jdf.z_down.values[0]

        if (x[i] < x[i + 1]):
            weight.append(xu)
        elif (x[i] > x[i + 1]):
            weight.append(xd)
        elif (y[i] < y[i + 1]):
            weight.append(yu)
        elif (y[i] > y[i + 1]):
            weight.append(yd)
        elif (z[i] < z[i + 1]):
            weight.append(zu)
        elif (z[i] > z[i + 1]):
            weight.append(zd)

    print("weight : ", weight)
    w_whole = sum(weight)
    nf = nf.append({"Max_nodes": dest[0], "From": 0.0, "To": dest[0], "Time": tim, "Cost": w_whole,
                    "Hopes": len(z) - 1}, ignore_index=True)
    print(nf)

nf.to_csv("Directional_greed_results.csv")
