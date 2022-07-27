import queue
import time
from collections import namedtuple
from random import randint
from random import seed
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
seed(5)
val=randint(0,10)
#print(val)
Edge = namedtuple('Edge', ['vertex', 'weight'])

# plot 3d space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
df = pd.read_csv ('125000_data.csv')
df = pd.DataFrame(df)


a=df.z.tolist()
b=df.y.tolist()
c=df.x.tolist()
ax.scatter(a,b,c, c='r', marker='o', alpha=0.1)



class GraphUndirectedWeighted(object):
    def __init__(self, vertex_count):
        self.vertex_count = vertex_count
        self.adjacency_list = [[] for _ in range(vertex_count)]

    def add_edge(self, source, dest, weight):
        assert source < self.vertex_count
        assert dest < self.vertex_count
        self.adjacency_list[source].append(Edge(dest, weight))
        self.adjacency_list[dest].append(Edge(source, weight))

    def get_edge(self, vertex):
        for e in self.adjacency_list[vertex]:
            yield e

    def get_vertex(self):
        for v in range(self.vertex_count):
            yield v


def dijkstra(graph, source, dest):
    q = queue.PriorityQueue()
    parents = []
    distances = []
    start_weight = float("inf")

    for i in graph.get_vertex():
        weight = start_weight
        if source == i:
            weight = 0
        distances.append(weight)
        parents.append(None)


    q.put(([0, source]))

    while not q.empty():
        v_tuple = q.get()
        v = v_tuple[1]

        for e in graph.get_edge(v):
            candidate_distance = distances[v] + e.weight
            if distances[e.vertex] > candidate_distance:
                distances[e.vertex] = candidate_distance
                parents[e.vertex] = v
                # primitive but effective negative cycle detection
                if candidate_distance < -1000:
                    raise Exception("Negative cycle detected")
                q.put(([distances[e.vertex], e.vertex]))

    shortest_path = []
    end = dest

    while end is not None:
        shortest_path.append(end)
        end = parents[end]

    shortest_path.reverse()
    print(shortest_path)
    return shortest_path, distances[dest]


columns_names=["Max_nodes","From","To","Time","Cost","Hopes"]
nf=pd.DataFrame(columns=columns_names)
def main():


   for h in range(2, 50):
      #h=50
      q = -1
      g = GraphUndirectedWeighted(((50*50)*(h+1)))
      print(g)

      global nf
      for k in range(50):

        for j in range(50):
            # x.append(i)

            for i in range(50):
                #combine.append([i, j, k])

                q=q+1
                #print(q)
                vetrtex =q
                vetrtex_x_up = q + 1
                vetrtex_y_up = q + 50
                vetrtex_z_up = q + 2500
                vetrtex_x_down = q - 1
                vetrtex_y_down = q - 50
                vetrtex_z_down = q - 2500
                ndf = df[(df['vert'] == q)]
                # print(ndf)

                z_u = ndf.z_up.values[0]
                z_d = ndf.z_down.values[0]
                y_u = ndf.y_up.values[0]
                y_d = ndf.y_down.values[0]
                x_u = ndf.X_up.values[0]
                x_d = ndf.X_down.values[0]
               # print(q)
                if(i==0 and j==0 and k==0):
                   # vetrtex_x_up=i+1,j,k
                   # vetrtex_y_up = i, j+1, k
                   # vetrtex_z_up = i, j, k+1

                    # print(vetrtex_x_up)
                    # vetrtex_x_up = str(vetrtex_x_up)
                    # print(vetrtex_x_up)

                    g.add_edge(vetrtex,vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)

                elif(i==0 and j==0 and k>0 and k<h):
                    # vetrtex_x_up=i+1,j,k
                    # vetrtex_y_up = i, j+1, k
                    # vetrtex_z_up = i, j, k+1
                    # vetrtex_z_down= i, j, k-1

                    g.add_edge(vetrtex,vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    print(vetrtex,",", vetrtex_z_up,",", z_u)
                    print(k)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)


                elif(i==0 and j>0 and j<49 and k==0):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j-1, k
                    # vetrtex_z_up = i, j, k - 1


                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                elif(i>0 and i<49 and j==0 and k==0):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_z_up = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)

                elif (i > 0 and i < 49 and j> 0 and j<49 and k == 0):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j-1, k
                    # vetrtex_z_up = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)

                elif (i > 0 and i < 49 and j ==0 and k > 0 and k < h):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j , k-1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)

                elif (i ==0 and j > 0 and j < 49 and k > 0 and k < h):
                    # vetrtex_x_up = i + 1, j, k
                    #
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i , j-1, k
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j , k-1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i > 0 and i < 49 and j > 0 and j < 49 and k > 0 and k<h):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j - 1, k
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i == 49 and j == 49 and k == h):

                    # vetrtex_x_down = i-1,j, k
                    #
                    # vetrtex_y_down = i, j - 1, k
                    #
                    # vetrtex_z_down = i, j, k - 1


                    g.add_edge(vetrtex, vetrtex_x_down, x_d)

                    g.add_edge(vetrtex, vetrtex_y_down, y_d)

                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i == 49 and j == 49 and k > 0 and k < h):

                    # vetrtex_x_down = i-1,j, k
                    #
                    # vetrtex_y_down = i, j - 1, k
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j, k - 1


                    g.add_edge(vetrtex, vetrtex_x_down, x_d)

                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down,z_d )
                elif (i == 49 and j > 0 and j < 49 and k == h):

                    # vetrtex_x_down = i-1,j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j - 1, k
                    #
                    # vetrtex_z_down = i, j, k - 1


                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)

                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i > 0 and i < 49 and j == 49 and k == h):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    #
                    # vetrtex_y_down = i, j - 1, k
                    #
                    # vetrtex_z_down = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)

                    g.add_edge(vetrtex, vetrtex_y_down, y_d)

                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i > 0 and i < 49 and j > 0 and j < 49 and k == h):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j - 1, k
                    #
                    # vetrtex_z_down = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)

                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i > 0 and i < 49 and j == 49 and k > 0 and k < h):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    #
                    # vetrtex_y_down = i, j - 1, k
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)

                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i == 49 and j > 0 and j < 49 and k > 0 and k < h):

                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j - 1, k
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j, k - 1


                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif(i==49 and j==0 and k>0 and k<h):

                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    #
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j, k - 1


                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)

                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif(i==0 and j==49 and  k>0 and k<h):
                    # vetrtex_x_up = i + 1, j, k
                    #
                    #
                    # vetrtex_y_down = i, j - 1, k
                    # vetrtex_z_up = i, j, k + 1
                    # vetrtex_z_down = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)

                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)
                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif(i==49 and j>0 and j>49 and k==0):

                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j - 1, k
                    # vetrtex_z_up = i, j, k + 1
                    #


                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)

                elif(i == 0 and j > 0 and j > 49 and k == h):
                    # vetrtex_x_up = i + 1, j, k
                    #
                    # vetrtex_y_up = i, j + 1, k
                    # vetrtex_y_down = i, j - 1, k
                    #
                    # vetrtex_z_down = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)

                    g.add_edge(vetrtex, vetrtex_y_up, y_u)
                    g.add_edge(vetrtex, vetrtex_y_down, y_d)

                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
                elif (i> 0 and i<49 and j ==49 and k == 0):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    #
                    # vetrtex_y_down = i, j - 1, k
                    # vetrtex_z_up = i, j, k + 1


                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)

                    g.add_edge(vetrtex, vetrtex_y_down, y_d)
                    g.add_edge(vetrtex, vetrtex_z_up, z_u)

                elif (i> 0 and i<49 and j == 0 and k == h):
                    # vetrtex_x_up = i + 1, j, k
                    # vetrtex_x_down = i-1, j, k
                    # vetrtex_y_up = i, j + 1, k
                    #
                    #
                    # vetrtex_z_down = i, j, k - 1

                    g.add_edge(vetrtex, vetrtex_x_up, x_u)
                    g.add_edge(vetrtex, vetrtex_x_down, x_d)
                    g.add_edge(vetrtex, vetrtex_y_up, y_u)

                    g.add_edge(vetrtex, vetrtex_z_down, z_d)
      print(g)




      sorce=0
      dest=50*50*h-1
      start = time.time()
      shortest_path, distance = dijkstra(g,sorce,dest)
      end = time.time()
      print("H :", h)
        #print("Time taken:", start, "end", end)
        #print("Total time taken for a path:", end - start)
        # assert shortest_path == [0, 1] and distance == randint(0, 10)
        #print(shortest_path)
        #print(distance)
      print(shortest_path)

      nf = nf.append({"Max_nodes":dest,"From":sorce,"To":dest,"Time":end-start,"Cost":distance,"Hopes":len(shortest_path)-1},ignore_index=True)
      print(nf)
   nf.to_csv("Dijkstre_results.csv")




if __name__ == "__main__":
    main()

