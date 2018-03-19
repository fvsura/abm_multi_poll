

def NodeAlike(NodeA, NodeB, nF, dF, nV, dV): #nF fix belief in distance pF, nV var belief in common in distance pV
    import numpy as np
    alike = []
    posF = []
    posV = []
    '''similarity in fixed belief'''
    if not NodeA.getLenFixB() == NodeB.getLenFixB(): #chek if node have same list dimension
        return -1
    if nF > NodeA.getLenFixB():
        return -1
    if not NodeA.getLenFixB() == 0: #if list empty skip
        k = 0
        while k < nF:
            temp = np.random.randint(0,NodeA.getLenFixB())
            if not temp in posF: #this avoid to draw twice the same position
                posF.append(temp)
                k = k+1
    for i in posF:
        if abs(NodeA.getOneFixB(i) - NodeB.getOneFixB(i)) <= dF:
            alike.append(1) 
        else:
            alike.append(0) #if too different add 0 in alike
    '''similarity in variable belief'''
    if not NodeA.getLenVarB() == NodeB.getLenVarB(): #chek if node have same list dimension
        return -1
    if nV > NodeA.getLenVarB():
        return -1
    if not NodeA.getLenVarB() == 0: #if list empty skip
        k = 0
        while k < nV:
            temp = np.random.randint(0,NodeA.getLenVarB())
            if not temp in posV: #this avoid to draw twice the same position
                posV.append(temp)
                k = k+1
    for i in posV:
        if abs(NodeA.getOneVarB(i) - NodeB.getOneVarB(i)) <= dV:
            alike.append(1) 
        else:
            alike.append(0) #if too different add 0 in alike
    '''if at least one parameter too different return 0'''
    if 0 in alike:
        return 0
    else:
        return 1 #if no problem return 1

def NodePosAlike(NodeA, NodeB, listPF, dF, listPV, dV): #listPF and listPV are list of position to look at in belief for similarity
    import numpy as np
    alike = []
    '''similarity in fixed belief'''
    if not NodeA.getLenFixB() == NodeB.getLenFixB(): #chek if node have same list dimension
        return -1
    if not NodeA.getLenFixB() == 0: #if list empty skip
        for i in listPF:
            if abs(NodeA.getOneFixB(i) - NodeB.getOneFixB(i)) <= dF:
                alike.append(1) 
            else:
                alike.append(0) #if too different add 0 in alike
    '''similarity in variable belief'''
    if not NodeA.getLenVarB() == NodeB.getLenVarB(): #chek if node have same list dimension
        return -1
    if not NodeA.getLenVarB() == 0: #if list empty skip
        for i in listPV:
            if abs(NodeA.getOneVarB(i) - NodeB.getOneVarB(i)) <= dV:
                alike.append(1) 
            else:
                alike.append(0) #if too different add 0 in alike   
    '''if at least one parameter too different return 0'''
    #print(alike) option for verbose
    if 0 in alike:
        return 0
    else:
        return 1 #if no problem return 1
        
        
def NodeInfluRand(NodeA, NodeB, nV, pV): #A influence B on nV belief with prob pV
    import numpy as np
    posV = []
    if not NodeA.getLenVarB() == NodeB.getLenVarB():
        return -1
    if nV >= NodeA.getLenVarB():
        return -1
    for i in range(nV):
        posV.append(np.random.randint(0,NodeA.getLenFixB()))
    for i in posV:
        if np.random.random() < pV: #that's a prob
            NodeB.setOneVarB(NodeA.getOneVarB(i))

def NodeBoundOne(NodeA, NodeB, bpos, pV, dV, rate, time, layer, r=0):#node, node, pos of belief to consider, random prob, distance of belief, change rate
    import numpy as np
    if not NodeA.getLenVarB() == NodeB.getLenVarB():
        return -1
    pos = np.random.choice(bpos) #shuffle list of belief
    if abs(NodeA.getOneVarB(pos)-NodeB.getOneVarB(pos)) < dV: #check distance between beliefs
        if np.random.random()<pV: #check random
            NodeB.setOneVarB(pos, NodeB.getOneVarB(pos)+rate*(NodeA.getOneVarB(pos)-NodeB.getOneVarB(pos))) #correct value of belief
            '''to keep track'''
            if r == 1:
                NodeB.recNewI(NodeA.getNum(), time, pos, NodeA.getOneVarB(pos), layer)
    del pos

def NodeBoundTwins(NodeA, NodeB, bpos, pV, dV, rate, time, layer, r=0):#node, node, pos of belief to consider, random prob, distance of belief, change rate, this simply process both nodes
    import numpy as np
    if not NodeA.getLenVarB() == NodeB.getLenVarB():
        return -1
    pos = np.random.choice(bpos) #shuffle list of belief
    if abs(NodeA.getOneVarB(pos)-NodeB.getOneVarB(pos)) < dV: #check distance between beliefs
        if np.random.random()<pV: #check random
            NodeB.setOneVarB(pos, NodeB.getOneVarB(pos)+rate*(NodeA.getOneVarB(pos)-NodeB.getOneVarB(pos))) #correct value of belief
            NodeA.setOneVarB(pos, NodeA.getOneVarB(pos)+rate*(NodeB.getOneVarB(pos)-NodeA.getOneVarB(pos)))
            '''to keep track'''
            if r == 1:
                NodeB.recNewI(NodeA.getNum(), time, pos, NodeA.getOneVarB(pos), layer)
                NodeA.recNewI(NodeA.getNum(), time, pos, NodeB.getOneVarB(pos), layer)
    del pos
            
def NodeBoundDrift(NodeA, NodeB, bpos, pV, dV, rate, time, layer, r=0): #experimental to see different way to converge
    import numpy as np
    if not NodeA.getLenVarB() == NodeB.getLenVarB():
        return -1
    pos = np.random.choice(bpos) #shuffle list of belief
    if abs(NodeA.getOneVarB(pos)-NodeB.getOneVarB(pos)) < dV: #check distance between beliefs
        if np.random.random()<pV: #check random
            NodeB.setOneVarB(pos, NodeB.getOneVarB(pos)+rate*(NodeA.getOneVarB(pos)-NodeB.getOneVarB(pos))) #correct value of belief
            '''to keep track'''
            if r == 1:
                NodeB.recNewI(NodeA.getNum(), time, pos, NodeA.getOneVarB(pos), layer)
    else:
        if np.random.random()<0.01*pV: #check random
            NodeB.setOneVarB(pos, NodeB.getOneVarB(pos)-rate*(NodeA.getOneVarB(pos)+NodeB.getOneVarB(pos))) #correct value of belief
            '''to keep track'''
            if r == 1:
                NodeB.recNewI(NodeA.getNum(), time, pos, -1*NodeA.getOneVarB(pos), layer)
    del pos

def EvolveInfluR(MG, iteration, cbpos, pV, dV, TheTime, nLayer): #evolve and influence the beliefs
    import numpy as np
    print('Evolving for ',iteration)
    G=[]
    TheTime = TheTime*100
    for lay in range(nLayer):
        G.append(ExtractLayer(MG, lay))
    for kk in range(iteration):
        if kk - 500*(kk//500) == 0 :
            #print(kk, TheTime)
            TheTime = TheTime+1
        for node in MG.nodes():
            try:
                lay = np.random.choice(node.getActiveLayer()) #choose a layer in wich the node is active
                g = G[lay]
                candidate = np.random.choice(g.neighbors(node)) #choose a random nighbor
                NodeBoundOne(candidate, node, cbpos[lay], pV, dV, node.getGull(), TheTime, lay)
                NodeBoundOne(node, candidate, cbpos[lay], pV, dV, candidate.getGull(), TheTime, lay)
            except:
                pass
    del G 
            
def CheckEdge(MG,Node,Candidate,Layer):
    flag = 0
    tG = ExtractLayer(MG,Layer)
    if Candidate==Node: #check two nodes aren't the same
        flag = 1
    if (Candidate, Node) in tG.edges(): #if edges don't already in layer
        flag = 1
    if (Node, Candidate) in tG.edges(): #if edges don't already in layer
        flag = 1   
    tG.clear()
    return flag
       
def ExtractLayer(MG, Layer): #return a single layer
    import networkx as nx
    temp = nx.Graph()
    for i in MG.edges(data='layer'):
        if i[2]==Layer:
            temp.add_edge(i[0],i[1])
    return temp
    del temp
    
def DrawLayer(MG, Nodes, Pos, nLayer): #draw representation of each layer
    import networkx as nx  
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,nLayer*3))
    axt=[]
    color = ['r','g','b','c','m','y', 'r','g','b','c','m','y', 'r','g','b','c','m','y']
    Label = {}
    for i in range(Nodes):
        Label[MG.nodes()[i]]=i
    mxt = fig.add_subplot(nLayer+1,1,1)
    mxt.set_title('complete graph')
    nx.draw(MG,ax=mxt, pos=Pos, labels=Label, node_size = 100)
    for i in range(nLayer):
        axt.append(fig.add_subplot(nLayer+1,1,i+2))
        axt[i].set_title(str('Layer '+str(i)))
        nx.draw(ExtractLayer(MG,i),ax=axt[i], edge_color=color[i], pos=Pos, labels=Label, node_size = 150)
    return plt.show()

def DrawLayerVote(MG, Nodes, Pos, nLayer, issue):
    import networkx as nx
    import matplotlib.pyplot as plt
    col = []
    for i in MG.nodes():
        col.append(i.getVote())
    fig = plt.figure(figsize=(15,nLayer*3))
    axt=[]
    color = ['r','g','b','c','m','y', 'r','g','b','c','m','y', 'r','g','b','c','m','y']
    Label = {}
    for i in range(Nodes):
        Label[MG.nodes()[i]]=i
    mxt = fig.add_subplot(nLayer+1,1,1)
    mxt.set_title('complete graph')
    nx.draw(MG,ax=mxt, pos=Pos, labels=Label, node_size = 100, node_color = col, cmap = 'bwr')
    for i in range(nLayer):
        col=[]
        G=ExtractLayer(MG,i)
        for j in G.nodes():
            col.append(j.getVote())
        axt.append(fig.add_subplot(nLayer+1,1,i+2))
        axt[i].set_title(str('Layer '+str(i)))
        nx.draw(G,ax=axt[i], edge_color=color[i], pos=Pos, labels=Label, node_size = 150, node_color=col, cmap='bwr')
    return plt.show()  
    
def PlotLayer(MG,nLayer): #plot degree distribution for each layer
    import networkx as nx  
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from sklearn.metrics import mean_squared_error
    from math import log
    fig = plt.figure(figsize=(10,nLayer*6.5))
    color = ['r','g','b','c','m','y', 'r','g','b','c','m','y', 'r','g','b','c','m','y']
    degree_sequence=sorted(nx.degree(MG).values(),reverse=True)
    log_degree_sequence=[]
    for i in degree_sequence:
        log_degree_sequence.append(float(i))        
    ax = fig.add_subplot(nLayer+1,1,1)
    ax.set_title("Degree rank Complete Graph")
    x=[]
    for i in range(len(degree_sequence)):
        x.append(float(i+1))
    fit = np.polyfit(x, log_degree_sequence,1)
    fit_fn = np.poly1d(fit)
    print ("Complete Graph; entry:",len(x), "; mean squared error to log-fit:", mean_squared_error(fit_fn(x), degree_sequence))
    plt.loglog(degree_sequence,'b-',marker='o')
    ax.set_ylabel("degree")
    ax.set_xlabel("rank")
    for i in range(nLayer):
        degree_sequence=sorted(nx.degree(ExtractLayer(MG,i)).values(),reverse=True)
        log_degree_sequence=[]
        for i in degree_sequence:
            log_degree_sequence.append(float(i))
        kurt = stats.kurtosis(degree_sequence)
        ax=fig.add_subplot(nLayer+1,1,i+2)
        ax.set_title(str('Layer '+str(i)+', Kurtosis: '+str(round(kurt,2))))
        plt.loglog(degree_sequence,color[i],marker='o')
        x=[]
        for i in range(len(degree_sequence)):
            x.append(float(i+1))
        fit = np.polyfit(x,log_degree_sequence,1)
        fit_fn = np.poly1d(fit)
        print ("Layer:", i,"; entry:",len(x), "; mean squared error to log-fit:", mean_squared_error(fit_fn(x), degree_sequence))
        
        plt.ylabel("degree")
        plt.xlabel("rank")
       
    return fig.show()

def BaseAnalisys(MG,nLayer):
    import networkx as nx
    import numpy as np
    output=[]
    print('complete assortativity ',nx.degree_assortativity_coefficient(MG))
    output.append([MG.number_of_nodes(),nx.degree_assortativity_coefficient(MG),np.average(list(nx.degree(MG).values()))])
    print()
    for i in range(nLayer):
        G=ExtractLayer(MG,i)
        print('Layer',i)
        print('assortativity ',nx.degree_assortativity_coefficient(G))
        print('clustering ',nx.average_clustering(G))
        print('averagedegree ',np.average(list(nx.degree(G).values())))
        print('nodes ', G.number_of_nodes())
        print()
        output.append([i,G.number_of_nodes(),nx.degree_assortativity_coefficient(G),nx.average_clustering(G),np.average(list(nx.degree(G).values()))])
    return(output)
        
def LayerAnalisys(MG, Layer):
    import networkx as nx
    import numpy as np
    temp = []
    temp.append(Layer)
    if ExtractLayer(MG,Layer).number_of_edges() != 0: #trying to avoid error for empty Layer
        temp.append(nx.degree_pearson_correlation_coefficient(ExtractLayer(MG,Layer)))
                #temp.append(nx.degree_assortativity_coefficient(ExtractLayer(MG,Layer)))
        temp.append(nx.average_clustering(ExtractLayer(MG,Layer)))
        temp.append(np.average(list(nx.degree(ExtractLayer(MG,Layer)).values())))
    else:
        temp.append(float('nan'))
        temp.append(float('nan'))
        temp.append(float('nan'))
        print('nan error! layer: ', Layer)
    return temp

def NetAnalisys(MG):
    import networkx as nx
    import numpy as np
    temp = []
    temp.append(nx.degree_assortativity_coefficient(MG))
    temp.append(np.average(list(nx.degree(MG).values())))
    return temp
        
def ShowDoubleEdges(MG):
    import networkx as nx
    for i in MG.edges(data='layer'):
        for j in range(Layer):
            if (i[0],i[1]) in ExtractLayer(MG,j).edges() or (i[1],i[0]) in ExtractLayer(MG,j).edges():
                if i[2]!=j:
                    print('layer',i[2],'and ',j,' edge ',i)
            
def GullDistr(MG, nLayer):
    #analysis of gullible distribution, should be uniform
    import matplotlib.plot as plt
    avg=[]
    for j in range(nLayer):
        gul=[]
        for i in ExtractLayer(MG, j).nodes():
            gul.append(i.getGull())
        avg.append(np.mean(gul))
        plt.hist(gul, 10) 
    print('Average: ', np.mean(avg))
    return plt.show()
    
def DegCorrAll(MG, bpos, nLayer): #calculate correlation of degree centrality for nodes in different layer
    import numpy as np
    import networkx as nx
    val=[]
    val.append(nx.degree_centrality(MG))
    for i in range(nLayer):
        G=ExtractLayer(MG, i)
        val.append(nx.degree_centrality(G))  
    print('Number of layers: ', len(val))
    matr=[]
    matr.append([])
    for n in MG.nodes():
        matr[0].append(val[0][n])
    for l in range(nLayer):
        matr.append([])
        l=l+1
        for n in MG.nodes():
            if n in val[l]:
                matr[l].append(val[l][n])
            else:
                matr[l].append(0)
    #print(shape(matr))        
    pears=[]
    print('Here a list of couples of layers with positive Pearson coefficient regarding degree centrality')
    for i in range(len(val)):
        for j in range(len(val)):
            if i!=0 and j!=0 and i!=j: #calc the pearson coeff of deg_centr only between different layers
                pears.append(np.corrcoef(matr[i],matr[j])[0,1])
                if (np.corrcoef(matr[i],matr[j])[0,1])>0:
                    print("check:",i,":",j," bpos",bpos[i],":",bpos[j]," coeff", np.corrcoef(matr[i],matr[j])[0,1])
    print()
    counter=sum(1 if x>0 else 0 for x in pears) #how many pos coeff
    print('Number of Pearson coefficient calculated: ',len(pears))
    print('Number of positive coefficient: ', counter)

    
''' NOT WORKING
def DegCorrActive(MG, nLayer): # same as DegCorrAll but only nodes active in both layers
    imprt numpy as np
    val=[]
    val.append(nx.degree_centrality(MG))
    for i in range(nLayer):
        G=ExtractLayer(MG2, i)
        val.append(nx.degree_centrality(G))  
    print("lenval:", len(val))

    G0=ExtractLayer(MG2, lay[0])
    G1=ExtractLayer(MG2, lay[1])
    commonN=[]

    for n in MG2.nodes():
        if n in G0.nodes() and n in G1.nodes():
            commonN.append(n)

    matrN=[]
    matrN.append([])
    for n in commonN:
        matrN[0].append(val[0][n])
    for l in range(Layer):
        matrN.append([])
        l=l+1
        for n in commonN:
            if n in val[l]:
                matrN[l].append(val[l][n])
            else:
                matrN[l].append(0)

    #matr=np.transpose(matr)
    print("shapematr:", shape(matrN))        

    plt.plot(matrN[lay[0]],matrN[lay[1]],'b1')
    corr=np.corrcoef(matrN)
    print("corrcoeff:", np.corrcoef(matrN[lay[0]],matrN[lay[1]])[0,1])

    print()
    pears=[]
    for i in range(len(val)):
        for j in range(len(val)):
            if i!=0 and j!=0 and i!=j: #calc the pearson coeff of deg_centr only between different layers
                pears.append(np.corrcoef(matrN[i],matrN[j])[0,1])
                if (np.corrcoef(matrN[i],matrN[j])[0,1])>0:
                    print("check:",i,":",j,", bpos",bpos[i],":",bpos[j],", coeff", np.corrcoef(matrN[i],matrN[j])[0,1]," was",np.corrcoef(matr[i],matr[j])[0,1] )

    print()
    counter=sum(1 if x>0 else 0 for x in pears) #how many pos coeff
    print(len(pears))
    print(counter)
'''

def GephiSearch(MG, Label):
    import matplotlib.pyplot as plt
    for nn in MG.nodes():
        if MG.node[nn]['Label']==Label:
            print(MG.node[nn])
            zz=nn
            print("found")
    print('Active layers: ', zz.getActiveLayer())
    print('Total neighbors: ', len(MG.neighbors(zz)))
    print('Neighbors per layer:')
    for i in zz.getActiveLayer():
        print('Neigh. ',len(ExtractLayer(MG, i).neighbors(zz)), 'layer ',i)      
    print('Vote: ', zz.getVote(), ' Gullible: ',zz.getGull()) 
    print('Layer in wich was influenced: ')
    ix=[]
    for i in zz.InfluHist:
        for j in zz.InfluHist[i]:
            ix.append(j)
    plt.hist(ix, 7)
    print(ix)
    print('List of influence: ', zz.InfluHist)
    return plt.show()
    
        
  
    