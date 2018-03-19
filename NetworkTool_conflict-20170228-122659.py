

def NodeAlike(NodeA, NodeB, nF, pF, nV, pV): #nF fix belief in distance pF, nV var belief in common in distance pV
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
        if abs(NodeA.getOneFixB(i) - NodeB.getOneFixB(i)) <= pF:
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
        if abs(NodeA.getOneVarB(i) - NodeB.getOneVarB(i)) <= pV:
            alike.append(1) 
        else:
            alike.append(0) #if too different add 0 in alike
    '''if at least one parameter too different return 0'''
    if 0 in alike:
        return 0
    else:
        return 1 #if no problem return 1

def NodePosAlike(NodeA, NodeB, posF, pF, posV, pV): #posF and posV are list of position to look at in belief for similarity
    import numpy as np
    alike = []
    '''similarity in fixed belief'''
    if not NodeA.getLenFixB() == NodeB.getLenFixB(): #chek if node have same list dimension
        return -1
    if not NodeA.getLenFixB() == 0: #if list empty skip
        for i in posF:
            if abs(NodeA.getOneFixB(i) - NodeB.getOneFixB(i)) <= pF:
                alike.append(1) 
            else:
                alike.append(0) #if too different add 0 in alike
    '''similarity in variable belief'''
    if not NodeA.getLenVarB() == NodeB.getLenVarB(): #chek if node have same list dimension
        return -1
    if not NodeA.getLenVarB() == 0: #if list empty skip
        for i in posV:
            if abs(NodeA.getOneVarB(i) - NodeB.getOneVarB(i)) <= pF:
                alike.append(1) 
            else:
                alike.append(0) #if too different add 0 in alike   
    '''if at least one parameter too different return 0'''
    if 0 in alike:
        return 0
    else:
        return 1 #if no problem return 1
        
        
def NodeInflu(NodeA, NodeB, nV, pV): #A influence B on nV belief with prob pV
    import numpy as np
    posV = []
    if not NodeA.getLenVarB() == NodeB.getLenVarB():
        return -1
    if nV >= NodeA.getLenVarB():
        return -1
    for i in range(nV):
        posV.append(np.random.randint(0,NodeA.getLenFixB()))
    for i in posV:
        if np.random.random() < pV:
            NodeB.setOneVarB(NodeA.getOneVarB(i))

def AttachForBelief(MG, Node, nFriends, Layer, nF, pF, nV, pV): #For Node in MG try to add nFriend
    import numpy as np
    import networkx as nx
    temp = []
    for n in range(MG.number_of_nodes()): 
        temp.append(n)
    i = 0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = MG.nodes()[temp.pop()] #choose random node removing from temp list
        if NodeAlike(Node, Candidate, nF, pF, nV, pV) == 1 and CheckEdge(MG,Node,Candidate,Layer) == 0: #check if alike and not connected
            MG.add_edge(Node, Candidate, layer = Layer) #if positive add edge in layer
            i = i+1

def AttachPositionVise(MG, Node, nFriends, Layer, posF, pF, posV, pV):
    import numpy as np
    import networkx as nx
    if type(posF)==int:
        tposF=[posF] #create list of single position
    else:
        tposF=posF[:]
    if type(posV)==int:
        tposV=[posV] #create list of single position
    else:
        tposV=posV[:] 
    temp = []
    for n in range(MG.number_of_nodes()): 
        temp.append(n)
    i=0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = MG.nodes()[temp.pop()] #choose random node removing from temp list
        if CheckEdge(MG, Node, Candidate, Layer) == 0:
            if NodePosAlike(Node, Candidate, posF, pF, posV, pV) == 1:
                MG.add_edge(Node, candidate, layer = Layer) #if positive add edge in layer
                i=i+1
    
def CheckEdge(MG,Node,Candidate,Layer):
    flag = 0
    tG = ExtractLayer(MG,Layer)
    if Candidate==Node: #check two nodes aren't the same
        flag = 1
    if (Candidate, Node) in tG.edges(): #if edges don't already in layer
        flag = 1
    if (Node, Candidate) in tG.edges(): #if edges don't already in layer
        flag = 1    
    return flag
       
def ExtractLayer(MG, Layer):
    import networkx as nx
    temp = nx.Graph()
    for i in MG.edges(data='layer'):
        if i[2]==Layer:
            temp.add_edge(i[0],i[1])
    return temp
    
def DrawLayer(MG, Nodes,Pos, Layer):
    import networkx as nx  
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,Layer*3))
    axt=[]
    color = ['r','g','b','c','m','y']
    Label = {}
    for i in range(Nodes):
        Label[MG.nodes()[i]]=i
    mxt = fig.add_subplot(Layer+1,1,1)
    mxt.set_title('complete graph')
    nx.draw(MG,ax=mxt, pos=Pos, labels=Label, node_size = 150)
    for i in range(Layer):
        axt.append(fig.add_subplot(Layer+1,1,i+2))
        axt[i].set_title(str('Layer '+str(i)))
        nx.draw(ExtractLayer(MG,i),ax=axt[i], edge_color=color[i], pos=Pos, labels=Label, node_size = 150)
    return plt.show()
    
    
    
    
    
    
    
    
    
    
