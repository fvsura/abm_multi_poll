
def AttachForBelief(MG, Node, nFriends, Layer, nF, dF, nV, dV, time): #For Node in MG try to add nFriend
    import numpy as np
    import networkx as nx
    import NetworkTool as nt
    temp = []
    for n in MG.nodes(): 
        temp.append(n)
    i = 0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = temp.pop() #choose random node removing from temp list
        if nt.NodeAlike(Node, Candidate, nF, dF, nV, dV) == 1 and nt.CheckEdge(MG,Node,Candidate,Layer) == 0: #check if alike and not connected
            MG.add_edge(Node, Candidate, layer = Layer) #if positive add edge in layer
            Node.setNewLayer(Layer)
            Candidate.setNewLayer(Layer)
            '''to keep track'''
            Node.aggF(time)
            Candidate.aggF(time)
            Node.recNewF(Layer, Candidate.getNum(), time)
            Candidate.recNewF(Layer, Node.getNum(), time)
            i = i+1

def AttachChoosenBelief(MG, Node, nFriends, Layer, listPF, dF, listPV, dV, time): #
    import numpy as np
    import networkx as nx
    import NetworkTool as nt
    temp = []
    for n in MG.nodes(): 
        temp.append(n)
    i=0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = temp.pop() #choose random node removing from temp list
        if nt.CheckEdge(MG, Node, Candidate, Layer) == 0:
            if nt.NodePosAlike(Node, Candidate, listPF, dF, listPV, dV) == 1:
                MG.add_edge(Node, Candidate, layer = Layer) #if positive add edge in layer
                Node.setNewLayer(Layer)
                Candidate.setNewLayer(Layer)
                '''to keep track'''
                Node.aggF(time)
                Candidate.aggF(time)
                Node.recNewF(Layer, Candidate.getNum(), time)
                Candidate.recNewF(Layer, Node.getNum(), time)
                i=i+1
                
def AttachCBRandomLayer(MG, Node, nFriends, nLayer, llPF, listDF, llPV, listDV, time): #listlistPF is a list of list
    import numpy as np
    import networkx as nx
    import NetworkTool as nt
    temp = []
    for n in MG.nodes(): 
        temp.append(n)
    i=0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = temp.pop() #choose random node removing from temp list
        tlay = np.random.randint(nLayer)
        tdF = listDF[tlay] #from the list extract the needed one
        tdV = listDV[tlay]
        listPF = llPF[tlay]
        listPV = llPV[tlay]
        if nt.CheckEdge(MG, Node, Candidate, tlay) == 0:
            if nt.NodePosAlike(Node, Candidate, listPF, tdF, listV, dV) == 1:
                MG.add_edge(Node, Candidate, layer = tlay) #if positive add edge in layer
                Node.setNewLayer(tlay)
                Candidate.setNewLayer(tlay)
                '''to keep track'''
                Node.aggF(time)
                Candidate.aggF(time)
                Node.recNewF(tlay, Candidate.getNum(), time)
                Candidate.recNewF(tlay, Node.getNum(), time)
                i=i+1

def AttachCBAnyLayer(MG, NodeA, nFriends, llPF, listDF, llPV, listDV, time): #listlistPF is a list of list
    import numpy as np
    import networkx as nx
    import NetworkTool as nt
    i=0
    temp = MG.nodes()[:]
    while i < nFriends and len(temp) > 0 : #if still there are nodes and not enough friends
        Candidate = temp.pop() #choose random node removing from temp list
        if len(Candidate.getActiveLayer()) != 0:
            tlay = np.random.choice(Candidate.getActiveLayer()) #layer in wich the node is active
            if nt.CheckEdge(MG, NodeA, Candidate, tlay) == 0:
                tdF = listDF[tlay] #from the list extract the needed one
                tdV = listDV[tlay]
                listPF = llPF[tlay]
                listPV = llPV[tlay]
                if nt.NodePosAlike(NodeA, Candidate, listPF, tdF, listPV, tdV) == 1:
                    MG.add_edge(NodeA, Candidate, layer = tlay) #if positive add edge in layer
                    NodeA.setNewLayer(tlay)
                    Candidate.setNewLayer(tlay)
                    '''to keep track'''
                    NodeA.aggF(time)
                    Candidate.aggF(time)
                    NodeA.recNewF(tlay, Candidate.getNum(), time)
                    Candidate.recNewF(tlay, NodeA.getNum(), time)
                    i=i+1
    del temp
                
def AttachFFAnyLayer(MG, NodeA, nFriends, llPF, listDF, llPV, listDV, time):
    import numpy as np
    import networkx as nx  
    import NetworkTool as nt
    temp = []
    for m in nx.all_neighbors(MG, NodeA):
        for n in nx.all_neighbors(MG, m): #create a list of neighbors of neighbors
            temp.append(n)
    i=0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = temp.pop() #choose random node removing from temp list                
        if len(Candidate.getActiveLayer()) != 0:
            tlay = np.random.choice(Candidate.getActiveLayer())
            if nt.CheckEdge(MG, NodeA, Candidate, tlay) == 0:
                tdF = listDF[tlay] #from the list extract the needed one
                tdV = listDV[tlay]
                listPF = llPF[tlay]
                listPV = llPV[tlay]
                if nt.NodePosAlike(NodeA, Candidate, listPF, tdF, listPV, tdV) == 1:
                    MG.add_edge(NodeA, Candidate, layer = tlay) #if positive add edge in layer
                    NodeA.setNewLayer(tlay)
                    '''to keep track'''
                    NodeA.aggF(time)
                    Candidate.aggF(time)
                    NodeA.recNewF(tlay, Candidate.getNum(), time)
                    Candidate.recNewF(tlay, NodeA.getNum(), time)
                    i=i+1
    if i < nFriends:
        AttachCBAnyLayer(MG, NodeA, nFriends-i, llPF, listDF, llPV, listDV, time)
    del temp
                
def AttachFriendFirst(MG, Node, nFriends, Layer, listPF, dF, listPV, dV, time):
    import numpy as np
    import networkx as nx  
    import NetworkTool as nt
    temp = []
    for m in nx.all_neighbors(MG, Node):
        for n in nx.all_neighbors(MG, m): #create a list of neighbors of neighbors
            temp.append(n)
    i=0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = temp.pop() #choose random node removing from temp list
        if nt.CheckEdge(MG, Node, Candidate, Layer) == 0:
            if nt.NodePosAlike(Node, Candidate, listPF, dF, listPV, dV) == 1:
                MG.add_edge(Node, Candidate, layer = Layer) #if positive add edge in layer
                Node.setNewLayer(Layer)
                Candidate.setNewLayer(Layer)
                '''to keep track'''
                Node.aggF(time)
                Candidate.aggF(time)
                Node.recNewF(Layer, Candidate.getNum(), time)
                Candidate.recNewF(Layer, Node.getNum(), time)
                i=i+1
    #if i>0: print(i) #chek if work
    if i < nFriends:
        AttachChoosenBelief(MG, Node, nFriends-i, Layer, listPF, dF, listPV, dV, time)
    
def  AttachFFRandomLayer(MG, Node, nFriends, nLayer, llPF, listDF, llPV, listDV, time):
    import numpy as np
    import networkx as nx 
    import NetworkTool as nt
    temp = []
    for m in nx.all_neighbors(MG, Node):
        for n in nx.all_neighbors(MG, m): #create a list of neighbors of neighbors
            temp.append(n)
    i=0
    while len(temp) > 0 and i < nFriends: #if still there are nodes and not enough friends
        np.random.shuffle(temp) #shuffle list of friend
        Candidate = temp.pop() #choose random node removing from temp list
        tlay = np.random.randint(nLayer)
        listPF = llPF[tlay]
        listPV = llPV[tlay]
        tdF = listDF[tlay]
        tdV = listDV[tlay]
        if nt.CheckEdge(MG, Node, Candidate, tlay) == 0:
            if nt.NodePosAlike(Node, Candidate, listPF, tdF, lisPV, tdV) == 1:
                MG.add_edge(Node, Candidate, layer = tlay) #if positive add edge in layer
                Node.setNewLayer(tlay)
                Candidate.setNewLayer(tlay)
                '''to keep track'''
                Node.aggF(time)
                Candidate.aggF(time)
                Node.recNewF(tlay, Candidate.getNum(), time)
                Candidate.recNewF(tlay, Node.getNum(), time)
                i=i+1
    #if i>0: print(i) #chek if work
    if i < nFriends:
        tlay = np.random.randint(nLayer)
        listPF = llPF[tlay]
        listPV = llPV[tlay]
        tdF = listDF[tlay]
        tdV = listDV[tlay]
        AttachChoosenBelief(MG, Node, nFriends-i, tlay, listPF, tdF, listPV, dV, time)
        
def RandomLink(MG, Node, nLayer, time):
    import numpy as np
    from NetworkTool import CheckEdge
    flag = 1
    while flag == 1:
        tlay = np.random.randint(nLayer)
        Candidate = np.random.choice(MG.nodes())
        flag = CheckEdge(MG, Node, Candidate, tlay)
    else:
        MG.add_edge(Node, Candidate, layer = tlay)
        Node.setNewLayer(tlay)
        Candidate.setNewLayer(tlay)
        '''to keep track'''
        Node.aggF(time)
        Candidate.aggF(time)
        Node.recNewF(tlay, Candidate.getNum(), time)
        Candidate.recNewF(tlay, Node.getNum(), time)
