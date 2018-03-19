
def AllVoteA(MG, issue):
    for nn in MG.nodes():
            nn.doVoteA(issue)
            
def SocialDesirability(NodeA, MG, md): #if too distant from neighbors lie
    import numpy as np
    vv=[]
    for nn in MG.neighbors(NodeA):
        vv.append(nn.getVote())
    if np.mean(vv)*NodeA.getVote()<0 and abs(np.mean(vv)*NodeA.getVote())>md:
        NodeA.setVote(np.mean(vv))
    del vv
            
def ConformUncertVote(NodeA, MG, bd, issue): #conform to neighbors if uncertain
    import numpy as np
    if abs(NodeA.getVote()) < bd:
        vv=[]
        for nn in MG.neighbors(NodeA):
            vv.append(nn.getVote())
        NodeA.setVote(np.mean(vv))
        del vv
        
def DifformUncertVote(NodeA, MG, bd, issue): #difform to neighbors if uncertain
    import numpy as np
    if abs(NodeA.getVote()) < bd:
        vv=[]
        for nn in MG.neighbors(NodeA):
            vv.append(nn.getVote())
        NodeA.setVote(-1*np.mean(vv))
        del vv

def BelVarDistance(NodeA, MG):
    import numpy as np
    vv = []
    for nn in MG.neighbors(NodeA):#chose the neighbors of nn only in net MG
        vv.append(np.linalg.norm(np.array(NodeA.getVarB())-np.array(nn.getVarB())))
    return np.mean(vv) 
    
def BelFixDistance(NodeA, MG):
    import numpy as np
    vv = []
    for nn in MG.neighbors(NodeA):
        vv.append(np.linalg.norm(np.array(nn.getFixB())-np.array(NodeA.getFixB())))
    return np.mean(vv)

def PollDesirability(MG, listLayer,listCall, listMd, issue, bd):
    G=[]
    iteration = len(listLayer)
    for tlay in listLayer:
        G.append(ExtractLayer(MG,tlay))
    Interviews=[]
    for i in iteration:
        for j in range(listCall[i]):
            Interviews[i].append(np.random.choice(G[i].nodes()))
        for n in Interviews[i]:
            SocialDesirability(n, G[i], listMd[i])
    for i in iteration:
        VoteAnalysis(G[i], issue, bd, 1)

def FixDistPrnt(MG, plot):
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    vv = []
    for nn in MG.nodes():
        vv.append(BelFixDistance(nn, MG))
    kur=round(stats.kurtosis(vv),2)
    skew=round(stats.skew(vv),2)        
    if plot == 1:
        print('avg dist ', np.mean(vv))
        plt.xlabel('Distance')
        plt.ylabel('# of nodes')
        print('kurtosis',kur)
        print('skewness',skew)
        plt.title('kurtosis: '+str(kur)+', skewness: '+str(skew))
        plt.hist(vv, bins=100)
        plt.show()
    return([np.mean(vv), kur, skew])
        
def VarDistPrnt(MG, plot):
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    vv = []
    for nn in MG.nodes():
        vv.append(BelVarDistance(nn, MG))
    kur=round(stats.kurtosis(vv),4)
    skew=round(stats.skew(vv),4)
    if plot == 1:
        print('avg dist ', np.mean(vv))
        plt.xlabel('Distance')
        plt.ylabel('# of nodes')
        print('kurtosis',kur)
        print('skewness',skew)
        plt.title('kurtosis: '+str(kur)+', skewness: '+str(skew))
        plt.hist(vv, bins=100)
        plt.show()
    return([np.mean(vv), kur, skew])
        
def VoteAnalysis(MG, issue, bd, do):
    import matplotlib.pyplot as plt
    y=0
    n=0
    a=0
    x=[]
    print(issue)#,' ',len(issue))
    if do == 1:
        for NN in MG.nodes():
            NN.doVoteA(issue)
            x.append(NN.getVote())                           #watch type of vote!
    else:
        for NN in MG.nodes():
            x.append(NN.getVote())
    for i in x:
        if i>bd:
            y=y+1
        if i<(-1*bd):
            n=n+1
        if i<bd and i>(-1*bd):
            a=a+1
    aa=100*a/len(x)
    yy=100*y/len(x)
    nn=100*n/len(x)
    print('astenuti: ',a,' ',aa,'%, favorevoli: ',y,' ',yy,'%, contrari: ',n,' ',nn,'%')
    print()
    if y>n:
        print('Passato')
    else:
        print('Respinto')
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1)
    labels = 'Favorevoli', 'Contrari', 'Astenuti'
    sizes = [y, n, a]
    explode = (0.1, 0.1, 0.1)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(x,bins=60)
    ax2.plot((bd, bd), (0, 20), 'r-')
    ax2.plot((-1*bd, -1*bd), (0, 20), 'r-')
    fig.savefig('VoteAnalysis.png',orientation='landscape',bbox_inches='tight',dpi='figure')
    plt.show()
    return [y,n,a]
    
def VoteAnalysisLayer(MG, issue, bd, nLayer): 
    from NetworkTool import ExtractLayer
    for layer in range(nLayer):   
        G = ExtractLayer(MG, layer)
        print('Analisys of Layer: ', layer)             
        VoteAnalysis(G, issue, bd)
        print('------------------------------------------------------------')        
        
        
def VoteALayerLight(MG, issue, bd, do, nLayer):
    import matplotlib.pyplot as plt
    from NetworkTool import ExtractLayer
    print(issue)#,' ',len(issue))
    for layer in range(nLayer):
        G = ExtractLayer(MG, layer)
        print(G.number_of_nodes())
        x=[]
        y=0
        n=0
        a=0
        if do == 1:
            for NN in MG.nodes():
                NN.doVoteA(issue)                          #watch type of vote!
        for NN in G.nodes():
            x.append(NN.getVote())
        for i in x:
            if i>bd:
                y=y+1
            if i<(-1*bd):
                n=n+1
            if i<bd and i>(-1*bd):
                a=a+1    
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(1,2,1)
        labels = 'Favorevoli', 'Contrari', 'Astenuti'
        sizes = [y, n, a]
        explode = (0.1, 0.1, 0.1)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        ax2 = fig.add_subplot(1,2,2)
        ax2.hist(x,bins=60,)
        ax2.plot((bd, bd), (0, 20), 'r-')
        ax2.plot((-1*bd, -1*bd), (0, 20), 'r-')
        ax1.set_title('layer: '+str(layer))
        ax2.set_title('layer: '+str(layer))
    #fig.savefig('LayerVoteA.png',orientation='landscape',bbox_inches='tight',dpi='figure')
    plt.show()       
        
def VoteDist(MG, plot):
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    vv = []
    for nn in MG.nodes():
        vv.append(nn.getVote())
    kur=round(stats.kurtosis(vv),2)
    skew=round(stats.skew(vv),2)
    if plot == 1:
        plt.xlabel('Vote')
        plt.ylabel('# of nodes')
        print('kurtosis',kur)
        print('skewness',skew)
        plt.title('kurtosis: '+str(kur)+', skewness: '+str(skew))
        plt.hist(vv, bins=60)
        plt.show()
    return([np.mean(vv), kur, skew])       
        
    
    
def PollLayers(MG, Call1,Call2, bd, issue, lay1,lay2, socDes):
    import numpy as np
    import matplotlib.pyplot as plt
    from NetworkTool import ExtractLayer
    G1=ExtractLayer(MG,lay1)
    G2=ExtractLayer(MG,lay2)
    Interviews1=[]
    Interviews2=[]
    vote1=[]
    vote2=[]
    print(G1.number_of_nodes())
    print(G2.number_of_nodes())
    VoteAnalysis(G1, issue, bd, 1)
    VoteAnalysis(G2, issue, bd, 1)
    for i in range(Call1):
        Interviews1.append(np.random.choice(G1.nodes()))
    for i in range(Call2):
        Interviews2.append(np.random.choice(G2.nodes()))
    if socDes!= 0:
        for n in Interviews1:
            SocialDesirability(n, G1, socDes)
            vote1.append(n.getVote())
        for n in Interviews2:
            SocialDesirability(n, G2, socDes)
            vote2.append(n.getVote())
    y1,n1,a1,y2,n2,a2 = 0,0,0,0,0,0 
    for i in vote1:
        if i>bd:
            y1=y1+1
        if i<(-1*bd):
            n1=n1+1
        if i<bd and i>(-1*bd):
            a1=a1+1
    for i in vote2:
        if i>bd:
            y2=y2+1
        if i<(-1*bd):
            n2=n2+1
        if i<bd and i>(-1*bd):
            a2=a2+1
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.hist(vote1, 60)
    ax1.plot((bd, bd), (0, 10), 'r-')
    ax1.plot((-1*bd, -1*bd), (0, 10), 'r-')
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(vote2, 60) 
    ax2.plot((bd, bd), (0, 10), 'r-')
    ax2.plot((-1*bd, -1*bd), (0, 10), 'r-')
    ax1.set_title('layer: '+str(lay1)+'; no: '+str(n1)+'; yes: '+str(y1))
    ax2.set_title('layer: '+str(lay2)+'; no: '+str(n2)+'; yes: '+str(y2))
    print('avg1: ', np.mean(vote1),'; std: ', np.std(vote1),'; avg2: ', np.mean(vote2),'; std: ', np.std(vote2))
    plt.show()
    return([n1,y1,a1,n2,y2,a2])

def ChiFreq(MG, bins, layer): #difference in vote distribution as chisq
    from NetworkTool import ExtractLayer
    G = ExtractLayer(MG, layer)
    x = []
    y = []
    xHist = []
    yHist = []
    for nn in MG.nodes():
        x.append(nn.getVote())
    for nn in G.nodes():
        y.append(nn.getVote())
    if max(x) > max(y):
        Max=max(x)
    else:
        Max=max(y)
    if min(x) < min(y):
        Min=min(x)
    else:
        Min=min(y)
    Span = Max - Min
    Size = Span/bins
    #print('span',Span,'size',Size,'bins',bins,'min-max', Min,Max)#-------------------
    for i in range(bins+1):
        xHist.append(0)
    for i in x:
        pos = int((i-Min)//Size)
        if i-Min >= Size*pos and i-Min <= Size*(pos+1):
            xHist[pos]=xHist[pos]+1
        else:
            print('error x ',i)#-------------------
    for i in range(bins+1):
        yHist.append(0)
    for i in y:
        pos = int((i-Min)//Size)
        if i-Min >= Size*pos and i-Min <= Size*(pos+1):
            yHist[pos]=yHist[pos]+1
        else:
            print('error y ',i)#------------------- 
    for i in range(bins+1):
        xHist[i]=xHist[i]
        yHist[i]=yHist[i]
    Chi=0
    Df=0
    for b in range(bins+1):
        if xHist[b] != 0 or yHist[b] != 0:
            Chi=Chi+(((xHist[b]-yHist[b])**2)/(xHist[b]+yHist[b]))#numerical recipes in c 0-521-43108-5 pag 622
        else:
            Df=Df+1
    print(Df)
    return([Chi,bins-Df, xHist,yHist, MG.number_of_nodes(), G.number_of_nodes()])
            
            
            
            
            
            
            
            
            
            
            
            