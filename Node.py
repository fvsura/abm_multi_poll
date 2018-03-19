
class Node:
    
    def __init__(self, num, FixBel, VarBel, Gull, time):
        
        self.FixBel = FixBel[:]
        self.nFixBel = len(self.FixBel)
        self.VarBel = VarBel[:] #un solo variabile(es voto) o più?
        self.nVarBel = len(self.VarBel)
        self.num = num
        self.born = time
        self.activeLayer=[]
        self.Vote = 0
        self.Gullible = Gull #[:] #first term distance, second term influence
        
        '''these two just to track evolution'''
        self.FriendHist = {}
        self.InfluHist = {}
    
    '''return value belief'''
    def getOneVarB(self, pBel): #get value in pBel position
        return self.VarBel[pBel]
    def getOneFixB(self, pBel):
        return self.FixBel[pBel]
    def getMoreVarB(self, n): #n is a list of position, return belief in position
        if np.sort(n)[-1] >= self.nVarBel:
            return -1
        temp = []
        for i in n:
            temp.append(self.VarBel[i])
        return temp[:]
    def getMoreFixB(self,n):
        if np.sort(n)[-1] >= self.nFixBel:
            return -1
        temp = []
        for i in n:
            temp.append(self.FixBel[i])
        return temp[:]
    
    '''various getter'''''
    def getFixB(self): #various getter to look in to the agent
        return self.FixBel[:]
    def getLenFixB(self):
        return self.nFixBel
    def getVarB(self):
        return self.VarBel[:]
    def getLenVarB(self):
        return self.nVarBel
    def getNum(self):
        return self.num
    def getBorn(self):
        return self.born
    def getOld(self, time):
        return time-self.born
    def getFriH(self):
        return dict(self.FriendHist)
    def getInfH(self):
        return dict(self.InfluHist)
    def getActiveLayer(self):
        return self.activeLayer[:]
    def getGull(self):
        return self.Gullible #[:]
    def getVote(self):
        return self.Vote
    
    '''voting algorithm''' #essentially a dot product normalized

    def doVoteC(self, issue):
        if len(issue)==(self.nFixBel+self.nVarBel):
            self.Vote = ((self.getFixVote(issue[:self.nFixBel])+self.getVarVote(issue[self.nVarBel:]))/2)
            #return(sum([(i*j)**(0.5) for (i, j) in zip(issue, self.FixBel+self.VarBel)])/(self.nFixBel+self.nVarBel))
        else:
            return(-1) 
    def doVoteA(self, issue):
        import numpy as np
        if len(issue)==(self.nFixBel+self.nVarBel):
            tT=list((self.VarBel+self.FixBel)/(np.linalg.norm(self.VarBel+self.FixBel)))
            tI=issue/np.linalg.norm(issue)
            self.Vote = np.inner(tT,tI)
        else:
            return(-1)
    def doVoteB(self):
        import numpy as np
        self.Vote = np.mean(self.VarBel+self.FixBel)
    def doVarVote(self,issue):
        if len(issue)==(self.nVarBel):
            self.Vote = (sum([(i*j)**(0.5) for (i, j) in zip(issue, self.VarBel)])/self.nVarBel)
        else:
            return(-1)
    def doVarVoteA(self,issue):
        import numpy as np
        if len(issue)==(self.nVarBel):
            tI=issue/np.linalg.norm(issue)
            tV=self.VarBel/np.linalg.norm(self.VarBel)
            self.Vote = np.inner(tV,tI)
        else:
            return -1
    def doVarVoteB(self):
        import numpy as np
        self.Vote = np.mean(self.VarBel)
    def doFixVote(self,issue):
        if len(issue)==self.nFixBel:
            self.Vote = (sum([(i*j)**(0.5) for (i, j) in zip(issue, self.FixBel)])/self.nFixBel)
        else:
            return(-1)
    def doFixVoteA(self,issue):
        import numpy as np
        if len(issue)==(self.nFixBel):
            tI=issue/np.linalg.norm(issue)
            tV=self.FixBel/np.linalg.norm(self.FixBel)
            self.Vote = np.inner(tV,tI)
        else:
            return(-1)
    def doFixVoteB(self):
        import numpy as np
        self.Vote = np.mean(self.FixBel)
                   
    '''change the val in pos pBel of variable belief'''
    def setOneVarB(self, pBel, vBel):
        self.VarBel[pBel] = vBel
    '''setter to change variable and fixed belief, not to be used'''
    def setFixB(self, FixB):
        self.FixBel = FixB[:]
        self.nFixBel = len(self.FixBel)
    def setVarB(self, VarB):
        self.VarBel = VarB[:]
        self.nVarBel = len(self.VarBel)
        
    def setNewLayer(self,layer):
        if layer not in self.activeLayer:
            self.activeLayer.append(layer)
    def setVote(self, extvote):
        self.Vote = extvote
    def setGull(self, Gull):
        self.Gullible = Gull #[:]

    ''' from here on is just to track evolution of the node'''''        
    def getFriStep(self, time): #get the dict of friend at a certain time
        if time in self.FriendHist:
            return dict(self.FriendHist[time])
        else:
            return -1
    def getInfStep(self, time): #get the dict of influence at a certain time
        if time in self.InfluHist:
            return dict(self.InfluHist[time])
        else:
            return -1
        
    def recNewF(self, layer, nFri, time): #keep track of a new friend added
        if not time in self.FriendHist: #it's a dict
            self.aggF(time) #copy the dict
        if not layer in self.FriendHist[time]:
            self.FriendHist[time][layer] = [] #it's a list
        self.FriendHist[time][layer].append(nFri)
      
    def aggF(self, time): #keep list of friend up to date with clock but without changes
        if not time-1 in self.FriendHist:
            self.FriendHist[time-1] = {}
        self.FriendHist[time] = dict(self.FriendHist[time-1])
            
    def delOldF(self, layer, nFri, time): #remove a friend
        if time in self.FriendHist:
            if not layer in self.FriendHist[time]:
                return -1 #if layer not present raise error
            if not nFri in self.FriendHist[time][layer]:
                return -1 #if nFri not present rise error
            self.FriendHist[time][layer].remove(nFri) #else remove selected from list
        else:
            if time-1 in self.FriendHist:
                self.FriendHist[time] = dict(self.FriendHist[time-1]) # tries to rebuilt actual FriendHist[time] dict
                if not layer in self.FriendHist[time]:
                    return -1 #if layer not present raise error
                if not nFri in self.FriendHist[time][layer]:
                    return -1 # if nFri not present, rise error
                self.FriendHist[time][layer].remove(nFri)
            else:
                return -1
        
    def recNewI(self, nFri, time, pBel, vBel, layer): #vBel it's a value in pos pBel in vector VarBel
        if not time in self.InfluHist: #InfluHist is a dict of dict
            self.InfluHist[time] = {}
        if not layer in self.InfluHist[time]:
            self.InfluHist[time][layer] = []
        self.InfluHist[time][layer].append([pBel,vBel, nFri]) #InfluHist[time][nFri] is a list


