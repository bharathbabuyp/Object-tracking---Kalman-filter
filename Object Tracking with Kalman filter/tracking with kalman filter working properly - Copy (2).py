import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import linear_sum_assignment
from pandas import DataFrame, read_csv
import pandas as pd
import time
from stream_thread_pi import PiVideoStream
from mygpio import blink
colours=[(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),(255,0,255),(123,54,222),(89,20,190)]
idd=1

def nextpredictstateusingkalman(X_K_1,   P_K_1,   Y_K_M,   R):
    t=1
    
#    A=np.array([[1,1],[0,1]])
    A=np.array([[1,0,t,0],[0,1,0,t],[0,0,1,0],[0,0,0,1]])
    
#    B=np.array([[.5*1],[1]])
    B=np.array([[.5*t**2,0],[0,.5*t**2],[t,0],[0,t]])
    
    
#    C=np.array([[1,0],[0,1]])
    C=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    
    ax=0
    ay=0
#    U_K=np.array([[ax]])    
    U_K=np.array([[ax],[ay]])
    
#    W_K=np.zeros_like(X_K_1)
#    Q_K=np.zeros_like(P_K_1)
    Z_M=np.zeros_like(Y_K_M)    
    
    W_K=np.random.random(X_K_1.shape)*.001
    Q_K=np.random.random(P_K_1.shape)*.00010
#    Z_M=np.random.random(Y_K_M.shape)*.01
#    Y_K_M=np.array([observation[count]]).transpose()
    X_K_P = A.dot( X_K_1) + B.dot(U_K) + (W_K) ############# 1
    P_K_P= A.dot(P_K_1).dot(A.transpose()) + (Q_K) ########### 2
    eye=np.identity(P_K_P.shape[0])
    P_K_P=P_K_P*eye
    #K=kalmangain(A,P_K_P,R)     ################### 4      
    K=P_K_P / (P_K_P + R)
    K[np.isnan(K)]=0        
    Y_K=C.dot(Y_K_M)+Z_M ################################### 5    
    X_K = X_K_P + K.dot(Y_K-X_K_P) ##################### 6        
    P_K=(np.ones_like(K)-K)*P_K_P ################# 7
    return X_K,P_K ################################ 8

class pointnode:
    def __init__(self,x,y,w,h,roi):
        global idd
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.vx=0
        self.vy=0
        self.colour=np.random.random_integers(0,255,3).tolist()
        self.trace=[[x,y]]
        self.pred_trace=[[x,y]]
        self.length=100       ######  Trace Length
        self.roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        self.idd=idd
        self.penalty=0
        self.penaltythreshold=70
        self.framecount=0
        self.framecountthreshold=5
        self.X_K_1=np.array([[self.x,self.y,self.vx,self.vy]]).transpose()
        self.Y_K_1=np.array([[self.x,self.y,self.vx,self.vy]]).transpose()
        self.P_K_1=np.array([[1.5,0,0,0],[0,1.5,0,0],[0,0,1.5,0],[0,0,0,1.5]])*10
        self.R=np.identity(4)*.1
#        print(self.X_K_1.shape)
        self.px=self.X_K_1[0][0]
        self.py=self.X_K_1[1][0]  
        self.ax=0
        self.ay=0
        self.v=0
        self.a=0
        
        idd+=1
        
    def update_without_measurement(self):
        t=1
        ax=0
        ay=0
        A=np.array([[1,0,t,0],[0,1,0,t],[0,0,1,0],[0,0,0,1]])
        B=np.array([[.5*t**2,0],[0,.5*t**2],[t,0],[0,t]])
        U_K=np.array([[ax],[ay]])        
#        W_K=np.zeros_like(self.X_K_1)
        W_K=np.random.random(self.X_K_1.shape)*.001
        X_K_P = A.dot( self.X_K_1) + B.dot(U_K) + (W_K) ############# 1
        self.px=self.X_K_1[0][0]
        self.py=self.X_K_1[1][0] 
        return X_K_P
    
    def update(self):
#        print(self.X_K_1[0],self.Y_K_1[0],self.X_K_1[1],self.Y_K_1[1],self.X_K_1[2],self.Y_K_1[2],self.X_K_1[3],self.Y_K_1[3])
#        print(self.Y_K_1)
        self.Y_K_1=np.array([[self.x,self.y,self.vx,self.vy]]).transpose()
        self.ax=self.X_K_1[2][0]
        self.ay=self.X_K_1[3][0]
        if self.penalty!=0:
            
            self.X_K_1=self.update_without_measurement()
        else:
            self.X_K_1,self.P_K_1=nextpredictstateusingkalman(self.X_K_1,self.P_K_1,self.Y_K_1,self.R)
        self.px=self.X_K_1[0][0]
        self.py=self.X_K_1[1][0]
        self.vx=self.X_K_1[2][0]
        self.vy=self.X_K_1[3][0]
        self.ax=self.X_K_1[2][0]-self.ax
        self.ay=self.X_K_1[3][0]-self.ay
        self.pred_trace.append([   int(self.X_K_1[0][0]),int(self.X_K_1[1][0])   ])
        self.v=math.sqrt((self.vx)**2+(self.vy)**2)
        self.a=math.sqrt((self.ax)**2+(self.ay)**2)
        
#        print([   int(self.X_K_1[0][0]),int(self.X_K_1[1][0])   ])

#        self.pred_trace.append([self.X_K_1[0],self.X_K_1[1]])

#        if self.Y_K_1[0][0]==self.x and self.Y_K_1[1][0]==self.y:
#            self.vx=0
#            self.vy=0

#        print(self.vx,self.vy)
#        self.Y_K_1=np.array([[self.x,self.y,self.vx,self.vy]]).transpose()
        
        
    
def get_points(contours,img):
    pnt=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        cx=int(x+w/2)
        cy=int(y+h/2)
        roi=img[y:y+h,x:x+w,:]
        pntnode=pointnode(cx,cy,w,h,roi)
        pnt.append(pntnode) ###########################
#        print(pnt)
    return pnt

def nodetolist(p):
    x=p.x
    y=p.y
    return [x,y]

def nodetolistarray(array):
    points=[]
    for i in array:
        points.append(nodetolist(i))
    return points

def drawpoints(img,points):
    if points:
        for j in range(len(points)):
#            print(i)
            i=points[j]
            x=i.x##################
            y=i.y##################
            w=i.w
            h=i.h
#            img=cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),i.colour,3)
            img[y-3:y+3,x-3:x+3]=i.colour
        for j in points:
#==============================================================================
#             for i in range(0,len(j.trace)-1):
#                 pnt1=j.trace[i]
#                 pnt2=j.trace[i+1]
#                 x1=j.trace[i][0]
#                 y1=j.trace[i][1]
#                 x2=j.trace[i+1][0]
#                 y2=j.trace[i+1][1]
#                 cv2.line(img,(x1,y1),(x2,y2),j.colour,3)
#==============================================================================
            for i in range(0,len(j.pred_trace)-1):
                if j.penalty==0:
                    pnt1=j.pred_trace[i]
                    pnt2=j.pred_trace[i+1]
                    x1=j.pred_trace[i][0]
                    y1=j.pred_trace[i][1]
                    x2=j.pred_trace[i+1][0]
                    y2=j.pred_trace[i+1][1]
                    cv2.line(img,(x1,y1),(x2,y2),j.colour,3)
                else:
                    pnt1=j.pred_trace[i]
                    pnt2=j.pred_trace[i+1]
                    x1=j.pred_trace[i][0]
                    y1=j.pred_trace[i][1]
                    x2=j.pred_trace[i+1][0]
                    y2=j.pred_trace[i+1][1]
                    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
#                img[y-1:y+1,x-1:x+1]=j.colour
    return img


def eucdist(p,q):
    return (math.sqrt((p.px-q.x)**2+(p.py-q.y)**2))

def BhattDist(p1,p2):
    hist1 = cv2.calcHist([p1.roi],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([p2.roi],[0],None,[256],[0,256])
    compare = cv2.compareHist(hist1,hist2,cv2.HISTCMP_BHATTACHARYYA)
    return compare



def compareandupdate(targets,detected):
    costmatrix=[]
    
    if targets==[] and detected==[]:
        targets= []
    elif targets==[] and detected!=[]:
        targets= detected
    elif targets!=[] and detected==[]:
        targets= []
    else:
        for i in targets:
            rowcost=[]
            for j in detected:
                rowcost.append(eucdist(i,j))
            costmatrix.append(rowcost)
        det=detected[:]
        tar=targets[:]
    #    pairs={}
        row,col=linear_sum_assignment(costmatrix)
        pairs=[]
        for i in range(len(row)):
            pairs.append([row[i],col[i]])
        for x,y in pairs:  ############################ update assigned ones 1.penalty=0  2.framecount++ 3.
            tar.remove(targets[x])
            det.remove(detected[y])
            dist_thresh=eucdist(targets[x],detected[y])
            if dist_thresh<50:
                targets[x].vx=detected[y].x-targets[x].x
                targets[x].vy=detected[y].y-targets[x].y
            
            
                targets[x].x=detected[y].x
                targets[x].y=detected[y].y
                targets[x].h=detected[y].h
                targets[x].w=detected[y].w
            
            
            
            
                targets[x].trace.append([detected[y].x,detected[y].y])
                if len(targets[x].trace)>targets[x].length:
                    targets[x].trace.pop(0)
                targets[x].penalty = 0
                targets[x].framecount+=1 
            else:
                targets[x].penalty+=1
                
                
        for i in det:
            i.framecount+=1
            targets.append(i)  # added non assigned ones
        for i in tar:
            i.penalty+=1
            if i.penalty>i.penaltythreshold:
                targets.remove(i)  #remove non updated ones
#    detectedlist=nodetolistarray(detected)
#    targetslist=nodetolistarray(targets)    
    return targets


def maxVelAcc(targets):
    vel=[]
    acc=[]
    try:
        for i in targets:
            if i.penalty==0:
                vel.append(i.v)
                acc.append(i.a)
        return [targets[vel.index(max(vel))],targets[acc.index(max(acc))]]
    except:
        return [None,None]
            
acct=2
font = cv2.FONT_HERSHEY_SIMPLEX
#vid=cv2.VideoCapture('bugs.mp4')
#vid=cv2.VideoCapture('test2.mp4')
#vid=cv2.VideoCapture('test.mov')
#vid=cv2.VideoCapture('lablow.avi')

vid = PiVideoStream(rotation=180,b=60,c=0).start()
time.sleep(2.0)

pointsrecord={}
fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()
targets=[]
framecount=0
qq={}
r=True
try:
    while(True):
#        r,frame=vid.read()
        frame=vid.read()
#        frame=cv2.resize(frame,(320,240))
        if r:
            fgmask = fgbg.apply(frame)
            kernel = np.ones((3,3),dtype=np.float32)/9
            kernel5 = np.ones((5,5),dtype=np.int8)/25
            fgmask = cv2.filter2D(fgmask,-1,kernel)
            erosion = cv2.erode(fgmask,kernel,iterations = 2)
            erosion[erosion>1]=255
            dilation = cv2.dilate(erosion,kernel,iterations = 6)
            dilation[dilation>2]=255
            img,contours, _ = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #        img,contours, _ = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
            bb=np.copy(frame)
            detected=get_points(contours,frame)
            detectedlength=len(detected)
            detectedlist=nodetolistarray(detected)
            targetslist=nodetolistarray(targets)
        
            targets=compareandupdate(targets,detected)
            for i in targets:
                i.update()
            for i in targets:
                if i.penalty==0:
    #                print(i.a)
                    frame=cv2.putText(frame,str([round(i.v,2),round(i.a,2)]),(int(i.x-i.w/2),int(i.y)), font, .7,i.colour,2,cv2.LINE_AA)
                    
                    
            
            v,a=maxVelAcc(targets)
            if v:
                print(v.a)
                if v.a>acct and v.penalty==0:
                    blink()
                    cc=np.copy(frame)
                    cc=cv2.rectangle(cc,(int(v.x-v.w/2),int(v.y-v.h/2)),(int(v.x+v.w/2),int(v.y+v.h/2)),v.colour,3)
                    t=time.asctime()+str(time.time()).split('.')[1][:3]
                    cv2.imwrite(t,cc)
            for i in targets:
                try:
                    pointsrecord[i.idd].append([(i.x,i.y),framecount])
                except:
                    pointsrecord[i.idd]=[]
                    pointsrecord[i.idd].append([(i.x,i.y),framecount])
            framecount+=1
    
    
            bb=drawpoints(bb,targets)
        
            cv2.imshow('dilation',dilation)
    #        cv2.imshow('boundingbox',bb)
    
            cv2.imshow('mask',fgmask)
    #        cv2.imshow('videoo',frame)
        #        cv2.imshow('mask2',fgmask2)
            
        
    #        k=cv2.waitKey(1)
            k=cv2.waitKey(1)
            if v:
                if v.a>acct:
                    print(v.a)
                    time.sleep(.05)
                    pass
        if r==False or k==ord('q'):
            cv2.destroyAllWindows()
            vid.release()
            break
        
except:
    cv2.destroyAllWindows()
    vid.release()
     
    
    
    
    
    















































#==============================================================================
# 
# ppp=pointsrecord
# 
# 
# 
# #myarray=np.zeros((1452,len(ppp))).astype('object')
# #mylist=myarray.tolist()
# #removejunk=[]
# #index=0
# #
# ##for i in ppp:    
# ##    for j in ppp[i]:
# ##        myarray[j[1],index]=[j[0]]
# ##    index+=1
# mycount=0
# 
# for i in ppp:
#     print(len(ppp[i]))
#     if len(ppp[i])>10:
#         mycount+=1
#         
# myarray=np.zeros((1452,mycount+1)).astype('object')
# mylist=myarray.tolist()
# removejunk=[]
# index=0
# 
# 
# for i in ppp:
#     print(len(ppp[i]))
#     if len(ppp[i])>10:
# #        removejunk.append(i)
#         for j in ppp[i]:            
#             myarray[j[1],index]=np.array([np.array([j[0]])])
#         index+=1
# 
# #            myarray[j[1],index]=[j[0]]np.array([j[0]])
# 
# c=[]
# for i in range(1,index+2):
#     c.append(i)
# 
# f=[]
# for i in range(1,framecount+1):
#     f.append(i)
#   
#     
# df=pd.DataFrame(myarray,index=f,columns=c)
# df.to_csv('tracks3.csv',index=True,header=True)
# 
# 
# #==============================================================================
# # myarray=np.zeros((1452,2348)).astype('object')
# # for i in ppp:
# #     for j in ppp[i]:
# #         myarray[j[1],i]=[j[0]]
# #         
# #         
# # c=[]
# # for i in range(1,2349):
# #     c.append(i)
# # 
# # f=[]
# # for i in range(1,framecount+1):
# #     f.append(i)
# #   
# #     
# # df=pd.DataFrame(myarray,index=f,columns=c)
# # 
# #==============================================================================
# 
# 
# a=np.array([1,2,3,4,5,6,7,8,9,10])
# my=myarray.transpose()
# 
# my1=np.roll(my,-1,axis=1)
# 
# #mmy1
# 
# 
# velocity=my1-my
# 
# for i in range(velocity.shape[0]):
#     for j in range(velocity.shape[1]):
#         if type(velocity[i][j])!=float:
#             print(i,j)
#             velocity[i][j]=0.0
#             break
#     for j in range(j+1,velocity.shape[1]):
#         if type(velocity[i][j])==float:
#             print(i,j)
#             velocity[i][j-1]=0.0
#             break
#         
# 
# 
# 
# 
# 
# 
# 
# velocitydata=velocity.transpose()
# 
# ac=np.copy(velocity)
# 
# ac1=np.roll(ac,-1,axis=1)
# 
# acceleration=ac1-ac
# 
# accelerationdata=acceleration.transpose()
# 
# df1=pd.DataFrame(velocitydata,index=f,columns=c)
# df1.to_csv('velocity.csv',index=True,header=True)
# 
# df2=pd.DataFrame(accelerationdata,index=f,columns=c)
# df2.to_csv('acceleration.csv',index=True,header=True)
# 
# def processres(v):
#     if type(v)==float:
#         return 0
#     else:
#         x=v[0,0,0]
#         y=v[0,0,1]
#         vr=math.sqrt(x**2+y**2)
#         if vr>3000 :
#             return 0
#         return vr
# 
# velocityresultant=np.zeros_like(velocity)
# flag=0
# for i in range(len(velocity)):
#     flag=0
#     for j in range(len(velocity[i])):
#         v=velocity[i][j]
#         v=processres(v)
#         velocityresultant[i,j]=v
#         
#         
# accelerationresultant=np.zeros_like(acceleration)
# 
# for i in range(len(acceleration)):
#     for j in range(len(acceleration[i])):
#         v=acceleration[i][j]
#         v=processres(v)
#         accelerationresultant[i,j]=v
#         
# np.save('v,a,vv,aa,pos.npy',[velocity,acceleration,velocityresultant,accelerationresultant,my])
# 
# 
# 
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import math
# from scipy.optimize import linear_sum_assignment
# from pandas import DataFrame, read_csv
# import pandas as pd
# 
# [velocity,acceleration,velocityresultant,accelerationresultant,my]=np.load('v,a,vv,aa,pos.npy')
# 
# 
# #==============================================================================
# # for i in range(len(velocityresultant)):
# #     plt.subplot(9,2,i+1)
# #     plt.plot(velocityresultant[i])
# #     
# # plt.show()
# #==============================================================================
# 
# for i in range(len(accelerationresultant)):
#     plt.subplot(9,2,i+1)
#     plt.plot(accelerationresultant[i])
#     
# plt.show()
# 
# 
# frameindexes=[]
# locations=np.zeros((accelerationresultant.shape[1],1),dtype=object).tolist()
# 
# 
# for i in range(accelerationresultant.shape[0]):
#     for j in range(accelerationresultant.shape[1]):
#         if accelerationresultant[i][j]>25:
#             if j not in frameindexes:
#                 locations[j].append(my[i,j])
#                 frameindexes.append(j)
# frameindexes.sort()
# 
# 
# vid2=cv2.VideoCapture('test.mov')
# f=0
# while(1):
#     ret,frame=vid2.read()
#     if ret:
#         if f in frameindexes:
#             frame=np.ones_like(frame)
#             x,y=locations[f][1][0][0]
#             frame[y-40:y+40,x-40:x+40]=255
#             print(x,y)
#             cv2.imshow('rapid change detection',frame)
#             time.sleep(.01)
#         cv2.imshow('rapid change detection',frame)
#         f+=1
#         k=cv2.waitKey(1)
#     
#     if not(ret) or k==ord('q'):
#         cv2.destroyAllWindows()
#         vid2.release()
#         break
# 
# 
# 
#==============================================================================
