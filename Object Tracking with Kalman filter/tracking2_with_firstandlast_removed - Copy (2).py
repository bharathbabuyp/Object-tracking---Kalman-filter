import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import linear_sum_assignment
from pandas import DataFrame, read_csv
import pandas as pd
import time
from stream_thread_pi import PiVideoStream

colours=[(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),(255,0,255),(123,54,222),(89,20,190)]
idd=1
class pointnode:
    def __init__(self,x,y,w,h,roi):
        global idd
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.colour=np.random.random_integers(0,255,3).tolist()
        self.trace=[[x,y]]
        self.length=100       ######  Trace Length
        self.roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        self.idd=idd
        self.penalty=0
        self.penaltythreshold=5
        self.framecount=0
        self.framecountthreshold=5
        self.nx=0
        self.ny=0
        self.vx=0
        self.vy=0
        self.ax=0
        self.ay=0
        
        idd+=1

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
            for i in range(0,len(j.trace)-1):
                pnt1=j.trace[i]
                pnt2=j.trace[i+1]
                x1=j.trace[i][0]
                y1=j.trace[i][1]
                x2=j.trace[i+1][0]
                y2=j.trace[i+1][1]
                cv2.line(img,(x1,y1),(x2,y2),j.colour,3)
#                img[y-1:y+1,x-1:x+1]=j.colour
    return img


def eucdist(p,q):
    return (math.sqrt((p.x-q.x)**2+(p.y-q.y)**2))

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
            targets[x].x=detected[y].x
            targets[x].y=detected[y].y
            targets[x].w=detected[y].h
            targets[x].h=detected[y].w
            targets[x].trace.append([detected[y].x,detected[y].y])
            if len(targets[x].trace)>targets[x].length:
                targets[x].trace.pop(0)
            targets[x].penalty = 0
            targets[x].framecount+=1 
                
                
        for i in det:
            targets.append(i)  # added non assigned ones
        for i in tar:
            targets.remove(i)  #remove non updated ones
#    detectedlist=nodetolistarray(detected)
#    targetslist=nodetolistarray(targets)    
    return targets
            


#vid=cv2.VideoCapture('bugs.mp4')
#vid=cv2.VideoCapture('test2.mp4')
#vid=cv2.VideoCapture('test.mov')
#vid=cv2.VideoCapture('lab.avi')
vid = PiVideoStream(rotation=180,b=60,c=0).start()
time.sleep(2.0)

pointsrecord={}
fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()
targets=[]
framecount=0
qq={}
r=True
while(True):
#    r,frame=vid.read()
    frame=vid.read()
#    frame=cv2.resize(frame,(320,240))
    if r:
        fgmask = fgbg.apply(frame,learningRate=.1)
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
            try:
                pointsrecord[i.idd].append([(i.x,i.y),framecount])
            except:
                pointsrecord[i.idd]=[]
                pointsrecord[i.idd].append([(i.x,i.y),framecount])
        framecount+=1

        
        bb=drawpoints(bb,targets)
    
#        cv2.imshow('dilation',dilation)
#        cv2.imshow('boundingbox',bb)
    #        cv2.imshow('videoo',frame)
        cv2.imshow('mask',fgmask)
    #        cv2.imshow('mask2',fgmask2)
        
    
#        k=cv2.waitKey(1)
        k=cv2.waitKey(1)
        
    if r==False or k==ord('q'):
        cv2.destroyAllWindows()
        vid.release()
        break
    
#==============================================================================
# 
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
