import cv2
import numpy as np
import time
from queue import Queue

#####相机初始化###
#cap = cv2.VideoCapture("z.avi") #0表示自己电脑的摄像头，1或2表示用其他摄像头
cap = cv2.VideoCapture(0)
width=480
height=640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
num = 300 #帧数

Lower = np.array([20, 43, 46])#要识别颜色的下限[18,43,46],[33,255,255]
Upper = np.array([33, 255, 255])#要识别的颜色的上限
kernel_4 = np.ones((4,4),np.uint8)#4x4的卷积核
#mid1_d = np.zeros((5,2)) #矩形中心
#mid2_d = np.zeros((5,2)) #最小矩形中心

mid1_xq = Queue(maxsize = 5)
mid1_yq = Queue(maxsize = 5)
for i in range(5):
    mid1_xq.put(0)
    mid1_yq.put(0)

mid3 = np.zeros((num,2)) #质心
FPS = np.zeros((num,1)) #存帧数

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('quanqidong_318.avi',fourcc, 20.0, (height,width)) 
fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('quanqidong_318_track.avi',fourcc1, 5.0, (height,width)) 

# 初始化测量坐标和预测坐标的数组
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)

kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
# 第一个参数是状态的维度，第二个是测量的，第三个是控制，默认0
# 加上速度是为了加快跟踪速度
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差

while(1):
    time_start=time.time()
    # 获取摄像头拍摄到的画面
    ret,frame= cap.read()
    #out.write(frame)
    if(ret==False):
        break
    
    # RGB空间转HSV
    f_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #cv2.imshow('f_hsv',f_hsv)

    mask = cv2.inRange(f_hsv, Lower, Upper)
    cv2.imshow('f_inrange',mask)

    # 开操作
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    # 闭操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
    dilation = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    cv2.imshow('filter',dilation)
    
    # 将滤波后的图像变成二值图像放在binary中
    ret, binary = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY) 
    # 在binary中发现轮廓，轮廓按照面积从小到大排列
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)     
    
    if(len(contours)>0):
        
        max_area = -1
        for j in range(len(contours)):
            area = cv2.contourArea(contours[j])
            if area > max_area:
                cnt = contours[j]
                max_area = area
        
        x,y,w,h = cv2.boundingRect(cnt)#将轮廓分解为识别对象的左上角坐标和宽、高
        # 在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,),2)
        w_mid,h_mid = int(w/2),int(h/2)
        #cv2.rectangle(frame,(x+w_mid-1,y+h_mid-1),(x+w_mid+1,y+h_mid+1),(0,255,),1)
        # 用红色表示有旋转角度的矩形框架
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  #坐标一定是整数
        #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        
        #质心坐标
        mid1_xq.get()
        mid1_yq.get()
        mid1_xq.put(x+w_mid)
        mid1_yq.put(y+h_mid)
        mid1_x = np.dot(np.array(mid1_xq.queue),[[0],[0],[0],[0],[1]])
        mid1_y = np.dot(np.array(mid1_yq.queue),[[0],[0],[0],[0],[1]])
        
        #kalman滤波估计
        last_prediction = current_prediction # 把当前预测存储为上一次预测
        last_measurement = current_measurement # 把当前测量存储为上一次测量
        #current_measurement = np.array([np.float32(rect[0][0]),np.float32(rect[0][1])]) # 当前测量
        current_measurement =  np.array([np.float32(mid1_x),np.float32(mid1_y)])# 当前测量
        kalman.correct(current_measurement) # 用当前测量来校正卡尔曼滤波器
        current_prediction = kalman.predict() # 计算卡尔曼预测值，作为当前预测
        
        lmx, lmy = last_measurement[0], last_measurement[1] # 上一次测量坐标
        cmx, cmy = current_measurement[0], current_measurement[1] # 当前测量坐标
        lpx, lpy = last_prediction[0], last_prediction[1] # 上一次预测坐标
        cpx, cpy = current_prediction[0], current_prediction[1] # 当前预测坐标

        # 绘制从上一次测量到当前测量以及从上一次预测到当前预测的两条线
        #cv2.line(frame, (lmx, lmy), (cmx, cmy), (255, 255, 0),2) # 青色线为测量值
        cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 255),5) # 红色线为预测值
        #if(i>=6):
        #    cv2.imshow('frame2',frame2)
    time_end=time.time()  
    fps =  1/(time_end - time_start)
    
    #实时显示帧数
    cv2.putText(frame, "FPS {0}".format(float('%.1f' % (fps))), (460, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.imshow('recognition',frame)
    out1.write(frame) #保持帧
    FPS[i] = fps
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(33)
cv2.destroyAllWindows()
print("最大帧数为%.1f, 最小帧数为%.1f, 平均帧数为%.1f"%(max(FPS),min(FPS),sum(FPS)/num))
out.release()
out1.release()
