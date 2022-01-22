import cv2
import numpy as np
import time

#####相机初始化###
cap = cv2.VideoCapture(0) #0表示自己电脑的摄像头，1或2表示用其他摄像头
width=480
height=640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
num = 300 #帧数

Lower = np.array([20, 43, 46])#要识别颜色的下限
Upper = np.array([32, 255, 255])#要识别的颜色的上限
kernel_4 = np.ones((4,4),np.uint8)#4x4的卷积核
mid1 = np.zeros((num,2)) #矩形中心
mid2 = np.zeros((num,2)) #最小矩形中心
mid3 = np.zeros((num,2)) #质心
FPS = np.zeros((num,1)) #存帧数

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('OI_test3.avi',fourcc, 20.0, (640,480))

#for i in range(num):
while(1): 
    time_start=time.time()
    # 获取摄像头拍摄到的画面
    ret,frame= cap.read()

    # RGB空间转HSV
    f_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow('f_hsv',f_hsv)

    mask = cv2.inRange(f_hsv, Lower, Upper)
    cv2.imshow('f_inrange',mask)

    # 开操作
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(30,30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    # 闭操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(30,30))
    dilation = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    cv2.imshow('filter',dilation)
    
    # 将滤波后的图像变成二值图像放在binary中
    ret, binary = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY) 
    # 在binary中发现轮廓，轮廓按照面积从小到大排列
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)     

    if(len(contours)>0):
        x,y,w,h = cv2.boundingRect(contours[-1])#将轮廓分解为识别对象的左上角坐标和宽、高
        # 在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,),3)
        w_mid,h_mid = int(w/2),int(h/2)
        # cv2.rectangle(frame,(x+w_mid-1,y+h_mid-1),(x+w_mid+1,y+h_mid+1),(0,255,),1)
        # 用红色表示有旋转角度的矩形框架
        rect = cv2.minAreaRect(contours[-1])
        box = cv2.boxPoints(rect)
        box = np.int0(box)  #坐标一定是整数
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        mid1[i,:] = [x+w_mid,h_mid]
        mid2[i,:] = np.array(rect[0])
    time_end=time.time()  
    fps =  1/(time_end - time_start)
    #实时显示帧数
    cv2.putText(frame, "FPS {0}".format(float('%.1f' % (fps))), (460, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.imshow('recognition',frame)
    out.write(frame) #保持帧
    FPS[i] = fps
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cv2.destroyAllWindows()
print("最大帧数为%.1f, 最小帧数为%.1f, 平均帧数为%.1f"%(max(FPS),min(FPS),sum(FPS)/num))