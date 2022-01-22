import cv2
import numpy as np
import time
from queue import Queue
import RPi.GPIO as GPIO

#####相机初始化###
cap = cv2.VideoCapture(0) #0表示自己电脑的摄像头，1或2表示用其他摄像头
width=480
height=640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)

###GPIO初始化###
GPIO.setmode(GPIO.BCM)
PINS_ALL=[6,13,19,26,21,20,16,12,25];
for i in PINS_ALL:
    GPIO.setup(i,GPIO.OUT);
state=[0,0];

# 视频帧数
num_real = 0 #实际帧数
frame_start = 20

#二值化范围
Lower = np.array([17, 50, 50])#要识别颜色的下qqqq限[18,50,50],[31,255,255]
Upper = np.array([33, 255, 255])#要识别的颜色的上限

#开闭操作滤波器大小
filter1 = 3#9
filter2 = 17#20

#移动平均滤波器
mid1_xq = Queue(maxsize = 5)
mid1_yq = Queue(maxsize = 5)
for i in range(5):
    mid1_xq.put(0)
    mid1_yq.put(0)


area_d = np.zeros((2,1)) #面积异常处理
mid_d = np.zeros((2,2)) #质心异常处理
exception_d = 0
area_yuzhi = 20000#400
distance_yuzhi = 200#170

# 初始化测量坐标和预测坐标q的数组
last_measurement = current_measurement = np.zeros((3, 1), np.float32)
last_prediction = current_prediction = np.zeros((3, 1), np.float32)
#如果dx初始化为-240或0且摄像头启动态也可转动，则会在启动态时，摄像头快速左转，如果车子在右边，则有可能会丢失目标。
#不管初始化为多少，由于启动态的收敛过程相对于图像中心有一定的距离，
#这个时候如果开启摄像头的转动，那么启动态中摄像头会有瞬间的较大的转动，可能丢失目标也可能损坏硬件。
#因此在启动的几帧要关闭摄像机转动，不一定要是启动态，只要在预测点距离中心不用太远。
#考虑到lman收敛速度快，暂定15-25帧。

#初始化kalman滤波器的维度
kalman = cv2.KalmanFilter(5, 3) # 4：状态数，包括（x，y，dx，dy, ldx）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
# 第一个参数是状态的维度，第二个是测量的，第三个是控制，默认0
# 加上速度是为了加快跟踪速度
############存储渐消因子###############
Lambda=1.0
multi=160
error_frame=[]
cnt_frame=0

#初始化状态转移和观测矩阵
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0],[0, 0, 0, 0, 1]], np.float32) # 系统测量矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0, -1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], np.float32) # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差
def draw_line(frame,pre,post,multi=160):
    x_pre=(int)(np.round(pre[0]*multi))
    y_pre=(int)(np.round(pre[1]*multi))
    x_post=(int)(np.round(post[0]*multi))
    y_post=(int)(np.round(post[1]*multi))
    cv2.line(frame,(x_pre,y_pre),(x_post,y_post),(0, 0, 255),5)
    return 0; 
def Roate(PIN,pulse=0.01,pause=0.001):
    ID=(int)(PIN);
    GPIO.output(ID,GPIO.HIGH);
    time.sleep(pulse)
    GPIO.output(ID,GPIO.LOW);
    time.sleep(pause);
    
def Turn(PINS,state,step,direct):
    Pulse=0.005#1step0.7°
    for i in range(step):
        if(direct==0):
            state+=1;
        else:
            state-=1;
        if(state>=4):
            state=0;
        if(state<0):
            state=3;
        Roate(PINS[state],pulse=Pulse);    
    return state;
    
def Ctrl_Turn(PINS_ALL,state,step_direct,hv):
    #hv=1:horizon
    #hv=0:verical
    #step=round(abs(step_direct))
    step =(int)( round(abs(step_direct/480*45)))
    if(step_direct>0):
        direct=0;
    else:
        direct=1;
    if(hv==0):
        state[0]=Turn(PINS_ALL[0:4],state[0],step,direct);
    else:
        state[1]=Turn(PINS_ALL[4:8],state[1],step,direct);
    return state;


while(1):
    time_start=time.time()#开始计时，计算帧数用
    
    # 获取摄像头拍摄到的画面
    ret,frame= cap.read()
    if ret == False:
        break
    frame = cv2.flip(cv2.transpose(frame),0)
    #目标识别
    # RGB空间转HSV
    f_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #cv2.imshow('f_hsv',f_hsv)

    #二值化
    mask = cv2.inRange(f_hsv, Lower, Upper)
    #cv2.imshow('f_inrange',mask)

    # 开操作
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(filter1,filter1)) #开操作滤波器
    mask_1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    # 闭操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(filter2,filter2))#闭操作滤波器
    dilation = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE, kernel2)
    #cv2.imshow('filter',dilation)
    
    # 将滤波后的图像变成二值图像放在binary中
    ret, binary = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY) 
    # 在binary中发现轮廓，轮廓按照面积从小到大排列
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)     
    
    #找出最大进行矩形框
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

        #质心坐标
        flag, mid_x, mid_y = 1, x+int(w/2), y+int(h/2)
        
        if(exception_d==0):
            area_d[0] = area_d[1] #上一时刻
            area_d[1] = max_area #这一时刻
            mid_d[0] = mid_d[1]
            mid_d[1] = np.array([mid_x, mid_y])
            distance = (sum((mid_d[1]-mid_d[0])**2))**0.5
            area_diff = max_area - area_d[0]
            #print(distance,area_diff)
        else:
            mid_exc = np.array([mid_x, mid_y])
            distance = (sum((mid_exc-mid_d[0])**2))**0.5
            area_diff = max_area - area_d[0]
            if distance <= distance_yuzhi and abs(area_diff)<area_yuzhi:
                area_d[1] = max_area #这一时刻
                mid_d[1] = np.array([mid_x, mid_y])
            #print(distance,area_diff)
         # if  (num_real <= frame_start or ((distance <= distance_yuzhi and abs(area_diff)<area_yuzhi ))):
        if(1):
         #识别到了，面积未突变，坐标未突变下的卡尔曼滤波跟踪
         #启动噪声不能当异常,因此异常启动过程不能有较大噪声
            exception_d = 0 #异常帧数清0
            #目标跟踪
            #kalman滤波估计
            last_prediction = current_prediction # 把当前预测存储为上一次预测
            last_measurement = current_measurement # 把当前测量存储为上一次测量
            current_measurement =  np.array([np.float32(mid_x),np.float32(mid_y),np.float32(mid_x-240)])/multi# 当前测量
            
            Y=np.array([[current_measurement[0]-last_prediction[0][0],current_measurement[1]-last_prediction[1][0],0]])
            C_hat=(Lambda)/(1+Lambda)*np.dot(Y,Y.T)
            C_no=np.dot(kalman.measurementMatrix,np.dot(kalman.errorCovPre,kalman.measurementMatrix.T))+kalman.measurementNoiseCov
            Lambda=max(1,np.trace(C_hat)/np.trace(C_no))
            kalman.errorCovPre=kalman.errorCovPre*Lambda
            
            kalman.correct(current_measurement) # 用当前测量来校正卡尔曼滤波器
            current_prediction = kalman.predict() # 计算卡尔曼预测值，作为当前预测
            '''
            lmx, lmy, lmdx = last_measurement[0], last_measurement[1], last_measurement[2] # 上一次测量坐标
            cmx, cmy, cmdx = current_measurement[0], current_measurement[1], current_measurement[2] # 当前测量坐标
            lpx, lpy, lpdx = last_prediction[0], last_prediction[1], last_prediction[2] # 上一次预测坐标
            cpx, cpy, cpdx = current_prediction[0], current_prediction[1], current_prediction[2] # 当前预测坐标
            cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 255),5) # 红色线为预测值
            '''
            draw_line(frame,last_prediction,current_prediction,multi)
            cv2.putText(frame, " {0}".format(int('%d' % (exception_d))), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
            
            error_frame.append(np.sum((np.array([current_measurement[0]-last_prediction[0][0],current_measurement[1]-last_prediction[1][0]]))**2)**0.5)
        else:
            exception_d = 1
            last_prediction = current_prediction # 把当前预测存储为上一次预测
            #current_measurement = np.array([np.float32(cpx),np.float32(cpy)])
            #kalman.correct(current_measurement) # 用当前测量来校正卡尔曼滤波器
            current_prediction = kalman.predict() # 计算卡尔曼预测值
            lpx, lpy, lpdx = last_prediction[0], last_prediction[1], last_prediction[2] # 上一次预测坐标
            cpx, cpy, cpdx = current_prediction[0], current_prediction[1], current_prediction[2] # 当前预测坐标
            cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 255),5) # 红色线为预测值
            cv2.putText(frame, " {0}".format(int('%d' % (exception_d))), (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
        
        if(num_real > frame_start): #收敛一定帧数后开启摄像头自主跟踪
            state = Ctrl_Turn(PINS_ALL, state, current_prediction[2][0]*multi, 1)

            #cpdx
            #continue
            
    time_end=time.time()  
    fps =  1/(time_end - time_start)
    
    #实时显示帧数
    cv2.putText(frame, "FPS {0}".format(float('%.1f' % (fps))), (300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.imshow('recognition',frame)
    #帧数计数
    num_real += 1
    
    #关闭窗口
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
np.save("data123.npy",error_frame)
cap.release()
cv2.destroyAllWindows()
