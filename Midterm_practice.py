#!/usr/bin/env python
# coding: utf-8
# In[1]:
import numpy as np
import cv2 as cv
# # 항목의 수 
# ## size 속성을 사용
# ## (주의) 항목의 수를 구하기 위해 len()함수를 사용할 경우 다차원 배열에서 문제 발생
# In[2]:
nums = np.array([1,4,2,6,3])
print(nums)
print(nums.ndim)
print(nums.shape)
print(len(nums.shape))
print(nums.size)
# # 슬라이싱 
# ## 배열명[start : stop]
# ## 배열명[start : stop : step]
# In[3]:
nums = np.array([1,2,4,3,2])
print(nums[::2])
# In[4]:
nums=np.array([[1,4,2],[7,5,2]])
print(nums)
# In[5]:
nums.ndim
# In[6]:
nums.shape
# In[7]:
len(nums.shape)
# In[8]:
nums.size
# In[9]:
len(nums)
# In[10]:
nums[0,2]
# In[11]:
nums[0][2]
# In[12]:
print(nums)
# In[13]:
np.array([[1,4,2],[7,5,3]])
print(nums)
# In[14]:
print(nums[0:1,])
# In[15]:
print(nums[0:1,:])
# In[16]:
print(nums[:,1:2])
# In[17]:
print(nums[:,1])
# In[18]:
print(nums[1:,1:])
# In[19]:
print(nums[1,1:])
# In[20]:
print(nums[1:,1])
# In[21]:
nums[1:,1:2].ndim
# In[22]:
nums[1:,1].ndim
# In[23]:
print(nums[:,2:])
# In[24]:
print(nums[1,:2])
# In[25]:
print(nums[1:,:2])
# In[26]:
np.array(3)
# In[27]:
nums=np.array(3)
print(nums)
# In[28]:
print(nums.ndim)
# In[29]:
print(nums.shape)
# # Numpy 배열 사용시 주의사항
# 
# ## 배열에 대한 검색/슬라이싱 결과는 참조만 할당
# ## 기존 배열이 변경될 경우 값이 같이 변경 됨
# 
# ## 독립적으로 사용해야할 경우 copy() 함수 사용
# In[30]:
nums = np.array([1,4,2,5,3])
# In[31]:
ref = nums[1:4]
print(ref)
# In[32]:
cpy = nums[1:4].copy()
print(cpy)
# In[33]:
nums[2] = 10
# In[34]:
print(ref) # 참조를 하고 있어기 때문에 변경된 값이 반영
# In[35]:
print(cpy) # nums[1:4].copy() 를 이용하면 독립적으로 사용가능
# In[36]:
print(np.zeros((2,2)))
# In[37]:
print(np.zeros(2))
# In[38]:
print(np.ones(1))
# In[39]:
print(np.ones((1,2)))
# In[40]:
print(np.full((2,2),7))
# In[41]:
print(np.full((2,2),0))
# In[42]:
np.identity(3)
# In[43]:
np.eye(3)
# In[44]:
np.eye(3,k=1)
# In[45]:
np.eye(3,k=3)
# In[46]:
np.eye(3,k=-2)
# In[47]:
np.random.random((2,2)) # 0 과 1 사이의 임의의 숫자 
# In[48]:
np.linspace(0,1,num=5,endpoint=True)
# In[49]:
np.linspace(0,1,num=5,endpoint=False)
# In[50]:
np.linspace(0,1,num=5,endpoint=False,retstep=True)
# In[51]:
np.arange(1,3,1)
# In[52]:
np.arange(0.4,1.1,0.1)
# In[53]:
sap=np.array(['mmm','aby','assdf','safsaf','mmm','aby','assdf','safsaf','mmm','aby','assdf','safsaf'])
len(sap.shape)
# In[54]:
len(sap)
# In[55]:
sap
# In[56]:
a= sap.reshape(3,4)
# In[57]:
len(a)
# In[58]:
a
# In[59]:
len(a)
# In[60]:
a.size
# In[61]:
a
# In[62]:
a.T
# In[63]:
b=sap.reshape(2,3,2)
# In[64]:
b
# In[65]:
b.T
# In[66]:
b.transpose(0,2,1)
# In[67]:
x=np.array([1,2,3])
y=np.array([3,2,1])
print(np.concatenate([x,y]))
# In[68]:
z=np.array([4,5,6])
# In[69]:
print(np.concatenate([x,y,z]))
# In[70]:
grid=np.array([[1,2,3],[4,5,6]])
# In[71]:
grid
# In[72]:
np.concatenate([grid,grid])
# In[73]:
np.concatenate([grid,grid],axis=1)
# In[74]:
np.vstack([grid,grid])
# In[75]:
np.hstack([grid,grid])
# In[76]:
y=np.array([[9],
         [8]])
grid=np.array([[1,2,3],
              [4,5,6]])
# In[77]:
np.hstack([y,grid])
# # Split(배열명, 분할 지점의 리스트)
# In[78]:
x=[1,2,3,99,99,3,2,1]
x1,x2,x3=np.split(x,[3,5])  # 인덱스 3에서 4까지(5전까지)
print(x1,x2,x3)
# In[79]:
grid=np.arange(16)
grid
# In[80]:
grid=grid.reshape((4,4))
# In[81]:
print(grid)
# In[82]:
upper,lower = np.split(grid,[2]) #2전 후로 나눔
# In[83]:
upper
# In[84]:
lower
# In[85]:
grid=np.arange(16).reshape((4,4))
# In[86]:
print(grid)
# In[87]:
left,right = np.hsplit(grid,[1]) # 수평으로 스플릿
# In[88]:
left
# In[89]:
right
# In[90]:
a=np.array([0,1,2])
b=np.array([5,5,5])
print(a+b)
# In[91]:
a*b
# In[92]:
5*b
# In[93]:
a*5
# In[94]:
np.eye(4)
# In[95]:
np.eye(4)+0.01
# In[96]:
np.eye(4)+0.01*np.ones(4,)
# In[97]:
np.ones((4,))

# In[98]:
import numpy as np
M=np.ones((3,2))
a=np.arange(3)
print(M)
print(a)
# In[99]:
M = np.ones((3,2))
a=np.arange(3)
# In[100]:
M.shape
# In[101]:
a.shape
# In[102]:
a=np.array([0,-1,2])
a
# In[103]:
abs(a)
# In[104]:
grid=np.array([[1,2,3],
          [4,6,2]])
np.mean(grid)
# In[105]:
np.mean(grid,axis=0)
# In[106]:
np.mean(grid,axis=1)
# In[107]:
x=np.array([1,2,3,4,5])
x<3
# In[108]:
a>=3
# In[109]:
x==3
# In[110]:
x!=3
# In[111]:
rng=np.random.RandomState(0)
rng
# In[112]:
x=rng.randint(10,size=(3,4))
x
# In[113]:
x<6
# In[114]:
np.count_nonzero(x<6) # coount_nonzero() 함수를 통해 True 개수 파악
# In[115]:
np.sum(x)
# In[116]:
np.sum(x<6)
# In[117]:
rng=np.random.RandomState(0)
x=rng.randint(10,size=(3,4))
x
# In[118]:
np.any(x>8)  # 하나라도 만족하는지
# In[119]:
np.all(x<6) #모든 값이 조건을 만족하는지
# In[120]:
np.all(x<6,axis=1)
# In[121]:
np.sum((x>=3)&(x<6))
# In[122]:
(x>3)&(x<6)
# In[123]:
x[x<5]
# In[124]:
x<5
# In[125]:
x=np.random.RandomState(0)
x
# In[126]:
x=x.randint(10,size=(3,4))
x
# In[127]:
x[x<6]
# In[128]:
def func1():
    img1=cv.imread('cat.bmp',cv.IMREAD_GRAYSCALE)
    
    if img1 is None:
        print('Image load failed!')
        return
    
    print('type(img1):', type(img1))
    print('img1.shape:', img1.shape)
    
    if len(img1.shape) == 2:
        print('img1 is a grayscale image')
    
    elif len(img1.shape) ==3:
        print('img1 is a truecolor image')
        
    cv.imshow('img1',img1)
    cv.waitKey()
    cv.destroyAllWindows()
func1()
# In[129]:
matrix = np.zeros((3, 3), dtype=int)
matrix2 = np.ones((3, 3), dtype=int)
mat3 = np.eye(3, dtype=int)
# In[130]:
matrix
# In[131]:
matrix2
# In[132]:
mat3
# In[133]:
def func2():
    img1 = np.empty((480,640), np.uint8) #grayscale image
    img2 = np.zeros((480, 640,3),np.uint8) #color image
    img3 = np.ones((480, 640), np.int32) #1's matrix
    img4 = np.full((480,640),0, np.float32) #Fill with 0.0
    
    mat1 = np.array([[11,12,13,14],
                     [21,22,33,24],
                     [31,32,33,34]]).astype(np.uint8)
    
    mat1[0,1] = 100 #element at x=1, y=0
    mat1[2,:] = 200 
    
    print(mat1)
    
func2()
# # 얕은 복사 (shallow copy)
# # 별도로 메모리를 할당하여 사용하려면 => copy() 사용
# In[135]:
# 행렬의 복사
def func3():
    img1=cv.imread('cat.bmp')
    
    img2=img1
    img3=img1.copy()
    
    img1[:,:] = (0,255,255) #g+r = yellow
    
    cv.imshow('img1',img1)
    cv.imshow('img2',img2)
    cv.imshow('img3',img3)
    
    cv.waitKey()
    cv.destroyAllWindows()
func3()
# In[138]:
# 부분 행렬 추출
def func4():
    img1=cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)
    
    img2= img1[200:400, 200:400]
    img3 = img1[200:400, 200:400].copy()
    
    img2+=50 
    
    cv.imshow('img1',img1)
    cv.imshow('img2',img2)
    cv.imshow('img3',img3)
    cv.waitKey()
    cv.destroyAllWindows()
    
func4()
# # 색상 반전 => 255 - 원래 값
# # 부분 행렬 추출 후 반전
# In[139]:
# 행렬 연산하기
def func6():
    mat1 = np.ones((3,4),np.int32) # 1's matrix
    mat2= np.arange(12).reshape(3,4)
    mat3=mat1+mat2
    mat4=mat2 * 2
    
    print('mat1:')
    print(mat1)
    print('mat2:')
    print(mat2)
    print('mat3:')
    print(mat3)
    print('mat4:')
    print(mat4)
    
func6()

# In[142]:
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()
    
print('Frame width:',int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
print('Frame height:',int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

# In[1]:
import cv2 as cv
#cap = cv.VideoCapture('stopwatch.avi')
#if not cap.isOpenedend():
    
# In[2]:
import numpy as np
import matplotlib.pyplot as plt
image = cv.imread('cat.bmp')
print(image.shape)
#핼열 정보만 저장
height, width = image.shape[:2]   
M= np.float32([[1,0,50],[0,1,10]])
dst=cv.warpAffine(image,M, (width,height))  #(x,y) = (열, 행) =(width, height) 순서 주의
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.show()
# In[3]:
expand=cv.resize(image,None, fx=2.0,fy=2.0, interpolation=cv.INTER_CUBIC)
plt.imshow(cv.cvtColor(expand, cv.COLOR_BGR2RGB))
plt.show()
print(expand.shape)
# In[4]:
shrink = cv.resize(image, None, fx=0.8, fy=0.8, interpolation=cv.INTER_AREA)
plt.imshow(cv.cvtColor(shrink, cv.COLOR_BGR2RGB))
plt.show()
print(shrink.shape)
# # 이미지 회전
# ## cv2.getRotationMatrix2D(center, angle, scale)
# ### 회전 중심, 회전 각도, 스케일(크기)
# In[5]:
M=cv.getRotationMatrix2D((width/2,height/2),90,0.5)
dst=cv.warpAffine(image, M, (width, height))
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.show()
print(M)
# In[6]:
print(image)
# In[7]:
image.size
# In[29]:
image.shape
# In[9]:
reshapingtest = image.reshape(960, 320,3)
plt.imshow(cv.cvtColor(reshapingtest, cv.COLOR_BGR2RGB))
plt.show()
# # 특정 범위 변경
# 
#         
# In[13]:
for i in range(0,100):
    for j in range(0,100):
        image[i,j] = [255,255,255]
plt.imshow(image)
# In[16]:
image[:100,100:200]=[0,0,0]
plt.imshow(image)
# In[18]:
image=cv.imread('cat.bmp')
# In[22]:
roi=image[200:300, 300:400]
plt.imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
plt.show()
image[0:100, 0:100]=roi
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.show()
# In[23]:
cap=cv.VideoCapture('stopwatch.avi')
if not cap.isOpened():
    print('Video open failed!')
    sys.exit()
    
print('Frame width:',int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
print('Frame height:',int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:',int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
fps=cap.get(cv.CAP_PROP_FPS)
print('FPS:',fps)
delay= round(1000/fps)
# In[24]:
while True:
    ret, frame = cap.read()  #cap.read() 각 프레임 읽기
    
    if not ret:
        break          
    
    inversed=~frame
    
    cv.imshow('frame',frame)
    cv.imshow('inversed',inversed)
    
    if cv.waitKey(delay) == 27:
        break
        
cv.destroyAllWindows()

# In[9]:
import numpy as np
import cv2 as cv
img = np.full((400,400,3), 255, np.uint8)
cv.line(img, (50,50),(200,50), (0,0,255))
cv.line(img, (50,100),(200,100), (255,0,255),3 )
cv.line(img, (50,150),(200,150), (255,0,0),10)
cv.line(img, (250,50),(350,100), (0,0,255),1, cv.LINE_4)
cv.line(img, (250,70),(350,120), (255,0,255),1,cv.LINE_8)
cv.line(img, (250,90),(350,140), (255,0,0),1,cv.LINE_AA)
cv.arrowedLine(img, (50,200),(150,200),(0,0,255),1)
cv.arrowedLine(img, (50,250),(350,250),(255,0,255),1)
cv.arrowedLine(img, (50,300),(350,300),(255,0,0),1,cv.LINE_8,0,0.05)
cv.drawMarker(img, (50, 350), (0,0,255), cv.MARKER_CROSS)
cv.drawMarker(img, (100,350), (0,0, 255),cv.MARKER_TILTED_CROSS)
cv.imshow('img',img)
cv.waitKey()
cv.destroyAllWindows()
# In[18]:
img = np.full((400,400,3),255,np.uint8)
cv.rectangle(img, (50,50), (150,100),(0,0,255),2)
cv.rectangle(img, (50,150),(150,200),(0,0,128),-1)
cv.circle(img,(300,120),30,(255,255,0), -1, cv.LINE_AA)
cv.circle(img,(300,120),60,(255,0,0), 3, cv.LINE_AA)
cv.ellipse(img, (120,300),(60,30),20,0,270,(255,255,0),cv.FILLED, cv.LINE_AA)
cv.ellipse(img, (120,300),(100,50),20,0,360,(0,255,0),2, cv.LINE_AA)
pts=np.array([[250,250],[300,250],[300,300],[350,300],[350,350],[250,350]])
cv.polylines(img,[pts],True, (255,0,255),2)
cv.imshow('img',img)
cv.waitKey()
cv.destroyAllWindows()
# In[6]:
import numpy as np
import cv2 as cv
img=np.full((500,800,3),255,np.uint8)
cv.putText(img, "Hello! world!", (20,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))  #문자열을 출력할 좌측 하단 좌표를 대입한다.
cv.imshow("img",img)
cv.waitKey(3000)
cv.destroyAllWindows()
# In[9]:
# 문자열 출력하기 출력할 문자열의 크길르 바탕으로 정중앙에 출력하기
img = np.full((200,640,3), 255, np.uint8)
text = "Hello,OpenCV"
fontFace = cv.FONT_HERSHEY_TRIPLEX
fontScale = 2.0
thickness = 1
sizeText, _ = cv.getTextSize(text, fontFace, fontScale, thinkness)
org=((img.shape[1]-sizeText[0])//2, (img.shape[0]+sizeText[1])//2)
cv.putText(img, text,org,fontFace,fontScale, (255,0,0), thickness)
cv.rectangle(img,org,(org[0]+sizeText[0], org[1]-sizeText[1]),(0,255,0),1)
cv.imshow('img',img)
cv.waitKey()
cv.destroyAllWindows()
import cv2 as cv
import numpy as np
def on_mouse(event,x,y,flags, param):
    global cnt, src_pts
    if event == cv.EVENT_LBUTTONDOWN:
        if cnt < 4:
            src_pts[cnt, :] = np.array([x,y]).astype(np.float32)
            cnt+=1
            
            cv.circle(src, (x,y),5, (0,0,255), -1)
            cv.imshow('src',src)
            
        if cnt == 4:
            w=200
            h=300
            
            dst_pts = np.array([[0,0],
                                [w-1,0],
                                [w-1, h-1],
                                [0, h-1]]).astype(np.float32)
            
            pers_mat = cv.getPerspectiveTransform(src_pts, dst_pts) #원본영상 4점과, #직접 점찍은 목표지점 4점을 입력, 3x3투시변환 행렬 생성해줌
            
            dst = cv.warpPerspective(src, pers_mat, (w,h))
            
            cv.imshow('dst',dst)
# In[13]:
cnt = 0 
src_pts = np.zeros([4,2], dtype=np.float32)
src=cv.imread('card.bmp')
if src is None:
    pritn('load failed')
    sys.exit()
    
cv.namedWindow('src')
cv.setMouseCallback('src', on_mouse)
cv.imshow('src',src)
cv.waitKey(0)
cv.destroyAllWindows()

# In[1]:
import cv2 as cv
import numpy as np
def affine_transform():
    src = cv.imread('tekapo.bmp')
    
    if src is None:
        print('failed!')
        return
    
    rows = src.shape[0]
    cols = src.shape[1]
    
    src_pts = np.array([[0,0],
                        [cols-1, 0],
                        [cols-1, rows-1]]).astype(np.float32) #tekapo.bmp의 왼쪽위, 오른쪽 위, 오른쪽 아래점을 잡음
    
    dst_pts = np.array([[50,50],
                        [cols - 100, 100],
                        [cols - 50, rows-50]]).astype(np.float32)
    
    affine_mat=cv.getAffineTransform(src_pts,dst_pts) #점 6개대입
    
    dst= cv.warpAffine(src, affine_mat,(0,0))
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.waitKey()
    cv.destroyAllWindows()
# In[2]:
affine_transform()
# # 이동 변환 Translation Transformation
# 
# # M= 1 0 a
# #       0 1 b      a와 b는 각각 x,y방향으로 이동하는 크기
# 
# ## opencv에서 이동변환은  2x3 실수 행렬 M을 만들고, warpAffine()함수 인자로 전달하면됨
# In[6]:
def affine_translation():
    src = cv.imread('tekapo.bmp')
    
    if src is None :
        print('load failed!')
        return 
    
    affine_mat = np.array([[1,0,150],
                           [0,1,100]]).astype(np.float32) #x 방향으로 150, y방향으로 100 평행이동
    
    dst = cv.warpAffine(src, affine_mat,(0,0))
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.waitKey()
    cv.destroyAllWindows()
# In[7]:
affine_translation()
# In[9]:
def affine_shear():
    src = cv.imread('tekapo.bmp')
    
    if src is None:
        print('failed!')
        return
    
    rows = src.shape[0]
    cols = src.shape[1]
    
    mx=0.3
    affine_mat = np.array([[1,mx,0],
                           [0,1,0]]).astype(np.float32)
    
    dst = cv.warpAffine(src, affine_mat, (int(cols+rows*mx),rows))
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.waitKey()
    cv.destroyAllWindows()
    
affine_shear()
# In[10]:
def affine_shear_y():
    src = cv.imread('tekapo.bmp')
    
    if src is None:
        print('failed!')
        return
    
    rows = src.shape[0]
    cols = src.shape[1]
    
    my=0.3
    affine_mat = np.array([[1,0,0],
                           [my,1,0]]).astype(np.float32)
    
    dst = cv.warpAffine(src, affine_mat, (cols, int(rows+my*cols)))
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.waitKey()
    cv.destroyAllWindows()
    
affine_shear_y()

# In[12]:
def mask_setTo():
    src = cv.imread('lenna.bmp', cv.IMREAD_COLOR)
    mask = cv.imread('mask_smile.bmp',cv.IMREAD_GRAYSCALE)
    
    if src is None or mask is None:
        print("failed!")
        return 
    
    src[mask>0]=(0,255,255) # mask영역에서 0이 아닌 부분에 모두 노란색을 src에 칠한다.
    
    cv.imshow('src',src)
    cv.imshow('mask',mask)
    cv.waitKey()
    cv.destroyAllWindows()
    
mask_setTo()
# In[13]:
# 마스크 영상에 의해 지정된 일부 영역만 복사하기
def mask_copyTo():
    src = cv.imread('airplane.bmp', cv.IMREAD_COLOR)
    mask = cv.imread('mask_plane.bmp', cv.IMREAD_GRAYSCALE)
    dst = cv.imread('field.bmp', cv.IMREAD_COLOR)
    
    if src is None or mask is None or dst is None:
        print('Image load failed!')
        return 
    
    dst[mask>0]=src[mask>0]
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.imshow('mask',mask)
    cv.waitKey()
    cv.destroyAllWindows()
mask_copyTo()
# In[16]:
def time_inverse():
    src = cv.imread('lenna.bmp',cv.IMREAD_GRAYSCALE)
    
    if src is None:
        print('Image load failed!')
        return 
    
    dst = np.empty(src.shape, dtype = src.dtype)
    
    tm=cv.TickMeter()
    tm.start()
    
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y,x]=255-src[y,x]
            
    tm.stop()
    print('Image inverse implementation took %4.3f ms' % tm.getTimeMilli())
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.waitKey()
    cv.destroyAllWindows()
# In[17]:
time_inverse()
# In[20]:
def time_inverse_numpy():
    src = cv.imread('lenna.bmp',cv.IMREAD_GRAYSCALE)
    
    if src is None:
        print('Image load failed!')
        return 
    
    dst = np.empty(src.shape, dtype = src.dtype)
    
    tm=cv.TickMeter()
    tm.start()
    
    dst = 255 - src
    
    tm.stop()
    print('Image inverse implementation took %4.3f ms' % tm.getTimeMilli())
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.waitKey()
    cv.destroyAllWindows()
# In[21]:
time_inverse_numpy()
# In[22]:
def time_inverse_numpy2():
    src = cv.imread('lenna.bmp',cv.IMREAD_GRAYSCALE)
    
    if src is None:
        print('Image load failed!')
        return 
    
    dst = np.empty(src.shape, dtype = src.dtype)
    
    tm=cv.TickMeter()
    tm.start()
    
    dst = ~src
    
    tm.stop()
    print('Image inverse implementation took %4.3f ms' % tm.getTimeMilli())
    
    cv.imshow('src',src)
    cv.imshow('dst',dst)
    cv.waitKey()
    cv.destroyAllWindows()
# In[23]:
time_inverse_numpy2()
# In[1]:
import cv2 as cv
import numpy as np
def useful_func():
    img=cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)
    
    if img is None:
        print('Image load failed!')
        return
    
    sum_img = np.sum(img)
    mean_img = np.mean(img, dtype=np.int32)
    print('Sum:',sum_img)
    print('Mean:',mean_img)
    
useful_func()
# In[3]:
def useful_func_with_mask():
    img = cv.imread('lenna.bmp', cv.IMREAD_GRAYSCALE)
    
    if img is None:
        print('Image load failed!')
        d
    
    # Create a binary mask. For demonstration, let's create a mask that selects the central part of the image.
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h//4:3*h//4, w//4:3*w//4] = 255  # Central region set to 255 (white)
    
    # Use cv.mean() with mask to compute the mean of the pixels in the masked region.
    mean_val_masked = cv.mean(img, mask=mask)
    
    print('Mean value without mask:', cv.mean(img)[0])
    print('Mean value with mask:', mean_val_masked[0])
    masked_image = cv.bitwise_and(img, img, mask=mask) 
    # Show the original image and the mask for visualization.
    cv.imshow('Original Image', img)
    cv.imshow('Mask', mask)
    cv.imshow('Mask_image',masked_image)
    cv.waitKey()
    cv.destroyAllWindows()
useful_func_with_mask()

# 엠보싱 필터링
import cv2 as cv
import numpy as np
src = cv.imread('rose.bmp',cv.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed!')
    sys.exit()
    
emboss = np.array([[-1,-1,0],
                   [-1,0,1],
                   [0,1,1]], np.float32)
dst = cv.filter2D(src, -1, emboss, delta=128) 
cv.imshow('src',src)
cv.imshow('dst',dst)
cv.waitKey()
cv.destroyAllWindows()

def blurring_mean():
    src = cv.imread('rose.bmp', cv.IMREAD_GRAYSCALE)
    
    if src is None:
        print('Image load failed!')
        return 
    
    cv.imshow('src',src)
    
    for ksize in range(3,9,2):
        dst = cv.blur(src, (ksize,ksize))
        
        desc = 'Mean : %dx%d' %(ksize,ksize)
        cv.putText(dst, desc, (10,30), cv.FONT_HERSHEY_SIMPLEX,
                  1.0, 255,1, cv.LINE_AA)
        
        cv.imshow('dst',dst)
        cv.waitKey()
        cv.destroyAllWindows()
        
blurring_mean()
# In[5]:
import cv2 as cv
import numpy as np
def calcGrayHist(img):
    channels = [0] #차원
    histSize = [256]
    histRange = [0,256]
    
    hist = cv.calcHist([img],channels, None, histSize, histRange)
    
    return hist
hist=calcGrayHist('cat.bmp')
# In[6]:
import cv2 as cv
import numpy as np
def defGrayHistImage(hist):
    _,histMax, _, _ = cv.minMaxLoc(hist)
    
    imgHist = np.ones((100,256), np.uint8)*255
    for x in range(imgHist.shape[1]):
        pt1 = (x,100)
        pt2 = (x,100-int(hist[x,0]*100/histMax))
        cv.line(imgHist, pt1,pt2,0)
        
        
    return imgHist
defGrayHistImage(hist)
# In[7]:
import matplotlib.pyplot as plt
# Creating a dummy histogram for visualization
dummy_hist = np.zeros((256,1), np.float32)
for i in range(256):
    dummy_hist[i,0] = 1000 * (np.sin(i * np.pi / 256) ** 2)
# Using the provided function to visualize the histogram
hist_image = defGrayHistImage(dummy_hist)
# Displaying the original dummy histogram and its visual representation
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(dummy_hist, color='black')
axs[0].set_title('Original Dummy Histogram')
axs[0].invert_yaxis()
axs[1].imshow(hist_image, cmap='gray')
axs[1].set_title('Visual Representation using defGrayHistImage()')
plt.tight_layout()
plt.show()
# In[10]:
nums=np.array([[[1,4,2],[5,6,7]], [[7,5,3],[8,2,9]]])
print(a.shape, a[1,1,0])
# In[ ]:
