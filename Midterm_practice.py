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


# # 브로드캐스팅 규칙
# 
# ## 1. 두 배열의 차원 수가 다르면 더 작은 수의 차원을 가진 배열 형상의 앞쪽(왼쪽)을 1로 채운다
# ## 2. 두 배열의 형상이 어떤 차원에서도 일치하지 않는다면 해당 차원의 형상이 1인 배열이 다른 형상과 일치하도록 늘어난다
# ## 3. 임의의 차원에서 크기가 일치하지 않고 1도 아니라면 오류가 발생한다

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


# # OpenCV and Matrix 03

# # 저장되어 있는 영상데이터를 파일로 저장하기 위해서는 imwrite()함수를 사용함
# ## bool imwrite(filename, img,params)

# # 이미지 타입 확인
# ## .bmp 로드하기 -흑백으로
# ## img1 의 type? - numpy.ndarray
# ## img1의 shape?
# ## Shape의 len으로 타입 확인 가능

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


# # OpenCV 주요 기능
# ## 카메라와 동영상 파일 다루기
# ## 다양한 그리기 함수
# ## 이벤트 처리 
# ## OpenCV 데이터 파일 입출력
# ## 유용한 OpenCV 기능
# 
# ## 동영상의 처리
# ## 동영상 : 일련의 정지 영상을 압축하여 파일로 저장한 형태
# ### 프레임 : 저장되어 있는 일련의 정지 영상
# 
# ## 동영상을 처리하는 작업 순서
# ### 프레임 추출 
# ### 각각의 프레임에 영상 처리 기법을 적용
# 
# ### 컴퓨터에 연결된 카메라 장치를 사용하는 작업도 카메라로부터 일정시간 간격으로 정지 영상 프레임을 받아와서 처리하느 형태
# 
# ### 카메라와 동영상 파일을 다루는 작업은 연속적인 프레임 영상을 받아와서 처리한다는 공통점이 있음
# 
# ## OpenCV 에서는 VideoCapture라는 하나의 클래스를 이용
# ### 카메라 또는 동영상 파일로부터 정지 영상 프레임을 받아올 수 있음
# 
# ## VideoCapture 클래스
# 
# ## VideoCapture(동영상 파일이름, 사용할 비디오 캡쳐 API 백엔드, 성공시True 실패시 false)
# ### 동영상 파일을 불러오기 위해서 VideoCapture 객체를 생성할 때 생성자에 동영상 파일 이름을 지정
# ### python에서는 open을 별도 호출 안해도되지만, isOpened()호출을 통해 제대로 파일을 불러왔는지 확인 필요
# ### cap.get(...) : 현재 열러 있는 카메라 장치 또는 동영상 파일로부터 여러가지 정보를 받아 오기 위해 사용
# 

# In[142]:


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()
    
print('Frame width:',int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
print('Frame height:',int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))


# ## waitKey(int) : 키 입력을 받을 때까지 대기할 시간을 지정
# 
# ## 동영상 파일은 초당 프레임 수FPS(Frames per second) 값을 가지고 있음
# ## FPS값을 고려하지 않을 경우 동영상이 너무 빠르거나 느리게 재생되는 경우가 발생
# ## FPS값을 확인하기 위해서는 cap.get(cv2.CAP_PROP_FPS) 활용
# 
# ## 동영상 파일의 FPS 값을 이용하면 매 프레임 사이의 시간 간격을 계산 가능
# 
# ### delay = round(1000 / fps) 1000ms=1s
# #### ex) 초당 30 프레임을 재생할 경우 delay는 33ms 이며, 각 프레임을 33ms 간격으로 출력해야함
# 
# #### 위에서 계산한 delay 값을 활용하여 추후 waitKey()함수의 인자로 사용

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


# # 다양한 그리기 함수
# 
# ## 직선 그리기
# ### line()함수를 사용하여 영상위에 직선을 그릴 수 있다
# 
# ### img = np.full((400,400,3), 255, np.uint8) => img는 numpy array
# 
# ### Line Types
# ### FILLED -1 내부를 채움 (직선 그리기 함수에는 사용 불가)
# ### LINE_4 4      4방향 연결
# ### LINE_8 4      8방향 연결
# ### LINE_AA 4   안티에일리어싱(anti-aliasing) -> 빈공간을 채워줌
# 
# 
# ## 화살표 형태의 직선을 그려야하는 경우 arrowedLine()함수를 사용
# 
# ### 인자 : 입출력 영상, 시작점, 끝점, 선 색상, 선 두께, 선타입(4,8,AA)중 하나, shift(그리기 좌표값 축소 비율), tipLength 전체 직선길이에 대한 화살표 길이의 비율
#  
# ## 직선 그리기 
# ### 마커를 그리는 경우 drawMarker()함수 사용

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


# ## 기하학적 변환(투시변환, 어파인 변환)
# 
# ### 영상의 기하학적 변환은 영상을 구성하는 픽셀의 배치 구조를 변경함으로써 전체 영상의 모양을 바꾸는 작업을 한다.
# 
# 
# # 투시 변환 
# 
# ## 널리 사용되는 영상의 기하학적 변환 중에는 어파인 변환보다 더 자유도가 높은 투시 변환이 있다. 투시 변환은 직사각형 형태의 영상을 임의의 "볼록 사각형 형태"로 변경할 수 있는 변환
# 
# ## 투시 변환에 의해 원본영상에 있던 직선은 결과 영상에서 그대로 직선성이 유지되지만, 두 직선의 평행 관계는 깨어질 수 있음
# ## 점 네개의 이동 관계에 의해 결정되는 투시 변환
# ## 투시 변환은 직선의 평행 관계가 유지되지 않기 때문에 결과 영상의 형태가 임의의 사각형으로 나타나게 됨
# ## 점 하나의 이동 관계로 부터 x 좌표에 대한 방정식 하나와 y 좌표에 대한 방정식 하나를 얻을 수 있으므로, 점 네개의 이동관계로부터 여덟개의 방정식을 얻을 수 있음
# ## 여덟 개의 방정식으로부터 투시 변환을 표현하는 파라미터 정보를 계산
# ## 투시 변환은 보통 3x3크기의 실수 행렬로 표현
# ## 투시 변환은 여덟개의 파라미터로 표현할 수 있지만, 좌표 계산의 편의상 아홉개의 원소를 갖는 3x3 행렬을 사용
# ### 행렬 수식에서 입력 좌표와 출력 좌표를 (x,y,1), (wx',wy',w)형태로 표현한 것을 동차 좌표계(homogeneous coordinates)라고 함
# ### -좌표 계산의 편의를 위해 사용하는 방식
# ### w는 결과 영상의 좌표를 표현할 때 사용되는 비례 상수
# #### -다음과 같은 형태로 계산 됨
# #### w=p31x+p32y+p33
# #### x'과 y'은 다음과 같이 구할 수 있음
# 
# #### x' = p11x+p12y+p13/w   y' = p21x+p22y+p23/w
# 
# ## 투시 변환 행렬을 구하는 함수와 실제 영상을 투시변환하는 함수를 모두 제공한다.
# 
# ## getPerspectiveTransform() : 투시 변환 행렬을 구하는 함수
# ### 입력 영상에서 네 점의 좌표와 이 점들이 이동한 결과 영상의 좌표 네 개를 입력으로 받아 3x3 투시 변환 행렬을 계산
# 
# 
# ## warpPerspective(src, dst, M, dsize, flags, boorderMode, borderValue);
# 
# ### src : 입력영상, dst : 결과 영상, M: 3x3 투시 변환 행렬, dsize : 결과 영상의 크기, flags : 보간법 알고리즘,
# ### borderMode : 가장자리 픽셀 확장 방식. BorderTypes 열거형 상수 중 하나를 지정함. 만약 BORDER_TRANSPARENT를 지정하면 입력 영상의 픽셀 값이 복사되지 않는 영역은 dst 픽셀 값을 그대로 유지한다.

# In[12]:


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


# ## (투시변환 ) 결과 해석
# 
# ### perspective 프로그램이 처음 실행되면 일단 src창만 화면에 나타난다.
# ### 이때 마우스를 이용한 좌표 선택은 카드의 좌측 상단 모서리 점부터 시작하여 시게방향 순서로 선택해야한다.
# ### 마우스로 클릭한 위치는 빨간색 원을 그려서 표시한다. 일반적인 카드의 가로 대 세로 크기 비율이 2:3이기 때문에 dst 창에 나타날 영상의 크기를 200x300으로 설정한다.
# ### 그 결과 dst 창에 다이아 K카드가 200x300 크기로 반듯하게 투시 변환되어 나타나는 것을 확인 할 수 있다.
# ### 참고로 3x3 투시 변환 행렬을 가지고 있을 때, 일부 점들이 투시 변환에 의해 어느 위치로 이동할 것인지 알고 싶다면         perspectiveTransform(src, dst, m)함수를 사용할 수 있다.
# ### src: 입력행렬, dst : 출력행렬, m : 변환 행렬 3x3또는 4x4

# ### 영상의 밝기 및 명암비 조절, 필터링 등의 변환=>픽셀위치 고정, 픽셀 값 변경!
# 
# ### 기하학전 변환(어파인 변환, 투시변환)=> 픽셀의 값 고정, 픽셀 위치를 변경!
# 
# # 어파인 변환 (이동변환,전단변환,크기변환,회전변환, 대칭변환)
# 
# ## 직선은 그대로 직선으로 나타나고(투시변환도 직선성은 유지됨), 직선간의 길이 비율과 평행관계가 그대로 유지됨(투시변환과의 차이점)
# 
# ## 직사각형 형태의 영상은 어파인 변환에 의해 "평행사변형"에 해당하는 모양으로 변경됨
# 
# 
# ## 여섯개의 파라미터를 이용한 수식으로 정의
# 
# ## 입력영상의 좌표를 나타내는 행렬 (x,y) 앞에 2x2 행렬 (ab cd)를 곱하고 그뒤에 2x1 행렬 (cxf)를 더하는 형태로 어파인 변환을 표현한다.
# 
# ## 수학적 편의를 위해 입력 영상의 좌표 (x,y)에 가상의 좌표 1을 하나 추가하여 (x,y,1)형태로 바꾸면, 앞 행렬 수식을 행렬 곱셈 형태로 바꿀 수 있다.
# 
# ## 어파인 변환 행렬 
# ### 앞 수식에서 여섯개의 파라미터로 구성된 2x3 행렬을 어파인 변환 행렬이라고 부름.
# ### 어파인 변환은 2x3 실수형 행렬 하나로 표현할 수 있음
# 
# ## 입력 영상과 어파인 변환 결과 영상으로부터 어파인 변환 행렬을 구하기 위해서는 최소 세점의 이동 관계를 알아야한다.
# 
# ## 점하나의 이동관계로부터 x좌표와 y좌표에 대한 변환 수식 두개를 얻을 수 있으므로, 점 세개의 이동 관계로부터 총 여섯개의 방정식을 구할 수 있음
# ## 점세개의 이동 관계를 알고 있다면 여섯개의 원소로 정의되는 어파인 변환행렬을 구할 수 있다.
# 
# ### 점세개만 있으면 되는 이유 : 직사각형 영상은 평행사변형 형태로 변환될 수 있기 떄문에 입력 영상의 남은 한 모서리 점은 자동을 결정될 수 있다.
# 
# ### getAffineTransform(src,dst) : 어파인 변환 행렬을 구하는 함수
# ### 입력 영상에서 세 점의 좌표와 이점들이 이동한 결과 영상의 좌표 세개를 입력으로 받아 2x3 어파인 변환 행렬을 계산
# ### warpAffine(src,dst,M,dsize, flags,borderMode, borderValue)
# ### src, dst, M 2x3어파인 변환 행렬, dsize: 결과 영상의 크기, flags : 보간법 알고리즘, borderMode : 가장자리 픽셀 확장 방식. BorderTypes 열거형 상수

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
# 결과 영상에서 호수 영상이 (150,100) 좌표부터 나타나는 것을 확인할 수 있고, 입력 영상의 픽셀 값이 복사되지 않은 영역은 검은 색으로
# 채워진 것을 확인할 수 있음


# In[7]:


affine_translation()


# # 전단변환 (shear Transformation)
# ## 직사각형 형태의 영상을 한쪽 방향으로 밀어서 평행사변형 모양으로 변형 (층밀림 변환)
# ### 가로 방향 또는 세로 방향으로 각각 정의할 수 있음
# ### 픽셀이 어느 위치에 있는가에 따라 이동 정도가 달라짐
# 
# ## 입력 영상에서 원점은 전단 변환에 의해 이동하지 않고 그대로 원점에 머물러 있음
# 
# ### y 좌표가 증가함에 따라 영상을 가로 방향으로 조금씩 밀어서 만드는 전단 변화 수식
# # x = 1 mx  x  + 0
# # y    0  1    y  + 0
# 
# ### x좌표가 증가함에 따라 영상을 세로 방향으로 조금씩 밀어서 만드는 전단 변환 수식
# # x' = x 
# # y' =myx + y 

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


# # 크기 변환 
# 
# # 보간법 : 결과 영상의 픽셀 값을 결정하기 위해, 입력 영상에서 주변 픽셀 값을 이용하는 방식을 의미
# 
# 
# ## resize() 함수 : 여섯번째 인자 interpolation 보간법 알고리즘 나타내는 interpolationFlags 열거거형 상수 지정, 양선형 보간법, 3차보간법 등..
# 
# ## 픽셀영역 리셈플릭 INTER_AREA => 축소시사용/ 무아게현상 방지 화질 면에서 유리

# # 회전 변환 (Rotation Transformation)
# ## 특정 좌표를 기준으로 영상을 원하는 각도만큼 회전하는 변환

# # ROI 마스크 연산
# 
# ## 임의의 모양을 갖는 ROI(Region - of - interest) 설정을 위해 일부 행렬 연산 함수에 대하여 마스크 연산을 지원함
# ## 입력 영상과 크기가 같고 깊이가 cv_8u인 마스크 영상을 함께 인자로 전달 받음
# 
# ## 마스크 영상이 주어질 경우, 마스크 영상의 픽셀값이 0이 아닌 좌표에 대해서만 연산이 수행됨.
# ### 일반적으로 사람의 눈으로도 구분이 쉽도록 픽셀값이 0 또는 255로 구성된 흑백 영상을 사용

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


# # 필터링
# ## 블러링 : 영상 부드럽게 하기
# 
# ## 샤프닝 : 영상 날카롭게 하기
# 
# ## 잡음 제거 필터링
# 
# # 필터링
# ### 영상에서 원하는 정보만 통과시키고 원치 않는 정보는 걸러내는 작업
# ### ex) 영상에서 지저분한 잡음을 걸러내어 영상을 깔끔하게 만드는 필터
# ### ex) 영상에서 부드러운 느낌의 성분을 제거하여 좀 더 날카로운 느낌이 나도록 변형
# 
# ## 영상의 필터링은 보통 마스크(mask)라고 부르는 작은 크기의 행렬을 이용
# 
# # 마스크
# ## 필터링의 성격을 정의하는 행렬
# ## 커널, 윈도우라고도 부름
# 
# ## 경우에 따라서 마스크 자체를 필터라고 부르기도함
# ### 다양한 크기와 모양으로 정의가능
# ### 행렬의 원소는 보통 실수로 구성
# 
# ## 필터링 연산 방법
# ### 1x3 or 3x1 형태의 직사각형 행렬을 사용하기도 하고 , 3x3,5x5등의 정방행렬을 사용하기도 함
# ### 필요시 십자가 모양의 마스크 사용가능
# ### 3x3 정방형 행렬이 다양한 필터링 연산에 널리 쓰임
# 
# ## 그림에 표시한 다양한 필터마스크에서 진한 색으로 표시한 위치는 고정점(anchor point)라고한다
# ## 고정점은 현재 필터링 작업을 수행하고 있는 기준 픽셀 위치를 나타내고, 대부분의 경우 마스크 행렬 정중앙을 고정점으로 사용한다.
# 
# ## 연산의 결과는 마스크 행렬의 모양과 원소 값에 의해 결정
# ### 마스크행렬의 정의에 따라 영상을 부드럽게 또는 날카롭게 만들 수 있음
# ### 잡음을 제거하거나 에지 성분만 나타나도록 만들수도 있음
# 
# ### 마스크 이용한 필터링은 입력 영상의 모든 픽셀 위로 마스크 행렬을 이동시키면서 마스크 연산을 수행하는 방식으로 처리
# ### 마스크 행렬의 모든 원소에 대하여 마스크 행렬 원소값과 같은 위치에 있는 입력 영상 픽셀 값을 서로 곱한 후, 그 결과를 모두 더하는 연산
# 
# ### 마스크 연산의 결과를 출력 영상에서 고정점 위치에 대응되는 픽셀 값으로 설정
# 
# ## 가장자리 픽셀의 처리
# 
# ### 가장자리 픽셀의 처리란 가장 왼쪽 또는 오른쪽 열, 가장 위쪽 또는 아래쪽 행에 있는 픽셀을 의미
# ### 예를 들어 (x,y) = (0,1)위치에서 3x3 크기의 마스크 연산을 수행하는 경우, x=-1인 위치에서의 픽셀 값, 즉 f(-1,0~2) 세 픽셀은 실제 영상에 존재하지 않음
# ### 영상의 가장자리 픽셀에 대해 필터링을 수행할 때에는 특별한 처리를 필요로 함
# 
# ### opencv는 영상의 필터링을 수행할 때, 영상의 가장자리 픽셀을 확장하여 영상 바깥쪽에 가상의 픽셀을 만듦
# ### 영상의 바깥쪽 가상의 픽셀 값을 어떻게 설정하는가에 따라 필터링 연산 결과가 달라짐
# ### 그림은 입력영상의 좌측 상단 부분을 확대하여 나타낸 것으로 각각의 사각형은 픽셀을 표현함
# 
# ### 가장자리 픽셀이 처리
# ### 실선으로 그려진 노란색 픽셀은 영상에에 실제 존재하는 픽셀이고, 점선으로 표현된 바깥쪽 분홍색 픽셀은 필터링 연산을 수행함
# ### 이 그림에서는 5x5 크기의 필터 마스크를 사용하는 필터링을 고려하여 영상 바깥쪽에 두 개씩의 가상 픽셀을 표현
# 
# ### 각각의 픽셀에 쓰여진 영문자는 픽셀 값을 나타내며, 가상의 픽셀 위치에는 값이 대칭 형태로 나타나도록 설정되어 있음
# ### Opencv는 이러한 가장자리 픽셀 확장 방법을 이용하여 영상의 가장자리 픽셀에 대해서도 문제없이 필터링 연산을 수행함
# ### 가장자리 픽셀의 처리
# ### 표에 나타난 상수는 BorderTypes라는 이름의 열거형 상수 중 일부
# ### BORDER_CONSTANT 0 0 0 a b c d e f g h 0 0 0
# ### BORDER_REPLICATE a a a a b c d e f g h h h h
# ### BORDER_REFLECT    c d a a b d d e f g h h g f
# ### BORDER_REFLECT    d c d a b c d e f g h g f e
# 
# 
# # 필터링 연산
# ### Opencv에서 필터 마스크를 사용하는 일반적인 필터링은 filter2D()함수를 이용하여 수행
# ### filter2D(src,dst,ddepth,kernel, anchor, delta, borderType);
# ### src, dst, ddepth : 결과 영상의 깊이, kernel : 필터링 커널, 1채널 실수형 행렬, anchor 고정점 좌표, Point(-1,-1)을 지정하면 커널 중심을 고정점으로 사용한다, delta : 필터링 연산 후 추가적으로 더할 값, borderType : 가장자리 픽셀 확장 방식
# 
# ### filter2D() 함수는 src 영상에 kernel 필터를 이용하여 필터링을 수행하고, 그 결과를 dst에 저장함
# ### 만약 src 인자와 dst 인자에 같은 변수를 지정하면 필터링 결과를 입력 영상에 덮어쓰게 됨
# 
# ### filter2D() 함수 인자 중에서 ddepth는 결과 영상의 깊이를 지정하는 용도로 사용
# ### ddepth에 -1을 지정하면 출력 영상의 깊이는 입력 영상과 같게 설정됨
# 
# 
# ## 엠보싱 필터링 
# ### 엠보싱 필터는 입력 영상을 엠보싱 느낌이 나도록 변환
# ### 픽셀 값 변화가 적은 평탄한 영역은 회색으로 설정
# ### 객체의 경계부분=> 밝거나 어둡게 처리하면 엠보싱 느낌이 남
# ### 필터링을 수행하면 대각선 방향으로 픽셀 값이 급격하게 변하는 부분에서 결과 영상 픽셀 값이 0보다 휠씬 크거나 또는 0보다 훨씬 작은 값을 가지게 됨
# ### 입력 영상에서 픽셀 값이 크게 바뀌지 않는 평탄한 영역에서는 결과 영상의 픽셀 값이 0에 가까운 값을 가지게 됨
# ### 구한 결과 영상을 그대로 화면에 나타내면 음수 값은 모두 포화 연산에 의해 0이 되어버리기 때문에 입체감이 크게 줄어 들게 됨
# ### 따라서 엠보싱 필터를 구현할 때에는 결과 영상에 128을 더하는 것이 좋음

# In[5]:


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
# 대각선 방향으로 픽셀 값이 급격하게 변하는 부분에서 결과 영상 픽셀값이 0보다 휠씬 크거나 또는 0보다 휠씬 작은 값을 가지게 된다
# 입력 영상에서는 픽셀값이 크게 바뀌지 않는 평탄한 영역에서는 결과영상의 픽셀 값이 0에 가까운 값을 가지게 됨
# 이렇게 구한 결과 영상을 그대로 화면에 나타내면 음수값은 모두 포화연산에 의해 0이 되어 버리기 때문에 입체감이 크게 줄어 들게 됨
# 따라서 엠보싱 구현할 때는 결과영상에 128을 더해 출력

cv.imshow('src',src)
cv.imshow('dst',dst)

cv.waitKey()
cv.destroyAllWindows()


# # 블러링
# ## 마치 초점이 맞지 않은 사진처럼 영상을 부드럽게 만드는 필터링 기법 
# ## 스무딩이라고도 함
# 
# ## 영상에서 인접한 픽셀간의 픽셀 값 변화가 크지 않은 경우 부드럽운 느낌을 받을 수 있음
# ### 블러링은 거친 느낌의 입력 영상을 부드럽게 만드는 용도로 사용되기도 하고, 혹은 입력 영상에 존재하는 잡음의 영향을 제거하는 전처리 과정으로도 사용됨
# 
# ## 평균값 필터
# ### 입력 영상에서 특정 픽셀과 주변 픽셀들의 산술 평균을 결과 영상 픽셀 값으로 설정하는 필터
# ### 평균값 필터에 의해 생성되는 결과 영상은 픽셀 값의 급격한 변화가 줄어들어 날카로운 에지가 무디어지고 잡음의 영향이 크게 사라지는 효과가 있음
# ### 너무 과도하게 사용할 경우 사물의 경계가 흐릿해지고 사물의 구분이 어려워 질 수 있음.
# ### 각각의 행렬은 모두 원소값이 1로 설정되어 있고, 행렬의 전체 원소개수로 각 행렬 원소값을 나누는 형태로 표현
# ### 평균값 필터는 마스크의 크기가 커지면 커질수록 더욱 부드러운 느낌의 결과 영상을 생성하며, 그 대신 연산량이 크게 증가할 수 있음
# ### blur (src, dst,  ksize, anchor, bordertype) 함수 이용하여 평균 값 필터링 수행 가능
# ### ksize : 블러링 커널 크기 , bordertype : 가장자리 픽셀 확장 방식

# In[7]:


#blurring_mean()함수는 3x3,5x5,7x7크기의 평균값 필터를 이용하여 rose.bmp 영상을 부드럽게 변환하고 그 결과를 화면에 출력
#평균값 필터의 크기가 커질 수 록 영상이 더욱 부드럽게 변경됨

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


# # 가우시안 분포
# ### 가우시안 분포는 평균을 중심으로 좌우 대칭의 종 모양을 갖는 확률 분포를 말하며 정규분포라고도 함
# ### 평균과 표준편차에 따라 분포 모양이 결정, 주로 평균 0을 사용
# ### 표준 편차가 작으면 그래프가 뾰족, 표준편차가 크편 완만
# ### 가우시안 분포 함수 값은 특정 x가 발생할 수 있는 확률
# ### 그래프 아래 면적을 모두 더하면 1이됨
# 
# ## 2차원 가우시안 분포함수는 1차우너 가우시안 분포함수의 곱으로 분리가 가능하고 이를 통해 가우시안 필터 연산량을 크게 줄일수 있다
# ## x축과 y축의 방향으로 1차원 가우시안 분포 함수의 곱으로 분리됨
# ### 입력 영상을 x축의 방향으로의 함수, y축 방향으로의 함수로 각각 1차원 마스크 연산을 수행하여 필터링 결과 영상을 얻을 수 있음
# ### 1차원 가우시안 함수로 부터 1x9 가우시안 마스크 행렬 g를 구하고 필터링을 수행한뒤, g의 전치행렬을 이용해서 필터링을 한번더 수행하면 2차원 가우시안 필터 마스크로 한번 필터링 한것과 같은 결과를 얻을 수 있다.
# ## 픽셀 하나당 곱셈 연산 횟수가 81번에서 18번으로 크게 감소한다.
# 

# # 히트토그램 구하기
# ## 흰색으로 초기화 된 256x100 크기로 영상 imgHist를 생성
# ## for문과 line()함수를 이용하여 그래프를 그림
# ### 히스토그램 영상 imgHist를 반환

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

a = ?

print(a.shape, a[1,1,0])


# In[ ]:




