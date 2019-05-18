import random
import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import math
import difflib
import chardet
#计算过程中差值矩阵的变化
diffsequencechange=list()
#记录嵌入水印的位置
Locationx=[]
Locationy=[]
#被嵌入水印
embed=''
#被提取水印
extracted=''
#获取图像一维nampy数组
def img_arry_flatten(image_path):
    '''
    :param imagepath: 图像
    :return:图像像素一维数组
    '''
    # 处理异常
    if os.path.splitext(image_path)[1]=='.bmp':
       im=Image.open(image_path).save('temp.bmp')
    im = Image.open('temp.bmp').convert('L')

    #获得图像的高和宽
    img_heigth=im.size[0]
    img_width=im.size[1]


    # 调用pylab中的属于numpy的array方法将灰度图像转化为二维灰度值的矩阵
    # 矩阵中每一个值都是0~255之间的数
    im_arry = np.array(im)

    # 调用array对象的flatten（）方法将二维灰度的矩阵转化为一维列表（类似数组）
    im_arry_flatten = im_arry.flatten()

    return  im_arry_flatten,img_heigth,img_width
#整数转化为特定长度的二进制字符串
def bin_value(a,length):
    '''
    将整数转化为指定长度的二进制字符串
    :param a: 待转换的整数
    :param length: 指定转化的二进制的长度
    :return: 转换的指定长度的二进制字符串
    '''
    result='{0:b}'.format(a)
    stresult=str(result)
    if len(stresult)>length:
        print('The length of the watermark is larger than %d'%length)
    while len(stresult)<length:
        stresult='0'+stresult
    return stresult
#直方图绘制
def drawhist(imagepath,i):
    '''
    根据像素概率数组输出图像的直方图
    :param imagepath: 图像的路径
    :return:
    '''

    plt.figure("hist of image")
    im_arry_flatten,im_heigth,im_width=img_arry_flatten(imagepath)
    '''
     参数
     arr: 需要计算直方图的一维数组
     bins: 直方图的柱数，可选项，默认为10
     normed: 是否将得到的直方图向量归一化。默认为0
     facecolor: 直方图颜色
     alpha: 透明度
     返回值 ：
     n: 直方图向量，是否归一化由参数设定
     bins: 返回各个bin的区间范围
     patches: 返回每个bin里面包含的数据，是一个list
    '''
    pixel=list(range(0,256))
    plt.hist(im_arry_flatten, bins=256, normed=1, facecolor='black', alpha=0.75)
    plt.xlabel('pixel')
    plt.ylabel('Percentage')
    # plt.show()
    savepath = '/Users/cclin/PycharmProjects/project/hist/' + 'hist' + str(i)+'.jpeg'
    plt.savefig(savepath)
#水印生成
def producewatermaerk(length):
    '''
    :param length: 水印长度
    :return: 限定长度的水印
    '''
    watermark = ""
    for i in range(int(length / 8)):
        watermark = watermark + chr(np.random.randint(ord('a'), ord('z')))
    watermarkfile=open('/Users/cclin/PycharmProjects/project/watermark/watermark1.txt','w')
    watermarkfile.write(watermark)
    return watermark
#图像预处理
def dealimage(image_path):
    #生成图像差值序列
    im_arry_flatten, img_heigth, img_width = img_arry_flatten(image_path)
    difsequence = [0] * img_heigth *int(img_width/2)
    j=0
    for i in range(0,img_width*img_heigth-1,2):
        # if i < 100:
        #   print(im_arry_flatten[i+1],":",im_arry_flatten[i])
        difsequence[j]=int(im_arry_flatten[i+1])-int(im_arry_flatten[i])
        j+=1
    diffsequencechange.append(difsequence)
    #计算每种差值的概率
    count=[0]*256
    rate=[0]*256
    for dif in difsequence:
        count[dif]+=1
        rate[dif]=count[dif]/len(difsequence)
    originrate=rate[:]
    rate.sort()
    # print(originrate)
    # print(rate)
    for min in rate:
        if min!=0:
            minrate=min
            break
    maxrate=max(rate)
    maxembedrate=maxrate
    #生成秘钥
    dmax=originrate.index(maxrate)
    for c in range(0,255):
        if c not in difsequence:
            dmin=c
            break
    tempmax=dmax
    tempmin=dmin
    if dmax<dmin    :
        tempmax=dmin
        tempmin=dmax
    difsequence1=[0]*img_width*int(img_width/2)
    #对差值序列进行处理，dmax,dmin之间的值+1
    for i in range(0,len(difsequence)):
        if difsequence[i]>tempmin and difsequence[i]<tempmax:
            difsequence1[i]=difsequence[i]+1
        else:
            difsequence1[i]=difsequence[i]
    diffsequencechange.append(difsequence1)
    #对图像进行预处理
    #I'(i,2j)=I(i,2j)+1(差值位于dmax,dmin)，
    for i in range(0,img_width*img_heigth-1,2):
        dif=int(im_arry_flatten[i+1])-int(im_arry_flatten[i])
        if dif<tempmax and dif>tempmin:
            im_arry_flatten[i+1]=im_arry_flatten[i+1]+1


    return im_arry_flatten,(dmin,dmax),difsequence1,originrate,(img_heigth,img_width),maxembedrate
#差值矩阵直方图
def histdiff():
    # 生成图像差值序列
  i=0
  for difsequence in diffsequencechange:
    plt.figure("hist of imagedifsequence")
    temp=difsequence[:]
    temp=sorted(temp)
    temp=list(set(temp))
    n, bins, patches = plt.hist(difsequence, bins=len(temp), normed=1, facecolor='g', alpha=0.75)
    arrymean=np.mean(difsequence)
    arrystd=np.std(difsequence)
    y = mlab.normpdf(bins, arrymean, arrystd)
    plt.plot(bins, y, 'b--')
    plt.legend(loc="lower right")
    plt.xlabel('Difference')
    plt.ylabel('Percentage')
    plt.title(r'Histogram of diffsequence: $\mu='+str(arrymean)+r'$, $\sigma='+str(arrystd)+r'$')
    # plt.show()
    savepath = '/Users/cclin/PycharmProjects/project/histdiff/' + 'histdif'+str(i)+'.jpeg'
    plt.savefig(savepath)
    i+=1
# 图像特征提取，即获得图像最大水印嵌入容量
def extractmaxcap(imagepath):
    '''
    图像特征提取，即获得图像最大水印嵌入容量
    :param imagepath: 图像路径
    :return:
    '''
    im_arry_flatten, (dmin, dmax), difsequence1, originrate, (img_heigth, img_width),maxembedrate=dealimage(imagepath)
    return  float(maxembedrate/2)
#嵌入水印
def embedwatermark(image_path,watermark,cap):
    embed=watermark
    print("The embed watermark is",watermark)
    im_arry_flatten,(dmin,dmax),difsubsequence,rate,(img_heigth,img_width),maxembedrate=dealimage(image_path)
    im_arry_flatten1,(dmin,dmax),difsubsequence,rate,(img_heigth,img_width),maxembedrate=dealimage(image_path)
    for i in range(0,len(im_arry_flatten1)):
        im_arry_flatten1[i]=255
    count=0
    for c in difsubsequence:
        if c==dmax:
            count+=1
    ratedmax=count/len(difsubsequence)
    # 把水印的每个字符转化为8位二进制字符串,防止溢出
    wmbinstr = ""
    i = 0
    for c in watermark:
        wmbinstr = wmbinstr + bin_value(ord(c), 8)
        i += 1
    index=0
    difsubsequence2=[0]*int(len(im_arry_flatten)/2)
    i=0
    j=0
    #记录需要嵌入水印但是像素值为255的像素值的位置
    Locationaviod = []
    #处理差值序列,嵌入水印
    if len(watermark)*8 > len(difsubsequence)*ratedmax:
        print("The length of watrermark is larger than the capacity")

    else:
      # 处理差值序列,嵌入水印
      for c in difsubsequence:
        difsubsequence2[i]=c
        if c==dmax and index<len(watermark)*8:
            if wmbinstr[index]=='0':
                difsubsequence2[i]=c
            elif wmbinstr[index]=='1':
                difsubsequence2[i]=c+1
            index+=1
        i+=1

      # 将改变映射到图像，在图像中嵌入水印
      for i in range(0, len(im_arry_flatten)- 1, 2):
        if j>len(wmbinstr)-1:
            break
        dif = int(im_arry_flatten[i + 1]) - int(im_arry_flatten[i])
        if dif == dmax and wmbinstr[j]=='1' and im_arry_flatten[i + 1]!=255:
               im_arry_flatten[i + 1] = im_arry_flatten[i + 1] + 1
               Locationaviod.append(i+1)
               #记录嵌入水印的位置
               Locationy.append((i+1)/512)
               Locationx.append((i+1)%512)
               im_arry_flatten1[i+1]=0
               j+=1
        elif dif==dmax and wmbinstr[j]=='0':
            im_arry_flatten[i + 1] = im_arry_flatten[i + 1]
            j += 1
    diffsequencechange.append(difsubsequence2)
    # 嵌入完所有信息，使用reshpe函数将包含水印的一维列表转化为numpy的二维数组
    im_array_embed = np.reshape(im_arry_flatten, (img_heigth, img_width))
    im_array_locate = np.reshape(im_arry_flatten1, (img_heigth, img_width))
    # 调用Image模块的fromarry方法将二维数组转化为图像保存
    im_embed = Image.fromarray(im_array_embed)
    im_locate=Image.fromarray(im_array_locate)
    path2 = image_path.split('/')[-1]
    savepath = '/Users/cclin/PycharmProjects/project/embed/' +str(cap)+ 'embed_' + path2
    savepath1='/Users/cclin/PycharmProjects/project/locate/' +str(cap)+ 'locate_' + path2
    im_embed.save(savepath)
    im_locate.save(savepath1)
    #嵌入位置
    Location()
    return (dmin,dmax),len(wmbinstr),savepath
#提取水印，恢复图像
def extractwatermark(image_path,dmax,dmin,wmlen):
    # 生成嵌入水印后图像的差值序列
    im_arry_flatten, img_heigth, img_width = img_arry_flatten(image_path)
    difsequence = [0] * img_heigth * int(img_width / 2)
    j = 0
    for i in range(0, img_width * img_heigth - 1, 2):
        difsequence[j] = int(im_arry_flatten[i + 1]) - int(im_arry_flatten[i])
        j+=1
    watermarkbin=""
    watermark=""
    totalwatermark=""
    #提取水印
    index=0
    for d in difsequence:
        if index>wmlen-1:
            break
        if index>0 and (index%8==0) and watermarkbin!="":
            watermark = watermark + chr(int(watermarkbin, 2))
            watermarkbin = ""
        if d==dmax+1:
            watermarkbin=watermarkbin+'1'
            totalwatermark = totalwatermark+'1'
            index+=1
        elif d==dmax:
            watermarkbin = watermarkbin + '0'
            totalwatermark = totalwatermark + '0'
            index+=1
    watermark = watermark + chr(int(watermarkbin, 2))
    print("The extracted watermark is:", watermark)
    extracted=watermark
    subsequence=[0]*int(len(im_arry_flatten)/2)
    #恢复图像
    #对嵌入过程进行恢复
    j=0
    for i in range(0, len(im_arry_flatten) - 1, 2):
        # print("wmbinstr[j]",wmbinstr[j])
        if j > len(totalwatermark)-1 :
            break
        dif = int(im_arry_flatten[i + 1]) - int(im_arry_flatten[i])

        if dif == dmax+1 :
            im_arry_flatten[i + 1] = im_arry_flatten[i + 1] - 1

            j += 1
        elif dif == dmax :
            im_arry_flatten[i + 1] = im_arry_flatten[i + 1]
            j += 1
    j = 0
    for i in range(0, img_width * img_heigth - 1, 2):
        subsequence[j] = int(im_arry_flatten[i + 1]) - int(im_arry_flatten[i])
        j += 1
    diffsequencechange.append(subsequence)
    #对预处理进行恢复
    subsequence1 = [0] * int(len(im_arry_flatten) / 2)
    tempmax = dmax
    tempmin = dmin
    if dmax < dmin:
        tempmax = dmin
        tempmin = dmax
    # I'(i,2j)=I(i,2j)-1(差值位于dmax,dmin)，
    for i in range(0, img_width * img_heigth - 1, 2):
        dif = int(im_arry_flatten[i + 1]) - int(im_arry_flatten[i])
        if dif <= tempmax and dif > tempmin:
            im_arry_flatten[i + 1] = im_arry_flatten[i + 1] - 1
    j = 0
    for i in range(0, img_width * img_heigth - 1, 2):
            subsequence1[j] = int(im_arry_flatten[i + 1]) - int(im_arry_flatten[i])
            j += 1
    diffsequencechange.append(subsequence1)
    # 提取完所有信息，使用reshpe函数将包含水印的一维列表转化为numpy的二维数组
    im_array_extracted = np.reshape(im_arry_flatten, (img_heigth, img_width))
    # 调用Image模块的fromarry方法将二维数组转化为图像保存
    im_extracted = Image.fromarray(im_array_extracted)
    path2 = image_path.split('/')[-1]
    savepath = '/Users/cclin/PycharmProjects/project/image/' + 'recover_' + path2
    #恢复图像的保存
    im_extracted.save(savepath)
    return savepath,watermark
#对比寻找原图像与恢复图像的差异
def comparison(img_path,recoverimg_path):
    orim_arry_flatten, img_heigth, img_width = img_arry_flatten(img_path)
    reim_arry_flatten, img_heigth, img_width = img_arry_flatten(recoverimg_path)

    pixeldifdict={}
    difflag=0
    for i in range(0,img_width*img_heigth):
        if orim_arry_flatten[i]==reim_arry_flatten[i]:
            continue
        else:
            pixeldifdict['index:'+str(i)]=str(int(orim_arry_flatten[i]))+' '+str(int(reim_arry_flatten[i]))
            difflag=1
            continue

    if difflag==0:
        print("There is no difference between original image and recovered image!")
    else:
        print("The difference is below:")
        for key,item in pixeldifdict.items():
            print(key,":",item,end=",")
    spot={}
    difflag1 = 0
    for i in range(0,len(embed)):
        if i>len(extracted):
            print("The embed-water is different with the extracted-watermark")
        if embed[i]==extracted[i]:
            continue
        else:
            difflag1=1
            spot['index:'+str(i)]=str(embed[i])+':'+str(extracted[i])
            continue
    if difflag1==0:
        print("There is no difference between embed watermark and extracted watermark!")

    else:
        print("Difference")
        for key, item in spot.items():
         print(key, ":", item, end=",")
#计算容量
def calculateER(pixelcount,embedcount):
    '''
    计算容量 bit/pixel
    :param pixelcount:
    :param embedcount:
    :return: 容量 单位 bit/pixel
    '''
    return embedcount*1/pixelcount
#计算PSNR
def calculatePSNR(originimg_path,dealimg_path):
    '''
    :param originimg_path: 算法处理前图片
    :param dealimg_path: 算法处理后图片
    :return:
    '''
    ori_image_arry_flatten,imgheigth,imawidth=img_arry_flatten(originimg_path)
    deal_image_arry_flatten,imgheigth1,imawidth1=img_arry_flatten(dealimg_path)

    temp1=0.0
    for index in range(0,len(ori_image_arry_flatten)):
        temp1 = temp1+ (deal_image_arry_flatten[index]-ori_image_arry_flatten[index])**2
    #MSE是原图像与处理图像间的均方误差
    #temp1=len(ori_image_arry_flatten)/2
    MSE=float(temp1/len(ori_image_arry_flatten))

    MAX=255
    if(MSE==0):
        PSNR=10*math.log(MAX*MAX,10)
    else:
      temp=(MAX*MAX)/MSE
      PSNR=10*math.log(temp,10)

    return PSNR
#计算SSIM相关性
def calculateSSIM(image_path1,image_path2):
    im1=cv2.imread(image_path1,0)
    im2=cv2.imread(image_path2,0)
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim
#绘制PSNR_ER_SSIM
def drawPSNR_ER_SSIM(originimg_path,maxcapacity):
    '''
    绘制PSNR~容量曲线
    :param originimg_path:
    :param dealimg_path:
    :return:
    '''

    im_arry_flatten, im_heigth, im_width = img_arry_flatten(originimg_path)
    #嵌入不同长度的水印，使图像具有不同水印容量。
    CAPACITY=[0.0]*19
    for i in range(0,19):
        CAPACITY[i]=float(maxcapacity*i/18)
    dealimg_path=[0]*19
    j=0
    for c in CAPACITY:
        watermark=producewatermaerk(im_heigth*im_width*c)
        (dmin,dmax),wmlen,dealimg_path[j]=embedwatermark(originimg_path,watermark,c)
        j+=1
    #绘制具有不同嵌入水印容量的直方图
    i=0
    for path in dealimg_path:
        drawhist(path,CAPACITY[i])
        i+=1
    #绘制PSNR_ER曲线
    PSNR=[0]*19
    SSIM=[0]*19
    for i in range(0,j):
        PSNR[i]=calculatePSNR(originimg_path,dealimg_path[i])
        SSIM[i]=calculateSSIM(originimg_path,dealimg_path[i])
    fig = plt.figure("PSNR~容量曲线")
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(CAPACITY, PSNR)
    plt.xlabel('CAPACITY')
    plt.ylabel('PSNR')

    fig2 = plt.figure("SSIM~容量曲线")
    ax = fig2.add_subplot(1, 2, 2)
    ax.plot(CAPACITY, SSIM)
    plt.xlabel('CAPACITY')
    plt.ylabel('SSIM')

    savepath='/Users/cclin/PycharmProjects/project/PSNR_ER/psnr_er.jpeg'
    plt.savefig(savepath)

    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    for c, m in [('r', 'o'), ('b', '^')]:
        ax.scatter(CAPACITY, SSIM,PSNR, c=c, marker=m)
    ax.set_xlabel('CAPACITY')
    ax.set_ylabel('SSIM')
    ax.set_zlabel('PSNR')
    savepath = '/Users/cclin/PycharmProjects/project/PSNR_ER/psnr_er_ssim.jpeg'
    plt.savefig(savepath)

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    ax.plot(CAPACITY, SSIM, PSNR, label='parametric curve')
    ax.legend()
    savepath = '/Users/cclin/PycharmProjects/project/PSNR_ER/psnr_er_ssim1.jpeg'
    plt.savefig(savepath)
#对比本算法与传统的基于直方图修改的水印嵌入算法
def campareER(image_path):
    im_arry_flatten,image_height,image_width,=img_arry_flatten(image_path)
    #计算获得图像中频率最大的像素值
    pixel=[0]*256
    rate=[0]*256
    # 获得零值点的个数
    countzero = 0
    for i in range(0,len(im_arry_flatten)):
        pixel[im_arry_flatten[i]]+=1
        if im_arry_flatten[i]==0:
            countzero+=1
    for i in range(0,256):
        rate[i]=pixel[i]/len(im_arry_flatten)
    maxpixel=rate.index(max(rate))
    maxpixelcount=pixel[maxpixel]
    #获得本实验算法Dmax（具有最大概率的差值的个数）/dmaxI(差值的值)
    #dmin
    im_arry_flatten, (dmin, dmax), difsequence1, originrate, (img_heigth, img_width), maxembedrate = dealimage(image_path)
    name=image_path.split('/')[-1].split('.')[0]
    print("测试图像:",name,"原始图像直方图的最大值个数:",maxpixelcount,"原始图像直方图零值点个数:",countzero,
          "差值矩阵的Dmax:",len(difsequence1)*maxembedrate,"dmax:",dmax,"dmin:",dmin)
#嵌入水印位置
def Location():
    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    #for c, m in [('r', 'o'), ('b', '^')]:
    ax.set_title('Embedded position distribution')
    ax.scatter(Locationx, Locationy,marker='x',c='b',cmap='b' )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #ax.set_zlabel('bit plane')
    savepath = '/Users/cclin/PycharmProjects/project/PSNR_ER/scatter.jpeg'
    plt.savefig(savepath)
#椒盐噪声

def PepperandSalt(img_path,percetage):
    '''
    椒盐噪声
    :param img_path: 图像路径
    :param percetage: 百分比
    :return: 椒盐噪声处理后的图片的存储路径
    '''
    noise_img=cv2.imread(img_path,0)
    img_height=noise_img.shape[0]
    img_width=noise_img.shape[1]
    NoiseNum=int(percetage*img_width*img_height)
    for i in range(NoiseNum):
        randX=random.randint(0,img_height-1)
        randY=random.randint(0,img_width-1)
        if random.randint(0,1)<=0.5:
            noise_img[randX,randY]=0
        else:
            noise_img[randX,randY]=255
    path2=img_path.split('/')[-1]
    pepperandsaltpath='/Users/cclin/PycharmProjects/project/pnoise/p_'+path2
    cv2.imwrite(pepperandsaltpath,noise_img)
    # cv2.imshow('pepperandsalt',noise_img)
    cv2.waitKey(0)
    return pepperandsaltpath
#高斯噪声
def GaussianNoise(img_path,means,sigma,percetage):
    '''
    高斯噪声
    :param img_path: 原图像路径
    :param means: 均值
    :param sigma: 标准差
    :param percetage: 百分比
    :return:gaussiansavepath 高斯噪声处理图像的存储路径
    '''
    noise_img=cv2.imread(img_path,0)
    img_height=noise_img.shape[0]
    img_width=noise_img.shape[1]
    NoiseNum = int(percetage * img_height * img_width)
    for i in range(NoiseNum):
        randX = random.randint(0, img_height - 1)
        randY = random.randint(0, img_width - 1)
        noise_img[randX, randY] = noise_img[randX, randY] + random.gauss(means, sigma)
        if noise_img[randX, randY] < 0:
            noise_img[randX, randY] = 0
        elif noise_img[randX, randY] > 255:
            noise_img[randX, randY] = 255
    path2 = img_path.split('/')[-1]
    gaussiansavepath='/Users/cclin/PycharmProjects/project/gnosie/g_'+path2
    cv2.imwrite( gaussiansavepath,noise_img)
    # cv2.imshow('gaussiannoise',noise_img)
    cv2.waitKey(0)
    return gaussiansavepath

def meanfilter(imagepath):
    img=cv2.imread(imagepath,0)
    #中值滤波
    result = cv2.blur(img, (7, 7))
    # result=cv2.GaussianBlur(img, (3, 3), 0)
    path2 = imagepath.split('/')[-1]
    savepath = '/Users/cclin/PycharmProjects/project/meanfilte/m_' + path2
    cv2.imwrite(savepath, result)
    return savepath
#计算误码率
def calculemistackerate(fi1,fi2):
    f1=open(fi1,'r')
    f2=open(fi2,'r')
    w1=f1.read()
    w2=f2.read()
    count=0
    for i in range(0,len(w1)):
        if i>=len(w2):
            break
        if w1[i]!=w2[i]:
            count+=1
    return count/len(w1)


def batchdeal():
     pathlist=[
         #'airplane512.bmp',
    #       'baboon512.bmp',
    #       'beach512.bmp',
    #       'boat1_512.bmp',
    #       'Boats512.bmp',
    #       'colar512.bmp',
    #        'desk512.bmp',
    #       'flowers512.bmp',
    #       'goldhill512.bmp',
          'jg512.bmp',
          'lena512.bmp',
          'peppers2_512.bmp',
          'peppers512.bmp',
          'Sailboat512.bmp',
          'scence512.bmp',
          'timg1_512.bmp',
          'timg2_512.bmp',
          'wheats512.bmp',
          'Zelda512.bmp']

     for path in pathlist:
        originpath='/Users/cclin/PycharmProjects/project/imagelib/'+path
        maxcaprate = extractmaxcap(originpath)
        watermark = producewatermaerk(512 * 512 * maxcaprate)
        (dmin, dmax), wmlen, savepath = embedwatermark(originpath,watermark,maxcaprate)
        # 图像差值矩阵直方图
        histdiff()
        # 提取水印，恢复图像
        resavepath,watermark1 = extractwatermark(savepath, dmax, dmin, wmlen)
        # 对比原图像与恢复后的图像，被嵌入水印与提取水印的差异
        comparison(originpath, resavepath)


if __name__=='__main__':
    originpath="/Users/cclin/PycharmProjects/project/imagelib/baboon512.bmp"
    maxcaprate=extractmaxcap(originpath)
    # #获得图像水印最大嵌入量比率,并根据最大嵌入量生成水印
    watermark=producewatermaerk(512*512*maxcaprate)
    # emwpath='/Users/cclin/PycharmProjects/project/ew/ew'+originpath.split('/')[-1].split('.')[0]+'.txt'
    # dmaxpath='/Users/cclin/PycharmProjects/project/dmax/'+originpath.split('/')[-1].split('.')[0]+'.txt'
    # dminpath='/Users/cclin/PycharmProjects/project/dmin/'+originpath.split('/')[-1].split('.')[0]+'.txt'
    # wmpath='/Users/cclin/PycharmProjects/project/wmlen/'+originpath.split('/')[-1].split('.')[0]+'.txt'
    # f = open(emwpath,'w')
    # f.write(watermark)
    # f.close()
    #预处理图像，嵌入水印
    (dmin, dmax), wmlen, savepath= embedwatermark(originpath,watermark,maxcaprate)
    # f2=open(dmaxpath,'w')
    # f2.write(str(dmax))
    # f2.close()
    # f3=open(dminpath,'w')
    # f3.write(str(dmin))
    # f3.close()
    # f5=open(wmpath,'w')
    # f5.write(str(wmlen))
    # f5.close()
    #图像差值矩阵直方图
    histdiff()
    # # 高斯噪声
    # # savepath = GaussianNoise(savepath, 2, 4, 0.2)
    # #savepath=PepperandSalt(savepath,0.2)
    # # savepath=meanfilter(savepath)
    # emwpath='/Users/cclin/PycharmProjects/project/ew/ew'+originpath.split('/')[-1].split('.')[0]+'.txt'
    # # savepath=
    # # 提取水印，恢复图像
    # f2 = open(dmaxpath, 'r')
    # dmax=int(f2.read())
    # f2.close()
    # f3 = open(dminpath, 'r')
    # dmin=int(f3.read())
    # f3.close()
    # f5 = open(wmpath,'r')
    # wmlen=int(f5.read())
    # f5.close()
    # #savepath='/Users/cclin/PycharmProjects/project/copy/boat1_512.bmp'
    resavepath,watermark1=extractwatermark(savepath,dmax,dmin,wmlen)
    # etwpath = '/Users/cclin/PycharmProjects/project/et/et' + originpath.split('/')[-1].split('.')[0] + '.txt'
    # f = open(etwpath, 'w')
    # f.write(watermark1)
    # f.close()
    #对比原图像与恢复后的图像，被嵌入水印与提取水印的差异
    comparison(originpath,resavepath)
    # f=open('/Users/cclin/PycharmProjects/project/subsequece.txt','w')
    # f.write(diffsequencechange)
    # for dif in diffsequencechange:
    #       print(diffsequencechange,end="\n\n")
    histdiff()
    PSNR=calculatePSNR(originpath,savepath)
    #计算嵌入量
    cap=extractmaxcap(originpath)
    print("嵌入量",cap)
    print("PSNR",PSNR)
    campareER(originpath)
    # drawPSNR_ER_SSIM(originpath,cap)
    ssim=calculateSSIM(originpath,savepath)
    print("SSIM",ssim)
    # drawhist('/Users/cclin/PycharmProjects/project/image/lena512.bmp',1)
    # drawhist('/Users/cclin/PycharmProjects/project/embed/embed_lena512.bmp',2)
    # drawhist('/Users/cclin/PycharmProjects/project/image/recover_embed_lena512.bmp',3)
    # batchdeal()
    # path="/Users/cclin/PycharmProjects/project/embed/0.0embed_peppers512.bmp"
    # path2=GaussianNoise(path,2,4,0.2)
    # comparison(path2)
