import numpy as np
import cv2
import os
import sys
import re
import random
import pdb
from argparse import ArgumentParser
from PIL import Image
import pickle

#二次元正規分布の確率密度を返す関数
def gaussian(x,sigma,mu):
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = x.ndim
    tmp = (x - mu).dot(inv)
    diag = np.sum(tmp*(x - mu),axis=1)
    return np.exp(-diag/2.0) / (np.sqrt((2 * np.pi) ** n * det))

def parse_args():
    parser = ArgumentParser(description='Generate toy data for Inpainting')
    parser.add_argument('-dspath','--dataSetPath',type=str,default="ToyData",help='Name of dataSet (default=ToyData)')
    parser.add_argument('-loadMask','--loadMaskPath',default="",help='using exist Mask Image, you have to set the path. (default="" (generate new mask))')
    # parser.add_argument('-loadTrain','--loadTrainPath',default="",help="他データセットの画像をtrainに使用")
    # parser.add_argument('-loadValid','--loadValidPath',default="",help="他データセットの画像をvalidに使用")
    # parser.add_argument('-loadTest','--loadTestPath',default="",help="他データセットの画像をtestに使用")
    # parser.add_argument('-loadSite','--loadSitePath',type=str,default="",help="")
    parser.add_argument('-train', '--train',type=int, default=1500,help='number of training images (default=1500)')
    parser.add_argument('-valid', '--valid',type=int, default=100,help='number of validation images (default=100)')
    parser.add_argument('-test', '--test',type=int, default=100,help='number of test images (default=100)')
    parser.add_argument('-masktype', '--masktype',type=str, default='same', help='generate same mask images or vary mask images (default=same)', choices=['same', 'vary'])
    parser.add_argument('-ratio','--maskratio',type=float,default=0.99,help='ratio of mask (default=0.99)')
    parser.add_argument('-posVariant','--positionVariant',action='store_true',help="Flag for making data position variant")
    parser.add_argument('-mixed','--isMixedGaussian',action='store_true',help="Flag for making data mixed-gaussian (if you use this, -posVariant is not available)")
    parser.add_argument('-noise','--isNoised',action='store_true',help="Flag for using noised data")
    parser.add_argument('-filter','--filterType',type=str,default="none",help="フィルターのタイプ(none,rect,gaus)")
    parser.add_argument('-loadFilter','--loadFilterPath',type=str,default="",help='')

    return  parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # shape = (512,512,1) # 画像のサイズ
    shape = (256,256,1)
    # shape = (128,128,1)
    keys = ["train","valid","test"]
    dsPath = "data" + os.sep + args.dataSetPath # データセットのパス
    if not os.path.isdir(dsPath):
        os.makedirs(dsPath)

    dataNum = {keys[0]:args.train,keys[1]:args.valid,keys[2]:args.test} # train,validation,testでの生成数 
    imagePath = [dsPath + os.sep + k + ".pickle" for k in keys]

    if args.loadFilterPath!="":
        # ext = args.loadFilterPath.split(".")[-1]
        filt = np.array(Image.open(args.loadFilterPath))/255
        

    # 指定されたマスク画像およびマスク画像のpickleデータをロードする
    if args.loadMaskPath!="":
        ext = args.loadMaskPath.split(".")[-1]
        if ext == "png" or ext == "jpg":
            existMask = np.array(Image.open(args.loadMaskPath))/255
        elif ext == "pickle" or ext == "pkl":
            existMask = pickle.load(open(args.loadMaskPath,"rb"))[0]

        # reshape
        print("Given mask image is reshaped.")
        if existMask.shape[0]!=shape[0] or existMask.shape[1]!=shape[1]:
            existMask = cv2.threshold(cv2.resize(existMask*255,(shape[0],shape[1])),127,255,cv2.THRESH_BINARY)[1]/255
        existMask = np.reshape(existMask,shape)


    # 混合分布と位置依存は同時に使えない
    if args.isMixedGaussian and args.positionVariant:
        print("you can't use option -mixed and -posVariant at the same time")
        sys.exit()

    if args.isMixedGaussian:
        pointNum = [3,4] # ガウス分布の数 ([2,3]なら2~3個のガウス分布による混合ガウスになる)
    else:
        pointNum = [1,1]

    if args.filterType=="rect":
        filt = np.ones(shape)
        filt[100:340,80:200,0] = 0
    elif args.filterType=="gaus":
        pass

    #================================================================================================
    # マスク画像の作成
    maskPath = [dsPath + os.sep+k+"_mask.pickle" for k in keys]
    maskRatio = args.maskratio # マスク部分の割合
    masktype = args.masktype

    def random_point_nodup(num): # 重複なしで座標を生成
        res = []
        while len(res) < num:
            x = random.randint(0,shape[0]-1)
            y = random.randint(0,shape[1]-1)
            dupflag = False # 重複フラグ
            for p in res:
                if p[0]==x and p[1]==y:
                    dupflag = True
                    break
            if not dupflag:
                res.append([x,y])

        return res
 
    obsNum = int(shape[0]*shape[1]*(1-maskRatio))
    if args.loadMaskPath == "": # 指定されたマスクがない場合
        if masktype == "same":
            mask = np.zeros(shape)
            # 観測点を生成し,マスクの要素を1 (画像として保存するので255) にする
            obs = random_point_nodup(obsNum)
            for p in obs:
                mask[p[0],p[1]] = 1
            mask = mask.astype("uint8")

    for path,key in zip(maskPath,keys):
        masks = []
        print("generate "+key+"_mask data")
        for i in range(dataNum[key]):
            print("\r progress:{0}/{1}".format(i+1,dataNum[key]),end="")
            if masktype=="same":
                if args.loadMaskPath !="": # 指定されたマスクがある場合
                    masks.append(existMask)
                else:
                    masks.append(mask)
            else:
                mask = np.zeros(shape)
                obs = random_point_nodup(obsNum)
                for p in obs:
                    mask[p[0],p[1]] = 1
                masks.append(mask.astype("uint8"))
            
        masks = np.array(masks)
        pickle.dump(masks,open(path,"wb"))
        print("")

    #=============================================================
    # 画像の作成
    # XY座標を生成
    x = np.arange(0, shape[0], 1)
    y = np.arange(0, shape[1], 1)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(),Y.ravel()]

    range_shape = np.array([shape[0],shape[1]])

    def sample_mu_sigma(mu): # 平均と分散のランダムサンプリング
        # mu = np.random.rand(2)*range_shape

        if args.positionVariant:
            delRange = [-25,25] # 作らない範囲（縦方向）
            centered_mu = mu-range_shape/2

            # if (centered_mu[1] < delRange[1]) and (centered_mu[1] > delRange[-1]):
            #     mu = [mu[0], range_shape[1]/2]
            #     if centered_mu[1] > 0: # 正なら範囲以上に
            #         mu[1] += 30 + np.random.rand()*60
            #     else: # 負なら範囲以下に
            #         mu[1] -= 30 + np.random.rand()*60

            if centered_mu[1]>0:
                xy = 0
                xx = 1800
                yy = 850
                label = 1
            elif centered_mu[1]<=0:
                xy = 0
                xx = 850
                yy = 850
                label = 2
            
        else:
            # randRange = [2000,10000]
            randRange = [500,1500]
            dis = randRange[1]-randRange[0]
            xx = random.random()*dis+randRange[0]
            yy = random.random()*dis+randRange[0]
            xy = (random.random()*2-1)*randRange[0]
            label = 0

        sigma = np.array([[xx,xy],[xy,yy]])
        return mu,sigma,label

    # データの生成
    for path,key in zip(imagePath,keys):
        imgs = []
        labels = []
        filters = []
        print("generate "+key+" data")
        for i in range(dataNum[key]):
            print("\r progress:{0}/{1}".format(i+1,dataNum[key]),end="")
            Z = np.zeros(shape)
            
            # mu = np.random.rand(2)*range_shape
            # mu = [np.random.rand()*50+105,np.random.rand()*256]
            mu = [128,np.random.rand()*256]

            labs = ""
            for _ in range(random.randint(pointNum[0],pointNum[1])):
                # ランダムにサンプリング
                rand_mu, rand_sigma,label = sample_mu_sigma(mu)
                # 生成
                Z += gaussian(XY,rand_sigma,rand_mu).reshape(shape)
                labs += str(label)

            # ラベルの保存
            labels.append(labs)

            weight = random.random()*128 + 127 # max:127 ~ 255
            img = (Z/np.max(Z)) * weight / 255 # max:0.5 ~ 1

            if args.isNoised:
                img += np.random.rand(shape[0],shape[1],shape[2])-0.5
                img[img < 0] = 0
                img[img > 1] = 1

            if args.filterType!="none" or args.loadFilterPath != "":
                pass
                # img = img + filt[:,:,np.newaxis]
                # img[img > 1] = 1
                # filters.append(filt)
                # savefig = img*filt*255
                # cv2.imwrite("data/sample_filterToyData/{0:04d}.png".format(i),img[:,:,0]*255)

            imgs.append(img)

        dumpData = {"images":np.array(imgs),"labels":labels,"filters":np.array(filters)}
        pickle.dump(dumpData,open(path,"wb"))
        print("")
