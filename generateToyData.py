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
    parser.add_argument('-loadMask','--loadMaskPath',default="",help='using exist Mask Image, you have to set the path. (default="" (generate new mask))')
    parser.add_argument('-train', '--train',type=int, default=1500,help='number of training images (default=1500)')
    parser.add_argument('-valid', '--valid',type=int, default=100,help='number of validation images (default=100)')
    parser.add_argument('-test', '--test',type=int, default=100,help='number of test images (default=100)')
    parser.add_argument('-masktype', '--masktype',type=str, default='same', help='generate same mask images or vary mask images (default=same)', choices=['same', 'vary'])
    parser.add_argument('-ratio','--maskratio',type=float,default=0.99,help='ratio of mask (default=0.99)')
    parser.add_argument('-dspath','--dataSetPath',type=str,default="gaussianToyData",help='Name of dataSet (default=gaussianToyData)')
    parser.add_argument('-posVariant','--positionVariant',action='store_true',help="Flag for making data position variant")
    parser.add_argument('-mixed','--isMixedGaussian',action='store_true',help="Flag for making data mixed-gaussian (if you use this, -posVariant is not available)")

    return  parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    shape = (512,512) # 画像のサイズ
    keys = ["train","valid","test"]
    dsPath = "data" + os.sep + args.dataSetPath # データセットのパス
    if not os.path.isdir(dsPath):
        os.makedirs(dsPath)

    dataNum = {keys[0]:args.train,keys[1]:args.valid,keys[2]:args.test} # train,validation,testでの生成数 
    imagePath = [dsPath + os.sep + k + ".pickle" for k in keys]

    # 指定されたマスク画像およびマスク画像のpickleデータをロードする
    if args.loadMaskPath!="":
        ext = args.loadMaskPath.split(".")[-1]
        if ext == "png" or ext == "jpg":
            existMask = np.array(Image.open(args.loadMaskPath))/255
        elif ext == "pickle" or ext == "pkl":
            existMask = pickle.load(open(args.loadMaskPath,"rb"))

        # サイズが合わなければreshape
        if existMask.shape[0] == shape[0] and existMask.shape[1] == shape[1]:
            print("Given mask image is reshaped.")
            existMask = np.reshape(existMask,shape)


    # 混合分布と位置依存は同時に使えない
    if args.isMixedGaussian and args.positionVariant:
        print("you can't use option -mixed and -posVariant at the same time")
        sys.exit()

    if args.isMixedGaussian:
        pointNum = [2,3] # ガウス分布の数 ([2,3]なら2~3個のガウス分布による混合ガウスになる)
    else:
        pointNum = [1,1]

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

        pickle.dump(np.array(masks),open(path,"wb"))
        print("")

    #=============================================================
    # 画像の作成
    # XY座標を生成
    x = np.arange(0, shape[0], 1)
    y = np.arange(0, shape[1], 1)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(),Y.ravel()]

    def sample_mu_sigma(): # 平均と分散のランダムサンプリング
        mu = np.random.rand(2)*np.array(shape)

        if args.positionVariant:
            centered_mu = mu-np.array(shape)/2
            if centered_mu[0]>0 and centered_mu[1]>0:
                xy = 0
                xx = 6000
                yy = 6000
                label = 1
            elif centered_mu[0]>0 and centered_mu[1]<0:
                xy = 0
                xx = 1500
                yy = 1500
                label = 2
            elif centered_mu[0]<0 and centered_mu[1]<0:
                sign = 1 if random.random() < 0.5 else -1
                xy = sign*1000
                xx = 1500
                yy = 1500
                label = 3
            elif centered_mu[0]<0 and centered_mu[1]>0:
                sign = 1 if random.random() < 0.5 else -1
                xy = sign*2000
                xx = 4000
                yy = 4000
                label = 4
        else:
            randRange = [2000,10000]
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
        print("generate "+key+" data")
        for i in range(dataNum[key]):
            print("\r progress:{0}/{1}".format(i+1,dataNum[key]),end="")
            Z = np.zeros(shape)

            labs = ""
            for _ in range(random.randint(pointNum[0],pointNum[1])):
                # ランダムにサンプリング
                rand_mu, rand_sigma,label = sample_mu_sigma()
                # 生成
                Z += gaussian(XY,rand_sigma,rand_mu).reshape(shape)
                labs += str(label)

            labels.append(labs)

            weight = random.random()*128+127
            img = (Z/np.max(Z))*weight
            imgs.append(img)
        dumpData = {"images":np.array(imgs),"labels":labels}
        pickle.dump(dumpData,open(path,"wb"))
        print("")


