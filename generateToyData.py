import numpy as np
import cv2
import os
import sys
import random
from argparse import ArgumentParser


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
    parser = ArgumentParser(description='Training script for PConv inpainting')
    parser.add_argument('-train', '--train',type=int, default=1500,help='number of training images (default=1500)')
    parser.add_argument('-valid', '--valid',type=int, default=100,help='number of validation images (default=100)')
    parser.add_argument('-test', '--test',type=int, default=100,help='number of test images (default=100)')
    parser.add_argument('-masktype', '--masktype',type=str, default='same', help='generate same mask images or vary mask images (default=same)', choices=['same', 'vary'])
    parser.add_argument('-ratio','--maskratio',type=float,default=0.99,help='ratio of mask (default=0.99)')
    parser.add_argument('-dspath','--dataSetPath',type=str,default="gaussianToyData",help='Path of dataSet')
    parser.add_argument('-posVariant','--positionVariant',action='store_true',help="Flag for making data position variant")

    return  parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dsPath = "data" + os.sep + args.dataSetPath # データセットのパス
    shape = (512,512) # 画像のサイズ
    dataNum = {"train":args.train,"valid":args.valid,"test":args.test} # train,validation,testでの生成数
 
    #================================================================================================
    # 画像の作成
    pointNum = [1,1] # ガウス分布の数 ([2,3]なら2~3個のガウス分布による混合ガウスになる)
    keys = ["train","valid","test"]
    dataPath = [dsPath+os.sep+k+os.sep+k+"_img" for k in keys]

    for DIR in dataPath:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

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
    for path,key in zip(dataPath,keys):
        print("generate "+key+" data")
        for i in range(dataNum[key]):
            print("\r progress:{0}/{1}".format(i+1,dataNum[key]),end="")
            Z = np.zeros(shape)

            labels = ""
            for _ in range(random.randint(pointNum[0],pointNum[1])):
                # ランダムにサンプリング
                rand_mu, rand_sigma,label = sample_mu_sigma()
                # 生成
                Z += gaussian(XY,rand_sigma,rand_mu).reshape(shape)
                labels += str(label)

            weight = random.random()*128+127
            img = (Z/np.max(Z))*weight
            cv2.imwrite(path+os.sep+"{0:04}_lab{1}.png".format(i,labels),img.astype("uint8"))
        print("")

    #================================================================================================
    # マスク画像の作成
    maskPath = [dsPath+os.sep+k+"_mask" for k in keys]
    maskRatio = args.maskratio # マスク部分の割合
    masktype = args.masktype

    for DIR in maskPath: # マスクのディレクトリを作成
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    def random_point_nodup(num): # 重複なしで座標を生成
        res = []
        while len(res) < num:
            x = random.randint(0,shape[0]-1)
            y = random.randint(0,shape[1]-1)
            dupflag = False
            for p in res:
                if p[0]==x and p[1]==y:
                    dupflag = True
                    break
            if not dupflag:
                res.append([x,y])

        return res
 
    obsNum = int(shape[0]*shape[1]*(1-maskRatio))
    if masktype == "same":
        mask = np.zeros(shape)
        # 観測点を生成し,マスクの要素を1 (画像として保存するので255) にする
        obs = random_point_nodup(obsNum)
        for p in obs:
            mask[p[0],p[1]] = 255
        mask = mask.astype("uint8")

    for path,key in zip(maskPath,keys):
        print("generate "+key+"_mask data")
        for i in range(dataNum[key]):
            print("\r progress:{0}/{1}".format(i+1,dataNum[key]),end="")
            if masktype=="same":
                cv2.imwrite(path+os.sep+"{0:04}.png".format(i),mask)
            else:
                mask = np.zeros(shape)
                obs = random_point_nodup(obsNum)
                for p in obs:
                    mask[p[0],p[1]] = 255
                mask = mask.astype("uint8")
                cv2.imwrite(path+os.sep+"{0:04}.png".format(i),mask)
        print("")
