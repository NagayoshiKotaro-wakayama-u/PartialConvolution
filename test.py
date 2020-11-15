import os
import gc
import datetime
import numpy as np
import cv2
import pdb
import glob
import sys

from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras import backend as K
from keras.utils import Sequence
from keras_tqdm import TQDMCallback

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from libs.pconv_model import PConvUnet
from libs.util import MaskGenerator,ImageChunker

import pickle
import cv2

def clip(x,sta=-0.1,end=0.1): # Clip the value
    x[x<sta] = sta
    x[x>end] = end
    dist = end-sta
    res = (x-sta)/dist
    return res

def nonhole(x,hole,opt=""): # 欠損部以外の値を取り出す
    shape = x.shape
    flatt = np.reshape(x,(np.product(shape)))
    holes = np.reshape(hole,(np.product(shape)))
    tmp = []
    x,y = [],[]
    for pix,hole,ite in zip(flatt,holes,[i for i in range(flatt.shape[0])]):
        if np.sum(hole) < 1e-10:
            continue
        tmp.append(pix)
        if opt=="xy":
            x.append(ite//shape[1])
            y.append(ite%shape[0])

    if opt=="xy":
        return np.array(tmp),np.array(x),np.array(y)

    return np.array(tmp)


def cmap(x,sta=[222,222,222],end=[255,0,0]): #x:gray-image([w,h]) , sta,end:[B,G,R]
    vec = np.array(end) - np.array(sta)
    res = []
    for i in range(x.shape[0]):
        tmp = []
        for j in range(x.shape[1]):
            tmp.append(np.array(sta)+x[i,j]*vec)
        res.append(tmp)
    res = np.array(res).astype("uint8")
    res[exist_rgb==0] = 255
    return res

def calcPCV1(x):
    x = np.array(np.where(x>pcv_thre))
    if 0 in x.shape:
        return np.array([[0,0]]).T , np.array([[0,0],[0,0]])

    center = np.mean(x,axis=1)[:,np.newaxis]
    xCe = x - center
    Cov = np.cov(xCe,bias=1)
    if True in np.isnan(Cov):
        print("nan")
        pdb.set_trace()
    elif True in np.isinf(Cov):
        print("inf")
        pdb.set_trace()
    V,D = np.linalg.eig(Cov)
    vec = D[:,[np.argmax(V)]]
    line = np.concatenate([vec*-256,vec*256],axis=1) + center
    return center,line


def analyse(x):#x = [n,d]
    maxs = np.max(x,axis=0)
    mins = np.min(x,axis=0)
    stds = np.std(x,axis=0)
    means = np.mean(x,axis=0)
    return maxs,mins,means,stds

def rangeError(pre,tru,domain=[-1.0,0.0],opt="MA"): # pred:予測値, true:真値 , domain:値域(domain[0]< y <=domain[1])
    # ある値域の真値のみで誤差を測る

    domain = [domain[0]/8,domain[1]/8] # normalyse
    inds = np.where(np.logical_and(tru>domain[0],tru<=domain[1])) # 
    if inds[0].shape[0]==0: # 値がない場合はNaN
        return np.NaN

    error_ = tru[inds[0],inds[1]]-pre[inds[0],inds[1]]
    if opt=="MA": # MAE
        error_ = np.mean(np.abs(error_))
    elif opt=="MS": # MSE
        error_ = np.mean(error_**2)
    elif opt=="A":
        error_ = np.abs(error_)

    return error_


def parse_args():
    parser = ArgumentParser(description="学習済みのパラメータでテストをし、真値との比較や分析結果の保存を行います")
    parser.add_argument('dir_name',help="実験名(ログファイルのディレクトリ名でxxxx_logsのxxxxの部分のみ)")
    parser.add_argument('model',help="学習済みのweightのファイル名(例：weights.150-0.13.h5)")
    parser.add_argument('-dataset','--dataset',type=str,default='gaussianToyData',help="データセットのディレクトリ名")
    parser.add_argument('-thre','--pcv_thre', type=float, default=0.2, help="固有ベクトル計算時の閾値")
    parser.add_argument('-test','--test',type=str,default="", help="テスト画像のパス")

    return  parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    dspath = args.dataset
    pcv_thre = args.pcv_thre
    test_imgs_path = ".{0}data{0}{1}{0}test{0}test_img{0}*.png".format(os.sep,dspath) if args.test=="" else args.test

    # path
    path = ".{0}experiment{0}{1}_logs{0}".format(os.sep,args.dir_name)
    if not os.path.isdir(path):
        os.makedirs(path)

    shape = (512,512)
    img_files = sorted(glob.glob(test_imgs_path)) # テスト画像のパス取得
    mask_files = sorted(glob.glob(".{0}data{0}{1}{0}test_mask{0}*.png".format(os.sep,dspath))) # テストのマスク画像のパス取得
    exist = np.ones(shape) # 観測部分が１となる画像(マスク画像とは別の未観測地点がある場合に使用(Toyデータでは使用しないため全て１))
    exist_rgb = np.tile(exist[:,:,np.newaxis],(1,1,3)) # カラー画像による可視化時に用いる
    BATCH_SIZE = 4

    names, imgs, masks = [], [], []

    for i,fname in enumerate(img_files):
        # normalyse
        img = np.array(Image.open(fname))/255
        mask = np.array(Image.open(mask_files[i]))/255

        names.append(fname.split(os.sep)[-1])
        imgs.append(img[:,:,np.newaxis])
        masks.append(mask[:,:,np.newaxis])

    # モデルをビルドし,学習した重みをロード
    model = PConvUnet(img_rows=img.shape[0],img_cols=img.shape[1],inference_only=True)
    model_name = args.model
    model.load(r"{}logs/{}_model/{}".format(path,dspath,model_name), train_bn=False)
    chunker = ImageChunker(512, 512, 30)

    # テスト結果の出力先ディレクトリを作成
    result_path = path+"result"+os.sep+"test"
    compare_path = path+"result"+os.sep+"comparison"
    pcv_path = path + "result"+os.sep+"pcv_thre{}_comparison".format(pcv_thre)
    for DIR in [result_path,compare_path,pcv_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)
    
    errors, maes, mses = [],[],[] # 偏差,MAE,MSE
    centers, lines = [], [] # XY座標による主成分分析時の平均・主成分ベクトル
    mae0, mae05_2, maes_sep = [],[],[] # 値域ごとのMAE
    cm_bwr = plt.get_cmap("bwr") # 青から赤へのカラーマップ

    # 予測してプロット
    for name,img,mask,ite in zip(names,imgs,masks,[i for i in range(len(names))]):
        print("\r progress : {}/{}".format(ite+1,len(names)),end="")

        masked = chunker.dimension_preprocess(deepcopy(img*mask))
        masks = chunker.dimension_preprocess(deepcopy(mask))
        pred = model.predict([masked,masks])[0,:,:,0] # 予測
        masked = masked[0,:,:,0] # 入力（マスクされた画像）
        mask_rgb = np.tile(mask,(1,1,3))
        mask = mask[:,:,0]
        img = img[:,:,0]

        width = 3 # 横のプロット数

        #================================================
        # 予測結果を出力
        #"""
        tmp = (pred*255).astype("uint8")
        cv2.imwrite(os.path.join(result_path,name), tmp)
        #"""
        #================================================

        #================================================
        # 非欠損部分の抽出・誤差の計算
        gt_nonh = nonhole(img,exist)
        pred_nonh = nonhole(pred,exist)

        mae_all = np.mean(np.abs(img-pred))

        err = pred-img
        mae_grand = np.mean(np.abs(gt_nonh-pred_nonh))
        mse_grand = np.mean((gt_nonh*8-pred_nonh*8)**2) # MSEは輝度=>応答スペクトルに変換して計算

        errors.append(err)
        maes.append(mae_grand)
        mses.append(mse_grand)

        #==========================================================================================
        # 入力・予測・真値の比較
        #"""
        x1 = cmap(masked)
        x1[mask_rgb==0] = 255
        xs = [x1]
        titles = ["masked","pred(MAE={0:.4f})".format(mae_grand)]
        xs.append(cmap(pred))

        xs.append(cmap(img))
        titles.append("original")

        
        _, axes = plt.subplots(3, width, figsize=(width*4+2, 15))
        for i,x in enumerate(xs):
            axes[0,i].imshow(x,vmin=0,vmax=255)
            axes[0,i].set_title(titles[i])

        # ヒストグラム
        bins = 20
        hs = []
        hs.append(nonhole(img,mask*exist))
        hs.append(nonhole(pred,exist))
        hs.append(nonhole(img,exist))
        tmp = np.concatenate(hs,axis=0)
        maxs = np.max(tmp)

        for i,h in enumerate(hs):
            axes[1,i].hist(h,bins=bins,range=(0,maxs))
        
        
        # 各震度値ごとのMAE 
        e0 = rangeError(pred,img,domain=[-1.0,0.0])
        e05_2 = rangeError(pred,img,domain=[0.5,2.0])
        sep_errs = [rangeError(pred,img,domain=[i*0.8,(i+1)*0.8]) for i in range(10)]
        mae0.append(e0)
        mae05_2.append(e05_2)
        maes_sep.append(sep_errs)
                
        # axes[2,0].text(0.2,0.05,"mae0:{0:.4f}".format(e0))
        # axes[2,0].text(0.2,0.125,"mae0.5_2:{0:.4f}".format(e05_2))
        # for i,er in enumerate(sep_errs):
        #     axes[2,0].text(0.2,0.2+i*0.075,"mae{0:.1f}_{1:.1f}:{2:.4f}".format(0.1*i,0.1*(i+1),er))

        axes[2,-1].plot(np.array([(i+1)*0.1 for i in range(10)]),sep_errs)

        # AE map
        err = cm_bwr(clip(err,-0.1,0.1))[:,:,:3] # -0.1~0.1の偏差をカラーに変換
        axes[2,1].imshow(err*exist_rgb,vmin=0,vmax=1.0)
        
        plt.savefig(os.path.join(compare_path,name))
        plt.close()
        #"""
        #==========================================================================================
        # calculate pcv and plot
        #"""
        _, axes = plt.subplots(2, width, figsize=(width*4+2, 11))

        # 主成分分析
        colors = ['g','b','r']
        pcv = [calcPCV1(masked), calcPCV1(pred*exist), calcPCV1(img)]

        # 対象の画像と主成分ベクトルを重ねてプロット
        for i,x in enumerate(xs):
            # 画像の上段
            axes[0,i].imshow(x,vmin=0,vmax=255)
            line = pcv[i][1] # 主成分ベクトル
            ce = pcv[i][0] # 平均
            axes[0,i].plot(line[1],line[0],colors[i]+'-')
            axes[0,i].scatter(ce[1],ce[0],c=colors[i])
            axes[0,i].set_title(titles[i])
            # 画像の下段 (固有ベクトルの比較)
            if i != len(xs)-1:
                axes[1,i].imshow(x,vmin=0,vmax=255)
                sr_l = pcv[0][1]
                tr_l = pcv[-1][1]
                axes[1,i].plot(sr_l[1],sr_l[0],colors[0]+'-')
                axes[1,i].plot(tr_l[1],tr_l[0],colors[-1]+'-')
                axes[1,i].plot(line[1],line[0],colors[i]+'-')
            else:
                for j,center in enumerate([v[0] for v in pcv]):
                    axes[1,i].scatter(center[1],center[0],c=colors[j])

        # 全部のサイズを調整
        for i in range(2):
            for j in range(width):
                axes[i,j].set_xlim(0,511)
                axes[i,j].set_ylim(511,0)

        plt.savefig(os.path.join(pcv_path,name))
        plt.close()
        #"""
        #==========================================================================================

    #"""
    # 全テスト結果のセルごとの誤差
    errors = np.array(errors)
    err = cm_bwr(clip(np.mean(errors,axis=0),-0.1,0.1))[:,:,:3]

    _,axes = plt.subplots(1,1,figsize=(6,5))
    axes.imshow(err*exist_rgb,vmin=0,vmax=1.0)
    plt.savefig(path+"result"+os.sep+"mae_map.png")
    #"""

    #"""
    # 分析結果
    summary_data = {
        "MAE":np.mean(np.array(maes)),
        "MSE":np.mean(np.array(mses)),
        "MAE0":np.mean(np.array(mae0)),
        "MAE-sep0.8":np.mean(np.array(maes_sep)),
        "MAE0.5~2":np.mean(np.array(mae05_2))
    }

    print("MSE={0:.8f}".format(summary_data["MSE"]))
    print("MAE={0:.8f}, MAE0.5~2={1:.8f}".format(summary_data["MAE"],summary_data["MAE0.5~2"]))
    
    pkl_path = os.path.join(path,"analysed_data.pickle")
    with open(pkl_path,"wb") as f:
        pickle.dump(summary_data,f)
    #"""