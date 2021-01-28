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
from libs.util import MaskGenerator,ImageChunker,rangeError,nonhole,cmap,calcPCV1,clip,calcLabeledError,PSNR
from libs.createSpatialHistogram import compSpatialHist, compKL

import pickle
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def analyse(x):#x = [n,d]
    maxs = np.max(x,axis=0)
    mins = np.min(x,axis=0)
    stds = np.std(x,axis=0)
    means = np.mean(x,axis=0)
    return maxs,mins,means,stds

def parse_args():
    parser = ArgumentParser(description="学習済みのパラメータでテストをし、真値との比較や分析結果の保存を行います")
    parser.add_argument('dir_name',help="実験名(ログファイルのディレクトリ名でxxxx_logsのxxxxの部分のみ)")
    parser.add_argument('model',help="学習済みのweightファイルのエポック番号( weights.〇〇.h5 の〇〇の部分)")
    parser.add_argument('-dataset','--dataset',type=str,default='gaussianToyData',help="データセットのディレクトリ名")
    parser.add_argument('-thre','--pcv_thre', type=float, default=0.4, help="固有ベクトル計算時の閾値")
    parser.add_argument('-test','--test',type=str,default="", help="テスト画像のパス")
    parser.add_argument('-posVar','--isPositionVariant',action='store_true',help='位置によってデータの種類が異なる場合に使用')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    dspath = "data" + os.sep + args.dataset + os.sep
    pcv_thre = args.pcv_thre
    test_imgs_path = ".{0}data{0}{1}{0}test{0}test_img{0}*.png".format(os.sep,dspath) if args.test=="" else args.test

    # path
    path = ".{0}experiment{0}{1}_logs{0}".format(os.sep,args.dir_name)
    if not os.path.isdir(path):
        os.makedirs(path)

    shape = (512,512)
    TEST_PICKLE = dspath+"test.pickle"
    TEST_MASK_PICKLE = dspath+"test_mask.pickle"
    if "quake" in dspath:
        exist = np.array(Image.open("data/sea.png"))/255
    else:
        # 観測部分が１となる画像(マスク画像とは別の未観測地点がある場合に使用(Toyデータでは使用しないため全て１))
        exist = np.ones(shape)
    exist_rgb = np.tile(exist[:,:,np.newaxis],(1,1,3)) # カラー画像による可視化時に用いる
    BATCH_SIZE = 4

    #pdb.set_trace()
    # names, imgs, masks, labels = [], [], [], []
    tmp = pickle.load(open(TEST_PICKLE,"rb"))
    imgs = tmp["images"]
    # names = [nm.split(os.sep)[-1] for nm in tmp["labels"]]
    names = ["{0:04d}.png".format(i) for i in range(imgs.shape[0])]
    masks = pickle.load(open(TEST_MASK_PICKLE,"rb"))

    # モデルをビルドし,学習した重みをロード
    model = PConvUnet(img_rows=imgs.shape[1],img_cols=imgs.shape[2],inference_only=True)
    model_name = "weights.{}.h5".format(args.model)
    model.load(r"{}logs/{}_model/{}".format(path,args.dataset,model_name), train_bn=False)
    chunker = ImageChunker(shape[0], shape[1], 30)

    # テスト結果の出力先ディレクトリを作成
    result_path = "result{}".format(args.model)
    predict_path = path + result_path + os.sep + "test"
    compare_path = path+ result_path +os.sep+"comparison"
    pcv_path = path + result_path +os.sep+"pcv_thre{}_comparison".format(pcv_thre)
    hist_path = path + result_path + os.sep + "spatialHist_thre{}".format(pcv_thre)
    for DIR in [predict_path,compare_path,pcv_path,hist_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)
    
    errors, maes, mses = [],[],[] # 偏差,MAE,MSE
    centers, lines = [], [] # XY座標による主成分分析時の平均・主成分ベクトル
    mae0, maes_sep = [],[] # 値域ごとのMAE(0の地点とその他0.1間隔での誤差)
    psnrs = [] # 非穴部分のPSNR
    cm_bwr = plt.get_cmap("bwr") # 青から赤へのカラーマップ
    labels = []

    exist_ = exist.astype("float32")[np.newaxis,:,:,np.newaxis]
    img_sph = compSpatialHist(imgs.astype("float32"),exist_) # original histogram
    masked_sph = compSpatialHist((imgs*masks).astype("float32"),exist_) # masked histogram
    
    sess = tf.Session()
    masked_p,img_p,kls = sess.run([masked_sph,img_sph,compKL(masked_sph,img_sph)])

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
        if args.isPositionVariant:
            label = int(name.split("_")[-1].split(".")[0][-1]) # ラベル番号抽出
            labels.append(label)

        width = 3 # 横のプロット数
        # pdb.set_trace()

        #================================================
        # 予測結果を出力
        #"""
        tmp = (pred*255).astype("uint8")
        cv2.imwrite(os.path.join(predict_path,name), tmp)
        #"""
        #================================================

        #================================================
        # 非欠損部分の抽出・誤差の計算
        gt_nonh = nonhole(img,exist) # ground truth の非穴部
        pred_nonh = nonhole(pred,exist) # predict の非穴部

        mae_all = np.mean(np.abs(img-pred))

        err = pred-img
        mae_grand = np.mean(np.abs(gt_nonh-pred_nonh))
        mse_grand = np.mean((gt_nonh-pred_nonh)**2) # MSEは輝度=>応答スペクトルに変換して計算
        psnr = PSNR(pred_nonh,gt_nonh)

        errors.append(err)
        maes.append(mae_grand)
        mses.append(mse_grand)
        psnrs.append(psnr)

        #==========================================================================================
        # 入力・予測・真値の比較
        x1 = cmap(masked)
        x1[mask_rgb==0] = 255
        xs = [x1,cmap(pred),cmap(img)]
        titles = ["masked","pred(MAE={0:.4f})".format(mae_grand),"original"]

        #"""
        _, axes = plt.subplots(3, width, figsize=(width*4+2, 15))
        for i,x in enumerate(xs):
            x[exist_rgb==0] = 255
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
        
        
        # # 各震度値ごとのMAE 
        e0 = rangeError(pred,img,domain=[-1.0,0.0]) # 負の値は存在しないので0のみのMAEを計算できる
        sep_errs = [rangeError(pred,img,domain=[i*0.1,(i+1)*0.1]) for i in range(10)]
        mae0.append(e0)
        maes_sep.append(sep_errs)
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
        pcv = [calcPCV1(masked,args.pcv_thre)
        , calcPCV1(pred*exist,args.pcv_thre)
        , calcPCV1(img,args.pcv_thre)]

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
        # Spatial Histogram
        pred_sph = compSpatialHist(pred.astype("float32")[np.newaxis,:,:,np.newaxis],exist_) # prediction histogram
        pred_p,klp = sess.run([pred_sph,compKL(pred_sph,img_sph[ite:ite+1])])

        fig = plt.figure()
        # masked
        ax=fig.add_subplot(2,3,1)
        im = ax.imshow(masked,vmin=0,vmax=1,cmap="jet")
        ax.set_title('sparse')

        ax=fig.add_subplot(2,3,4)
        im = ax.imshow(masked_p[ite],cmap="jet")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_title(f'hist of sparse, kl={kls[ite]:.4f}')

        # prediction
        ax=fig.add_subplot(2,3,2)
        im = ax.imshow(pred,vmin=0,vmax=1,cmap="jet")
        ax.set_title('pred')

        ax=fig.add_subplot(2,3,5)
        im = ax.imshow(pred_p[0],cmap="jet")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_title(f'hist of pred, kl={klp[0]:.4f}')

        # original
        ax=fig.add_subplot(2,3,3)
        ax.imshow(img,vmin=0,vmax=1,cmap="jet")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_title('dense')

        ax=fig.add_subplot(2,3,6)
        im = ax.imshow(img_p[ite],cmap="jet")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_title(f'hist of original')

        plt.tight_layout()
        plt.savefig(os.path.join(hist_path,name))
        plt.close()
        #==========================================================================================

    #"""
    # 全テスト結果のセルごとの誤差
    errors = np.array(errors)
    err = cm_bwr(clip(np.mean(errors,axis=0),-0.1,0.1))[:,:,:3]

    _,axes = plt.subplots(1,1,figsize=(6,5))
    axes.imshow(err*exist_rgb,vmin=0,vmax=1.0)
    plt.savefig(path+result_path+os.sep+"mae_map.png")
    #"""

    #"""
    # 分析結果の保存・表示
    summary_data = {
        "PSNR":np.mean(np.array(psnrs)),
        "MAE":np.mean(np.array(maes)),
        "MSE":np.mean(np.array(mses)),
        "MAE0":np.mean(np.array(mae0)),
        "MAE-sep0.1":np.mean(np.array(maes_sep))
    }

    print("\nPSNR={0:.10f}".format(summary_data["PSNR"]))
    print("MSE={0:.10f}, MAE={1:.10f}".format(summary_data["MSE"],summary_data["MAE"]))

    if args.isPositionVariant:
        labMAEs = calcLabeledError(errors,labels,opt="MA")
        labMSEs = calcLabeledError(errors,labels,opt="MS")
        summary_data["labMAEs"] = labMAEs
        summary_data["labMSEs"] = labMSEs
        summary_data["labels"] = labels

    pkl_path = os.path.join(path,"analysed_data.pickle")
    with open(pkl_path,"wb") as f:
        pickle.dump(summary_data,f)
    #"""
