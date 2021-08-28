import os
import gc
import datetime
import numpy as np
import cv2
import pickle
import glob
import pdb
import sys
import tensorflow as tf

from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback,Callback
from keras_tqdm import TQDMCallback
from keras.models import Model

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from libs.pconv_model import PConvUnet,sitePConvUnet,PKConvUnet
from libs.util import MaskGenerator,resize_images,standardize
from PIL import Image

def cmap(x,sta=[222,222,222],end=[255,0,0]): #x:gray-image([w,h]) , sta,end:[B,G,R]
    vec = np.array(end) - np.array(sta)
    res = []
    for i in range(x.shape[0]):
        tmp = []
        for j in range(x.shape[1]):
            tmp.append(np.array(sta)+x[i,j]*vec)
        res.append(tmp)
    res = np.array(res).astype("uint8")
    return res

def calcPCV1(x): # 第一主成分ベクトルを導出し
    pcv_thre=args.KLthre
    x = np.array(np.where(x>pcv_thre))
    if 0 in x.shape:
        return np.array([[0,0]]).T , np.array([[0,0],[0,0]])
    center = np.mean(x,axis=1)[:,np.newaxis]
    xCe = x - center
    Cov = np.cov(xCe,bias=1)
    if True in np.isnan(Cov):
        raise Exception("Cov is nan.")
    elif True in np.isinf(Cov):
        raise Exception("Cov is inf.")
    V,D = np.linalg.eig(Cov)
    vec = D[:,[np.argmax(V)]]
    line = np.concatenate([vec*-(img_h/2),vec*img_h/2],axis=1) + center
    return center,line

# def resize_images(xs,size): # xs=[N,W,H,C]
#     res = []
#     for x in xs:
#         chanXs = []
#         for i in range(x.shape[2]):
#             chanXs.append(cv2.resize(x[:,:,i],size)[:,:,np.newaxis])
#         chanXs = np.concatenate(chanXs,axis=2)
#         res.append(chanXs)

#     return np.array(res)

def loadSiteImage(paths):#paths=[str1,str2,...,strN]
    _images = []
    for path in paths:
        _img = np.array(Image.open(f"{path}"))[:,:,np.newaxis]
        
        if args.stdSite: # 標準化
            # 海洋部を除いて標準化
            if "quake" in dataset:
                _exist = np.array(Image.open(SEA_PATH))/255
            else:
                _exist=None
            # pdb.set_trace()
            _img = standardize(_img,exist=_exist)

        else: # 線形に正規化
            _img = _img*args.siteScale + args.siteBias

        _images.append(_img)
    return _images


def parse_args():
    parser = ArgumentParser(description='Training script for PConv inpainting')
    parser.add_argument('experiment',type=str,help='name of experiment, e.g. \'normal_PConv\'')
    parser.add_argument('-dataset','--dataset',type=str,default='gaussianToyData',help='name of dataset directory (default=gaussianToyData)')
    parser.add_argument('-stage', '--stage',type=str, default='train', help='Which stage of training to run', choices=['train', 'finetune'])
    parser.add_argument('-train', '--train', type=str, default="", help='Folder with training images')
    parser.add_argument('-valid', '--validation', type=str, default="", help='Folder with validation images')
    parser.add_argument( '-test', '--test', type=str, default="", help='Folder with testing images')
    parser.add_argument('-trainmask', '--trainmask', type=str, default="", help='Folder with training mask images')
    parser.add_argument('-validmask', '--validmask', type=str, default="", help='Folder with validation mask images')
    parser.add_argument( '-testmask', '--testmask', type=str, default="", help='Folder with testing mask images')
    parser.add_argument('-checkpoint', '--checkpoint',type=str, help='Previous weights to be loaded onto model')
    parser.add_argument('-epochs','--epochs',type=int,default=100,help='training epoch')
    parser.add_argument('-imgw','--imgw',type=int,default=512,help='input width')
    parser.add_argument('-imgh','--imgh',type=int,default=512,help='input height')
    parser.add_argument('-lr','--lr',type=float,default=0.0002,help='learning rate')

    # EarlyStopping
    parser.add_argument('-es','--isEarlyStopOn',action='store_true',help="Flag for using Early stopping")
    parser.add_argument('-esEpoch','--earlyStopEpoch',type=int,default=10)
    # 対数尤度
    parser.add_argument('-llh','--LLH',action='store_true')
    parser.add_argument('-llhonly','--LLHonly',action='store_true')
    # KL
    parser.add_argument('-t1','--truefirst',action='store_true',help="KL距離において真値を第一引数にするかどうか")
    parser.add_argument('-p1','--predfirst',action='store_true',help="KL距離において予測値を第一引数にするかどうか")
    parser.add_argument('-klbias','--KLbias',action='store_true',help="KL-divergenceを用いる際、真値と予測値が同じ時に勾配が０になるためのバイアス")
    parser.add_argument('-klonly','--KLonly',action='store_true',help="KL「のみ」を損失として使用する")
    parser.add_argument('-KLthre', '--KLthre',type=float, default=0.0,help='threshold value of KLloss ※使用しない') # 使用しない
    parser.add_argument('-KL','--KL',action='store_true',help="Flag for using KL-loss function")
    # histogram KL
    parser.add_argument('-histKL','--histKL',action='store_true',help="Flag for using spatial Histogram KL-loss function")
    parser.add_argument('-histFilterSize','--histFilterSize',type=int,default=64,help="size of filter to make a histogram (default=64)" )

    # 学習途中でテストのプロットをするかどうか
    parser.add_argument('-plotTest','--plotTest',action='store_true')
    # サイト特性を考慮したモデルを使用するかどうか
    parser.add_argument('-pchan','--posEmbChan',type=int,default=1,help='channnels of position code (learnable)')
    parser.add_argument('-sitePConv','--sitePConv',action='store_true')
    parser.add_argument('-posKernel','--positionalKernel',action='store_true')
    parser.add_argument('-eachChannel','--eachChannel',action='store_true')
    parser.add_argument('-posKernelOpe','--posKernelOpe',type=str,default="add")
    parser.add_argument('-PKlayers','--PKlayers',type=lambda x:list(map(int,x.split(","))),default=[3],help="list of PKConvlayer number. ex:3,4,5")
    parser.add_argument('-loadSite','--loadSitePath',type=lambda x:list(map(str,x.split(","))),default="")
    parser.add_argument('-encFNum','--encFNum',type=lambda x:list(map(int,x.split(","))),default="64,128,256,512,512")

    parser.add_argument('-useSiteNorm','--useSiteNorm',action='store_true')
    parser.add_argument('-useSiteCNN','--useSiteCNN',action='store_true')
    parser.add_argument('-stdSite','--stdSite',action='store_true',help="ロードした位置特性を標準化する。これをオンにすると正規化は行われない。")
    parser.add_argument('-sCNNFNum','--sCNNFNum',type=lambda x:list(map(int,x.split(","))),default="1,1,1,1,1")
    parser.add_argument('-sCNNBias','--sCNNBias',action="store_true")
    parser.add_argument('-sCNNAct','--sCNNAct',default=None)
    parser.add_argument('-sCNNSinglePath','--sCNNSinglePath',action="store_true")

    parser.add_argument('-sScale','--siteScale',type=float,default=1/255)
    parser.add_argument('-sBias','--siteBias',type=float,default=0)


    return  parser.parse_args()


# paths = [Image And Site, mask]
def Generator(paths, batchSize):
    imagePath = paths[0]
    maskPath = paths[1]
    data = pickle.load(open(imagePath,"rb"))
    mask = pickle.load(open(maskPath,"rb"))

    while True:
        for B in range(0, len(data), batchSize):
            ori = data["images"][B:B+batchSize] # original images

            # Y = data["labels"][B:B+batchSize] # labels
            mask = mask[B:B+batchSize] # masks
            masked = mask*ori # masked image

            # pdb.set_trace()
            if useSite or args.useSiteCNN:
                # 初めの層のサイズに合わせてサイズ変更
                devide = 2**(args.PKlayers[0]-1)
                layer_shape = tuple([int(s/devide) for s in shape])
                site = np.tile(posEmb,[batchSize,1,1,1])
                site = resize_images(site,layer_shape)
                inp = (masked, mask, site)
            else:
                inp = (masked, mask)

            out = ori

            yield inp, out


class PSNREarlyStopping(ModelCheckpoint):
    def __init__(self,savepath,log_path,dataset):
        super(PSNREarlyStopping,self).__init__(
            os.path.join(log_path, dataset+'_model', 'weights.{epoch:02d}.h5'),
            monitor='val_PSNR',
            save_best_only=False,
            save_weights_only=True,
            period = 1
        )
        self.best_val_PSNR = -1000
        self.history_val_PSNR = []
        self.best_weights   = None
        self.now_epoch = 0
        self.limitRatio = 0.05

        # ロスなどの記録
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        self.path = os.path.join(savepath,"training_losses.pickle")
        self.types = ["loss","PSNR","loss_KL","original","loss_Djs"]
        # self.types = ["output_img_"+t for t in self.types]
        # training用
        self.trainingLoss = dict([(t,[]) for t in self.types])
        # validation用
        self.validationLoss = dict([(t,[]) for t in self.types])

    def on_train_batch_end(self, batch, logs={}): # バッチ終了時に呼び出される
        # pdb.set_trace()
        # 訓練時のロスの保存
        for t in self.types:
            self.trainingLoss[t].append(logs.get(t))

    def on_epoch_end(self, epoch, logs=None):
        print("epoch{}".format(epoch))
        self.saveModelPath = self._get_file_path(epoch, logs)
        self.now_epoch += 1
        # 検証時のロスの保存
        for t in self.types:
            self.validationLoss[t].append(logs.get("val_"+t))

        # 過学習の検知
        #=================================================================
        if args.isEarlyStopOn:
            self.epochs_since_last_save += 1
            val_PSNR = logs['val_PSNR']
            self.history_val_PSNR.append(val_PSNR)
            
            # 検証データのPSNRの最大値を取得
            if val_PSNR > self.best_val_PSNR:
                self.best_val_PSNR = val_PSNR
                
            # 指定されたエポック(nエポック)以降で、最大値と比べて5%以上下がっているなら終了
            if (epoch+1) >= args.earlyStopEpoch:
                # pdb.set_trace()
                if self.best_val_PSNR*(1-self.limitRatio) > val_PSNR:
                    # print("best:{}, current:{}".format(self.best_val_PSNR,np.max(self.history_val_PSNR[-5:])))
                    self.model.stop_training = True
                    self.on_train_end()
                    sys.exit()
        #=================================================================

        # 10回に1回モデルを保存
        if (epoch+1)%10==0:
            # pdb.set_trace()
            # self._save_model(epoch=epoch, logs=logs)
            self.model.save_weights(self.saveModelPath, overwrite=True, options=self._options)

    def on_train_end(self,logs=None):
        # self._save_model(epoch=epoch, logs=logs)
        self.model.save_weights(self.saveModelPath, overwrite=True, options=self._options)

        summary = {
            "epochs":epochs,
            "end_epoch":self.now_epoch,
            "steps_per_epoch":steps_per_epoch
        }

        # summary に学習・検証の損失のデータを加える
        for t in self.types:
            summary[t] = self.trainingLoss[t]
            summary["val_"+t] = self.validationLoss[t]

        with open(self.path,"wb") as f:
            pickle.dump(summary,f)
        
        for lossName in self.types:
            loss = summary[lossName]
            plt.plot(range(self.now_epoch*steps_per_epoch),loss)
            plt.xlabel('Iteration (1epoch={}ite)'.format(steps_per_epoch))
            plt.ylabel(lossName)
            plt.title(args.experiment)
            plt.savefig(os.path.join(loss_path,lossName+".png"))
            plt.close()

            loss = summary["val_"+lossName]
            plt.plot(range(self.now_epoch),loss)
            plt.xlabel('Epoch')
            plt.ylabel("val_"+lossName)
            plt.title(args.experiment)
            plt.savefig(os.path.join(loss_path,"val_"+lossName+".png"))
            plt.close()


# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    if args.stage == 'finetune' and not args.checkpoint:
        raise AttributeError('If you are finetuning your model, you must supply a checkpoint file')

    # 実験に用いるディレクトリを作成
    experiment_path = ".{0}experiment{0}{1}_logs".format(os.sep,args.experiment)
    loss_path = f"{experiment_path}{os.sep}losses"
    log_path = f"{experiment_path}{os.sep}logs"
    test_path = f"{experiment_path}{os.sep}test_samples"
    # site_path = f"data{os.sep}siteImages{os.sep}"
    site_path = f"data{os.sep}new_siteImages{os.sep}"

    for DIR in [experiment_path,loss_path,log_path,test_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    epochs = args.epochs
    dataset = args.dataset # データセットのディレクトリ
    dspath = ".{0}data{0}{1}{0}".format(os.sep,dataset)

    # 各pickleデータのパス
    TRAIN_PICKLE = dspath+"train.pickle" if args.train=="" else args.train
    TRAIN_MASK_PICKLE = dspath+"train_mask.pickle" if args.trainmask=="" else args.trainmask
    VALID_PICKLE = dspath+"valid.pickle" if args.validation=="" else args.valid
    VALID_MASK_PICKLE = dspath+"valid_mask.pickle" if args.validmask=="" else args.validmask
    TEST_PICKLE = dspath+"test.pickle" if args.test=="" else args.test
    TEST_MASK_PICKLE = dspath+"test_mask.pickle" if args.testmask=="" else args.testmask

    # 地震データの時は海洋部のマスクをロード
    
    SEA_PATH = ".{0}data{0}sea.png".format(os.sep) if "quake" in dataset else ""

    # 位置エンベッディング画像のロード
    useSite = False
    if args.loadSitePath[0]!="":
        if not args.useSiteCNN:
            useSite = True

        # pdb.set_trace()
        posEmb = loadSiteImage([f"{site_path}{p}" for p in args.loadSitePath])
        posEmb = np.concatenate(posEmb,axis=2) if len(args.loadSitePath)>1 else posEmb[0] # 複数ロードする場合はチャネル方向に結合
        posEmb = posEmb[np.newaxis] # [1,H,W,C]
        args.posEmbChan = posEmb.shape[3]


    train_Num = pickle.load(open(TRAIN_PICKLE,"rb"))["images"].shape[0] # 画像の枚数をカウント
    valid_Num = pickle.load(open(VALID_PICKLE,"rb"))["images"].shape[0]
    img_w = args.imgw
    img_h = args.imgh
    shape = (img_h, img_w)

    # バッチサイズはメモリサイズに合わせて調整が必要
    batchsize = 5 # バッチサイズ
    steps_per_epoch = train_Num//batchsize # 1エポック内のiteration数

    # generatorを作成
    trainPaths = [TRAIN_PICKLE,TRAIN_MASK_PICKLE]
    validPaths = [VALID_PICKLE,VALID_MASK_PICKLE]
    testPaths = [TEST_PICKLE,TEST_MASK_PICKLE]
    train_generator = Generator(trainPaths,batchsize) # Create training generator
    val_generator = Generator(validPaths,batchsize) # Create validation generator
    test_generator = Generator(testPaths,batchsize) # Create testing generator

    # Pick out an example to be send to test samples folder
    test_data = next(test_generator)

    if useSite or args.useSiteCNN:
        (masked, mask, site), ori = test_data
    else:
        (masked, mask), ori = test_data

    mask_rgb = np.tile(np.reshape(mask[0],[shape[0],shape[1],1]),(1,1,3)) #カラーで可視化する際に用いるマスク

    # 学習途中のテスト結果が必要ない場合はmodel.fit_generator内で
    # 以下の plot_callback() を呼び出している行のコメントアウトをして下さい
    def plot_callback(model, path, epoch):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""
        
        # Get samples & Display them
        if useSite:
            pred_img = model.predict([masked, mask, site])
        else:
            pred_img = model.predict([masked, mask])

        # Clear current output and display test images
        for i in range(1):
            img = ori[i,:,:,0]
            pred = pred_img[i,:,:,0]

            masked_img = cmap(img)
            masked_img[mask_rgb==0] = 255 # 
            titles = ['Masked','Predicted','Original']
            # colors = ['g','b','r']
            xs = [masked_img,cmap(pred),cmap(img)]
            # pcv = [calcPCV1(masked[i,:,:,0]),calcPCV1(pred),calcPCV1(img)]

            _, axes = plt.subplots(1, 3, figsize=(20, 5))

            for i,x in enumerate(xs):
                axes[i].imshow(x)
                axes[i].set_title(titles[i])
                # line = pcv[i][1]
                # ce = pcv[i][0]
                # axes[i].plot(line[1],line[0],colors[i]+'-')
                # axes[i].scatter(ce[1],ce[0],c=colors[i])
                # axes[i].set_xlim(0,img_w-1)
                # axes[i].set_ylim(img_h-1,0)

            plt.savefig(os.path.join(path, 'img{}_epoch{}.png'.format(i, epoch)))
            plt.close()

    # pdb.set_trace()
    # Build the model
    if args.positionalKernel:
        model = PKConvUnet(img_rows=img_h,img_cols=img_w,lr=args.lr,use_site=useSite,exist_point_file=SEA_PATH,
        exist_flag=True,posEmbChan=args.posEmbChan,opeType=args.posKernelOpe,PKConvlayer=args.PKlayers,
        encFNum=args.encFNum,sCNNFNum=args.sCNNFNum,eachChannel=args.eachChannel,useSiteCNN=args.useSiteCNN,
        sCNNBias=args.sCNNBias,sCNNActivation=args.sCNNAct,sCNNSinglePath=args.sCNNSinglePath, useSiteNormalize=args.useSiteNorm)
    elif args.sitePConv:
        model = sitePConvUnet(img_rows=img_h,img_cols=img_w,use_site=useSite,exist_point_file=SEA_PATH,
        exist_flag=True,posEmbChan=args.posEmbChan)
    else:
        model = PConvUnet(img_rows=img_h,img_cols=img_w,KLthre=args.KLthre,isUsedKL= args.KL,
        isUsedHistKL=args.histKL,isUsedLLH=args.LLH,LLHonly=args.LLHonly,exist_point_file=SEA_PATH,exist_flag=True,
        histFSize=args.histFilterSize,truefirst=args.truefirst,predfirst=args.predfirst,KLbias=args.KLbias,KLonly=args.KLonly)   


    # Loading of checkpoint（デフォルトではロードせずに初めから学習する）
    if args.checkpoint:
        if args.stage == 'train':
            model.load(args.checkpoint)
        elif args.stage == 'finetune':
            model.load(args.checkpoint, train_bn=False, lr=args.lr)

    # callback の設定
    callbacks = [
        TensorBoard(
            log_dir=os.path.join(log_path, dataset+'_model'),
            write_graph=False
        ),
        PSNREarlyStopping(loss_path,log_path,dataset),
        TQDMCallback()
    ]

    # 学習時にテスト結果をプロットするかどうか
    if args.plotTest:
        callbacks.append(
            LambdaCallback(on_epoch_end=lambda epoch,logs: plot_callback(model, test_path,epoch))
        )


    # モデルの学習
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=valid_Num,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks
    )

