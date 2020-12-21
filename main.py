import os
import gc
import datetime
import numpy as np
import cv2
import pickle
import glob
import pdb

from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback,Callback
from keras_tqdm import TQDMCallback
from keras.models import Model

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from libs.pconv_model import PConvUnet
from libs.util import MaskGenerator
from PIL import Image

# Sample call

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
    parser.add_argument('-KLthre', '--KLthre',type=float, default=0.4,help='threshold value of KLloss')
    parser.add_argument('-KL','--KL',action='store_true',help="Flag for using KL-loss function")
    parser.add_argument('-histKL','--histKL',action='store_true',help="Flag for using spatial Histogram KL-loss function")
    parser.add_argument('-epochs','--epochs',type=int,default=100,help='training epoch')
        
    return  parser.parse_args()

# ディレクトリから画像を読み込み学習に使用する為のクラス（マスクの変形なども可能）
class AugmentingDataGenerator(ImageDataGenerator):
    """Wrapper for ImageDataGenerator to return mask & image"""
    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, color_mode='grayscale', *args, **kwargs) 
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            # Get augmentend image samples
            ori = next(generator)
            
            # Get masks for each image sample            
            mask = np.stack([
                mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )[:,:,:,np.newaxis]

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 0
            gc.collect()
            yield [masked, mask], ori

def Generator(imagePath, maskPath, batchSize):
    data = pickle.load(open(imagePath,"rb"))
    mask = pickle.load(open(maskPath,"rb"))

    while True:
        for B in range(0, len(data), batchSize):
            ori = data["images"][B:B+batchSize] # original images
            # Y = data["labels"][B:B+batchSize] # labels
            mask = mask[B:B+batchSize] # masks
            masked = mask*ori # masked image
            yield [masked, mask], ori # Returning mask

# 損失の監視・保存に用いるクラス
class LossHistory(Callback):
    def __init__(self,savepath):
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        self.path = os.path.join(savepath,"training_losses.pickle")
        self.histKL = []
        self.KL = []
        self.PSNR = []
        self.totalLoss = []

    def on_batch_end(self, batch, logs={}): # バッチ終了時に呼び出される
        self.histKL.append(logs.get("loss_spatialHistKL"))
        self.KL.append(logs.get("loss_KL"))
        self.PSNR.append(logs.get("PSNR"))
        self.totalLoss.append(logs.get("loss"))

    def on_train_end(self, logs={}): # 学習終了時に全損失をpickleとして保存
        types = ["Total","PSNR","KL","spaHistKL"]
        summary = {
            "epochs":epochs,
            "steps_per_epoch":steps_per_epoch,
            types[0]:np.array(self.totalLoss),
            types[1]:np.array(self.PSNR),
            types[2]:np.array(self.KL),
            types[3]:np.array(self.histKL)
        }

        with open(self.path,"wb") as f:
            pickle.dump(summary,f)
        
        for lossName in types:
            loss = summary[lossName]
            plt.plot(range(epochs*steps_per_epoch),loss)
            plt.xlabel('Iteration (1epoch={}ite)'.format(steps_per_epoch))
            plt.ylabel(lossName)
            plt.title(args.experiment)
            plt.savefig(os.path.join(loss_path,lossName+".png"))
            plt.close()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    if args.stage == 'finetune' and not args.checkpoint:
        raise AttributeError('If you are finetuning your model, you must supply a checkpoint file')

    # 実験に用いるディレクトリを作成
    experiment_path = ".{0}experiment{0}{1}_logs".format(os.sep,args.experiment)
    loss_path = "{0}{1}losses".format(experiment_path,os.sep)
    log_path = "{0}{1}logs".format(experiment_path,os.sep)
    test_path = "{0}{1}test_samples".format(experiment_path,os.sep)
    for DIR in [experiment_path,loss_path,log_path,test_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    epochs = args.epochs

    dataset = args.dataset # データセットのディレクトリ
    dspath = ".{0}data{0}{1}{0}".format(os.sep,dataset)

    TRAIN_PICKLE = dspath+"train.pickle" if args.train=="" else args.train
    TRAIN_MASK_PICKLE = dspath+"train_mask.pickle" if args.trainmask=="" else args.trainmask
    VALID_PICKLE = dspath+"valid.pickle" if args.validation=="" else args.valid
    VALID_MASK_PICKLE = dspath+"valid_mask.pickle" if args.validmask=="" else args.validmask
    TEST_PICKLE = dspath+"test.pickle" if args.test=="" else args.test
    TEST_MASK_PICKLE = dspath+"test_mask.pickle" if args.testmask=="" else args.testmask

    # SEA_PATH = ".{0}data{0}sea.png".format(os.sep)
    SEA_PATH = ""
    train_Num = pickle.load(open(TRAIN_PICKLE,"rb"))["images"].shape[0] # 画像の枚数をカウント
    valid_Num = pickle.load(open(VALID_PICKLE,"rb"))["images"].shape[0]
    img_w = 512
    img_h = 512
    shape = (img_h, img_w)

    # バッチサイズはメモリサイズに合わせて調整が必要
    batchsize = 5 # バッチサイズ
    steps_per_epoch = train_Num//batchsize # 1エポック内のiteration数

    train_generator = Generator(TRAIN_PICKLE,TRAIN_MASK_PICKLE,batchsize) # Create training generator
    val_generator = Generator(VALID_PICKLE,VALID_MASK_PICKLE,batchsize) # Create validation generator
    test_generator = Generator(TEST_PICKLE,TEST_MASK_PICKLE,batchsize) # Create testing generator

    # Pick out an example to be send to test samples folder
    test_data = next(test_generator)
    (masked, mask), ori = test_data
    mask_rgb = np.tile(np.reshape(mask[0],[shape[0],shape[1],1]),(1,1,3)) #カラーで可視化する際に用いるマスク

    # 学習途中のテスト結果が必要ない場合はmodel.fit_generator内で
    # 以下の plot_callback() を呼び出している行のコメントアウトをして下さい
    def plot_callback(model, path):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""
        
        # Get samples & Display them        
        pred_img = model.predict([masked, mask])
        pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Clear current output and display test images
        for i in range(ori.shape[0]):
            img = ori[i,:,:,0]
            pred = pred_img[i,:,:,0]

            masked_img = cmap(img)
            masked_img[mask_rgb==0] = 255 # 
            titles = ['Masked','Predicted','Original']
            colors = ['g','b','r']
            xs = [masked_img,cmap(pred),cmap(img)]
            pcv = [calcPCV1(masked[i,:,:,0]),calcPCV1(pred),calcPCV1(img)]

            _, axes = plt.subplots(1, 3, figsize=(20, 5))

            for i,x in enumerate(xs):
                axes[i].imshow(x)
                axes[i].set_title(titles[i])
                line = pcv[i][1]
                ce = pcv[i][0]
                axes[i].plot(line[1],line[0],colors[i]+'-')
                axes[i].scatter(ce[1],ce[0],c=colors[i])
                axes[i].set_xlim(0,511)
                axes[i].set_ylim(511,0)

            plt.savefig(os.path.join(path, 'img_{}_{}.png'.format(i, pred_time)))
            plt.close()

    # 損失の記録
    history = LossHistory(loss_path)
    
    # Build the model
    model = PConvUnet(img_rows=img_h,img_cols=img_w,KLthre=args.KLthre,isUsedKL= args.KL,isUsedHistKL=args.histKL,exist_point_file=SEA_PATH,exist_flag=True)
    
    # Loading of checkpoint（デフォルトではロードせずに初めから学習する）
    if args.checkpoint:
        if args.stage == 'train':
            model.load(args.checkpoint)
        elif args.stage == 'finetune':
            model.load(args.checkpoint, train_bn=False, lr=0.00005)

    # Fit model
    model.fit_generator(
        train_generator, 
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=valid_Num,
        epochs=epochs,
        verbose=0,
        callbacks=[
            TensorBoard(
                log_dir=os.path.join(log_path, dataset+'_model'),
                write_graph=False
            ),
            ModelCheckpoint(
                os.path.join(log_path, dataset+'_model', 'weights.{epoch:02d}-{loss:.2f}.h5'),
                monitor='val_loss', 
                save_best_only=False, 
                save_weights_only=True,
                period = 10
            ),
            LambdaCallback(on_epoch_end=lambda epoch, logs: plot_callback(model, test_path)), # 学習中のテスト出力が不必要ならこの行をコメントアウト
            history,
            TQDMCallback()
        ]
    )
        
