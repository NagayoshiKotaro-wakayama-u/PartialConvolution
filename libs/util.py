import os
from random import randint, seed
import numpy as np
import cv2
import pdb
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback,Callback


def rangeError(pre,tru,domain=[-1.0,0.0],opt="MA"): # 欠損部含めた誤差 pred:予測値, true:真値 , domain:値域(domain[0]< y <=domain[1])
    # ある値域の真値のみで誤差を測る
    domain = [domain[0],domain[1]] # normalyse
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

def nonhole(x,hole): # 欠損部以外の値を取り出す
    shape = x.shape
    flatt = np.reshape(x,(np.product(shape)))
    holes = np.reshape(hole,(np.product(shape)))
    tmp = []
    x,y = [],[]
    for pix,hole,ite in zip(flatt,holes,[i for i in range(flatt.shape[0])]):
        if np.sum(hole) < 1e-10:
            continue
        tmp.append(pix)

    return np.array(tmp)

def cmap(x,exist_rgb=None,sta=[222,222,222],end=[255,0,0]): #x:gray-image([w,h]) , sta,end:[B,G,R]
    vec = np.array(end) - np.array(sta)
    res = []
    for i in range(x.shape[0]):
        tmp = []
        for j in range(x.shape[1]):
            tmp.append(np.array(sta)+x[i,j]*vec)
        res.append(tmp)
    res = np.array(res).astype("uint8")
    if exist_rgb != None:
        res[exist_rgb==0] = 255
    return res

def calcPCV1(x,pcv_thre): # 第一主成分を抽出
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
    vec = D[:,[np.argmax(V)]] # 第一主成分を抽出
    line = np.concatenate([vec*-256,vec*256],axis=1) + center
    return center,line

def clip(x,sta=-0.1,end=0.1): # Clip the value
    x[x<sta] = sta
    x[x>end] = end
    dist = end-sta
    res = (x-sta)/dist
    return res

def calcLabeledError(errors,labels,opt="MA"):
    # pdb.set_trace()
    labs = np.array(list(set(labels)))
    results = [[] for _ in range(labs.shape[0])] # labelの種類だけリストを作成

    for lab,error in zip(labels,errors):
        ind = np.arange(labs.shape[0])[labs==lab][0]
        results[ind].append(np.abs(error)) # ラベル別で誤差を保存
    # pdb.set_trace()

    if opt=="MA": # MAE
        results = [np.mean(res) for res in results]
    elif opt=="MS": # MSE
        results = [np.mean(np.array(res)**2) for res in results]
    elif opt=="A":
        results = results

    return results,labs

def PSNR(y_pred,y_true):
    return - 10.0 * np.log(np.mean(np.square(y_pred - y_true))) / np.log(10.0)

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
        
        # ロスなどの記録
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        self.path = os.path.join(savepath,"training_losses.pickle")
        self.types = ["loss","PSNR","loss_KL","loss_spaHistKL"]    
        # training用
        self.trainingLoss = {
            self.types[0]:[],
            self.types[1]:[],
            self.types[2]:[],
            self.types[3]:[]
        }
        # validation用
        self.validationLoss = {
            self.types[0]:[],
            self.types[1]:[],
            self.types[2]:[],
            self.types[3]:[]
        }

    def on_batch_end(self, batch, logs={}): # バッチ終了時に呼び出される
        for t in self.types:
            self.trainingLoss[t].append(logs.get(t))

    def on_epoch_end(self, epoch, logs=None):
        # 検証ロスの保存
        self.now_epoch += 1
        for t in self.types:
            self.validationLoss[t].append(logs.get("val_"+t))

        self.epochs_since_last_save += 1
        val_PSNR = logs['val_PSNR']
        self.history_val_PSNR = np.append(self.history_val_PSNR,val_PSNR)
        # pdb.set_trace()

        # 検証データのPSNRの最大値を取得
        if val_PSNR > self.best_val_PSNR:
            self.best_val_PSNR = val_PSNR
            
        # 最大値より 1.5 以上小さいと終了
        if self.best_val_PSNR - 1.5 > val_PSNR: 
            self.model.stop_training = True
            self._save_model(epoch=epoch, logs=logs)
            self.on_train_end()
            sys.exit()

        # 10エポック以降で過去5エポックの最大値と比べてPSNRの変化が0.01以下なら終了
        if (epoch+1) >= 10:
            if self.best_val_PSNR < self.history_val_PSNR[:-5].max() + 0.01: 
                self.model.stop_training = True
                self._save_model(epoch=epoch, logs=logs)
                self.on_train_end()
                sys.exit()

    def on_train_end(self,logs=None):
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
            plt.plot(range(epochs*steps_per_epoch),loss)
            plt.xlabel('Iteration (1epoch={}ite)'.format(steps_per_epoch))
            plt.ylabel(lossName)
            plt.title(args.experiment)
            plt.savefig(os.path.join(loss_path,lossName+".png"))
            plt.close()

            loss = summary["val_"+lossName]
            plt.plot(range(epochs),loss)
            plt.xlabel('Epoch')
            plt.ylabel("val_"+lossName)
            plt.title(args.experiment)
            plt.savefig(os.path.join(loss_path,"val_"+lossName+".png"))
            plt.close()

class MaskGenerator():

    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training
        
        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width
        
        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))        

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")
        
        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
            
        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
        
        return 1-img

    def _load_mask(self, rotation=False, dilation=False, cropping=False):
        """Loads a mask from disk, and optionally augments it"""

        # Read image
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]),cv2.IMREAD_GRAYSCALE)
        
        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            
        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8) 
            mask = cv2.erode(mask, kernel, iterations=1)
            
        # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y+self.height, x:x+self.width]

        return (mask > 1).astype(np.uint8)

    def sample(self, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask()
        else:
            return self._generate_mask()

class ImageChunker(object): 
    
    def __init__(self, rows, cols, overlap):
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
    
    def perform_chunking(self, img_size, chunk_size):
        """
        Given an image dimension img_size, return list of (start, stop) 
        tuples to perform chunking of chunk_size
        """
        chunks, i = [], 0
        while True:
            chunks.append((i*(chunk_size - self.overlap/2), i*(chunk_size - self.overlap/2)+chunk_size))
            i+=1
            if chunks[-1][1] > img_size:
                break
        n_count = len(chunks)        
        chunks[-1] = tuple(x - (n_count*chunk_size - img_size - (n_count-1)*self.overlap/2) for x in chunks[-1])
        chunks = [(int(x), int(y)) for x, y in chunks]
        return chunks
    
    def get_chunks(self, img, scale=1):
        """
        Get width and height lists of (start, stop) tuples for chunking of img.
        """
        x_chunks, y_chunks = [(0, self.rows)], [(0, self.cols)]        
        if img.shape[0] > self.rows:
            x_chunks = self.perform_chunking(img.shape[0], self.rows)
        else:
            x_chunks = [(0, img.shape[0])]
        if img.shape[1] > self.cols:
            y_chunks = self.perform_chunking(img.shape[1], self.cols)
        else:
            y_chunks = [(0, img.shape[1])]
        return x_chunks, y_chunks    
    
    def dimension_preprocess(self, img, padding=True):
        """
        In case of prediction on image of different size than 512x512,
        this function is used to add padding and chunk up the image into pieces
        of 512x512, which can then later be reconstructed into the original image
        using the dimension_postprocess() function.
        """
    
        # Assert single image input
        assert len(img.shape) == 3, "Image dimension expected to be (H, W, C)"
    
        # Check if we are adding padding for too small images
        if padding:
            
            # Check if height is too small
            if img.shape[0] < self.rows:
                padding = np.ones((self.rows - img.shape[0], img.shape[1], img.shape[2]))
                img = np.concatenate((img, padding), axis=0)
    
            # Check if width is too small
            if img.shape[1] < self.cols:
                padding = np.ones((img.shape[0], self.cols - img.shape[1], img.shape[2]))
                img = np.concatenate((img, padding), axis=1)
    
        # Get chunking of the image
        x_chunks, y_chunks = self.get_chunks(img)
    
        # Chunk up the image
        images = []
        for x in x_chunks:
            for y in y_chunks:
                images.append(
                    img[x[0]:x[1], y[0]:y[1], :]
                )
        images = np.array(images)        
        return images
    
    def dimension_postprocess(self, chunked_images, original_image, scale=1, padding=True):
        """
        In case of prediction on image of different size than 512x512,
        the dimension_preprocess  function is used to add padding and chunk 
        up the image into pieces of 512x512, and this function is used to 
        reconstruct these pieces into the original image.
        """
    
        # Assert input dimensions
        assert len(original_image.shape) == 3, "Image dimension expected to be (H, W, C)"
        assert len(chunked_images.shape) == 4, "Chunked images dimension expected to be (B, H, W, C)"
        
        # Check if we are adding padding for too small images
        if padding:
    
            # Check if height is too small
            if original_image.shape[0] < self.rows:
                new_images = []
                for img in chunked_images:
                    new_images.append(img[0:scale*original_image.shape[0], :, :])
                chunked_images = np.array(new_images)
    
            # Check if width is too small
            if original_image.shape[1] < self.cols:
                new_images = []
                for img in chunked_images:
                    new_images.append(img[:, 0:scale*original_image.shape[1], :])
                chunked_images = np.array(new_images)
            
        # Put reconstruction into this array
        new_shape = (
            original_image.shape[0]*scale,
            original_image.shape[1]*scale,
            original_image.shape[2]
        )
        reconstruction = np.zeros(new_shape)
            
        # Get the chunks for this image    
        x_chunks, y_chunks = self.get_chunks(original_image)
        
        i = 0
        s = scale
        for x in x_chunks:
            for y in y_chunks:
                
                prior_fill = reconstruction != 0
                chunk = np.zeros(new_shape)
                chunk[x[0]*s:x[1]*s, y[0]*s:y[1]*s, :] += chunked_images[i]
                chunk_fill = chunk != 0
                
                reconstruction += chunk
                reconstruction[prior_fill & chunk_fill] = reconstruction[prior_fill & chunk_fill] / 2
    
                i += 1
        
        return reconstruction
