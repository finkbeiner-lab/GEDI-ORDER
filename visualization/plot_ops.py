import matplotlib.pyplot as plt
import numpy as np
import param_gedi as param
import imageio
import os


class Plotty:
    def __init__(self, model_timestamp):
        self.p = param.Param()
        self.model_timestamp = model_timestamp

    def show_batch(self, image_batch, label_batch):
        """Vgg montage"""
        plt.figure(figsize=(4, 4))
        for n in range(self.p.BATCH_SIZE):
            ax = plt.subplot(5, 5, n + 1)
            img = image_batch[n]
            print('mx', np.max(img))
            print(np.min(img))
            b, g, r = img[..., 0], img[..., 1], img[..., 2]
            b = b + self.p.VGG_MEAN[0]
            g = g + self.p.VGG_MEAN[1]
            r = r + self.p.VGG_MEAN[2]
            rgb = np.dstack((r, g, b))
            print('mx rgb', np.max(rgb))
            print(np.min(rgb))
            rgb = np.uint8(rgb)
            plt.imshow(rgb)
            # plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
            plt.axis('off')
        plt.show()

    def make_montage(self, im_lbl_lst, title, size=16):
        side = int(np.sqrt(size))
        montage = np.zeros((self.p.target_size[0] * side,
                            self.p.target_size[1] * side,
                            self.p.target_size[2]), dtype=np.uint8)
        step = self.p.target_size[0]
        chklbls = []
        partitions = len(im_lbl_lst) // size
        for k in range(partitions):
            savepath = os.path.join(self.p.confusion_dir, self.model_timestamp + '_' + title + '_' + str(k) + '.tif')

            im_split = im_lbl_lst[k * size: (k + 1) * size]
            for cnt, lst in enumerate(im_split):
                i = int(cnt % np.sqrt(size))
                j = int(cnt // np.sqrt(size))
                img = lst[0]
                lbl = lst[1]
                chklbls.append(np.argmax(lbl))
                b, g, r = img[..., 0], img[..., 1], img[..., 2]
                b = b + self.p.VGG_MEAN[0]
                g = g + self.p.VGG_MEAN[1]
                r = r + self.p.VGG_MEAN[2]
                rgb = np.dstack((r, g, b))
                rgb = np.uint8(rgb)
                montage[step * i:step * (i + 1), step * j: step * (j + 1), :] = rgb
            assert np.all(np.array(chklbls) == 1) or np.all(np.array(chklbls) == 0)
            imageio.imwrite(savepath, montage)
            # plt.imshow(montage)
            # plt.title(title)
            # plt.show()
