import tensorflow as tf
import pandas as pd
import os
import numpy as np

def _parse_embedding_function(img_filename, txt_filename, embedding_filename):
        """Parse function read embeddings.

        It will just read embeddings and do nothing to other parameters.

        Args:
            img_filename: String(Tensor). Image filename.
            txt_filename: String(Tensor). Text description filename.
            embedding: String(Tensor). Text embedding filename. 

        Returns:
            image_resized: Tensor. Image resized to 229x229.
            text_embed: Tensor (of full context bytes). Text description.  
            embedding: Numpy array. Text embedding. 
        """

        embedding_arr = np.load(embedding_filename.decode())

        return img_filename, txt_filename, embedding_arr

def _parse_img_and_txt_function(img_filename, txt_filename, embedding):
        """Parse function to read and preprocess image, corresponding text description 
            and embedding.

        It will use tensorflow built in api to read image and pandas to read description.

        Args:
            img_filename: String. Image filename.
            txt_filename: String. Text description filename.
            embedding: Numpy array(Tensor). Text embedding. 

        Returns:
            image_resized: Tensor. Image resized to 229x229.
            text_embed: Tensor (of full context bytes). Text description.  
            embedding: Tensor. Text embedding. 
        """

        image_bytes = tf.read_file(img_filename)
        image_decoded = tf.image.decode_image(image_bytes, channels=3)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 229, 229)

        text_bytes = tf.read_file(txt_filename)

        return image_resized, text_bytes, embedding
    
class DataReader():
    def __init__(self, mode, dataset_prefix):
        ''' TODO
        
        Args:
            mode: String. 'train', 'valid' or 'test'.
            dataset_prefix: String. Where your dataset locate.
        '''
        img_path_prefix = 'birds/CUB_200_2011/images/'
        txt_path_prefix = 'birds/text_c10/'
        embed_path_prefix = 'birds/embeddings/'
        print(os.path.join(dataset_prefix, 'birds', mode, 'char-CNN-RNN-embeddings.pickle'))
        
        embeddings_pkl_filename = os.path.join(dataset_prefix, 'birds', mode, 'char-CNN-RNN-embeddings.pickle')
        #class_info_pkl_filename = os.path.join('birds', mode, 'class_info.pickle')
        file_pkl_filename = os.path.join(dataset_prefix, 'birds', mode, 'filenames.pickle')
        embeddings = pd.read_pickle(embeddings_pkl_filename)
        #class_info = pd.read_pickle(class_info_pkl_filename)
        file_filenames = pd.read_pickle(file_pkl_filename)
        
        self.img_pathes = [os.path.join(dataset_prefix, img_path_prefix, f+'.jpg') for f in file_filenames]
        self.txt_pathes = [os.path.join(dataset_prefix, txt_path_prefix, f+'.txt') for f in file_filenames]
        self.embed_pathes = [os.path.join(dataset_prefix, embed_path_prefix, f+'.npy') for f in file_filenames]
        
        self.dataset = None
        self.batch_size = None
        self.num_epochs = None
        
    def configure(self, batch_size, num_epochs, workers=6, shuffle=True, buffer_size=10000):
        ''' Dataset reader configuration.
        
        Args:
            batch_size: Integer.
            num_epochs: Integer.
            workers: Integer.
        '''
        # Public member
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.img_pathes, self.txt_pathes, self.embed_pathes)
        )
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.workers = workers
        self.shuffle = True
        self.buffer_size = buffer_size
        self.dataset = self.dataset.map(
            map_func = lambda i, t, e: tuple(tf.py_func(
                    _parse_embedding_function, [i ,t, e], [i.dtype, t.dtype, tf.float32])),
            num_parallel_calls = self.workers
        )
        self.dataset = self.dataset.map(
            map_func = _parse_img_and_txt_function,
            num_parallel_calls = self.workers
        )
        if self.shuffle is True:
            self.dataset = self.dataset.shuffle(buffer_size=self.buffer_size)
        self.dataset = self.dataset.repeat(self.num_epochs)
        self.dataset = self.dataset.batch(self.batch_size)
    
    def get_next(self):
        self.iterator = self.dataset.make_one_shot_iterator()
        next_element = self.iterator.get_next()
        return next_element
        
# Testing module
if __name__ == '__main__':
    mode = 'train'
    dataset_prefix = '/home/meirtz/datasets/CuoNet/raw_data/'
    
    idx = np.random.randint(0, 100) # Index for testing
    batch_size = 64
    num_epochs = 100
    
    reader = DataReader(mode, dataset_prefix)
    reader.configure(batch_size=batch_size, num_epochs=num_epochs)
    next_element = reader.get_next()
    
    with tf.Session() as sess:
        while True:
          try:
            imgs, txts, embeds = sess.run(next_element)
            # Do something...
          except tf.errors.OutOfRangeError:
            print('End of dataset.')
    
    
        