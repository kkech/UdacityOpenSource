import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import argparse



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map,width,height,args):

  """Visualizes input image, segmentation map and overlay view."""

  seg_image = label_to_color_image(seg_map).astype(np.uint8)

  sample=seg_image
  black_pixels_mask = np.all(sample == [0, 0, 0], axis=-1)
  img = np.asarray(image).astype(np.uint8)
  img[black_pixels_mask] = [255, 255, 255]
  cropped_input_img=img.copy()
  create_bin_mask = img
  create_bin_mask[black_pixels_mask] = [255, 255, 255]
  create_bin_mask[black_pixels_mask == False] = [0, 0, 0]
  background = Image.open(args.background_path)
  background = background.resize((img.shape[1],img.shape[0]), Image.ANTIALIAS)
  background = cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB)
  crop_background = np.array(background)
  crop_background[black_pixels_mask==False] = [0, 0, 0]
  original_img=np.asarray(image).astype(np.uint8)
  original_img[black_pixels_mask] = [0, 0, 0]
  final_image = crop_background + original_img
  img_pth=args.image_path
  cropped_img_pth='./cropped_image/'+ (img_pth.rsplit('/', 1)[1])
  #save image to the destination
  Image.fromarray(cropped_input_img).resize((width, height), Image.ANTIALIAS).save(cropped_img_pth)
  #save pasted image
  pasted_image_path='./pasted_image/'+ (img_pth.rsplit('/', 1)[1])
  Image.fromarray(final_image).resize((width, height), Image.ANTIALIAS).save(pasted_image_path)


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def run_visualization(args):
    """Inferences DeepLab model and visualizes result."""
    try:
        # f = urllib.request.urlopen(url)
        # jpeg_str = f.read()
        # original_im = Image.open(BytesIO(jpeg_str))


        original_im = Image.open(args.image_path)
        width, height = original_im.size

    except IOError:
        print('Cannot retrieve image. Please check url: ' + args.image_path)
        return

    print('running deeplab on image %s...' % args.image_path)
    resized_im, seg_map = MODEL.run(original_im)

    vis_segmentation(resized_im, seg_map,width,height,args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Crop you image\
             by giving the path to your image\
              ")
    parser.add_argument("--image_path",
                        required=True,
                        help="full path to your image")

    parser.add_argument("--background_path",
                        required=True,
                        help="full path to the background image")

    args = parser.parse_args()

    MODEL_NAME = 'xception_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = 'deeplab_model.tar.gz'

    # model_dir = tempfile.mkdtemp()

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'model_dir')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        tf.gfile.MakeDirs(final_directory)

        download_path = os.path.join(final_directory, _TARBALL_NAME)
        print('downloading model, this might take a while...')
        try:

            urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],download_path)
        except Exception as e:
            print(e)

        print('download completed! loading DeepLab model...')

        MODEL = DeepLabModel(download_path)
        print('model loaded successfully!')
    else:
        MODEL = os.path.join(current_directory, r'model_dir') + '/deeplab_model.tar.gz'
        MODEL = DeepLabModel(MODEL)
        print('model loaded successfully!')

    SAMPLE_IMAGE = 'image3'  # @param ['image1', 'image2', 'image3']
    IMAGE_URL = ''  # @param {type:"string"}


    # _SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
    #                'deeplab/g3doc/img/%s.jpg?raw=true')




    # image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE'
    image_url = ''
    run_visualization(args)







