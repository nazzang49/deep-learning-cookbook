# using local image
from sklearn.neighbors import NearestNeighbors
import os
from sklearn.decomposition import TruncatedSVD
from keras.models import Model
import pickle
try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote
from PIL import Image
try:
    from io import BytesIO
except ImportError:
    from io import StringIO as BytesIO
import numpy as np
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image

# local image dir
dir_path = "C:/dogs-vs-cats/dump/"
images = os.listdir(dir_path)

def center_crop_resize(img, new_size):
    w, h = img.size
    s = min(w, h)
    y = (h - s) // 2
    x = (w - s) // 2
    img = img.crop((x, y, s, s))
    return img.resize((new_size, new_size))

def fetch_image(file_name):
    try:
        img = Image.open(dir_path + file_name)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return center_crop_resize(img, 299)
    except IOError:
        pass
    return None

valid_images = []
valid_image_names = []
for image_name in tqdm(images):
    img = fetch_image(image_name)
    if img:
        valid_images.append(img)
        valid_image_names.append(image_name)

print(valid_image_names)

# https://keras.io/api/applications/inceptionv3/
base_model = InceptionV3(weights='imagenet', include_top=True)
base_model.summary()

# select pooling
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

def get_vector(img):
    if not type(img) == list:
        images = [img]
    else:
        images = img
    target_size = int(max(model.input.shape[1:]))
    images = [img.resize((target_size, target_size), Image.ANTIALIAS) for img in images]
    np_imgs = [image.img_to_array(img) for img in images]
    pre_processed = preprocess_input(np.asarray(np_imgs))
    return model.predict(pre_processed)

# example
x = get_vector(valid_images[4])
print(x.shape)

# batch size
chunks = [get_vector(valid_images[i:i+30]) for i in range(0, len(valid_images), 30)]
vectors = np.concatenate(chunks)
print(vectors.shape)

nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vectors)

with open('../data/image_similarity.pck', 'wb') as fout:
    pickle.dump({'nbrs': nbrs, 'image_names': valid_image_names}, fout)

cat = get_vector(Image.open('../data/cat.jpg'))
distances, indices = nbrs.kneighbors(cat)

if True:
    images = [Image.open('../data/cat.jpg')]
    target_size = int(max(model.input.shape[1:]))
    images = [img.resize((target_size, target_size), Image.ANTIALIAS) for img in images]
    np_imgs = [image.img_to_array(img) for img in images]
    pre_processed = preprocess_input(np.asarray(np_imgs))
    x = model.predict(pre_processed)

print(pre_processed)

nbrs64 = NearestNeighbors(n_neighbors=64, algorithm='ball_tree').fit(vectors)
distances64, indices64 = nbrs64.kneighbors(cat)
vectors64 = np.asarray([vectors[idx] for idx in indices64[0]])
print("================== NearestNeighbors ==================")
print(distances64)
print(indices64)

# print out image name
for name in indices64[0]:
    print("img name : " + valid_image_names[name])

# reducing dim
svd = TruncatedSVD(n_components=2)
vectors64_transformed = svd.fit_transform(vectors64)
print("================== TruncatedSVD ==================")
print(vectors64_transformed)
print(vectors64_transformed.shape)

# show images on 8 x 8 checker board
img64 = Image.new('RGB', (8 * 75, 8 * 75), (180, 180, 180))
mins = np.min(vectors64_transformed, axis=0)
maxs = np.max(vectors64_transformed, axis=0)
xys = (vectors64_transformed - mins) / (maxs - mins)

for idx, (x, y) in zip(indices64[0], xys):
    x = int(x * 7) * 75
    y = int(y * 7) * 75
    img64.paste(valid_images[idx].resize((75, 75)), (x, y))

img64.show()


