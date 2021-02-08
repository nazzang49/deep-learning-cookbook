# using online wikimedia image
from sklearn.neighbors import NearestNeighbors
import random
import os
import base64
from sklearn.decomposition import TruncatedSVD
from keras.models import Model
from hashlib import md5
import pickle
try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote
from PIL import Image
import requests
try:
    from io import BytesIO
except ImportError:
    from io import StringIO as BytesIO
from IPython.display import HTML
import numpy as np
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
import json

query = """SELECT DISTINCT ?pic
WHERE
{
    ?item wdt:P31 ?class . 
    ?class wdt:P18 ?pic
}"""

url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
data = requests.get(url, params={'query': query, 'format': 'json'}).text
data = json.loads(data, strict=False)

images = [x['pic']['value'] for x in data['results']['bindings']]
len(images), random.sample(images, 10)

IMAGE_DIR = 'wp_images'
if not os.path.isdir(IMAGE_DIR):
    os.mkdir(IMAGE_DIR)

def center_crop_resize(img, new_size):
    w, h = img.size
    s = min(w, h)
    y = (h - s) // 2
    x = (w - s) // 2
    img = img.crop((x, y, s, s))
    return img.resize((new_size, new_size))

def fetch_image(image_cache, image_url):
    image_name = image_url.rsplit('/', 1)[-1]
    local_name = image_name.rsplit('.', 1)[0] + '.jpg'
    local_path = os.path.join(image_cache, local_name)
    if os.path.isfile(local_path):
        img = Image.open(local_path)
        img.load()
        return center_crop_resize(img, 299)
    image_name = unquote(image_name).replace(' ', '_')
    m = md5()
    m.update(image_name.encode('utf8'))
    c = m.hexdigest()
    for prefix in 'http://upload.wikimedia.org/wikipedia/en', 'http://upload.wikimedia.org/wikipedia/commons':
        url = '/'.join((prefix, c[0], c[0:2], image_name))
        r = requests.get(url)
        if r.status_code != 404:
            try:
                img = Image.open(BytesIO(r.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(local_path)
                return center_crop_resize(img, 299)
            except IOError:
                pass
    return None

fetch_image(IMAGE_DIR, images[0])

valid_images = []
valid_image_names = []
for image_name in tqdm(images):
    img = fetch_image(IMAGE_DIR, image_name)
    if img:
        valid_images.append(img)
        valid_image_names.append(image_name)

len(valid_images)

# https://keras.io/api/applications/inceptionv3/
base_model = InceptionV3(weights='imagenet', include_top=True)
base_model.summary()

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

chunks = [get_vector(valid_images[i:i+256]) for i in range(0, len(valid_images), 256)]
vectors = np.concatenate(chunks)
print(vectors.shape)

nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vectors)

with open('../data/image_similarity.pck', 'wb') as fout:
    pickle.dump({'nbrs': nbrs,
                 'image_names': valid_image_names
                 }, fout)

cat = get_vector(Image.open('data/cat.jpg'))
distances, indices = nbrs.kneighbors(cat)

if True:
    images = [Image.open('../data/cat.jpg')]
    target_size = int(max(model.input.shape[1:]))
    images = [img.resize((target_size, target_size), Image.ANTIALIAS) for img in images]
    np_imgs = [image.img_to_array(img) for img in images]
    pre_processed = preprocess_input(np.asarray(np_imgs))
    x = model.predict(pre_processed)

print(pre_processed)

html = []
for idx, dist in zip(indices[0], distances[0]):
    b = BytesIO()
    valid_images[idx].save(b, format='jpeg')
    html.append("<img src='data:image/jpg;base64,{0}'/>".format(base64.b64encode(b.getvalue()).decode('utf-8')))
HTML(''.join(html))

b = BytesIO()
valid_images[0].save(b, format='png')
HTML("<img src='data:image/png;base64,{0}'/>".format(base64.b64encode(b.getvalue()).decode('utf-8')))

nbrs64 = NearestNeighbors(n_neighbors=64, algorithm='ball_tree').fit(vectors)
distances64, indices64 = nbrs64.kneighbors(cat)

vectors64 = np.asarray([vectors[idx] for idx in indices64[0]])

svd = TruncatedSVD(n_components=2)
vectors64_transformed = svd.fit_transform(vectors64)
vectors64_transformed.shape

img64 = Image.new('RGB', (8 * 75, 8 * 75), (180, 180, 180))

mins = np.min(vectors64_transformed, axis=0)
maxs = np.max(vectors64_transformed, axis=0)
xys = (vectors64_transformed - mins) / (maxs - mins)

for idx, (x, y) in zip(indices64[0], xys):
    x = int(x * 7) * 75
    y = int(y * 7) * 75
    img64.paste(valid_images[idx].resize((75, 75)), (x, y))
