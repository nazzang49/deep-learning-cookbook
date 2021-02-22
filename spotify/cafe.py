# annoy from spotify
# https://github.com/nazzang49/annoy
# https://zsunn.tistory.com/entry/AI-Annoy-Approximate-Nearest-Neighbors-Oh-Yeah-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EC%98%88%EC%A0%9C
import os
from keras.models import Model
from PIL import Image
import numpy as np
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from annoy import AnnoyIndex

# local image dir
dir_path = "C:/cafe-img/dump/"
images = os.listdir(dir_path)

# split test
files = os.listdir("C:/cafe-img/")
print(files[0].split("_")[0])

# length of files
print(len(images))

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
        # should save file names on additional list
        valid_image_names.append(image_name)

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
print(valid_image_names)
print(valid_images)

# batch size
chunks = [get_vector(valid_images[i:i+30]) for i in range(0, len(valid_images), 30)]
vectors = np.concatenate(chunks)
print(vectors.shape)

# annoy
vector_size = 2048
index = AnnoyIndex(vector_size, 'dot')
data = []
for idx in range(len(vectors)):
  data.append({'idx': idx, 'img': valid_images[idx], 'name': valid_image_names[idx], 'vector': vectors[idx]})
  if idx <= 80:
    index.add_item(idx, vectors[idx])

index.build(50)
index.save('cafe_similarity_analysis.annoy')

# evaluation
load_index = AnnoyIndex(vector_size, 'dot')
load_index.load('cafe_similarity_analysis.annoy')

result = load_index.get_nns_by_vector(data[77]['vector'], 20)
print(result)

# img = Image.new('RGB', (8 * 75, 8 * 75), (180, 180, 180))
# img.paste(data[77]['img'])
# img.show()

for idx in result:
    print(data[idx]['name'])


