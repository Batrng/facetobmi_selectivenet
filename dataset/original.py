import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
p = pickle.load(open('C:/Users/nguyen/Downloads/test.pickle', 'rb'), encoding='latin1')
n_items = p['height'].shape[0]
for i, img_id in enumerate(p["image_id"]):
        if str(img_id.decode('latin1')) == "rm998672640":
            index = i
            print(i)
            break

print('n_items, ', n_items)
print(p.keys())
#index = index
#print(index)
image_id = str(p["image_id"][index].decode('latin1'))
height = p["height"][index]
gender = p["gender"][index]
actor_id = p["actor_id"][index].decode('latin1')
pose = p["pose_2d"][index]

img = mpimg.imread('C:/Users/nguyen/Downloads/rm998672640.jpg')
plt.imshow(img)
pose_n = pose.reshape(-1,3) * [img.shape[1], img.shape[0], 1]

plt.imshow(img)
plt.scatter(pose_n[:,0], pose_n[:,1])

print("Gender: ", "male" if gender else "female")
print("Height: ", height)
print("Check https://www.imdb.com/name/{}/ for more information on the person".format(actor_id))
print(f"See the image on imdb: https://www.imdb.com/name/{actor_id}/mediaviewer/{image_id}")
min_x = np.min(pose_n[:,0][pose_n[:,0]!=0])
max_x = np.max(pose_n[:,0][pose_n[:,0]!=0])
min_y = np.min(pose_n[:,1][pose_n[:,1]!=0])
max_y = np.max(pose_n[:,1][pose_n[:,1]!=0])
print(min_x, max_x, min_y, max_y)
plt.imshow(img[int(min_y):int(max_y), int(min_x):int(max_x), :])