import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 

def addpadding(image, image_id, target_size=(256, 256)):
    # Get the original dimensions
    original_h, original_w = image.shape[:2]

    # Calculate the scale factor
    scale = min(target_size[0] / original_h, target_size[1] / original_w)
    new_size = (int(original_w * scale), int(original_h * scale))
    # Resize the image
    resized_image = cv2.resize(image, new_size)

    # Create a square canvas of the target size with black padding
    padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    # Calculate the position to place the resized image
    y_offset = (target_size[0] - new_size[1]) // 2
    x_offset = (target_size[1] - new_size[0]) // 2

    # Place the resized image on the canvas
    padded_image[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized_image
    cv2.imwrite("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/fullbody/"+image_id, padded_image)
    

def croppedBody(index):
    p = pickle.load(open('train.pickle', 'rb'), encoding='latin1')
    #n_items = p['height'].shape[0]
    #print('n_items, ', n_items)
    #print(p.keys()) 
    '''
    for i, img_id in enumerate(p["image_id"]):
        if str(img_id.decode('latin1')) == "rm1364189184":
            index = i
            break
    print(index)
    '''
    
    image_id = str(p["image_id"][index].decode('latin1'))
    #height = p["height"][index]
    #gender = p["gender"][index]
    #actor_id = p["actor_id"][index].decode('latin1')
    pose = p["pose_2d"][index]
    #poseid = p["pose_id"][index]
    img = mpimg.imread("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/imdb_images/"+image_id + '.jpg')
    
    pose_n = pose.reshape(-1,3) * [img.shape[1], img.shape[0], 1]

    #landmark_n = landmark.reshape(-1,2) * [img.shape[1], img.shape[0]]
    #plt.scatter(pose_n[:,0], pose_n[:,1])
    #plt.scatter(landmark_n[0], landmark_n[1], color='red')

    min_x = np.min(pose_n[:,0][pose_n[:,0]!=0])
    max_x = np.max(pose_n[:,0][pose_n[:,0]!=0])
    min_y = np.min(pose_n[:,1][pose_n[:,1]!=0])
    max_y = np.max(pose_n[:,1][pose_n[:,1]!=0])
    #print(min_x, max_x, min_y, max_y)
    img=img[int(min_y):int(max_y), int(min_x):int(max_x), :]
    addpadding(img, image_id)
    #plt.imshow(addpadding(img, landmark_n))
    #plt.show()

def croppedBodywholeimgheight(index):
    p = pickle.load(open('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/train.pickle', 'rb'), encoding='latin1')
    #index=64758
    image_id = str(p["image_id"][index].decode('latin1'))
    pose = p["pose_2d"][index]
    img = mpimg.imread('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/imdb_images/' + image_id + '.jpg')
    
    pose_n = pose.reshape(-1,3) * [img.shape[1], img.shape[0], 1]
    #plt.scatter(pose_n[:,0], pose_n[:,1])

    min_x = np.min(pose_n[:,0][pose_n[:,0]!=0])
    max_x = np.max(pose_n[:,0][pose_n[:,0]!=0])
    min_y = np.min(0)
    max_y = np.max(pose_n[:,1][pose_n[:,1]!=0])
    img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
    cv2.imwrite("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/preface/" + image_id + ".jpg", img)
    #addpadding(img, pose_n)
    #plt.imshow(img)
    #plt.show()
    return image_id



def croppedFace(image_id):
    img = cv2.imread("/home/nguyenbt/nobackup/data/height/preface/" + image_id)

    # Detect face using a face detection model (e.g., OpenCV Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Assuming you take the first detected face (you can find the one closest to the head joint)
    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        break  # For demonstration, just use the first face found
    
    # Display or save the cropped face
    cv2.imshow("Cropped Face", face_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #croppedBodywholeimgheight()
    #croppedFace()
    #croppedBody()
    p = pickle.load(open('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/train.pickle', 'rb'), encoding='latin1')
    for i, img_id in enumerate(p["image_id"]):
        index = i
        croppedBodywholeimgheight(i)
        croppedBody(i)
        

