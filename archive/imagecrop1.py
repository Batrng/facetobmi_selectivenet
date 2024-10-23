import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import numpy as np
import cv2
import shutil
import numpy as np
import os

def addpadding(image, image_id, target_size=(256, 256)):
    # Convert grayscale image to RGB if necessary
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,) * 3, axis=-1)  # Convert to RGB

    original_h, original_w = image.shape[:2]

    # Calculate scale to resize the image to fit within target size
    scale_w = target_size[0] / original_w
    scale_h = target_size[1] / original_h
    scale = min(scale_w, scale_h)  # Maintain aspect ratio

    # Calculate new dimensions
    new_width = int(original_w * scale)
    new_height = int(original_h * scale)

    # Resize the image to fit within target dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Initialize a final image filled with black (target size)
    final_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    # Center the resized image in the final padded image
    y_offset = (target_size[0] - new_height) // 2
    x_offset = (target_size[1] - new_width) // 2

    # Place the resized image in the center of the black canvas
    final_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    # Save the final image
    #output_path = f"/home/nguyenbt/nobackup/face-to-bmi-vit/{image_id}.jpg"
    output_path = f"/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_fullbody_v4/{image_id}.jpg"
    img_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)




'''def addpadding(image, image_id, folder, target_size=(256, 256)):
    # Get the original dimensions
    if len(image.shape) == 2:  # grayscale image
        image = np.stack((image,) * 3, axis=-1)
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
    img_bgr = cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/" + folder +"/" + image_id + ".jpg", img_bgr)
    '''

def croppedBody(index):
    p = pickle.load(open('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/val.pickle', 'rb'), encoding='latin1')
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
    try:
        img = mpimg.imread("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/imdb_images/" + image_id + '.jpg')
    except Exception as e:
            print(f"Error reading image {image_id}: {e}")
            return  # Skip to the next image
    
    pose_n = pose.reshape(-1,3) * [img.shape[1], img.shape[0], 1]

    #landmark_n = landmark.reshape(-1,2) * [img.shape[1], img.shape[0]]
    #plt.scatter(pose_n[:,0], pose_n[:,1])
    #plt.scatter(landmark_n[0], landmark_n[1], color='red')

    min_x = np.min(pose_n[:,0][pose_n[:,0]!=0])
    max_x = np.max(pose_n[:,0][pose_n[:,0]!=0])
    min_y = np.min(pose_n[:,1][pose_n[:,1]!=0])
    max_y = np.max(pose_n[:,1][pose_n[:,1]!=0])
    height, width = img.shape[:2]
    #print(min_x, max_x, min_y, max_y)
    if len(img.shape) == 2:  # Grayscale image (2D array)
        img = img[int(min_y):int(max_y), :]
        if int(min_x) > int(width - max_x):
            img = img[:, int(min_x)-int(width - max_x)]
        elif int(min_x) < int(width - max_x):
            img = img[:, :int(max_x) + int(min_x)]
    elif len(img.shape) == 3:  # RGB image (3D array)
        img = img[int(min_y):int(max_y), :, :]
        if int(min_x) > int(width - max_x):
            img = img[:, int(min_x)-int(width - max_x):, :]
        elif int(min_x) < int(width - max_x):
            img = img[:, :int(max_x) + int(min_x), :]
    
    print(f"Width: {width}, Height: {height}")
    print(f"mixx: {min_x}, maxx: {width-max_x}")
    print(max_x-min_x)
    if len(img.shape) > 1:
        img = resize_image_to_target(img)
        #output_path = f"/home/nguyenbt/nobackup/face-to-bmi-vit/{image_id}_fullbody.jpg"
        #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(output_path, img_bgr)
        addpadding(img, image_id)
        #plt.imshow(addpadding(img, landmark_n))
        #plt.show()

def resize_image_to_target(image, target_size=256):
    # Get original dimensions
    
    if len(image.shape) == 2:
        # Convert grayscale to RGB by stacking along the third axis
        image = np.stack((image,) * 3, axis=-1)
    original_height, original_width = image.shape[:2]
    # Calculate scale factors for width and height
    scale_width = target_size / original_width
    scale_height = target_size / original_height

    # Choose the smaller scale factor to maintain aspect ratio
    scale = min(scale_width, scale_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

    # Example usage
    #img = cv2.imread('path_to_your_image.jpg')
    #resized_img = resize_image_to_target(img)

    # Save or display the resized image
    #cv2.imwrite('path_to_save_resized_image.jpg', resized_img)

def croppedBodywholeimgheight(index):
    p = pickle.load(open('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/train.pickle', 'rb'), encoding='latin1')
    #index=64758
    image_id = str(p["image_id"][index].decode('latin1'))
    pose = p["pose_2d"][index]
    try:
        img = mpimg.imread('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/imdb_images/' + image_id + '.jpg')
    except Exception as e:
            print(f"Error reading image {image_id}: {e}")
            return  # Skip to the next image
    
    pose_n = pose.reshape(-1,3) * [img.shape[1], img.shape[0], 1]
    #plt.scatter(pose_n[:,0], pose_n[:,1])

    min_x = np.min(pose_n[:,0][pose_n[:,0]!=0])
    max_x = np.max(pose_n[:,0][pose_n[:,0]!=0])
    min_y = np.min(0)
    max_y = np.max(pose_n[:,1][pose_n[:,1]!=0])
    if len(img.shape) == 2:  # Grayscale image (2D array)
        img = img[int(min_y):int(max_y), int(min_x):int(max_x)]
    elif len(img.shape) == 3:  # RGB image (3D array)
        img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]

    #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/preface_test/" + image_id + ".jpg", img)
    #addpadding(img, pose_n)
    #plt.imshow(img)
    #plt.show()
    #print(image_id)
    croppedFace(image_id)
    return image_id



def croppedFace(image_id):
    img = cv2.imread("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/preface_test/" + image_id + ".jpg")

    # Detect face using a face detection model (e.g., OpenCV Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces)==0:
         profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
         faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Assuming you take the first detected face (you can find the one closest to the head joint)
    if len(faces)==0:
         face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        break  # For demonstration, just use the first face found
    if len(faces)!=0:
        addpadding(face_crop, image_id, "face_test")

    # Display or save the cropped face
    #cv2.imshow("Cropped Face", face_crop)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def process_images(input_folder="/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_fullbody_v4/", output_folder="/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_fullbody_new/"):
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the source folder
    for filename in os.listdir(input_folder):
        source_file = os.path.join(input_folder, filename)
        
        # Check if the file exists in folder_z
        if os.path.isfile(source_file) and os.path.exists(os.path.join("/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_face", filename)):
            # Copy the file to the destination folder
            shutil.copy2(source_file, output_folder)
            print(f"Copied {filename} to {output_folder}")
        else:
            print(f"{filename} does not exist in {"/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_face"}, skipping.")


if __name__ == "__main__":
    process_images()
    #croppedBodywholeimgheight()
    #croppedFace()
    #croppedBody(25441) #43192
    #croppedBody(43192) #43192
    #roppedBody(8423)

    '''

    p = pickle.load(open('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/train.pickle', 'rb'), encoding='latin1')
    print(len(str(p["image_id"])))
    for i, img_id in enumerate(p["image_id"]):
        if str(img_id.decode('latin1')) == "rm1058127616":
            index = i
            break
    print(index)
    
   
    #for i, img_id in enumerate(p["image_id"][38781:], start=38781):
    
    p = pickle.load(open('/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/val.pickle', 'rb'), encoding='latin1')
    #print("Total number of images:", len(p["image_id"]))
    for i, img_id in enumerate(p["image_id"]):
        index = i
        #print(i)
        #croppedBodywholeimgheight(i)
        croppedBody(i) 
        #croppedFace(i)
     '''