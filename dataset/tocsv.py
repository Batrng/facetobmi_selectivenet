import pickle
import pandas as pd
'''
#df = pd.read_csv('C:/Users/nguyen/slurmgit/facetobmi_selectivenet-regression/height.csv')
p = pickle.load(open('C:/Users/nguyen/Downloads/test.pickle', 'rb'), encoding='latin1')
height = p["height"]
image_id = p["image_id"]

df = pd.DataFrame({
    'image_id': image_id,
    'height': height
})
df.to_csv('C:/Users/nguyen/slurmgit/facetobmi_selectivenet-regression/height_test.csv', index=False)
df = pd.read_csv('C:/Users/nguyen/slurmgit/facetobmi_selectivenet-regression/height_test.csv')
df['image_id'] = df['image_id'].apply(lambda x: x.encode('utf-8').decode('unicode_escape').strip("b'")+ '.jpg')
df.to_csv('C:/Users/nguyen/slurmgit/facetobmi_selectivenet-regression/height_test.csv', index=False)

import os
import pandas as pd

def filter_csv_by_image_names(csv_file, image_column, folder_path, output_csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # List all files in the folder (get just the filenames)
    image_files = set(os.listdir(folder_path))

    # Filter the dataframe to only include rows where the image name exists in the folder
    df_filtered = df[df[image_column].isin(image_files)]

    # Save the filtered DataFrame back to a new CSV file
    df_filtered.to_csv(output_csv_file, index=False)

    #print(f"Filtered CSV saved as {output_csv_file}")

# Paths to your CSV file and image folder


# Call the function to filter the CSV
filter_csv_by_image_names("../height.csv", "image_id", "/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_face", "../height.csv")
'''

import os
import pandas as pd

def delete_non_csv_images(image_folder, csv_file, image_column):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the image names from the specified column and normalize them
    csv_image_names = df[image_column].astype(str).str.strip().tolist()

    # Get the list of image files in the folder
    image_files = os.listdir(image_folder)

    # Create a set for faster lookup
    csv_image_set = set(csv_image_names)

    # Loop through the images in the folder
    for img in image_files:
        # Check if the image name (without extension) is not in the CSV
        if img not in csv_image_set:
            # Construct the full path of the image
            img_path = os.path.join(image_folder, img)
            try:
                # Delete the image
                os.remove(img_path)
                print(f"Deleted: {img_path}")
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")

# Example usage
image_folder = "/home/nguyenbt/nobackup/data/2019_Mhse_Height_Data/combined_fullbody/"  # Replace with your image folder path
csv_file = "../height.csv"                   # Replace with your CSV file path
image_column = "image_id"           # Replace with the actual name of your column

delete_non_csv_images(image_folder, csv_file, image_column)


