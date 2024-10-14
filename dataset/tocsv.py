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
'''
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


