import pickle
import pandas as pd

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


