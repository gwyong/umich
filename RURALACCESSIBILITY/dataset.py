import os, shutil, random, json, requests, time
import pandas as pd
from collections import defaultdict
from PIL import Image
from io import BytesIO
from tqdm import tqdm, trange

MI_LIHTC_R1_folder_path = "../../RollingsAccessibility/Images/MI_LIHTC/Round 1"
MI_LIHTC_R1_folder_names = [name for name in os.listdir(MI_LIHTC_R1_folder_path) if os.path.isdir(os.path.join(MI_LIHTC_R1_folder_path, name))]
MI_LIHTC_R2_folder_path = "../../RollingsAccessibility/Images/MI_LIHTC/Round 2"
MI_LIHTC_R2_folder_names = [name for name in os.listdir(MI_LIHTC_R2_folder_path) if os.path.isdir(os.path.join(MI_LIHTC_R2_folder_path, name))]
MI_USDA_RD_folder_path = "../../RollingsAccessibility/Images/MI_USDA_RD"
df_south_jersey_300_path = "../../RollingsAccessibility/Addresses/nj_intersections_tl_2020_south_jersey_sample_300.csv"
df_elizabeth_300_path    = "../../RollingsAccessibility/Addresses/nj_intersections_tl_2020_elizabeth_sample_300.csv"

def count_image_files():
    png_count, jpg_count = 0, 0
    for folder_name in MI_LIHTC_R1_folder_names:
        folder_path = os.path.join(MI_LIHTC_R1_folder_path, folder_name)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.PNG') or file_name.endswith('.png'):
                png_count += 1
            elif file_name.endswith('.JPG') or file_name.endswith('.jpg'):
                jpg_count += 1

    print(f"{"MI_LIHTC/Round1"} | # of files: {png_count+jpg_count}; # of avg files: {(png_count+jpg_count)/len(MI_LIHTC_R1_folder_names)} # of PNG files: {png_count}; # of JPG files: {jpg_count}")

    png_count, jpg_count = 0, 0
    for folder_name in MI_LIHTC_R2_folder_names:
        folder_path = os.path.join(MI_LIHTC_R2_folder_path, folder_name)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.PNG') or file_name.endswith('.png'):
                png_count += 1
            elif file_name.endswith('.JPG') or file_name.endswith('.jpg'):
                jpg_count += 1

    print(f"{"MI_LIHTC/Round2"} | # of files: {png_count+jpg_count}; # of avg files: {(png_count+jpg_count)/len(MI_LIHTC_R2_folder_names)} # of PNG files: {png_count}; # of JPG files: {jpg_count}")

    png_count, jpg_count = 0, 0
    for file_name in os.listdir(MI_USDA_RD_folder_path):
        if file_name.endswith('.PNG') or file_name.endswith('.png'):
            png_count += 1
        elif file_name.endswith('.JPG') or file_name.endswith('.jpg'):
            jpg_count += 1
    
    print(f"{"MI_USDA_RD"} | # of files: {png_count+jpg_count}; # of PNG files: {png_count}; # of JPG files: {jpg_count}")
    return

def make_testing_set(source_folder_path, target_folder_path="../../RollingsAccessibility/test_images", num_addresses=5):
    source_folder_names = [name for name in os.listdir(source_folder_path) if os.path.isdir(os.path.join(source_folder_path, name))]
    test_folders = random.sample(source_folder_names, num_addresses)
    
    file_mode = "a" if os.path.exists(os.path.join(target_folder_path, "test_folders.txt")) else "w"
    with open(os.path.join(target_folder_path, "test_folders.txt"), file_mode) as file:
        for folder_name in test_folders:
            copy_source_folder_path = os.path.join(source_folder_path, folder_name)
            for file_name in os.listdir(copy_source_folder_path):
                if file_name.endswith('.PNG') or file_name.endswith('.png') or file_name.endswith('.JPG') or file_name.endswith('.jpg'):
                    file_path = os.path.join(copy_source_folder_path, file_name)
                    shutil.copyfile(file_path, os.path.join(target_folder_path, file_name))
            file.write(copy_source_folder_path + "\n")
    print(f"Test images are saved in {target_folder_path} from {source_folder_path}")
    return

def count_labels_from_jsons(json_folder_path):
    label_counts = defaultdict(int)

    for file_name in os.listdir(json_folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(json_folder_path, file_name), "r") as file:
                json_data = json.load(file)
                for shape in json_data.get('shapes', []):
                    label = shape.get('label', 'unknown')
                    label_counts[label] += 1
    
    total_count = sum(label_counts.values())
    print(label_counts)
    print(f"Total count: {total_count}")
    return

def collect_GSV_images(api_key, df_path, output_path="./outputs/GSVs"):
    df  = pd.read_csv(df_path)
    url = f"https://maps.googleapis.com/maps/api/streetview"
    headings = [(0, "north"), (90, "east"), (180, "south"), (270, "west")]
    
    for idx in trange(len(df)):
        if idx <= 285:
            continue
        latitude = df.loc[idx, 'latitude']
        longitude = df.loc[idx, 'longitude']
        address = df.loc[idx, 'street_names']
        
        for heading in headings:
            params = {
                'location': f'{latitude}, {longitude}',
                'size': '640x640',
                'heading': heading[0],
                'fov': 90,
                'pitch': 0,
                'key': api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(output_path, f"{address+"_"+heading[1]}.png"))
            else:
                print(f"Error: {response.status_code} | {address} | {heading[1]}")
            time.sleep(0.001)
    return

if __name__ == "__main__":
    # count_image_files()
    # make_testing_set(MI_LIHTC_R1_folder_path, num_addresses=24)
    # make_testing_set(MI_LIHTC_R2_folder_path, num_addresses=5)
    # make_testing_set(MI_USDA_RD_folder_path, num_addresses=5)
    # count_labels_from_jsons("../../RollingsAccessibility/test_images")
    collect_GSV_images("API_key", df_elizabeth_300_path)

"""
MI_LIHTC/Round1 | # of files: 417; # of avg files: 3.475 # of PNG files: 414; # of JPG files: 3
MI_LIHTC/Round2 | # of files: 175; # of avg files: 7.608695652173913 # of PNG files: 170; # of JPG files: 5
MI_USDA_RD | # of files: 83; # of PNG files: 83; # of JPG files: 0
3 hours 32 mins / 65 images
2 hours 45 mins / 60 images
defaultdict(<class 'int'>, {'parking_lot': 221, 'townhome': 243, 'sidewalk': 84, 'step_free': 25, 'curb_cut': 27, 'entrance': 57, 'first_story': 85, 'apartment': 321, 'second_story': 69, 'stair': 14, 'third_story': 28, 'air_conditioning': 8, 'fourth_story': 2, 'ramp': 1})
Total count: 1185
1256 - 128 = 1128
1088 - 13 = 1075
"""