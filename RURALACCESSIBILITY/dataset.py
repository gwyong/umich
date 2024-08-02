import os, random

MI_LIHTC_R1_folder_path = "../../RollingsAccessibility/Images/MI_LIHTC/Round 1"
MI_LIHTC_R1_folder_names = [name for name in os.listdir(MI_LIHTC_R1_folder_path) if os.path.isdir(os.path.join(MI_LIHTC_R1_folder_path, name))]
MI_LIHTC_R2_folder_path = "../../RollingsAccessibility/Images/MI_LIHTC/Round 2"
MI_LIHTC_R2_folder_names = [name for name in os.listdir(MI_LIHTC_R2_folder_path) if os.path.isdir(os.path.join(MI_LIHTC_R2_folder_path, name))]
MI_USDA_RD_folder_path = "../../RollingsAccessibility/Images/MI_USDA_RD"

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

if __name__ == "__main__":
    count_image_files()

"""
MI_LIHTC/Round1 | # of files: 417; # of avg files: 3.475 # of PNG files: 414; # of JPG files: 3
MI_LIHTC/Round2 | # of files: 175; # of avg files: 7.608695652173913 # of PNG files: 170; # of JPG files: 5
MI_USDA_RD | # of files: 83; # of PNG files: 83; # of JPG files: 0
"""