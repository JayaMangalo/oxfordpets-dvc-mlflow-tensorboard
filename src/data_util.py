import os
import urllib.request
import shutil
import tarfile

# Step 1: Download Oxford-IIIT Pet Dataset
def download_oxford_pets():
    url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
    data_dir = 'data/oxford-iiit-pet'
    filename = os.path.join(data_dir, 'images.tar.gz')
    
    # Check if the file already exists
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already exists!")

# Step 2: Extract the downloaded tar file
def extract_data():
    data_dir = 'data/oxford-iiit-pet'
    tar_path = os.path.join(data_dir, 'images.tar.gz')
    raw_dir = os.path.join(data_dir, 'raw')

    if not os.path.exists(raw_dir):
        print("Extracting data...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=raw_dir)
        print("Data extracted successfully.")
    else:
        print("Data already extracted.")

# Step 3: Organize images into class-based folders inside 'images/'
def organize_images():
    data_dir = 'data/oxford-iiit-pet'
    raw_dir = os.path.join(data_dir, 'raw/images')  # Original folder that contains images
    output_dir = os.path.join(data_dir, 'images')  # The output folder for breed-specific folders

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a mapping of breeds (from image file names)
    for image in os.listdir(raw_dir):
        image_path = os.path.join(raw_dir, image)
        
        # Skip non-file entries (directories or non-image files)
        if not os.path.isfile(image_path):
            continue
        
        # Extract breed name from image file name (e.g., 'Abyssinian_1.jpg' -> 'Abyssinian')
        breed_name = image.split('_')[0]
        
        # Create breed directory if it doesn't exist
        breed_output_path = os.path.join(output_dir, breed_name)
        if not os.path.exists(breed_output_path):
            os.makedirs(breed_output_path)

        # Move image to the corresponding breed folder inside 'images/'
        shutil.copy(image_path, breed_output_path)

    print(f"Images organized into breed-specific folders inside {output_dir}")

# Step 4: Execute all functions
if __name__ == '__main__':
    download_oxford_pets()
    extract_data()
    organize_images()
    print("Process completed.")
