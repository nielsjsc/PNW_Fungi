import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

# Get the current working directory
cwd = os.getcwd()

# Specify the directory where you want to save the images
save_dir = os.path.join(cwd, 'PNW_mushrooms\\fungi_photos_rgb')

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the CSV files
df1 = pd.read_csv('C:\\Users\\User\\Desktop\\Coding\\Pytorch\\PNW_mushrooms\\fungi_links.csv')

# Extract the image URLs and species names
image_urls = df1['identifier']
species_urls = df1['references']
species_list_enum = []

# Set the limit for the number of images to download
download_limit = 100000
downloaded_count = 5123

# Keep track of downloaded filenames
downloaded_filenames = set()

# Check if the downloaded CSV file exists and load it if it does
label_csv_path = os.path.join(cwd, 'image_labels.csv')
if os.path.exists(label_csv_path):
    label_df = pd.read_csv(label_csv_path)
    downloaded_filenames = set(label_df['filename'])
    downloaded_count = len(downloaded_filenames)

# Find the index to start downloading from
start_index = downloaded_count

# ... (previous code)

# Download the images, convert to black and white, resize, and save them to disk
for i, row in df1.iloc[start_index:].iterrows():
    url, spec_url = row['identifier'], row['references']

    if pd.isna(url) or pd.isna(spec_url):
        print(f"Skipping entry {i + 1} as URL or spec_url is NaN.")
        continue

    if downloaded_count >= download_limit:
        break  # Break out of the loop once the download limit is reached

    # Extract filename from URL
    img_filename = os.path.basename(urlparse(url).path)

    if img_filename in downloaded_filenames:
        print(f"Skipping already downloaded image {img_filename}.")
        continue

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        img = Image.open(BytesIO(response.content))

        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Resize to 224x224 pixels
        img = img.resize((224, 224))

        # Save the modified image with the species name as the file label
        response = requests.get(spec_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        species_tag = soup.find('span', {'class': 'sciname'})
        species_name = species_tag.text.strip() if species_tag is not None else f'unknown_species_{i}'
        img_filename = f'{species_name}_image{i}.jpg'
        
        img.save(os.path.join(save_dir, img_filename))
        
        species_list_enum.append(species_name)
        
        downloaded_count += 1
        downloaded_filenames.add(img_filename)

    except requests.RequestException as e:
        print(f"Error downloading image {img_filename}: {e}")

# ... (rest of the code)

# Create a DataFrame with image filenames and corresponding species names
label_df = pd.DataFrame({'filename': [f'{species}_image{i}.jpg' for i, species in enumerate(species_list_enum, start=start_index)],
                         'species': species_list_enum})

# Save the DataFrame to a CSV file for reference during model training
label_df.to_csv(label_csv_path, index=False)
