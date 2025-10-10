from bs4 import BeautifulSoup
import os
import requests
import json
from multimodal_rag.common.config import get_settings

settings = get_settings()

# Get all HTML files from raw directory
data_dir = settings.data_dir
image_dir = data_dir + '/images'

def split_text_with_overlap(text, max_length=256, overlap_percentage=0.25):
        """Split text into chunks with overlap."""
        if len(text) <= max_length:
            return [text]

        overlap_size = int(max_length * overlap_percentage)
        chunks = []

        # Calculate total number of potential chunks
        chunk_starts = range(0, len(text), max_length - overlap_size)

        for start in chunk_starts:
            # Take a chunk of max_length or remaining text
            chunk = text[start:start + max_length]

            # If not the last chunk, try to break at a space
            if start + max_length < len(text):
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    chunk = chunk[:last_space]

            chunks.append(chunk.strip())

        return chunks

def parse_html_content_fine_grained(html_content):
    """
    Parse HTML content and extract structured content with sections and paragraphs.

    Args:
        html_content (str): Raw HTML content to parse

    Returns:
        list: List of dictionaries containing structured content
    """
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get article title
    article_title = soup.find('title').get_text().strip() if soup.find('title') else "Untitled"

    # Initialize variables
    structured_content = []
    current_section = "Main"  # Default section if no headers found

    # Find all headers and paragraphs
    content_elements = soup.find_all(['h1', 'h2', 'h3', 'p'])

    for element in content_elements:
        if element.name in ['h1', 'h2', 'h3']:
            current_section = element.get_text().strip()
        elif element.name == 'p' and element.get_text().strip():

            text = element.get_text().strip()
            # Split text into chunks with overlap
            text_chunks = split_text_with_overlap(text)

            for chunk in text_chunks:
                structured_content.append({
                    'article_title': article_title,
                    'section': current_section,
                    'text': chunk
                })

    return structured_content

def parse_html_content(html_content):
    """
    Parse HTML content and extract structured content with sections and paragraphs.

    Args:
        html_content (str): Raw HTML content to parse

    Returns:
        list: List of dictionaries containing structured content
    """
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get article title
    article_title = soup.find('title').get_text().strip() if soup.find('title') else "Untitled"

    # Initialize variables
    structured_content = []
    current_section = "Main"  # Default section if no headers found

    # Find all headers and text content
    content_elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol'])

    for element in content_elements:
        if element.name in ['h1', 'h2', 'h3']:
            current_section = element.get_text().strip()
        elif element.name in ['p', 'ul', 'ol']:
            text = element.get_text().strip()
            # Only add non-empty content that's at least 30 characters long
            if text and len(text) >= 30:
                structured_content.append({
                    'article_title': article_title,
                    'section': current_section,
                    'text': text
                })

    return structured_content

def parse_html_images(html_content):
    """
    Parse HTML content and extract images with their captions.

    Args:
        html_content (str): Raw HTML content to parse

    Returns:
        list: List of dictionaries containing images and their metadata
    """
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get article title
    article_title = soup.find('title').get_text().strip() if soup.find('title') else "Untitled"

    # Initialize variables
    structured_content = []
    current_section = "Main"  # Default section if no headers found

    # Find all headers and images
    content_elements = soup.find_all(['h1', 'h2', 'h3', 'img', 'figure'])

    for element in content_elements:
        if element.name in ['h1', 'h2', 'h3']:
            current_section = element.get_text().strip()
        elif element.name == 'img':
            # Get image path
            image_url = element.get('src', '')

            if image_url:  # Only proceed if there's an actual image URL
                # Download the image
                response = requests.get(image_url)
                if response.status_code == 200:
                    # Create images directory if it doesn't exist
                    os.makedirs(image_dir, exist_ok=True)

                    # Extract image filename from URL
                    image_filename = os.path.basename(image_url)
                    if "." not in image_filename:
                        image_filename = f"{image_filename}.jpg"

                    # Define the local file path
                    local_image_path = os.path.join(image_dir, image_filename)

                    # Save the image to the local file path
                    with open(local_image_path, 'wb') as f:
                        f.write(response.content)

                    # Store the local file path in the dictionary
                    image_path = local_image_path
                else:
                    image_path = ''

            # Try to get caption from alt text or figure caption
            caption = element.get('alt', '')
            if not caption and element.parent.name == 'figure':
                figcaption = element.parent.find('figcaption')
                if figcaption:
                    caption = figcaption.get_text().strip()

            if image_path:  # Only add if there's an actual image path
                structured_content.append({
                    'article_title': article_title,
                    'section': current_section,
                    'image_path': image_path,
                    'caption': caption or "No caption available"
                })

    return structured_content

def save_to_json(structured_content, output_file='output.json'):
    """
    Save structured content to a JSON file.

    Args:
        structured_content (list): List of dictionaries containing structured content
        output_file (str): Path to the output JSON file (default: 'output.json')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    # Save to JSON file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_content, f, indent=4, ensure_ascii=False)

def load_from_json(input_file):
    """
        Load structured content from a JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)
