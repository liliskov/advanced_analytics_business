from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os

def resize_images(input_dir, output_dir, target_size,crop_size, mean, std):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    resize_transform = transforms.Resize(size=target_size, interpolation=InterpolationMode.BICUBIC)
    crop_transform = transforms.CenterCrop(size=crop_size)
    composed_transform = transforms.Compose([
        resize_transform,
        crop_transform,
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize(mean=mean, std=std)
    ])
    # Iterate over files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.webp') or filename.endswith('.png'):  # Adjust file extensions as needed
            # Open image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img = composed_transform(img)
            img_pil = transforms.functional.to_pil_image(img)
            # Save resized image to output directory
            output_path = os.path.join(output_dir, filename)
            img_pil.save(output_path)


# Example usage
#for RESNET50:
input_directory = 'images/'
output_directory = 'ResNet_Input'
target_size = [232]  # Specify the target size (width, height) in pixels
crop_size = [224]
mean = [0.485, 0.456, 0.406]
std =[0.229, 0.224, 0.225]
resize_images(input_directory, output_directory, target_size, crop_size, mean, std)
#for EfiicientNET: