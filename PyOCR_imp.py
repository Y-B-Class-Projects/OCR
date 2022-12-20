from PIL import Image
import sys
import pyocr
import pyocr.builders

# Create an OCR tool
tool = pyocr.get_available_tools()[0]
print("Will use tool '%s'" % (tool.get_name()))

# Load an image containing text
image = Image.open('data/LD_02.jpg')

# Use the OCR tool to recognize the text in the image
text = tool.image_to_string(
    image,
    lang='eng+vie',
    builder=pyocr.builders.TextBuilder()
)

# Print the recognized text
print(text)
