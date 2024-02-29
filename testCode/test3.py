import cairo
from PIL import Image

# Define image dimensions
width, height = 400, 400

# Create a Cairo surface with a white background
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
ctx = cairo.Context(surface)
ctx.set_source_rgb(1, 1, 1)  # White background
ctx.rectangle(0, 0, width, height)
ctx.fill()

# Set drawing parameters for the line
ctx.set_source_rgb(0, 0, 0)  # Black line color
ctx.set_line_width(5)

# Draw the line
ctx.move_to(50, 50)
ctx.line_to(350, 250)
ctx.stroke()

# Convert Cairo surface to PIL image
data = surface.get_data()
img = Image.frombytes("RGBA", (width, height), data)

# Save the image as "pycario.png"
img.save("pycario.png")

print("Image saved as pycario.png")
