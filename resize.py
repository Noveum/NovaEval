from PIL import Image

# Increase PIL's decompression bomb limit to handle large images
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely
# Alternative: Set a higher limit if you prefer
# Image.MAX_IMAGE_PIXELS = 500000000  # 500 million pixels

# Input and output paths
input_path = (
    "./examples/MMLU_REPORT/plots/overall_model_performance_comparison_bar_chart.png"
)
output_path = "examples/MMLU_REPORT/plots/overall_model_performance_comparison_bar_chart_small.png"

# Open the image
img = Image.open(input_path)

# LinkedIn requires smaller than 6012x6012. Let's safely resize with max dimension = 3000px to avoid issues
max_size = 3000
img.thumbnail((max_size, max_size))

# Save resized image
img.save(output_path)

print(f"Image resized and saved to: {output_path}")
