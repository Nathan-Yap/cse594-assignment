import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

# --- CONFIGURATION ---
CSV_FILE = 'data/labels.csv'        # Path to your CSV file
IMAGE_DIR = './data'             # Directory where your images are stored

class ImageViewer:
    def __init__(self, csv_path, img_dir):
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.current_idx = 0
        self.total_images = len(self.df)

        # Setup the plot figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2) # Make room for buttons at the bottom

        # Create axes for buttons [left, bottom, width, height]
        ax_prev = plt.axes([0.3, 0.05, 0.15, 0.075])
        ax_next = plt.axes([0.55, 0.05, 0.15, 0.075])

        # Initialize Buttons
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')

        # Attach button click events
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

        # Display the first image
        self.show_image()

    def show_image(self):
        """Loads and displays the image at the current index."""
        self.ax.clear() # Clear the previous image
        self.ax.axis('off') # Hide grid lines and axes

        # Get data for the current row
        row = self.df.iloc[self.current_idx]
        img_name = row['imageName']
        description = row['description']

        # Construct full path and load image
        img_path = os.path.join(self.img_dir, img_name)

        try:
            img = mpimg.imread(img_path)
            self.ax.imshow(img)

            # Set the title to show progress and description
            title_text = f"[{self.current_idx + 1}/{self.total_images}] {img_name}\nDesc: {description.title()}"
            self.ax.set_title(title_text, fontsize=12, pad=10)

        except FileNotFoundError:
            # Handle cases where the image file is missing
            self.ax.text(0.5, 0.5, f"Image not found:\n{img_name}",
                         ha='center', va='center', fontsize=14, color='red')
            self.ax.set_title(f"[{self.current_idx + 1}/{self.total_images}] Missing Image")

        # Redraw the canvas
        self.fig.canvas.draw()

    def next_image(self, event):
        """Goes to the next image (loops back to start if at the end)."""
        self.current_idx = (self.current_idx + 1) % self.total_images
        print(self.current_idx)
        self.show_image()

    def prev_image(self, event):
        """Goes to the previous image (loops to the end if at the start)."""
        self.current_idx = (self.current_idx - 1) % self.total_images
        self.show_image()

if __name__ == '__main__':
    # Initialize and run the viewer
    viewer = ImageViewer(CSV_FILE, IMAGE_DIR)
    plt.show()