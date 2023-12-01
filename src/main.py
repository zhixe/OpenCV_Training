# Version 1.26
import cv2, os, sys, numpy as np,matplotlib.pyplot as plt

# ==================== OOP - REFACTORED SOLID DESIGN MODEL ====================
# ==================== MAIN LOGIC ====================
class ImageProcessor:
    """
    The ImageProcessor class is designed to handle different tasks related to image processing.

    Attributes:
        image (np.array): The image to be processed.

    Methods:
        remove_text(image): Removes text from the image.
        is_text(contour): Determines if a contour is text based on its area.
        apply_color_mask(image, colors): Applies color masks to the image.
        apply_mask(image, lower, upper): Applies a color mask to the image.
        rgb_to_hsv_range(rgb_color): Converts an RGB color to an HSV range.
        to_grayscale(image): Converts the image to grayscale.
        threshold(gray): Applies a binary inverse OTSU threshold to a grayscale image.
        find_contours(thresh): Finds contours in a binary image.
        display_and_save_overall_results_question1(image_without_text, color_masks, output_path): Displays and saves the overall results for question 1.
        save_objects_results_question1(image_without_text, color_masks, output_folder): Displays and saves the results for question 1.
        save_object_results_question2(output_folder, colors_q2): Displays and saves the results for question 2.
        save_results_question2_histogram(color_masks, output_folder): Displays and saves the histogram results for question 2.
    """
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Warning: Could not load the image at {image_path}. Proceed to terminate the program!")
            sys.exit()
        else:
            self.image = cv2.imread(image_path)

    @staticmethod
    def remove_text(image):
        gray = ImageProcessor.to_grayscale(image)
        _, thresh = ImageProcessor.threshold(gray)
        contours, _ = ImageProcessor.find_contours(thresh)
        image_without_text = image.copy()
        for contour in contours:
            if ImageProcessor.is_text(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_without_text, (x, y), (x + w, y + h), (255, 255, 255), -1)
        return image_without_text

    @staticmethod
    def is_text(contour):
        _, _, w, h = cv2.boundingRect(contour)
        return w * h < 500

    @staticmethod
    def apply_color_mask(image, colors):
        results = {}
        for color, (lower, upper) in colors.items():
            result = ImageProcessor.apply_mask(image, lower, upper)
            results[color] = result
        return results

    @staticmethod
    def apply_mask(image, lower, upper):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((6, 6), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        colored_objects = cv2.bitwise_and(image, image, mask=mask)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 25, 255]))
        white_mask_inv = cv2.bitwise_not(white_mask)
        colored_objects[white_mask_inv == 0] = [0, 0, 0]
        return colored_objects

    @staticmethod
    def rgb_to_hsv_range(rgb_color):
        hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)
        h, s, v = hsv_color[0][0]
        lower = [h - 10, max(0, s - 50), max(0, v - 50)]
        upper = [h + 10, min(255, s + 50), min(255, v + 50)]
        return (lower, upper)

    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def threshold(gray):
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    @staticmethod
    def find_contours(thresh):
        return cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    @staticmethod
    def responsivePlotText(texts, fig=None, minimal=1):
        if not fig: fig = plt.gcf()
        fontsizes = [t.get_fontsize() for t in texts]
        _, windowheight = fig.get_size_inches()*fig.dpi

        def resize(event=None):
            scale = event.height / windowheight
            for i in range(len(texts)):
                newsize = np.max([int(fontsizes[i]*scale), minimal])
                texts[i].set_fontsize(newsize)
        return resize

    @staticmethod
    def display_and_save_overall_results_question1(image_without_text, color_masks, output_path):
        fig = plt.figure(num='Question 1', figsize=(14, 10))

        # Calculate the number of images per row
        n = len(color_masks) + 1
        n_per_row = n // 2 if n % 2 == 0 else n // 2 + 1

        # Display image with non-text objects
        ax1 = plt.subplot(2, n_per_row, 1)
        plt.imshow(cv2.cvtColor(image_without_text, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image with Non-Text Objects')
        ax1.axis('off')

        # Display colored objects
        axs = [ax1]
        for i, (color, result) in enumerate(color_masks.items(), start=2):
            ax = plt.subplot(2, n_per_row, i)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax.set_title(f'{color} Objects')
            ax.axis('off')
            axs.append(ax)

        plt.tight_layout()
        plt.savefig(output_path)

        # Add responsive text title for each plot
        plt.gcf().canvas.mpl_connect("resize_event", ImageProcessor.responsivePlotText([ax.title for ax in axs]))

        plt.show()

    @staticmethod
    def save_objects_results_question1(image_without_text, color_masks, output_folder):
        # Save the image with non-text objects
        cv2.imwrite(os.path.join(output_folder, 'Q1_Output_Image_With_Non-Text_Objects.jpg'), image_without_text)

        # Save each color mask image
        for seq, (color, result) in enumerate(color_masks.items(), start=1):
            file_name = f'Q1_Output_Image_{seq}_{color}_Object.jpg'
            cv2.imwrite(os.path.join(output_folder, file_name), result)

    @staticmethod
    def save_object_results_question2(output_folder, colors_q2):
        kernel = np.ones((5,5),np.uint8)
        for seq, color in enumerate(colors_q2.keys(), start=1):
            image_path = os.path.join(output_folder, f'Q1_Output_Image_{seq}_{color}_Object.jpg')
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load the image at {image_path}. Skipping this image!")
                continue
            gray = ImageProcessor.to_grayscale(image)
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            contours, _ = ImageProcessor.find_contours(opening)
            i = 1
            for contour in contours:
                # Filter contours by area
                if cv2.contourArea(contour) < 500:  # Adjust the threshold as needed
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cropped = image[y:y+h, x:x+w]
                file_name = f'Q2_Output_Image_{seq}_{color}_{i}_Cropped_Object.jpg'
                cv2.imwrite(os.path.join(output_folder, file_name), cropped)
                i += 1

    @staticmethod
    def save_results_question2_histogram(output_folder, original_image):
        """
        This method generates and saves histograms of the RGB color channels of the original image (image_in_q1).

        Parameters:
        output_folder (str): The path to the folder where the histogram image will be saved.
        original_image (ndarray): The original image for which the histograms will be calculated.

        The method does the following:
        1. Creates a figure with 4 subplots.
        2. Calculates and plots the histogram for each RGB color channel in the first 3 subplots.
        3. Plots a combined histogram of all RGB color channels in the 4th subplot.
        4. Saves the figure as 'Q2_Output_Image_RGB_Dist_Object_Histogram.jpg' in the specified output folder.

        Note: The image is saved in the output folder and not returned by the function.
        """
        # Create a figure and specify the layout
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))  # 4, 1 Change the number of subplots to 4

        color = ('r', 'g', 'b')

        # Calculate and plot the histogram for each RGB color channel (3 subplots)
        for i, col in enumerate(color):
            hist = cv2.calcHist([original_image], [i], None, [256], [0, 256])
            axs[i].plot(range(256), hist.ravel(), color=col, linewidth=0.8)
            axs[i].set_xlim([0, 256])
            axs[i].set_title(f'{col.upper()} Distribution of Object Histogram', fontsize=14)
            axs[i].set_xlabel('Color Intensity (0-255)', fontsize=12)
            axs[i].set_ylabel('Number of Pixels', fontsize=12)

        # Plot the combined RGB color channel histogram (1 subplot)
        for i, col in enumerate(color):
            hist = cv2.calcHist([original_image], [i], None, [256], [0, 256])
            axs[3].plot(range(256), hist.ravel(), color=col, linewidth=0.8)  # Use color for the combined plot
            axs[3].set_xlim([0, 256])
            axs[3].set_title('RGB Distribution of Object Histogram', fontsize=14)
            axs[3].set_xlabel('Color Intensity (0-255)', fontsize=12)
            axs[3].set_ylabel('Number of Pixels', fontsize=12)

        plt.subplots_adjust(left=0.15, hspace=0.5)  # Adjust left margin of Y-axis label and space between subplots
        plt.savefig(os.path.join(output_folder, 'Q2_Output_Image_RGB_Dist_Object_Histogram.jpg'))
        plt.clf()


# ==================== ANSWERS LOGIC ====================
class Question1: ## QUESTION 1 ##
    """
    The Question1 class is designed to handle the tasks related to the first question.

    Attributes:
        image_path (str): The path to the image to be processed.
        output_path (str): The output path for saving processed images.

    Methods:
        apply_text_removal_and_color_masking(image, colors): Removes text from the image and applies color masking, then returns the processed image and color masks.
    """
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

    def apply_text_removal_and_color_masking(self, image, colors):
        image_without_text = ImageProcessor.remove_text(image)
        color_masks = ImageProcessor.apply_color_mask(image, colors)
        return image_without_text, color_masks


class Question2: ## QUESTION 2 ##
    """
    The Question2 class is designed to handle the tasks related to the second question.

    Attributes:
        image_path (str): The path to the image to be processed.
        output_path (str): The output path for saving processed images.

    Methods:
        apply_color_masking(image, colors): Applies color masking to the image, then returns the color masks.
    """
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

    def apply_color_masking(self, image, colors):
        color_masks = ImageProcessor.apply_color_mask(image, colors)
        return color_masks


# ==================== PROCESS FLOW LOGIC ====================
class TaskHandler:
    """
    The TaskHandler class is designed to handle different tasks related to image processing.

    Attributes:
        processor (ImageProcessor): An instance of the ImageProcessor class.
        output_path (str): The general output path for saving processed images.
        image_out_q1 (str): The output path for the results of question 1.
        colors (list): A list of colors used for color masking in the image processing tasks.

    Methods:
        process_question1(): Applies text removal and color masking to the image for question 1, then displays and saves the results.
        process_question2(colors_q2): Applies color masking to the image for question 2, then displays and saves the results.
        run_all(): Runs all tasks if an image is available for processing.
    """
    def __init__(self, image_path, colors, image_out_q1, output_path):
        self.processor = ImageProcessor(image_path)
        self.output_path = output_path
        self.image_out_q1 = image_out_q1
        self.colors = colors

    def process_question1(self):
        if self.processor.image is None:
            return  # Skip processing if the image couldn't be loaded

        question1_processor = Question1(self.processor.image, self.output_path)
        image_without_text, color_masks = question1_processor.apply_text_removal_and_color_masking(self.processor.image, self.colors)
        
        ImageProcessor.display_and_save_overall_results_question1(image_without_text, color_masks, self.image_out_q1)
        ImageProcessor.save_objects_results_question1(image_without_text, color_masks, self.output_path)

    def process_question2(self, colors_q2):
        if self.processor.image is None:
            return  # Skip processing if the image couldn't be loaded

        question2_processor = Question2(self.processor.image, self.output_path)
        question2_processor.apply_color_masking(self.processor.image, colors_q2)
            
        ImageProcessor.save_object_results_question2(self.output_path, colors_q2)
        ImageProcessor.save_results_question2_histogram(self.output_path, self.processor.image) 

    def run_all(self):
        if self.processor.image is not None:
            self.process_question1()
            self.process_question2(colors_q2)
        else:
            print("No image to process.")



# ==================== DEFINED GLOBAL VARIABLE[S]/SCOPE[S] AND EXECUTE ALL LOGIC ====================
"""
This script is designed to handle different tasks related to image processing.

The script first clears the terminal screen and prints a message indicating that it is running. It then defines the input and output paths, as well as the color palettes for the tasks. The color palettes are defined in the RGB color space and then converted to the HSV range using the `rgb_to_hsv_range` method from the `ImageProcessor` class.

An instance of the `TaskHandler` class is created with the input image, color palette, output image for question 1, and output folder as parameters. The `run_all` method of the `TaskHandler` class is then called to run all the processing tasks.

Finally, the terminal screen is cleared again and a message is printed indicating that the script has successfully run and is about to exit.
"""
if __name__ == "__main__":
    os.system('cls') # Clear terminal screen
    print("Script is currently running...")

    # Input Path
    image_in_q1 = fr'input\Figure1.jpg'
    
    # Output Path
    output_folder = fr'output'
    os.makedirs(output_folder, exist_ok=True)
    image_out_q1 = fr'output\Q1_Final_Output_Overall_Result.jpg'

    # Defined default colours pallete
    colors = {
        'Yellow': ImageProcessor.rgb_to_hsv_range([250, 234, 24]),
        'Green': ImageProcessor.rgb_to_hsv_range([46, 175, 74]),
        'Blue': ImageProcessor.rgb_to_hsv_range([26, 119, 188]),
        'Red': ImageProcessor.rgb_to_hsv_range([234, 35, 40]),
        'Orange': ImageProcessor.rgb_to_hsv_range([243, 128, 35]),
        'Purple': ImageProcessor.rgb_to_hsv_range([105, 63, 149]),
        'Pink': ImageProcessor.rgb_to_hsv_range([236, 36, 142])
    }

    # Defined colours pallete for Question 2
    colors_q2 = {
        'Yellow': ImageProcessor.rgb_to_hsv_range([250, 234, 24]),
        'Green': ImageProcessor.rgb_to_hsv_range([46, 175, 74]),
        'Blue': ImageProcessor.rgb_to_hsv_range([26, 119, 188]),
        'Red': ImageProcessor.rgb_to_hsv_range([234, 35, 40])
    }

    # Create an instance of the processing class
    processing_tasks = TaskHandler(image_in_q1, colors, image_out_q1, output_folder)
    processing_tasks.run_all()

    print("Script successfully ran")
    print(f"Please check your output at: \n{os.path.join(os.getcwd(), output_folder)}\n")
    print("Proceed to exit the program...")
