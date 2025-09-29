import os
import shutil
from PIL import Image

import cv2
from IPython.display import display


class Utils:
    @staticmethod
    def extract_images(source_path: str, target_path: str) -> None:
        """
        Extracts images from video and copies existing images.

        Args:
            source_path (str): Directory to extract from.
            target_path (str): Directory to extract to.

        Returns:
            None
        """
        if os.path.exists(target_path):  # deleting target folder, if it exists
            shutil.rmtree(target_path)

        # going through folders with videos and images
        for folder_name in os.listdir(source_path):
            out_dir = f"{target_path}/scale_1/{folder_name}"
            os.makedirs(out_dir)

            total_img_num = 0

            for file_name in os.listdir(f"{source_path}/{folder_name}"):
                if file_name.endswith(".mp4"):
                    cap = cv2.VideoCapture(f"{source_path}/{folder_name}/{file_name}")
                    if not cap.isOpened():
                        print(
                            "Error: Could not open video file {0}/{1}/{2}".format(
                                source_path, folder_name, file_name
                            )
                        )
                        return

                    frame_count = 0

                    while cap.grab():
                        frame = cap.retrieve()[1]

                        frame_filename = os.path.join(out_dir, f"{total_img_num}.jpg")
                        cv2.imwrite(frame_filename, frame)

                        frame_count += 1
                        total_img_num += 1

                    cap.release()

                    print(
                        "   {0} frames extracted from {1}/{2}/{3}".format(
                            frame_count, source_path, folder_name, file_name
                        )
                    )

            img_num = 0

            for file_name in os.listdir(f"{source_path}/{folder_name}"):
                if file_name.endswith(".jpg"):
                    shutil.copyfile(
                        f"{source_path}/{folder_name}/{file_name}",
                        os.path.join(out_dir, f"{img_num}.jpg"),
                    )

                    img_num += 1
                    total_img_num += 1

            if img_num > 0:
                print(f"   {img_num} images copied from {source_path}/{folder_name}")

            print(f"{total_img_num} images total in {out_dir}")

    @staticmethod
    def show_altered_image(path: str, scale: float, remove_color: bool = False) -> None:
        """
        Shows rescaled image with or without color.

        Args:
            path (str): Directory to extract from.
            scale (float): Directory to extract to.
            remove_color (bool): If color should be removed. Default: False

        Returns:
            None
        """
        img = cv2.imread(path)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        print(f"Image size: {img.shape[1]}:{img.shape[0]}")

        if remove_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        display(Image.fromarray(img))

    @staticmethod
    def alter_images(
        source_path: str, scale: float, remove_color: bool = False
    ) -> None:
        """
        Extracts images from video and copies existing images.

        Args:
            source_path (str): Directory to extract from.
            scale (float): Directory to extract to.
            remove_color (bool): If color should be removed. Default: False

        Returns:
            None
        """
        target_path = f"{source_path}/scale_{scale}"

        if os.path.exists(target_path):  # deleting target folder, if it exists
            shutil.rmtree(target_path)

        for folder_name in os.listdir(f"{source_path}/scale_1"):
            out_dir = f"{target_path}/{folder_name}"
            os.makedirs(out_dir)

            for file_name in os.listdir(f"{source_path}/scale_1/{folder_name}"):
                if file_name.endswith(".jpg"):
                    img = cv2.imread(f"{source_path}/scale_1/{folder_name}/{file_name}")
                    img = cv2.resize(img, None, fx=scale, fy=scale)

                    if remove_color:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(f"{target_path}/{folder_name}/{file_name}", img)

        print(f"Images with scale {scale} created")
