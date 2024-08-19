import cv2
import numpy as np


class image_processing:
    def get_contours(self,combined_masks, epsilon = 0.005):
        added_mask = np.sum(combined_masks, axis=0).astype(np.uint8)

        smooth_mask = cv2.morphologyEx(added_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        min_contour_area = 100  # Adjust this value based on your needs
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        output = smooth_mask.copy()

        approx_contours = []
        for contour in filtered_contours:
            epsilon_val = epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon_val, True)
            cv2.drawContours(output, [approx], -1, (255, 255, 255), 5)
            approx_contours.append(approx)

        return smooth_mask,approx_contours,output