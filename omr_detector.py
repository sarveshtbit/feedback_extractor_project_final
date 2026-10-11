import cv2
import numpy as np

class OMRDetector:
    def __init__(self, filled_threshold=0.25, review_threshold=0.18):
        """
        filled_threshold: Minimum ink density ratio (black pixels / total area) to consider a bubble filled.
        review_threshold: Density ratio above which a bubble requires human review (borderline state).
        """
        self.filled_threshold = filled_threshold
        self.review_threshold = review_threshold

    def calculate_bubble_density(self, bubble_crop, radius):
        """
        Calculates the ink density ratio inside the circular region of interest.
        """
        if len(bubble_crop.shape) == 3:
            gray = cv2.cvtColor(bubble_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = bubble_crop.copy()

        # Thresholding to binary (ink = 255)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Create circular mask corresponding to inner bubble
        h, w = thresh.shape
        cx, cy = w // 2, h // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        # Inner 80% radius to ignore outer printed border
        inner_r = int(radius * 0.75)
        cv2.circle(mask, (cx, cy), inner_r, 255, -1)

        total_inner_pixels = np.sum(mask > 0)
        if total_inner_pixels == 0:
            return 0.0

        ink_pixels = np.sum((thresh > 0) & (mask > 0))
        density = float(ink_pixels / total_inner_pixels)
        return density

    def process_question(self, question_options):
        """
        Processes options for a single question.
        Returns:
          selected_option (str or None)
          status ("SINGLE_ANSWER", "MULTIPLE", "UNANSWERED", "REVIEW")
          confidence (float)
          densities (dict)
        """
        densities = {}
        filled_options = []
        review_options = []

        for opt in question_options:
            label = opt["label"]
            crop = opt["crop"]
            radius = opt["radius"]
            density = self.calculate_bubble_density(crop, radius)
            densities[label] = density

            if density >= self.filled_threshold:
                filled_options.append((label, density))
            elif density >= self.review_threshold:
                review_options.append((label, density))

        if len(filled_options) == 1:
            selected, conf = filled_options[0]
            return selected, "SINGLE_ANSWER", round(conf, 4), densities
        elif len(filled_options) > 1:
            # Multiple bubbles filled
            return None, "MULTIPLE", 0.0, densities
        elif len(review_options) > 0:
            # Borderline fill requiring human review
            best_review, conf = max(review_options, key=lambda x: x[1])
            return best_review, "REVIEW", round(conf, 4), densities
        else:
            # Unanswered
            return None, "UNANSWERED", 0.0, densities

    def process_all_questions(self, omr_feedback_data):
        """
        Processes all OMR feedback questions.
        """
        results = {}
        for q in omr_feedback_data:
            q_id = q["question_id"]
            label = q["label"]
            selected, status, conf, densities = self.process_question(q["options"])
            results[q_id] = {
                "question_id": q_id,
                "label": label,
                "selected_option": selected,
                "status": status,
                "confidence": conf,
                "densities": densities
            }
        return results

if __name__ == "__main__":
    detector = OMRDetector()
    # Test dummy empty vs filled bubble
    empty_crop = np.full((60, 60, 3), 255, dtype=np.uint8)
    cv2.circle(empty_crop, (30, 30), 28, (0, 0, 0), 3)

    filled_crop = empty_crop.copy()
    cv2.circle(filled_crop, (30, 30), 22, (0, 0, 0), -1)

    print("Empty bubble density:", detector.calculate_bubble_density(empty_crop, 28))
    print("Filled bubble density:", detector.calculate_bubble_density(filled_crop, 28))
