import os
import shutil
import cv2
import numpy as np
from typing import Optional

class ScratchManager:
    """
    Manages job-isolated scratch directories under scratch/<job_id>/
    Structure:
      scratch/<job_id>/original/
      scratch/<job_id>/variants/
      scratch/<job_id>/crops/
      scratch/<job_id>/ocr/
      scratch/<job_id>/debug/

    Crops and debug images are only generated and saved when actually required.
    """
    
    BASE_SCRATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scratch"))

    @classmethod
    def get_job_dir(cls, job_id: str, category: Optional[str] = None) -> str:
        job_dir = os.path.join(cls.BASE_SCRATCH_DIR, str(job_id))
        if category:
            job_dir = os.path.join(job_dir, category)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    @classmethod
    def save_image(cls, job_id: str, category: str, filename: str, image: np.ndarray) -> str:
        target_dir = cls.get_job_dir(job_id, category)
        filepath = os.path.join(target_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath

    @classmethod
    def reset_stale_scratch(cls):
        """
        Deletes existing stale crop/scratch files if any exist at top level of scratch directory.
        Prevents legacy/stale crops from interfering with new extraction jobs.
        """
        if os.path.exists(cls.BASE_SCRATCH_DIR):
            for item in os.listdir(cls.BASE_SCRATCH_DIR):
                item_path = os.path.join(cls.BASE_SCRATCH_DIR, item)
                if os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                    except Exception:
                        pass
                elif os.path.isdir(item_path) and item.startswith("legacy_"):
                    try:
                        shutil.rmtree(item_path)
                    except Exception:
                        pass

    @classmethod
    def clean_job_scratch(cls, job_id: str, keep_debug: bool = False):
        """
        Cleans up temporary job scratch files when extraction completes if keep_debug is False.
        """
        job_dir = os.path.join(cls.BASE_SCRATCH_DIR, str(job_id))
        if not os.path.exists(job_dir):
            return

        if not keep_debug:
            try:
                shutil.rmtree(job_dir)
            except Exception as e:
                print(f"[ScratchManager] Warning cleaning job scratch {job_id}: {e}")
        else:
            # Clean only heavy temporary variants
            variants_dir = os.path.join(job_dir, "variants")
            if os.path.exists(variants_dir):
                try:
                    shutil.rmtree(variants_dir)
                except Exception:
                    pass
