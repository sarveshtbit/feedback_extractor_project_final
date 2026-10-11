import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np

def create_validation_datasets():
    print("=== STAGE 12: CREATING INDEPENDENT VALIDATION DATASET ===")
    
    out_dir = "storage/stage12_independent_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    template_pdf = r"D:\Downloads\BANNARI AMMAN INSTITUTE OF TECHNOLOGY (1).pdf"
    if not os.path.exists(template_pdf):
        raise FileNotFoundError(f"Template PDF not found at {template_pdf}")
        
    ground_truth = {
        "doc_1": {
            "form_1": {
                "student_name": "KAVITHA M",
                "school_name": "ST JOSEPHS HIGHER SEC SCHOOL",
                "date": "15/10/2026",
                "student_mobile": "9876543210",
                "parent_mobile": "9443322110",
                "email": "kavitha.m@testmail.com",
                "cutoff_marks": "192.75",
                "interested_in_bit": "Yes",
                "programme_usefulness": "4",
                "speaker_effectiveness": "5",
                "choice_filling_assistance": "No",
                "notes": "Standard clean digitally typed form with high ratings"
            },
            "form_2": {
                "student_name": "RAJESH K",
                "school_name": "BHARATHI VIDYA BHAVAN",
                "date": "22/04/2026",
                "student_mobile": "8765432109",
                "parent_mobile": None,  # DELIBERATELY BLANK
                "email": "rajesh.k@sample.edu",
                "cutoff_marks": "165.50",
                "interested_in_bit": "No",
                "programme_usefulness": None,  # DELIBERATELY UNANSWERED
                "speaker_effectiveness": "3",
                "choice_filling_assistance": "AMBIGUOUS",  # DELIBERATELY AMBIGUOUS (both marked)
                "notes": "Contains blank parent mobile, unanswered Q2, and ambiguous Q4"
            }
        },
        "doc_2": {
            "form_1": {
                "student_name": "DINESH KUMAR V",
                "school_name": "MODEL GOVT BOYS HR SEC SCHOOL",
                "date": "05/11/2026",
                "student_mobile": "7890123456",
                "parent_mobile": "7123456789",
                "email": None,  # DELIBERATELY BLANK
                "cutoff_marks": "181.00",
                "interested_in_bit": "Yes",
                "programme_usefulness": "3",
                "speaker_effectiveness": None,  # DELIBERATELY UNANSWERED
                "choice_filling_assistance": "Yes",
                "notes": "Contains blank email and unanswered Q3"
            },
            "form_2": {
                "student_name": "SNEHA R",
                "school_name": "KENDRIYA VIDYALAYA NO 2",
                "date": "18/12/2026",
                "student_mobile": "9123456780",
                "parent_mobile": "9842109876",
                "email": "sneha.r26@domain.org",
                "cutoff_marks": "198.25",
                "interested_in_bit": "Yes",
                "programme_usefulness": "5",
                "speaker_effectiveness": "5",
                "choice_filling_assistance": "Yes",
                "notes": "Standard clean digitally typed form with high cutoff"
            }
        }
    }
    
    # Save answer keys separately
    answer_keys_path = os.path.join(out_dir, "answer_keys.json")
    with open(answer_keys_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved independent answer keys to {answer_keys_path}")
    
    # Precise box coordinates and clear-rects
    date_rects_f1 = [
        (36.4, 72.0, 50.5, 89.9), (52.5, 72.0, 66.5, 89.9),
        (81.4, 72.0, 95.5, 89.9), (97.4, 72.0, 111.5, 89.9),
        (126.4, 72.0, 140.4, 89.9), (142.4, 72.0, 156.5, 89.9),
        (158.5, 72.0, 172.5, 89.9), (174.5, 72.0, 188.5, 89.9)
    ]
    date_rects_f2 = [
        (36.4, 458.5, 50.5, 476.4), (52.5, 458.5, 66.5, 476.4),
        (81.4, 458.5, 95.5, 476.4), (97.4, 458.5, 111.5, 476.4),
        (126.4, 458.5, 140.4, 476.4), (142.4, 458.5, 156.5, 476.4),
        (158.5, 458.5, 172.5, 476.4), (174.5, 458.5, 188.5, 476.4)
    ]
    
    sm_rects_f1 = [
        (36.4, 146.3, 50.4, 164.1), (52.4, 146.3, 66.4, 164.1),
        (68.4, 146.3, 82.4, 164.1), (84.4, 146.3, 98.4, 164.1),
        (100.4, 146.3, 114.5, 164.1), (116.4, 146.3, 130.5, 164.1),
        (132.5, 146.3, 146.5, 164.1), (148.5, 146.3, 162.5, 164.1),
        (164.5, 146.3, 178.5, 164.1), (180.5, 146.3, 194.5, 164.1)
    ]
    pm_rects_f1 = [
        (312.5, 146.3, 326.5, 164.1), (328.5, 146.3, 342.5, 164.1),
        (344.5, 146.3, 358.5, 164.1), (360.5, 146.3, 374.5, 164.1),
        (376.5, 146.3, 390.6, 164.1), (392.5, 146.3, 406.6, 164.1),
        (408.6, 146.3, 422.6, 164.1), (424.6, 146.3, 438.6, 164.1),
        (440.6, 146.3, 454.6, 164.1), (456.6, 146.3, 470.6, 164.1)
    ]
    
    sm_rects_f2 = [
        (36.4, 532.9, 50.4, 550.7), (52.4, 532.9, 66.4, 550.7),
        (68.4, 532.9, 82.4, 550.7), (84.4, 532.9, 98.4, 550.7),
        (100.4, 532.9, 114.5, 550.7), (116.4, 532.9, 130.5, 550.7),
        (132.5, 532.9, 146.5, 550.7), (148.5, 532.9, 162.5, 550.7),
        (164.5, 532.9, 178.5, 550.7), (180.5, 532.9, 194.5, 550.7)
    ]
    pm_rects_f2 = [
        (312.5, 532.9, 326.5, 550.7), (328.5, 532.9, 342.5, 550.7),
        (344.5, 532.9, 358.5, 550.7), (360.5, 532.9, 374.5, 550.7),
        (376.5, 532.9, 390.6, 550.7), (392.5, 532.9, 406.6, 550.7),
        (408.6, 532.9, 422.6, 550.7), (424.6, 532.9, 438.6, 550.7),
        (440.6, 532.9, 454.6, 550.7), (456.6, 532.9, 470.6, 550.7)
    ]
    
    cutoff_rects_f1 = [
        (348.0, 201.5, 381.4, 219.0), (383.4, 201.5, 416.9, 219.0),
        (418.9, 201.5, 452.3, 219.0), (487.1, 201.5, 509.1, 219.0),
        (511.1, 201.5, 530.3, 219.0)
    ]
    cutoff_rects_f2 = [
        (355.7, 588.1, 389.1, 605.8), (391.1, 588.1, 424.6, 605.8),
        (426.6, 588.1, 452.9, 605.8), (487.1, 588.1, 516.7, 605.8),
        (518.7, 588.1, 545.1, 605.8)
    ]

    def populate_pdf(doc_key, data):
        doc = fitz.open(template_pdf)
        page = doc[0]
        
        layout_pts = {
            "form_1": {
                "name": (106.0, 114.0),
                "school": (360.0, 114.0),
                "email": (78.0, 196.0),
                "date_rects": date_rects_f1,
                "sm_rects": sm_rects_f1,
                "pm_rects": pm_rects_f1,
                "cutoff_rects": cutoff_rects_f1,
                "q1_bubbles": {"Yes": (306.5, 243.5), "No": (351.5, 243.5)},
                "q2_bubbles": {"5": (81.8, 285.3), "4": (190.2, 285.3), "3": (298.6, 285.3), "2": (407.1, 285.3), "1": (515.5, 285.3)},
                "q3_bubbles": {"5": (81.8, 340.8), "4": (190.2, 340.8), "3": (298.6, 340.8), "2": (407.1, 340.8), "1": (515.5, 340.8)},
                "q4_bubbles": {"Yes": (354.8, 360.9), "No": (400.0, 360.9)}
            },
            "form_2": {
                "name": (106.0, 500.5),
                "school": (360.0, 500.5),
                "email": (78.0, 582.5),
                "date_rects": date_rects_f2,
                "sm_rects": sm_rects_f2,
                "pm_rects": pm_rects_f2,
                "cutoff_rects": cutoff_rects_f2,
                "q1_bubbles": {"Yes": (306.5, 630.0), "No": (351.5, 630.0)},
                "q2_bubbles": {"5": (81.8, 672.0), "4": (190.2, 672.0), "3": (298.6, 672.0), "2": (407.1, 672.0), "1": (515.5, 672.0)},
                "q3_bubbles": {"5": (81.8, 727.8), "4": (190.2, 727.8), "3": (298.6, 727.8), "2": (407.1, 727.8), "1": (515.5, 727.8)},
                "q4_bubbles": {"Yes": (354.8, 747.6), "No": (400.0, 747.6)}
            }
        }
        
        for f_key in ["form_1", "form_2"]:
            cfg = layout_pts[f_key]
            f_data = data[f_key]
            
            # Helper to draw white box and insert centered digit
            def insert_boxed_digit(rect_coords, digit_char):
                r = fitz.Rect(*rect_coords)
                page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1))
                # Center text inside rect
                tx = r.x0 + 3.0
                ty = r.y1 - 2.5
                page.insert_text((tx, ty), digit_char, fontsize=15, fontname="times-bold", color=(0.05, 0.1, 0.45))
                
            # 1. Student Name
            if f_data.get("student_name"):
                page.insert_text(cfg["name"], f_data["student_name"], fontsize=12, fontname="times-bold", color=(0.05, 0.1, 0.45))
                
            # 2. School Name
            if f_data.get("school_name"):
                page.insert_text(cfg["school"], f_data["school_name"], fontsize=12, fontname="times-bold", color=(0.05, 0.1, 0.45))
                
            # 3. Email
            if f_data.get("email"):
                page.insert_text(cfg["email"], f_data["email"], fontsize=12, fontname="times-roman", color=(0.05, 0.1, 0.45))
                
            # 4. Date digits (DDMMYYYY)
            if f_data.get("date"):
                d_str = f_data["date"].replace("/", "")
                for idx, ch in enumerate(d_str[:8]):
                    insert_boxed_digit(cfg["date_rects"][idx], ch)
                    
            # 5. Student Mobile
            if f_data.get("student_mobile"):
                sm_str = f_data["student_mobile"]
                for idx, ch in enumerate(sm_str[:10]):
                    insert_boxed_digit(cfg["sm_rects"][idx], ch)
                    
            # 6. Parent Mobile (might be None)
            if f_data.get("parent_mobile"):
                pm_str = f_data["parent_mobile"]
                for idx, ch in enumerate(pm_str[:10]):
                    insert_boxed_digit(cfg["pm_rects"][idx], ch)
                    
            # 7. Cutoff Marks (XXX.XX -> 5 digits)
            if f_data.get("cutoff_marks"):
                co_str = f_data["cutoff_marks"].replace(".", "")
                for idx, ch in enumerate(co_str[:5]):
                    insert_boxed_digit(cfg["cutoff_rects"][idx], ch)
                    
            # Helper to draw filled bubble circle
            def draw_bubble(center, fill_color=(0.05, 0.12, 0.45), r=2.6):
                cx, cy = center
                page.draw_circle((cx, cy), r, color=fill_color, fill=fill_color)
                
            # 8. Bubbles
            q1_ans = f_data.get("interested_in_bit")
            if q1_ans in cfg["q1_bubbles"]:
                draw_bubble(cfg["q1_bubbles"][q1_ans])
                
            q2_ans = f_data.get("programme_usefulness")
            if q2_ans in cfg["q2_bubbles"]:
                draw_bubble(cfg["q2_bubbles"][q2_ans])
                
            q3_ans = f_data.get("speaker_effectiveness")
            if q3_ans in cfg["q3_bubbles"]:
                draw_bubble(cfg["q3_bubbles"][q3_ans])
                
            q4_ans = f_data.get("choice_filling_assistance")
            if q4_ans == "AMBIGUOUS":
                draw_bubble(cfg["q4_bubbles"]["Yes"])
                draw_bubble(cfg["q4_bubbles"]["No"])
            elif q4_ans in cfg["q4_bubbles"]:
                draw_bubble(cfg["q4_bubbles"][q4_ans])

        pdf_out_path = os.path.join(out_dir, f"{doc_key}.pdf")
        doc.save(pdf_out_path)
        doc.close()
        print(f"Generated {pdf_out_path}")
        return pdf_out_path

    # Generate PDF 1 and PDF 2
    pdf1 = populate_pdf("val_doc_unseen_1", ground_truth["doc_1"])
    pdf2 = populate_pdf("val_doc_unseen_2", ground_truth["doc_2"])
    
    # Render to 300 DPI images with realistic physical perturbations
    def render_and_perturb(pdf_path, out_img_path, angle_deg=0.0, brightness_offset=0, contrast_scale=1.0):
        doc = fitz.open(pdf_path)
        pix = doc[0].get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()
        
        # 1. Skew / Rotation
        if abs(angle_deg) > 0.01:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            
        # 2. Contrast & Brightness
        if contrast_scale != 1.0 or brightness_offset != 0:
            img = cv2.convertScaleAbs(img, alpha=contrast_scale, beta=brightness_offset)
            
        cv2.imwrite(out_img_path, img)
        print(f"Saved perturbed image {out_img_path} (rotation={angle_deg}°, brightness={brightness_offset}, contrast={contrast_scale})")
        return out_img_path

    render_and_perturb(pdf1, os.path.join(out_dir, "val_doc_unseen_1.png"), angle_deg=0.35, brightness_offset=4, contrast_scale=1.01)
    render_and_perturb(pdf2, os.path.join(out_dir, "val_doc_unseen_2.png"), angle_deg=-0.45, brightness_offset=-3, contrast_scale=1.02)
    
    print("\nIndependent validation dataset successfully created with 4 forms across 2 documents!")

if __name__ == "__main__":
    create_validation_datasets()
