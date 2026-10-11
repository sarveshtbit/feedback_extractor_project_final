import os
import sys
import cv2
import yaml
import numpy as np

def main():
    print("=== TASK 3: DESIGN AND VISUALLY VERIFY REFINED FIELD BOUNDARIES ===")
    template_path = "bannari_amman_form_template.png"
    tmpl = cv2.imread(template_path)
    h, w = tmpl.shape[:2]
    
    with open("config/form_template.yaml", "r") as f:
        orig_cfg = yaml.safe_load(f)
        
    # Let's inspect the fields on canvas (w=2479, h=3508)
    # Original field normalized bounds:
    # student_name: [0.0815, 0.1830, 0.5000, 0.2130] -> [202, 641, 1239, 747] (prompt "1. Name of the Student :" is x 202 to 910)
    # Refined student_name write-in: x in [910, 1250], y in [641, 747] -> norm: [0.3670, 0.1830, 0.5040, 0.2130]
    
    # school_name: [0.5200, 0.1830, 0.9200, 0.2130] -> [1289, 641, 2280, 747] (write-in space only) -> norm: [0.5200, 0.1830, 0.9200, 0.2130]
    
    # date: [0.0815, 0.1431, 0.4776, 0.1822] -> [202, 501, 1183, 639] (prompt "Date of Birth :" is x 202 to 570; 8 boxes are x 580 to 1180)
    # Refined date (boxes only): x in [580, 1183], y in [501, 639] -> norm: [0.2340, 0.1431, 0.4776, 0.1822]
    
    # student_mobile: [0.0815, 0.2144, 0.5785, 0.2535] -> [202, 752, 1434, 889] (prompt "Mobile No :" is x 202 to 460; 10 boxes are x 470 to 1434)
    # Refined student_mobile (boxes only): x in [470, 1434], y in [752, 889] -> norm: [0.1895, 0.2144, 0.5785, 0.2535]
    
    # parent_mobile: [0.5857, 0.2144, 0.9950, 0.2535] -> [1451, 752, 2466, 889] (prompt "Parent's Mobile :" is x 1451 to 1840; 10 boxes are x 1850 to 2466)
    # Refined parent_mobile (boxes only): x in [1850, 2466], y in [752, 889] -> norm: [0.7460, 0.2144, 0.9950, 0.2535]
    
    # email: [0.0815, 0.2540, 0.4776, 0.2840] -> [202, 891, 1183, 996] (prompt "3. E-mail :" is x 202 to 550; write-in is x 560 to 1183)
    # Refined email: x in [560, 1183], y in [891, 996] -> norm: [0.2260, 0.2540, 0.4776, 0.2840]
    
    # cutoff_marks: [0.5750, 0.2851, 0.8550, 0.3224] -> [1425, 1000, 2119, 1130] (boxes only) -> norm: [0.5750, 0.2851, 0.8550, 0.3224]

    refined_cfg = yaml.safe_load(yaml.dump(orig_cfg))
    refined_fields = {
        "student_name": {
            "type": "text_region",
            "normalized_bbox": [0.3670, 0.1830, 0.5040, 0.2130],
            "description": "Student name write-in area excluding prompt label"
        },
        "school_name": {
            "type": "text_region",
            "normalized_bbox": [0.5200, 0.1830, 0.9200, 0.2130],
            "description": "School name write-in area"
        },
        "date": {
            "type": "character_boxes",
            "count": 8,
            "normalized_bbox": [0.2340, 0.1431, 0.4776, 0.1822],
            "boxes": orig_cfg['fields']['date']['boxes'],
            "description": "Date 8 character boxes excluding prompt label"
        },
        "student_mobile": {
            "type": "character_boxes",
            "count": 10,
            "normalized_bbox": [0.1895, 0.2144, 0.5785, 0.2535],
            "boxes": orig_cfg['fields']['student_mobile']['boxes'],
            "description": "Student mobile 10 character boxes excluding prompt label"
        },
        "parent_mobile": {
            "type": "character_boxes",
            "count": 10,
            "normalized_bbox": [0.7460, 0.2144, 0.9950, 0.2535],
            "boxes": orig_cfg['fields']['parent_mobile']['boxes'],
            "description": "Parent mobile 10 character boxes excluding prompt label"
        },
        "email": {
            "type": "text_region",
            "normalized_bbox": [0.2260, 0.2540, 0.4776, 0.2840],
            "description": "Email write-in line excluding prompt label"
        },
        "cutoff_marks": {
            "type": "character_boxes",
            "count": 6,
            "normalized_bbox": [0.5750, 0.2851, 0.8550, 0.3224],
            "boxes": orig_cfg['fields']['cutoff_marks']['boxes'],
            "description": "Cutoff marks 6 boxes"
        }
    }
    refined_cfg['fields'] = refined_fields
    
    os.makedirs("config", exist_ok=True)
    with open("config/form_template_experimental.yaml", "w") as f:
        yaml.dump(refined_cfg, f, sort_keys=False)
        
    print("Saved refined configuration to config/form_template_experimental.yaml")
    
    # Draw visual comparison on blank template and form1
    out_dir = "storage/stage8_paddleocr_verification/visual_crops"
    os.makedirs(out_dir, exist_ok=True)
    
    for img_name, img_path in [("template", template_path), ("form1", "cnn_dataset/form1.jpeg")]:
        img_raw = cv2.imread(img_path)
        vis_img = img_raw.copy()
        
        # Draw Original BBoxes in RED
        for f_name, f_data in orig_cfg['fields'].items():
            b = f_data['normalized_bbox']
            x1, y1, x2, y2 = int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(vis_img, f"ORIG: {f_name}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Draw Refined BBoxes in GREEN
        for f_name, f_data in refined_fields.items():
            b = f_data['normalized_bbox']
            x1, y1, x2, y2 = int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(vis_img, f"REFINED: {f_name}", (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            
        out_vis_path = f"{out_dir}/{img_name}_crops_comparison.png"
        cv2.imwrite(out_vis_path, vis_img)
        print(f"Saved visual overlay comparison to {out_vis_path}")

if __name__ == '__main__':
    main()
