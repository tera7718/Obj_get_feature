import json

# --- JSONの読み込み ---
with open("/home/hma/tera_ws/1_skill/tera_obj_feature/io/results.json", "r") as f:
    scene_data = json.load(f)

with open("/home/hma/docker_skill/detic-ros-docker/detic_ros/output/detic_results.json", "r") as f:
    detic_data = json.load(f)

# --- マージ処理 ---
for scene_name, content in detic_data.items():
    labels = content["labels"]
    boxes = content["positions"]

    for group in scene_data:
        for obj in group:
            # sceneIDにscene_nameが含まれている場合のみ対象にする（例: test_trash_can1 → sample_mroom）
            if scene_name in obj["sceneID"] or obj["sceneID"] in scene_name:
                for lbl, box in zip(labels, boxes):
                    # アンダースコアとスペースの違いを無視して比較
                    if obj["label"].replace("_", " ").lower() == lbl.lower():
                        obj["bbox"] = box

# --- 保存 ---
output_path = "/home/hma/tera_ws/1_skill/tera_obj_feature/io/results.json"
with open(output_path, "w") as f:
    json.dump(scene_data, f, indent=4)

print(f"✅ Merged JSON saved to {output_path}")
