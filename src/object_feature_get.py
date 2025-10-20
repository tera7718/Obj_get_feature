import openai
import base64
import os
import cv2
import re
import matplotlib.pyplot as plt
import re
import json
from pathlib import Path
from PIL import Image

   

class GetFeatureObjBot:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
    


    def encode_image(self, output_path):
        image_list = []
        for image_path in [output_path]:
            image_list.append(image_path)
        print(image_path)
        self.base64_image_list = []
        for image_path in (image_list):
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                self.base64_image_list.append(base64_image)
        return  

    def resize_image(self, input_path, output_path, new_size=(640, 480)):

        # 開く
        img = Image.open(input_path)

        # サイズを指定してリサイズ（幅, 高さ）
        new_size = (1024, 768)
        resized_img = img.resize(new_size)

        # 保存
        resized_img.save(output_path)

        print(f"画像を {new_size} にリサイズして保存しました: {output_path}")



    def set_inital_prompt(self, scene_id) -> None:
        self.messages = [
            {"role": "system", "content": "Get features of objects in the image and ouput the features and output features in json format."},
            {"role": "system", "content": "four features: sceneID, label, text, color, shape, position, material. If you cannot recognize the object, output Unknown."},
            {"role": "system", "content": f"In this time, sceneID is {scene_id}"},
            {"role": "system", "content": "If the text is not specified on the object, abusolutely output Unknown."},
            {"role": "system", "content": "If there are same objects, You must change labels of objects acording to the number at objects labels"},
            {"role": "system", "content": "Output the result in json format like [{'sceneID': 'test_scene1', 'label': 'trash_can1', 'text': 'Recycle', 'color': 'green', 'shape': 'cylindrical', 'position': 'left side', 'material': 'plastic'}, {'sceneID': 'test_scene1', 'label': 'trash_can2', 'text': 'Trash', 'color': 'black', 'shape': 'cylindrical', 'position': 'right side', 'material': 'metal'}]"},

        ]


    def create_chat(self, input_image_path, scene_id="test"):       
        self.encode_image(input_image_path)
        
        self.set_inital_prompt(scene_id)
        print("sent info to gpt-4o")
        self.messages.append(
            {"role": "user","content":
               [
                    # {"type": "text", "text": ""},
                    {"type": "image_url","image_url": {"url":  f"data:image/jpeg;base64,{self.base64_image_list[0]}",},
                    },
                ],
            },
        )

        
        response = openai.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=self.messages,
            temperature=0,
        )
        print("completed")
        print("response:", response.choices[0].message.content)
        return response.choices[0].message.content

    def append_result(self, new_result, filename="../io/results.json"):
        """
        JSONファイルに結果を追記して保存する関数。
        既存ファイルがなければ新規作成する。
        """
        file_path = Path(filename)

        # 既存ファイルを読み込み（なければ空リスト）
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        if isinstance(new_result, str):
            try:
                new_result = json.loads(new_result)
            except json.JSONDecodeError:
                print("⚠️ new_result が正しいJSON文字列ではありません。スキップします。")
                return
            # 結果を追加
        data.append(new_result)

        # 上書き保存
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"新しい結果を {filename} に追加しました。")
    

if __name__ == "__main__":
    scene_id = "test_trash_can1"
    input_image_path  = f"../test_pig/{scene_id}.png"
    resize_image_path = "../test_pig/resized_image.png"
    # 1) 新しいインスタンスで初期化
    response = GetFeatureObjBot()
    # 2) チャットの開始
    response.resize_image(input_image_path, resize_image_path)
    answer = response.create_chat(resize_image_path, scene_id)
    # 3) 結果の保存
    response.append_result(answer)



