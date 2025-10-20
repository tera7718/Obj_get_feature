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

   

class TargetSpecificLLMBot:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def open_json(self, filename="../io/results.json"):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            # print(f"Loaded JSON data from {filename}: {data}")
        return data


    def set_inital_prompt(self) -> None:
        json_data = self.open_json()
        self.messages = [
            {"role": "system", "content": "You are a brilliant mathematician."},
            {"role": "system", "content": "You will now need to identify what users are seeking."},
            {"role": "system", "content": f"The following is the object information list: {json_data}."},
            {"role": "system", "content": "Identify what the user is seeking based on the object information and the user's command. If you can identify the target, output the object information in information lists."},
            {"role": "system", "content": "If the user's command does not narrow down the target to a single item, please tell me which attributes from the list should be asked to the user to narrow down the candidates."}, 
            {"role": "system", "content": "You must answer only questions for finding the target object or the object information in json format."},
            {"role": "system", "content": "When you asked questions, youm must ask formatting talking and determine one attributes from color, shape, material."},
            ]


    def create_chat(self):       
        self.set_inital_prompt()
        print("sent info to gpt-4o")
        self.messages.append(
            {"role": "user","content":
               [
                    {"type": "text", "text": "Please tell me a trash can"},
                    # {"type": "image_url","image_url": {"url":  f"data:image/jpeg;base64,{self.base64_image_list[0]}",},
                    # },
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

    # def append_result(self, new_result, filename="../io/results.json"):
    #     """
    #     JSONファイルに結果を追記して保存する関数。
    #     既存ファイルがなければ新規作成する。
    #     """
    #     file_path = Path(filename)

    #     # 既存ファイルを読み込み（なければ空リスト）
    #     if file_path.exists():
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             try:
    #                 data = json.load(f)
    #             except json.JSONDecodeError:
    #                 data = []
    #     else:
    #         data = []
    #     if isinstance(new_result, str):
    #         try:
    #             new_result = json.loads(new_result)
    #         except json.JSONDecodeError:
    #             print("⚠️ new_result が正しいJSON文字列ではありません。スキップします。")
    #             return
    #         # 結果を追加
    #     data.append(new_result)

    #     # 上書き保存
    #     with open(file_path, "w", encoding="utf-8") as f:
    #         json.dump(data, f, ensure_ascii=False, indent=4)

    #     print(f"新しい結果を {filename} に追加しました。")
    

if __name__ == "__main__":
    # 1) 新しいインスタンスで初期化
    response = TargetSpecificLLMBot()
    # 2) チャットの開始
    answer = response.create_chat()
    # # 3) 結果の保存
    # response.append_result(answer)