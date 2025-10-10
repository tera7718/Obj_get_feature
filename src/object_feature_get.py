import openai
import base64
import os
import cv2
import re
import matplotlib.pyplot as plt
import re

   

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




    def set_inital_prompt(self) -> None:
        self.messages = [
            {"role": "system", "content": "Get features of objects in the image and ouput the features and output features in json format."},
            {"role": "system", "content": "four features: label, color, shape, position, material. If you cannot recognize the object, output Unknown."}, 
            {"role": "system", "content": "If the label is not specified on the object, abusolutely output Unknown."},
        ]


    def create_chat(self, input_image_path):       
        self.encode_image(input_image_path)
        
        self.set_inital_prompt()
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


if __name__ == "__main__":
    input_image_path  = "../test_pig/bbox_scene1.jpg"
    # 1) 新しいインスタンスで初期化
    response = GetFeatureObjBot()
    # 2) チャットの開始
    answer = response.create_chat(input_image_path)


