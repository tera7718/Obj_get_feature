import json
import numpy as np
import faiss
from openai import OpenAI
from collections import defaultdict


class SceneRAGRetriever:
    def __init__(self, json_path: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
        self.data = self._load_json(json_path)
        self.texts = self._to_texts(self.data)
        self.embeddings = self._create_embeddings(self.texts)
        self.index = self._build_index(self.embeddings)

    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Loaded {len(data)} objects from {path}")
        return data

    def _to_texts(self, data):
        texts = []
        for obj in data:
            text = ", ".join([f"{k}: {v}" for k, v in obj.items() if k != "sceneID" and v])
            texts.append(text)
        return texts

    def _create_embeddings(self, texts):
        print("[INFO] Creating embeddings...")
        embeddings = []
        for text in texts:
            emb = self.client.embeddings.create(
                input=text,
                model=self.model
            ).data[0].embedding
            embeddings.append(emb)
        embeddings = np.array(embeddings).astype("float32")
        print("[INFO] Embeddings created.")
        return embeddings

    def _build_index(self, embeddings):
        print("[INFO] Building FAISS index (cosine similarity mode)...")
        # 🔹 コサイン類似度では内積(IndexFlatIP)を使用
        faiss.normalize_L2(embeddings)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        print(f"[INFO] Index built with {len(embeddings)} vectors.")
        return index

    def search_by_scene(self, query: str, threshold: float = 0.3):
        print(f"\n[QUERY] {query}")
        query_emb = self.client.embeddings.create(
            input=query, model=self.model
        ).data[0].embedding
        query_emb = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(query_emb)  # 🔹 正規化してコサイン類似度に対応

        D, I = self.index.search(query_emb, len(self.data))  # 類似度の高い順
        print(D[0])

        Non_filtered = {idx: (self.data[idx], D[0][i]) for i, idx in enumerate(I[0])}
        for idx, (obj, sim) in Non_filtered.items():
            scene = obj.get("sceneID", "unknown")
            print(f"Scene: {scene} | 類似度: {sim:.3f} | 候補: {obj}")

        # 各sceneIDの最も類似度が高いものを取得
        best_per_scene = defaultdict(lambda: (None, -1.0))
        for idx, sim in zip(I[0], D[0]):
            item = self.data[idx]
            scene = item.get("sceneID", "unknown")
            if sim > best_per_scene[scene][1]:
                best_per_scene[scene] = (item, sim)

        # threshold以上のものだけ残す（類似度が高い＝近い）
        filtered = {
            scene: (obj, sim)
            for scene, (obj, sim) in best_per_scene.items()
            if sim > threshold
        }

        print("\n[RESULTS: Each Scene Top Candidate]")
        if filtered:
            for scene, (obj, sim) in filtered.items():
                print(f"Scene: {scene} | 類似度: {sim:.3f} | 候補: {obj}")
        else:
            print("No candidates above threshold.")

        return filtered, Non_filtered

    def refine_search(self, query: str, candidates: dict):
        print("\n[REFINE] 再検索を実行中...")
        sub_data = [obj for obj, _ in candidates.values()]
        sub_texts = self._to_texts(sub_data)
        sub_embeddings = self._create_embeddings(sub_texts)
        faiss.normalize_L2(sub_embeddings)
        sub_index = faiss.IndexFlatIP(sub_embeddings.shape[1])
        sub_index.add(sub_embeddings)

        query_emb = self.client.embeddings.create(
            input=query, model=self.model
        ).data[0].embedding
        query_emb = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(query_emb)

        D, I = sub_index.search(query_emb, len(sub_data))
        print("\n[REFINE RESULTS]")
        for idx, sim in zip(I[0], D[0]):
            print(f"類似度: {sim:.3f} | 再候補: {sub_data[idx]}")

        # 最も類似度が高い（simが最大）の候補を選択
        best_idx = I[0][np.argmax(D[0])]
        best_sim = np.max(D[0])
        best_obj = sub_data[best_idx]

        print("\n✅ 最も類似度が高い候補:")
        print(f"類似度: {best_sim:.3f} | オブジェクト: {best_obj}")

        return best_obj, best_sim


if __name__ == "__main__":
    retriever = SceneRAGRetriever(json_path="./assets/object.json")

    query = input("\n命令文を入力してください: ")
    filtered, non_filtered = retriever.search_by_scene(query, threshold=0.1)

    refine_query = input("\n再検索の命令文を入力してください（またはEnterでスキップ）: ")
    if refine_query.strip():
        retriever.refine_search(refine_query, filtered)