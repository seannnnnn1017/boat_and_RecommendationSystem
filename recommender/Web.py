from typing import Dict, List, Set
from sentence_transformers import SentenceTransformer, util
import streamlit as st
# 載入預訓練的多語言 SBERT 模型
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 課程數據
courses = {
    # 核心課程
    "A": {"name": "基礎程式設計", "desc": "學習 Python 的基礎語法和簡單程式設計概念", "prereqs": set(), "topics": {"程式設計", "Python"}, "type": "核心"},
    "B": {"name": "資料結構", "desc": "介紹基本的資料結構，如陣列、鏈結串列和樹", "prereqs": {"A"}, "topics": {"程式設計", "資料結構"}, "type": "核心"},
    "C": {"name": "機器學習入門", "desc": "學習機器學習的基本概念和簡單模型，使用 Python 實作", "prereqs": {"A"}, "topics": {"機器學習", "Python", "AI"}, "type": "核心"},
    "D": {"name": "進階程式設計", "desc": "深入探討物件導向程式設計和演算法", "prereqs": {"A"}, "topics": {"程式設計", "演算法"}, "type": "核心"},
    "E": {"name": "人工智慧概論", "desc": "介紹 AI 的基礎理論和應用，不含實作", "prereqs": set(), "topics": {"AI", "理論"}, "type": "核心"},

    # 通識課程
    "GE1": {"name": "科技與社會", "desc": "探討科技對社會的影響，涵蓋 AI 和數位化趨勢", "prereqs": set(), "topics": {"科技", "社會", "AI"}, "type": "通識"},
    "GE2": {"name": "環境科學入門", "desc": "介紹環境問題與可持續發展的基本概念", "prereqs": set(), "topics": {"環境", "科學"}, "type": "通識"},
    "GE3": {"name": "世界經濟", "desc": "介紹國際世界觀，經濟脈絡", "prereqs": set(), "topics": {"國際", "經濟"}, "type": "通識"},
    "GE4": {"name": "數據素養與決策", "desc": "培養數據思維，理解資料背後的意義與決策依據", "prereqs": set(), "topics": {"資料分析", "決策", "數據素養"}, "type": "通識"},
    "GE5": {"name": "心理學概論", "desc": "認識人類行為與思維的基本心理學原理", "prereqs": set(), "topics": {"心理學", "人文"}, "type": "通識"},

    # 選修課程
    "EL1": {"name": "網頁開發基礎", "desc": "學習 HTML、CSS 和 JavaScript 打造簡單網頁", "prereqs": {"A"}, "topics": {"程式設計", "網頁開發"}, "type": "選修"},
    "EL2": {"name": "深度學習概論", "desc": "介紹深度學習的基本原理和應用，使用 Python", "prereqs": {"C"}, "topics": {"機器學習", "AI", "Python"}, "type": "選修"},
    "EL3": {"name": "遊戲設計入門", "desc": "學習遊戲設計基礎，包含程式設計與創意發想", "prereqs": {"A"}, "topics": {"程式設計", "遊戲設計"}, "type": "選修"},

    # ✅ 統計相關課程
    "ST1": {"name": "統計學基礎", "desc": "學習描述統計、機率分布與假設檢定的基本概念", "prereqs": set(), "topics": {"統計", "資料分析", "數學"}, "type": "核心"},
    "ST2": {"name": "機率論", "desc": "探討隨機變數、期望值、機率分布及其應用", "prereqs": {"ST1"}, "topics": {"機率", "統計", "數學"}, "type": "選修"},
    "ST3": {"name": "R 語言資料分析", "desc": "學習使用 R 語言進行資料清理與視覺化", "prereqs": {"ST1"}, "topics": {"統計", "資料分析", "R"}, "type": "選修"},

    # ✅ 管理學系課程
    "MG1": {"name": "管理學原理", "desc": "介紹組織管理的基本原則、領導、規劃與控制", "prereqs": set(), "topics": {"管理", "組織"}, "type": "核心"},
    "MG2": {"name": "行銷學概論", "desc": "探討行銷策略、市場分析與顧客行為", "prereqs": set(), "topics": {"行銷", "管理", "市場"}, "type": "選修"},
    "MG3": {"name": "財務管理", "desc": "學習資金運用、財務報表分析與投資決策", "prereqs": {"MG1"}, "topics": {"財務", "管理", "投資"}, "type": "選修"}



}

# 建立課程語意向量
course_embeddings: Dict[str, any] = {}
for cid, info in courses.items():
    topic_text = "、".join(info["topics"])
    full_text = f"{info['name']}：{info['desc']}。主題包括：{topic_text}"
    course_embeddings[cid] = sbert_model.encode(full_text, convert_to_tensor=True)

# 先修檢查
def check_prereqs(course_id: str, completed: Set[str]) -> bool:
    return courses[course_id]["prereqs"].issubset(completed)

# 課程推薦
def recommend_courses(completed: Set[str], interests: str, top_n: int):
    user_text = f"我對{interests}有興趣"
    interest_embedding = sbert_model.encode(user_text, convert_to_tensor=True)

    scored_courses = []
    for cid, info in courses.items():
        if cid in completed:
            continue  # ✅ 跳過已修過的課程
        if check_prereqs(cid, completed):
            sim = util.pytorch_cos_sim(interest_embedding, course_embeddings[cid]).item()
            scored_courses.append((cid, sim, info["type"]))

    scored_courses.sort(key=lambda x: (x[1], {"核心": 0, "選修": 1, "通識": 2}.get(x[2], 3)), reverse=True)

    results = []
    for cid, sim, ctype in scored_courses[:top_n]:
        course = courses[cid]
        results.append(f"✅ 推薦「{course['name']}」({ctype})，語意相似度為 {sim:.2f}")
    return results


# === Streamlit UI ===
st.title("📚 選課推薦系統（SBERT 語意分析）")

selected_completed = st.multiselect("✅ 請勾選你已修過的課程：",
                                    options=list(courses.keys()),
                                    format_func=lambda x: f"{x} - {courses[x]['name']}")

user_interest = st.text_input("💡 請輸入你感興趣的領域（如：AI、資料分析、管理...）：")

top_n = st.selectbox("📌 想要回傳幾門推薦課程？", options=[1, 3, 5, 10], index=2)

if st.button("🚀 產生推薦"):
    if not user_interest.strip():
        st.warning("請先輸入興趣內容！")
    else:
        recommendations = recommend_courses(set(selected_completed), user_interest, top_n)
        st.subheader("📋 推薦結果：")
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.info("沒有符合興趣與先修條件的課程。")