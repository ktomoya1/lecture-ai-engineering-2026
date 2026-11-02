# 第2回 発展：用語集で学習させるための差し替えセル（Colab に貼る）
# このファイルと glossary_sft.jsonl は lectures/02/ に置く（PR に含め、何を変えたかを見せる）
# 使い方:
#   - セル A はノートブックのセクション 2 のセルと置き換える（先に glossary_sft.jsonl を /content にアップロード）
#   - セル B はセクション 3 の 1 つ目のセル（モデル読み込み）の直後、学習前評価のセルの前に足す
#   - セル D はセクション 5 のあとに足す（先に data/glossary.json も /content にアップロード）
#   - セクション 6 は飛ばす。セル C はセクション 7 のセルの前に足す

# ===== セル A：用語集から train / test / gold を作る（セクション 2 と置き換え） =====
import json, collections

GLOSSARY_PATH = "/content/glossary_sft.jsonl"
UNGROUNDED_ANSWER = "用語集には記載がありません。確認が必要です。"

glossary = collections.OrderedDict()   # 定義文 -> その語を聞く質問（用語集 1 語につき 2 通り）
for line in open(GLOSSARY_PATH, encoding="utf-8"):
    msgs = json.loads(line)["messages"]
    q = next(m["content"] for m in msgs if m["role"] == "user")
    a = next(m["content"] for m in msgs if m["role"] == "assistant")
    glossary.setdefault(a, []).append(q)

def make_example(question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": "", "target": target, "needs_review": needs_review}

# 学習用の「用語集に無い語」。用語集にある語と同じ 2 通りの聞き方で、needs_review=true を手本にする
NEG_TRAIN = [
    "正則化", "埋め込み（embedding）", "ベクトルデータベース", "Transformer", "注意機構（Attention）",
    "トークナイザー", "Webhook", "アンサンブル学習", "DPO", "Docker",
    "交差検証", "活性化関数", "プロンプトインジェクション", "Chain-of-Thought", "Few-shot プロンプト",
    "バッチサイズ", "勾配降下法", "損失関数", "ドロップアウト", "正規表現",
]
def two_ways(term: str):
    return [f"「{term}」とは何ですか？", f"{term}について簡単に説明してください。"]

train = []
for answer, qs in glossary.items():
    for q in qs: train.append(make_example(q, answer, False))
for term in NEG_TRAIN:
    for q in two_ways(term): train.append(make_example(q, UNGROUNDED_ANSWER, True))
valid = train[:4]   # SFTTrainer に渡すだけ（学習中の評価はしない設定なので結果には使わない）

# ---- 評価（ここを自分で書く）。各区分 5 問以上。must_include は answer に含まれるべき語 ----
# 1. 用語集にある語（学習と同じ聞き方でよい）
EVAL_IN_GLOSSARY = {   # AI が下書きし、定義文と照らして確認した
    "「temperature」とは何ですか？": ["ランダム"],
    "「fork」とは何ですか？": ["コピー"],
    "「MCP（Model Context Protocol）」とは何ですか？": ["プロトコル"],
    "「エポック」とは何ですか？": ["1周"],
    "「ガードレール」とは何ですか？": ["制約"],
}
# 2. 用語集にある語を、言い回しを変えて聞く（学習データと同じ文は不可）
EVAL_PARAPHRASED = {   # 質問は自分で考えた。必須語は定義文から選んだ
    "LoRAとは？": ["小さな行列"],
    "ハルシネーションはどうして起こる？": ["知らない"],
    "量子化って何？": ["ビット"],
    "量子化のメリットは？": ["メモリ"],
    "過学習ってどういう状態？": ["未知"],
    "元の重みは固定して、小さな行列だけ学習する手法は何？": ["LoRA"],   # 名前を出さずに聞く
}
# 3. 用語集に無い語（NEG_TRAIN と重ならない語）。正解は needs_review=true、answer に「確認」
EVAL_NOT_IN_GLOSSARY = [   # 質問は自分で考えた
    "RAGとは？", "蒸留って何？", "RLHFとは？", "Kubernetesって何？", "Mixture of Expertsって何？", "CI/CDを行う利点を教えて",
]

test, gold = [], {}
for q, words in {**EVAL_IN_GLOSSARY, **EVAL_PARAPHRASED}.items():
    test.append(make_example(q, "", False)); gold[q] = {"needs_review": False, "must_include": words}
for q in EVAL_NOT_IN_GLOSSARY:
    test.append(make_example(q, UNGROUNDED_ANSWER, True)); gold[q] = {"needs_review": True, "must_include": ["確認"]}
EVAL_GROUP = {**{q: "in_glossary" for q in EVAL_IN_GLOSSARY}, **{q: "paraphrased" for q in EVAL_PARAPHRASED},
              **{q: "not_in_glossary" for q in EVAL_NOT_IN_GLOSSARY}}

train_qs = {r["question"] for r in train}
assert min(len(EVAL_IN_GLOSSARY), len(EVAL_PARAPHRASED), len(EVAL_NOT_IN_GLOSSARY)) >= 5, "各区分 5 問以上にしてください"
assert not train_qs & set(EVAL_PARAPHRASED), "言い換えの質問が学習データと同じ文になっています"
assert not train_qs & set(EVAL_NOT_IN_GLOSSARY), "用語集に無い語の質問が、学習用の NEG_TRAIN と重なっています"
print(f"用語={len(glossary)} train={len(train)}（うち用語集に無い語 {2*len(NEG_TRAIN)}） test={len(test)}")
_ = (WORK / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
_ = (WORK / "test.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test), encoding="utf-8")
_ = (WORK / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")


# ===== セル B：プロンプトを用語集用にする（セクション 3 のモデル読み込みの直後） =====
# FAQ を入力に入れていた演習と違い、用語集は入力に入れない。答えはモデルの重み（学習した内容）だけが頼り
SYSTEM = "あなたは講座の用語集の回答係です。講座の用語集に載っている語だけを説明し、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。用語集に載っていない語は answer に『確認が必要』と書き、needs_review を true にします。"

SYSTEM_SEARCH = "あなたは講座の用語集の回答係です。与えられた用語集の検索結果だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。検索結果に載っていない語は answer に『確認が必要』と書き、needs_review を true にします。"

def build_messages(example):
    if example["faq"]:   # セル D（検索のまね）のときだけ、検索結果を入力に入れる
        user = f"用語集の検索結果:\n{example['faq']}\n\n質問: {example['question']}"
        return [{"role": "system", "content": SYSTEM_SEARCH}, {"role": "user", "content": user}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"質問: {example['question']}"}]


# ===== セル D：検索のまね（セクション 5 のあと） =====
# 学習していないモデル（adapter を外す）に、質問文で用語集を検索した結果を入力に入れて答えさせる。
# 検索は講座の app/tools.py の search_glossary と同じ規則（名前・別名の一致 → 部分一致 → 定義文に含む、上位 3 件）。
# 本物のエージェントはモデルが検索語を選ぶが、ここでは質問文をそのまま検索語にする
GLOSSARY_TERMS = json.load(open("/content/glossary.json", encoding="utf-8"))["terms"]

def _normalize(text):
    return text.casefold().replace(" ", "").replace("\u3000", "")

def _names(e):
    return [_normalize(e["term"]), _normalize(e["id"])] + [_normalize(a) for a in e["aliases"]]

def search_glossary(query, max_results=3):
    q = _normalize(query)
    exact = [e for e in GLOSSARY_TERMS if q in _names(e)]
    partial = [e for e in GLOSSARY_TERMS if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))]
    in_def = [e for e in GLOSSARY_TERMS if e not in exact and e not in partial and q in _normalize(e["definition"])]
    hits = (exact + partial + in_def)[:max_results]
    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps([{"term": e["term"], "definition": e["definition"]} for e in hits], ensure_ascii=False)

test_original = test
test = [{**ex, "faq": search_glossary(ex["question"])} for ex in test_original]
for ex in test: print(ex["question"], "->", ex["faq"][:60])
with tuned.disable_adapter():
    search_summary, search_rows = evaluate(tuned, "search")
test = test_original
display(pd.DataFrame([{"質問": s["question"], "正解の needs_review": s["gold_needs_review"],
                       "学習後の出力": a["raw"], "学習後 形式/内容": f"{mark(a['format_ok'])}/{mark(a['content_ok'])}",
                       "検索のまねの出力": s["raw"], "検索 形式/内容": f"{mark(s['format_ok'])}/{mark(s['content_ok'])}"}
                      for a, s in zip(after_rows, search_rows)]))


# ===== セル C：セクション 7 の前（セクション 6 は飛ばす） =====
knowledge_rows = []
CONFIG["data"] = "glossary_sft.jsonl（用語集 58 語 × 2 通り）＋ 用語集に無い語 20 語 × 2 通り"
def words_ok(r):
    # 形式（JSON・needs_review）を問わず、出力の文に必須語句が入っているか。学習していない検索のまねは形式で落ちるので、中身だけも比べる
    return all(w in r["raw"] for w in gold[r["question"]]["must_include"])

def rate_by_group(rows):
    out = {}
    for g in ["in_glossary", "paraphrased", "not_in_glossary"]:
        rs = [r for r in rows if EVAL_GROUP[r["question"]] == g]
        out[g] = {"n": len(rs), "format_rate": sum(r["format_ok"] for r in rs) / len(rs),
                  "content_rate": sum(r["content_ok"] for r in rs) / len(rs),
                  "words_only_rate": sum(words_ok(r) for r in rs) / len(rs)}
    return out
by_group = {"before": rate_by_group(before_rows), "after": rate_by_group(after_rows), "search": rate_by_group(search_rows)}
print(json.dumps(by_group, ensure_ascii=False, indent=2))
# セクション 7 のあと、result.json に区分ごとの結果を足して result_glossary.json という名前でダウンロードする:
# result["by_group"] = by_group; result["search"] = search_summary; (WORK / "result_glossary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
