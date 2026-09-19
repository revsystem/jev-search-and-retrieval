"""Shared instruction text for the search and retrieval judgements."""

RELEVANCE_LEVELS = [
    "無関係。クエリの主題とも参照先とも一致しない。",
    "話題は重なるが、クエリへの回答根拠にはならない。",
    "部分的な根拠を含む。回答の一部を裏づけるが単独では不十分。",
    "直接的な根拠を含む。この記述だけでクエリに答えられる。",
]

RELEVANCE_CRITERIA = {
    "true": "クエリが指す対象そのものを扱い、回答を裏づける事実・数値・定義を含む。",
    "false": "話題が重なるだけ、参照先が異なる、または根拠となる情報を含まない。",
}

NOUL_INSTRUCTION = (
    'state.candidates["{key}"] は state.query に答えるための根拠として役に立ちますか。'
)

SCORE_INSTRUCTION = (
    'state.query に答えるための根拠として、state.candidates["{key}"] の有用性を判定してください。'
    "話題が重なるだけの文章は低く、クエリが指す対象そのものを扱い数値・定義・事実を示す文章を高く評価します。"
)
