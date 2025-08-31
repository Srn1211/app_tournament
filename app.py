# -*- coding: utf-8 -*-

import io
import random
import string
from typing import List, Tuple, Union

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
import matplotlib
from matplotlib import font_manager

# ======================= フォント設定 =======================
FONT_CANDIDATES = [
    Path("NotoSansJP-Thin.otf"),
    Path("fonts/NotoSansJP-Thin.otf"),
]
FALLBACK_NAMES = [
    "Noto Sans JP", "Noto Sans CJK JP",
    "IPAexGothic", "Source Han Sans JP", "TakaoPGothic",
]

def _set_jp_font():
    for p in FONT_CANDIDATES:
        if p.exists():
            try:
                font_manager.fontManager.addfont(str(p))
                fp = font_manager.FontProperties(fname=str(p))
                matplotlib.rcParams["font.family"] = fp.get_name()
                return
            except Exception:
                pass
    for f in font_manager.fontManager.ttflist:
        if f.name in FALLBACK_NAMES:
            matplotlib.rcParams["font.family"] = f.name
            return

# ======================= Utilities =======================
ZEN_MAP = str.maketrans("0123456789-", "０１２３４５６７８９－")
def to_zen_digits(x: int) -> str:
    return str(x).translate(ZEN_MAP)

def df_to_png_bytes(
    df: pd.DataFrame,
    width_px: int = 1200,
    dpi: int = 200,
    font_size: int = 11,
    cell_pad: float = 0.5,
    header_cell_pad: float | None = None,
    header_bg: str = "#66cdaa",                       # 1行目（ヘッダ）の背景
    stripe_bg: tuple[str, str] = ("#FFFFFF", "#F7F7F7"),  # データ行の縞々（白/薄グレー）
) -> bytes:
    """DataFrame→PNG（日本語フォント＋セル余白＋縞々背景）"""
    _set_jp_font()

    # 図サイズ
    width_in = max(6.0, width_px / dpi)
    header_in = 0.85
    row_in = 0.40
    height_in = header_in + max(1, len(df)) * row_in

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # 余白設定
    if header_cell_pad is None:
        header_cell_pad = cell_pad
    for (r, _c), cell in table.get_celld().items():
        cell.PAD = header_cell_pad if r == 0 else cell_pad

    # 背景色：ヘッダ=カラー、以降は縞々
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_edgecolor("black")
            cell.set_linewidth(1.0)
            cell.get_text().set_fontweight("bold")
        else:
            bg = stripe_bg[(r - 1) % 2]  # r は1からデータ行
            cell.set_facecolor(bg)
            cell.set_edgecolor("black")
            cell.set_linewidth(0.9)

    # スケールと列幅
    table.scale(1.06, 1.36 + cell_pad * 0.4)
    try:
        table.auto_set_column_width(col=list(range(len(df.columns))))
    except Exception:
        pass

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.28, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def display_df_with_index_from_one(df: pd.DataFrame):
    df_show = df.copy()
    df_show.index = range(1, len(df_show)+1)
    st.dataframe(df_show, use_container_width=True)

# === 結果保持用 ===
if "result" not in st.session_state:
    st.session_state["result"] = None  # dict | None

def _save_result(df: pd.DataFrame, label_png: str, fname_png: str):
    st.session_state["result"] = {
        "df": df,
        "label_png": label_png,
        "fname_png": fname_png,
    }

def _render_result():
    r = st.session_state.get("result")
    if not r:
        st.info("まだ組み合わせを生成していません。")
        return
    df = r["df"]
    display_df_with_index_from_one(df)
    st.download_button(r["label_png"], data=df_to_png_bytes(df, 1200, 200),
                       file_name=r["fname_png"], mime="image/png")

# ======================= 表示用整形 =======================
SinglesPair = Tuple[str, str]
DoublesTeam = Tuple[str, str]
DoublesPair = Tuple[DoublesTeam, DoublesTeam]
AnyPair = Union[SinglesPair, DoublesPair]

def _fmt_team(x: Union[str, DoublesTeam], singles: bool) -> str:
    if singles:
        return str(x)
    if isinstance(x, tuple) and len(x) == 2:
        return "・".join([str(x[0]), str(x[1])]).strip("・")
    return str(x)

def to_display_strings(pair: AnyPair, singles: bool) -> Tuple[str, str]:
    left, right = pair  # type: ignore
    if singles:
        return str(left), str(right)
    else:
        return _fmt_team(left, False), _fmt_team(right, False)

def build_list_table(rounds: List[List[AnyPair]], court_count: int, singles: bool, continuous_match_no: bool=True) -> pd.DataFrame:
    records = []
    match_no_global = 0
    for r_idx, pairings in enumerate(rounds, start=1):
        play_rows, rest_rows = [], []
        for p in pairings:
            left, right = to_display_strings(p, singles)
            is_rest = (right in ["休憩","休"]) or (left in ["休憩","休"]) or ("休憩" in right) or ("休憩" in left) or (right == "BYE") or (left == "BYE")
            target_list = rest_rows if is_rest else play_rows
            shown_match_no = ""
            if not is_rest:
                match_no_global += 1
                shown_match_no = match_no_global if continuous_match_no else ""
            court_str = "-" if is_rest else f"{to_zen_digits(((len(play_rows)) % court_count) + 1)}コート"
            target_list.append({
                "ラウンド": to_zen_digits(r_idx),
                "試合番号": (f"第{to_zen_digits(shown_match_no)}試合" if shown_match_no != "" else ""),
                "コート番号": court_str,
                "名前１": left, "名前２": right
            })
        records.extend(play_rows + rest_rows)
    return pd.DataFrame.from_records(records, columns=["ラウンド","試合番号","コート番号","名前１","名前２"])

# ======================= 総当たり（奇数対応） =======================
def circle_method_rounds_any(participants: List[Union[str, DoublesTeam]], doubles: bool) -> List[List[AnyPair]]:
    arr = participants[:]
    if len(arr) % 2 == 1:
        arr.append(("休憩","休憩") if doubles else "休憩")
    n = len(arr)
    rounds: List[List[AnyPair]] = []
    for _ in range(n - 1):
        pairs: List[AnyPair] = []
        for i in range(n // 2):
            a, b = arr[i], arr[-(i+1)]
            pairs.append((a, b))  # type: ignore
        rounds.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]  # 先頭固定の右回転
    return rounds

# ======================= ランダム（余りは休憩） =======================
def random_round_singles(players: List[str], court_count: int) -> List[SinglesPair]:
    order = players[:]
    random.shuffle(order)
    pairs: List[SinglesPair] = []
    i = 0
    while i + 1 < len(order) and len(pairs) < court_count:
        pairs.append((order[i], order[i+1]))
        i += 2
    for name in order[i:]:
        pairs.append((name, "休憩"))
    return pairs

def random_round_doubles(players: List[str], court_count: int) -> List[DoublesPair]:
    order = players[:]
    random.shuffle(order)
    teams: List[DoublesTeam] = []
    i = 0
    while i + 1 < len(order):
        teams.append((order[i], order[i+1]))
        i += 2
    pairs: List[DoublesPair] = []
    j = 0
    while j + 1 < len(teams) and len(pairs) < court_count:
        pairs.append((teams[j], teams[j+1]))
        j += 2
    rest_pairs: List[DoublesPair] = []
    if j < len(teams):
        rest_pairs.append((teams[j], ("休憩","休憩")))
        j += 1
    if i < len(order):
        for name in order[i:]:
            rest_pairs.append(((name, "休憩"), ("休憩","休憩")))
    return pairs + rest_pairs

# ======================= Names editor（初期2・最小2） =======================
def _ensure_object(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "名前" not in df.columns:
        df["名前"] = pd.Series([], dtype="object")
    df["名前"] = df["名前"].astype("object").where(df["名前"].notna(), "")
    return df

def _nonempty_count(df: pd.DataFrame) -> int:
    df = _ensure_object(df)
    return int((df["名前"].astype(str).str.strip() != "").sum())

def _resize_preserve_head(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    df = _ensure_object(df)
    names = df["名前"].astype(str).tolist()
    if rows <= 0:
        return pd.DataFrame({"名前": []}, dtype="object")
    if len(names) < rows:
        names = names + [""]*(rows - len(names))
    else:
        names = names[:rows]
    return pd.DataFrame({"名前": names}, dtype="object")

def _grid_height_for_rows(rows: int) -> int:
    return int(48 + rows * 38 + 6)

# 初期 state（2行）
if "names_display" not in st.session_state:
    st.session_state["names_display"] = pd.DataFrame({"名前": ["", "" ]}, dtype="object")
if "target_participants" not in st.session_state:
    st.session_state["target_participants"] = max(_nonempty_count(st.session_state["names_display"]), 2)

def _on_change_target():
    target = max(2, int(st.session_state["target_participants"]))  # 最低2
    st.session_state["target_participants"] = target
    df = st.session_state.get("names_display", pd.DataFrame({"名前": []}, dtype="object"))
    filled = _nonempty_count(df)
    required_rows = target
    if filled > target:
        st.session_state["__trim_notice"] = f"入力済み {filled} 名 -> 目標 {target} 名に合わせて先頭 {target} 名だけ残しました。"
    st.session_state["names_display"] = _resize_preserve_head(df, required_rows)

st.subheader("参加者の入力")
col0, col2 = st.columns([3,1])
with col0:
    st.number_input(
        "参加人数",
        min_value=2, max_value=200,
        step=1,
        key="target_participants",
        on_change=_on_change_target,
        help="参加人数は最少２人となっています"
    )
# データフレーム/エディタのツールバー（CSVなど）を全て非表示
st.markdown("""
<style>
[data-testid="stElementToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# 一度だけトリム通知
if "__trim_notice" in st.session_state:
    st.toast(st.session_state["__trim_notice"], icon="⚠️")
    del st.session_state["__trim_notice"]

# 固定行エディタ（高さを行数に連動）
names_df = st.session_state["names_display"]
names_rows = len(names_df)
edited_display = st.data_editor(
    names_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    height=_grid_height_for_rows(names_rows),
    key="names_editor_fixed",
    column_config={
        "名前": st.column_config.TextColumn("名前", required=False, width="medium")
    }
)

# 保存用：末尾空白を除去
saved = _ensure_object(edited_display)
saved = saved[saved["名前"].astype(str).str.strip() != ""]
players = [str(x).strip() for x in saved["名前"].tolist() if str(x).strip()]

# ======================= Settings =======================
with st.sidebar:
    st.header("設定")
    mode = st.radio("方式", ["総当たり", "トーナメント", "ランダム組み合わせ"])
    kind = st.radio("種別", ["シングル", "ダブルス"], horizontal=True)
    doubles = (kind == "ダブルス")
    court_count = st.number_input("コート数", min_value=1, max_value=20, value=2, step=1)
    rounds_count = 1
    if mode == "ランダム組み合わせ":
        rounds_count = st.number_input("ラウンド数（ランダム）", min_value=1, max_value=50, value=1, step=1)

# --- 結果の自動クリア（方式/種別の変更時） ---
mode_changed = False
kind_changed = False
if "last_mode" not in st.session_state:
    st.session_state["last_mode"] = mode
else:
    if mode != st.session_state["last_mode"]:
        mode_changed = True
        st.session_state["last_mode"] = mode
if "last_kind" not in st.session_state:
    st.session_state["last_kind"] = kind
else:
    if kind != st.session_state["last_kind"]:
        kind_changed = True
        st.session_state["last_kind"] = kind
if mode_changed or kind_changed:
    st.session_state["result"] = None

# ---------------- Doubles pairs（行追加禁止） ----------------
pairs_df = None
if doubles and mode != "ランダム組み合わせ":
    st.subheader("ダブルスのペア指定")
    required_pairs = max(1, len(players) // 2)
    team_labels = list(string.ascii_uppercase)[:required_pairs]
    init_pairs = pd.DataFrame({
        "チーム": team_labels,
        "名前１": players[:required_pairs] + [""] * max(0, required_pairs - len(players[:required_pairs])),
        "名前２": players[required_pairs:2*required_pairs] + [""] * max(0, required_pairs - len(players[required_pairs:2*required_pairs])),
    })
    pairs_key = f"pairs_editor_{required_pairs}"
    pairs_df = st.data_editor(
        init_pairs,
        use_container_width=True,
        num_rows="fixed", 
        hide_index=True,
        height=_grid_height_for_rows(required_pairs),
        column_config={
            "チーム": st.column_config.TextColumn("チーム", disabled=True, help="自動ラベル（編集不可）"),
            "名前１": st.column_config.SelectboxColumn("名前１", options=players, required=False),
            "名前２": st.column_config.SelectboxColumn("名前２", options=players, required=False),
        },
        key=pairs_key,
    )

# ======================= 生成ボタン =======================
generate = st.button("組み合わせを生成")

# ======================= Tournament helpers：シード =======================
def _is_seed(x: Union[str, DoublesTeam], doubles: bool) -> bool:
    if doubles:
        return isinstance(x, tuple) and len(x) == 2 and x[0] == "シード" and x[1] == "シード"
    else:
        return isinstance(x, str) and x == "シード"

def _winner_with_seed(pair: AnyPair, doubles: bool):
    left, right = pair  # type: ignore
    left_seed = _is_seed(left, doubles)
    right_seed = _is_seed(right, doubles)
    if left_seed and not right_seed:
        return right
    if right_seed and not left_seed:
        return left
    if left_seed and right_seed:
        return ("シード","シード") if doubles else "シード"
    return None  # 試合で決まる

# ======================= Results =======================
st.subheader("結果")

if generate:
    try:
        if len(players) < 2:
            st.error("２名以上入力してください。")
        else:
            if mode == "総当たり":
                if doubles:
                    if pairs_df is None:
                        st.error("ペア表が見つかりません。")
                    else:
                        teams = [
                            (str(r["名前１"]).strip(), str(r["名前２"]).strip())
                            for _, r in pairs_df.iterrows()
                            if str(r["名前１"]).strip() and str(r["名前２"]).strip()
                        ]
                        if len(teams) < 2:
                            st.error("２チーム以上のペアを入力してください。")
                        else:
                            rounds = circle_method_rounds_any(teams, doubles=True)
                            df = build_list_table(rounds, court_count=court_count, singles=False, continuous_match_no=True)
                            _save_result(df, "表を保存（PNG）", "match_list.png")
                else:
                    participants = players[:]
                    random.shuffle(participants)
                    rounds = circle_method_rounds_any(participants, doubles=False)
                    df = build_list_table(rounds, court_count=court_count, singles=True, continuous_match_no=True)
                    _save_result(df, "表を保存（PNG）", "match_list.png")

            elif mode == "トーナメント":
                if doubles:
                    if pairs_df is None:
                        st.error("ペア表が見つかりません。")
                        st.stop()
                    participants_any: List[Union[str, DoublesTeam]] = [
                        (str(r["名前１"]).strip(), str(r["名前２"]).strip())
                        for _, r in pairs_df.iterrows()
                        if str(r["名前１"]).strip() and str(r["名前２"]).strip()
                    ]
                else:
                    participants_any = players[:]

                if len(participants_any) < 2:
                    st.error("２名以上入力してください。")
                else:
                    if not doubles:
                        random.shuffle(participants_any)
                    n = len(participants_any)
                    next_pow = 1 << (n - 1).bit_length()
                    seeds_needed = next_pow - n
                    for _ in range(seeds_needed):
                        seed_name = ("シード","シード") if doubles else "シード"
                        participants_any.append(seed_name)

                    rounds: List[List[AnyPair]] = []
                    current: List[Union[str, DoublesTeam, str]] = participants_any[:]
                    ridx = 1
                    while len(current) > 1:
                        ps: List[AnyPair] = []
                        winners: List[Union[str, DoublesTeam, str]] = []
                        for i in range(0, len(current), 2):
                            left = current[i]
                            right = current[i+1]
                            pair = (left, right)  # type: ignore
                            ps.append(pair)
                            w = _winner_with_seed(pair, doubles)
                            winners.append(w if w is not None else f"勝者{ridx}-{(i//2)+1}")
                        rounds.append(ps)
                        current = winners
                        ridx += 1

                    df = build_list_table(rounds, court_count=court_count, singles=not doubles, continuous_match_no=True)
                    _save_result(df, "表を保存（PNG）", "tournament_list.png")

            else:  # ランダム組み合わせ
                rounds: List[List[AnyPair]] = []
                if doubles:
                    for _ in range(int(rounds_count)):
                        rounds.append(random_round_doubles(players, court_count))
                    df = build_list_table(rounds, court_count=court_count, singles=False, continuous_match_no=True)
                    _save_result(df, "表を保存（PNG）", "random_doubles.png")
                else:
                    for _ in range(int(rounds_count)):
                        rounds.append(random_round_singles(players, court_count))
                    df = build_list_table(rounds, court_count=court_count, singles=True, continuous_match_no=True)
                    _save_result(df, "表を保存（PNG）", "random_singles.png")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.exception(e)

# 直近の結果を常に表示（ダウンロード押下で再実行されても保持）
_render_result = _render_result  # avoid accidental name typo
_render_result()

