# @file message.py
# @brief Warnings of tensight, shown once for each text (NFR-008)
# @author Kohei Yamaguchi, Claude

import warnings



MESSAGE_REGISTRY = {
    "MSG-001": "観測点のパターン「{pattern}」に当てはまる層がありません",
    "MSG-002": "観測点「{point}」の記録の計算に失敗したため、このステップでは記録しませんでした: {error}",
    "MSG-003": "読み取り部品「{adapter}」が失敗したため、量子化の情報を「なし」にしました: {error}",
    "MSG-004": "記録フォルダ {path} に書き込めないため、記録を止めます。学習は続けます: {error}",
    "MSG-005": "比較用の保存が 200 MB を超えたため、比較用の保存を止めました",
    "MSG-006": "{function} が run_start の前に呼ばれたため、何もしませんでした",
    "MSG-007": "実行中に run_start が呼ばれたため、前の実行を閉じて新しい実行を始めました",
    "MSG-008": "更新量を記録するには、watch_model に optimizer を渡してください。更新量は記録しません",
    "MSG-009": "観測点「{point}」は {ndim} 次元のため、並べ方の式「{pattern}」の地図を記録しません",
    "MSG-010": "型 {type} の値は記録できないため、記録しません。converter_register で変換の関数を登録すると記録できます",
    "MSG-011": "型 {type} の変換の関数が失敗したため、その値を記録しませんでした: {error}",
    "MSG-012": "観測点「{point}」は {reason} のため、記録しません",
    "MSG-013": "観測点「{point}」は、層の種類 {layer} から地図の作り方を選べないため、並べ方の式「{pattern}」で記録します。maps で作り方を指定できます",
    "MSG-014": "観測点「{point}」の値は、{reason}ため、一部のテンソルを記録しません",
    "MSG-015": "記録する観測点が1つもありません。run_start の points と kinds を確かめてください",
}

_seen: set[str] = set()



class TensightWarning(UserWarning):
    '''
    #* The warning of tensight.
    '''


def message_warn(code: str, **fields: object) -> None:
    '''
    #* Show one warning, but only the first time for the same text (02a 1章).
    #! code must be a key of MESSAGE_REGISTRY, and fields must fill every name in its text.
    '''
    
    # Make the text, and stop if it was already shown
    text = MESSAGE_REGISTRY[code].format(**fields)
    if text in _seen:
        return None
    
    # Show it once
    _seen.add(text)
    warnings.warn("[tensight " + code + "] " + text, TensightWarning, stacklevel=2)
    return None


def message_reset() -> None:
    '''
    #* Forget the shown warnings, so that a new run shows them again.
    #? Called from run_start().
    '''
    
    _seen.clear()
    return None
